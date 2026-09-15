"""
Comprehensive Unit Test Suite for AeroFlow-v2.
Verifies all modules, math guarantees, edge cases, and gradient flows.
"""

import math
import pytest
import torch
import torch.nn.functional as F

from aeroflow.frontend.text_norm import TextNormalizer
from aeroflow.frontend.phonemizer import Phonemizer, ALL_TOKENS
from aeroflow.models.encoder import ConformerEncoder
from aeroflow.models.alignment import (
    maximum_path_viterbi,
    alignment_to_durations,
    EnergyConstrainedDurationPredictor,
    expand_text_representations
)
from aeroflow.models.flow_matching import (
    VectorFieldNetwork,
    NonUniformHeunSolver,
    OptimalTransportCFM
)
from aeroflow.models.decoder import ComplexSTFTDecoder
from aeroflow.models.istft import iSTFTSynthesizer, STFTAnalysis
from aeroflow.losses.losses import AeroFlowLoss, InstantaneousFrequencyLoss
from aeroflow.models.pipeline import AeroFlowTTS
from aeroflow.dataset.dataset import create_synthetic_batch


def test_text_normalization():
    norm = TextNormalizer()
    assert "doctor" in norm.normalize("Dr. Smith")
    assert "one hundred dollars" in norm.normalize("$100")
    assert "three point one four" in norm.normalize("3.14")
    assert "twenty-first" in norm.normalize("21st")
    assert "nineteen eighty-four" in norm.normalize("1984")
    assert norm.normalize("") == ""


def test_phonemizer():
    p = Phonemizer()
    assert len(p.vocab) == 84
    tokens = p.text_to_sequence("Peter Piper picked a peck.")
    assert len(tokens) > 0
    assert all(0 <= t < 84 for t in tokens)
    # Test unknown word fallback
    oov_tokens = p.text_to_sequence("Supercalifragilistic")
    assert len(oov_tokens) > 0
    assert all(0 <= t < 84 for t in oov_tokens)


def test_conformer_encoder():
    encoder = ConformerEncoder(vocab_size=84, d_model=192, num_blocks=2)
    tokens = torch.randint(1, 84, (2, 20))
    lengths = torch.tensor([20, 15])
    H, mask = encoder(tokens, lengths=lengths)
    assert H.shape == (2, 20, 192)
    assert mask is not None
    assert mask.shape == (2, 20)
    # Gradient check
    H.sum().backward()


def test_viterbi_mas():
    B, N, T = 2, 4, 12
    scores = torch.randn(B, N, T)
    text_lens = torch.tensor([4, 3])
    audio_lens = torch.tensor([12, 10])
    path = maximum_path_viterbi(scores, text_lens, audio_lens)
    assert path.shape == (B, N, T)
    # Each valid audio frame must have exactly one phoneme assigned
    assert (path[0, :, :12].sum(dim=0) == 1.0).all()
    durations = alignment_to_durations(path)
    assert (durations[0] >= 1.0).all()


def test_duration_predictor():
    pred = EnergyConstrainedDurationPredictor(text_dim=192)
    H = torch.randn(2, 10, 192)
    dur = pred.predict_durations(H, alpha=1.0, d_min=1, d_max=80)
    assert dur.shape == (2, 10)
    assert (dur >= 1).all()
    assert (dur <= 80).all()


def test_monotonic_expansion():
    H = torch.randn(2, 5, 192)
    dur = torch.tensor([[2, 1, 3, 1, 2], [1, 2, 1, 3, 1]], dtype=torch.long)
    C = expand_text_representations(H, dur)
    assert C.shape[0] == 2
    assert C.shape[1] == 192
    assert C.shape[2] == max(dur.sum(dim=1).tolist())


def test_flow_matching():
    ot = OptimalTransportCFM()
    z = torch.randn(2, 32, 25)
    x_t, t, u_t, x_0 = ot.sample_trajectory(z)
    assert x_t.shape == (2, 32, 25)
    assert u_t.shape == (2, 32, 25)

    vf = VectorFieldNetwork(latent_dim=32, model_dim=192, num_blocks=2)
    C = torch.randn(2, 192, 25)
    v = vf(x_t, t, C)
    assert v.shape == (2, 32, 25)

    solver = NonUniformHeunSolver(num_steps=6, rho=1.5)
    z_sol = solver.solve(vf, x_0, C)
    assert z_sol.shape == (2, 32, 25)


def test_complex_stft_decoder():
    decoder = ComplexSTFTDecoder(latent_dim=32, model_dim=128, num_bins=513, num_blocks=2)
    z = torch.randn(2, 32, 20)
    S, mag, cos_phi, sin_phi = decoder(z)
    assert S.shape == (2, 513, 20)
    assert S.is_complex()
    # Unit phase check
    unit = cos_phi ** 2 + sin_phi ** 2
    assert torch.allclose(unit, torch.ones_like(unit), atol=1e-5)


def test_istft_cola():
    n_fft = 1024
    hop = 240
    analysis = STFTAnalysis(n_fft=n_fft, hop_length=hop)
    synth = iSTFTSynthesizer(n_fft=n_fft, hop_length=hop)
    audio = torch.randn(2, 24000)
    S, _, _ = analysis(audio)
    recon = synth(S, length=audio.shape[-1])
    diff = (recon - audio).abs().max().item()
    assert diff < 1e-4, f"COLA error too large: {diff}"


def test_loss_suite():
    loss_fn = AeroFlowLoss()
    B, T_frames, N_tokens = 2, 30, 10
    T_samples = T_frames * 240

    v_pred = torch.randn(B, 32, T_frames, requires_grad=True)
    u_target = torch.randn(B, 32, T_frames)
    log_dur_pred = torch.randn(B, N_tokens, requires_grad=True)
    dur_target = torch.randint(1, 5, (B, N_tokens)).float()
    y_audio = torch.randn(B, T_samples)
    y_hat_audio = torch.randn(B, T_samples, requires_grad=True)

    loss_total, metrics = loss_fn(
        v_pred, u_target, log_dur_pred, dur_target, y_audio, y_hat_audio
    )
    assert torch.isfinite(loss_total)
    loss_total.backward()
    assert v_pred.grad is not None
    assert log_dur_pred.grad is not None
    assert y_hat_audio.grad is not None


def test_pipeline_end_to_end():
    model = AeroFlowTTS()
    batch = create_synthetic_batch(batch_size=2, audio_dur_s=1.0)
    out = model.forward_train(
        batch["phoneme_tokens"],
        batch["audio"],
        batch["text_lengths"],
        batch["audio_lengths"]
    )
    assert out["v_pred"].shape[0] == 2
    assert out["audio_hat"].shape[-1] == batch["audio"].shape[-1]

    # Synthesis test
    with torch.no_grad():
        wav = model.synthesize("Peter Piper picked a peck of pickled peppers.")
        assert wav.dim() == 1
        assert wav.shape[0] > 0
        assert torch.isfinite(wav).all()


def test_adversarial_normalizer_currency():
    norm = TextNormalizer()
    assert "one million dollars" in norm.normalize("$1,000,000")
    assert "one million dollars fifty cents" in norm.normalize("$1,000,000.50")
    assert "twelve dollars fifty cents" in norm.normalize("$12.50")
    assert "one million" in norm.normalize("1,000,000")
    assert "saint John" in norm.normalize("St. John lived on Maple St.")
    assert "Maple street" in norm.normalize("St. John lived on Maple St.")


def test_adversarial_phonemizer_vocabulary():
    p = Phonemizer()
    assert p.phonemize_word("have") == ["HH", "AE1", "V"]
    assert p.phonemize_word("would") == ["W", "UH1", "D"]
    assert p.phonemize_word("said") == ["S", "EH1", "D"]
    assert p.phonemize_word("don't") == ["D", "OW1", "N", "T"]
    assert p.phonemize_word("could") == ["K", "UH1", "D"]
    assert p.phonemize_word("been") == ["B", "IH1", "N"]
    assert p.phonemize_word("were") == ["W", "ER1"]


def test_adversarial_viterbi_lower_bound():
    # Test path search where audio frames are close to text length
    N, T = 10, 12
    scores = torch.randn(1, N, T)
    text_lens = torch.tensor([N])
    audio_lens = torch.tensor([T])
    path = maximum_path_viterbi(scores, text_lens, audio_lens)
    durations = alignment_to_durations(path, text_lengths=text_lens)
    # Every phoneme must be allocated at least 1 frame
    assert (durations[0, :N] >= 1.0).all()
    assert durations[0, :N].sum() == T


def test_adversarial_if_loss_nan_immunity():
    if_loss = InstantaneousFrequencyLoss()
    # 1. Exact zero input
    y_zero = torch.zeros(2, 24000)
    y_hat_zero = torch.zeros(2, 24000, requires_grad=True)
    l_zero = if_loss(y_zero, y_hat_zero)
    l_zero.backward()
    assert torch.isfinite(l_zero)
    assert torch.isfinite(y_hat_zero.grad).all()

    # 2. Subnormal floats (1e-15)
    y_sub = torch.full((2, 24000), 1e-15)
    y_hat_sub = torch.full((2, 24000), 1e-15, requires_grad=True)
    l_sub = if_loss(y_sub, y_hat_sub)
    l_sub.backward()
    assert torch.isfinite(l_sub)
    assert torch.isfinite(y_hat_sub.grad).all()


def test_prior_alignment_gradient_flow():
    model = AeroFlowTTS()
    loss_fn = AeroFlowLoss()
    batch = create_synthetic_batch(batch_size=2, audio_dur_s=1.0)
    out = model.forward_train(
        batch["phoneme_tokens"],
        batch["audio"],
        batch["text_lengths"],
        batch["audio_lengths"]
    )
    loss, metrics = loss_fn(
        v_pred=out["v_pred"],
        u_target=out["u_target"],
        log_dur_pred=out["log_dur_pred"],
        dur_target=out["dur_target"],
        y_audio=out["audio_gt"],
        y_hat_audio=out["audio_hat"],
        text_mask=out["text_mask"],
        frame_mask=out["frame_mask"],
        audio_mask=out["audio_mask"],
        text_proj_expanded=out["text_proj_expanded"],
        z_target=out["z_target"]
    )
    loss.backward()
    assert model.text_to_latent_proj.weight.grad is not None
    assert model.text_to_latent_proj.weight.grad.norm().item() > 0.0
    assert "loss_prior" in metrics
    assert torch.isfinite(metrics["loss_prior"])


def test_stft_frame_count_exactness():
    model = AeroFlowTTS()
    audio_len = 24000
    hop = 240
    expected_frames = 1 + audio_len // hop  # 101
    audio = torch.randn(1, audio_len)
    S, _, _ = model.stft_analysis(audio)
    assert S.shape[-1] == expected_frames == 101


def test_spectral_loss_masked_silence_grad_finite():
    """Realistic worst case: speech + trailing padded silence under audio_mask.

    Masked-out (exact-zero) regions must not poison gradients via 0 * NaN on
    any backend/BLAS.
    """
    torch.manual_seed(0)
    loss_fn = AeroFlowLoss()
    speech = torch.randn(2, 12000) * 0.5
    y = torch.zeros(2, 24000)
    y[:, :12000] = speech
    y_hat = y.clone().detach().requires_grad_(True)
    audio_mask = torch.zeros(2, 24000, dtype=torch.bool)
    audio_mask[:, :12000] = True

    l_mr = loss_fn.mr_stft_loss(y, y_hat, audio_mask=audio_mask)
    l_if = loss_fn.if_loss(y, y_hat, audio_mask=audio_mask)
    assert torch.isfinite(l_mr) and torch.isfinite(l_if)
    (l_mr + l_if).backward()
    assert torch.isfinite(y_hat.grad).all()


def test_attention_mask_fp16_no_overflow():
    """Kaggle regression: -1e9 mask overflows FP16 (max 65504) under AMP.

    The mask fill must use a dtype-aware minimum so masked_fill works in
    half precision and masked positions still softmax to zero.
    """
    from aeroflow.models.encoder import MultiHeadSelfAttentionRoPE
    torch.manual_seed(0)
    attn = MultiHeadSelfAttentionRoPE(d_model=192, num_heads=4).half()
    x = torch.randn(2, 10, 192, dtype=torch.float16)
    mask = torch.ones(2, 10)
    mask[:, 8:] = 0
    out = attn(x, mask=mask)
    assert out.dtype == torch.float16
    assert torch.isfinite(out.float()).all()

