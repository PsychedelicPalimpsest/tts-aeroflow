"""Regression checks for audible aliasing, phase objectives and reconstruction."""

import numpy as np
import pytest
import soundfile as sf
import torch

from aeroflow.dataset.dataset import HiFiTTSDataset
from aeroflow.dataset.hf_hifi_tts import _resample_mono
from aeroflow.losses.losses import InstantaneousFrequencyLoss
from aeroflow.models.istft import STFTAnalysis, iSTFTSynthesizer


def test_resampling_rejects_out_of_band_tone_in_both_loaders(tmp_path):
    t = np.arange(44100) / 44100
    # 16 kHz would alias to 8 kHz when decimated to 24 kHz.
    low = np.sin(2 * np.pi * 1000 * t)
    high = np.sin(2 * np.pi * 16000 * t)
    mixed = (0.25 * (low + high)).astype(np.float32)
    path = tmp_path / "tones.wav"
    sf.write(path, mixed, 44100, subtype="FLOAT")
    outputs = [_resample_mono(mixed, 44100), HiFiTTSDataset()._load_audio(str(path))]
    for output in outputs:
        spectrum = np.abs(np.fft.rfft(output[2400:-2400]))
        # Interior spans exactly 0.8 s; integer frequency bins avoid leakage.
        assert spectrum[6400] / spectrum[800] < 0.002
        assert output.dtype == np.float32
        assert len(output) == 24000
    passed = _resample_mono(low.astype(np.float32), 44100)
    assert np.sqrt(np.mean(passed[2400:-2400] ** 2)) == pytest.approx(2 ** -0.5, rel=0.01)


def test_resampling_empty_and_same_rate():
    assert _resample_mono(np.empty(0), 44100).size == 0
    assert _resample_mono(np.ones(1), 44100).size == 0
    x = np.arange(20, dtype=np.float32)
    np.testing.assert_array_equal(_resample_mono(x, 24000), x)


def test_phase_loss_silence_is_zero_and_gradients_finite():
    predicted = torch.randn(1, 2400, requires_grad=True)
    loss = InstantaneousFrequencyLoss()(torch.zeros_like(predicted), predicted)
    loss.backward()
    assert loss.item() == 0
    assert torch.isfinite(predicted.grad).all()


def test_phase_loss_is_invariant_to_extra_padding():
    torch.manual_seed(4)
    target = torch.randn(2, 4800)
    predicted = torch.randn_like(target)
    # Keep real audio away from the STFT boundary so this tests weighting,
    # rather than the difference between reflected and zero-padded boundaries.
    target[:, -1200:] = 0
    predicted[:, -1200:] = 0
    mask = torch.ones_like(target, dtype=torch.bool)
    fn = InstantaneousFrequencyLoss()
    original = fn(target, predicted, mask)
    pad = lambda x: torch.nn.functional.pad(x, (0, 4800))
    padded = fn(pad(target), pad(predicted), pad(mask))
    assert torch.allclose(original, padded, atol=1e-6)


def test_phase_refinement_preserves_valid_spectrum_and_length():
    torch.manual_seed(9)
    audio = torch.randn(1, 4800)
    spectrum = STFTAnalysis()(audio)[0]
    synth = iSTFTSynthesizer()
    assert torch.equal(synth.refine_phase(spectrum, 0, 4800), synth(spectrum, 4800))
    refined = synth.refine_phase(spectrum, 4, 4800)
    assert refined.shape == audio.shape
    assert torch.allclose(refined, audio, atol=3e-6)
    silent = synth.refine_phase(torch.zeros_like(spectrum), 4, 4800)
    assert torch.isfinite(silent).all() and silent.count_nonzero() == 0
    with pytest.raises(ValueError):
        synth.refine_phase(spectrum, -1)


def test_phase_refinement_reduces_magnitude_residual():
    torch.manual_seed(10)
    analysis, synth = STFTAnalysis(), iSTFTSynthesizer()
    spectrum = analysis(torch.randn(1, 4800))[0]
    perturbed = spectrum * torch.exp(1j * torch.randn_like(spectrum.real))
    before = analysis(synth(perturbed, 4800))[1]
    after = analysis(synth.refine_phase(perturbed, 8, 4800))[1]
    assert (after - spectrum.abs()).norm() < (before - spectrum.abs()).norm()
