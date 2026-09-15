"""
AeroFlow-v2 Unified End-to-End TTS Architecture.
Combines:
1. Conformer Text Encoder with RoPE.
2. Viterbi Monotonic Alignment Search (MAS) & Energy-Constrained Duration Predictor.
3. 32-Channel Continuous Acoustic Latent Encoder.
4. Optimal Transport ConvNeXt-ODE Vector Field with AdaLN.
5. 6-Step Non-Uniform Heun ODE Solver (rho = 1.5).
6. ConvNeXt-V2 Complex STFT Decoder (Log-Mag + Unit Phase Vectors).
7. Alias-Free COLA-compliant iSTFT Waveform Synthesizer (N_fft=1024, Hop=240).
"""

from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from aeroflow.frontend.phonemizer import Phonemizer
from aeroflow.models.encoder import ConformerEncoder
from aeroflow.models.alignment import (
    EnergyConstrainedDurationPredictor,
    maximum_path_viterbi,
    alignment_to_durations,
    expand_text_representations
)
from aeroflow.models.flow_matching import (
    VectorFieldNetwork,
    NonUniformHeunSolver,
    OptimalTransportCFM
)
from aeroflow.models.decoder import ComplexSTFTDecoder
from aeroflow.models.istft import iSTFTSynthesizer, STFTAnalysis


class AcousticLatentEncoder(nn.Module):
    """
    Encodes full-band STFT magnitude spectra (513 bins) into 32-channel continuous latents z.
    Provides the continuous acoustic manifold target for Optimal Transport Flow Matching.
    """

    def __init__(self, in_bins: int = 513, hidden_dim: int = 256, latent_dim: int = 32):
        super().__init__()
        self.conv1 = nn.Conv1d(in_bins, hidden_dim, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(1, hidden_dim)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(1, hidden_dim)
        self.out_proj = nn.Conv1d(hidden_dim, latent_dim, kernel_size=1)
        self.out_norm = nn.LayerNorm(latent_dim)

    def forward(self, mag: torch.Tensor) -> torch.Tensor:
        # mag: [B, 513, T]
        # Operate in log-magnitude space for linear dynamic range
        log_mag = torch.log(mag.clamp(min=1e-5))
        h = F.gelu(self.norm1(self.conv1(log_mag)))
        h = F.gelu(self.norm2(self.conv2(h)))
        z = self.out_proj(h)  # [B, 32, T]
        z = self.out_norm(z.transpose(1, 2)).transpose(1, 2)
        return z


class AeroFlowTTS(nn.Module):
    """
    Unified AeroFlow-v2 Text-to-Speech Engine.
    Engineered for ultra-fast, broadcast-quality 24 kHz synthesis on multi-core CPUs.
    """

    def __init__(
        self,
        vocab_size: int = 84,
        text_dim: int = 192,
        latent_dim: int = 32,
        decoder_dim: int = 256,
        n_fft: int = 1024,
        hop_length: int = 240,
        heun_steps: int = 6,
        heun_rho: float = 1.5
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.text_dim = text_dim
        self.latent_dim = latent_dim
        self.n_fft = n_fft
        self.hop_length = hop_length

        # 1. Text & Alignment Front-End
        self.phonemizer = Phonemizer()
        self.encoder = ConformerEncoder(
            vocab_size=vocab_size,
            d_model=text_dim,
            num_blocks=4,
            num_heads=4
        )
        self.duration_predictor = EnergyConstrainedDurationPredictor(
            text_dim=text_dim,
            hidden_dim=text_dim
        )
        self.text_to_latent_proj = nn.Linear(text_dim, latent_dim)

        # 2. Acoustic Processing & Flow Matching
        self.stft_analysis = STFTAnalysis(n_fft=n_fft, hop_length=hop_length)
        self.acoustic_encoder = AcousticLatentEncoder(
            in_bins=n_fft // 2 + 1,
            hidden_dim=decoder_dim,
            latent_dim=latent_dim
        )
        self.ot_cfm = OptimalTransportCFM(sigma_min=1e-4)
        self.vector_field = VectorFieldNetwork(
            latent_dim=latent_dim,
            model_dim=text_dim,
            num_blocks=6
        )
        self.solver = NonUniformHeunSolver(num_steps=heun_steps, rho=heun_rho)

        # 3. Complex STFT Decoder & Alias-Free iSTFT
        self.decoder = ComplexSTFTDecoder(
            latent_dim=latent_dim,
            model_dim=decoder_dim,
            num_bins=n_fft // 2 + 1,
            num_blocks=4
        )
        self.istft = iSTFTSynthesizer(n_fft=n_fft, hop_length=hop_length)

    def compute_mas_alignment(
        self,
        text_repr: torch.Tensor,
        z_target: torch.Tensor,
        text_lengths: torch.Tensor,
        frame_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes Monotonic Alignment Search between text embeddings and acoustic latents.
        text_repr: [B, N, D] text embeddings H_text or projected latents [B, N, latent_dim]
        z_target: [B, latent_dim, T] acoustic latent targets
        Returns:
            path: [B, N, T] binary alignment matrix
            durations: [B, N] integer duration targets
        """
        if text_repr.shape[-1] != self.latent_dim:
            text_proj = self.text_to_latent_proj(text_repr)  # [B, N, 32]
        else:
            text_proj = text_repr

        z_t = z_target.transpose(1, 2)               # [B, T, 32]

        # Compute negative L2 distance matrix
        dist = torch.cdist(text_proj, z_t, p=2.0)    # [B, N, T]
        neg_dist = -0.5 * (dist ** 2)

        path = maximum_path_viterbi(neg_dist, text_lengths, frame_lengths)
        durations = alignment_to_durations(path, text_lengths=text_lengths)
        return path, durations

    def forward_train(
        self,
        phoneme_tokens: torch.Tensor,
        audio_24k: torch.Tensor,
        text_lengths: Optional[torch.Tensor] = None,
        audio_lengths: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Full differentiable training pass.
        phoneme_tokens: [B, N] token IDs
        audio_24k: [B, T_samples] 24 kHz target waveform
        Returns:
            dictionary containing all predictions and targets required for loss computation.
        """
        B, N = phoneme_tokens.shape
        device = phoneme_tokens.device

        # 1. Text representations
        H_text, text_mask = self.encoder(phoneme_tokens, lengths=text_lengths)
        text_proj = self.text_to_latent_proj(H_text)  # [B, N, latent_dim]

        # 2. Extract ground-truth STFT and acoustic latents
        S_gt, mag_gt, phase_gt = self.stft_analysis(audio_24k)
        T_frames = S_gt.shape[-1]
        z_target = self.acoustic_encoder(mag_gt)  # [B, 32, T_frames]

        # Exact centered STFT frame count: 1 + audio_lengths // hop_length
        if text_lengths is None:
            text_lengths = torch.full((B,), N, dtype=torch.long, device=device)
        if audio_lengths is None:
            frame_lengths = torch.full((B,), T_frames, dtype=torch.long, device=device)
            audio_mask = torch.ones((B, audio_24k.shape[-1]), dtype=torch.bool, device=device)
            frame_mask = torch.ones((B, T_frames), dtype=torch.bool, device=device)
        else:
            frame_lengths = (1 + audio_lengths // self.hop_length).clamp(min=1, max=T_frames)
            time_idx = torch.arange(audio_24k.shape[-1], device=device).unsqueeze(0).expand(B, -1)
            audio_mask = time_idx < audio_lengths.unsqueeze(1)
            frame_idx = torch.arange(T_frames, device=device).unsqueeze(0).expand(B, -1)
            frame_mask = frame_idx < frame_lengths.unsqueeze(1)

        # 3. Monotonic Alignment Search (Viterbi MAS)
        with torch.no_grad():
            path, dur_target = self.compute_mas_alignment(
                text_proj.detach(), z_target.detach(), text_lengths, frame_lengths
            )

        # 4. Duration Prediction
        log_dur_pred = self.duration_predictor(H_text, mask=text_mask)

        # 5. Monotonic Expansion using MAS duration targets
        C = expand_text_representations(H_text, dur_target.long())
        if C.shape[-1] < T_frames:
            C = F.pad(C, (0, T_frames - C.shape[-1]))
        elif C.shape[-1] > T_frames:
            C = C[..., :T_frames]

        # Prior alignment expansion: connect text_to_latent_proj to autograd
        text_proj_expanded = expand_text_representations(text_proj, dur_target.long())
        if text_proj_expanded.shape[-1] < T_frames:
            text_proj_expanded = F.pad(text_proj_expanded, (0, T_frames - text_proj_expanded.shape[-1]))
        elif text_proj_expanded.shape[-1] > T_frames:
            text_proj_expanded = text_proj_expanded[..., :T_frames]

        # 6. Optimal Transport Flow Matching (detach z_target to prevent latent collapse)
        x_t, t, u_target, x_0 = self.ot_cfm.sample_trajectory(z_target.detach())
        v_pred = self.vector_field(x_t, t, C)

        # 7. Complex STFT Reconstruction & Alias-Free iSTFT
        S_hat, mag_hat, cos_hat, sin_hat = self.decoder(z_target)
        audio_hat = self.istft(S_hat, length=audio_24k.shape[-1])

        return {
            "v_pred": v_pred,
            "u_target": u_target,
            "log_dur_pred": log_dur_pred,
            "dur_target": dur_target,
            "audio_gt": audio_24k,
            "audio_hat": audio_hat,
            "text_mask": text_mask,
            "frame_mask": frame_mask,
            "audio_mask": audio_mask,
            "z_target": z_target,
            "text_proj_expanded": text_proj_expanded,
            "S_gt": S_gt,
            "S_hat": S_hat
        }

    @torch.no_grad()
    def synthesize(
        self,
        text_or_tokens: Union[str, torch.Tensor],
        alpha: float = 1.0
    ) -> torch.Tensor:
        """
        End-to-end synthesis from raw text string or token tensor to 24 kHz audio.
        text_or_tokens: Input English text string or [1, N] token tensor.
        alpha: Speaking tempo modifier (1.0 = normal tempo, >1.0 slower, <1.0 faster).
        Returns:
            audio: [N_samples] 24,000 Hz 1D waveform tensor.
        """
        device = next(self.parameters()).device

        if isinstance(text_or_tokens, str):
            token_ids = self.phonemizer.text_to_sequence(text_or_tokens)
            tokens = torch.tensor([token_ids], dtype=torch.long, device=device)
        else:
            tokens = text_or_tokens.to(device)
            if tokens.dim() == 1:
                tokens = tokens.unsqueeze(0)

        # 1. Phoneme Encoding
        H_text, _ = self.encoder(tokens)  # [1, N, 192]

        # 2. Duration Prediction with hard bounds [1, 80]
        durations = self.duration_predictor.predict_durations(H_text, alpha=alpha, d_min=1, d_max=80)

        # 3. Monotonic Expansion (strictly monotonic, zero skipping, zero looping)
        C = expand_text_representations(H_text, durations)  # [1, 192, T]
        T = C.shape[-1]

        # 4. Sample Continuous Prior Noise
        x_0 = torch.randn(1, self.latent_dim, T, device=device)

        # 5. 6-Step Non-Uniform Heun ODE Flow Matching
        z = self.solver.solve(self.vector_field, x_0, C)  # [1, 32, T]

        # 6. Complex STFT Decoding & Alias-Free iSTFT Synthesis
        S_complex, _, _, _ = self.decoder(z)  # [1, 513, T]
        audio = self.istft(S_complex)          # [1, N_samples]

        return audio.squeeze(0)  # [N_samples] @ 24 kHz
