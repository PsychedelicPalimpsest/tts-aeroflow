"""
AeroFlow-v2 Convex Spectral Regression Loss Suite.
Eliminates adversarial GAN instability with 100% convex spectral losses:
1. Optimal Transport Conditional Flow Matching (CFM) Loss (L_cfm)
2. Monotonic Duration L1 Regression Loss (L_dur)
3. Text-to-Latent Prior Alignment Loss (L_prior)
4. Multi-Resolution Complex STFT Loss (L_mr_stft)
5. Instantaneous Frequency Unit Phasor Cosine Distance Loss (L_if)
Total Loss: L_total = L_cfm + 1.0 * L_dur + 1.0 * L_prior + 1.0 * L_mr_stft + 0.5 * L_if
"""

import math
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class CFMLoss(nn.Module):
    """Optimal Transport Conditional Flow Matching loss (mean squared velocity error)."""

    def __init__(self):
        super().__init__()

    def forward(
        self,
        v_pred: torch.Tensor,
        u_target: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        v_pred: [B, 32, T] predicted vector field
        u_target: [B, 32, T] target velocity vector
        mask: [B, T] or [B, 1, T] frame mask
        Returns:
            loss: scalar MSE loss
        """
        diff = (v_pred - u_target) ** 2
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1)
            diff = diff * mask.to(diff.dtype)
            return diff.sum() / (mask.sum() * diff.shape[1] + 1e-5)
        return diff.mean()


class DurationLoss(nn.Module):
    """L1 log-duration regression against Viterbi MAS alignment targets."""

    def __init__(self):
        super().__init__()

    def forward(
        self,
        log_dur_pred: torch.Tensor,
        dur_target: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        log_dur_pred: [B, N] predicted log duration
        dur_target: [B, N] target duration from MAS (integer frames)
        mask: [B, N] phoneme sequence mask
        Returns:
            loss: scalar L1 loss
        """
        log_dur_target = torch.log(dur_target.clamp(min=1.0) + 1e-3)
        diff = torch.abs(log_dur_pred - log_dur_target)

        if mask is not None:
            diff = diff * mask.to(diff.dtype)
            return diff.sum() / (mask.sum() + 1e-5)
        return diff.mean()


class PriorAlignmentLoss(nn.Module):
    """
    L2 alignment loss between expanded text representations and acoustic latents.
    Connects text_to_latent_proj to autograd to guide text embeddings towards the acoustic manifold.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        text_proj_expanded: torch.Tensor,
        z_target: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        text_proj_expanded: [B, 32, T] expanded text projection
        z_target: [B, 32, T] acoustic latent targets (detached)
        mask: [B, T] or [B, 1, T] frame mask
        """
        diff = (text_proj_expanded - z_target.detach()) ** 2
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1)
            diff = diff * mask.to(diff.dtype)
            return diff.sum() / (mask.sum() * diff.shape[1] + 1e-5)
        return diff.mean()


class SingleResolutionSTFTLoss(nn.Module):
    """STFT loss for a single FFT resolution combining spectral convergence, log-mag, and complex distance."""

    def __init__(self, n_fft: int, hop_length: int, win_length: int):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.register_buffer("window", torch.hann_window(win_length, periodic=True))

    @torch.amp.autocast('cuda', enabled=False)
    def forward(
        self,
        y: torch.Tensor,
        y_hat: torch.Tensor,
        audio_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        y: [B, T_samples] ground-truth audio
        y_hat: [B, T_samples] synthesized audio
        audio_mask: [B, T_samples] optional boolean/float audio sequence mask
        Returns:
            sc_loss: spectral convergence
            log_mag_loss: L1 log magnitude distance
            complex_loss: Frobenius norm of complex difference
        """
        y = y.float()
        y_hat = y_hat.float()

        # Ensure identical lengths
        min_len = min(y.shape[-1], y_hat.shape[-1])
        y = y[..., :min_len]
        y_hat = y_hat[..., :min_len]

        S = torch.stft(
            y, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length,
            window=self.window, center=True, normalized=False, onesided=True, return_complex=True
        )
        S_hat = torch.stft(
            y_hat, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length,
            window=self.window, center=True, normalized=False, onesided=True, return_complex=True
        )

        mag = torch.abs(S)
        mag_hat = torch.abs(S_hat)
        T_frames = mag.shape[-1]

        frame_mask = None
        if audio_mask is not None:
            audio_mask = audio_mask[..., :min_len]
            valid_audio = audio_mask.float().sum(dim=-1)
            valid_frames = (1 + valid_audio.long() // self.hop_length).clamp(max=T_frames)
            f_idx = torch.arange(T_frames, device=y.device).unsqueeze(0).expand(y.shape[0], -1)
            frame_mask = (f_idx < valid_frames.unsqueeze(1)).unsqueeze(1).to(mag.dtype)

        # 1. Spectral Convergence
        diff_mag = mag - mag_hat
        if frame_mask is not None:
            diff_mag = diff_mag * frame_mask
            mag_norm = mag * frame_mask
            sc_loss = torch.norm(diff_mag, p="fro", dim=(-2, -1)) / (torch.norm(mag_norm, p="fro", dim=(-2, -1)) + 1e-5)
        else:
            sc_loss = torch.norm(diff_mag, p="fro", dim=(-2, -1)) / (torch.norm(mag, p="fro", dim=(-2, -1)) + 1e-5)
        sc_loss = sc_loss.mean()

        # 2. Log-Magnitude L1
        log_mag = torch.log(mag + 1e-5)
        log_mag_hat = torch.log(mag_hat + 1e-5)
        diff_log = torch.abs(log_mag - log_mag_hat)
        if frame_mask is not None:
            diff_log = diff_log * frame_mask
            log_mag_loss = diff_log.sum() / (frame_mask.sum() * diff_log.shape[1] + 1e-5)
        else:
            log_mag_loss = diff_log.mean()

        # 3. Complex STFT Frobenius difference
        complex_diff = S - S_hat
        complex_err = torch.sqrt(complex_diff.real ** 2 + complex_diff.imag ** 2 + 1e-5)
        if frame_mask is not None:
            complex_err = complex_err * frame_mask
            complex_fro = complex_err.sum() / (frame_mask.sum() * complex_err.shape[1] + 1e-5)
        else:
            complex_fro = complex_err.mean()

        return sc_loss, log_mag_loss, complex_fro


class MultiResolutionSTFTLoss(nn.Module):
    """
    Multi-Resolution STFT Loss evaluated over 3 resolutions:
    - (N_fft=512,  Hop=128, Win=512)
    - (N_fft=1024, Hop=256, Win=1024)
    - (N_fft=2048, Hop=512, Win=2048)
    """

    def __init__(
        self,
        resolutions: Tuple[Tuple[int, int, int], ...] = (
            (512, 128, 512),
            (1024, 256, 1024),
            (2048, 512, 2048)
        )
    ):
        super().__init__()
        self.losses = nn.ModuleList([
            SingleResolutionSTFTLoss(n_fft, hop, win)
            for n_fft, hop, win in resolutions
        ])

    def forward(
        self,
        y: torch.Tensor,
        y_hat: torch.Tensor,
        audio_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        total_loss = 0.0
        for loss_fn in self.losses:
            sc, log_mag, comp = loss_fn(y, y_hat, audio_mask=audio_mask)
            total_loss = total_loss + (sc + log_mag + comp)
        return total_loss / len(self.losses)


class InstantaneousFrequencyLoss(nn.Module):
    """
    Instantaneous Frequency Loss enforcing phase gradient consistency across adjacent STFT frames.
    Uses normalized phasor cosine distance to eradicate NaN gradients and atan2 singularities.
    """

    def __init__(self, n_fft: int = 1024, hop_length: int = 240, win_length: int = 1024):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.register_buffer("window", torch.hann_window(win_length, periodic=True))

    @torch.amp.autocast('cuda', enabled=False)
    def forward(
        self,
        y: torch.Tensor,
        y_hat: torch.Tensor,
        audio_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        y = y.float()
        y_hat = y_hat.float()

        min_len = min(y.shape[-1], y_hat.shape[-1])
        y = y[..., :min_len]
        y_hat = y_hat[..., :min_len]

        S = torch.stft(
            y, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length,
            window=self.window, center=True, normalized=False, onesided=True, return_complex=True
        )
        S_hat = torch.stft(
            y_hat, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length,
            window=self.window, center=True, normalized=False, onesided=True, return_complex=True
        )

        # Cross-frame product across temporal dimension: S(t) * conj(S(t-1))
        diff_t = S[:, :, 1:] * torch.conj(S[:, :, :-1])
        diff_t_hat = S_hat[:, :, 1:] * torch.conj(S_hat[:, :, :-1])

        # Normalized phasors (eliminating torch.angle and atan2 gradient blowups)
        p_gt = diff_t / (torch.abs(diff_t) + 1e-5)
        p_pred = diff_t_hat / (torch.abs(diff_t_hat) + 1e-5)

        # Unit phasor cosine distance: 1.0 - Re(p_gt * conj(p_pred))
        # Re(A * conj(B)) = A.real * B.real + A.imag * B.imag
        cos_sim = p_gt.real * p_pred.real + p_gt.imag * p_pred.imag
        phase_dist = 1.0 - cos_sim  # [B, num_bins, T-1], bounded in [0, 2]

        T_diff = diff_t.shape[-1]
        frame_mask = None
        if audio_mask is not None:
            audio_mask = audio_mask[..., :min_len]
            valid_audio = audio_mask.float().sum(dim=-1)
            valid_frames = (1 + valid_audio.long() // self.hop_length).clamp(max=S.shape[-1])
            valid_diff = (valid_frames - 1).clamp(min=1, max=T_diff)
            f_idx = torch.arange(T_diff, device=y.device).unsqueeze(0).expand(y.shape[0], -1)
            frame_mask = (f_idx < valid_diff.unsqueeze(1)).unsqueeze(1).to(phase_dist.dtype)

        # Weight by magnitude to focus on voiced speech harmonics
        mag_weight = torch.sqrt(torch.abs(S[:, :, 1:]) * torch.abs(S_hat[:, :, 1:]) + 1e-5)
        if frame_mask is not None:
            mag_weight = mag_weight * frame_mask
        mag_norm = mag_weight / (mag_weight.mean(dim=(-2, -1), keepdim=True) + 1e-5)

        weighted_loss = phase_dist * mag_norm
        if frame_mask is not None:
            weighted_loss = weighted_loss * frame_mask
            return weighted_loss.sum() / (frame_mask.sum() * weighted_loss.shape[1] + 1e-5)
        return weighted_loss.mean()


class AeroFlowLoss(nn.Module):
    """
    Unified Non-Adversarial Loss Suite for AeroFlow-v2.
    L_total = L_cfm + lambda_dur * L_dur + lambda_prior * L_prior + lambda_mr_stft * L_mr_stft + lambda_if * L_if
    """

    def __init__(
        self,
        lambda_cfm: float = 1.0,
        lambda_dur: float = 1.0,
        lambda_prior: float = 1.0,
        lambda_mr_stft: float = 1.0,
        lambda_if: float = 0.5
    ):
        super().__init__()
        self.lambda_cfm = lambda_cfm
        self.lambda_dur = lambda_dur
        self.lambda_prior = lambda_prior
        self.lambda_mr_stft = lambda_mr_stft
        self.lambda_if = lambda_if

        self.cfm_loss = CFMLoss()
        self.duration_loss = DurationLoss()
        self.prior_loss = PriorAlignmentLoss()
        self.mr_stft_loss = MultiResolutionSTFTLoss()
        self.if_loss = InstantaneousFrequencyLoss()

    def forward(
        self,
        v_pred: torch.Tensor,
        u_target: torch.Tensor,
        log_dur_pred: torch.Tensor,
        dur_target: torch.Tensor,
        y_audio: torch.Tensor,
        y_hat_audio: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
        frame_mask: Optional[torch.Tensor] = None,
        audio_mask: Optional[torch.Tensor] = None,
        text_proj_expanded: Optional[torch.Tensor] = None,
        z_target: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Computes all loss terms and returns scalar total loss and dictionary of individual metrics.
        """
        l_cfm = self.cfm_loss(v_pred, u_target, mask=frame_mask)
        l_dur = self.duration_loss(log_dur_pred, dur_target, mask=text_mask)

        if text_proj_expanded is not None and z_target is not None:
            l_prior = self.prior_loss(text_proj_expanded, z_target, mask=frame_mask)
        else:
            l_prior = torch.tensor(0.0, device=v_pred.device, dtype=v_pred.dtype)

        l_mr_stft = self.mr_stft_loss(y_audio, y_hat_audio, audio_mask=audio_mask)
        l_if = self.if_loss(y_audio, y_hat_audio, audio_mask=audio_mask)

        l_total = (
            self.lambda_cfm * l_cfm +
            self.lambda_dur * l_dur +
            self.lambda_prior * l_prior +
            self.lambda_mr_stft * l_mr_stft +
            self.lambda_if * l_if
        )

        loss_dict = {
            "loss_total": l_total.detach(),
            "loss_cfm": l_cfm.detach(),
            "loss_dur": l_dur.detach(),
            "loss_prior": l_prior.detach(),
            "loss_mr_stft": l_mr_stft.detach(),
            "loss_if": l_if.detach()
        }

        return l_total, loss_dict
