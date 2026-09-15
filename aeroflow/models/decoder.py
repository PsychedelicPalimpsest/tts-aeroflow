"""
AeroFlow-v2 ConvNeXt-V2 Complex STFT Decoder.
Features:
- Global Response Normalization (GRN) to enhance feature diversity.
- 4 ConvNeXt-V2 blocks mapping 32-channel continuous latents to complex STFT spectra.
- Direct prediction of full-band Log-Magnitude and Continuous Phase Unit Vectors (cos phi, sin phi).
- Guaranteed unit-norm phase projection eliminating phase wrap discontinuities.
"""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class GRN(nn.Module):
    """Global Response Normalization (GRN) layer for ConvNeXt-V2."""

    def __init__(self, dim: int):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1))
        self.beta = nn.Parameter(torch.zeros(1, dim, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, dim, T]
        # Compute L2 norm across temporal dimension
        Gx = torch.norm(x, p=2, dim=-1, keepdim=True)  # [B, dim, 1]
        Nx = Gx / (Gx.mean(dim=1, keepdim=True) + 1e-5)  # [B, dim, 1]
        return self.gamma * (x * Nx) + self.beta + x


class ConvNeXtV2Block(nn.Module):
    """ConvNeXt-V2 block with depthwise convolution and GRN."""

    def __init__(self, dim: int = 256, kernel_size: int = 7):
        super().__init__()
        self.dwconv = nn.Conv1d(
            dim, dim, kernel_size=kernel_size,
            padding=kernel_size // 2, groups=dim
        )
        self.norm = nn.LayerNorm(dim)
        self.pwconv1 = nn.Conv1d(dim, dim * 4, kernel_size=1)
        self.act = nn.GELU()
        self.grn = GRN(dim * 4)
        self.pwconv2 = nn.Conv1d(dim * 4, dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, dim, T]
        res = x
        x = self.dwconv(x)
        x = self.norm(x.transpose(1, 2)).transpose(1, 2)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        return res + x


class ComplexSTFTDecoder(nn.Module):
    """
    Decodes 32-channel acoustic latents to full-band continuous complex STFT spectra.
    Outputs Log-Magnitude and Unit Phase vectors (cos phi, sin phi).
    """

    def __init__(
        self,
        latent_dim: int = 32,
        model_dim: int = 256,
        num_bins: int = 513,
        num_blocks: int = 4
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.model_dim = model_dim
        self.num_bins = num_bins

        self.in_proj = nn.Conv1d(latent_dim, model_dim, kernel_size=1)
        self.blocks = nn.ModuleList([
            ConvNeXtV2Block(model_dim, kernel_size=7)
            for _ in range(num_blocks)
        ])
        # Outputs: Log-Mag (num_bins) + Phase Real (num_bins) + Phase Imag (num_bins)
        self.out_proj = nn.Conv1d(model_dim, num_bins * 3, kernel_size=1)

    def forward(
        self,
        z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        z: [B, 32, T] acoustic latent tensor
        Returns:
            S_complex: [B, num_bins, T] complex STFT tensor
            mag: [B, num_bins, T] linear magnitude
            cos_phi: [B, num_bins, T] unit-norm real phase component
            sin_phi: [B, num_bins, T] unit-norm imaginary phase component
        """
        h = self.in_proj(z)
        for block in self.blocks:
            h = block(h)
        out = self.out_proj(h)

        log_mag, pr, pi = out.chunk(3, dim=1)

        # Clamped exponential for numerical stability (prevent overflow/underflow)
        log_mag_clamped = torch.clamp(log_mag, min=-10.0, max=9.0)
        mag = torch.exp(log_mag_clamped)

        # Unit-vector projection for continuous phase angle (eliminates phase wrapping)
        phase_norm = torch.clamp(torch.sqrt(pr ** 2 + pi ** 2), min=1e-4)
        cos_phi = pr / phase_norm
        sin_phi = pi / phase_norm

        real = mag * cos_phi
        imag = mag * sin_phi
        S_complex = torch.complex(real, imag)

        return S_complex, mag, cos_phi, sin_phi
