"""
AeroFlow-v2 Optimal Transport Flow-Matching Engine & Non-Uniform Heun ODE Solver.
Features:
- Adaptive Layer Normalization (AdaLN) conditioned on time and phoneme context.
- ConvNeXt-ODE Vector Field with dilated depthwise convolutions and residual scaling.
- Optimal Transport conditional flow matching trajectory sampling.
- 6-step Non-Uniform Heun ODE Solver with polynomial schedule (rho = 1.5).
"""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaLN(nn.Module):
    """Adaptive Layer Normalization conditioned on time and phoneme context."""

    def __init__(self, dim: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.proj = nn.Linear(cond_dim, dim * 2)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T], cond: [B, cond_dim, T]
        x_t = x.transpose(1, 2)
        cond_t = cond.transpose(1, 2)
        scale, shift = self.proj(cond_t).chunk(2, dim=-1)
        x_norm = self.norm(x_t) * (1.0 + scale) + shift
        return x_norm.transpose(1, 2)


class ConvNeXtODEBlock(nn.Module):
    """Dilated Depthwise-Separable ConvNeXt block for ODE Vector Field."""

    def __init__(
        self,
        dim: int = 192,
        cond_dim: int = 192,
        kernel_size: int = 7,
        dilation: int = 1
    ):
        super().__init__()
        self.dwconv = nn.Conv1d(
            dim, dim,
            kernel_size=kernel_size,
            padding=dilation * (kernel_size - 1) // 2,
            dilation=dilation,
            groups=dim
        )
        self.adaln = AdaLN(dim, cond_dim)
        self.pwconv1 = nn.Conv1d(dim, dim * 4, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv1d(dim * 4, dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.full((dim, 1), 1e-6))

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        res = x
        x = self.dwconv(x)
        x = self.adaln(x, cond)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        return res + self.gamma * x


class VectorFieldNetwork(nn.Module):
    """6-Block ConvNeXt-ODE predicting continuous vector velocities in R^32."""

    def __init__(
        self,
        latent_dim: int = 32,
        model_dim: int = 192,
        num_blocks: int = 6
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.model_dim = model_dim
        self.in_proj = nn.Conv1d(latent_dim, model_dim, kernel_size=1)
        self.time_mlp = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, model_dim)
        )
        dilations = [1, 2, 4, 1, 2, 4]
        self.blocks = nn.ModuleList([
            ConvNeXtODEBlock(
                model_dim, model_dim,
                dilation=dilations[i % len(dilations)]
            )
            for i in range(num_blocks)
        ])
        self.out_norm = nn.LayerNorm(model_dim)
        self.out_proj = nn.Conv1d(model_dim, latent_dim, kernel_size=1)

    def _sinusoidal_embedding(self, t: torch.Tensor, dim: int) -> torch.Tensor:
        # t: [B]
        half_dim = dim // 2
        freqs = torch.exp(
            -torch.arange(half_dim, dtype=torch.float32, device=t.device) *
            (math.log(10000.0) / (half_dim - 1))
        )
        args = t[:, None] * freqs[None, :]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        C: torch.Tensor
    ) -> torch.Tensor:
        """
        x_t: [B, 32, T] latent at continuous time t
        t: [B] time in [0, 1]
        C: [B, 192, T] frame-level conditioning
        Returns:
            v: [B, 32, T] predicted vector velocity
        """
        t_emb = self.time_mlp(self._sinusoidal_embedding(t, self.model_dim)).unsqueeze(-1)
        cond = C + t_emb  # Broadcast time embedding across temporal frames
        h = self.in_proj(x_t)
        for block in self.blocks:
            h = block(h, cond)
        h = self.out_norm(h.transpose(1, 2)).transpose(1, 2)
        return self.out_proj(h)


class NonUniformHeunSolver:
    """
    Second-order Heun solver with power-law schedule to preserve high-frequency transients.
    rho = 1.5 concentrates evaluation steps near t -> 1.0.
    """

    def __init__(self, num_steps: int = 6, rho: float = 1.5):
        self.num_steps = num_steps
        self.rho = rho
        # Polynomial time schedule: t_k = 1 - (1 - k/N)^rho
        steps = torch.linspace(0, 1, num_steps + 1)
        self.t_schedule = 1.0 - torch.pow(1.0 - steps, rho)

    @torch.no_grad()
    def solve(
        self,
        v_theta: nn.Module,
        x_0: torch.Tensor,
        C: torch.Tensor
    ) -> torch.Tensor:
        """
        Executes 6-step Heun integration from prior x_0 to target acoustic latent z.
        x_0: [B, 32, T]
        C: [B, 192, T]
        Returns:
            z: [B, 32, T] synthesized acoustic latents
        """
        x = x_0
        batch_size = x.shape[0]
        device = x.device

        for k in range(self.num_steps):
            t_curr = torch.full((batch_size,), self.t_schedule[k].item(), device=device)
            t_next = torch.full((batch_size,), self.t_schedule[k + 1].item(), device=device)
            dt = (self.t_schedule[k + 1] - self.t_schedule[k]).item()

            # Predictor step (Euler forward)
            k1 = v_theta(x, t_curr, C)
            x_pred = x + dt * k1

            # Corrector step (Trapezoidal quadrature)
            k2 = v_theta(x_pred, t_next, C)
            x = x + 0.5 * dt * (k1 + k2)

        return x


class OptimalTransportCFM(nn.Module):
    """
    Optimal Transport Conditional Flow Matching (OT-CFM) helper.
    Computes straight-line interpolation trajectories and target velocity fields.
    """

    def __init__(self, sigma_min: float = 1e-4):
        super().__init__()
        self.sigma_min = sigma_min

    def sample_trajectory(
        self,
        z: torch.Tensor,
        x_0: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        z: [B, 32, T] ground-truth acoustic latents
        x_0: [B, 32, T] prior noise (optional, sampled if None)
        Returns:
            x_t: [B, 32, T] interpolated latent state
            t: [B] sampled timesteps in [0, 1]
            u_t: [B, 32, T] target velocity vector
            x_0: [B, 32, T] initial noise
        """
        B, C, T = z.shape
        device = z.device

        if x_0 is None:
            x_0 = torch.randn_like(z)

        # Sample continuous time t uniformly in [0, 1]
        t = torch.rand(B, device=device)
        t_expand = t.view(B, 1, 1)

        # OT straight path: x_t = (1 - (1 - sigma_min) * t) * x_0 + t * z
        x_t = (1.0 - (1.0 - self.sigma_min) * t_expand) * x_0 + t_expand * z

        # Target conditional velocity field: u_t = z - (1 - sigma_min) * x_0
        u_t = z - (1.0 - self.sigma_min) * x_0

        return x_t, t, u_t, x_0
