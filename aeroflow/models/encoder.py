"""
AeroFlow-v2 Conformer Phoneme Encoder with Rotary Position Embeddings (RoPE).
Features:
- Embedding layer mapping 84 phoneme tokens to d_model=192.
- 4 Conformer Blocks with Macaron-style Feed-Forward networks.
- Multi-Head Self-Attention with Rotary Position Embeddings (RoPE).
- Depthwise-Separable Convolution modules with Gated Linear Units (GLU).
- Zero padding / attention masking support.
"""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) for relative position awareness."""

    def __init__(self, dim: int, max_seq_len: int = 4096, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, dtype=torch.float32, device=self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq)  # [seq_len, dim // 2]
        # Repeat freqs to match [seq_len, dim]
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # q, k: [B, num_heads, N, head_dim]
        seq_len = q.shape[-2]
        if seq_len > self.cos_cached.shape[0]:
            self._build_cache(max(seq_len, self.cos_cached.shape[0] * 2))

        cos = self.cos_cached[:seq_len, :].to(q.dtype).unsqueeze(0).unsqueeze(1)
        sin = self.sin_cached[:seq_len, :].to(q.dtype).unsqueeze(0).unsqueeze(1)

        def rotate_half(x: torch.Tensor) -> torch.Tensor:
            x1 = x[..., : self.dim // 2]
            x2 = x[..., self.dim // 2 :]
            return torch.cat((-x2, x1), dim=-1)

        q_rot = (q * cos) + (rotate_half(q) * sin)
        k_rot = (k * cos) + (rotate_half(k) * sin)
        return q_rot, k_rot


class MultiHeadSelfAttentionRoPE(nn.Module):
    """Multi-Head Self-Attention with Rotary Positional Embeddings."""

    def __init__(self, d_model: int = 192, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.rope = RotaryEmbedding(self.head_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: [B, N, d_model]
        # mask: [B, N] (1 for valid tokens, 0 for pad)
        B, N, C = x.shape
        q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE
        q, k = self.rope(q, k)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [B, H, N, N]

        if mask is not None:
            # mask: [B, N] -> [B, 1, 1, N]
            attn_mask = mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, N]
            scores = scores.masked_fill(attn_mask == 0, -1e9)

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)  # [B, H, N, head_dim]
        out = out.transpose(1, 2).contiguous().view(B, N, C)
        return self.out_proj(out)


class FeedForwardModule(nn.Module):
    """Macaron-style Feed-Forward Network."""

    def __init__(self, d_model: int = 192, expansion: int = 4, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_model * expansion)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_model * expansion, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(self.act(self.linear1(x))))


class ConformerConvModule(nn.Module):
    """Conformer Convolution module with GLU and Depthwise Conv1D."""

    def __init__(self, d_model: int = 192, kernel_size: int = 7, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.pointwise_conv1 = nn.Conv1d(d_model, d_model * 2, kernel_size=1)
        self.glu = nn.GLU(dim=1)
        self.depthwise_conv = nn.Conv1d(
            d_model, d_model, kernel_size=kernel_size,
            padding=kernel_size // 2, groups=d_model
        )
        self.norm2 = nn.LayerNorm(d_model)  # Token-wise LayerNorm over channels
        self.act = nn.GELU()
        self.pointwise_conv2 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: [B, N, d_model]
        h = self.norm(x)
        if mask is not None:
            h = h * mask.unsqueeze(-1).to(h.dtype)
        h = h.transpose(1, 2)  # [B, d_model, N]
        h = self.pointwise_conv1(h)
        h = self.glu(h)
        h = self.depthwise_conv(h)
        h = self.norm2(h.transpose(1, 2)).transpose(1, 2)
        h = self.act(h)
        h = self.pointwise_conv2(h)
        h = self.dropout(h)
        out = h.transpose(1, 2)  # [B, N, d_model]
        if mask is not None:
            out = out * mask.unsqueeze(-1).to(out.dtype)
        return out


class ConformerBlock(nn.Module):
    """Single Conformer block with Macaron FFN, RoPE-Attention, and Conv Module."""

    def __init__(self, d_model: int = 192, num_heads: int = 4, kernel_size: int = 7, dropout: float = 0.1):
        super().__init__()
        self.ffn1 = FeedForwardModule(d_model, expansion=4, dropout=dropout)
        self.norm_ffn1 = nn.LayerNorm(d_model)

        self.attn = MultiHeadSelfAttentionRoPE(d_model, num_heads=num_heads, dropout=dropout)
        self.norm_attn = nn.LayerNorm(d_model)

        self.conv = ConformerConvModule(d_model, kernel_size=kernel_size, dropout=dropout)

        self.ffn2 = FeedForwardModule(d_model, expansion=4, dropout=dropout)
        self.norm_ffn2 = nn.LayerNorm(d_model)

        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 1. Macaron FFN half-step
        x = x + 0.5 * self.ffn1(self.norm_ffn1(x))

        # 2. Self-Attention with RoPE
        x = x + self.attn(self.norm_attn(x), mask=mask)

        # 3. Convolution Module
        x = x + self.conv(x, mask=mask)

        # 4. Macaron FFN second half-step
        x = x + 0.5 * self.ffn2(self.norm_ffn2(x))

        return self.final_norm(x)


class ConformerEncoder(nn.Module):
    """
    Complete Conformer Phoneme Encoder.
    Maps phoneme integer IDs in {0..83} to continuous representations H_text in R^(B x N x 192).
    """

    def __init__(
        self,
        vocab_size: int = 84,
        d_model: int = 192,
        num_blocks: int = 4,
        num_heads: int = 4,
        kernel_size: int = 7,
        dropout: float = 0.1
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.scale = math.sqrt(d_model)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            ConformerBlock(d_model=d_model, num_heads=num_heads, kernel_size=kernel_size, dropout=dropout)
            for _ in range(num_blocks)
        ])
        self.out_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        tokens: torch.Tensor,
        lengths: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        tokens: [B, N] phoneme token IDs
        lengths: [B] sequence lengths (optional)
        Returns:
            H_text: [B, N, d_model]
            mask: [B, N] boolean mask (True for valid tokens)
        """
        B, N = tokens.shape
        mask = None
        if lengths is not None:
            # Create [B, N] mask
            idx = torch.arange(N, device=tokens.device).unsqueeze(0).expand(B, N)
            mask = idx < lengths.unsqueeze(1)
        elif (tokens == 0).any():
            mask = (tokens != 0)

        x = self.embedding(tokens) * self.scale
        x = self.dropout(x)

        for block in self.blocks:
            x = block(x, mask=mask)

        x = self.out_norm(x)

        if mask is not None:
            x = x * mask.unsqueeze(-1).to(x.dtype)

        return x, mask
