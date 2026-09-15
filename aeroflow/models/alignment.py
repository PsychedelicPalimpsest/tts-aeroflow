"""
AeroFlow-v2 Monotonic Alignment & Duration Modeling.
Contains:
1. Dynamic Programming Viterbi Monotonic Alignment Search (MAS).
2. Energy-Constrained Monotonic Duration Predictor with clamped bounds [d_min=1, d_max=80].
3. Monotonic Expansion module for mapping phoneme embeddings to frame conditioning.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def maximum_path_viterbi(
    neg_cent: torch.Tensor,
    text_lengths: torch.Tensor,
    audio_lengths: torch.Tensor
) -> torch.Tensor:
    """
    Vectorized dynamic programming for Monotonic Alignment Search (Viterbi MAS).
    neg_cent: [B, N, T] log-likelihood or negative distance matrix between phonemes N and frames T.
    text_lengths: [B] number of phonemes per sequence
    audio_lengths: [B] number of frames per sequence
    Returns:
        path: [B, N, T] binary alignment matrix with exactly one 1 per column (audio frame).

    The DP is vectorized over phonemes N and batch B (one [B, N] sweep per
    frame t) with the monotonic band constraint applied as a mask. This is
    bit-identical to the scalar triple-loop formulation but ~100x faster,
    since per-cell kernel launches dominated both CPU and CUDA step time.
    """
    B, N, T = neg_cent.shape
    device = neg_cent.device
    NEG = -1e9

    n_len = text_lengths.to(device).long().clamp(min=0)
    t_len = audio_lengths.to(device).long().clamp(min=0)
    # Guard against shorter audio than phonemes (guarantee t_len >= n_len)
    n_eff = torch.minimum(n_len, t_len)

    Q = torch.full((B, N, T), NEG, dtype=torch.float32, device=device)
    backtrack = torch.zeros((B, N, T), dtype=torch.int64, device=device)
    path = torch.zeros((B, N, T), dtype=torch.float32, device=device)

    b_idx = torch.arange(B, device=device)
    n_grid = torch.arange(N, device=device).view(1, N)
    neg_col = torch.full((B, 1), NEG, dtype=torch.float32, device=device)

    # Base condition: start at (0, 0)
    valid0 = (n_eff > 0) & (t_len > 0)
    if valid0.any():
        vb = b_idx[valid0]
        Q[vb, 0, 0] = neg_cent[vb, 0, 0].float()

    scores_t = neg_cent.float()
    for t in range(1, T):
        active_t = t < t_len  # [B]
        if not bool(active_t.any()):
            break
        # Monotonic band: n in [max(0, n_eff - (t_len - t)), min(t + 1, n_eff)]
        n_min = torch.clamp(n_eff - (t_len - t), min=0).unsqueeze(1)  # [B, 1]
        n_max = torch.minimum(torch.full_like(n_eff, t + 1), n_eff).unsqueeze(1)
        in_band = (n_grid >= n_min) & (n_grid < n_max) & active_t.unsqueeze(1)

        prev = Q[:, :, t - 1]  # [B, N]
        stay = prev
        step = torch.cat([neg_col, prev[:, :-1]], dim=1)
        take_step = step > stay  # ties keep stay (matches scalar `stay >= step`)
        upd = torch.maximum(stay, step) + scores_t[:, :, t]

        Q[:, :, t] = torch.where(in_band, upd, Q[:, :, t])
        backtrack[:, :, t] = torch.where(in_band, take_step.to(torch.int64), backtrack[:, :, t])

    # Backtrack from (n_eff - 1, t_len - 1), vectorized over the batch
    curr = (n_eff - 1).clamp(min=0)  # [B]
    has_content = (n_eff > 0) & (t_len > 0)
    for t in range(T - 1, -1, -1):
        active = has_content & (t < t_len)
        if not bool(active.any()):
            continue
        ab = b_idx[active]
        cn = curr[active]
        path[ab, cn, t] = 1.0
        if t > 0:
            stepped = backtrack[ab, cn, t] == 1
            dec_idx = ab[stepped]
            curr[dec_idx] = (curr[dec_idx] - 1).clamp(min=0)

    return path


def alignment_to_durations(
    path: torch.Tensor,
    text_lengths: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Extracts integer duration counts per phoneme from alignment path matrix.
    path: [B, N, T]
    text_lengths: [B] optional valid phoneme lengths
    Returns:
        durations: [B, N] integer durations d_n >= 0 (strictly >= 1 for valid path).
    """
    durations = path.sum(dim=-1)
    if text_lengths is not None:
        B, N, _ = path.shape
        idx = torch.arange(N, device=path.device).unsqueeze(0).expand(B, N)
        mask = idx < text_lengths.unsqueeze(1)
        durations = torch.where(mask, durations.clamp(min=1.0), torch.zeros_like(durations))
    else:
        durations = durations.clamp(min=1.0)
    return durations


class EnergyConstrainedDurationPredictor(nn.Module):
    """
    Predicts phoneme duration from text representations with hard quantile bounds:
    d_min >= 1 frame (10 ms), d_max <= 80 frames (800 ms).
    """

    def __init__(self, text_dim: int = 192, hidden_dim: int = 192, kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(text_dim, hidden_dim, kernel_size=kernel_size, padding=kernel_size // 2)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=kernel_size, padding=kernel_size // 2)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.proj = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        H_text: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Predicts raw log durations.
        H_text: [B, N, text_dim]
        mask: [B, N] boolean mask
        Returns:
            log_durations: [B, N]
        """
        # Conv operates on [B, text_dim, N]
        x = H_text.transpose(1, 2)
        x = self.conv1(x).transpose(1, 2)
        x = F.relu(self.norm1(x))
        x = self.dropout(x)

        x = x.transpose(1, 2)
        x = self.conv2(x).transpose(1, 2)
        x = F.relu(self.norm2(x))
        x = self.dropout(x)

        log_dur = self.proj(x).squeeze(-1)  # [B, N]

        if mask is not None:
            log_dur = log_dur.masked_fill(~mask, 0.0)

        return log_dur

    def predict_durations(
        self,
        H_text: torch.Tensor,
        alpha: float = 1.0,
        mask: Optional[torch.Tensor] = None,
        d_min: int = 1,
        d_max: int = 80
    ) -> torch.Tensor:
        """
        Quantizes predicted log durations into integer frames with anti-collapse bounds.
        d_min: Minimum frames per phoneme (default 1 frame = 10 ms @ 100 Hz).
        d_max: Maximum frames per phoneme (default 80 frames = 800 ms).
        alpha: Speaking rate multiplier (1.0 = normal, >1.0 = slower, <1.0 = faster).
        Returns:
            quantized_durations: [B, N] torch.LongTensor
        """
        log_dur = self.forward(H_text, mask=mask)
        raw_dur = torch.exp(log_dur) * alpha
        quantized = torch.floor(raw_dur + 0.5)
        clamped = torch.clamp(quantized, min=float(d_min), max=float(d_max))

        if mask is not None:
            clamped = clamped.masked_fill(~mask, 0.0)

        return clamped.long()


def expand_text_representations(
    H_text: torch.Tensor,
    durations: torch.Tensor
) -> torch.Tensor:
    """
    Monotonically expands text representations to frame-level conditioning C.
    H_text: [B, N, D]
    durations: [B, N] integer frame counts
    Returns:
        C: [B, D, T] where T = max(sum(durations))
    """
    B, N, D = H_text.shape
    device = H_text.device

    if B == 1:
        # Fast path for single item inference
        d = durations[0]
        valid_idx = d > 0
        h_valid = H_text[0, valid_idx]
        d_valid = d[valid_idx]
        if len(d_valid) == 0:
            return torch.zeros((1, D, 1), device=device)
        C = torch.repeat_interleave(h_valid, d_valid, dim=0).unsqueeze(0)  # [1, T, D]
        return C.transpose(1, 2)  # [1, D, T]

    # Batched path: expand each batch item and pad to max_T
    total_frames = durations.sum(dim=1)  # [B]
    max_T = int(total_frames.max().item())

    expanded_list = []
    for b in range(B):
        d_b = durations[b]
        # Only take valid tokens with d > 0
        valid_idx = d_b > 0
        h_valid = H_text[b, valid_idx]
        d_valid = d_b[valid_idx]

        if len(d_valid) == 0:
            c_b = torch.zeros((max_T, D), device=device)
        else:
            c_b = torch.repeat_interleave(h_valid, d_valid, dim=0)  # [T_b, D]
            if c_b.shape[0] < max_T:
                pad_t = max_T - c_b.shape[0]
                c_b = F.pad(c_b, (0, 0, 0, pad_t))
            elif c_b.shape[0] > max_T:
                c_b = c_b[:max_T]
        expanded_list.append(c_b)

    C = torch.stack(expanded_list, dim=0)  # [B, max_T, D]
    return C.transpose(1, 2)  # [B, D, max_T]
