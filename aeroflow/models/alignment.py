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
    frame t) with the monotonic band constraint precomputed once as a mask.
    Backtracking walks all batch rows together with plain gather/scatter
    (no per-frame fancy indexing) and builds the path via broadcasting.
    Bit-identical to the scalar triple-loop formulation, at a fraction of
    the kernel launches that dominated both CPU and CUDA step time.

    The body is intentionally TorchScript-compatible (no data-dependent
    Python control flow): :func:`scripted_maximum_path` compiles it for the
    training path. Use that on the training path.
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

    b_idx = torch.arange(B, device=device)
    n_grid = torch.arange(N, device=device).view(1, N, 1)
    neg_col = torch.full((B, 1), NEG, dtype=torch.float32, device=device)

    # Base condition: start at (0, 0)
    valid0 = (n_eff > 0) & (t_len > 0)
    Q[:, 0, 0] = torch.where(valid0, neg_cent[:, 0, 0].float(), Q[:, 0, 0])

    # Monotonic band for every frame, computed once (no per-frame mask math):
    # n in [max(0, n_eff - (t_len - t)), min(t + 1, n_eff)]
    tt = torch.arange(T, device=device).view(1, 1, T)
    n_eff_3 = n_eff.view(B, 1, 1)
    t_len_3 = t_len.view(B, 1, 1)
    n_min_all = torch.clamp(n_eff_3 - (t_len_3 - tt), min=0)
    n_max_all = torch.minimum(tt + 1, n_eff_3)
    band = (n_grid >= n_min_all) & (n_grid < n_max_all)

    scores_t = neg_cent.float()
    for t in range(1, T):
        prev = Q[:, :, t - 1]  # [B, N]
        stay = prev
        step = torch.cat([neg_col, prev[:, :-1]], dim=1)
        take_step = step > stay  # ties keep stay (matches scalar `stay >= step`)
        upd = torch.maximum(stay, step) + scores_t[:, :, t]

        in_band = band[:, :, t]
        Q[:, :, t] = torch.where(in_band, upd, Q[:, :, t])
        backtrack[:, :, t] = torch.where(in_band, take_step.to(torch.int64), backtrack[:, :, t])

    # Backward walk from (n_eff - 1, t_len - 1) for all rows at once.
    # Padded frames carry bt == 0 so curr freezes there; invalid rows/frames
    # are zeroed once at the end instead of masked per frame.
    has_content = (n_eff > 0) & (t_len > 0)
    curr = (n_eff - 1).clamp(min=0)  # [B]
    cols = torch.zeros((B, T), dtype=torch.int64, device=device)
    for t in range(T - 1, 0, -1):
        cols[:, t] = curr
        stepped = backtrack[b_idx, curr, t]
        curr = curr - stepped
    cols[:, 0] = curr

    frame_ok = (tt.view(1, T) < t_len.view(B, 1)) & has_content.view(B, 1)
    path = (cols.view(B, 1, T) == n_grid).to(torch.float32)
    path = path * frame_ok.view(B, 1, T).to(torch.float32)

    return path


_scripted_mas = None


def scripted_maximum_path(
    neg_cent: torch.Tensor,
    text_lengths: torch.Tensor,
    audio_lengths: torch.Tensor
) -> torch.Tensor:
    """
    TorchScript-compiled :func:`maximum_path_viterbi` for the training path.

    Compiles lazily on first call (then cached); falls back to the eager
    vectorized implementation if scripting is unavailable. Bit-identical
    output (covered by the brute-force regression test, which runs both).
    """
    global _scripted_mas
    if _scripted_mas is None:
        try:
            _scripted_mas = torch.jit.script(maximum_path_viterbi)
        except Exception:
            _scripted_mas = False
    if _scripted_mas is False:
        return maximum_path_viterbi(neg_cent, text_lengths, audio_lengths)
    return _scripted_mas(neg_cent, text_lengths, audio_lengths)


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

    # Batched path: fully vectorized. Frame f belongs to the smallest
    # phoneme n with cumsum(durations)[n] > f, i.e. seg[b, f] counts how
    # many cumulative ends fall at or before f (zero-duration phonemes can
    # never be selected since their repeated end value keeps them out).
    # A handful of launches total instead of ~5 per batch item.
    total_frames = durations.sum(dim=1)  # [B]
    max_T = int(total_frames.max().item())
    if max_T <= 0:
        return torch.zeros((B, D, 1), device=device, dtype=H_text.dtype)

    ends = durations.cumsum(dim=1)  # [B, N]
    pos = torch.arange(max_T, device=device).view(1, 1, -1)  # [1, 1, T]
    seg = (ends.unsqueeze(2) <= pos).sum(dim=1).clamp(max=N - 1)  # [B, T]
    valid = (pos.squeeze(1) < total_frames.unsqueeze(1)).to(H_text.dtype)  # [B, T]
    C = H_text.gather(1, seg.unsqueeze(-1).expand(B, max_T, D)) * valid.unsqueeze(-1)
    return C.transpose(1, 2)  # [B, D, max_T]
