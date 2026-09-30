"""Length-aware acoustic operations, retaining existing checkpoint parameters."""

import torch


def mask_frames(x, mask):
    return x if mask is None else x * mask[:, None, :].to(x.dtype)


def masked_group_norm(x, norm, mask):
    """GroupNorm(1, C) using only valid frames, with FP32 statistics under AMP."""
    if mask is None:
        return norm(x)
    valid = mask[:, None, :].to(torch.float32)
    value = x.float()
    count = (valid.sum(dim=-1, keepdim=True) * x.shape[1]).clamp_min(1)
    mean = (value * valid).sum(dim=(1, 2), keepdim=True) / count
    variance = ((value - mean).square() * valid).sum(dim=(1, 2), keepdim=True) / count
    value = (value - mean) * torch.rsqrt(variance + norm.eps)
    value = value * norm.weight[None, :, None] + norm.bias[None, :, None]
    return (value * valid).to(x.dtype)


def frame_mask(lengths, frames, hop_length):
    return torch.arange(frames, device=lengths.device)[None] < (1 + lengths[:, None] // hop_length)
