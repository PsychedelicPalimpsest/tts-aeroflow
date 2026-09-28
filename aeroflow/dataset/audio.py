"""Shared, band-limited audio preprocessing for local and Hub datasets."""

import math

import numpy as np
from scipy.signal import resample_poly


def resample_mono(waveform: np.ndarray, src_sr: int, tgt_sr: int = 24000) -> np.ndarray:
    """Mix channels, then low-pass before resampling; preserve floor-length convention."""
    if src_sr <= 0 or tgt_sr <= 0:
        raise ValueError("Sample rates must be positive")
    data = np.asarray(waveform, dtype=np.float32)
    if data.ndim not in (1, 2):
        raise ValueError("Expected [samples] or [samples, channels] audio")
    if data.ndim == 2:
        data = data.mean(axis=-1)
    if src_sr == tgt_sr or data.size == 0:
        return data
    length = len(data) * tgt_sr // src_sr
    if length == 0:
        return np.empty(0, dtype=np.float32)
    divisor = math.gcd(src_sr, tgt_sr)
    return resample_poly(data, tgt_sr // divisor, src_sr // divisor)[:length].astype(np.float32)
