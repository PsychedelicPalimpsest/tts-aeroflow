"""
Length-bucketed batching for variable-length TTS training.

Problem: Hi-Fi TTS clips range 0.3-19 s. Random batches pad every item to
the longest clip, so one 12 s clip forces 288k-sample tensors (and 1201 MAS
frames) on the whole batch. Step times swing 5x batch-to-batch and the mean
is dragged up by padding waste (measured x1.77 memory at batch 256).

Fix: sort by length, group into similar-length buckets, shuffle bucket
order each epoch. Every batch is then near-uniform length: minimal padding,
stable step times, faster mean. Standard practice in ASR/TTS training.

Only works for map-style datasets with known lengths (streaming has no
lengths upfront). DDP-aware: all ranks build the identical global batch
list from (seed, epoch), then take a strided slice - no communication.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import torch
from torch.utils.data import Sampler

__all__ = ["BucketBatchSampler", "dataset_lengths"]


class BucketBatchSampler(Sampler[List[int]]):
    """
    Yields batches of similarly-sized items.

    Args:
        lengths: per-item audio length (samples); index-aligned with dataset.
        batch_size: items per batch.
        bucket_width: buckets hold ``batch_size * bucket_width`` items;
            larger = tighter length grouping but less shuffling.
        shuffle: shuffle bucket order each epoch (within-bucket order stays
            sorted so batches stay uniform).
        seed: base RNG seed (combined with epoch).
        rank / world_size: DDP sharding; rank r takes batches[r::world_size].
        drop_last: drop partial batches AND trim the tail so every rank gets
            the same batch count (required: DDP hangs on uneven step counts).
    """

    def __init__(
        self,
        lengths: Sequence[int],
        batch_size: int,
        bucket_width: int = 20,
        shuffle: bool = True,
        seed: int = 42,
        rank: int = 0,
        world_size: int = 1,
        drop_last: bool = True,
    ):
        super().__init__()
        if len(lengths) == 0:
            raise ValueError("BucketBatchSampler needs non-empty lengths")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.lengths = [int(v) for v in lengths]
        self.batch_size = batch_size
        self.bucket_width = max(1, bucket_width)
        self.shuffle = shuffle
        self.seed = seed
        self.rank = rank
        self.world_size = max(1, world_size)
        self.drop_last = drop_last
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def _global_batches(self) -> List[List[int]]:
        n = len(self.lengths)
        order = sorted(range(n), key=self.lengths.__getitem__)
        chunk = self.batch_size * self.bucket_width
        chunks = [order[i: i + chunk] for i in range(0, n, chunk)]
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self._epoch)
            perm = torch.randperm(len(chunks), generator=g).tolist()
            chunks = [chunks[i] for i in perm]
        batches: List[List[int]] = []
        for ch in chunks:
            for i in range(0, len(ch), self.batch_size):
                piece = ch[i: i + self.batch_size]
                if len(piece) == self.batch_size or not self.drop_last:
                    batches.append(piece)
        if not batches:  # fewer items than one batch: yield all at once
            batches = [order]
        if self.drop_last and self.world_size > 1:
            usable = (len(batches) // self.world_size) * self.world_size
            if usable:
                batches = batches[:usable]
            else:
                # Fewer batches than ranks: every rank repeats the one batch.
                # (Identical input on all ranks averages correctly; a rank
                # with zero batches would hang DDP instead.)
                batches = [batches[0]] * self.world_size
        return batches

    def __iter__(self):
        batches = self._global_batches()
        return iter(batches[self.rank:: self.world_size])

    def __len__(self) -> int:
        batches = self._global_batches()
        mine = batches[self.rank:: self.world_size]
        return len(mine)


def dataset_lengths(dataset) -> Optional[List[int]]:
    """
    Best-effort per-item audio lengths (samples) for bucketing.

    Understands ``audio_lengths`` (HF map adapter), ``get_audio_lengths()``
    (manifest adapter, header-only scan), and ``audio_lengths`` on the
    synthetic fallback. Returns None when unavailable (e.g. streaming).
    """
    if hasattr(dataset, "audio_lengths"):
        return [int(v) for v in dataset.audio_lengths]
    getter = getattr(dataset, "get_audio_lengths", None)
    if callable(getter):
        return [int(v) for v in getter()]
    return None
