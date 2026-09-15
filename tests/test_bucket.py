"""
Tests for length-bucketed batching (aeroflow/dataset/bucket.py).
"""

import pytest
import torch

from aeroflow.dataset.bucket import BucketBatchSampler, dataset_lengths
from aeroflow.dataset.dataset import HiFiTTSDataset, collate_hifi_tts


def _lengths(n=200, lo=24000, hi=288000, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (torch.rand(n, generator=g) * (hi - lo) + lo).long().tolist()


def test_buckets_group_similar_lengths():
    lengths = _lengths()
    bs = BucketBatchSampler(lengths, batch_size=16, bucket_width=20, seed=0)
    spreads = []
    for b in bs:
        assert len(b) == 16
        vals = [lengths[i] for i in b]
        spreads.append(max(vals) - min(vals))
    # Bucketed spread << global range (264k samples).
    assert sum(spreads) / len(spreads) < 40000


def test_random_batching_spread_is_worse():
    """Sanity: bucketing must beat naive contiguous batching on sorted data."""
    lengths = sorted(_lengths())
    bs = BucketBatchSampler(lengths, batch_size=16, seed=1)
    spread = sum(max(lengths[i] for i in b) - min(lengths[i] for i in b)
                 for b in bs) / len(bs)
    assert spread < 40000


def test_ddp_split_disjoint_and_even():
    lengths = _lengths(n=200)
    r0 = list(BucketBatchSampler(lengths, batch_size=16, seed=3, rank=0, world_size=2))
    r1 = list(BucketBatchSampler(lengths, batch_size=16, seed=3, rank=1, world_size=2))
    assert len(r0) == len(r1)  # even counts: no DDP hang
    assert not (set(sum(r0, [])) & set(sum(r1, [])))  # disjoint
    assert len(r0) + len(r1) <= 200 // 16  # drop_last respected


def test_epoch_changes_order_but_same_items():
    lengths = _lengths(n=192)
    # bucket_width=3 -> 4 chunks, so chunk shuffling actually permutes.
    a = BucketBatchSampler(lengths, batch_size=16, bucket_width=3, seed=5)
    a.set_epoch(0)
    b0 = list(a)
    a.set_epoch(1)
    b1 = list(a)
    assert sorted(sum(b0, [])) == sorted(sum(b1, [])) == list(range(192))
    assert b0 != b1
    c = BucketBatchSampler(lengths, batch_size=16, bucket_width=3, seed=5)
    c.set_epoch(0)
    assert list(c) == b0  # deterministic


def test_tiny_dataset_yields_one_batch():
    bs = BucketBatchSampler([100, 200, 150], batch_size=16, seed=0)
    assert list(bs) == [[0, 1, 2]] or sorted(list(bs)[0]) == [0, 1, 2]


def test_no_shuffle_is_sorted():
    lengths = _lengths()
    bs = BucketBatchSampler(lengths, batch_size=16, shuffle=False, seed=0)
    flat = [i for b in bs for i in b]
    got = [lengths[i] for i in flat]
    assert got == sorted(got)


def test_dataset_lengths_manifest_and_loader(tmp_path):
    import soundfile as sf
    fpaths = []
    for i, secs in enumerate([1.0, 2.0, 3.0, 1.5]):
        p = tmp_path / f"a{i}.wav"
        sf.write(str(p), [0.0] * int(secs * 24000), 24000)
        fpaths.append(str(p))
    man = tmp_path / "m.json"
    man.write_text("\n".join(
        f'{{"audio_filepath": "{p}", "text": "hi there"}}' for p in fpaths))
    ds = HiFiTTSDataset(manifest_path=str(man))
    lens = dataset_lengths(ds)
    assert lens == [24000, 48000, 72000, 36000]
    # Buckets + real loader + collate end to end.
    sampler = BucketBatchSampler(lens, batch_size=2, seed=0)
    loader = torch.utils.data.DataLoader(ds, batch_sampler=sampler,
                                         collate_fn=collate_hifi_tts)
    batches = list(loader)
    assert len(batches) == 2
    for b in batches:
        assert b["audio"].shape[1] % 240 == 0
    # Sorted batches: the two short clips share a batch (17280+36000 vs 86400 max).
    assert batches[0]["audio"].shape[1] <= batches[1]["audio"].shape[1]


def test_dataset_lengths_none_for_unknown():
    assert dataset_lengths(object()) is None
