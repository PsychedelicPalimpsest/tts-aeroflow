"""
Tests for the local LJSpeech adapter (metadata.csv + audio/ or wavs/).
"""

import numpy as np
import pytest
import soundfile as sf
import torch

from aeroflow.dataset.dataset import collate_hifi_tts
from aeroflow.dataset.hf_hifi_tts import create_hifi_tts_dataset
from aeroflow.dataset.ljspeech import (
    LJSpeechDataset,
    parse_ljspeech_row,
)


def _write_wav(path, seconds=1.0, sr=22050, freq=220.0):
    path.parent.mkdir(parents=True, exist_ok=True)
    n = int(seconds * sr)
    t = np.linspace(0, seconds, n, endpoint=False).astype(np.float32)
    wave = (0.4 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    sf.write(str(path), wave, sr)


def _make_root(tmp_path, audio_subdir="audio"):
    root = tmp_path / "LJSpeech-1.1"
    _write_wav(root / audio_subdir / "LJ001-0001.wav", seconds=1.0)
    _write_wav(root / audio_subdir / "LJ001-0002.wav", seconds=0.8)
    meta = (
        "LJ001-0001|Printing, in the only sense with which we are concerned.|printing in the only sense with which we are concerned.\n"
        "LJ001-0002|Hello there.|Hello there.\n"
        "LJ001-9999|Missing audio file.|missing audio file.\n"
        "LJ001-0003||\n"
    )
    (root / "metadata.csv").write_text(meta, encoding="utf-8")
    return root


def test_parse_row_prefers_normalized():
    fid, text = parse_ljspeech_row(["LJ001-0001", "Raw HERE.", "normalized here."])
    assert fid == "LJ001-0001"
    assert text == "normalized here."
    # Two-column fallback.
    assert parse_ljspeech_row(["LJ001-0002", "Just raw."])[1] == "Just raw."
    # Raw opt-out.
    assert parse_ljspeech_row(["LJ001-0001", "Raw.", "Norm."], use_normalized=False)[1] == "Raw."
    with pytest.raises(ValueError):
        parse_ljspeech_row(["only-one-col"])
    with pytest.raises(ValueError):
        parse_ljspeech_row(["LJ001-0003", "", ""])


def test_loads_audio_layout_and_skips_bad_rows(tmp_path):
    root = _make_root(tmp_path, "audio")
    ds = LJSpeechDataset(root=root)
    assert len(ds) == 2  # missing wav + empty transcript skipped
    assert ds.items[0]["text"].startswith("printing in the only")
    item = ds[0]
    assert item["tokens"].dtype == torch.long and len(item["tokens"]) > 0
    assert item["audio"].dtype == torch.float32
    assert len(item["audio"]) % 240 == 0
    assert abs(len(item["audio"]) - 24000) <= 240  # 1s @22050 -> 24k
    assert torch.isfinite(item["audio"]).all()
    assert item["file"] == "LJ001-0001"
    assert item["speaker"] == "LJSpeech"


def test_classic_wavs_subdir_and_explicit_audio_dir(tmp_path):
    root = _make_root(tmp_path, "wavs")
    assert len(LJSpeechDataset(root=root)) == 2
    # Explicit metadata path + explicit audio dir (the `audio/` issue layout).
    root2 = _make_root(tmp_path / "other", "audio")
    ds = LJSpeechDataset(metadata_path=root2 / "metadata.csv",
                         audio_dir=root2 / "audio")
    assert len(ds) == 2
    # Root pointing straight at the csv file.
    assert len(LJSpeechDataset(root=root2 / "metadata.csv")) == 2


def test_duration_filter_and_lengths(tmp_path):
    root = _make_root(tmp_path, "audio")
    _write_wav(root / "audio" / "LJ001-long.wav", seconds=20.0)
    with open(root / "metadata.csv", "a", encoding="utf-8") as f:
        f.write("LJ001-long|A very long clip.|a very long clip.\n")
    ds = LJSpeechDataset(root=root, max_duration_s=12.0)
    assert len(ds) == 2
    assert len(ds.audio_lengths) == 2 == len(ds.get_audio_lengths())
    assert all(v % 240 == 0 for v in ds.audio_lengths)
    batch = collate_hifi_tts([ds[0], ds[1]])
    assert batch["audio"].dim() == 2 and batch["audio"].shape[1] % 240 == 0
    assert batch["phoneme_tokens"].dim() == 2


def test_factory_routing(tmp_path):
    root = _make_root(tmp_path, "audio")
    ds = create_hifi_tts_dataset("ljspeech", root=root)
    assert isinstance(ds, LJSpeechDataset)
    assert len(ds) == 2
    assert create_hifi_tts_dataset("lj", root=root).audio_lengths
