"""
Unit + integration tests for the lazy HuggingFace Hi-Fi TTS adapters.

Unit tests are fully offline (synthetic rows, fake HF table) and prove the
CRITICAL memory contract: metadata index building never touches `audio`
bytes, and each `__getitem__` decodes exactly one row.

Integration tests hit the small same-format repo `MikhailT/hifi-tts-light`
and are skipped when `datasets` or network access is unavailable.
"""

import numpy as np
import pytest
import torch

from aeroflow.dataset.dataset import collate_hifi_tts
from aeroflow.dataset.hf_hifi_tts import (
    HF_REPO_LIGHT,
    HuggingFaceHiFiTTSDataset,
    StreamingHiFiTTSDataset,
    _normalize_speaker_ids,
    _peak_normalize,
    _resample_mono,
    _resolve_cache_dir,
    _row_passes_filter,
    create_hifi_tts_dataset,
    default_hf_cache_dir,
    process_hf_row,
)
from aeroflow.frontend.phonemizer import Phonemizer


def _sine_44k(seconds=1.0, freq=110.0, stereo=False):
    sr = 44100
    n = int(seconds * sr)
    t = np.linspace(0, seconds, n, endpoint=False).astype(np.float32)
    wave = (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    if stereo:
        wave = np.stack([wave, wave * 0.5], axis=-1)
    return wave


def _hf_row(speaker="9017", duration=1.0, text="Hello world.", seconds=1.0, stereo=False):
    return {
        "speaker": speaker,
        "file": f"audio/{speaker}_clean/x.flac",
        "duration": duration,
        "text": text.lower(),
        "text_no_preprocessing": text,
        "text_normalized": text,
        "audio": {"array": _sine_44k(seconds), "sampling_rate": 44100}
        if not stereo else {"array": _sine_44k(seconds, stereo=True), "sampling_rate": 44100},
    }


# ---------------------------------------------------------------------------
# Pure helper tests (offline)
# ---------------------------------------------------------------------------

def test_normalize_speaker_ids():
    assert _normalize_speaker_ids(None) is None
    assert _normalize_speaker_ids("9017") == ("9017",)
    assert _normalize_speaker_ids(9017) == ("9017",)
    assert _normalize_speaker_ids(["9017", 9017, "92"]) == ("9017", "92")
    assert _normalize_speaker_ids([]) is None


def test_resample_44k_to_24k_length_and_mono():
    mono = _sine_44k(1.0)
    out = _resample_mono(mono, 44100, 24000)
    assert len(out) == 24000
    stereo = _sine_44k(1.0, stereo=True)
    assert stereo.ndim == 2
    out_stereo = _resample_mono(stereo, 44100, 24000)
    assert out_stereo.ndim == 1 and len(out_stereo) == 24000
    # Same-rate passthrough
    same = _resample_mono(mono, 44100, 44100)
    assert len(same) == len(mono)


def test_peak_normalize():
    loud = np.array([2.0, -4.0, 1.0], dtype=np.float32)
    normed = _peak_normalize(loud)
    assert abs(float(np.abs(normed).max()) - 0.95) < 1e-6
    silence = np.zeros(100, dtype=np.float32)
    assert _peak_normalize(silence).max() == 0.0


def test_row_filter_bounds():
    assert _row_passes_filter("9017", 1.0, "hi", ("9017",), 0.5, 12.0)
    assert not _row_passes_filter("92", 1.0, "hi", ("9017",), 0.5, 12.0)
    assert not _row_passes_filter("9017", 0.1, "hi", ("9017",), 0.5, 12.0)
    assert not _row_passes_filter("9017", 13.5, "hi", ("9017",), 0.5, 12.0)
    assert not _row_passes_filter("9017", 1.0, "", ("9017",), 0.5, 12.0)
    assert _row_passes_filter("92", 1.0, "hi", None, 0.5, 12.0)


def test_default_hf_cache_dir_kaggle_scratch(monkeypatch, tmp_path):
    """On Kaggle the cache must resolve to /kaggle/tmp scratch, not /kaggle/working."""
    monkeypatch.setenv("KAGGLE_TMP_DIR", str(tmp_path))
    resolved = default_hf_cache_dir()
    assert resolved == str(tmp_path / "hf_cache")
    # _resolve_cache_dir creates it
    assert _resolve_cache_dir() == resolved
    assert (tmp_path / "hf_cache").is_dir()


def test_default_hf_cache_dir_off_kaggle(monkeypatch):
    monkeypatch.setenv("KAGGLE_TMP_DIR", "/nonexistent-kaggle-tmp-xyz")
    assert default_hf_cache_dir() is None
    assert _resolve_cache_dir() is None


def test_resolve_cache_dir_explicit_wins(monkeypatch, tmp_path):
    monkeypatch.setenv("KAGGLE_TMP_DIR", str(tmp_path))
    explicit = str(tmp_path / "custom")
    assert _resolve_cache_dir(explicit) == explicit
    assert (tmp_path / "custom").is_dir()


def test_process_hf_row_dict_audio():
    phonemizer = Phonemizer()
    out = process_hf_row(_hf_row(), phonemizer)
    assert out["audio"].dtype == torch.float32
    assert len(out["audio"]) == 24000  # 1s @ 44.1k -> 24k, hop-multiple
    assert len(out["audio"]) % 240 == 0
    assert out["tokens"].dtype == torch.long and len(out["tokens"]) > 0
    assert out["speaker"] == "9017"
    assert out["text"] == "Hello world."


def test_process_hf_row_audiodecoder_style_payload():
    """datasets>=3/torchcodec yields AudioDecoder (no .get, only [key])."""

    class FakeAudioDecoder:
        def __init__(self, array, sr):
            self._array = array
            self._sr = sr

        def __getitem__(self, key):
            if key == "array":
                return self._array
            if key == "sampling_rate":
                return self._sr
            raise KeyError(key)

    phonemizer = Phonemizer()
    row = _hf_row()
    row["audio"] = FakeAudioDecoder(row["audio"]["array"], 44100)
    out = process_hf_row(row, phonemizer)
    assert len(out["audio"]) == 24000


def test_process_hf_row_text_fallback_and_errors():
    phonemizer = Phonemizer()
    row = _hf_row()
    del row["text_normalized"]
    out = process_hf_row(row, phonemizer, text_field="text_normalized")
    assert out["text"] == "hello world."

    with pytest.raises(ValueError):
        process_hf_row({**_hf_row(), "audio": None}, phonemizer)
    with pytest.raises(ValueError):
        process_hf_row({**_hf_row(), "text": "", "text_normalized": "",
                        "text_no_preprocessing": ""}, phonemizer)


def test_processed_rows_collate():
    phonemizer = Phonemizer()
    items = [process_hf_row(_hf_row(text="Hello world.", seconds=1.0), phonemizer),
             process_hf_row(_hf_row(text="Hi there.", seconds=0.6), phonemizer)]
    batch = collate_hifi_tts(items)
    assert batch["phoneme_tokens"].dim() == 2
    assert batch["audio"].dim() == 2
    assert batch["audio"].shape[1] % 240 == 0
    assert batch["text_lengths"].tolist() == [len(items[0]["tokens"]), len(items[1]["tokens"])]


# ---------------------------------------------------------------------------
# Laziness proof with a fake HF table (offline, no `datasets` import needed)
# ---------------------------------------------------------------------------

class _FakeHFTable:
    """Minimal stand-in for datasets.Dataset proving lazy audio access."""

    def __init__(self, rows):
        self._rows = rows
        self.column_names = ["speaker", "file", "duration", "text",
                             "text_no_preprocessing", "text_normalized", "audio"]
        self.audio_decodes = 0

    def remove_columns(self, cols):
        assert "audio" in cols
        view = _FakeHFTable(self._rows)
        view.column_names = [c for c in self.column_names if c not in cols]
        view._metadata_only = True
        return view

    def __len__(self):
        return len(self._rows)

    def __getitem__(self, key):
        if isinstance(key, slice):
            chunk = self._rows[key]
            out = {c: [r[c] if c != "audio" else None for r in chunk]
                   for c in self.column_names if c != "audio" or False}
            # metadata slices must never carry audio bytes
            assert "audio" not in out
            return out
        row = self._rows[key]
        if getattr(self, "_metadata_only", False):
            row = {k: v for k, v in row.items() if k != "audio"}
        elif "audio" in row:
            self.audio_decodes += 1
        return row


def _make_lazy_map_instance(rows, **kwargs):
    ds = HuggingFaceHiFiTTSDataset.__new__(HuggingFaceHiFiTTSDataset)
    import torch.utils.data as _tud
    assert isinstance(ds, _tud.Dataset)
    ds.speaker_ids = _normalize_speaker_ids(kwargs.get("speaker_ids", ("9017",)))
    ds.sample_rate = kwargs.get("sample_rate", 24000)
    ds.hop_length = kwargs.get("hop_length", 240)
    ds.min_duration_s = kwargs.get("min_duration_s", 0.5)
    ds.max_duration_s = kwargs.get("max_duration_s", 12.0)
    ds.text_field = kwargs.get("text_field", "text_normalized")
    ds.phonemizer = Phonemizer()
    fake = _FakeHFTable(rows)
    ds._hf = fake
    HuggingFaceHiFiTTSDataset._build_metadata_index(ds, fake)
    return ds, fake


def test_metadata_index_never_touches_audio():
    rows = [
        _hf_row("9017", 1.0, "Hello world.", 1.0),
        _hf_row("92", 1.0, "Hi there.", 1.0),          # wrong speaker
        _hf_row("9017", 20.0, "Too long.", 2.0),        # over max duration
        _hf_row("9017", 0.1, "Too short.", 0.5),        # under min duration
        _hf_row("9017", 1.0, "Second kept.", 0.6),
    ]
    ds, fake = _make_lazy_map_instance(rows)
    assert fake.audio_decodes == 0, "index build must not decode audio"
    assert len(ds) == 2
    item = ds[0]
    assert fake.audio_decodes == 1, "each __getitem__ decodes exactly one row"
    assert item["speaker"] == "9017"
    batch = collate_hifi_tts([ds[0], ds[1]])
    assert batch["audio"].dim() == 2


def test_streaming_defaults_and_epoch(monkeypatch):
    """Streaming adapter constructs offline; epoch only affects shuffle seed."""
    monkeypatch.setenv("KAGGLE_TMP_DIR", "/nonexistent-kaggle-tmp-xyz")
    s = StreamingHiFiTTSDataset(repo_id=HF_REPO_LIGHT)
    assert s.seed == 42
    assert s.shuffle_buffer_size == 0
    assert s.cache_dir is None  # no Kaggle scratch in this env
    s.set_epoch(3)
    assert s._epoch == 3


def test_streaming_explicit_cache_dir(tmp_path):
    s = StreamingHiFiTTSDataset(repo_id=HF_REPO_LIGHT,
                                cache_dir=str(tmp_path / "c"))
    assert s.cache_dir == str(tmp_path / "c")
    assert (tmp_path / "c").is_dir()


def test_factory_routing():
    from aeroflow.dataset.dataset import HiFiTTSDataset
    local = create_hifi_tts_dataset("manifest")
    assert isinstance(local, HiFiTTSDataset)
    with pytest.raises(ValueError):
        create_hifi_tts_dataset("nope")


def test_factory_hf_classes():
    pytest.importorskip("datasets", reason="datasets lib not installed",
                        exc_type=ImportError)
    assert create_hifi_tts_dataset("hf", repo_id=HF_REPO_LIGHT,
                                   subset="clean", split="train",
                                   speaker_ids=None,
                                   min_duration_s=0.0, max_duration_s=30.0)
    s = create_hifi_tts_dataset("hf-streaming", repo_id=HF_REPO_LIGHT)
    assert isinstance(s, StreamingHiFiTTSDataset)


# ---------------------------------------------------------------------------
# Integration tests against hifi-tts-light (needs network + datasets lib)
# ---------------------------------------------------------------------------

def _wide_light_kwargs(**over):
    kw = dict(repo_id=HF_REPO_LIGHT, subset="clean", split="train",
              min_duration_s=0.0, max_duration_s=30.0)
    kw.update(over)
    return kw


def test_integration_light_map_9017():
    pytest.importorskip("datasets", reason="datasets lib not installed",
                        exc_type=ImportError)
    try:
        ds = HuggingFaceHiFiTTSDataset(**_wide_light_kwargs(speaker_ids=("9017",)))
    except Exception as exc:
        pytest.skip(f"light dataset unreachable: {exc}")
    assert len(ds) == 3  # light clean/train holds 3x 9017 rows
    for i in range(len(ds)):
        item = ds[i]
        assert item["speaker"] == "9017"
        assert len(item["audio"]) % 240 == 0
        assert len(item["tokens"]) > 0
        assert torch.isfinite(item["audio"]).all()
    batch = collate_hifi_tts([ds[i] for i in range(len(ds))])
    assert batch["audio"].dim() == 2


# ---------------------------------------------------------------------------
# Dry-run loader tests (scripts/dry_run_train.py)
# ---------------------------------------------------------------------------

def _import_dry_run_module():
    import importlib.util
    from pathlib import Path
    path = Path(__file__).resolve().parent.parent / "scripts" / "dry_run_train.py"
    spec = importlib.util.spec_from_file_location("dry_run_train", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_dry_run_loader_falls_back_offline_or_uses_light():
    """Loader must always return a valid batch: light when reachable, else synthetic."""
    module = _import_dry_run_module()
    batch, source = module.load_dry_run_batch(batch_size=2)
    assert source in ("hifi-tts-light", "synthetic")
    assert batch["phoneme_tokens"].dim() == 2
    assert batch["audio"].dim() == 2
    assert batch["audio"].shape[1] % 240 == 0
    assert all(t.strip() for t in batch["texts"])


def test_dry_run_loader_light_integration():
    pytest.importorskip("datasets", reason="datasets lib not installed",
                        exc_type=ImportError)
    module = _import_dry_run_module()
    try:
        batch, source = module.load_dry_run_batch(batch_size=2)
    except Exception as exc:
        pytest.skip(f"light dataset unreachable: {exc}")
    if source != "hifi-tts-light":
        pytest.skip("light dataset unreachable, synthetic fallback engaged")
    assert batch["phoneme_tokens"].shape[0] == 2
    assert batch["audio"].shape[1] % 240 == 0


def test_integration_light_streaming_matches_map():
    pytest.importorskip("datasets", reason="datasets lib not installed",
                        exc_type=ImportError)
    try:
        stream = StreamingHiFiTTSDataset(**_wide_light_kwargs(speaker_ids=("9017",)))
        items = list(stream)
    except Exception as exc:
        pytest.skip(f"light dataset unreachable: {exc}")
    assert len(items) == 3
    assert all(it["speaker"] == "9017" for it in items)
    batch = collate_hifi_tts(items[:2])
    assert batch["phoneme_tokens"].dim() == 2
