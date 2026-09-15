"""
Lazy HuggingFace Hi-Fi TTS dataset adapters (MikhailT/hifi-tts).

Covers the switch-out compatibility for the new dataset format:

- Full training set: ``MikhailT/hifi-tts`` (~324k rows, ~40 GB audio).
- Lightweight same-format set for testing: ``MikhailT/hifi-tts-light`` (84 rows, ~12 MB).

CRITICAL memory constraint: the full 40 GB corpus must NEVER be loaded
into RAM. Both adapters below are lazy:

- :class:`HuggingFaceHiFiTTSDataset` (map-style) keeps the Arrow table
  memory-mapped on disk and decodes/resamples **one row at a time** in
  ``__getitem__``. Speaker/duration filtering at init time only touches
  lightweight metadata columns (never the ``audio`` bytes).
- :class:`StreamingHiFiTTSDataset` (iterable-style) uses
  ``load_dataset(..., streaming=True)`` so not even the full parquet
  cache needs to sit on disk; RAM stays O(1) per worker.

Row schema (both repos, all ``clean``/``other``/``all`` configs):

- ``speaker``: string id, e.g. ``"9017"`` (John Van Stan), ``"6097"``, ``"92"``.
- ``file``: original relative path, e.g. ``audio/9017_clean/... .flac``.
- ``duration``: float seconds (at the native 44.1 kHz rate).
- ``text`` / ``text_no_preprocessing`` / ``text_normalized``: transcripts.
- ``audio``: HF ``Audio`` feature ``{"array", "sampling_rate", "path"}``.
  Native sampling rate is **44100 Hz**; both adapters resample to 24 kHz mono.

Subset/split layout (from the dataset configs)::

    subset="clean" -> splits: train / test / dev
    subset="other" -> splits: train / test / dev
    subset="all"   -> splits: train.clean / train.other / test.clean /
                               test.other / dev.clean / dev.other

Output item format matches :class:`HiFiTTSDataset` so the existing
``collate_hifi_tts`` works unchanged::

    {"tokens": LongTensor, "audio": FloatTensor @ 24kHz,
     "text": str, "speaker": str, "file": str}

Storage note (Kaggle): ``/kaggle/working`` is only ~20 GB while the full
corpus cache is ~40 GB. Both adapters therefore resolve a ``None``
``cache_dir`` to ``/kaggle/tmp/hf_cache`` whenever ``/kaggle/tmp`` exists
(see :func:`default_hf_cache_dir`), keeping checkpoints (small, must
persist) in ``/kaggle/working`` and the bulky ephemeral parquet cache on
``/kaggle/tmp`` scratch. Pass an explicit ``cache_dir`` (or set
``KAGGLE_TMP_DIR``) to override.
"""

from __future__ import annotations

import itertools
import os
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset

from aeroflow.frontend.phonemizer import Phonemizer

HF_REPO_FULL = "MikhailT/hifi-tts"
HF_REPO_LIGHT = "MikhailT/hifi-tts-light"

HF_NATIVE_SAMPLE_RATE = 44100

#: Kaggle scratch mount used for the bulky ephemeral HF cache
#: (``/kaggle/working`` at ~20 GB cannot hold the ~40 GB corpus cache).
KAGGLE_TMP_DIR = "/kaggle/tmp"

#: Subdirectory of the Kaggle scratch mount holding the HF datasets cache.
HF_CACHE_SUBDIR = "hf_cache"

__all__ = [
    "HF_REPO_FULL",
    "HF_REPO_LIGHT",
    "HF_NATIVE_SAMPLE_RATE",
    "KAGGLE_TMP_DIR",
    "HF_CACHE_SUBDIR",
    "default_hf_cache_dir",
    "HuggingFaceHiFiTTSDataset",
    "StreamingHiFiTTSDataset",
    "create_hifi_tts_dataset",
    "process_hf_row",
]


# ---------------------------------------------------------------------------
# Shared pure-python helpers (no `datasets` dependency -> unit testable)
# ---------------------------------------------------------------------------

def default_hf_cache_dir() -> Optional[str]:
    """
    Returns the preferred HF datasets cache directory, or None for default.

    On Kaggle (``/kaggle/tmp`` present, override via ``KAGGLE_TMP_DIR`` env)
    this is ``/kaggle/tmp/hf_cache`` so the ~40 GB corpus cache lands on
    scratch instead of overflowing the ~20 GB ``/kaggle/working`` mount.
    Off Kaggle, returns None (use the standard HF cache location).
    """
    kaggle_tmp = os.environ.get("KAGGLE_TMP_DIR", KAGGLE_TMP_DIR)
    if kaggle_tmp and os.path.isdir(kaggle_tmp):
        return os.path.join(kaggle_tmp, HF_CACHE_SUBDIR)
    return None


def _resolve_cache_dir(cache_dir: Optional[str] = None) -> Optional[str]:
    """Resolves an explicit ``cache_dir`` or the Kaggle-aware default (created)."""
    resolved = cache_dir or default_hf_cache_dir()
    if resolved:
        os.makedirs(resolved, exist_ok=True)
    return resolved
# Shared pure-python helpers (no `datasets` dependency -> unit testable)
# ---------------------------------------------------------------------------

def _normalize_speaker_ids(
    speaker_ids: Optional[Union[str, int, Sequence[Union[str, int]]]],
) -> Optional[Tuple[str, ...]]:
    """Normalizes speaker filter to a tuple of strings, or None for all."""
    if speaker_ids is None:
        return None
    if isinstance(speaker_ids, (str, int)):
        speaker_ids = (speaker_ids,)
    normalized = tuple(sorted({str(s) for s in speaker_ids}))
    return normalized or None


def _pick_text(row: Dict[str, Any], text_field: str = "text_normalized") -> str:
    """Picks transcript with fallback: requested -> text_normalized -> text -> raw."""
    for key in (text_field, "text_normalized", "text", "text_no_preprocessing"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _resample_mono(
    waveform: np.ndarray,
    src_sr: int,
    tgt_sr: int = 24000,
) -> np.ndarray:
    """Resamples mono float32 audio with linear interpolation (matches dataset.py)."""
    data = np.asarray(waveform, dtype=np.float32)
    if data.ndim > 1:
        data = data.mean(axis=-1).astype(np.float32)
    if src_sr != tgt_sr and len(data) > 0:
        num_target = int(len(data) * tgt_sr / src_sr)
        if num_target <= 0:
            return np.zeros((0,), dtype=np.float32)
        data = np.interp(
            np.linspace(0, len(data), num_target, endpoint=False),
            np.arange(len(data)),
            data,
        ).astype(np.float32)
    return data


def _audio_array_and_sr(
    audio_payload: Any,
    row: Dict[str, Any],
) -> Tuple[np.ndarray, int]:
    """
    Extracts (waveform, sampling_rate) from the HF ``audio`` feature.

    Handles the plain ``dict`` form (cached/local datasets) as well as the
    ``AudioDecoder`` object returned by ``datasets`` >= 3 streaming and
    torchcodec-backed map datasets (both support ``payload["array"]`` /
    ``payload["sampling_rate"]`` but only ``dict`` has ``.get``).
    """
    if audio_payload is None:
        raise ValueError(f"HF row missing audio array (file={row.get('file')!r})")
    if isinstance(audio_payload, np.ndarray):
        # Raw waveform passed directly (unit tests / pre-decoded paths).
        src_sr = row.get("sampling_rate") or HF_NATIVE_SAMPLE_RATE
        return np.asarray(audio_payload, dtype=np.float32), int(src_sr)
    if isinstance(audio_payload, dict):
        raw_array = audio_payload.get("array")
        src_sr = audio_payload.get("sampling_rate") or HF_NATIVE_SAMPLE_RATE
        if raw_array is None:
            raise ValueError(f"HF row missing audio array (file={row.get('file')!r})")
        return np.asarray(raw_array, dtype=np.float32), int(src_sr)
    # AudioDecoder / mapping-like payloads.
    try:
        raw_array = audio_payload["array"]
        src_sr = audio_payload["sampling_rate"]
        if raw_array is None:
            raise ValueError(f"HF row missing audio array (file={row.get('file')!r})")
        return np.asarray(raw_array, dtype=np.float32), int(src_sr)
    except (KeyError, TypeError, IndexError, ValueError) as exc:
        if isinstance(exc, ValueError):
            raise
    # Last resort: attribute-style payloads.
    raw_array = getattr(audio_payload, "array", None)
    src_sr = getattr(audio_payload, "sampling_rate", HF_NATIVE_SAMPLE_RATE)
    if raw_array is None:
        raise ValueError(f"HF row missing audio array (file={row.get('file')!r})")
    return np.asarray(raw_array, dtype=np.float32), int(src_sr or HF_NATIVE_SAMPLE_RATE)


def _peak_normalize(waveform: np.ndarray, peak: float = 0.95) -> np.ndarray:
    data = np.asarray(waveform, dtype=np.float32)
    max_abs = float(np.abs(data).max()) if data.size else 0.0
    if max_abs > 1e-4:
        data = (peak * (data / max_abs)).astype(np.float32)
    return data


def process_hf_row(
    row: Dict[str, Any],
    phonemizer: Phonemizer,
    sample_rate: int = 24000,
    hop_length: int = 240,
    text_field: str = "text_normalized",
) -> Dict[str, Any]:
    """
    Converts one raw HF row into a training item (lazy, single-row, O(1) RAM).

    Expects ``row["audio"]`` as an HF Audio dict ``{"array", "sampling_rate"}``
    or an ``AudioDecoder`` exposing ``payload["array"]`` /
    ``payload["sampling_rate"]`` (datasets >= 3 / torchcodec), plus transcript
    fields. Raises ``ValueError`` for rows with missing audio/text (callers
    decide skip vs. fail).
    """
    raw_array, src_sr = _audio_array_and_sr(row.get("audio"), row)

    text = _pick_text(row, text_field)
    if not text:
        raise ValueError(f"HF row missing transcript (file={row.get('file')!r})")

    wave = _resample_mono(np.asarray(raw_array, dtype=np.float32), src_sr, sample_rate)
    wave = _peak_normalize(wave)

    num_frames = len(wave) // hop_length
    wave = wave[: num_frames * hop_length]

    tokens = phonemizer.text_to_sequence(text)
    if len(tokens) == 0:
        raise ValueError(f"HF row produced zero phoneme tokens (file={row.get('file')!r})")

    speaker = str(row.get("speaker", ""))
    return {
        "tokens": torch.tensor(tokens, dtype=torch.long),
        "audio": torch.tensor(wave, dtype=torch.float32),
        "text": text,
        "speaker": speaker,
        "file": str(row.get("file", "")),
    }


def _row_passes_filter(
    speaker: Any,
    duration: Any,
    text: str,
    speaker_ids: Optional[Tuple[str, ...]],
    min_duration_s: float,
    max_duration_s: float,
) -> bool:
    if speaker_ids is not None and str(speaker) not in speaker_ids:
        return False
    try:
        dur = float(duration)
    except (TypeError, ValueError):
        return False
    if not (min_duration_s <= dur <= max_duration_s):
        return False
    if not text:
        return False
    return True


# ---------------------------------------------------------------------------
# Map-style lazy dataset (random access, memory-mapped Arrow)
# ---------------------------------------------------------------------------

class HuggingFaceHiFiTTSDataset(Dataset):
    """
    Map-style lazy adapter over ``MikhailT/hifi-tts`` (or ``-light``).

    Laziness contract: the underlying HF Arrow table stays memory-mapped;
    audio bytes are decoded/resampled for exactly one row per ``__getitem__``
    call. Init only scans metadata columns (speaker/duration/text), so even
    the 324k-row / 40 GB corpus never materializes in RAM.

    Args:
        repo_id: HF dataset repo (full or light).
        subset: ``"clean"`` (default), ``"other"``, or ``"all"``.
        split: split name for the subset, e.g. ``"train"`` for
            ``clean``/``other``, or ``"train.clean"`` for ``all``.
        speaker_ids: speaker id(s) to keep, e.g. ``("9017",)`` (default:
            John Van Stan single-speaker voice). ``None`` keeps all speakers.
        sample_rate: target rate (model needs 24000).
        hop_length: audio truncated to a multiple of this (model needs 240).
        min_duration_s / max_duration_s: metadata pre-filter bounds.
        text_field: preferred transcript column (fallback chain built in).
        cache_dir / token: forwarded to ``load_dataset``. A None cache_dir
            auto-resolves to ``/kaggle/tmp/hf_cache`` on Kaggle (see
            :func:`default_hf_cache_dir`) so the ~40 GB cache does not
            overflow the ~20 GB ``/kaggle/working`` mount.
    """

    def __init__(
        self,
        repo_id: str = HF_REPO_FULL,
        subset: str = "clean",
        split: str = "train",
        speaker_ids: Optional[Union[str, int, Sequence[Union[str, int]]]] = ("9017",),
        sample_rate: int = 24000,
        hop_length: int = 240,
        min_duration_s: float = 0.5,
        max_duration_s: float = 12.0,
        text_field: str = "text_normalized",
        cache_dir: Optional[str] = None,
        token: Optional[Union[str, bool]] = None,
    ):
        super().__init__()
        try:
            from datasets import load_dataset  # lazy: keeps `import aeroflow` light
        except ImportError as exc:
            raise ImportError(
                "HuggingFaceHiFiTTSDataset needs the `datasets` package "
                "(e.g. `pip install 'datasets[audio]'`)."
            ) from exc

        self.repo_id = repo_id
        self.subset = subset
        self.split = split
        self.speaker_ids = _normalize_speaker_ids(speaker_ids)
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.min_duration_s = min_duration_s
        self.max_duration_s = max_duration_s
        self.text_field = text_field
        self.phonemizer = Phonemizer()
        self.cache_dir = _resolve_cache_dir(cache_dir)

        load_kwargs: Dict[str, Any] = {"trust_remote_code": False}
        if self.cache_dir is not None:
            load_kwargs["cache_dir"] = self.cache_dir
        if token is not None:
            load_kwargs["token"] = token

        hf_ds = load_dataset(repo_id, subset, split=split, streaming=False, **load_kwargs)
        self._hf = hf_ds
        self._index: List[int] = self._build_metadata_index(hf_ds)

    def _build_metadata_index(self, hf_ds: Any) -> List[int]:
        """
        Scans only metadata columns in batches (never touches `audio` bytes).
        Batched slicing keeps init fast even for 324k rows.
        """
        speaker_ids = self.speaker_ids
        col_names: List[str] = list(getattr(hf_ds, "column_names", []) or [])
        meta = hf_ds
        if "audio" in col_names:
            try:
                meta = hf_ds.remove_columns(["audio"])
            except Exception:
                meta = hf_ds

        total = len(meta)
        index: List[int] = []
        batch_size = 5000
        for start in range(0, total, batch_size):
            chunk = meta[start: start + batch_size]
            speakers = chunk.get("speaker", [None] * min(batch_size, total - start))
            durations = chunk.get("duration", [None] * len(speakers))
            n = len(speakers)
            for key in (self.text_field, "text_normalized", "text", "text_no_preprocessing"):
                if key in chunk:
                    texts = chunk.get(key)
                    if texts is not None and any(
                        isinstance(t, str) and t.strip() for t in texts
                    ):
                        break
            else:
                texts = [""] * n
            for j in range(n):
                text_val = texts[j] if j < len(texts) else ""
                text_val = text_val.strip() if isinstance(text_val, str) else ""
                dur_val = durations[j] if j < len(durations) else None
                if _row_passes_filter(
                    speakers[j], dur_val, text_val,
                    speaker_ids, self.min_duration_s, self.max_duration_s,
                ):
                    index.append(start + j)
        return index

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        hf_idx = self._index[idx]
        row = self._hf[hf_idx]
        return process_hf_row(
            row,
            self.phonemizer,
            sample_rate=self.sample_rate,
            hop_length=self.hop_length,
            text_field=self.text_field,
        )

    @property
    def hf_index(self) -> List[int]:
        """Underlying HF row ids backing each positional index (for debugging)."""
        return list(self._index)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(repo_id={self.repo_id!r}, subset={self.subset!r}, "
            f"split={self.split!r}, speaker_ids={self.speaker_ids!r}, "
            f"num_rows={len(self)})"
        )


# ---------------------------------------------------------------------------
# Streaming iterable dataset (no local copy, O(1) RAM)
# ---------------------------------------------------------------------------

class StreamingHiFiTTSDataset(IterableDataset):
    """
    Iterable-style lazy adapter with ``streaming=True``.

    Nothing is downloaded/cached up front; rows flow from the Hub parquet
    shards and only the current row occupies RAM. Ideal for the full 40 GB
    corpus on disk-constrained trainers (e.g. Kaggle).

    Supports ``torch.utils.data.DataLoader(num_workers=N)`` sharding via
    ``itertools.islice`` and multi-node DDP via ``rank``/``world_size``
    (uses HF ``IterableDataset.shard`` when available).

    Call :meth:`set_epoch` once per epoch (like a sampler) so the optional
    shuffle buffer reshuffles between passes over the stream.
    """

    def __init__(
        self,
        repo_id: str = HF_REPO_FULL,
        subset: str = "clean",
        split: str = "train",
        speaker_ids: Optional[Union[str, int, Sequence[Union[str, int]]]] = ("9017",),
        sample_rate: int = 24000,
        hop_length: int = 240,
        min_duration_s: float = 0.5,
        max_duration_s: float = 12.0,
        text_field: str = "text_normalized",
        rank: int = 0,
        world_size: int = 1,
        seed: int = 42,
        shuffle_buffer_size: int = 0,
        cache_dir: Optional[str] = None,
        token: Optional[Union[str, bool]] = None,
    ):
        super().__init__()
        self.repo_id = repo_id
        self.subset = subset
        self.split = split
        self.speaker_ids = _normalize_speaker_ids(speaker_ids)
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.min_duration_s = min_duration_s
        self.max_duration_s = max_duration_s
        self.text_field = text_field
        self.rank = rank
        self.world_size = max(1, world_size)
        self.seed = seed
        self.shuffle_buffer_size = shuffle_buffer_size
        self.cache_dir = _resolve_cache_dir(cache_dir)
        self.token = token
        self.phonemizer = Phonemizer()
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Sets the epoch index (reshuffles the stream when shuffling is on)."""
        self._epoch = int(epoch)

    def _open_stream(self) -> Iterable[Dict[str, Any]]:
        from datasets import load_dataset  # lazy import

        load_kwargs: Dict[str, Any] = {"trust_remote_code": False}
        if self.cache_dir is not None:
            load_kwargs["cache_dir"] = self.cache_dir
        if self.token is not None:
            load_kwargs["token"] = self.token
        ds = load_dataset(
            self.repo_id, self.subset, split=self.split, streaming=True, **load_kwargs
        )
        if self.world_size > 1 and hasattr(ds, "shard"):
            try:
                ds = ds.shard(num_shards=self.world_size, index=self.rank)
            except Exception:
                pass
        if self.shuffle_buffer_size and self.shuffle_buffer_size > 0 and hasattr(ds, "shuffle"):
            try:
                ds = ds.shuffle(
                    buffer_size=self.shuffle_buffer_size,
                    seed=self.seed + self._epoch,
                )
            except Exception:
                pass
        return ds

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        stream = self._open_stream()
        if worker_info is not None and worker_info.num_workers > 1:
            stream = itertools.islice(
                stream, worker_info.id, None, worker_info.num_workers
            )
        for row in stream:
            text = _pick_text(row, self.text_field)
            if not _row_passes_filter(
                row.get("speaker"), row.get("duration"), text,
                self.speaker_ids, self.min_duration_s, self.max_duration_s,
            ):
                continue
            try:
                yield process_hf_row(
                    row,
                    self.phonemizer,
                    sample_rate=self.sample_rate,
                    hop_length=self.hop_length,
                    text_field=self.text_field,
                )
            except ValueError:
                continue

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(repo_id={self.repo_id!r}, subset={self.subset!r}, "
            f"split={self.split!r}, speaker_ids={self.speaker_ids!r})"
        )


# ---------------------------------------------------------------------------
# Switch-out factory
# ---------------------------------------------------------------------------

def create_hifi_tts_dataset(
    source: str = "hf",
    **kwargs: Any,
):
    """
    One-line switch between dataset backends (all share the collate format).

    Args:
        source: ``"manifest"`` -> legacy local :class:`HiFiTTSDataset`
            (pass ``manifest_path``/``audio_dir``); ``"hf"`` (default) ->
            lazy map-style :class:`HuggingFaceHiFiTTSDataset` over
            ``MikhailT/hifi-tts``; ``"hf-streaming"``/``"streaming"`` ->
            :class:`StreamingHiFiTTSDataset` (no local copy, O(1) RAM).
        **kwargs: forwarded to the selected class.

    Examples:
        >>> ds = create_hifi_tts_dataset("hf")  # full 9017 clean/train
        >>> light = create_hifi_tts_dataset("hf", repo_id=HF_REPO_LIGHT)
        >>> stream = create_hifi_tts_dataset("hf-streaming")
        >>> local = create_hifi_tts_dataset("manifest", manifest_path="m.json")
    """
    normalized = source.lower().replace("_", "-")
    if normalized in ("manifest", "local", "disk"):
        from aeroflow.dataset.dataset import HiFiTTSDataset

        return HiFiTTSDataset(
            manifest_path=kwargs.pop("manifest_path", None),
            audio_dir=kwargs.pop("audio_dir", None),
            sample_rate=kwargs.pop("sample_rate", 24000),
            hop_length=kwargs.pop("hop_length", 240),
            max_duration_s=kwargs.pop("max_duration_s", 12.0),
            min_duration_s=kwargs.pop("min_duration_s", 0.5),
        )
    if normalized in ("hf", "hf-map", "map"):
        return HuggingFaceHiFiTTSDataset(**kwargs)
    if normalized in ("hf-streaming", "streaming", "iterable"):
        return StreamingHiFiTTSDataset(**kwargs)
    raise ValueError(
        f"Unknown dataset source {source!r}. "
        "Expected 'manifest', 'hf', or 'hf-streaming'."
    )
