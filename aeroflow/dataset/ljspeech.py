"""
LJSpeech dataset adapter (LJSpeech-1.1 layout + ``audio/`` variant).

Expected on-disk layout::

    <root>/
      metadata.csv          # ``ID|raw transcript|normalized transcript`` per line
      audio/*.wav           # or ``wavs/*.wav`` (classic LJSpeech-1.1), 22050 Hz
      # e.g. LJ001-0001|Printing, in the only sense...|printing in the only...

The stock ``HiFiTTSDataset`` manifest loader cannot read this directly: its
``|`` fallback treats ``parts[0]`` as an audio path (LJSpeech stores a bare
id with no extension/dir there) and picks ``parts[1]`` (raw) instead of the
preferred normalized column. This adapter resolves those two mismatches and
otherwise matches the :class:`HiFiTTSDataset` output format so the existing
``collate_hifi_tts`` and length-bucketed batching work unchanged::

    {"tokens": LongTensor, "audio": FloatTensor @ 24kHz,
     "text": str, "file": str, "speaker": str}

Audio handling mirrors the other adapters: mix to mono, resample to 24 kHz
via :func:`aeroflow.dataset.audio.resample_mono`, peak-normalize to 0.95,
truncate to a multiple of ``hop_length``. Duration filtering at init time is
header-only (``soundfile.info``, no audio decoded).
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

from aeroflow.dataset.audio import resample_mono
from aeroflow.frontend.phonemizer import Phonemizer

__all__ = [
    "LJSPEECH_METADATA_NAME",
    "LJSPEECH_AUDIO_SUBDIRS",
    "LJSpeechDataset",
    "parse_ljspeech_row",
]

#: Default metadata filename inside a dataset root.
LJSPEECH_METADATA_NAME = "metadata.csv"

#: Audio subdirectories probed (in order) under the dataset root when no
#: explicit ``audio_dir`` is given. ``"audio"`` first because that is the
#: layout in the reported issue; ``"wavs"`` is the classic LJSpeech-1.1 name.
LJSPEECH_AUDIO_SUBDIRS: Tuple[str, ...] = ("audio", "wavs")

#: Single-speaker tag carried on each item (mirrors the ``speaker`` field of
#: the HF adapters so downstream logging stays uniform).
LJSPEECH_SPEAKER = "LJSpeech"


def parse_ljspeech_row(
    parts: Sequence[str],
    use_normalized: bool = True,
) -> Tuple[str, str]:
    """
    Splits one ``|``-separated metadata row into ``(file_id, text)``.

    LJSpeech rows are ``ID|raw|normalized``; some exports only have
    ``ID|transcript``. With ``use_normalized=True`` (default) the third
    column wins when non-empty, otherwise the second column is used.
    Raises ``ValueError`` for rows with a missing id or transcript.
    """
    cleaned = [p.strip() for p in parts]
    if len(cleaned) < 2:
        raise ValueError(f"LJSpeech row needs at least 2 columns, got {parts!r}")
    file_id = cleaned[0]
    raw = cleaned[1] if len(cleaned) > 1 else ""
    normalized = cleaned[2] if len(cleaned) > 2 else ""
    if not file_id:
        raise ValueError(f"LJSpeech row missing file id: {parts!r}")
    if use_normalized and normalized:
        text = normalized
    else:
        text = raw or normalized
    if not text:
        raise ValueError(f"LJSpeech row missing transcript (id={file_id!r})")
    return file_id, text


def _candidate_audio_paths(
    file_id: str,
    search_dirs: Sequence[Path],
) -> List[Path]:
    """Ordered candidate files for one metadata id (no I/O performed)."""
    fid = file_id.strip()
    names: List[str] = []
    if os.path.splitext(fid)[1]:
        # Id already carries an extension (e.g. "LJ001-0001.wav").
        names.append(fid)
    else:
        names.extend((f"{fid}.wav", f"{fid}.flac"))
    # Bare filename that already includes a subdir (e.g. "audio/x.wav").
    candidates: List[Path] = []
    for directory in search_dirs:
        for name in names:
            candidates.append(directory / name)
    # Absolute/relative path used verbatim as a last resort.
    verbatim = Path(fid)
    if verbatim.suffix:
        candidates.append(verbatim)
    return candidates


class LJSpeechDataset(Dataset):
    """
    Map-style loader for a local LJSpeech directory (``metadata.csv`` + wav dir).

    Args:
        root: dataset root holding ``metadata.csv`` (and usually ``audio/``
            or ``wavs/``). May also point directly at the metadata file.
        metadata_path: explicit metadata file; defaults to
            ``<root>/metadata.csv``. Accepts the ``metadata.csv`` path itself
            (then ``root`` is inferred as its parent unless given).
        audio_dir: explicit wav directory. Defaults to the first existing
            probe of ``<root>/audio``, ``<root>/wavs``, then ``<root>``.
        sample_rate: target rate (model needs 24000; LJSpeech natives 22050).
        hop_length: audio truncated to a multiple of this (model needs 240).
        min_duration_s / max_duration_s: header-only duration filter bounds
            (native rate; resampled estimates are stored for bucketing).
        use_normalized: prefer the normalized (3rd) transcript column.
        audio_subdirs: subdirectories probed under ``root`` for audio.
    """

    def __init__(
        self,
        root: Optional[Union[str, Path]] = None,
        metadata_path: Optional[Union[str, Path]] = None,
        audio_dir: Optional[Union[str, Path]] = None,
        sample_rate: int = 24000,
        hop_length: int = 240,
        min_duration_s: float = 0.5,
        max_duration_s: float = 12.0,
        use_normalized: bool = True,
        audio_subdirs: Sequence[str] = LJSPEECH_AUDIO_SUBDIRS,
    ):
        super().__init__()
        self.sample_rate = int(sample_rate)
        self.hop_length = int(hop_length)
        self.min_duration_s = float(min_duration_s)
        self.max_duration_s = float(max_duration_s)
        self.use_normalized = bool(use_normalized)
        self.phonemizer = Phonemizer()

        self.root, self.metadata_file = self._resolve_root_and_metadata(root, metadata_path)
        self.audio_search_dirs = self._resolve_audio_dirs(audio_dir, audio_subdirs)

        self.items: List[Dict[str, Any]] = []
        self._lengths: List[int] = []  # resampled estimates, index-aligned
        if self.metadata_file is not None:
            self._build_index()

    # ------------------------------------------------------------------
    # Path resolution (pure, no audio I/O)
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_root_and_metadata(
        root: Optional[Union[str, Path]],
        metadata_path: Optional[Union[str, Path]],
    ) -> Tuple[Optional[Path], Optional[Path]]:
        meta: Optional[Path] = Path(metadata_path) if metadata_path else None
        base: Optional[Path] = Path(root) if root else None
        if base is not None and base.is_file():
            # `root` pointed straight at the csv file.
            meta = base if meta is None else meta
            base = meta.parent
        if meta is None and base is not None:
            candidate = base / LJSPEECH_METADATA_NAME
            meta = candidate if candidate.is_file() else None
            if meta is None and base.name == LJSPEECH_METADATA_NAME and base.is_file():
                meta = base
                base = base.parent
        if base is None and meta is not None:
            base = meta.parent
        if meta is not None and not meta.is_file():
            meta = None
        return base, meta

    def _resolve_audio_dirs(
        self,
        audio_dir: Optional[Union[str, Path]],
        audio_subdirs: Sequence[str],
    ) -> List[Path]:
        dirs: List[Path] = []
        if audio_dir is not None:
            dirs.append(Path(audio_dir))
        if self.root is not None:
            for sub in audio_subdirs:
                dirs.append(self.root / sub)
            dirs.append(self.root)
        # De-dupe while preserving order.
        seen: set = set()
        unique: List[Path] = []
        for d in dirs:
            key = str(d)
            if key not in seen:
                seen.add(key)
                unique.append(d)
        return unique

    # ------------------------------------------------------------------
    # Index building (header-only scan, never decodes audio bytes)
    # ------------------------------------------------------------------

    def _resolve_audio_file(self, file_id: str) -> Optional[str]:
        for candidate in _candidate_audio_paths(file_id, self.audio_search_dirs):
            if candidate.is_file():
                return str(candidate)
        return None

    def _build_index(self) -> None:
        assert self.metadata_file is not None
        missing = 0
        skipped_text = 0
        skipped_duration = 0
        with open(self.metadata_file, "r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f, delimiter="|")
            for parts in reader:
                if not parts or not any(p.strip() for p in parts):
                    continue
                try:
                    file_id, text = parse_ljspeech_row(parts, self.use_normalized)
                except ValueError:
                    skipped_text += 1
                    continue
                audio_path = self._resolve_audio_file(file_id)
                if audio_path is None:
                    missing += 1
                    continue
                try:
                    info = sf.info(audio_path)
                    duration_s = float(info.frames) / float(info.samplerate or 1)
                except Exception:
                    missing += 1
                    continue
                if not (self.min_duration_s <= duration_s <= self.max_duration_s):
                    skipped_duration += 1
                    continue
                est = int(duration_s * self.sample_rate)
                self.items.append({"audio_path": audio_path, "text": text, "file": file_id})
                self._lengths.append((est // self.hop_length) * self.hop_length)
        if (missing or skipped_text or skipped_duration) and len(self.items) == 0:
            raise FileNotFoundError(
                f"No usable LJSpeech rows in {self.metadata_file} "
                f"(missing_audio={missing}, bad_text={skipped_text}, "
                f"out_of_duration={skipped_duration}, searched={self.audio_search_dirs})"
            )

    # ------------------------------------------------------------------
    # Dataset protocol (mirrors HiFiTTSDataset output + speaker/file tags)
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.items)

    @property
    def audio_lengths(self) -> List[int]:
        """Resampled audio-length estimates (samples, for bucketed batching)."""
        return list(self._lengths)

    def get_audio_lengths(self) -> List[int]:
        """Alias matching :meth:`HiFiTTSDataset.get_audio_lengths`."""
        return self.audio_lengths

    def _load_audio(self, audio_path: str) -> np.ndarray:
        """Loads and normalizes audio to ``sample_rate`` mono float32."""
        data, sr = sf.read(audio_path, dtype="float32")
        data = resample_mono(np.asarray(data, dtype=np.float32), int(sr), self.sample_rate)
        peak = float(np.abs(data).max()) if data.size else 0.0
        if peak > 1e-4:
            data = (0.95 * (data / peak)).astype(np.float32)
        return data

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.items[idx]
        audio = self._load_audio(item["audio_path"])
        tokens = self.phonemizer.text_to_sequence(item["text"])
        num_frames = len(audio) // self.hop_length
        audio = audio[: num_frames * self.hop_length]
        return {
            "tokens": torch.tensor(tokens, dtype=torch.long),
            "audio": torch.tensor(audio, dtype=torch.float32),
            "text": item["text"],
            "file": item.get("file", ""),
            "speaker": LJSPEECH_SPEAKER,
        }

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(metadata_file={str(self.metadata_file)!r}, "
            f"audio_dirs={[str(d) for d in self.audio_search_dirs]!r}, "
            f"num_rows={len(self)})"
        )
