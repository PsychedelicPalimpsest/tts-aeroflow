"""Offline pronunciation decisions and a filtered training dataset.

The audit file is intentionally small: it stores IDs and phoneme tokens, never audio.
Only accepted rows appear in the training index. A text digest prevents applying a
decision to a changed transcript.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterator, Mapping

import torch
from torch.utils.data import Dataset, IterableDataset

from aeroflow.frontend.phonemizer import PHONEME_TO_ID


def decision_key(speaker: str, file: str) -> str:
    return json.dumps([str(speaker), str(file)], ensure_ascii=False)


def text_digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_decisions(path: str | Path) -> Dict[str, Dict[str, Any]]:
    decisions: Dict[str, Dict[str, Any]] = {}
    with open(path, encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("status") != "accepted":
                continue
            key = decision_key(row["speaker"], row["file"])
            phones = row["phonemes"]
            if (key in decisions or not isinstance(phones, list) or not phones
                    or any(p not in PHONEME_TO_ID for p in phones)):
                raise ValueError(f"Invalid or duplicate pronunciation decision at line {line_number}")
            decisions[key] = {"text_sha256": row["text_sha256"], "phonemes": phones}
    if not decisions:
        raise ValueError("Pronunciation manifest contains no accepted clips")
    return decisions


def _metadata(dataset: Dataset, index: int, hf_metadata: Any = None) -> tuple[str, str, str]:
    if hasattr(dataset, "items"):
        item = dataset.items[index]
        speaker = "LJSpeech" if type(dataset).__name__ == "LJSpeechDataset" else ""
        return speaker, str(item.get("audio_path", "")), str(item["text"])
    if hasattr(dataset, "_hf") and hasattr(dataset, "_index"):
        row = (hf_metadata if hf_metadata is not None else dataset._hf)[dataset._index[index]]
        text = ""
        for field in (dataset.text_field, "text_normalized", "text", "text_no_preprocessing"):
            if isinstance(row.get(field), str) and row[field].strip():
                text = row[field].strip()
                break
        return str(row.get("speaker", "")), str(row.get("file", "")), text
    raise TypeError("Pronunciation filtering supports manifest, LJSpeech, and Hi-Fi TTS datasets")


def _approved(item: Mapping[str, Any], decisions: Mapping[str, Dict[str, Any]]) -> Dict[str, Any] | None:
    key = decision_key(item.get("speaker", ""), item.get("file", ""))
    row = decisions.get(key)
    if row is None or row["text_sha256"] != text_digest(str(item["text"])):
        return None
    result = dict(item)
    result["tokens"] = torch.tensor([PHONEME_TO_ID[p] for p in row["phonemes"]], dtype=torch.long)
    return result


class PronunciationFilteredDataset(Dataset):
    def __init__(self, dataset: Dataset, decisions: Mapping[str, Dict[str, Any]]):
        self.dataset = dataset
        self.decisions = decisions
        self.indices = []
        lengths = (dataset.get_audio_lengths() if hasattr(dataset, "get_audio_lengths")
                   else getattr(dataset, "audio_lengths", []))
        self._lengths = []
        hf_metadata = None
        if hasattr(dataset, "_hf") and "audio" in dataset._hf.column_names:
            hf_metadata = dataset._hf.remove_columns(["audio"])
        for index in range(len(dataset)):
            speaker, file, text = _metadata(dataset, index, hf_metadata)
            row = decisions.get(decision_key(speaker, file))
            if row and row["text_sha256"] == text_digest(text):
                self.indices.append(index)
                if lengths:
                    self._lengths.append(lengths[index])
        if not self.indices:
            raise ValueError("No clips match the pronunciation manifest and current transcripts")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        item = self.dataset[self.indices[index]]
        # Local and LJSpeech loaders expose a short `file` field, but the audit
        # uses the resolved audio path so separate datasets cannot collide.
        if hasattr(self.dataset, "items"):
            item = dict(item)
            item["file"] = self.dataset.items[self.indices[index]]["audio_path"]
            if type(self.dataset).__name__ == "LJSpeechDataset":
                item["speaker"] = "LJSpeech"
        approved = _approved(item, self.decisions)
        if approved is None:
            raise ValueError("Pronunciation manifest no longer matches dataset item")
        return approved

    def get_audio_lengths(self) -> list[int]:
        return list(self._lengths)


class PronunciationFilteredStream(IterableDataset):
    def __init__(self, dataset: IterableDataset, decisions: Mapping[str, Dict[str, Any]]):
        self.dataset = dataset
        self.decisions = decisions
        if hasattr(dataset, "_open_stream"):
            # Let the HF adapter reject rows before decoding their audio.
            dataset.pronunciation_decisions = decisions

    def set_epoch(self, epoch: int) -> None:
        if hasattr(self.dataset, "set_epoch"):
            self.dataset.set_epoch(epoch)

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        for item in self.dataset:
            approved = _approved(item, self.decisions)
            if approved is not None:
                yield approved
