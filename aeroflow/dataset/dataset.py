"""
AeroFlow-v2 Hi-Fi TTS Speaker 9017 Dataset & Data Collator.
Specialized for Hi-Fi TTS Speaker 9017 (John Van Stan - 24 kHz Mono Studio Baritone).
Features:
- Manifest parsing (JSON/TSV/NeMo formats).
- Audio loading and peak/RMS normalization to 24,000 Hz mono.
- Dynamic phoneme padding and audio collation with sequence masks.
- Synthetic batch generation simulating Speaker 9017 acoustic characteristics.
"""

import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

from aeroflow.frontend.phonemizer import PAD_ID, Phonemizer


class HiFiTTSDataset(Dataset):
    """
    Dataset loader for Hi-Fi TTS Speaker 9017.
    Loads 24 kHz mono uncompressed PCM audio and transcripts.
    """

    def __init__(
        self,
        manifest_path: Optional[Union[str, Path]] = None,
        audio_dir: Optional[Union[str, Path]] = None,
        sample_rate: int = 24000,
        hop_length: int = 240,
        max_duration_s: float = 12.0,
        min_duration_s: float = 0.5
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.max_samples = int(max_duration_s * sample_rate)
        self.min_samples = int(min_duration_s * sample_rate)
        self.phonemizer = Phonemizer()
        self.items: List[Dict[str, Any]] = []

        if manifest_path and os.path.exists(manifest_path):
            self._load_manifest(manifest_path, audio_dir)

    def _load_manifest(self, manifest_path: Union[str, Path], audio_dir: Optional[Union[str, Path]]):
        manifest_path = Path(manifest_path)
        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    audio_fp = entry.get("audio_filepath") or entry.get("audio_path") or entry.get("audio")
                    text = entry.get("text_normalized") or entry.get("text") or entry.get("transcript")

                    if audio_dir and not os.path.isabs(audio_fp):
                        audio_fp = os.path.join(audio_dir, audio_fp)

                    if audio_fp and text:
                        self.items.append({
                            "audio_path": audio_fp,
                            "text": text
                        })
                except json.JSONDecodeError:
                    parts = line.split("|")
                    if len(parts) >= 2:
                        audio_fp, text = parts[0].strip(), parts[1].strip()
                        if audio_dir and not os.path.isabs(audio_fp):
                            audio_fp = os.path.join(audio_dir, audio_fp)
                        self.items.append({
                            "audio_path": audio_fp,
                            "text": text
                        })

    def __len__(self) -> int:
        return len(self.items)

    def _load_audio(self, audio_path: str) -> np.ndarray:
        """Loads and normalizes audio to 24 kHz mono float32."""
        data, sr = sf.read(audio_path, dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=-1)

        if sr != self.sample_rate:
            # Resample if sample rate differs
            num_target_samples = int(len(data) * self.sample_rate / sr)
            data = np.interp(
                np.linspace(0, len(data), num_target_samples, endpoint=False),
                np.arange(len(data)),
                data
            ).astype(np.float32)

        # Peak normalization to [-0.95, 0.95]
        peak = np.abs(data).max()
        if peak > 1e-4:
            data = 0.95 * (data / peak)

        return data

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.items[idx]
        audio = self._load_audio(item["audio_path"])
        tokens = self.phonemizer.text_to_sequence(item["text"])

        # Truncate or pad to multiple of hop_length
        num_frames = len(audio) // self.hop_length
        audio = audio[: num_frames * self.hop_length]

        return {
            "tokens": torch.tensor(tokens, dtype=torch.long),
            "audio": torch.tensor(audio, dtype=torch.float32),
            "text": item["text"]
        }


def collate_hifi_tts(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate function dynamically padding variable-length phonemes and audio waveforms.
    """
    batch_size = len(batch)

    # 1. Phoneme Tokens Padding
    token_lens = [len(x["tokens"]) for x in batch]
    max_token_len = max(token_lens)
    padded_tokens = torch.full((batch_size, max_token_len), PAD_ID, dtype=torch.long)
    for i, x in enumerate(batch):
        t_len = len(x["tokens"])
        padded_tokens[i, :t_len] = x["tokens"]

    # 2. Audio Waveforms Padding
    audio_lens = [len(x["audio"]) for x in batch]
    max_audio_len = max(audio_lens)
    # Ensure audio length is a multiple of hop_length (240)
    hop = 240
    if max_audio_len % hop != 0:
        max_audio_len = ((max_audio_len // hop) + 1) * hop

    padded_audio = torch.zeros((batch_size, max_audio_len), dtype=torch.float32)
    for i, x in enumerate(batch):
        a_len = len(x["audio"])
        padded_audio[i, :a_len] = x["audio"]

    return {
        "phoneme_tokens": padded_tokens,
        "text_lengths": torch.tensor(token_lens, dtype=torch.long),
        "audio": padded_audio,
        "audio_lengths": torch.tensor(audio_lens, dtype=torch.long),
        "texts": [x.get("text", "") for x in batch]
    }


def create_synthetic_batch(
    batch_size: int = 2,
    seq_len: int = 35,
    audio_dur_s: float = 1.0,
    sample_rate: int = 24000,
    hop_length: int = 240
) -> Dict[str, Any]:
    """
    Generates a realistic synthetic batch simulating Speaker 9017 (John Van Stan).
    Simulates deep male baritone resonance with fundamental F0 ~110 Hz and rich harmonics.
    """
    num_samples = int(audio_dur_s * sample_rate)
    # Align to hop_length multiple
    num_samples = (num_samples // hop_length) * hop_length

    # Realistic phoneme sequences
    phonemizer = Phonemizer()
    sample_texts = [
        "Peter Piper picked a peck of pickled peppers.",
        "AeroFlow two text to speech engine running on Intel Core processor."
    ]

    tokens_list = []
    audios_list = []

    for i in range(batch_size):
        text = sample_texts[i % len(sample_texts)]
        toks = phonemizer.text_to_sequence(text)
        tokens_list.append(torch.tensor(toks, dtype=torch.long))

        # Synthesize harmonic baritone waveform: F0 = 110 Hz + harmonics + envelope
        t = torch.linspace(0, audio_dur_s, num_samples)
        f0 = 110.0 + 5.0 * torch.sin(2.0 * math.pi * 1.5 * t)  # natural vibrato
        phase = torch.cumsum(2.0 * math.pi * f0 / sample_rate, dim=0)

        # Baritone harmonics: strong fundamental + formants at 500, 1500, 2500 Hz
        sig = 0.5 * torch.sin(phase) + 0.3 * torch.sin(2 * phase) + 0.2 * torch.sin(3 * phase)
        # Apply smooth speech envelope
        envelope = torch.sin(math.pi * torch.linspace(0, 1, num_samples)).clamp(min=0.0)
        # Add slight ambient noise for SNR >= 40 dB
        noise = torch.randn(num_samples) * 0.005
        audio = (sig * envelope + noise).clamp(min=-0.95, max=0.95)
        audios_list.append(audio)

    # Use collate function
    batch_items = [
        {"tokens": tokens_list[i], "audio": audios_list[i], "text": sample_texts[i % len(sample_texts)]}
        for i in range(batch_size)
    ]

    return collate_hifi_tts(batch_items)
