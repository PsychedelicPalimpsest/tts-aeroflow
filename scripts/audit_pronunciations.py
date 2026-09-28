"""Audit recorded speech against plausible ARPAbet pronunciations.

Run this once before TTS training. The HuPER model hears phones from audio;
CMUdict supplies plausible word variants. Ambiguous or poorly matching clips
are rejected. Decisions are written as JSONL for --pronunciation-manifest.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aeroflow.dataset.audio import resample_mono
from aeroflow.dataset.dataset import HiFiTTSDataset
from aeroflow.dataset.hf_hifi_tts import StreamingHiFiTTSDataset
from aeroflow.dataset.ljspeech import LJSpeechDataset
from aeroflow.dataset.pronunciation import decision_key, text_digest
from aeroflow.frontend.phonemizer import PHONEME_TO_ID, PUNCTUATION_TOKENS, Phonemizer

DEFAULT_MODEL = "huper29/huper_recognizer"
SPECIAL_LABELS = {"<PAD>", "<UNK>", "<BOS>", "<EOS>", "|"}


def plain(phone: str) -> str:
    return re.sub(r"[012]$", "", phone.upper())


def spoken_words(text: str, phonemizer: Phonemizer) -> tuple[list[str], list[str]]:
    normalized = phonemizer.normalizer.normalize(text)
    parts = re.findall(r"[\w'-]+|[.,!?;:\"']", normalized)
    return parts, [part for part in parts if part not in PUNCTUATION_TOKENS]


def variants(word: str, phonemizer: Phonemizer, dictionary: dict) -> list[list[str]]:
    baseline = phonemizer.phonemize_word(word)
    candidates = [baseline]
    for candidate in dictionary.get(word.lower(), []):
        phones = list(candidate)
        if phones not in candidates and all(phone in PHONEME_TO_ID for phone in phones):
            candidates.append(phones)
    # Conservative regional/allophonic options. Acoustic evidence must still
    # beat the baseline to select one; these are never applied by spelling alone.
    for source in list(candidates):
        for index, phone in enumerate(source):
            # English /ɚ, ɝ/ may be recognized as one ER phone or as a
            # vowel followed by R. Preserve the source stress on the vowel.
            if plain(phone) == "ER" and phone[-1] in "012":
                for vowel in ("AH", "EH"):
                    alternate = source[:index] + [vowel + phone[-1], "R"] + source[index + 1:]
                    if alternate not in candidates:
                        candidates.append(alternate)
            if (phone == "R" and index > 0 and plain(source[index - 1]) in
                    {"AA", "AE", "AH", "AO", "EH", "ER", "IH", "IY", "UH", "UW"}
                    and (index == len(source) - 1 or plain(source[index + 1]) not in
                         {"AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW"})):
                alternate = source[:index] + source[index + 1:]
                if alternate and alternate not in candidates:
                    candidates.append(alternate)
            if (phone in {"T", "D"} and 0 < index < len(source) - 1 and
                    re.match(r"^(AA|AE|AH|AO|AW|AY|EH|ER|EY|IH|IY|OW|OY|UH|UW)[012]$", source[index - 1]) and
                    re.match(r"^(AA|AE|AH|AO|AW|AY|EH|ER|EY|IH|IY|OW|OY|UH|UW)[012]$", source[index + 1])):
                alternate = source[:index] + ["DX"] + source[index + 1:]
                if alternate not in candidates:
                    candidates.append(alternate)
    return candidates


def _advance(costs: list[int], phones: Sequence[str], observed: Sequence[str]) -> tuple[list[int], list[int]]:
    """Edit-distance transitions; retain the starting observation offset."""
    size = len(observed)
    prior = list(costs)
    origins = list(range(size + 1))
    for phone in phones:
        current = [prior[0] + 1] + [0] * size
        current_origins = [origins[0]] + [0] * size
        for j in range(1, size + 1):
            choices = (
                (prior[j - 1] + (phone != observed[j - 1]), origins[j - 1]),
                (prior[j] + 1, origins[j]),
                (current[j - 1] + 1, current_origins[j - 1]),
            )
            current[j], current_origins[j] = min(choices, key=lambda choice: choice[0])
        prior, origins = current, current_origins
    return prior, origins


def select_variants(groups: list[list[list[str]]], heard: list[str]) -> tuple[list[int], int, list[tuple[int, int]]]:
    """Find the lowest phone edit distance across all dictionary variants."""
    observed = [plain(phone) for phone in heard]
    costs = list(range(len(observed) + 1))
    history: list[list[tuple[int, int]]] = []
    for group in groups:
        best = [10**9] * len(costs)
        picks = [(0, 0)] * len(costs)
        for variant_index, candidate in enumerate(group):
            next_cost, origins = _advance(costs, [plain(p) for p in candidate], observed)
            for j, value in enumerate(next_cost):
                if value < best[j]:
                    best[j] = value
                    picks[j] = (origins[j], variant_index)
        costs = best
        history.append(picks)
    offset = len(observed)
    chosen = [0] * len(groups)
    spans = [(0, 0)] * len(groups)
    for word_index in range(len(groups) - 1, -1, -1):
        end = offset
        offset, chosen[word_index] = history[word_index][end]
        spans[word_index] = (offset, end)
    return chosen, costs[-1], spans


def render_phonemes(parts: list[str], selected: Sequence[Sequence[str]]) -> list[str]:
    result = ["<bos>"]
    word_index = 0
    for i, part in enumerate(parts):
        if part in PUNCTUATION_TOKENS:
            result.append(part)
        else:
            result.extend(selected[word_index])
            word_index += 1
            if i + 1 < len(parts) and parts[i + 1] not in PUNCTUATION_TOKENS:
                result.append("<sp>")
    result.append("<eos>")
    return result


class PhoneRecognizer:
    def __init__(self, model_id: str, device: str):
        from transformers import Wav2Vec2Processor, WavLMForCTC

        self.device = torch.device(device)
        self.processor = Wav2Vec2Processor.from_pretrained(model_id)
        self.model = WavLMForCTC.from_pretrained(model_id).to(self.device).eval()
        self.blank = self.processor.tokenizer.pad_token_id

    @torch.inference_mode()
    def recognize(self, audio_24k: np.ndarray) -> list[str]:
        audio_16k = resample_mono(audio_24k, 24000, 16000)
        inputs = self.processor(audio_16k, sampling_rate=16000, return_tensors="pt")
        logits = self.model(**{key: value.to(self.device) for key, value in inputs.items()}).logits
        ids = logits.argmax(dim=-1)[0].tolist()
        phones = []
        previous = None
        for token_id in ids:
            if token_id != self.blank and token_id != previous:
                label = self.model.config.id2label.get(token_id)
                if label is None:
                    label = self.model.config.id2label.get(str(token_id))
                if label is None:
                    label = self.processor.tokenizer.convert_ids_to_tokens(token_id)
                if isinstance(label, str) and label.upper() not in SPECIAL_LABELS:
                    phones.append(label.upper())
            previous = token_id
        return phones


class TranscriptChecker:
    """Independent word recognizer used only for borderline phone matches."""

    def __init__(self, model_name: str, device: str, download_root: str | None = None):
        from faster_whisper import WhisperModel

        compute_type = "float16" if device.startswith("cuda") else "int8"
        self.model = WhisperModel(model_name, device=device, compute_type=compute_type,
                                  download_root=download_root)
        self.normalizer = Phonemizer().normalizer

    def word_error_ratio(self, audio_16k: np.ndarray, transcript: str) -> float:
        segments, _ = self.model.transcribe(audio_16k, beam_size=3, language="en",
                                            vad_filter=False)
        heard = " ".join(segment.text.strip() for segment in segments)
        expected_words = re.findall(r"[a-z']+", self.normalizer.normalize(transcript).lower())
        heard_words = re.findall(r"[a-z']+", self.normalizer.normalize(heard).lower())
        costs = list(range(len(heard_words) + 1))
        for i, word in enumerate(expected_words, 1):
            next_costs = [i] + [0] * len(heard_words)
            for j, other in enumerate(heard_words, 1):
                next_costs[j] = min(costs[j] + 1, next_costs[j - 1] + 1,
                                    costs[j - 1] + (word != other))
            costs = next_costs
        return costs[-1] / max(len(expected_words), 1)


def audit_item(item: dict, phonemizer: Phonemizer, dictionary: dict,
               recognizer: PhoneRecognizer, max_error_ratio: float,
               min_correction_gain: int, transcript_checker: TranscriptChecker | None = None,
               rescue_max_error_ratio: float = 0.20, max_asr_wer: float = 0.10) -> dict:
    parts, words = spoken_words(item["text"], phonemizer)
    if not words:
        return {"status": "rejected", "reason": "no_words"}
    groups = [variants(word, phonemizer, dictionary) for word in words]
    audio = item["audio"].numpy()
    if len(audio) < 12000 or not np.isfinite(audio).all() or np.sqrt(np.mean(audio**2)) < 0.003:
        return {"status": "rejected", "reason": "invalid_or_silent_audio"}
    heard = recognizer.recognize(audio)
    if not heard or any(plain(p) not in {plain(x) for x in PHONEME_TO_ID} for p in heard):
        return {"status": "rejected", "reason": "unusable_phone_recognition", "heard": heard}
    chosen, cost, spans = select_variants(groups, heard)
    baseline_phones = [plain(p) for group in groups for p in group[0]]
    baseline_cost = _advance(list(range(len(heard) + 1)), baseline_phones,
                             [plain(p) for p in heard])[0][-1]
    selected = [group[index] for group, index in zip(groups, chosen)]
    changed = sum(index != 0 for index in chosen)
    if changed and baseline_cost - cost < min_correction_gain:
        return {"status": "rejected", "reason": "ambiguous_variant", "error": cost,
                "baseline_error": baseline_cost, "heard": heard}
    expected_count = sum(len(phones) for phones in selected)
    error_ratio = cost / max(len(heard), expected_count, 1)
    asr_wer = None
    if error_ratio > max_error_ratio:
        if transcript_checker is None or error_ratio > rescue_max_error_ratio:
            return {"status": "rejected", "reason": "audio_text_phone_mismatch",
                    "error_ratio": round(error_ratio, 4), "heard": heard}
        asr_wer = transcript_checker.word_error_ratio(
            resample_mono(audio, 24000, 16000), item["text"])
        if asr_wer > max_asr_wer:
            return {"status": "rejected", "reason": "asr_text_mismatch",
                    "error_ratio": round(error_ratio, 4), "asr_wer": round(asr_wer, 4),
                    "heard": heard}
    # A good sentence-level score can hide one wrong word in a long clip.
    normalized_heard = [plain(p) for p in heard]
    uncertain_words = []
    for word, phones, (start, end) in zip(words, selected, spans):
        segment = normalized_heard[start:end]
        local_cost = _advance(list(range(len(segment) + 1)),
                              [plain(p) for p in phones], segment)[0][-1]
        if local_cost >= 2 and local_cost / max(len(phones), len(segment), 1) > 0.5:
            uncertain_words.append(word)
    if uncertain_words:
        return {"status": "rejected", "reason": "word_phone_mismatch",
                "words": uncertain_words, "error_ratio": round(error_ratio, 4),
                "asr_wer": round(asr_wer, 4) if asr_wer is not None else None,
                "heard": heard}
    phones = render_phonemes(parts, selected)
    changes = [
        {"index": word_index, "word": word, "from": group[0], "to": group[index]}
        for word_index, (word, group, index) in enumerate(zip(words, groups, chosen))
        if index != 0
    ]
    return {"status": "accepted", "phonemes": phones, "changed_words": changed,
            "changes": changes, "error_ratio": round(error_ratio, 4),
            "asr_wer": round(asr_wer, 4) if asr_wer is not None else None,
            "heard": heard}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["manifest", "ljspeech", "hf-streaming"], required=True)
    parser.add_argument("--manifest-path")
    parser.add_argument("--audio-dir")
    parser.add_argument("--ljspeech-root")
    parser.add_argument("--hf-repo-id", default="MikhailT/hifi-tts")
    parser.add_argument("--hf-subset", default="clean")
    parser.add_argument("--hf-split", default="train")
    parser.add_argument("--hf-speaker", default="9017")
    parser.add_argument("--hf-min-duration", type=float, default=0.5)
    parser.add_argument("--hf-max-duration", type=float, default=10.0,
                        help="Match train_kaggle.py Hi-Fi TTS duration filter (default 10 s)")
    parser.add_argument("--output", required=True)
    parser.add_argument("--lexicon-output", help="Write high-consensus word overrides for inference")
    parser.add_argument("--resume-from", help="Previous JSONL audit; copy decisions and skip those clips")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-error-ratio", type=float, default=0.15)
    parser.add_argument("--rescue-max-error-ratio", type=float, default=0.20,
                        help="Borderline phone limit requiring independent ASR agreement")
    parser.add_argument("--max-asr-wer", type=float, default=0.10)
    parser.add_argument("--asr-model", default="base.en",
                        help="Faster-Whisper model for borderline clips; set 'none' to disable")
    parser.add_argument("--asr-device", default="cpu", help="Device for borderline transcript check")
    parser.add_argument("--asr-download-root", default=None)
    parser.add_argument("--min-correction-gain", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0, help="Audit first N clips for calibration")
    parser.add_argument("--new-items", type=int, default=0,
                        help="Stop after N new decisions, useful for Kaggle session limits")
    args = parser.parse_args()
    if (not 0 <= args.max_error_ratio <= args.rescue_max_error_ratio < 1
            or not 0 <= args.max_asr_wer < 1 or args.min_correction_gain < 1):
        parser.error("Invalid phone/ASR thresholds or correction gain")
    if args.source == "manifest":
        if not args.manifest_path:
            parser.error("--manifest-path is required for manifest source")
        dataset = HiFiTTSDataset(manifest_path=args.manifest_path, audio_dir=args.audio_dir)
    elif args.source == "ljspeech":
        if not args.ljspeech_root:
            parser.error("--ljspeech-root is required for ljspeech source")
        dataset = LJSpeechDataset(root=args.ljspeech_root)
    else:
        dataset = StreamingHiFiTTSDataset(repo_id=args.hf_repo_id, subset=args.hf_subset,
                                          split=args.hf_split, speaker_ids=[args.hf_speaker],
                                          min_duration_s=args.hf_min_duration,
                                          max_duration_s=args.hf_max_duration)
    import cmudict
    dictionary = cmudict.dict()
    phonemizer = Phonemizer()
    recognizer = PhoneRecognizer(args.model, args.device)
    checker = (None if args.asr_model.lower() == "none" else
               TranscriptChecker(args.asr_model, args.asr_device, args.asr_download_root))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    prior = {}
    if args.resume_from:
        with open(args.resume_from, encoding="utf-8") as previous:
            for line in previous:
                if line.strip():
                    row = json.loads(line)
                    prior[decision_key(row["speaker"], row["file"])] = row.get("text_sha256")
        if Path(args.resume_from).resolve() == output.resolve():
            raise ValueError("--resume-from and --output must be different files")
    written = 0
    with output.open("w", encoding="utf-8") as stream:
        if args.resume_from:
            with open(args.resume_from, encoding="utf-8") as previous:
                shutil.copyfileobj(previous, stream)
            with open(args.resume_from, "rb") as previous_bytes:
                previous_bytes.seek(0, 2)
                if previous_bytes.tell():
                    previous_bytes.seek(-1, 2)
                    if previous_bytes.read(1) != b"\n":
                        stream.write("\n")
        source_items = (enumerate(dataset) if args.source == "hf-streaming"
                        else ((index, None) for index in range(len(dataset))))
        for index, streamed_item in source_items:
            if args.limit and index >= args.limit:
                break
            if args.source == "hf-streaming":
                item = streamed_item
                speaker, file = item["speaker"], item["file"]
            else:
                speaker = "LJSpeech" if args.source == "ljspeech" else ""
                file = dataset.items[index]["audio_path"]
            text = (item["text"] if args.source == "hf-streaming"
                    else dataset.items[index]["text"])
            if prior.get(decision_key(speaker, file)) == text_digest(text):
                continue
            if args.new_items and written >= args.new_items:
                break
            if args.source != "hf-streaming":
                try:
                    item = dataset[index]
                except (OSError, ValueError, RuntimeError) as exc:
                    stream.write(json.dumps({"speaker": speaker, "file": file,
                                             "text_sha256": text_digest(text),
                                             "status": "rejected", "reason": "audio_load_error",
                                             "detail": str(exc)}) + "\n")
                    counts["audio_load_error"] += 1
                    written += 1
                    continue
            base = {"speaker": speaker, "file": file, "text": item["text"],
                    "text_sha256": text_digest(item["text"])}
            try:
                result = audit_item(item, phonemizer, dictionary, recognizer,
                                    args.max_error_ratio, args.min_correction_gain,
                                    transcript_checker=checker,
                                    rescue_max_error_ratio=args.rescue_max_error_ratio,
                                    max_asr_wer=args.max_asr_wer)
            except (OSError, ValueError, RuntimeError) as exc:
                result = {"status": "rejected", "reason": "processing_error", "detail": str(exc)}
            row = {**base, **result}
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            counts[row.get("reason", row["status"])] += 1
            written += 1
            if (index + 1) % 100 == 0:
                stream.flush()
                print(f"Audited {index + 1}: {dict(counts)}", flush=True)
    print(f"Wrote {output}: {dict(counts)}")
    if args.lexicon_output:
        usage: dict[str, Counter] = {}
        with output.open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("status") != "accepted" or not row.get("text"):
                    continue
                _, words = spoken_words(row["text"], phonemizer)
                corrected = {change["index"]: change["to"] for change in row.get("changes", [])}
                for word_index, word in enumerate(words):
                    phones = tuple(corrected.get(word_index, phonemizer.phonemize_word(word)))
                    usage.setdefault(word.lower(), Counter())[phones] += 1
        overrides = {}
        for word, counts_for_word in usage.items():
            total = sum(counts_for_word.values())
            phones, count = counts_for_word.most_common(1)[0]
            if (count >= 3 and count / total >= 0.8 and
                    list(phones) != phonemizer.phonemize_word(word)):
                overrides[word] = list(phones)
        lexicon_path = Path(args.lexicon_output)
        lexicon_path.parent.mkdir(parents=True, exist_ok=True)
        lexicon_path.write_text(json.dumps({"model": args.model, "overrides": overrides},
                                           ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {lexicon_path}: {len(overrides)} word overrides")


if __name__ == "__main__":
    main()
