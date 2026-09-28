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
    normalized = re.sub(r"-{2,}", ", ", normalized)
    parts = re.findall(r"[\w'-]+|[.,!?;:\"']", normalized)
    return parts, [part for part in parts if part not in PUNCTUATION_TOKENS and re.search(r"\w", part)]


FUNCTION_WORD_VARIANTS: dict[str, list[list[str]]] = {
    "and": [["AH0", "N"], ["IH0", "N"], ["AH0", "N", "D"], ["IH0", "N", "D"], ["AE1", "N"], ["AE0", "N"], ["AE0", "N", "D"]],
    "that": [["DH", "AH0", "T"], ["DH", "IH0", "T"], ["DH", "AE1", "DX"], ["DH", "AH0", "DX"], ["DH", "IH0", "DX"],
             ["DH", "EH0", "DX"], ["DH", "EH1", "DX"], ["DH", "AE1"], ["DH", "AH0"], ["DH", "EH1"], ["DH", "EH0"]],
    "had": [["HH", "AH0", "D"], ["HH", "IH0", "D"], ["HH", "EH0", "D"], ["HH", "EH1", "D"],
            ["HH", "AE1", "DX"], ["HH", "AH0", "DX"], ["HH", "EH1"], ["HH", "EH0"], ["HH", "AH0"], ["HH", "AE1"],
            ["AH0", "D"], ["IH0", "D"], ["AE1", "D"]],
    "at": [["AH0", "T"], ["IH0", "T"], ["AE1", "DX"], ["AH0", "DX"], ["IH0", "DX"], ["AH0"], ["IH0"], ["AE1"]],
    "it": [["IH1", "DX"], ["IH0", "DX"], ["IH1"], ["IH0"], ["AH0", "T"], ["AH0", "DX"]],
    "for": [["F", "ER0"], ["F", "AH0", "R"], ["F", "AO0", "R"], ["F", "AH0"], ["F", "AO0"]],
    "should": [["SH", "AH0", "D"], ["SH", "IH0", "D"], ["SH", "UH1"], ["SH", "AH0"], ["SH", "IH0"]],
    "would": [["W", "AH0", "D"], ["W", "IH0", "D"], ["W", "UH1"], ["W", "AH0"], ["W", "IH0"]],
    "could": [["K", "AH0", "D"], ["K", "IH0", "D"], ["K", "UH1"], ["K", "AH0"], ["K", "IH0"]],
    "of": [["AH0", "V"], ["AH0"], ["AH1"]],
    "to": [["T", "AH0"], ["T", "IH0"], ["DX", "AH0"], ["DX", "UW0"], ["T", "UW0"]],
    "in": [["IH0", "N"], ["AH0", "N"]],
    "as": [["AH0", "Z"], ["IH0", "Z"], ["AE0", "Z"]],
    "with": [["W", "IH0", "DH"], ["W", "IH0", "TH"], ["W", "AH0", "DH"], ["W", "AH0", "TH"]],
    "from": [["F", "ER0", "M"], ["F", "AH0", "M"], ["F", "R", "AH0", "M"], ["F", "R", "AH1", "M"]],
    "or": [["ER0"], ["AH0", "R"], ["AO0"]],
    "not": [["N", "AA1", "DX"], ["N", "AH0", "T"], ["N", "AA1"]],
    "but": [["B", "AH0", "T"], ["B", "AH0", "DX"], ["B", "AH0"]],
    "he": [["IY1"], ["IY0"], ["HH", "IY0"]],
    "her": [["ER0"], ["HH", "ER0"]],
    "him": [["IH1", "M"], ["IH0", "M"]],
    "his": [["IH1", "Z"], ["IH0", "Z"]],
    "them": [["DH", "AH0", "M"], ["AH0", "M"]],
    "was": [["W", "AH0", "Z"], ["W", "IH0", "Z"], ["W", "AA0", "Z"]],
    "have": [["HH", "AH0", "V"], ["AH0", "V"], ["HH", "AE0", "V"]],
    "has": [["HH", "AH0", "Z"], ["AH0", "Z"], ["HH", "AE0", "Z"]],
    "are": [["ER0"], ["AA0", "R"]],
}


def variants(word: str, phonemizer: Phonemizer, dictionary: dict) -> list[list[str]]:
    w_clean = word.lower().strip()
    candidates: list[list[str]] = []

    # 1. Primary dictionary pronunciations
    if w_clean in dictionary:
        for candidate in dictionary[w_clean]:
            phones = list(candidate)
            if phones not in candidates and all(phone in PHONEME_TO_ID for phone in phones):
                candidates.append(phones)

    # 2. Hyphenated compound words
    if "-" in w_clean and not w_clean.startswith("-") and not w_clean.endswith("-"):
        subwords = [sw for sw in w_clean.split("-") if sw]
        sub_cands = []
        for sw in subwords:
            if sw in dictionary:
                sub_cands.append(dictionary[sw])
            else:
                sub_cands.append([phonemizer.phonemize_word(sw)])
        import itertools
        for combo in itertools.product(*sub_cands):
            cand = [ph for sub in combo for ph in sub]
            if cand not in candidates and all(phone in PHONEME_TO_ID for phone in cand):
                candidates.append(cand)

    # 3. English possessive 's inflection
    if w_clean.endswith("'s"):
        stem = w_clean[:-2]
        if stem in dictionary:
            for candidate in dictionary[stem]:
                last_p = plain(candidate[-1])
                if last_p in {"S", "Z", "SH", "ZH", "CH", "JH"}:
                    cand = list(candidate) + ["IH0", "Z"]
                elif last_p in {"P", "T", "K", "F", "TH"}:
                    cand = list(candidate) + ["S"]
                else:
                    cand = list(candidate) + ["Z"]
                if cand not in candidates and all(phone in PHONEME_TO_ID for phone in cand):
                    candidates.append(cand)

    # 4. Morphological 2-part compound decomposition for words not in CMUdict
    if not candidates and len(w_clean) >= 6:
        for split_idx in range(3, len(w_clean) - 2):
            w1, w2 = w_clean[:split_idx], w_clean[split_idx:]
            if w1 in dictionary and w2 in dictionary:
                for p1 in dictionary[w1]:
                    for p2 in dictionary[w2]:
                        comb = list(p1) + [re.sub(r"1$", "2", p) if p.endswith("1") else p for p in p2]
                        if comb not in candidates and all(phone in PHONEME_TO_ID for phone in comb):
                            candidates.append(comb)

    # 5. Baseline from phonemizer
    baseline = phonemizer.phonemize_word(word)
    if baseline not in candidates and all(phone in PHONEME_TO_ID for phone in baseline):
        candidates.insert(0, baseline)
    elif not candidates:
        candidates.append(baseline)

    # 6. Function word weak forms and reductions
    if w_clean in FUNCTION_WORD_VARIANTS:
        for fw_cand in FUNCTION_WORD_VARIANTS[w_clean]:
            if fw_cand not in candidates and all(phone in PHONEME_TO_ID for phone in fw_cand):
                candidates.append(fw_cand)

    # 7. Conservative regional and allophonic options.
    # Acoustic evidence must still beat the baseline to select one; these are never applied by spelling alone.
    for source in list(candidates):
        for index, phone in enumerate(source):
            # English /ɚ, ɝ/ may be recognized as one ER phone or as a
            # vowel followed by R. Preserve the source stress on the vowel.
            if plain(phone) == "ER" and phone[-1] in "012":
                for vowel in ("AH", "EH"):
                    alternate = source[:index] + [vowel + phone[-1], "R"] + source[index + 1:]
                    if alternate not in candidates:
                        candidates.append(alternate)
            # Postvocalic R-deletion
            if (phone == "R" and index > 0 and plain(source[index - 1]) in
                    {"AA", "AE", "AH", "AO", "EH", "ER", "IH", "IY", "UH", "UW"}
                    and (index == len(source) - 1 or plain(source[index + 1]) not in
                         {"AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW"})):
                alternate = source[:index] + source[index + 1:]
                if alternate and alternate not in candidates:
                    candidates.append(alternate)
            # Intervocalic and word-final flapping (DX)
            if (phone in {"T", "D"} and index > 0 and
                    re.match(r"^(AA|AE|AH|AO|AW|AY|EH|ER|EY|IH|IY|OW|OY|UH|UW)[012]$", source[index - 1])):
                if (index < len(source) - 1 and
                        re.match(r"^(AA|AE|AH|AO|AW|AY|EH|ER|EY|IH|IY|OW|OY|UH|UW)[012]$", source[index + 1])):
                    alternate = source[:index] + ["DX"] + source[index + 1:]
                    if alternate not in candidates:
                        candidates.append(alternate)
                elif index == len(source) - 1:
                    alternate = source[:index] + ["DX"]
                    if alternate not in candidates:
                        candidates.append(alternate)
            # Unstressed weak vowel alternation (weak vowel merger: AH0 <-> IH0)
            if phone == "AH0":
                alternate = source[:index] + ["IH0"] + source[index + 1:]
                if alternate not in candidates:
                    candidates.append(alternate)
            elif phone == "IH0":
                alternate = source[:index] + ["AH0"] + source[index + 1:]
                if alternate not in candidates:
                    candidates.append(alternate)
            # Father-bother / cot-caught merger before R
            if phone.startswith("AO") and index < len(source) - 1 and plain(source[index + 1]) == "R":
                alternate = source[:index] + ["AA" + phone[-1]] + source[index + 1:]
                if alternate not in candidates:
                    candidates.append(alternate)
            elif phone.startswith("AA") and index < len(source) - 1 and plain(source[index + 1]) == "R":
                alternate = source[:index] + ["AO" + phone[-1]] + source[index + 1:]
                if alternate not in candidates:
                    candidates.append(alternate)
            # Yod-coalescence: S Y -> SH, Z Y -> ZH, T Y -> CH, D Y -> JH
            if index < len(source) - 1 and plain(source[index + 1]) == "Y":
                coalesce_map = {"S": "SH", "Z": "ZH", "T": "CH", "D": "JH"}
                p_plain = plain(phone)
                if p_plain in coalesce_map:
                    alternate = source[:index] + [coalesce_map[p_plain]] + source[index + 2:]
                    if alternate not in candidates:
                        candidates.append(alternate)
            # Nasal raising / pin-pen merger: EH -> IH before N/M/NG
            if plain(phone) == "EH" and index < len(source) - 1 and plain(source[index + 1]) in {"N", "M", "NG"}:
                alternate = source[:index] + ["IH" + phone[-1]] + source[index + 1:]
                if alternate not in candidates:
                    candidates.append(alternate)
            # Coronal stop cluster simplification in word-final position
            if index == len(source) - 1 and len(source) >= 2:
                prev_p = plain(source[index - 1])
                curr_p = plain(phone)
                if (curr_p == "D" and prev_p == "N") or (curr_p == "T" and prev_p in {"S", "F", "P", "K"}):
                    alternate = source[:-1]
                    if alternate and alternate not in candidates:
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

    def _normalize_words(self, words: list[str]) -> list[str]:
        out = []
        i = 0
        while i < len(words):
            if len(words[i]) == 1 and i + 1 < len(words) and len(words[i + 1]) == 1:
                acronym = words[i]
                while i + 1 < len(words) and len(words[i + 1]) == 1:
                    i += 1
                    acronym += words[i]
                out.append(acronym)
                i += 1
            else:
                out.append(words[i])
                i += 1
        return out

    def _align_compounds(self, w1: list[str], w2: list[str]) -> tuple[list[str], list[str]]:
        i, j = 0, 0
        res1, res2 = [], []
        while i < len(w1) and j < len(w2):
            if w1[i] == w2[j]:
                res1.append(w1[i])
                res2.append(w2[j])
                i += 1
                j += 1
            elif j + 1 < len(w2) and w1[i] == w2[j] + w2[j + 1]:
                res1.append(w1[i])
                res2.append(w1[i])
                i += 1
                j += 2
            elif i + 1 < len(w1) and w1[i] + w1[i + 1] == w2[j]:
                res1.append(w2[j])
                res2.append(w2[j])
                i += 2
                j += 1
            else:
                res1.append(w1[i])
                res2.append(w2[j])
                i += 1
                j += 1
        while i < len(w1):
            res1.append(w1[i])
            i += 1
        while j < len(w2):
            res2.append(w2[j])
            j += 1
        return res1, res2

    def _word_match(self, w1: str, w2: str) -> int:
        if w1 == w2:
            return 0
        if len(w1) >= 4 and len(w2) >= 4 and abs(len(w1) - len(w2)) <= 1:
            costs = list(range(len(w2) + 1))
            for i, c1 in enumerate(w1, 1):
                nc = [i] + [0] * len(w2)
                for j, c2 in enumerate(w2, 1):
                    nc[j] = min(costs[j] + 1, nc[j - 1] + 1, costs[j - 1] + (c1 != c2))
                costs = nc
            if costs[-1] <= 1:
                return 0
        return 1

    def _compute_costs(self, exp: list[str], hrd: list[str]) -> float:
        e_aligned, h_aligned = self._align_compounds(self._normalize_words(exp), self._normalize_words(hrd))
        costs = list(range(len(h_aligned) + 1))
        for i, word in enumerate(e_aligned, 1):
            next_costs = [i] + [0] * len(h_aligned)
            for j, other in enumerate(h_aligned, 1):
                next_costs[j] = min(costs[j] + 1, next_costs[j - 1] + 1,
                                    costs[j - 1] + self._word_match(word, other))
            costs = next_costs
        return costs[-1] / max(len(exp), 1)

    def word_error_ratio(self, audio_16k: np.ndarray, transcript: str) -> float:
        segments, _ = self.model.transcribe(audio_16k, beam_size=3, language="en",
                                            vad_filter=False)
        raw_heard = " ".join(segment.text.strip() for segment in segments)
        expected_words = re.findall(r"[a-z']+", self.normalizer.normalize(transcript).lower())

        heard_words_std = re.findall(r"[a-z']+", self.normalizer.normalize(raw_heard).lower())
        wer_std = self._compute_costs(expected_words, heard_words_std)

        digit_words = {"0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
                       "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine"}
        raw_digits_expanded = re.sub(r"\d", lambda m: " " + digit_words[m.group(0)] + " ", raw_heard)
        heard_words_digits = re.findall(r"[a-z']+", self.normalizer.normalize(raw_digits_expanded).lower())
        wer_digits = self._compute_costs(expected_words, heard_words_digits)

        return min(wer_std, wer_digits)


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
        denom = max(len(phones), len(segment), 1)
        if local_cost >= 2 and local_cost / denom > 0.5:
            # Check boundary slack (+/- 1 phone) to ensure alignment boundary jitter
            # does not falsely flag a cleanly spoken word.
            slack_costs = [local_cost]
            for s_off in (-1, 0, 1):
                for e_off in (-1, 0, 1):
                    if s_off == 0 and e_off == 0:
                        continue
                    ns = max(0, start + s_off)
                    ne = min(len(normalized_heard), max(ns, end + e_off))
                    seg = normalized_heard[ns:ne]
                    slack_costs.append(_advance(list(range(len(seg) + 1)),
                                                [plain(p) for p in phones], seg)[0][-1])
            min_slack_cost = min(slack_costs)
            if min_slack_cost >= 2 and min_slack_cost / max(len(phones), 1) > 0.5:
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
