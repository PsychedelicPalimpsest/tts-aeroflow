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
| Accepted directly by phone score and local word checks | 11 |
| Accepted after Whisper transcript check and local word checks | 7 |
| Rejected for a local word-to-phone mismatch | 16 |
| Rejected for phone error above 0.20 | 9 |
| Rejected because Whisper transcript did not match sufficiently | 5 |
| **Total** | **48** |

The final audit accepted **18/48 (37.5%)** and rejected **30/48 (62.5%)**.
All accepted clips had a selected pronunciation change relative to the
built-in phonemizer; the accepted clips contained 116 changed word instances.
The output consensus lexicon proposed two word overrides, `movable` and
`with`. These are model proposals rather than verified ground-truth labels.

Examples:

- `lj-001-0004` passed the Whisper check with zero transcript word errors
  after a phone error ratio of 0.1552; candidates included `books` as
  `B UH1 K S`.
- `lj-001-0005` passed directly at 0.1386; it selected
  `M UW1 V AH0 B AH0 L` for `movable`.
- `lj-004-0133` was rejected despite zero Whisper transcript word errors:
  the local phone match for `and` remained uncertain.
- `lj-017-0154` was rejected for the same reason at `that`, with phone error
  ratio 0.20 and zero Whisper transcript word errors.

The first strict phone-only pass accepted 11 of 48. Whisper confirmed the
transcript within 10% word error on 27 of the 37 clips that strict pass
rejected, so phone score alone was discarding many plausible transcripts.
After adding the transcript check, the local word guard still discarded
clips whose *phoneme labels* could not be assigned confidently. This is why
Whisper confirmation does not authorize replacing the phonemes wholesale.

## Training integration and limits

`PronunciationFilteredDataset` loaded all 18 accepted decisions from the final
JSONL, returned audio and phoneme tokens for a real clip, and reported 18
lengths for bucketed batching. The sample is too small to estimate a full
corpus retention rate or a pronunciation error rate. Neither the dataset nor
this test provides verified phone-level labels; listening or a phone-labeled
benchmark is needed to determine whether the selected variants sound right.
The CPU run does not measure preprocessing overhead on a Kaggle 2×T4 session.
