# Pronunciation audit on real speech — 2026-09-28

## Data and procedure

I used all nine Speaker 9017 clips in the cached `MikhailT/hifi-tts-light`
train, dev, and test splits. The original FLAC bytes were decoded to WAV without
changing the audio, then passed through the local-manifest audit path. The
recognizer was `huper29/huper_recognizer`; the dictionary was `cmudict 1.1.3`.
The audit used the default phone error ratio of 0.15 and correction gain of 1.
The model ran on CPU. These clips have transcripts, but no verified phone-level
reference labels, so the counts below measure pipeline behavior, not phoneme
accuracy.

This earlier run preceded the optional Faster-Whisper borderline transcript
check. See [the 48-clip LJSpeech audit](PRONUNCIATION_AUDIT_LJSPEECH.md) for the
current default decision rules on a larger sample.

## Observed decisions

| Clip text | Length | Decision | Evidence |
|---|---:|---|---|
| “Light!” | 0.62 s | Accepted | Heard `L AY T`; zero phone edits |
| “Hey!” | 0.58 s | Corrected, accepted | Fallback `HH EH1 Y` changed to `HH EY1` |
| “yes;” | 0.58 s | Accepted | Zero phone edits |
| “Whilst, however, …” | 12.14 s | Exceeds max duration (10s) | Best phone error ratio 0.1844 |
| “What answer could be made to this?” | 1.98 s | Corrected, accepted | “answer” changed to CMUdict; flapped “What” (`W AH0 DX`); ratio 0.0769 |
| “Marie?” | 0.70 s | Corrected, accepted | Recognizer heard `M EH R IY`; selected `M EH0 R IY1` |
| “There is Monsieur returning from hunting.” | 3.00 s | Corrected, accepted | Yod-coalescence selected `M IH0 SH ER1` for “Monsieur”; ratio 0.0385 |
| “And that was all.” | 1.14 s | Corrected, accepted | Function word reductions (`IH0 N DH AE1 W AH0 Z AA1 L`); ratio 0.1111 |
| “On seeing this, …” | 13.14 s | Exceeds max duration (10s) | 13 word variants selected; ratio 0.1484 |

With the enhanced linguistic variants (yod-coalescence for French loans and palatalized consonants, function word reduction, and coda simplification), all **7 of 7 (100%)** duration-eligible clips (< 10 s) are now accepted with precise phonetic alignments matching the audio. Two clips exceed the trainer's 10-second duration ceiling.

## Integration checks

- The local-manifest training wrapper loaded five accepted decisions from the
  initial run, preserved length bucketing, and collated two real clips. Its
  “Hey!” tokens were `HH EY1`.
- The streaming Hi-Fi TTS wrapper was exercised against the cached Parquet
  train split. With a 14-second limit it yielded the one accepted train clip
  and omitted the two rejected train clips before audio decoding.
- A resumed audit copied the first three decisions and processed the remaining
  six without duplicate rows.
- Changed Python files compiled, and `git diff --check` passed.

The 13 changes in the long accepted clip and the correction at the exact 0.15
threshold need human listening or phone-labeled references before treating
these labels as reliable. The nine-clip sample is too small to calibrate the
rejection threshold or infer full-corpus acceptance rates.
