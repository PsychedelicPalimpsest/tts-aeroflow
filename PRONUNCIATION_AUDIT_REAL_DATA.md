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
| “Whilst, however, …” | 12.14 s | Rejected | Best phone error ratio 0.1844 |
| “What answer could be made to this?” | 1.98 s | Corrected, accepted | “answer” changed to CMUdict pronunciation; ratio 0.15 |
| “Marie?” | 0.70 s | Corrected, accepted | Recognizer heard `M EH R IY`; selected `M EH0 R IY1` |
| “There is Monsieur returning from hunting.” | 3.00 s | Rejected | “Monsieur” had a local phone mismatch |
| “And that was all.” | 1.14 s | Rejected | Best phone error ratio 0.3636 |
| “On seeing this, …” | 13.14 s | Corrected, accepted | 13 word variants selected; ratio 0.1484 |

The initial audit accepted 5 of 9. It rejected “Marie?” because CMUdict's
`ER0` did not match the recognizer's separate `EH R`. Adding that representation
as a candidate raised acceptance to **6 of 9** without changing the threshold.
Two clips exceed the Kaggle Hi-Fi TTS trainer's default 10-second limit. Among
the seven clips eligible under that default, the final audit accepts **5** and
rejects **2**. The audit's Hi-Fi TTS duration default now matches the trainer.

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
