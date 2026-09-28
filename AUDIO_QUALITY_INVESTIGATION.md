# Checkpoint audio investigation — 2026-09-28

Checkpoint: `/home/mitch/Downloads/checkpoint_best.pt`, loaded strictly with all
257 state entries matching. Training metadata: step 8,000, epoch 142, best loss
1.911767. SHA-256:
`75b7aaebaaa7b2e73a40c0509069e4eeb677573d85b31c624d7a8304f8ab789c`.
The checkpoint is unchanged. No replacement weights have been trained.

## How this project works

Text normalization and the built-in ARPAbet lexicon/rule fallback feed a Conformer.
MAS supplies training durations; a duration predictor supplies inference timing.
An acoustic encoder compresses **log magnitude only** into 32 channels. A
conditional flow model learns these latents, and a ConvNeXt decoder predicts
magnitude plus phase. PyTorch iSTFT combines the frames into 24 kHz audio.
The decoder trains on reference latents; inference uses flow-generated latents.
This gives three separate things to investigate: text/timing, latent generation,
and acoustic reconstruction.

The design documents' claims of guaranteed absence of artifacts are not established
by the code. Unit-normalized phase does not enforce coherent adjacent frames.
Hann windows with hop 240 and FFT 1024 do not have an exactly constant squared
overlap sum, but PyTorch iSTFT normalizes that envelope; this alone is not a bug.
The neural training objective is also not convex in the model parameters.

## Confirmed defects and fixes

1. **Training resampling had no anti-aliasing filter.** Both dataset backends used
   linear interpolation for 44.1→24 kHz conversion. They now share polyphase FIR
   resampling. A unit-amplitude 16 kHz test tone previously left −6.48 dBFS RMS
   of aliased energy; the replacement leaves −63.37 dBFS over the interior,
   approximately 57 dB less. A separate test verifies 1 kHz amplitude retention.
   This proves a preprocessing fix, not that aliasing explains all of this
   checkpoint's sound. Existing weights require continued training to benefit.
   SciPy is now an explicit dependency in the Kaggle setup instructions.
2. **Phase-loss weighting depended on predicted magnitude.** This allowed gradients
   to alter energy allocation instead of only correcting phase. Weights now use
   reference energy from both adjacent frames. Silent references have zero weight.
3. **Phase-loss normalization counted padding twice.** Normalization now divides
   the weighted error by reference weight once per utterance, making it invariant
   to extra zero padding when STFT boundary conditions are unchanged.

## Checkpoint experiments

All inference uses CPU FP32, evaluation mode and fixed seeds. The checkpoint lacks
dataset/speaker metadata; speaker 9017 is the project's documented default, not
verified training provenance. Six reference clips came from the clean dev/test
splits of `MikhailT/hifi-tts-light`; they are diagnostic examples, not a quality
benchmark, and possible training overlap is unknown.

On an initial fixed-noise “quick brown fox” synthesis, normalized complex STFT
inconsistency was 0.5147 with 6 flow steps versus 0.5137 with 24. The output
was 2.57 s, peak 0.9255 at 6 steps, so clipping did not explain this sample.
Inconsistency is `||S - STFT(iSTFT(S))|| / ||S||`; it identifies reconstruction
disagreement, not a perceptual score or proof of the audible cause.

Optional Griffin–Lim iterations initialized with learned phase preserve predicted
magnitude. Reference reconstruction relative magnitude errors were:

| Reference | Original | 4 iterations | 16 iterations |
|---|---:|---:|---:|
| Light! | 0.782 | 0.718 | 0.704 |
| Hey! | 0.515 | 0.430 | 0.417 |
| yes; | 0.424 | 0.448 | 0.450 |
| Whilst, however, the horses… (12.14 s) | 0.390 | 0.337 | 0.326 |
| What answer could be made to this? | 0.490 | 0.408 | 0.406 |
| Marie? | 0.484 | 0.333 | 0.315 |

Five of six improve by this metric; one worsens. Log-magnitude error also worsens
on some examples. The option remains **off by default**, with unchanged checkpoint
compatibility. Listening is required to decide whether it reduces the reported
buzz/metallic quality. No auditory judgment or perceptual-quality improvement is
claimed from these numerical measurements.

## Reproduce and listen

```bash
python scripts/diagnose_audio.py \
  --checkpoint /home/mitch/Downloads/checkpoint_best.pt
```

Use repeated `--reference /path/to/real_speech.wav` for acoustic reconstruction,
or repeated `--text 'Sentence.'` for custom prompts. No network is needed by this
script. It strictly loads the original checkpoint and writes to
`artifacts/tts_diagnosis/comparison/`:

- Three prompts × two flow-step counts × three phase settings by default.
- PCM listening files matched in RMS within each phase comparison, with a shared
  gain to avoid clipping. Cross-prompt or cross-step comparisons are not matched.
- `_raw.wav` float files preserving actual model levels, and `report.json` with
  checkpoint hash, prompts, seed and metrics calculated before gain adjustment.

Start with `text_0_steps_6_phase_0.wav` and `text_0_steps_6_phase_4.wav`.
Inference API: `model.eval(); model.synthesize(text, phase_iterations=4)`.
Omitting the option preserves the original synthesis.

## Validation and next training decision

62 tests passed (three network integration tests deselected). Coverage includes
alias suppression, passband preservation, silent/padded phase loss, valid-spectrum
round trips, phase-refinement residual reduction, full training gradients, and
existing checkpoint resume behavior. An additional forward/backward pass with the
actual checkpoint and real speech checks the changed objective before training.

The next substantive model experiment is continued training on the confirmed
original speaker/corpus using corrected preprocessing and loss, with a fixed
held-out listening set. The current trainer selects “best” using a single training
batch's total loss; that is not a reliable perceptual ranking. Since the loss
definition changed, historical best-loss values are not directly comparable.
Use a separate output directory and reset best-loss bookkeeping (the trainer's
`--finetune` mode does this) for that experiment. No GPU was available locally,
and the corpus provenance was not supplied, so a production fine-tune was not run.

If reconstruction itself remains metallic after this controlled fine-tune, phase
modeling/decoder training deserves priority over increasing flow steps. Replacing
the decoder or adding adversarial losses would be a separate experiment requiring
training and listening validation, not an immediate checkpoint repair.

References: [SciPy polyphase resampling](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.resample_poly.html),
[PyTorch Griffin–Lim](https://docs.pytorch.org/audio/main/generated/torchaudio.transforms.GriffinLim.html),
[Vocos paper](https://arxiv.org/abs/2306.00814).
