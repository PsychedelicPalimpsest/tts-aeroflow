# Acoustic decoder repair

For **new LJSpeech and HiFi 9017 models from scratch**, follow
[KAGGLE_TRAINING_GUIDE.md](KAGGLE_TRAINING_GUIDE.md): train the complete TTS model
with `train_kaggle.py --no-auto-resume`, then use its checkpoint to initialize
the decoder stage described here. Existing pretrained voices are optional; the
decoder trainer itself always needs an already learned encoder and flow.

The confirmed padding bug is fixed. The new training path addresses the suspected
perceptual-loss problem; a trained cure for robotic timbre has **not** yet been
demonstrated. Existing checkpoints still load strictly and the repair exports
work with the ordinary synthesis CLI.

## Changes

- Each recording's STFT reflects its own endpoint before spectra are padded.
- Acoustic GroupNorm excludes padded frames and masks convolution inputs and
  outputs. Decoder GRN and convolution states also exclude padding.
- Batched inverse STFT reconstructs each recording with its own frame count and
  length, avoiding overlap-normalization changes at padded endpoints.
- Phase-vector normalization uses FP32 arithmetic and a finite derivative at
  zero. Parameter names and tensor sizes are unchanged.
- `scripts/train_vocoder.py` trains only the decoder. It freezes the acoustic
  encoder, text encoder, duration predictor, and flow network, preserving the
  latent representation learned by existing checkpoints.
- Its objective is `45 * log_mel_L1 + ramp * (adversarial + 2 * feature_matching)`.
  Five period critics (2, 3, 5, 7, 11) and three complex-spectrum critics
  (FFT 512, 1024, 2048) learn natural waveform structure. The critic uses hinge
  loss; the generator uses the non-saturating negative critic score. Feature
  matching averages over layers and sums over critics. Exact complex-spectrum
  regression and the legacy instantaneous-frequency loss are absent.

This is a local implementation of the established multi-period/multi-resolution
vocoder approach, not an exact reproduction of Vocos. See
[HiFi-GAN](https://arxiv.org/abs/2010.05646) and
[Vocos](https://arxiv.org/html/2306.00814v3) for the underlying approaches.

Training reconstructs complete recordings with valid lengths, then crops aligned
real/generated segments for the discriminator and mel losses. Crops cannot
include batch padding. This retains the encoder's full-utterance normalization
and avoids training on a different latent representation produced by isolated
crops. Default discriminator crop length is 16,320 samples (0.68 s at 24 kHz).
The default duration filter caps recordings at 12 seconds.

Critics are only used in training; inference cost is unchanged. Changing the
decoder still requires checking whether improvements transfer to flow-generated
latents. Fixed text samples are saved alongside recording reconstructions.

## Data available in this workspace

As inspected September 30, 2026:

- `/tmp/lj` contains five WAVs, although metadata.csv lists all 13,100 LJ entries.
  The adapter uses the five recordings actually present.
- `MikhailT/hifi-tts-light`, clean/train, contains nine rows across three
  speakers, including three for **9017**. Only two pass the diagnostic duration
  filter. This repository is a tiny sample, not the user's approximately 20 GB
  training corpus. The full corpus must be supplied for substantive training.
- CUDA is unavailable on this machine.

These inputs are suitable for implementation checks. Do not interpret thousands
of repeated updates on them as a production voice repair.

## GPU training with complete data

Start with the Stage A checkpoint (or an existing trained checkpoint) for the
matching speaker. Use a separate
output directory for each voice. For LJ, populate a complete dataset directory:

```bash
python scripts/train_vocoder.py \
  --dataset lj --data-root /path/to/complete/LJSpeech-1.1 \
  --checkpoint checkpoints/lj_joint/checkpoint_latest.pt \
  --output checkpoints/lj_vocoder --device cuda \
  --steps 100000 --batch-size 8
```

For HiFi, use the full `MikhailT/hifi-tts` repository with the existing adapter,
preserving speaker 9017. The supplied light repository can be used for
short diagnostics via `--hf-repo MikhailT/hifi-tts-light`; it is not sufficient
for this training command:

```bash
python scripts/train_vocoder.py \
  --dataset hifi --hf-repo MikhailT/hifi-tts --speaker 9017 \
  --checkpoint checkpoints/hifi9017_joint/checkpoint_latest.pt \
  --output checkpoints/hifi_vocoder --device cuda \
  --steps 100000 --batch-size 8
```

100,000 steps is a starting budget, not a demonstrated convergence requirement.
Inspect validation audio early and regularly. Reduce batch size if GPU memory is
limited. The critics train immediately; their contribution to decoder updates
ramps over the first 1,000 steps. The pretrained decoder uses LR 1e-4 and critics
use 2e-4, with constant rates so a resumed run does not accidentally cycle an
expired cosine schedule. Both can be set explicitly at the start of a run.

Resume with the same dataset, split, seed, optimizer settings and architecture:

```bash
python scripts/train_vocoder.py \
  --dataset lj --data-root /path/to/complete/LJSpeech-1.1 \
  --resume checkpoints/lj_vocoder/checkpoint_latest.pt \
  --output checkpoints/lj_vocoder --device cuda \
  --steps 100000 --batch-size 8
```

`--steps` is the total number of updates, including previous sessions. For a run
started with non-default settings, repeat those settings when resuming. Sampling
and crop positions are derived from seed and step, so worker prefetch cannot
shift the sequence. Dataset filename order is fingerprinted to reject a changed
split on resume. Checkpoints atomically save both optimizers, critics, AMP
scalers, model weights, RNG, step, configuration and original checkpoint/code
provenance. Saves occur every 500 steps by default.

## Evaluation and exports

The trainer reserves a deterministic set of recordings from repair updates and
writes `split.json`. These clips may have appeared in the original checkpoint's
training; they are held out from the **repair** run. For small subsets, at least
one clip is held out. The baseline is saved before any update. Every 1,000 steps,
it evaluates complete validation recordings and writes:

- Reference/reconstruction audio, in raw FLOAT and RMS-matched PCM pairs.
- A fixed-seed text synthesis sample (`text.wav` / `text_raw.wav`).
- Log-mel and relative magnitude errors, peak values, filenames and transcripts.
- `model_latest.pt`: latest evaluated model, directly usable for inference.
- `model_best_mel.pt`: best validation log-mel checkpoint, including the untouched
  baseline if none of the updates improves that metric. This name deliberately
  does not imply best perceived quality.

Use `checkpoint_latest.pt` to resume this trainer. Its step counts decoder
updates, not the original text/flow training steps. Inference exports contain no
optimizer; do not feed them to the legacy joint trainer's resume command. Further
joint training with the old losses could undo the decoder repair.

```bash
python -m aeroflow "The sound of the wind was soft and low." \
  --checkpoint checkpoints/lj_vocoder/model_latest.pt -o repaired.wav
```

Listen to the fixed original/reconstruction pairs first, then compare fixed text
samples. Improved reconstruction metrics alone do not establish reduced robotic
timbre. If direct reconstruction remains poor after a controlled repair run,
the next experiment is encoder/bottleneck capacity or a known pretrained
vocoder with its matching frontend; changing the encoder would also require
adapting the flow.

## Local validation

The automated checks cover padding independence (including nonzero GRN
parameters), polarity-invariant mel reconstruction, finite zero-phase gradients,
gradient isolation between critics/generator/frozen modules, valid audio crops,
disjoint splits, and deterministic resumed batch sampling. Existing model,
dataset, CLI, resumption and loss tests are also run.

Real CPU diagnostic runs completed on both supplied sources with small critics,
and a separate LJ run completed with the default full-size critics, full crop
length and a batch of two unequal-length recordings. Resume and exported-model
compatibility were checked. These runs establish executable training and
checkpoint behavior, not perceptual improvement. The short HiFi run's validation
mel loss worsened (0.340 to 0.364), which is why the unchanged baseline remains its
best-mel export.

Using original checkpoints, padding a real recording to twice its length now
changes acoustic latents by only approximately 2e-7 relative L2 (numerical
roundoff). Before the fix, the HiFi probe changed by 0.1165; its reconstructed
waveform now differs by only 5.9e-6. Measurements are saved under
`artifacts/timbre_investigation_20260930/padding_after_fix.json`.
