# AeroFlow project context

Updated September 30, 2026. This file describes the implemented system. For
commands and dataset preparation, use [KAGGLE_TRAINING_GUIDE.md](KAGGLE_TRAINING_GUIDE.md).
The older master plan and Track Alpha documents are historical design proposals;
their quality guarantees and speed targets are not established properties.

## Voices and data

The current work concerns two independent English voices: LJSpeech and HiFi
speaker 9017. Train each on its own complete corpus and use separate checkpoints.
LJSpeech uses `metadata.csv` plus `wavs/` or `audio/`. The full HiFi adapter uses
`MikhailT/hifi-tts`, clean/train, filtered to speaker 9017. The `hifi-tts-light`
repository and the five-recording local `/tmp/lj` copy are diagnostic samples.

Both loaders mix to mono, resample with an anti-aliasing filter, peak-normalize
to 0.95 and prepare 24 kHz audio. RAM use and on-disk cache capacity are separate
constraints: lazy decoding does not eliminate full-split download/cache costs.

## Model

- Text normalization and ARPAbet phonemization feed a four-block Conformer.
- Monotonic alignment search supplies training durations; a duration predictor
  supplies inference timing.
- An acoustic encoder compresses log STFT magnitude into 32 channels per frame.
- A conditional flow network learns acoustic latents from noise and expanded
  text conditioning. The default inference solver uses six Heun steps.
- A four-block ConvNeXt decoder predicts full-band magnitude and phase vectors.
- PyTorch inverse STFT produces audio using FFT 1024, hop 240 and a Hann window.

The inspected model has 9,104,420 trainable parameters before freezing any
components. Duration bounds allocate frames to tokens but do not guarantee
correct pronunciation. Unit phase vectors do not ensure consistency between
frames. The Hann/240 configuration does not have an exactly constant squared
window overlap; PyTorch normalizes the overlap envelope. Neither this transform
nor the neural loss guarantees artifact-free audio or globally convex training.
No fixed CPU real-time factor is guaranteed by the architecture.

## New-model training workflow

1. **Stage A — joint TTS training from random weights.** Run
   `scripts/train_kaggle.py` with an explicit dataset, a fresh output directory,
   `--no-auto-resume`, and no `--resume-path` or `--finetune`. This learns text,
   alignment, durations, the acoustic representation, flow and decoder using the
   existing regression objectives. Its “best” checkpoint is based on a single
   training batch; assess fixed listening samples as well.
2. **Stage B — adversarial acoustic decoder training.** Initialize
   `scripts/train_vocoder.py --checkpoint` from that voice's Stage A checkpoint.
   Only the decoder and training-only critics learn. The text network, acoustic
   encoder and flow remain frozen, keeping the latent representation stable.
   Mel reconstruction, adversarial and feature-matching losses replace exact
   phase regression in this stage. This trainer cannot bootstrap a complete TTS
   model from random weights.
3. **Evaluate and export.** Compare original/reconstructed recordings and fixed
   text samples. Use Stage B's `model_latest.pt` or a listening-selected export
   in the normal CLI. `model_best_mel.pt` denotes a metric ranking, not a listening
   judgment. Further legacy Stage A training can undo Stage B's decoder changes.

The documented Kaggle workflow uses one GPU per run. The vocoder trainer has no
DDP or session watchdog. The joint trainer supports a watchdog and includes a
DDP path, but local checks did not validate multi-GPU training. Session limits,
GPU availability and storage capacity should be checked in the actual notebook.

## Confirmed fixes and remaining validation

Padding previously altered the acoustic target for a recording because global
normalization included padded frames. The model now respects valid lengths in
STFT analysis, acoustic GroupNorm, decoder GRN, intermediate convolution states
and inverse STFT. Existing checkpoint parameter names and shapes are preserved.
Phase-vector normalization also avoids a singular derivative at zero.

The normalization fix reduces the measured HiFi padding-induced latent change
from approximately 0.1165 relative L2 to approximately 2e-7. Unit checks, real
CPU updates on both sample datasets, full-size critic updates, deterministic
resume and inference exports passed. Full-corpus GPU convergence and reduced
robotic timbre still require training and listening evaluation. The current
adversarial objective is a supported experiment, not a proven perceptual cure.

The initial investigation is in
[AUDIO_QUALITY_INVESTIGATION.md](AUDIO_QUALITY_INVESTIGATION.md); current acoustic
training details and local findings are in [VOCODER_REPAIR.md](VOCODER_REPAIR.md).
