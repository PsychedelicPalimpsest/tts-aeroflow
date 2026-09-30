# AeroFlow TTS

## Train a new voice

See [the Kaggle training guide](KAGGLE_TRAINING_GUIDE.md) for complete LJSpeech and
HiFi speaker 9017 workflows. Each voice starts with joint TTS training from random
weights (`train_kaggle.py --no-auto-resume`), followed by adversarial decoder
training (`train_vocoder.py --checkpoint <stage-A-checkpoint>`). The second stage
requires a trained acoustic encoder and flow; it cannot train a complete model
from scratch by itself.

Use complete datasets and separate output directories for each voice and stage.
`MikhailT/hifi-tts-light` is a diagnostic sample. Full-corpus GPU training and
listening evaluation are still needed to establish the improvement in timbre.

## Generate speech

Generate 24 kHz mono speech from text with the trained AeroFlow checkpoint.
From the project directory, run:

```bash
python -m aeroflow "Hello there. This text is spoken aloud." -o hello.wav
```

The CLI automatically uses `~/Downloads/checkpoint_best.pt` when it exists,
otherwise it checks `checkpoints/checkpoint_best.pt`. Choose another checkpoint
with `--checkpoint /path/to/checkpoint_best.pt`.

For a long passage, the command packs complete sentences together into
manageable phoneme-count chunks, synthesizes each one, and joins them with short
silences and click-smoothing fades. Oversized sentences split only between
whitespace-separated words. A single unusually long word stays intact even if it
exceeds the target. The default target is 240 phoneme tokens per chunk; adjust it
with `--chunk-tokens`.

```bash
python -m aeroflow --text-file chapter.txt -o chapter.wav
cat chapter.txt | python -m aeroflow -o chapter.wav
python -m aeroflow "A little slower." --alpha 1.15
python -m aeroflow --help
```

VoiceFixer speech cleanup is optional and runs after all text chunks have been
joined. Install the package when you want to use it:

```bash
python -m pip install git+https://github.com/haoheliu/voicefixer.git
python -m aeroflow --text-file chapter.txt -o chapter_clean.wav --voicefixer
```

The first VoiceFixer run may download its pretrained weights. Mode 0 is the
default; `--voicefixer-mode 1` adds high-frequency preprocessing. This stage can
change the voice character, so compare the cleaned WAV with the default output.
VoiceFixer remains optional and is not imported unless `--voicefixer` is used.

Input may be supplied as positional text, with `--text-file`, or through standard
input. Output defaults to `speech.wav`. The CLI supports `--device auto`, `cpu`,
or `cuda`; `auto` selects CUDA when available, then falls back to CPU.

For details on the checkpoint investigation and optional phase refinement, see
[AUDIO_QUALITY_INVESTIGATION.md](AUDIO_QUALITY_INVESTIGATION.md).

For the padding fix and adversarial acoustic decoder training, see
[VOCODER_REPAIR.md](VOCODER_REPAIR.md). The repair trainer supports local LJSpeech
and the HiFi Hugging Face adapter, preserves the existing text/flow weights, and
exports checkpoints for this CLI. Full-corpus retraining and listening evaluation
are required to determine the improvement in timbre.
