# AeroFlow TTS

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
input. Output defaults to `speech.wav`. The CLI supports `auto`, `cpu`, `cuda`, and
device; `auto` selects CUDA when available, then falls back to CPU.

For details on the checkpoint investigation and optional phase refinement, see
[AUDIO_QUALITY_INVESTIGATION.md](AUDIO_QUALITY_INVESTIGATION.md).
