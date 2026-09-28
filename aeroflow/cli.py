"""Friendly command-line interface for AeroFlow text-to-speech."""

from __future__ import annotations

import argparse
import math
import re
import sys
import tempfile
from pathlib import Path
from typing import Iterable

import numpy as np
import soundfile as sf
import torch

from aeroflow.dataset.audio import resample_mono
from aeroflow.models.pipeline import AeroFlowTTS


def chunk_text(text: str, phonemizer, max_tokens: int = 240) -> list[str]:
    """Pack whole sentences first; split oversized sentences only between words.

    A single long word is kept intact as its own chunk even when it exceeds the
    token budget. That budget is a target for manageable inference lengths, not
    permission to alter the user's text.
    """
    if max_tokens < 1:
        raise ValueError("max_tokens must be at least 1")
    text = text.strip()
    if not text:
        return []

    sentences = re.split(r"(?<=[.!?])\s+|\s*\n+\s*", text)
    chunks: list[str] = []
    current = ""

    def count(value: str) -> int:
        return len(phonemizer.text_to_sequence(value))

    def flush() -> None:
        nonlocal current
        if current:
            chunks.append(current)
            current = ""

    for sentence in sentences:
        if not sentence.strip():
            continue
        sentence = sentence.strip()
        candidate = f"{current} {sentence}".strip()
        if count(candidate) <= max_tokens:
            current = candidate
            continue

        # Prefer an intact sentence boundary whenever the sentence itself fits.
        if count(sentence) <= max_tokens:
            flush()
            current = sentence
            continue

        # Only oversized sentences need word-boundary subdivision.
        flush()
        words = sentence.split()
        for word in words:
            candidate = f"{current} {word}".strip()
            if count(candidate) <= max_tokens:
                current = candidate
                continue

            flush()
            # An unusually long word remains whole; max_tokens is a soft limit
            # for this case so a chunk boundary never changes a word.
            current = word
        flush()
    flush()
    return chunks


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    device = torch.device(requested)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("This pipeline currently supports CPU and CUDA devices")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but PyTorch cannot access a CUDA GPU")
    return device


def _default_checkpoint() -> Path:
    candidates = [
        Path.home() / "Downloads" / "checkpoint_best.pt",
        Path("checkpoints/checkpoint_best.pt"),
    ]
    return next((path for path in candidates if path.is_file()), candidates[0])


def _read_input(text: str | None, text_file: Path | None) -> str:
    if text_file is not None:
        try:
            return text_file.read_text(encoding="utf-8")
        except OSError as exc:
            raise ValueError(f"Cannot read text file {text_file}: {exc}") from exc
    if text is not None:
        return text
    if sys.stdin.isatty():
        raise ValueError("Provide TEXT, --text-file PATH, or pipe text through stdin")
    return sys.stdin.read()


def _assemble_audio(chunks: Iterable[torch.Tensor], sample_rate: int = 24000) -> torch.Tensor:
    parts: list[torch.Tensor] = []
    silence = torch.zeros(round(0.075 * sample_rate))
    fade_samples = round(0.005 * sample_rate)
    for chunk in chunks:
        # Copy out of inference mode before applying the edge fade in-place.
        audio = chunk.detach().float().cpu().flatten().clone()
        if audio.numel() == 0 or not torch.isfinite(audio).all():
            raise ValueError("Model returned empty or non-finite audio")
        if audio.numel() > 2 * fade_samples:
            fade = torch.linspace(0.0, 1.0, fade_samples)
            audio[:fade_samples] *= fade
            audio[-fade_samples:] *= fade.flip(0)
        parts.append(audio)
        parts.append(silence)
    if parts:
        parts.pop()
    return torch.cat(parts) if parts else torch.empty(0)


def _restore_with_voicefixer(
    audio: torch.Tensor,
    output_path: Path,
    *,
    cuda: bool,
    mode: int,
) -> torch.Tensor:
    """Run optional VoiceFixer cleanup on the complete, joined utterance."""
    try:
        from voicefixer import VoiceFixer
    except ImportError as exc:
        raise RuntimeError(
            "VoiceFixer was requested but is not installed. Install it with "
            "`python -m pip install voicefixer` and run the command again."
        ) from exc

    try:
        with tempfile.TemporaryDirectory(prefix="aeroflow-voicefixer-") as temp_dir:
            temp_dir = Path(temp_dir)
            input_path = temp_dir / "aeroflow_input.wav"
            restored_path = temp_dir / "voicefixer_output.wav"
            sf.write(input_path, audio.numpy(), 24000, subtype="PCM_16")
            print("Running VoiceFixer cleanup (first run may download its model weights)...", file=sys.stderr)
            restorer = VoiceFixer()
            restorer.restore(
                input=str(input_path),
                output=str(restored_path),
                cuda=cuda,
                mode=mode,
            )
            restored, sample_rate = sf.read(restored_path, dtype="float32")
            restored = resample_mono(restored, sample_rate, 24000)
            if restored.size == 0 or not np.isfinite(restored).all():
                raise RuntimeError("VoiceFixer returned empty or non-finite audio")
            print(f"VoiceFixer output prepared for {output_path}.", file=sys.stderr)
            return torch.from_numpy(restored)
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(f"VoiceFixer cleanup failed: {exc}") from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m aeroflow",
        description="Turn text into 24 kHz speech with an AeroFlow checkpoint.",
        epilog=(
            "Examples:\n"
            "  python -m aeroflow 'Hello there.' -o hello.wav\n"
            "  python -m aeroflow --text-file chapter.txt -o chapter.wav\n"
            "  cat chapter.txt | python -m aeroflow -o chapter.wav\n"
            "  python -m aeroflow 'Slow down.' --alpha 1.15 --chunk-tokens 140"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("text", nargs="?", help="Text to speak; defaults to standard input")
    parser.add_argument("-f", "--text-file", type=Path, help="Read UTF-8 text from a file")
    parser.add_argument("-o", "--output", type=Path, default=Path("speech.wav"), help="Output WAV path (default: speech.wav)")
    parser.add_argument("-c", "--checkpoint", type=Path, default=None, help="Checkpoint path (auto-finds ~/Downloads/checkpoint_best.pt or checkpoints/checkpoint_best.pt)")
    parser.add_argument("--device", default="auto", help="auto, cpu, or cuda (auto uses CUDA when available)")
    parser.add_argument("--alpha", type=float, default=1.0, help="Speech tempo: larger is slower (default: 1.0)")
    parser.add_argument("--chunk-tokens", type=int, default=240, help="Target phoneme-token limit per chunk (default: 240); complete sentences and words stay intact")
    parser.add_argument("--phase-iterations", type=int, default=0, help="Optional phase-refinement iterations (default: off)")
    parser.add_argument("--pronunciation-lexicon", type=Path, help="Audited speaker pronunciation overrides JSON")
    parser.add_argument("--voicefixer", action="store_true", help="Optional speech-restoration pass after all text chunks are joined (requires voicefixer package)")
    parser.add_argument("--voicefixer-mode", type=int, choices=(0, 1), default=0, help="VoiceFixer mode: 0 standard (default), 1 with high-frequency preprocessing")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed for repeatable synthesis")
    parser.add_argument("--threads", type=int, default=None, help="CPU threads; by default PyTorch chooses")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.text is not None and args.text_file is not None:
            raise ValueError("Use positional TEXT or --text-file, not both")
        if args.alpha <= 0 or not math.isfinite(args.alpha):
            raise ValueError("--alpha must be a finite positive number")
        if args.phase_iterations < 0:
            raise ValueError("--phase-iterations must be non-negative")
        if args.threads is not None:
            if args.threads < 1:
                raise ValueError("--threads must be at least 1")
            torch.set_num_threads(args.threads)
        text = _read_input(args.text, args.text_file)
        checkpoint_path = args.checkpoint or _default_checkpoint()
        if not checkpoint_path.is_file():
            raise ValueError(
                f"Checkpoint not found: {checkpoint_path}\n"
                "Pass its location with --checkpoint /path/to/checkpoint_best.pt"
            )
        device = _resolve_device(args.device)

        torch.manual_seed(args.seed)
        model = AeroFlowTTS().to(device).eval()
        if args.pronunciation_lexicon:
            model.phonemizer.load_lexicon_overrides(args.pronunciation_lexicon)
        try:
            # AeroFlow training checkpoints also contain NumPy/Python RNG state,
            # which the restricted weights-only loader cannot deserialize.
            # Only load a checkpoint path explicitly selected by the user or the
            # well-known local Downloads/checkpoints locations above.
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        except Exception as exc:
            raise ValueError(f"Could not load checkpoint {checkpoint_path}: {exc}") from exc
        state = checkpoint.get("model") if isinstance(checkpoint, dict) else None
        if not isinstance(state, dict):
            raise ValueError("This file is not an AeroFlow training checkpoint (missing 'model' weights)")
        try:
            model.load_state_dict(state, strict=True)
        except RuntimeError as exc:
            raise ValueError(f"Checkpoint architecture does not match this AeroFlow version: {exc}") from exc
        model.to(device)

        chunks = chunk_text(text, model.phonemizer, args.chunk_tokens)
        if not chunks:
            raise ValueError("Input text is empty")
        print(f"Loaded {checkpoint_path} on {device}; synthesizing {len(chunks)} chunk(s).", file=sys.stderr)
        generated: list[torch.Tensor] = []
        with torch.inference_mode():
            for index, chunk in enumerate(chunks, start=1):
                print(f"  [{index}/{len(chunks)}] {chunk[:72]}", file=sys.stderr)
                generated.append(model.synthesize(
                    chunk,
                    alpha=args.alpha,
                    phase_iterations=args.phase_iterations,
                ))
        audio = _assemble_audio(generated)
        if args.voicefixer:
            audio = _restore_with_voicefixer(
                audio,
                args.output,
                cuda=device.type == "cuda",
                mode=args.voicefixer_mode,
            )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        peak = float(audio.abs().max()) if audio.numel() else 0.0
        if peak > 1.0:
            audio = audio / peak * 0.999
        sf.write(args.output, audio.numpy(), 24000, subtype="PCM_16")
        print(f"Wrote {args.output} ({audio.numel() / 24000:.1f} s, 24 kHz mono).", file=sys.stderr)
        return 0
    except (ValueError, OSError, RuntimeError) as exc:
        parser.error(str(exc))
    return 2
