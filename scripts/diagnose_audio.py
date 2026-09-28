"""Generate reproducible checkpoint/phase comparisons without changing weights.

python scripts/diagnose_audio.py --checkpoint /path/checkpoint_best.pt
Add --reference speech.wav to test the acoustic encoder/decoder without text.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aeroflow.dataset.audio import resample_mono
from aeroflow.models.alignment import expand_text_representations
from aeroflow.models.flow_matching import NonUniformHeunSolver
from aeroflow.models.pipeline import AeroFlowTTS


def rms(audio):
    return audio.square().mean().sqrt().clamp_min(1e-12)


@torch.inference_mode()
def compare(model, spectrum, length, name, output, iterations, reference=None):
    baseline = model.istft(spectrum, length=length)
    records = []
    for count in iterations:
        audio = model.istft.refine_phase(spectrum, count, length=length)
        projected = model.stft_analysis(audio)[0]
        record = {
            "name": name, "phase_iterations": count,
            "seconds": audio.shape[-1] / 24000,
            "raw_peak": audio.abs().max().item(),
            "raw_rms": rms(audio).item(),
            "decoder_magnitude_residual": (
                (projected.abs() - spectrum.abs()).norm() / spectrum.abs().norm().clamp_min(1e-8)
            ).item(),
        }
        if reference is not None:
            target = model.stft_analysis(reference)[1]
            record["reference_magnitude_error"] = (
                (projected.abs() - target).norm() / target.norm().clamp_min(1e-8)
            ).item()
            record["reference_log_magnitude_error"] = (
                projected.abs().clamp_min(1e-5).log() - target.clamp_min(1e-5).log()
            ).abs().mean().item()
        # Store raw float audio for analysis, and RMS-matched PCM for listening.
        # The baseline peak is brought to 0.8; variants share its target RMS.
        raw_path = output / f"{name}_phase_{count}_raw.wav"
        sf.write(raw_path, audio[0].cpu().numpy(), 24000, subtype="FLOAT")
        matched = audio * rms(baseline) / rms(audio)
        records.append((record, matched, raw_path))
    peak = max(float(item[1].abs().max()) for item in records)
    gain = 0.8 / max(peak, 1e-8)
    result = []
    for record, matched, raw_path in records:
        path = output / f"{name}_phase_{record['phase_iterations']}.wav"
        sf.write(path, (matched * gain)[0].cpu().numpy(), 24000, subtype="PCM_16")
        record.update(file=path.name, raw_file=raw_path.name)
        result.append(record)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/tts_diagnosis/comparison"))
    parser.add_argument("--text", action="append", help="Repeat for multiple prompts")
    parser.add_argument("--reference", type=Path, action="append", default=[])
    parser.add_argument("--steps", type=int, nargs="+", default=[6, 24])
    parser.add_argument("--phase-iterations", type=int, nargs="+", default=[0, 4, 16])
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if any(n < 1 for n in args.steps) or any(n < 0 for n in args.phase_iterations):
        parser.error("steps must be positive and phase iterations non-negative")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    model = AeroFlowTTS().eval()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    prompts = args.text or [
        "The quick brown fox jumps over the lazy dog.",
        "Peter Piper picked a peck of pickled peppers.",
        "The sound of the wind was soft and low.",
    ]
    report = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "global_step": checkpoint.get("global_step"), "seed": args.seed,
        "prompts": prompts, "references": [str(p.resolve()) for p in args.reference],
        "torch_version": torch.__version__, "results": [],
        "note": "Metrics are not perceptual quality scores. Listening WAVs are RMS-matched within each comparison; raw FLOAT WAVs retain original gain.",
    }
    with torch.inference_mode():
        for i, text in enumerate(prompts):
            tokens = torch.tensor([model.phonemizer.text_to_sequence(text)])
            hidden, _ = model.encoder(tokens)
            durations = model.duration_predictor.predict_durations(hidden)
            condition = expand_text_representations(hidden, durations)
            generator = torch.Generator().manual_seed(args.seed + i)
            noise = torch.randn(1, model.latent_dim, condition.shape[-1], generator=generator)
            for steps in args.steps:
                latent = NonUniformHeunSolver(steps).solve(model.vector_field, noise.clone(), condition)
                spectrum = model.decoder(latent)[0]
                records = compare(model, spectrum, None, f"text_{i}_steps_{steps}", args.output, args.phase_iterations)
                report["results"].extend(records)
                print(f"Generated prompt {i}, {steps} flow steps", flush=True)
        for i, path in enumerate(args.reference):
            array, rate = sf.read(path, dtype="float32")
            audio = torch.from_numpy(resample_mono(array, rate)).unsqueeze(0)
            if audio.shape[-1] <= model.n_fft // 2:
                raise ValueError(f"Reference too short for STFT: {path}")
            audio = audio * (0.95 / audio.abs().max().clamp_min(1e-4))
            sf.write(args.output / f"reference_{i}.wav", audio[0].numpy(), 24000, subtype="FLOAT")
            spectrum = model.decoder(model.acoustic_encoder(model.stft_analysis(audio)[1]))[0]
            report["results"].extend(compare(
                model, spectrum, audio.shape[-1], f"reconstruction_{i}",
                args.output, args.phase_iterations, reference=audio,
            ))
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote listening comparisons and report to {args.output}")


if __name__ == "__main__":
    main()
