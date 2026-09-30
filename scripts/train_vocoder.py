"""Repair an AeroFlow decoder on recordings, preserving its text/flow/encoder weights.

Example:
  python scripts/train_vocoder.py --dataset lj --data-root /tmp/lj \
      --checkpoint ~/Downloads/checkpoint_lj.pt --output checkpoints/lj_vocoder

Training-only critics add no inference cost. This intentionally does not use
the legacy complex-spectrum or instantaneous-frequency regression objectives.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import soundfile as sf
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aeroflow.dataset.dataset import collate_hifi_tts
from aeroflow.dataset.ljspeech import LJSpeechDataset
from aeroflow.dataset.hf_hifi_tts import HuggingFaceHiFiTTSDataset
from aeroflow.models.pipeline import AeroFlowTTS
from aeroflow.models.discriminators import VocoderDiscriminators
from aeroflow.losses.vocoder import MelReconstructionLoss, discriminator_loss, generator_losses


def audio_identities(dataset):
    """Read stable file IDs in one metadata pass, without audio decoding."""
    if isinstance(dataset, LJSpeechDataset):
        return [str(item["audio_path"]) for item in dataset.items]
    files = dataset._hf.select_columns(["file"])["file"]
    return [str(files[index]) for index in dataset.hf_index]


def split_indices(count, validation_items, seed):
    if count < 2:
        raise ValueError("At least two recordings are required for a disjoint validation split")
    order = torch.randperm(count, generator=torch.Generator().manual_seed(seed)).tolist()
    heldout = min(validation_items, max(1, count // 10))
    return order[heldout:], order[:heldout]


class StepBatches:
    """Reproducible sampling with replacement, independent of worker prefetch."""
    def __init__(self, indices, batch_size, first_step, last_step, seed):
        self.indices, self.batch_size = indices, batch_size
        self.first_step, self.last_step, self.seed = first_step, last_step, seed

    def __iter__(self):
        for step in range(self.first_step, self.last_step):
            rng = torch.Generator().manual_seed(self.seed + step)
            draws = torch.randint(len(self.indices), (self.batch_size,), generator=rng)
            yield [self.indices[i] for i in draws.tolist()]

    def __len__(self):
        return self.last_step - self.first_step


def paired_crops(real, generated, lengths, samples, seed):
    """Aligned crops entirely inside real speech lengths; never train on batch padding."""
    rng = torch.Generator().manual_seed(seed)
    real_crops, generated_crops = [], []
    for i, length in enumerate(lengths.detach().cpu().tolist()):
        if length < samples:
            raise ValueError("Recording shorter than crop; raise dataset min_duration_s")
        start = int(torch.randint(length - samples + 1, (), generator=rng))
        real_crops.append(real[i, start:start + samples])
        generated_crops.append(generated[i, start:start + samples])
    return torch.stack(real_crops), torch.stack(generated_crops)


def atomic_save(value, path):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


@torch.inference_mode()
def validate(model, dataset, indices, loss_fn, device, folder, identities=None):
    """Full-utterance metrics and raw/RMS-matched listening pairs; no random crops."""
    folder.mkdir(parents=True, exist_ok=True)
    model.eval()
    identities = audio_identities(dataset) if identities is None else identities
    records = []
    for number, index in enumerate(indices):
        item = dataset[index]
        real = item["audio"].to(device)[None]
        predicted = model.reconstruct(real)
        magnitude = model.stft_analysis(real)[1]
        estimated = model.stft_analysis(predicted)[1]
        record = {
            "index": index, "file": identities[index], "text": item["text"],
            "mel_l1": float(loss_fn(real, predicted)),
            "relative_magnitude_error": float((magnitude - estimated).norm() / magnitude.norm().clamp_min(1e-8)),
            "peak": float(predicted.abs().max()),
        }
        records.append(record)
        signals = {"reference": real[0].cpu(), "reconstruction": predicted[0].cpu()}
        normalized = {key: y / y.square().mean().sqrt().clamp_min(1e-8) for key, y in signals.items()}
        gain = .9 / max(float(y.abs().max().clamp_min(1e-8)) for y in normalized.values())
        for key, y in signals.items():
            sf.write(folder / f"{number}_{key}_raw.wav", y.numpy(), 24000, subtype="FLOAT")
            sf.write(folder / f"{number}_{key}.wav", (normalized[key] * gain).numpy(), 24000, subtype="PCM_16")
    result = {"mel_l1": sum(r["mel_l1"] for r in records) / len(records), "items": records,
              "note": "Validation magnitude losses are not perceptual scores. Listen to fixed references."}
    # Also monitor transfer to the existing flow without changing training RNG.
    devices = [device.index or 0] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(1234)
        prompt = "The quick brown fox jumps over the lazy dog."
        speech = model.synthesize(prompt).cpu()
        sf.write(folder / "text_raw.wav", speech.numpy(), 24000, subtype="FLOAT")
        speech = speech * (.9 / speech.abs().max().clamp_min(1e-8))
        sf.write(folder / "text.wav", speech.numpy(), 24000, subtype="PCM_16")
        result["text_prompt"] = prompt
    (folder / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", choices=("lj", "hifi"), required=True)
    p.add_argument("--data-root", default="/tmp/lj")
    p.add_argument("--hf-repo", default="MikhailT/hifi-tts-light")
    p.add_argument("--hf-split", default="train")
    p.add_argument("--speaker", default="9017")
    p.add_argument("--cache-dir")
    p.add_argument("--checkpoint", type=Path, help="Original AeroFlow checkpoint; required for a new run")
    p.add_argument("--resume", type=Path, help="Resume this trainer's checkpoint_latest.pt")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--steps", type=int, default=100000, help="Total update count, including resumed steps")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--crop-samples", type=int, default=16320)
    p.add_argument("--lr", type=float, default=0.0001)
    p.add_argument("--disc-lr", type=float, default=0.0002)
    p.add_argument("--disc-channels", type=int, default=32)
    p.add_argument("--mel-weight", type=float, default=45)
    p.add_argument("--feature-weight", type=float, default=2)
    p.add_argument("--adversarial-weight", type=float, default=1)
    p.add_argument("--adversarial-warmup", type=int, default=1000,
                   help="Train critics immediately; ramp their generator contribution over these steps")
    p.add_argument("--validation-items", type=int, default=16)
    p.add_argument("--validate-every", type=int, default=1000)
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--threads", type=int, default=4)
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    if bool(args.checkpoint) == bool(args.resume):
        p.error("Supply exactly one of --checkpoint or --resume")
    for key in ("steps", "batch_size", "disc_channels", "validation_items", "validate_every", "save_every", "threads"):
        if getattr(args, key) < 1:
            p.error(f"--{key.replace('_', '-')} must be positive")
    if args.crop_samples < 2048 or args.workers < 0 or args.adversarial_warmup < 0:
        p.error("crop-samples must be >= 2048; workers and adversarial-warmup must be nonnegative")
    for key in ("lr", "disc_lr", "mel_weight", "feature_weight", "adversarial_weight"):
        if not math.isfinite(getattr(args, key)) or getattr(args, key) <= 0:
            p.error(f"--{key.replace('_', '-')} must be positive and finite")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        p.error("CUDA is unavailable")
    checkpoint = torch.load(args.resume or args.checkpoint, map_location="cpu", weights_only=False)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    if args.resume:
        if checkpoint.get("trainer") != "aeroflow_vocoder_v1":
            p.error("--resume requires a vocoder training checkpoint, not an original TTS checkpoint")
        # Steps may be extended; optimizer LR is fixed, so no expired cosine cycle.
        mutable = {"steps", "resume", "checkpoint", "output", "device", "threads", "workers", "save_every", "validate_every", "cache_dir"}
        differences = [k for k, value in checkpoint["config"].items() if k not in mutable and config[k] != value]
        if differences:
            p.error("Resume configuration differs: " + ", ".join(differences))
    elif (args.output / "checkpoint_latest.pt").exists():
        p.error("Output already has a run; use --resume or another output directory")
    args.output.mkdir(parents=True, exist_ok=True)
    minimum = max(0.5, (args.crop_samples + 240) / 24000)
    if args.dataset == "lj":
        dataset = LJSpeechDataset(root=args.data_root, min_duration_s=minimum)
    else:
        dataset = HuggingFaceHiFiTTSDataset(repo_id=args.hf_repo, split=args.hf_split,
                    speaker_ids=(args.speaker,), min_duration_s=minimum, cache_dir=args.cache_dir)
    train_indices, validation_indices = split_indices(len(dataset), args.validation_items, args.seed)
    identities = audio_identities(dataset)
    fingerprint = hashlib.sha256("\n".join(identities).encode()).hexdigest()
    if args.resume and fingerprint != checkpoint["dataset_fingerprint"]:
        p.error("Dataset file list/order changed since checkpoint; use the same dataset to resume")
    manifest = {"fingerprint": fingerprint, "training_count": len(train_indices),
                "validation_files": [identities[i] for i in validation_indices],
                "validation_indices": validation_indices, "seed": args.seed}
    (args.output / "split.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Dataset: {len(dataset)} available recordings, {len(train_indices)} train / {len(validation_indices)} validation", flush=True)
    if len(train_indices) < 100:
        print("Small diagnostic subset: this run cannot establish production voice quality.", flush=True)
    model = AeroFlowTTS().to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.requires_grad_(False)
    model.decoder.requires_grad_(True)
    critics = VocoderDiscriminators(channels=args.disc_channels).to(device)
    mel = MelReconstructionLoss().to(device)
    generator_optimizer = torch.optim.AdamW(model.decoder.parameters(), lr=args.lr, betas=(0.8, 0.99))
    discriminator_optimizer = torch.optim.AdamW(critics.parameters(), lr=args.disc_lr, betas=(0.8, 0.99))
    use_amp = device.type == "cuda"
    generator_scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    discriminator_scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    step, best_mel = 0, float("inf")
    provenance = checkpoint.get("provenance") if args.resume else {
        "source_checkpoint": str(args.checkpoint.resolve()),
        "source_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "source_training_steps": checkpoint.get("global_step"),
        "git_revision": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
        "torch": torch.__version__,
        "code_sha256": {str(path.relative_to(Path(__file__).resolve().parents[1])):
            hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [Path(__file__).resolve(), *sorted((Path(__file__).resolve().parents[1] / "aeroflow").rglob("*.py"))]},
    }
    if args.resume:
        critics.load_state_dict(checkpoint["discriminators"])
        generator_optimizer.load_state_dict(checkpoint["generator_optimizer"])
        discriminator_optimizer.load_state_dict(checkpoint["discriminator_optimizer"])
        generator_scaler.load_state_dict(checkpoint["generator_scaler"])
        discriminator_scaler.load_state_dict(checkpoint["discriminator_scaler"])
        step, best_mel = checkpoint["global_step"], checkpoint["best_mel"]
        torch.set_rng_state(checkpoint["rng_cpu"])
        if use_amp and checkpoint.get("rng_cuda") is not None:
            torch.cuda.set_rng_state_all(checkpoint["rng_cuda"])
    if step >= args.steps:
        p.error("--steps must exceed the checkpoint's completed update count")
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    def save():
        atomic_save({"trainer": "aeroflow_vocoder_v1", "model": model.state_dict(),
            "discriminators": critics.state_dict(), "generator_optimizer": generator_optimizer.state_dict(),
            "discriminator_optimizer": discriminator_optimizer.state_dict(),
            "generator_scaler": generator_scaler.state_dict(), "discriminator_scaler": discriminator_scaler.state_dict(),
            "global_step": step, "best_mel": best_mel, "config": config, "provenance": provenance,
            "dataset_fingerprint": fingerprint, "rng_cpu": torch.get_rng_state(),
            "rng_cuda": torch.cuda.get_rng_state_all() if use_amp else None}, args.output / "checkpoint_latest.pt")

    if not args.resume:
        baseline = validate(model, dataset, validation_indices, mel, device, args.output / "baseline", identities)
        best_mel = baseline["mel_l1"]
        atomic_save({"model": model.state_dict(), "global_step": 0, "vocoder_step": 0,
                     "validation": baseline, "provenance": provenance}, args.output / "model_best_mel.pt")
        save()
    batches = StepBatches(train_indices, args.batch_size, step, args.steps, args.seed)
    loader = DataLoader(dataset, batch_sampler=batches, collate_fn=collate_hifi_tts,
                        num_workers=args.workers, pin_memory=use_amp)
    start_time = time.monotonic()
    for batch in loader:
        model.eval()
        model.decoder.train()
        critics.train().requires_grad_(True)
        real = batch["audio"].to(device)
        lengths = batch["audio_lengths"].to(device)
        generator_optimizer.zero_grad(set_to_none=True)
        discriminator_optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            generated = model.reconstruct(real, lengths)
            real_crop, fake_crop = paired_crops(real, generated, lengths, args.crop_samples, args.seed + step)
            d_loss = discriminator_loss(critics(real_crop), critics(fake_crop.detach()))
        if not torch.isfinite(d_loss):
            raise FloatingPointError(f"Non-finite discriminator loss at update {step}")
        discriminator_scaler.scale(d_loss).backward()
        discriminator_scaler.unscale_(discriminator_optimizer)
        torch.nn.utils.clip_grad_norm_(critics.parameters(), 10)
        discriminator_scaler.step(discriminator_optimizer)
        discriminator_scaler.update()
        discriminator_optimizer.zero_grad(set_to_none=True)
        critics.requires_grad_(False)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            with torch.no_grad():
                real_features = critics(real_crop)
            adversarial, matching = generator_losses(real_features, critics(fake_crop))
            reconstruction = mel(real_crop, fake_crop)
            ramp = min(1.0, (step + 1) / max(1, args.adversarial_warmup))
            g_loss = args.mel_weight * reconstruction + ramp * (
                args.adversarial_weight * adversarial + args.feature_weight * matching)
        if not torch.isfinite(g_loss):
            raise FloatingPointError(f"Non-finite generator loss at update {step}")
        generator_scaler.scale(g_loss).backward()
        generator_scaler.unscale_(generator_optimizer)
        norm = torch.nn.utils.clip_grad_norm_(model.decoder.parameters(), 10)
        generator_scaler.step(generator_optimizer)
        generator_scaler.update()
        step += 1
        metrics = {"step": step, "mel": float(reconstruction.detach()), "feature": float(matching.detach()),
                   "adversarial": float(adversarial.detach()), "discriminator": float(d_loss.detach()),
                   "generator": float(g_loss.detach()), "gradient_norm": float(norm), "adversarial_ramp": ramp}
        with (args.output / "train.jsonl").open("a") as handle:
            handle.write(json.dumps(metrics) + "\n")
        if step == 1 or step % 10 == 0 or step == args.steps:
            print(json.dumps(metrics), flush=True)
        if step % args.validate_every == 0 or step == args.steps:
            evaluation = validate(model, dataset, validation_indices, mel, device, args.output / f"validation_{step:07d}", identities)
            export = {"model": model.state_dict(), "global_step": step, "vocoder_step": step,
                      "validation": evaluation, "provenance": provenance}
            atomic_save(export, args.output / "model_latest.pt")
            if evaluation["mel_l1"] < best_mel:
                best_mel = evaluation["mel_l1"]
                atomic_save(export, args.output / "model_best_mel.pt")
            print(f"Validation log-mel L1: {evaluation['mel_l1']:.5f}; rank perceptual quality by listening.", flush=True)
        if step % args.save_every == 0 or step % args.validate_every == 0 or step == args.steps:
            save()
    print(f"Completed {step} decoder updates; session elapsed {time.monotonic() - start_time:.1f}s. "
          f"Inference checkpoint: {args.output / 'model_latest.pt'}", flush=True)


if __name__ == "__main__":
    main()
