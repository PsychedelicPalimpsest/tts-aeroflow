"""
AeroFlow-v2 Kaggle 2x T4 Distributed Data Parallel (DDP) Training System.
Features:
- Dual-T4 DDP multi-GPU scaling via torchrun (--nproc_per_node=2) with single-GPU/CPU fallback.
- Hardware environment variables: NCCL_P2P_DISABLE=1, NCCL_IB_DISABLE=1, OMP_NUM_THREADS=2.
- FP16 Mixed Precision training with torch.amp.GradScaler('cuda') and FP32 spectral loss computation.
- 4-vCPU balanced DataLoader (2 workers per GPU process, persistent workers, prefetch factor 2).
- Automated Atomic Checkpointing (temp_path -> os.replace) saving full model, optimizer, scaler,
  scheduler, step, epoch, and CPU/CUDA/NumPy/Python RNG states.
- Automated Checkpoint Retrieval & Resumption searching /kaggle/input/** and /kaggle/working/checkpoints.
- 11.2-Hour Wall-Clock Watchdog preventing Kaggle 12-hour session timeout with clean exit code 0.
"""

import os
import sys

# 1. Enforce Kaggle multi-GPU PCIe & thread constraints BEFORE importing torch
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["OMP_NUM_THREADS"] = "2"

import argparse
import glob
import math
from pathlib import Path
import random
import time
from typing import Dict, Optional, Tuple, Any

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

# Ensure aeroflow is importable from repository root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aeroflow import (
    AeroFlowTTS,
    AeroFlowLoss,
    HiFiTTSDataset,
    HuggingFaceHiFiTTSDataset,
    StreamingHiFiTTSDataset,
    collate_hifi_tts,
    create_synthetic_batch
)
from torch.utils.data import IterableDataset


class SyntheticHiFiTTSDataset(Dataset):
    """Fallback synthetic dataset simulating Speaker 9017 for dry-run verification."""

    def __init__(self, num_samples: int = 100, sample_rate: int = 24000, hop_length: int = 240):
        super().__init__()
        self.num_samples = num_samples
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.phonemizer = AeroFlowTTS().phonemizer
        self.sentences = [
            "Peter Piper picked a peck of pickled peppers.",
            "AeroFlow two text to speech engine running on Intel Core processor.",
            "Deep resonant American male voice synthesis with optimal transport flow matching.",
            "The quick brown fox jumps over the lazy dog.",
            "High fidelity audio reproduction at twenty-four kilohertz."
        ]

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        text = self.sentences[idx % len(self.sentences)]
        tokens = self.phonemizer.text_to_sequence(text)

        dur_s = 1.0 + (idx % 3) * 0.5  # 1.0s, 1.5s, 2.0s
        n_samples = int(dur_s * self.sample_rate)
        n_samples = (n_samples // self.hop_length) * self.hop_length

        t = torch.linspace(0, dur_s, n_samples)
        f0 = 110.0 + 5.0 * torch.sin(2.0 * math.pi * 1.5 * t)
        phase = torch.cumsum(2.0 * math.pi * f0 / self.sample_rate, dim=0)
        sig = 0.5 * torch.sin(phase) + 0.3 * torch.sin(2 * phase) + 0.2 * torch.sin(3 * phase)
        envelope = torch.sin(math.pi * torch.linspace(0, 1, n_samples)).clamp(min=0.0)
        noise = torch.randn(n_samples) * 0.005
        audio = (sig * envelope + noise).clamp(min=-0.95, max=0.95)

        return {
            "tokens": torch.tensor(tokens, dtype=torch.long),
            "audio": audio.float(),
            "text": text
        }


def setup_environment(seed: int = 42) -> Tuple[torch.device, int, int, int, bool]:
    """
    Initializes DDP process group or single-device environment.
    Returns: (device, rank, world_size, local_rank, is_distributed)
    """
    torch.set_num_threads(2)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        if torch.cuda.is_available():
            device = torch.device(f"cuda:{local_rank}")
            torch.cuda.set_device(device)
            torch.cuda.manual_seed_all(seed)
            dist.init_process_group("nccl", rank=rank, world_size=world_size)
        else:
            device = torch.device("cpu")
            dist.init_process_group("gloo", rank=rank, world_size=world_size)

        is_distributed = True
    elif torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
        torch.cuda.manual_seed_all(seed)
        rank = 0
        world_size = 1
        local_rank = 0
        is_distributed = False
    else:
        device = torch.device("cpu")
        rank = 0
        world_size = 1
        local_rank = 0
        is_distributed = False

    return device, rank, world_size, local_rank, is_distributed


def save_atomic_checkpoint(
    checkpoint_path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    scheduler: Optional[Any],
    global_step: int,
    epoch: int,
    best_loss: float,
    is_distributed: bool
):
    """
    Saves checkpoint atomically to prevent corruption using temp_path -> os.replace.
    Serializes complete model weights, optimizer momentum, scaler, and full RNG states.
    """
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = checkpoint_path.with_suffix(".tmp")

    raw_model = model.module if is_distributed else model

    rng_state = {
        "cpu": torch.get_rng_state(),
        "numpy": np.random.get_state(),
        "python": random.getstate()
    }
    if torch.cuda.is_available():
        rng_state["cuda"] = torch.cuda.get_rng_state_all()

    state = {
        "model": raw_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None and hasattr(scaler, "state_dict") else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "global_step": global_step,
        "epoch": epoch,
        "best_loss": best_loss,
        "rng_state": rng_state,
        "timestamp": time.time()
    }

    torch.save(state, temp_path)
    os.replace(temp_path, checkpoint_path)


def find_existing_checkpoint(search_dirs: list) -> Optional[Path]:
    """
    Searches for existing checkpoints in Kaggle input directories and working directory.
    Checks /kaggle/input/**/checkpoint_latest.pt and working directory.
    """
    # 1. Look for user-provided paths or common Kaggle input paths
    candidate_patterns = [
        "/kaggle/input/**/checkpoint_latest.pt",
        "/kaggle/working/**/checkpoint_latest.pt",
    ]
    for search_dir in search_dirs:
        candidate_patterns.append(str(Path(search_dir) / "checkpoint_latest.pt"))
        candidate_patterns.append(str(Path(search_dir) / "**" / "checkpoint_latest.pt"))

    for pattern in candidate_patterns:
        matches = glob.glob(pattern, recursive=True)
        if matches:
            # Sort by modification time to find most recent
            matches.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            return Path(matches[0])

    return None


def resume_from_checkpoint(
    checkpoint_path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    scheduler: Optional[Any],
    device: torch.device,
    is_distributed: bool
) -> Tuple[int, int, float]:
    """
    Restores model weights, optimizer momentum, scaler scale, scheduler state, and RNG states.
    Returns: (global_step, epoch, best_loss)
    """
    checkpoint = torch.load(str(checkpoint_path), map_location=device, weights_only=False)

    raw_model = model.module if is_distributed else model
    raw_model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    if scaler is not None and checkpoint.get("scaler") is not None and hasattr(scaler, "load_state_dict"):
        scaler.load_state_dict(checkpoint["scaler"])

    if scheduler is not None and checkpoint.get("scheduler") is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])

    global_step = checkpoint.get("global_step", 0)
    epoch = checkpoint.get("epoch", 0)
    best_loss = checkpoint.get("best_loss", float("inf"))

    rng = checkpoint.get("rng_state", {})
    if "cpu" in rng:
        torch.set_rng_state(rng["cpu"])
    if "numpy" in rng:
        np.random.set_state(rng["numpy"])
    if "python" in rng:
        random.setstate(rng["python"])
    if torch.cuda.is_available() and "cuda" in rng:
        try:
            torch.cuda.set_rng_state_all(rng["cuda"])
        except Exception:
            pass

    return global_step, epoch, best_loss


def train():
    parser = argparse.ArgumentParser(description="AeroFlow-v2 Kaggle 2x T4 DDP Training")
    parser.add_argument("--manifest-path", type=str, default=None, help="Path to Hi-Fi TTS JSON manifest")
    parser.add_argument("--audio-dir", type=str, default=None, help="Directory containing audio files")
    parser.add_argument("--dataset-source", type=str, default="auto",
                        choices=["auto", "manifest", "hf", "hf-streaming", "synthetic"],
                        help="Dataset backend: 'manifest' (local JSON), 'hf' (lazy map over "
                             "MikhailT/hifi-tts, Arrow memory-mapped, one-row RAM), "
                             "'hf-streaming' (streaming=True, no local copy, O(1) RAM), "
                             "'synthetic' (fallback), or 'auto' (manifest if found else synthetic)")
    parser.add_argument("--hf-repo-id", type=str, default="MikhailT/hifi-tts",
                        help="HF repo for --dataset-source hf/hf-streaming "
                             "(use MikhailT/hifi-tts-light for small-format testing)")
    parser.add_argument("--hf-subset", type=str, default="clean",
                        help="HF config: 'clean', 'other', or 'all'")
    parser.add_argument("--hf-split", type=str, default="train",
                        help="HF split: 'train'/'test'/'dev' for clean/other, "
                             "or 'train.clean'/'train.other'/... for subset 'all'")
    parser.add_argument("--hf-speaker", type=str, default="9017",
                        help="Comma-separated speaker id(s) to keep (default '9017' single-voice). "
                             "Use 'all' for every speaker.")
    parser.add_argument("--hf-min-duration", type=float, default=0.5)
    parser.add_argument("--hf-max-duration", type=float, default=12.0)
    parser.add_argument("--checkpoint-dir", type=str, default="/kaggle/working/checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--resume-path", type=str, default=None, help="Explicit checkpoint path to resume from")
    parser.add_argument("--auto-resume", action="store_true", default=True, help="Auto-search for existing checkpoints")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size per GPU")
    parser.add_argument("--epochs", type=int, default=100, help="Total number of training epochs")
    parser.add_argument("--lr", type=float, default=2e-4, help="AdamW learning rate")
    parser.add_argument("--save-interval-steps", type=int, default=500, help="Save latest checkpoint every N steps")
    parser.add_argument("--max-hours", type=float, default=11.2, help="Watchdog timeout in hours (default 11.2h)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--synthetic-samples", type=int, default=200, help="Samples for synthetic fallback dataset")
    args = parser.parse_args()

    # 1. Environment & DDP Setup
    device, rank, world_size, local_rank, is_distributed = setup_environment(seed=args.seed)
    is_master = (rank == 0)

    if is_master:
        print("=" * 80)
        print("AEROFLOW-v2: KAGGLE 2x T4 DISTRIBUTED DATA PARALLEL TRAINING SYSTEM")
        print(f"Distributed DDP: {is_distributed} | World Size: {world_size} | Device: {device}")
        print(f"OMP Threads: {os.environ.get('OMP_NUM_THREADS')} | NCCL P2P Disabled: {os.environ.get('NCCL_P2P_DISABLE')}")
        print("=" * 80)

    checkpoint_dir = Path(args.checkpoint_dir)
    if is_master:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # 2. Dataset & DataLoader (4-vCPU balanced: 2 workers per GPU process)
    # Backends share the same collate format so this is a pure switch-out.
    source = args.dataset_source
    if source == "auto":
        source = "manifest" if (args.manifest_path and os.path.exists(args.manifest_path)) else "synthetic"

    is_streaming = False
    if source == "manifest":
        if args.manifest_path and os.path.exists(args.manifest_path):
            if is_master:
                print(f"[Dataset] Loading Hi-Fi TTS manifest: {args.manifest_path}")
            dataset = HiFiTTSDataset(
                manifest_path=args.manifest_path,
                audio_dir=args.audio_dir,
                sample_rate=24000,
                hop_length=240
            )
        else:
            if is_master:
                print(f"[Dataset] Manifest not specified or not found. Using SyntheticHiFiTTSDataset ({args.synthetic_samples} items).")
            dataset = SyntheticHiFiTTSDataset(
                num_samples=args.synthetic_samples,
                sample_rate=24000,
                hop_length=240
            )
    elif source in ("hf", "hf-streaming"):
        speaker_arg = (args.hf_speaker or "").strip().lower()
        speaker_ids = None if speaker_arg in ("", "all", "none") else [s.strip() for s in args.hf_speaker.split(",")]
        hf_common = dict(
            repo_id=args.hf_repo_id,
            subset=args.hf_subset,
            split=args.hf_split,
            speaker_ids=speaker_ids,
            sample_rate=24000,
            hop_length=240,
            min_duration_s=args.hf_min_duration,
            max_duration_s=args.hf_max_duration,
        )
        if source == "hf":
            if is_master:
                print(f"[Dataset] Loading lazy HF map dataset: {args.hf_repo_id} "
                      f"(subset={args.hf_subset}, split={args.hf_split}, speakers={speaker_ids}). "
                      f"Arrow memory-mapped, audio decoded one row at a time (40GB never in RAM).")
            dataset = HuggingFaceHiFiTTSDataset(**hf_common)
        else:
            if is_master:
                print(f"[Dataset] Loading HF streaming dataset: {args.hf_repo_id} "
                      f"(subset={args.hf_subset}, split={args.hf_split}, speakers={speaker_ids}). "
                      f"streaming=True, no local copy, O(1) RAM.")
            dataset = StreamingHiFiTTSDataset(
                **hf_common, rank=rank, world_size=world_size,
            )
            is_streaming = True
        if is_master:
            print(f"[Dataset] HF backend ready ({len(dataset) if not is_streaming else 'streaming'} rows). "
                  f"For format testing use --hf-repo-id MikhailT/hifi-tts-light.")
    else:
        if is_master:
            print(f"[Dataset] Using SyntheticHiFiTTSDataset ({args.synthetic_samples} items).")
        dataset = SyntheticHiFiTTSDataset(
            num_samples=args.synthetic_samples,
            sample_rate=24000,
            hop_length=240
        )

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True) if (is_distributed and not isinstance(dataset, IterableDataset)) else None
    use_cuda = device.type == "cuda"
    num_workers = 2 if use_cuda else 0

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(sampler is None and not isinstance(dataset, IterableDataset)),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=use_cuda,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
        collate_fn=collate_hifi_tts
    )

    # 3. Model Architecture
    model = AeroFlowTTS(
        vocab_size=84,
        text_dim=192,
        latent_dim=32,
        decoder_dim=256,
        n_fft=1024,
        hop_length=240,
        heun_steps=6,
        heun_rho=1.5
    ).to(device)

    if is_distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    # 4. Convex Spectral Loss Suite & Optimizer
    loss_fn = AeroFlowLoss(
        lambda_cfm=1.0,
        lambda_dur=1.0,
        lambda_prior=1.0,
        lambda_mr_stft=1.0,
        lambda_if=0.5
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.8, 0.99),
        weight_decay=0.01
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs * max(1, len(loader)),
        eta_min=1e-5
    )

    # FP16 Mixed Precision Scaler
    use_amp = use_cuda
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # 5. Automated Checkpoint Resumption
    global_step = 0
    start_epoch = 0
    best_loss = float("inf")

    ckpt_to_load = None
    if args.resume_path and os.path.exists(args.resume_path):
        ckpt_to_load = Path(args.resume_path)
    elif args.auto_resume:
        ckpt_to_load = find_existing_checkpoint([checkpoint_dir, "./checkpoints", "/kaggle/working/checkpoints"])

    if ckpt_to_load:
        if is_master:
            print(f"\n[Checkpoint Resume] Found existing checkpoint at: {ckpt_to_load}")
            print(f"                    Restoring weights, optimizer, scaler, and RNG states...")
        global_step, start_epoch, best_loss = resume_from_checkpoint(
            ckpt_to_load, model, optimizer, scaler, scheduler, device, is_distributed
        )
        if is_master:
            print(f"[Checkpoint Resume] Successfully resumed from Global Step {global_step}, Epoch {start_epoch}!")
            print(f"                    Previous best loss: {best_loss:.4f}\n")

    # 6. Wall-Clock Watchdog
    start_wall_time = time.time()
    max_watchdog_seconds = args.max_hours * 3600.0

    if is_master:
        print(f"[Watchdog Config] Limit: {args.max_hours:.1f} hours ({max_watchdog_seconds:.0f}s)")
        print(f"[Training] Starting training from epoch {start_epoch} to {args.epochs}...\n")

    latest_ckpt_path = checkpoint_dir / "checkpoint_latest.pt"
    best_ckpt_path = checkpoint_dir / "checkpoint_best.pt"

    for epoch in range(start_epoch, args.epochs):
        if is_distributed and sampler is not None:
            sampler.set_epoch(epoch)

        model.train()
        epoch_loss = 0.0
        batches_in_epoch = 0

        for batch_idx, batch in enumerate(loader):
            # Watchdog check
            elapsed_time = time.time() - start_wall_time
            if elapsed_time >= max_watchdog_seconds:
                if is_master:
                    print(f"\n" + "!" * 80)
                    print(f"[Watchdog Triggered] Elapsed time {elapsed_time/3600:.2f}h >= {args.max_hours:.2f}h limit!")
                    print(f"[Watchdog] Gracefully saving final checkpoint before Kaggle session cutoff...")
                    save_atomic_checkpoint(
                        latest_ckpt_path, model, optimizer, scaler, scheduler,
                        global_step, epoch, best_loss, is_distributed
                    )
                    print(f"[Watchdog] Saved {latest_ckpt_path} successfully. Exiting code 0.")
                    print("!" * 80)
                if is_distributed:
                    dist.barrier()
                sys.exit(0)

            step_start = time.perf_counter()
            tokens = batch["phoneme_tokens"].to(device, non_blocking=True)
            audio = batch["audio"].to(device, non_blocking=True)
            text_lens = batch["text_lengths"].to(device, non_blocking=True)
            audio_lens = batch["audio_lengths"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            # Mixed precision forward pass
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                raw_model = model.module if is_distributed else model
                out = raw_model.forward_train(
                    phoneme_tokens=tokens,
                    audio_24k=audio,
                    text_lengths=text_lens,
                    audio_lengths=audio_lens
                )

                # Loss suite operates in FP32
                loss_total, metrics = loss_fn(
                    v_pred=out["v_pred"],
                    u_target=out["u_target"],
                    log_dur_pred=out["log_dur_pred"],
                    dur_target=out["dur_target"],
                    y_audio=out["audio_gt"],
                    y_hat_audio=out["audio_hat"],
                    text_mask=out["text_mask"],
                    frame_mask=out["frame_mask"],
                    audio_mask=out["audio_mask"],
                    text_proj_expanded=out["text_proj_expanded"],
                    z_target=out["z_target"]
                )

            # Gradient backward & unscale
            if use_amp:
                scaler.scale(loss_total).backward()
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss_total.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            scheduler.step()
            global_step += 1
            epoch_loss += loss_total.item()
            batches_in_epoch += 1
            step_duration_ms = (time.perf_counter() - step_start) * 1000.0

            # Logging
            if is_master and (global_step % 20 == 0 or global_step == 1):
                lr_curr = optimizer.param_groups[0]["lr"]
                print(
                    f"[Ep {epoch:03d} | Step {global_step:06d}] "
                    f"Loss: {loss_total.item():.4f} "
                    f"(CFM: {metrics['loss_cfm'].item():.3f}, "
                    f"Dur: {metrics['loss_dur'].item():.3f}, "
                    f"Prior: {metrics['loss_prior'].item():.3f}, "
                    f"MR-STFT: {metrics['loss_mr_stft'].item():.3f}, "
                    f"IF: {metrics['loss_if'].item():.3f}) | "
                    f"Grad: {grad_norm.item():.2f} | "
                    f"LR: {lr_curr:.2e} | "
                    f"Time: {step_duration_ms:.1f}ms"
                )

            # Atomic checkpoint saving
            if is_master and (global_step % args.save_interval_steps == 0):
                save_atomic_checkpoint(
                    latest_ckpt_path, model, optimizer, scaler, scheduler,
                    global_step, epoch, best_loss, is_distributed
                )
                print(f"  --> [Checkpoint] Saved latest checkpoint to {latest_ckpt_path} (step {global_step})")

                if loss_total.item() < best_loss:
                    best_loss = loss_total.item()
                    save_atomic_checkpoint(
                        best_ckpt_path, model, optimizer, scaler, scheduler,
                        global_step, epoch, best_loss, is_distributed
                    )
                    print(f"  --> [Checkpoint] New best loss {best_loss:.4f}! Saved to {best_ckpt_path}")

        # End of epoch checkpoint
        if is_master:
            avg_epoch_loss = epoch_loss / max(1, batches_in_epoch)
            print(f"[Epoch {epoch:03d} Complete] Avg Loss: {avg_epoch_loss:.4f}")
            save_atomic_checkpoint(
                latest_ckpt_path, model, optimizer, scaler, scheduler,
                global_step, epoch + 1, best_loss, is_distributed
            )

    if is_master:
        print("\n" + "=" * 80)
        print("AEROFLOW-v2 TRAINING COMPLETE (EXIT CODE 0)")
        print(f"Total steps: {global_step:,} | Best Loss: {best_loss:.4f}")
        print(f"Final checkpoint: {latest_ckpt_path}")
        print("=" * 80)

    if is_distributed:
        dist.destroy_process_group()

    return 0


if __name__ == "__main__":
    sys.exit(train())
