"""
AeroFlow-v2 CPU pipeline profiler: finds whether training steps are
data-bound or compute-bound and names the hottest stage.

Stages:
  1. micro      - per-item costs: resample, phonemize, peak-norm, collate
                  (incl. batch-256 padding waste), vectorized MAS.
  2. loader     - sustained DataLoader batches/sec (workers/prefetch matrix).
  3. step       - torch.profiler CPU breakdown of forward_train + loss +
                  backward on a small synthetic batch.

Runs on CPU with the synthetic path by default (no network, no `datasets`
package needed). If `datasets` + network are available, --hf-light adds the
real MikhailT/hifi-tts-light map dataset to stage 2.

Usage:
    python3 scripts/profile_pipeline.py [--stages micro,loader,step]
                                        [--batch-size 8] [--steps 20]
"""

import argparse
import resource
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aeroflow import Phonemizer, collate_hifi_tts, create_synthetic_batch
from aeroflow.dataset.hf_hifi_tts import _peak_normalize, _resample_mono
from aeroflow.models.alignment import maximum_path_viterbi


def rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def bench(fn, repeats=5, warmup=1, label=""):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000.0)
    print(f"  {label}: {np.mean(times):9.2f} ms  (min {np.min(times):.2f}, n={repeats})")
    return float(np.mean(times))


def stage_micro(batch_size):
    print("\n[Stage 1] Micro-benchmarks (per-item pipeline costs)")
    sr441, sr24 = 44100, 24000
    t = np.linspace(0, 3.0, sr441 * 3, endpoint=False).astype(np.float32)
    wave44 = (0.5 * np.sin(2 * np.pi * 110 * t)).astype(np.float32)
    stereo = np.stack([wave44, wave44 * 0.5], axis=-1)
    phon = Phonemizer()
    sent = "There is Monsieur returning from hunting."

    bench(lambda: _resample_mono(wave44, sr441, sr24), label="resample 44.1k->24k mono 3s")
    bench(lambda: _resample_mono(stereo, sr441, sr24), label="resample 44.1k->24k stereo 3s")
    bench(lambda: phon.text_to_sequence(sent), label="phonemize 40-char sentence")
    bench(lambda: _peak_normalize(wave44), label="peak-normalize 3s")

    # Collate at scale: mixed 1s..12s clips expose padding waste.
    rng = np.random.default_rng(0)
    items = []
    raw_bytes = 0
    for i in range(batch_size):
        secs = float(rng.uniform(1.0, 12.0))
        n = int(secs * sr24)
        audio = torch.randn(n)
        toks = torch.randint(1, 84, (int(rng.integers(20, 150)),))
        raw_bytes += audio.numel() * 4 + toks.numel() * 8
        items.append({"tokens": toks, "audio": audio, "text": sent})
    rss0 = rss_mb()
    holder = {}
    bench(lambda: holder.setdefault("b", collate_hifi_tts(items)),
          repeats=3, label=f"collate batch={batch_size} mixed 1-12s")
    b = holder["b"]
    padded_bytes = b["audio"].numel() * 4 + b["phoneme_tokens"].numel() * 8
    print(f"  collate: padded {padded_bytes/1e6:.1f} MB vs raw {raw_bytes/1e6:.1f} MB "
          f"(waste x{padded_bytes/max(1,raw_bytes):.2f}), peak RSS +{rss_mb()-rss0:.0f} MB")

    # Vectorized MAS at realistic training shapes.
    torch.manual_seed(0)
    for (B, N, T) in [(2, 40, 300), (32, 80, 600), (256, 80, 600)]:
        scores = torch.randn(B, N, T)
        tl = torch.full((B,), N)
        al = torch.full((B,), T)
        bench(lambda: maximum_path_viterbi(scores, tl, al),
              repeats=3, label=f"MAS B={B} N={N} T={T}")


class _SynthItems(torch.utils.data.Dataset):
    """On-the-fly items with realistic per-item CPU work (synth + phonemize)."""

    def __init__(self, n=200):
        self.n = n
        self.phon = Phonemizer()
        self.sents = ["There is Monsieur returning from hunting.",
                      "Peter Piper picked a peck of pickled peppers."]

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        rng = np.random.default_rng(i)
        secs = float(rng.uniform(1.0, 6.0))
        n = int(secs * 24000)
        t = np.linspace(0, secs, n, endpoint=False).astype(np.float32)
        audio = torch.from_numpy(
            (0.5 * np.sin(2 * np.pi * 110 * t)).astype(np.float32))
        text = self.sents[i % 2]
        return {"tokens": torch.tensor(self.phon.text_to_sequence(text)),
                "audio": audio, "text": text}


def loader_rate(ds, batch_size, workers, prefetch, steps):
    kw = dict(batch_size=batch_size, shuffle=False, collate_fn=collate_hifi_tts,
              num_workers=workers)
    if workers > 0:
        kw.update(persistent_workers=True, prefetch_factor=prefetch)
    loader = torch.utils.data.DataLoader(ds, **kw)
    it = iter(loader)
    next(it)  # warmup (worker spin-up)
    t0 = time.perf_counter()
    for _ in range(steps):
        try:
            b = next(it)
        except StopIteration:
            it = iter(loader)
            b = next(it)
    dt = time.perf_counter() - t0
    return dt / steps, b


def stage_loader(batch_size, steps, hf_light):
    print("\n[Stage 2] DataLoader throughput (sec/batch, lower is better)")
    ds = _SynthItems()
    for workers, prefetch in [(0, None), (2, 2), (2, 4)]:
        s_per_batch, b = loader_rate(ds, batch_size, workers, prefetch or 2, steps)
        print(f"  synthetic workers={workers} prefetch={prefetch}: "
              f"{s_per_batch*1000:8.1f} ms/batch  (audio {tuple(b['audio'].shape)})")
    if hf_light:
        try:
            from aeroflow import HuggingFaceHiFiTTSDataset
            hf = HuggingFaceHiFiTTSDataset(
                repo_id="MikhailT/hifi-tts-light", subset="clean", split="train",
                speaker_ids=None, min_duration_s=0.0, max_duration_s=30.0)
            s_per_batch, b = loader_rate(hf, 8, 0, 2, 4)
            print(f"  hifi-tts-light workers=0: {s_per_batch*1000:8.1f} ms/batch "
                  f"(audio {tuple(b['audio'].shape)})")
        except Exception as exc:
            print(f"  hifi-tts-light skipped: {exc}")


def stage_step():
    print("\n[Stage 3] Training-step CPU breakdown (synthetic batch=2 x 1.0s)")
    from aeroflow import AeroFlowLoss, AeroFlowTTS
    torch.manual_seed(42)
    device = torch.device("cpu")
    model = AeroFlowTTS().to(device)
    loss_fn = AeroFlowLoss().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4)
    batch = create_synthetic_batch(batch_size=2, audio_dur_s=1.0)
    tok, aud = batch["phoneme_tokens"], batch["audio"]
    tl, al = batch["text_lengths"], batch["audio_lengths"]

    # Manual per-phase timers.
    model.train()
    opt.zero_grad()
    t0 = time.perf_counter()
    out = model.forward_train(tok, aud, tl, al)
    t_fwd = time.perf_counter() - t0
    t0 = time.perf_counter()
    loss, _ = loss_fn(out["v_pred"], out["u_target"], out["log_dur_pred"],
                      out["dur_target"], out["audio_gt"], out["audio_hat"])
    t_loss = time.perf_counter() - t0
    t0 = time.perf_counter()
    loss.backward()
    t_bwd = time.perf_counter() - t0
    print(f"  forward_train: {t_fwd*1000:.0f} ms | loss: {t_loss*1000:.0f} ms | "
          f"backward: {t_bwd*1000:.0f} ms | total: {(t_fwd+t_loss+t_bwd)*1000:.0f} ms")

    # Op-level profile of one more step.
    opt.zero_grad()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],
                                record_shapes=False) as prof:
        out = model.forward_train(tok, aud, tl, al)
        loss, _ = loss_fn(out["v_pred"], out["u_target"], out["log_dur_pred"],
                          out["dur_target"], out["audio_gt"], out["audio_hat"])
        loss.backward()
    print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=18))


def main(argv=None):
    ap = argparse.ArgumentParser(description="AeroFlow-v2 CPU pipeline profiler")
    ap.add_argument("--stages", default="micro,loader,step")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--hf-light", action="store_true")
    args = ap.parse_args(argv)
    print(f"torch {torch.__version__} | threads {torch.get_num_threads()} | "
          f"RSS {rss_mb():.0f} MB")
    stages = {s.strip() for s in args.stages.split(",")}
    if "micro" in stages:
        stage_micro(args.batch_size)
    if "loader" in stages:
        stage_loader(min(args.batch_size, 8), args.steps, args.hf_light)
    if "step" in stages:
        stage_step()
    print("\nDone.")


if __name__ == "__main__":
    main()
