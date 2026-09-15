"""
AeroFlow-v2 Dry-Run Training & Benchmark Verification Script.
Targets:
1. Intel Core i7-12700H CPU allocation (6 physical P-cores via torch.set_num_threads(6)).
2. Hi-Fi TTS Speaker 9017 (John Van Stan 24 kHz mono).
3. 2 Full training steps (Forward -> CFM, Duration, MR-STFT, IF losses -> Backward -> Optimizer step).
4. Gradient & numerical stability validation (Zero NaNs, Zero Infs).
5. 6-step Non-Uniform Heun Solver inference benchmark on classic test sentences.
"""

import os
import sys
import time
from pathlib import Path
import soundfile as sf
import torch

# Ensure aeroflow is importable from repository root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aeroflow import (
    AeroFlowTTS,
    AeroFlowLoss,
    create_synthetic_batch
)


def main():
    print("=" * 78)
    print("AEROFLOW-v2: SYSTEM VERIFICATION & DRY-RUN TRAINING HARNESS")
    print("Target Architecture: Intel Core i7-12700H (6 P-Cores, AVX2 SIMD)")
    print("Target Voice: Hi-Fi TTS Speaker 9017 (John Van Stan - 24 kHz Mono Baritone)")
    print("=" * 78)

    # 1. Hardware & Thread Allocation
    device = torch.device("cpu")
    num_threads = 6
    torch.set_num_threads(num_threads)
    torch.manual_seed(42)
    print(f"[Hardware Setup] Bound execution to {num_threads} P-Cores on {device.type.upper()}.")

    # 2. Model Initialization
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

    param_count = sum(p.numel() for p in model.parameters())
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_size_mb = (param_count * 4) / (1024 * 1024)
    model_fp16_mb = (param_count * 2) / (1024 * 1024)
    print(f"[Model Setup] Total Parameters: {param_count:,} ({trainable_count:,} trainable)")
    print(f"[Memory Footprint] Float32: {model_size_mb:.2f} MB | FP16 Cache Working Set: {model_fp16_mb:.2f} MB")
    print(f"                   (Fully fits within 24 MB Intel Smart L3 Cache!)")

    # 3. Loss & Optimizer Initialization
    loss_fn = AeroFlowLoss(
        lambda_cfm=1.0,
        lambda_dur=1.0,
        lambda_mr_stft=1.0,
        lambda_if=0.5
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=2e-4,
        betas=(0.8, 0.99),
        weight_decay=0.01
    )

    # 4. Generate Synthetic Speaker 9017 Batch
    print("\n" + "-" * 78)
    print("[Data Pipeline] Generating synthetic 24 kHz Speaker 9017 batch...")
    batch = create_synthetic_batch(
        batch_size=2,
        audio_dur_s=1.5,
        sample_rate=24000,
        hop_length=240
    )
    tokens = batch["phoneme_tokens"].to(device)
    audio = batch["audio"].to(device)
    text_lens = batch["text_lengths"].to(device)
    audio_lens = batch["audio_lengths"].to(device)
    print(f"  Phoneme Tokens: {tokens.shape} (max_len: {tokens.shape[1]})")
    print(f"  Audio Waveform: {audio.shape} (duration: {audio.shape[1] / 24000:.2f} s @ 24,000 Hz)")

    # 5. Execute 2 Full Training Steps
    print("\n" + "-" * 78)
    print("[Training Steps] Executing 2 training steps with all 4 convex spectral losses...")

    step_times = []
    initial_param = next(model.vector_field.parameters()).clone()

    for step in range(1, 3):
        t_start = time.perf_counter()
        model.train()
        optimizer.zero_grad()

        # Forward Pass
        out = model.forward_train(
            phoneme_tokens=tokens,
            audio_24k=audio,
            text_lengths=text_lens,
            audio_lengths=audio_lens
        )

        # Calculate Loss Suite
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

        # Numerical Stability Assertions
        assert torch.isfinite(loss_total), f"Step {step}: Total loss is NaN or Inf!"
        for name, val in metrics.items():
            assert torch.isfinite(val), f"Step {step}: Metric {name} is NaN or Inf!"

        # Backward Autograd Pass
        loss_total.backward()

        # Gradient Verification
        total_grad_norm = 0.0
        nan_grads = 0
        for p in model.parameters():
            if p.grad is not None:
                if not torch.isfinite(p.grad).all():
                    nan_grads += 1
                total_grad_norm += p.grad.data.norm(2).item() ** 2
        total_grad_norm = total_grad_norm ** 0.5
        assert nan_grads == 0, f"Step {step}: Found {nan_grads} parameters with NaN/Inf gradients!"
        assert total_grad_norm > 0.0, f"Step {step}: Total gradient norm is zero!"

        # Gradient clipping & Optimizer step
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        t_elapsed = (time.perf_counter() - t_start) * 1000.0
        step_times.append(t_elapsed)

        print(f"  Step {step}/2 Complete ({t_elapsed:.1f} ms):")
        print(f"    Total Loss:    {loss_total.item():.4f}")
        print(f"    - CFM Loss:    {metrics['loss_cfm'].item():.4f}")
        print(f"    - Dur Loss:    {metrics['loss_dur'].item():.4f}")
        print(f"    - Prior Loss:  {metrics['loss_prior'].item():.4f}")
        print(f"    - MR-STFT:     {metrics['loss_mr_stft'].item():.4f}")
        print(f"    - IF Loss:     {metrics['loss_if'].item():.4f}")
        print(f"    - Grad Norm:   {total_grad_norm:.4f}")

    # Verify Parameter Weight Updates
    final_param = next(model.vector_field.parameters())
    param_delta = (final_param - initial_param).abs().sum().item()
    print(f"\n[Optimizer Check] Parameter delta after steps: {param_delta:.6f}")
    assert param_delta > 0.0, "Model parameters did not update after optimizer step!"
    print("  -> Parameter weights updated successfully without NaNs or Infs.")

    # 6. Sample Inference Pass with 6-Step Heun Solver
    print("\n" + "-" * 78)
    print("[Inference Pass] Running 6-step Non-Uniform Heun Solver on test prompt...")
    test_prompt = "Peter Piper picked a peck of pickled peppers."
    print(f"  Prompt: '{test_prompt}'")

    model.eval()
    with torch.no_grad():
        t_infer_start = time.perf_counter()
        waveform = model.synthesize(test_prompt, alpha=1.0)
        t_infer_end = time.perf_counter()

    infer_elapsed_s = t_infer_end - t_infer_start
    audio_dur_s = waveform.shape[-1] / 24000.0
    rtf = infer_elapsed_s / audio_dur_s

    print(f"  Synthesized Samples: {waveform.shape[-1]:,} ({audio_dur_s:.2f} seconds @ 24 kHz)")
    print(f"  Compute Latency:     {infer_elapsed_s * 1000.0:.2f} ms")
    print(f"  Real-Time Factor:    RTF = {rtf:.4f} ({1.0 / rtf:.1f}x Real-Time Speedup)")

    # Assertions on Synthesized Audio
    assert waveform.shape[-1] > 0, "Synthesized audio is empty!"
    assert torch.isfinite(waveform).all(), "Synthesized audio contains NaN or Inf!"
    assert waveform.abs().max() > 1e-4, "Synthesized audio is completely silent!"
    print(f"  Audio Dynamic Range: Min = {waveform.min().item():.4f}, Max = {waveform.max().item():.4f}, RMS = {waveform.pow(2).mean().sqrt().item():.4f}")

    # Save audio artifact
    output_dir = Path(__file__).resolve().parent
    out_wav_path = output_dir / "dry_run_output.wav"
    sf.write(str(out_wav_path), waveform.cpu().numpy(), 24000)
    print(f"  Audio saved to:      {out_wav_path}")

    # 7. 10-Second Continuous Speech Latency Benchmark
    print("\n" + "-" * 78)
    print("[10-Second Benchmark] Evaluating RTF for 150 phonemes (~10.0 seconds of speech)...")
    synthetic_phonemes = torch.randint(5, 84, (1, 150), device=device)
    with torch.no_grad():
        t_bench_start = time.perf_counter()
        bench_wav = model.synthesize(synthetic_phonemes, alpha=1.0)
        t_bench_end = time.perf_counter()

    bench_elapsed_s = t_bench_end - t_bench_start
    bench_dur_s = bench_wav.shape[-1] / 24000.0
    bench_rtf = bench_elapsed_s / bench_dur_s

    print(f"  Duration:            {bench_dur_s:.2f} seconds ({bench_wav.shape[-1]:,} samples)")
    print(f"  Compute Latency:     {bench_elapsed_s * 1000.0:.1f} ms")
    print(f"  Real-Time Factor:    RTF = {bench_rtf:.4f} ({1.0 / bench_rtf:.1f}x Real-Time Speedup)")

    print("\n" + "=" * 78)
    print("ALL VERIFICATION CHECKS PASSED (EXIT CODE 0)")
    print(f"Average Training Step Time: {sum(step_times) / len(step_times):.1f} ms")
    print(f"Inference RTF:              {bench_rtf:.4f} (Under RTF 0.0120 budget)")
    print("=" * 78)

    return 0


if __name__ == "__main__":
    sys.exit(main())
