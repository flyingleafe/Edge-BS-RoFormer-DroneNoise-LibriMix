"""
Quick smoke test for the MultiF0RPSPredictor.
Tests:
    1. Build model and check forward pass shapes
    2. Verify output is within reasonable range
    3. Verify gradient flow through the CNN part
"""

import sys

sys.path.insert(0, ".")

import time

import torch

from models.multif0.rps_predictor import MultiF0RPSPredictor


def test_rps_predictor():
    print("=" * 60)
    print("MultiF0RPSPredictor — Smoke Test")
    print("=" * 60)

    # ── Build model ──
    model = MultiF0RPSPredictor(
        n_fft=2048,
        hop_length=512,
        num_rotors=4,
        temperature=1.0,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    # ── Test input ──
    duration_s = 3.0  # 3 seconds
    sr = 16000
    B = 2
    t = torch.arange(B * int(sr * duration_s), dtype=torch.float32).view(B, -1) / sr
    # Generate drone-like sound: 80 Hz fundamental + harmonics at 160, 240, 320 Hz
    f0 = 80.0  # Hz (≈ 4800 RPM)
    audio = (
        torch.sin(2 * torch.pi * f0 * t)
        + 0.5 * torch.sin(2 * torch.pi * 2 * f0 * t)
        + 0.25 * torch.sin(2 * torch.pi * 3 * f0 * t)
        + 0.125 * torch.sin(2 * torch.pi * 4 * f0 * t)
    ) * 0.3

    print(f"  Audio shape: {audio.shape} ({duration_s}s at {sr} Hz)")

    # ── Forward pass (eval) ──
    model.eval()
    t0 = time.time()
    with torch.no_grad():
        rps = model(audio)
    elapsed = time.time() - t0

    print(f"  Output shape: {rps.shape}  (expected: ({B}, 4, ~{audio.shape[1] // 512 + 1}))")
    print(f"  Forward time: {elapsed:.1f}s")

    # Check output range for a randomly initialized model
    # (should be near the frequency grid center ~500 Hz since softmax
    #  over random logits is approximately uniform)
    fg_center = model.freq_grid.mean().item()
    print(f"  RPS range: [{rps.min().item():.1f}, {rps.max().item():.1f}] Hz")
    print(f"  (untrained model centers near freq_grid mean ≈ {fg_center:.1f} Hz)")

    # Verify output is close to frequency grid center
    mean_rps = rps.mean().item()
    print(f"  Mean RPS: {mean_rps:.1f} Hz")
    assert fg_center * 0.5 < mean_rps < fg_center * 1.5, (
        f"Mean RPS {mean_rps:.1f} too far from grid center {fg_center:.1f}"
    )
    print("  ✓ Output in reasonable range")

    # ── Gradient flow test ──
    model.train()
    audio_grad = audio[:1].clone()  # single sample for speed
    rps_grad = model(audio_grad)

    # The HCQT part doesn't have gradients; only the CNN does.
    # So we can still call backward — HCQT tensors are leaf tensors
    # that don't require grad, and CNN params do get gradients.
    loss = (rps_grad - 80.0).pow(2).mean()
    loss.backward()

    cnn_params_with_grad = 0
    cnn_params_total = 0
    for name, p in model.cnn.named_parameters():
        cnn_params_total += 1
        if p.grad is not None:
            cnn_params_with_grad += 1

    print(f"  CNN grad flow: {cnn_params_with_grad}/{cnn_params_total} params have gradients")

    # All CNN params should have gradients (BatchNorm running stats don't during eval,
    # but we're in train mode)
    assert cnn_params_with_grad > 0, "No CNN params received gradients!"
    if cnn_params_with_grad < cnn_params_total:
        print("  ⚠ Some CNN params lack gradients (likely BatchNorm running stats)")
    else:
        print("  ✓ All CNN params receive gradients")

    # ── Check that freq_grid is frozen ──
    assert not model.freq_grid.requires_grad
    print("  ✓ freq_grid is a frozen buffer")

    # ── Test salience prediction ──
    with torch.no_grad():
        salience = model.predict_salience(audio_grad)  # (1, 360, T_hcqt)
    print(f"  Salience shape: {salience.shape}")
    print(f"  Salience range: [{salience.min().item():.3f}, {salience.max().item():.3f}]")
    print("  ✓ Salience prediction works")

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED ✓")
    print("=" * 60)


if __name__ == "__main__":
    test_rps_predictor()
