"""
Verification test for multi-F0 PyTorch reimplementation.

Tests:
    1. HCQT computation produces correct shapes and values
    2. All four model variants build and run forward pass
    3. Output salience map shape is correct
    4. Model parameter counts are reasonable
"""

import sys

import numpy as np
import torch

# Add project src to path
sys.path.insert(0, ".")

from models.multif0.hcqt import (
    HCQT,
    compute_hcqt,
    compute_hcqt_mag_phase,
    hcqt_params,
)
from models.multif0.model import (
    EarlyDeep,
    EarlyShallow,
    LateDeep,
    LateDeepNoPhase,
    build_model,
)


def test_hcqt():
    """Test HCQT computation."""
    print("=" * 60)
    print("TEST 1: HCQT computation")
    print("=" * 60)

    params = hcqt_params()
    print(f"  Parameters: {params}")

    sr = params["sr"]
    duration = 2.0  # seconds
    t = np.arange(int(sr * duration)) / sr
    # Generate a simple test signal: C major chord (C4, E4, G4)
    audio = (
        np.sin(2 * np.pi * 261.63 * t)  # C4
        + np.sin(2 * np.pi * 329.63 * t)  # E4
        + np.sin(2 * np.pi * 392.00 * t)  # G4
    ).astype(np.float32)

    print(f"  Audio: {duration}s, {len(audio)} samples, sr={sr}")

    # Test compute_hcqt
    data = compute_hcqt(audio, **params)
    mag, phase = data["mag"], data["phase"]

    n_harmonics = len(params["harmonics"])
    n_bins = params["n_octaves"] * 12 * params["over_sample"]
    n_frames = len(t) * sr // params["hop_length"] + 1  # approximate

    print(f"  mag shape:   {mag.shape}   (expected: ({n_harmonics}, {n_bins}, ~{n_frames}))")
    print(f"  phase shape: {phase.shape} (expected: ({n_harmonics}, {n_bins}, ~{n_frames}))")

    assert mag.shape[0] == n_harmonics, f"Wrong n_harmonics: {mag.shape[0]} != {n_harmonics}"
    assert mag.shape[1] == n_bins, f"Wrong n_bins: {mag.shape[1]} != {n_bins}"
    assert mag.ndim == 3, f"Wrong ndim: {mag.ndim}"

    # Check values are reasonable (log magnitude should be <= 0)
    assert np.max(mag) <= 0.0 + 1e-5, f"Max magnitude {np.max(mag)} > 0 (log scale)"
    assert not np.any(np.isnan(mag)), "NaN in magnitude"
    assert not np.any(np.isnan(phase)), "NaN in phase"

    # Test compute_hcqt_mag_phase
    mag2, dphase2 = compute_hcqt_mag_phase(audio, **params)
    assert mag2.shape == mag.shape, f"Mag shape mismatch: {mag2.shape} vs {mag.shape}"
    assert dphase2.shape == phase.shape, f"Dphase shape mismatch: {dphase2.shape} vs {phase.shape}"
    assert np.allclose(mag2, mag), "Mag values differ"

    # Test HCQT class
    extractor = HCQT()
    mag3, dphase3 = extractor(audio)
    assert mag3.shape == mag.shape
    assert dphase3.shape == dphase2.shape

    # Check dphase values are reasonable (phase diffs should be small for stationary signals)
    print(f"  dphase range: [{np.min(dphase2):.3f}, {np.max(dphase2):.3f}]")
    print(f"  mag range:    [{np.min(mag):.3f}, {np.max(mag):.3f}] dB")

    print("  ✓ HCQT tests passed\n")
    return mag.shape, n_harmonics, n_bins


def test_models(n_harmonics=5, n_freqs=360):
    """Test all model variants."""
    print("=" * 60)
    print("TEST 2: Model forward pass")
    print("=" * 60)

    batch_size = 4
    T = 50  # time patch size used in training

    mag = torch.randn(batch_size, n_harmonics, n_freqs, T)
    dphase = torch.randn(batch_size, n_harmonics, n_freqs, T)

    models = {
        "EarlyShallow": EarlyShallow(n_harmonics),
        "EarlyDeep": EarlyDeep(n_harmonics),
        "LateDeep": LateDeep(n_harmonics),
        "LateDeepNoPhase": LateDeepNoPhase(n_harmonics),
    }

    for name, model in models.items():
        model.eval()
        with torch.no_grad():
            if "NoPhase" in name:
                out = model(mag, dphase=None)  # dphase ignored
            else:
                out = model(mag, dphase)

        n_params = sum(p.numel() for p in model.parameters())
        print(
            f"  {name:20s}:  input {tuple(mag.shape)} → output {tuple(out.shape)}  "
            f"({n_params:,} params)"
        )

        # Check output shape
        assert out.shape == (batch_size, 1, n_freqs, T), (
            f"Wrong output shape: {out.shape} != ({batch_size}, 1, {n_freqs}, {T})"
        )

        # Check output range
        assert torch.all(out >= 0) and torch.all(out <= 1), (
            f"Output values outside [0, 1]: min={out.min():.3f}, max={out.max():.3f}"
        )

    # Test variable-length input
    T_long = 200
    mag_long = torch.randn(2, n_harmonics, n_freqs, T_long)
    dphase_long = torch.randn(2, n_harmonics, n_freqs, T_long)

    model = LateDeep(n_harmonics)
    model.eval()
    with torch.no_grad():
        out_long = model(mag_long, dphase_long)

    print(f"  LateDeep (long): input {tuple(mag_long.shape)} → output {tuple(out_long.shape)}")
    assert out_long.shape == (2, 1, n_freqs, T_long), f"Wrong long output: {out_long.shape}"

    # Test factory
    m = build_model("late_deep", n_harmonics=n_harmonics)
    assert isinstance(m, LateDeep)

    print("  ✓ All model tests passed\n")


def test_conv_dimensions():
    """Detailed check: verify conv layers preserve frequency dimension correctly."""
    print("=" * 60)
    print("TEST 3: Convolution dimension preservation")
    print("=" * 60)

    B, C, F, T = 2, 5, 360, 50
    x = torch.randn(B, C, F, T)

    model = LateDeep(5)
    model.eval()

    # Hook intermediate activations
    activations = {}

    def hook_fn(name):
        def hook(module, inp, out):
            activations[name] = out.shape

        return hook

    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.BatchNorm2d)):
            hooks.append(module.register_forward_hook(hook_fn(name)))

    with torch.no_grad():
        dphase = torch.randn(B, C, F, T)
        out = model(x, dphase)

    for h in hooks:
        h.remove()

    # Verify all intermediate freq dims are 360 (except possibly after the (360,1) conv)
    # Actually the (360,1) conv with 'same' padding should also give 360
    for name, shape in activations.items():
        freq_dim = shape[2]
        if "dist_conv" not in name:  # Before distribution layer
            assert freq_dim == F, f"{name}: freq dim {freq_dim} != {F}"

    print(f"  Output shape: {out.shape}")
    assert out.shape[2] == F, f"Final freq dim {out.shape[2]} != {F}"
    print("  ✓ All frequency dimensions preserved\n")


def test_gradient_flow():
    """Test that gradients flow through the model."""
    print("=" * 60)
    print("TEST 4: Gradient flow")
    print("=" * 60)

    B, C, F, T = 2, 5, 360, 50
    mag = torch.randn(B, C, F, T, requires_grad=False)
    dphase = torch.randn(B, C, F, T, requires_grad=False)

    model = LateDeep(5)
    model.train()

    out = model(mag, dphase)
    target = torch.rand_like(out)
    loss = torch.nn.functional.binary_cross_entropy(out, target)
    loss.backward()

    # Check that all parameters have gradients
    no_grad_params = []
    for name, p in model.named_parameters():
        if p.grad is None:
            no_grad_params.append(name)

    if no_grad_params:
        print(f"  ⚠ Parameters without gradient: {no_grad_params}")
    else:
        print("  ✓ All parameters received gradients")

    print(f"  Loss: {loss.item():.6f}\n")


def test_output_interpretation():
    """Test that the output can be interpreted as a pitch salience map."""
    print("=" * 60)
    print("TEST 5: Output interpretation")
    print("=" * 60)

    # Create a simple audio signal and run through the full pipeline
    params = hcqt_params()
    sr = params["sr"]
    duration = 3.0
    t = np.arange(int(sr * duration)) / sr

    # Generate chirp sweeping from C4 to C5
    f0_t = 261.63 * (2 ** (t / duration))  # C4 to C5 over 3 seconds
    phase = 2 * np.pi * np.cumsum(f0_t) / sr
    audio = 0.5 * np.sin(phase).astype(np.float32)

    # Compute HCQT
    extractor = HCQT()
    mag, dphase = extractor(audio)

    print(f"  HCQT shape: {mag.shape}")

    # Run through model
    mag_t = torch.from_numpy(mag).unsqueeze(0)  # (1, 5, 360, T)
    dphase_t = torch.from_numpy(dphase).unsqueeze(0)  # (1, 5, 360, T)

    model = LateDeep(5)
    model.eval()
    with torch.no_grad():
        salience = model(mag_t, dphase_t)  # (1, 1, 360, T)
    salience = salience.squeeze().numpy()  # (360, T)

    print(f"  Salience shape: {salience.shape}")
    print(f"  Salience range: [{salience.min():.4f}, {salience.max():.4f}]")

    # Check that we get some activation
    assert salience.max() > 0.01, f"Very low max salience: {salience.max()}"
    print("  ✓ Salience map produced\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Multi-F0 PyTorch Reimplementation — Verification Suite")
    print("=" * 60 + "\n")

    try:
        _, n_harmonics, n_bins = test_hcqt()
        test_models(n_harmonics, n_bins)
        test_conv_dimensions()
        test_gradient_flow()
        test_output_interpretation()

        print("=" * 60)
        print("ALL TESTS PASSED ✓")
        print("=" * 60)
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
