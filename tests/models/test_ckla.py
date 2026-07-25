"""Tests for the CKLA (Complex Kalman Linear Attention) layer and model.

Ground truth is an *independent* NumPy complex128 implementation of the
recursion in docs/ckla-design.md §1 (direct complex arithmetic, written from
the design doc — not from the torch op), plus a NumPy port of the real-KLA
``flat_step`` (fkla reference, arXiv 2602.10743) for the zero-rotation
equivalence check.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from models.ckla import (
    ComplexKLALayer,
    SimpleConvV2CKLA,
    TemporalCKLAHead,
    complex_kla_scan,
)
from models.rps_predictor import SimpleConvV2Transformer

# ─── NumPy fp64 references ───────────────────────────────────────────────────


def _ref_complex_scan(abar_mag, omega, pbar, k, v, lam_v, q, eps=1e-6):
    """Design §1 recursion in direct complex128 arithmetic."""
    B, T, _ = omega.shape
    D = v.shape[-1]
    N = abar_mag.shape[0]
    eta = np.zeros((B, N, D), dtype=np.complex128)
    lam = np.zeros((B, N, D), dtype=np.float64)
    ys = np.zeros((B, T, D), dtype=np.complex128)
    for t in range(T):
        abar = abar_mag[None] * np.exp(1j * omega[:, t, :, None])  # (B, N, D)
        den = abar_mag[None] ** 2 + pbar[None] * lam
        phi = k[:, t, :, None] ** 2 * lam_v[:, t, None, :]
        kappa = k[:, t, :, None] * lam_v[:, t, None, :] * v[:, t, None, :]
        eta = abar * eta / den + kappa
        lam = lam / den + phi
        mu = eta / np.maximum(lam, eps)
        ys[:, t] = np.einsum("bn,bnd->bd", q[:, t], mu)
    return ys.real, ys.imag


def _flat_step(eta, lam, abar, pbar, phi, kappa):
    """Real-KLA flat recursion — NumPy port of fkla.reference.flat_step."""
    den = abar**2 + pbar * lam
    return abar * eta / den + kappa, lam / den + phi


def _random_inputs(rng, B, T, N, D):
    gamma = rng.uniform(0.05, 3.0, size=(N, D))
    dt = rng.uniform(0.005, 0.5, size=(N, D))
    abar_mag = np.exp(-gamma * dt)
    pbar = rng.uniform(1e-3, 0.5, size=(N, D))
    k = rng.standard_normal((B, T, N))
    q = rng.standard_normal((B, T, N))
    v = rng.standard_normal((B, T, D))
    lam_v = rng.uniform(0.01, 2.0, size=(B, T, D))
    omega = rng.uniform(-np.pi, np.pi, size=(B, T, N))
    return abar_mag, omega, pbar, k, v, lam_v, q


def _torch_scan(abar_mag, omega, pbar, k, v, lam_v, q, dtype):
    am, cw, sw, pb, kk, vv, lv, qq = (
        torch.as_tensor(a, dtype=dtype)
        for a in (abar_mag, np.cos(omega), np.sin(omega), pbar, k, v, lam_v, q)
    )
    return complex_kla_scan(am, cw, sw, pb, kk, vv, lv, qq)


def _mixers(model: SimpleConvV2CKLA) -> list[ComplexKLALayer]:
    """Typed accessor for the head's mixer layers (pyright-friendly)."""
    out = []
    for blk in model.head.blocks:
        mixer = blk.mixer
        assert isinstance(mixer, ComplexKLALayer)
        out.append(mixer)
    return out


# ─── 1. Property test against the complex reference ─────────────────────────


@pytest.mark.parametrize("B", [1, 2])
@pytest.mark.parametrize("T", [1, 5, 40])
@pytest.mark.parametrize("N", [1, 4])
@pytest.mark.parametrize("D", [1, 8])
def test_scan_matches_complex128_reference(B, T, N, D):
    rng = np.random.default_rng(1000 * B + 100 * T + 10 * N + D)
    inputs = _random_inputs(rng, B, T, N, D)
    ref_re, ref_im = _ref_complex_scan(*inputs)

    # fp64: the op must reproduce the reference to numerical identity.
    y_re, y_im = _torch_scan(*inputs, dtype=torch.float64)
    np.testing.assert_allclose(y_re.numpy(), ref_re, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(y_im.numpy(), ref_im, rtol=1e-10, atol=1e-12)

    # fp32 (the training dtype): ~1e-5 relative over the recursion depth.
    y_re, y_im = _torch_scan(*inputs, dtype=torch.float32)
    np.testing.assert_allclose(y_re.numpy(), ref_re, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(y_im.numpy(), ref_im, rtol=1e-4, atol=1e-5)


# ─── 2. Zero-rotation equivalence with real KLA ──────────────────────────────


def test_zero_rotation_matches_real_kla():
    rng = np.random.default_rng(7)
    B, T, N, D = 2, 25, 4, 8
    abar_mag, _, pbar, k, v, lam_v, q = _random_inputs(rng, B, T, N, D)
    omega = np.zeros((B, T, N))

    eta = np.zeros((B, N, D))
    lam = np.zeros((B, N, D))
    ys = np.zeros((B, T, D))
    for t in range(T):
        phi = k[:, t, :, None] ** 2 * lam_v[:, t, None, :]
        kappa = k[:, t, :, None] * lam_v[:, t, None, :] * v[:, t, None, :]
        eta, lam = _flat_step(eta, lam, abar_mag[None], pbar[None], phi, kappa)
        mu = eta / np.maximum(lam, 1e-6)
        ys[:, t] = np.einsum("bn,bnd->bd", q[:, t], mu)

    y_re, y_im = _torch_scan(abar_mag, omega, pbar, k, v, lam_v, q, dtype=torch.float64)
    np.testing.assert_allclose(y_re.numpy(), ys, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(y_im.numpy(), 0.0, atol=1e-12)


# ─── 3. gradcheck ────────────────────────────────────────────────────────────


def test_scan_gradcheck():
    torch.manual_seed(0)
    B, T, N, D = 1, 3, 2, 2
    abar_mag = torch.rand(N, D, dtype=torch.float64) * 0.8 + 0.1
    ang = torch.rand(B, T, N, dtype=torch.float64) * 2 * math.pi
    cos_w, sin_w = torch.cos(ang), torch.sin(ang)
    pbar = torch.rand(N, D, dtype=torch.float64) * 0.5 + 0.01
    k = torch.randn(B, T, N, dtype=torch.float64)
    v = torch.randn(B, T, D, dtype=torch.float64)
    lam_v = torch.rand(B, T, D, dtype=torch.float64) + 0.1
    q = torch.randn(B, T, N, dtype=torch.float64)
    inputs = tuple(t.requires_grad_(True) for t in (abar_mag, cos_w, sin_w, pbar, k, v, lam_v, q))
    assert torch.autograd.gradcheck(complex_kla_scan, inputs, atol=1e-6)


# ─── 4. Layer forward ────────────────────────────────────────────────────────


def test_layer_forward_shape():
    torch.manual_seed(0)
    layer = ComplexKLALayer(32, n_state=8)
    x = torch.randn(2, 20, 32)
    y = layer(x)
    assert y.shape == (2, 20, 32)
    assert y.dtype == torch.float32
    assert torch.isfinite(y).all()


def test_layer_scan_runs_fp32_under_bf16_autocast():
    torch.manual_seed(0)
    layer = ComplexKLALayer(32, n_state=8)
    layer.capture = []
    x = torch.randn(2, 20, 32)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        # Probe: only meaningful if autocast actually downcasts linears here.
        if layer.k_proj(x).dtype != torch.bfloat16:
            pytest.skip("CPU bf16 autocast inactive on this torch build")
        y = layer(x)
    cap = layer.capture[-1]
    for name in ("k", "q", "v", "lam_v", "cos_w", "sin_w"):
        assert cap[name].dtype == torch.float32, f"{name} not cast to fp32"
    assert cap["abar_mag"].dtype == torch.float32
    assert torch.isfinite(y.float()).all()


def test_head_shape_contract():
    torch.manual_seed(0)
    head = TemporalCKLAHead(in_ch=128, d_model=64, num_rotors=4, n_layers=2, n_state=8)
    x = torch.randn(2, 128, 30)
    y = head(x)
    assert y.shape == (2, 4, 30)


# ─── 5. Model end-to-end ─────────────────────────────────────────────────────


def test_model_end_to_end_and_param_budget():
    torch.manual_seed(0)
    model = SimpleConvV2CKLA()
    audio = torch.randn(2, 64000)
    out = model(audio)
    assert out.shape == (2, 4, 126)

    loss = out.pow(2).mean()
    loss.backward()
    for name, p in model.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), f"non-finite grad in {name}"
    for mixer in _mixers(model):
        # omega_proj is zero-init, but its grad flows through s ⊙ Wh; s must
        # receive gradient (rotation path connected).
        assert mixer.s.grad is not None
        assert mixer.omega_proj.weight.grad is not None

    n_ckla = sum(p.numel() for p in model.parameters())
    ref = SimpleConvV2Transformer()
    n_ref = sum(p.numel() for p in ref.parameters())
    head_ckla = sum(p.numel() for p in model.head.parameters())
    head_ref = sum(p.numel() for p in ref.head.parameters())
    print(
        f"params: CKLA full {n_ckla} vs transformer {n_ref} "
        f"(ratio {n_ckla / n_ref:.3f}); heads {head_ckla} vs {head_ref}"
    )
    # 1841552 vs 1476400 → ratio 1.247: within the ±25% budget (deterministic).
    assert abs(n_ckla - n_ref) / n_ref <= 0.25


def test_registry_builds_both_variants():
    from models.registry import build_model

    m_if = build_model("simple_conv_v2_ckla", n_fft=256, hop_length=64, num_rotors=4)
    assert isinstance(m_if, SimpleConvV2CKLA)
    assert m_if.frontend.out_channels == 2  # stft_mag_if
    m_mag = build_model("simple_conv_v2_ckla_mag", n_fft=256, hop_length=64, num_rotors=4)
    assert isinstance(m_mag, SimpleConvV2CKLA)
    assert m_mag.frontend.out_channels == 1  # stft_mag


# ─── 6. Rotation is wired ────────────────────────────────────────────────────


def test_rotation_shift_changes_output():
    torch.manual_seed(0)
    model = SimpleConvV2CKLA(n_fft=256, hop_length=64)
    model.eval()
    audio = torch.randn(1, 8000)
    with torch.no_grad():
        y1 = model(audio)
        for mixer in _mixers(model):
            mixer.omega0.add_(2.0)
        y2 = model(audio)
    assert (y1 - y2).abs().max().item() > 1e-5
