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
    phase_diff_features,
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


# ─── 6. State instrumentation (return_state / capture_state) ────────────────


def test_return_state_parity_and_shapes():
    rng = np.random.default_rng(42)
    B, T, N, D = 2, 17, 4, 8
    abar_mag, omega, pbar, k, v, lam_v, q = _random_inputs(rng, B, T, N, D)
    y_re0, y_im0 = _torch_scan(abar_mag, omega, pbar, k, v, lam_v, q, dtype=torch.float64)

    am, cw, sw, pb, kk, vv, lv, qq = (
        torch.as_tensor(a, dtype=torch.float64)
        for a in (abar_mag, np.cos(omega), np.sin(omega), pbar, k, v, lam_v, q)
    )
    y_re, y_im, state = complex_kla_scan(am, cw, sw, pb, kk, vv, lv, qq, return_state=True)
    # Default path must be byte-identical: same ops, same order.
    assert torch.equal(y_re, y_re0) and torch.equal(y_im, y_im0)
    for name in ("lam", "eta_re", "eta_im"):
        assert state[name].shape == (B, T, N, D)
    assert state["contrib"].shape == (B, T, N)
    assert (state["lam"] >= 0).all()
    # contrib_t[n] = q_t[n]·‖μ_t[n,:]‖ — recompute at the last step.
    mu_re = state["eta_re"][:, -1] / state["lam"][:, -1].clamp(min=1e-6)
    mu_im = state["eta_im"][:, -1] / state["lam"][:, -1].clamp(min=1e-6)
    expect = torch.as_tensor(q, dtype=torch.float64)[:, -1] * (mu_re**2 + mu_im**2).sum(-1).sqrt()
    np.testing.assert_allclose(state["contrib"][:, -1].numpy(), expect.numpy(), rtol=1e-10)


def test_layer_capture_state():
    torch.manual_seed(0)
    layer = ComplexKLALayer(32, n_state=8)
    x = torch.randn(2, 20, 32)
    with torch.no_grad():
        y_plain = layer(x)
        layer.capture = []
        layer.capture_state = True
        y_cap = layer(x)
    assert torch.equal(y_plain, y_cap)  # capture must not perturb the output
    cap = layer.capture[-1]
    assert cap["omega"].shape == (2, 20, 8)
    assert cap["lam"].shape == (2, 20, 8, 32)
    assert cap["eta_re"].shape == (2, 20, 8, 32)
    assert cap["eta_im"].shape == (2, 20, 8, 32)
    assert cap["contrib"].shape == (2, 20, 8)
    # omega is the pre-cos/sin phase: cos(omega) must equal the captured cos_w.
    np.testing.assert_allclose(np.cos(cap["omega"].numpy()), cap["cos_w"].numpy(), atol=1e-6)


# ─── 7. Rotation is wired ────────────────────────────────────────────────────


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


# ─── 8. Phase-differential readout ───────────────────────────────────────────


def test_default_readout_param_names_unchanged():
    """readout="complex_mean" must be the pre-change layer exactly: same
    parameter names, same mix shape (byte-identical default code path)."""
    torch.manual_seed(0)
    d_model, n_state = 32, 8
    layer = ComplexKLALayer(d_model, n_state=n_state)
    assert layer.readout == "complex_mean"
    expected = {
        "conv.weight",
        "conv.bias",
        "k_proj.weight",
        "q_proj.weight",
        "v_proj.weight",
        "lamv_proj.weight",
        "lamv_proj.bias",
        "omega_proj.weight",
        "omega_proj.bias",
        "s",
        "omega0",
        "mix.weight",
        "gate_proj.weight",
        "out_proj.weight",
        "norm.weight",
        "a_param",
        "p_param",
        "dt_param",
    }
    assert set(layer.state_dict().keys()) == expected
    assert layer.mix.weight.shape == (d_model, 2 * d_model)


def test_readout_rejects_unknown():
    with pytest.raises(ValueError, match="readout"):
        ComplexKLALayer(32, readout="argmax")


@pytest.mark.parametrize("readout,mix_mult", [("phase_diff", 4), ("phase_only", 2)])
def test_phase_readout_layer_forward_and_grads(readout, mix_mult):
    torch.manual_seed(0)
    d_model = 32
    layer = ComplexKLALayer(d_model, n_state=8, readout=readout)
    assert layer.mix.weight.shape == (d_model, mix_mult * d_model)
    x = torch.randn(2, 20, d_model, requires_grad=True)
    y = layer(x)
    assert y.shape == (2, 20, d_model)
    assert torch.isfinite(y).all()
    y.pow(2).mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    for name, p in layer.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), f"non-finite grad in {name}"


@pytest.mark.parametrize("readout", ["phase_diff", "phase_only"])
def test_phase_readout_model_end_to_end(readout):
    torch.manual_seed(0)
    model = SimpleConvV2CKLA(n_fft=256, hop_length=64, readout=readout)
    audio = torch.randn(2, 8000)
    out = model(audio)
    assert out.shape[:2] == (2, 4)
    assert torch.isfinite(out).all()
    out.pow(2).mean().backward()
    for name, p in model.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), f"non-finite grad in {name}"


def test_phase_diff_features_recovers_rotation_rate():
    """Feed a pure rotating phasor y_t = A·e^{iωt}: arg d must recover ω per
    step (and 0 at t = 0), mag must recover A."""
    B, T, D = 2, 30, 3
    omegas = torch.tensor([0.3, -1.1, 2.5])  # per-channel rotation rates
    amps = torch.tensor([1.0, 0.5, 3.0])
    t = torch.arange(T, dtype=torch.float32)[None, :, None]  # (1, T, 1)
    phase = omegas[None, None, :] * t
    y_re = (amps * torch.cos(phase)).expand(B, T, D)
    y_im = (amps * torch.sin(phase)).expand(B, T, D)

    arg_d, mag = phase_diff_features(y_re, y_im)
    assert arg_d.shape == (B, T, D) and mag.shape == (B, T, D)
    # d_0 = y_0·conj(y_0) ⇒ arg exactly 0.
    assert torch.equal(arg_d[:, 0], torch.zeros(B, D))
    # One-step differential recovers the (wrapped) rotation rate.
    wrapped = torch.atan2(torch.sin(omegas), torch.cos(omegas))
    torch.testing.assert_close(
        arg_d[:, 1:], wrapped[None, None, :].expand(B, T - 1, D), atol=1e-5, rtol=0
    )
    torch.testing.assert_close(mag, amps[None, None, :].expand(B, T, D), atol=1e-5, rtol=0)


def test_phase_diff_features_zero_input_is_finite():
    y = torch.zeros(1, 5, 4, requires_grad=True)
    arg_d, mag = phase_diff_features(y, y)
    assert torch.equal(arg_d, torch.zeros(1, 5, 4))  # atan2(0, 0) = 0
    # |y| is guarded by sqrt(|y|² + 1e−12): grad finite even at y = 0.
    # (arg d's grad at the exact origin is NaN by atan2's nature — measure
    # zero; the spec deliberately leaves it unguarded.)
    mag.sum().backward()
    grad = y.grad
    assert grad is not None and torch.isfinite(grad).all()


def test_registry_builds_phase_readout_variants():
    from models.registry import build_model

    for name, readout in [
        ("simple_conv_v2_ckla_phasediff", "phase_diff"),
        ("simple_conv_v2_ckla_phaseonly", "phase_only"),
    ]:
        m = build_model(name, n_fft=256, hop_length=64, num_rotors=4, p_init=1.0)
        assert isinstance(m, SimpleConvV2CKLA)
        for mixer in _mixers(m):
            assert mixer.readout == readout
