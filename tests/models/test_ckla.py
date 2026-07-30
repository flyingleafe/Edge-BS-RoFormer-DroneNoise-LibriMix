"""Tests for the CKLA (Complex Kalman Linear Attention) layer and model.

Ground truth is an *independent* NumPy complex128 implementation of the
recursion in docs/ckla-design.md §1 (direct complex arithmetic, written from
the design doc — not from the torch op), plus a NumPy port of the real-KLA
``flat_step`` (fkla reference, arXiv 2602.10743) for the zero-rotation
equivalence check.
"""

from __future__ import annotations

import importlib
import math
import os

import numpy as np
import pytest
import torch

from models.ckla import (
    ComplexKLALayer,
    SimpleConvV2CKLA,
    TemporalCKLAHead,
    complex_kla_scan,
    complex_kla_scan_parallel,
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
    # capture_state routes through the sequential reference scan; pin the
    # plain forward to it too so the no-perturbation check stays byte-exact.
    layer.use_parallel_scan = False
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


# ─── 9. Parallel (associative-scan) path ─────────────────────────────────────


def _edge_inputs(rng, B, T, N, D):
    """Random inputs seeded with the adversarial structure the parallel scan
    must survive: whole zero-φ steps (k = 0), p̄ near 0, |ā| near 1."""
    abar_mag, omega, pbar, k, v, lam_v, q = _random_inputs(rng, B, T, N, D)
    abar_mag[: max(1, N // 3)] = 1.0 - 1e-6  # near-unit decay
    pbar[N // 2 :] = 1e-12  # near-zero process noise
    k[:, ::3, :] = 0.0  # φ = κ = 0 on every third step
    return abar_mag, omega, pbar, k, v, lam_v, q


def _torch_inputs(
    np_inputs, dtype, requires_grad=False
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """(abar_mag, omega, pbar, k, v, lam_v, q) → the 8 scan tensor args."""
    abar_mag, omega, pbar, k, v, lam_v, q = np_inputs
    am, cw, sw, pb, kk, vv, lv, qq = (
        torch.as_tensor(a, dtype=dtype)
        for a in (abar_mag, np.cos(omega), np.sin(omega), pbar, k, v, lam_v, q)
    )
    if requires_grad:
        am, cw, sw, pb, kk, vv, lv, qq = (
            t.clone().requires_grad_(True) for t in (am, cw, sw, pb, kk, vv, lv, qq)
        )
    return am, cw, sw, pb, kk, vv, lv, qq


@pytest.mark.parametrize("T", [1, 2, 3, 31, 250])
def test_parallel_scan_matches_sequential_forward(T):
    rng = np.random.default_rng(123 + T)
    B, N, D = 3, 16, 32
    inputs = _edge_inputs(rng, B, T, N, D)
    for dtype in (torch.float64, torch.float32):
        tens = _torch_inputs(inputs, dtype)
        y_re_s, y_im_s = complex_kla_scan(*tens)
        y_re_p, y_im_p = complex_kla_scan_parallel(*tens)
        np.testing.assert_allclose(y_re_p.numpy(), y_re_s.numpy(), rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(y_im_p.numpy(), y_im_s.numpy(), rtol=1e-4, atol=1e-5)


_GRAD_NAMES = ("abar_mag", "cos_w", "sin_w", "pbar", "k", "v", "lam_v", "q")


@pytest.mark.parametrize("T", [3, 31, 250])
def test_parallel_scan_grads_match_sequential(T):
    rng = np.random.default_rng(31 + T)
    B, N, D = 3, 16, 32
    inputs = _edge_inputs(rng, B, T, N, D)
    w_re = rng.standard_normal((B, T, D))
    w_im = rng.standard_normal((B, T, D))

    def grads(fn, dtype):
        tens = _torch_inputs(inputs, dtype, requires_grad=True)
        y_re, y_im = fn(*tens)
        wr = torch.as_tensor(w_re, dtype=dtype)
        wi = torch.as_tensor(w_im, dtype=dtype)
        ((y_re * wr).mean() + (y_im * wi).mean()).backward()
        out = []
        for t in tens:
            assert t.grad is not None
            out.append(t.grad.double().numpy())
        return out

    # fp64 (strictest case): elementwise parallel == sequential.
    truth = grads(complex_kla_scan, torch.float64)
    par64 = grads(complex_kla_scan_parallel, torch.float64)
    for name, g_s, g_p in zip(_GRAD_NAMES, truth, par64):
        np.testing.assert_allclose(
            g_p, g_s, rtol=1e-3, atol=1e-5, err_msg=f"fp64 grad mismatch: {name}"
        )

    # fp32 (the training dtype): elementwise comparison of two fp32 graphs is
    # dominated by summation-order rounding on near-zero entries, so measure
    # each implementation against the fp64 truth instead — the parallel
    # scan's rounding error must stay within a small factor of the
    # sequential loop's own (measured ≤ ~2.3× across inputs and T).
    seq32 = grads(complex_kla_scan, torch.float32)
    par32 = grads(complex_kla_scan_parallel, torch.float32)
    for name, g_t, g_s, g_p in zip(_GRAD_NAMES, truth, seq32, par32):
        err_seq = np.abs(g_s - g_t).max()
        err_par = np.abs(g_p - g_t).max()
        scale = np.abs(g_t).max()
        assert err_par <= max(4 * err_seq, 1e-6 * scale, 1e-7), (
            f"fp32 grad {name}: parallel err {err_par:.3e} vs sequential err {err_seq:.3e} "
            f"(scale {scale:.3e})"
        )


def test_parallel_return_state_falls_back_to_sequential():
    rng = np.random.default_rng(5)
    inputs = _edge_inputs(rng, 2, 17, 4, 8)
    tens = _torch_inputs(inputs, torch.float64)
    y_re_s, y_im_s, st_s = complex_kla_scan(*tens, return_state=True)
    y_re_p, y_im_p, st_p = complex_kla_scan_parallel(*tens, return_state=True)
    assert torch.equal(y_re_p, y_re_s) and torch.equal(y_im_p, y_im_s)
    for name in ("lam", "eta_re", "eta_im", "contrib"):
        assert torch.equal(st_p[name], st_s[name])


def test_layer_parallel_optin_and_env(monkeypatch):
    # Parallel scan is opt-in (span-product blowup on real batches — see the
    # attribute note in ComplexKLALayer.__init__); sequential is the default.
    assert ComplexKLALayer(16, n_state=4).use_parallel_scan is False
    monkeypatch.setenv("CKLA_PARALLEL_SCAN", "1")
    assert ComplexKLALayer(16, n_state=4).use_parallel_scan is True


def test_head_parallel_matches_sequential_path():
    torch.manual_seed(0)
    head = TemporalCKLAHead(in_ch=64, d_model=64, num_rotors=4, n_layers=2, n_state=8)
    x = torch.randn(2, 64, 50)
    with torch.no_grad():
        y_seq = head(x)  # default: sequential scan (CPU input)
        for blk in head.blocks:
            mixer = blk.mixer
            assert isinstance(mixer, ComplexKLALayer)
            assert not mixer.use_parallel_scan
            mixer.use_parallel_scan = True
        y_par = head(x)
    torch.testing.assert_close(y_par, y_seq, rtol=1e-4, atol=1e-5)


def test_parallel_scan_span_overflow_guard_finite():
    """Small ā with tiny-nonzero k: span-gain products overflow fp32 in a naive
    associative scan (the sequential loop never materializes them). The combine's
    1e10 magnitude caps must keep the parallel output finite; outputs are allowed
    to differ from sequential here (both are in a diverged regime)."""
    torch.manual_seed(0)
    B, T, N, D = 2, 250, 8, 16
    cos_w, sin_w = torch.ones(B, T, N), torch.zeros(B, T, N)
    v = torch.randn(B, T, D)
    lam_v = torch.rand(B, T, D) + 0.5
    q = torch.randn(B, T, N)
    for a_val in (0.3, 0.5, 0.7):
        a = torch.full((N, D), a_val)
        p = torch.full((N, D), 1e-12)
        k = torch.rand(B, T, N) * 1e-15
        y_re, y_im = complex_kla_scan_parallel(a, cos_w, sin_w, p, k, v, lam_v, q)
        assert torch.isfinite(y_re).all() and torch.isfinite(y_im).all()
        # k exactly zero: sequential is exactly zero and parallel must match it.
        kz = torch.zeros(B, T, N)
        yz_re, yz_im = complex_kla_scan_parallel(a, cos_w, sin_w, p, kz, v, lam_v, q)
        assert torch.equal(yz_re, torch.zeros_like(yz_re))
        assert torch.equal(yz_im, torch.zeros_like(yz_im))


# ─── 10. Fused Triton op (CPU interpreter + CUDA) ────────────────────────────


def _load_ckla_triton(interpret: bool):
    """(Re)import ``models.ckla_triton`` with TRITON_INTERPRET set/cleared.

    Triton picks interpreter vs compiled mode at ``@triton.jit`` decoration
    time, so the env var must be in place before the module (re)import.
    """
    if interpret:
        os.environ["TRITON_INTERPRET"] = "1"
    else:
        os.environ.pop("TRITON_INTERPRET", None)
    import models.ckla_triton as mod

    return importlib.reload(mod)


def _interp_probe_reason() -> str | None:
    """None iff the triton CPU interpreter can run the fused op end-to-end."""
    try:
        mod = _load_ckla_triton(interpret=True)
        if not mod.HAS_TRITON:
            return "triton not installed"
        n, d = 2, 3
        y_re, y_im = mod.complex_kla_scan_triton(
            torch.full((n, d), 0.5),
            torch.ones(1, 3, n),
            torch.zeros(1, 3, n),
            torch.full((n, d), 0.1),
            torch.ones(1, 3, n),
            torch.ones(1, 3, d),
            torch.ones(1, 3, d),
            torch.ones(1, 3, n),
            BD=4,
            BT=2,
        )
        if not (torch.isfinite(y_re).all() and torch.isfinite(y_im).all()):
            return "triton CPU interpreter produced non-finite output"
        return None
    except Exception as exc:  # pragma: no cover - env-dependent escape hatch
        return f"triton CPU interpreter unavailable: {exc!r}"


_TRITON_INTERP_SKIP = _interp_probe_reason()
_interp = pytest.mark.skipif(_TRITON_INTERP_SKIP is not None, reason=str(_TRITON_INTERP_SKIP))

# B=2, N=8, D=20 with BD=8 exercises D-masking (20 = 2·8 + 4); BT=4 with
# T ∈ {1, 5, 31} exercises T-chunk padding in both kernels.
_TRITON_B, _TRITON_N, _TRITON_D = 2, 8, 20


@_interp
@pytest.mark.parametrize("T", [1, 5, 31])
def test_triton_interp_forward_matches_sequential(T):
    mod = _load_ckla_triton(interpret=True)
    rng = np.random.default_rng(9000 + T)
    inputs = _random_inputs(rng, _TRITON_B, T, _TRITON_N, _TRITON_D)
    tens = _torch_inputs(inputs, torch.float32)
    y_re_s, y_im_s = complex_kla_scan(*tens)
    y_re_t, y_im_t = mod.complex_kla_scan_triton(*tens, BD=8, BT=4)
    np.testing.assert_allclose(y_re_t.numpy(), y_re_s.numpy(), rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(y_im_t.numpy(), y_im_s.numpy(), rtol=1e-4, atol=1e-5)


@_interp
@pytest.mark.parametrize("T", [1, 5, 31])
def test_triton_interp_grads_match_sequential(T):
    mod = _load_ckla_triton(interpret=True)
    rng = np.random.default_rng(9100 + T)
    inputs = _random_inputs(rng, _TRITON_B, T, _TRITON_N, _TRITON_D)
    w_re = torch.as_tensor(rng.standard_normal((_TRITON_B, T, _TRITON_D)), dtype=torch.float32)
    w_im = torch.as_tensor(rng.standard_normal((_TRITON_B, T, _TRITON_D)), dtype=torch.float32)

    def grads(fn, **kw):
        tens = _torch_inputs(inputs, torch.float32, requires_grad=True)
        y_re, y_im = fn(*tens, **kw)
        ((y_re * w_re).sum() + (y_im * w_im).sum()).backward()
        out = []
        for t in tens:
            assert t.grad is not None
            out.append(t.grad.numpy())
        return out

    g_seq = grads(complex_kla_scan)
    g_tri = grads(mod.complex_kla_scan_triton, BD=8, BT=4)
    for name, g_s, g_t in zip(_GRAD_NAMES, g_seq, g_tri):
        np.testing.assert_allclose(
            g_t, g_s, rtol=1e-3, atol=1e-5, err_msg=f"triton grad mismatch: {name}"
        )


@_interp
def test_triton_interp_zero_k_gives_exact_zero_y():
    mod = _load_ckla_triton(interpret=True)
    rng = np.random.default_rng(9200)
    abar_mag, omega, pbar, k, v, lam_v, q = _random_inputs(rng, _TRITON_B, 9, _TRITON_N, _TRITON_D)
    k = np.zeros_like(k)
    tens = _torch_inputs((abar_mag, omega, pbar, k, v, lam_v, q), torch.float32)
    y_re, y_im = mod.complex_kla_scan_triton(*tens, BD=8, BT=4)
    assert torch.equal(y_re, torch.zeros_like(y_re))
    assert torch.equal(y_im, torch.zeros_like(y_im))


def test_layer_triton_default_and_env_optout(monkeypatch):
    assert ComplexKLALayer(16, n_state=4).use_triton_scan is True
    monkeypatch.setenv("CKLA_NO_TRITON", "1")
    assert ComplexKLALayer(16, n_state=4).use_triton_scan is False


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for compiled triton op")
def test_triton_cuda_matches_sequential():
    mod = _load_ckla_triton(interpret=False)
    if not mod.HAS_TRITON:
        pytest.skip("triton not installed")
    rng = np.random.default_rng(4242)
    B, T, N, D = 4, 250, 16, 128
    inputs = _random_inputs(rng, B, T, N, D)
    w_re = torch.as_tensor(rng.standard_normal((B, T, D)), dtype=torch.float32, device="cuda")
    w_im = torch.as_tensor(rng.standard_normal((B, T, D)), dtype=torch.float32, device="cuda")

    def run(fn, **kw):
        tens = tuple(t.cuda().requires_grad_(True) for t in _torch_inputs(inputs, torch.float32))
        y_re, y_im = fn(*tens, **kw)
        ((y_re * w_re).sum() + (y_im * w_im).sum()).backward()
        grads = []
        for t in tens:
            assert t.grad is not None
            grads.append(t.grad.cpu().numpy())
        return y_re.detach().cpu().numpy(), y_im.detach().cpu().numpy(), grads

    yr_s, yi_s, g_seq = run(complex_kla_scan)
    yr_t, yi_t, g_tri = run(mod.complex_kla_scan_triton)
    np.testing.assert_allclose(yr_t, yr_s, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(yi_t, yi_s, rtol=1e-4, atol=1e-5)
    # T=250 fp32 accumulation: grads compared at a slightly looser atol than
    # the small interpreter sizes (summation-order rounding over 250 steps).
    for name, g_s, g_t in zip(_GRAD_NAMES, g_seq, g_tri):
        np.testing.assert_allclose(
            g_t, g_s, rtol=1e-3, atol=1e-4, err_msg=f"triton CUDA grad mismatch: {name}"
        )
