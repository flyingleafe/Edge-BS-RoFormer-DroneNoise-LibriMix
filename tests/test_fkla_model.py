"""Tests for the vendored flat-KLA op/layer and the FKLA RPS model.

Ground truth for the op is an independent NumPy fp64 port of the sequential
real-KLA flat recursion (``fkla.reference.flat_step``, arXiv 2602.10743 —
the same reference the CKLA zero-rotation test uses), checked against the
vendored parallel dyadic-tree implementation.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from models.ckla import SimpleConvV2CKLA
from models.fkla import FKLARPSModel, FlatKLALayer, TemporalFKLAHead, flat_kla

# ─── 1. Op vs sequential NumPy reference ─────────────────────────────────────


def _ref_flat_scan(abar, pbar, k, v, lam_v, q, eps=1e-8):
    """Sequential real-KLA flat recursion (fkla.reference.flat_step port)."""
    B, T, N = k.shape
    D = v.shape[-1]
    eta = np.zeros((B, N, D), dtype=np.float64)
    lam = np.zeros((B, N, D), dtype=np.float64)
    ys = np.zeros((B, T, D), dtype=np.float64)
    for t in range(T):
        phi = k[:, t, :, None] ** 2 * lam_v[:, t, None, :]
        kappa = k[:, t, :, None] * lam_v[:, t, None, :] * v[:, t, None, :]
        den = abar[None] ** 2 + pbar[None] * lam
        eta = abar[None] * eta / den + kappa
        lam = lam / den + phi
        mu = eta / np.maximum(lam, eps)
        ys[:, t] = np.einsum("bn,bnd->bd", q[:, t], mu)
    return ys


def _random_inputs(rng, B, T, N, D):
    gamma = rng.uniform(0.05, 3.0, size=(N, D))
    dt = rng.uniform(0.005, 0.5, size=(N, D))
    abar = np.exp(-gamma * dt)
    pbar = rng.uniform(1e-3, 0.5, size=(N, D))
    k = rng.standard_normal((B, T, N))
    q = rng.standard_normal((B, T, N))
    v = rng.standard_normal((B, T, D))
    lam_v = rng.uniform(0.01, 2.0, size=(B, T, D))
    return abar, pbar, k, v, lam_v, q


@pytest.mark.parametrize("B", [1, 2])
@pytest.mark.parametrize("T", [1, 5, 40])
@pytest.mark.parametrize("N", [1, 4])
@pytest.mark.parametrize("D", [1, 8])
def test_flat_kla_matches_sequential_reference(B, T, N, D):
    rng = np.random.default_rng(1000 * B + 100 * T + 10 * N + D)
    inputs = _random_inputs(rng, B, T, N, D)
    ref = _ref_flat_scan(*inputs)

    # fp64: the parallel tree op must reproduce the sequential recursion.
    abar, pbar, k, v, lam_v, q = (torch.as_tensor(a, dtype=torch.float64) for a in inputs)
    y64 = flat_kla(abar, pbar, k, v, lam_v, q)
    np.testing.assert_allclose(y64.numpy(), ref, rtol=1e-8, atol=1e-10)

    # fp32 (the training dtype): loose tolerance over the composition depth.
    abar, pbar, k, v, lam_v, q = (torch.as_tensor(a, dtype=torch.float32) for a in inputs)
    y32 = flat_kla(abar, pbar, k, v, lam_v, q)
    np.testing.assert_allclose(y32.numpy(), ref, rtol=1e-3, atol=1e-4)


def test_flat_kla_fold_weight_scales_output():
    rng = np.random.default_rng(3)
    abar, pbar, k, v, lam_v, q = (
        torch.as_tensor(a, dtype=torch.float64) for a in _random_inputs(rng, 2, 12, 4, 8)
    )
    y = flat_kla(abar, pbar, k, v, lam_v, q)
    fold_w = torch.rand(2, 12, dtype=torch.float64)
    yw = flat_kla(abar, pbar, k, v, lam_v, q, fold_weight=fold_w)
    np.testing.assert_allclose(yw.numpy(), (fold_w.unsqueeze(-1) * y).numpy(), rtol=1e-12)


# ─── 2. Layer / head shape contracts ─────────────────────────────────────────


def test_layer_forward_shape():
    torch.manual_seed(0)
    layer = FlatKLALayer(32, n_state=8)
    x = torch.randn(2, 20, 32)
    y = layer(x)
    assert y.shape == (2, 20, 32)
    assert y.dtype == torch.float32
    assert torch.isfinite(y).all()


def test_head_shape_contract():
    torch.manual_seed(0)
    head = TemporalFKLAHead(in_ch=128, d_model=64, num_rotors=4, n_layers=2, n_state=8)
    x = torch.randn(2, 128, 30)
    y = head(x)
    assert y.shape == (2, 4, 30)


# ─── 3. Model end-to-end: forward shape, finite grads, param budget ─────────


def test_model_end_to_end_and_param_budget():
    torch.manual_seed(0)
    model = FKLARPSModel(p_init=1.0)
    audio = torch.randn(2, 64000)
    out = model(audio)
    assert out.shape == (2, 4, 126)  # same contract as SimpleConvV2CKLA

    loss = out.pow(2).mean()
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, f"no grad for {name}"
        assert torch.isfinite(p.grad).all(), f"non-finite grad in {name}"

    n_fkla = sum(p.numel() for p in model.parameters())
    ref = SimpleConvV2CKLA(p_init=1.0)
    n_ckla = sum(p.numel() for p in ref.parameters())
    head_fkla = sum(p.numel() for p in model.head.parameters())
    head_ckla = sum(p.numel() for p in ref.head.parameters())
    print(
        f"params: FKLA full {n_fkla} vs CKLA {n_ckla} (ratio {n_fkla / n_ckla:.3f}); "
        f"heads {head_fkla} vs {head_ckla}"
    )
    # FKLA lacks CKLA's rotation machinery + complex mix — must stay within
    # 10% of the CKLA arm for the comparison to be parameter-fair.
    assert abs(n_fkla - n_ckla) / n_ckla <= 0.10


def test_registry_builds_fkla():
    from models.registry import build_model

    m = build_model("simple_conv_v2_fkla", n_fft=256, hop_length=64, num_rotors=4, p_init=1.0)
    assert isinstance(m, FKLARPSModel)
    assert getattr(m.frontend, "out_channels", None) == 2  # stft_mag_if, as the CKLA arms


# ─── 4. Overfit smoke: loss decreases on random data ─────────────────────────


def test_overfit_smoke_20_steps():
    torch.manual_seed(0)
    model = FKLARPSModel(n_fft=256, hop_length=64, num_rotors=4, p_init=1.0)
    audio = torch.randn(4, 4000)
    with torch.no_grad():
        t_frames = model(audio).shape[-1]
    target = torch.rand(4, 4, t_frames) * 2 - 1

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    losses = []
    for _ in range(20):
        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(model(audio), target)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert all(np.isfinite(losses))
    assert losses[-1] < losses[0], f"loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"
