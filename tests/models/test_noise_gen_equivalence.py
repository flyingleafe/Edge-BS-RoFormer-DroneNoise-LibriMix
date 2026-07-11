"""Numerical-equivalence guard for the noise-generator forward optimizations.

The generator forward was sped up (~2x on CPU, far more on GPU) by three edits
that must NOT change what the model computes — three E6 arms are mid-training on
these exact state-dict keys and their checkpoints must keep producing the same
audio:

1. ``dsp.freqs_to_phasors``: ``exp(1j * dphi)`` -> ``torch.polar(1, dphi)`` (same
   ``cumprod``; ``polar`` is bit-identical to the imaginary ``exp``).
2. ``dsp.upsample_with_windows``: the ``F.fold``/``col2im`` overlap-add replaced
   by a fold-free 50%-overlap slice-add (``math_utils.overlap_add_50pct``).
3. ``HarmonicNoiseGenNew._apply_rps_jitter``: the Python OU scan replaced by a
   vectorized FIR (train-mode only).

This test reconstructs the *original* implementations as reference callables and
asserts the optimized model matches them within ``rtol=1e-4, atol=1e-5``.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

import models.generative.dsp as dsp
from models.generative.math_utils import overlap_add_50pct, overlap_and_add
from models.registry import build_noise_gen_model

SR = 16000
RTOL, ATOL = 1e-4, 1e-5


# ── reference (pre-optimization) implementations ───────────────────────────────


def _ref_freqs_to_phasors(freq: torch.Tensor, sr: int) -> torch.Tensor:
    """Original phasor construction: complex ``exp`` then ``cumprod``."""
    phase_diff = freq * 2 * torch.pi / sr
    return torch.cumprod(torch.exp(1j * phase_diff), -1)


def _ref_ou_scan(alpha, noise_scale, sigma, bo, n_intervals, dtype, device):
    """Original Ornstein-Uhlenbeck Python scan (draw order: eps, then d0)."""
    eps = torch.randn(bo, n_intervals, device=device, dtype=dtype)
    d = sigma * torch.randn(bo, device=device, dtype=dtype)
    cols = [d]
    for n in range(n_intervals):
        d = d * alpha + noise_scale * eps[:, n]
        cols.append(d)
    return torch.stack(cols, dim=1)


def _fixed_inputs(b=8, r=4, m=8, t=SR):
    torch.manual_seed(0)
    rps = torch.rand(b, r, t) * 40 + 60  # 60-100 Hz per rotor
    rotor = torch.tensor([[0.2, 0.2, 0.0], [-0.2, 0.2, 0.0], [-0.2, -0.2, 0.0], [0.2, -0.2, 0.0]])[
        :r
    ]
    mic = torch.randn(m, 3) * 0.03  # mic array ~3 cm around origin
    rel = (mic[None, :, None, :] - rotor[None, None, :, :]).expand(b, m, r, 3).contiguous()
    names = ["dregon", "michaels"] * (b // 2)
    initial_phases = torch.rand(b, r, 100) * 2 * math.pi
    return rps, rel, names, initial_phases


# ── unit-level: the two "exact" swaps really are (near-)bit-identical ───────────


def test_polar_matches_exp_phasors():
    torch.manual_seed(0)
    freq = (torch.rand(4, 100, 4000) * 40 + 60) * torch.arange(1, 101).view(1, 100, 1)
    opt = dsp.freqs_to_phasors(freq, SR)
    ref = _ref_freqs_to_phasors(freq, SR)
    assert torch.allclose(opt.real, ref.real, rtol=RTOL, atol=ATOL)
    assert torch.allclose(opt.imag, ref.imag, rtol=RTOL, atol=ATOL)


def test_overlap_add_50pct_matches_fold():
    torch.manual_seed(0)
    hop = 485
    windowed = torch.randn(200, 34, 2 * hop)
    assert torch.equal(overlap_add_50pct(windowed, hop), overlap_and_add(windowed, hop))


# ── OU vectorization matches the original scan (same RNG draw order) ────────────


def test_rps_jitter_vectorized_matches_scan():
    from models.generative import HarmonicNoiseGenNew

    sigma, tau = 0.6, 0.016
    emitter = HarmonicNoiseGenNew(
        n_harmonics=8, sample_rate=SR, n_oscillators=1, rps_jitter_sigma=sigma, rps_jitter_tau=tau
    )
    b, o, t = 1, 16, SR
    f0s = torch.full((b, o, t), 90.0)

    # Recompute the exact control-grid constants the method uses.
    duration = t / SR
    ctrl_dt = min(tau / 10.0, 1.0 / 50.0)
    n_intervals = max(1, int(np.ceil(duration / ctrl_dt)))
    dt = duration / n_intervals
    alpha = 1.0 - dt / tau
    noise_scale = sigma * float(np.sqrt(2.0 * dt / tau))

    torch.manual_seed(7)
    got = (emitter._apply_rps_jitter(f0s) - f0s)[0]  # [o, t]
    torch.manual_seed(7)
    ref_ctrl = _ref_ou_scan(alpha, noise_scale, sigma, b * o, n_intervals, f0s.dtype, f0s.device)
    ref = F.interpolate(
        ref_ctrl.reshape(b * o, 1, n_intervals + 1), size=t, mode="linear", align_corners=True
    ).reshape(o, t)
    assert torch.allclose(got, ref, rtol=1e-3, atol=1e-4)


# ── full-model: optimized == original, per the prompt's equivalence contract ───


def test_forward_equivalent_to_reference_implementation():
    """Build the composite, drive it with fixed inputs + seeded RNG, and assert
    the optimized forward matches the original (exp-phasor + fold-OLA) forward."""
    model = build_noise_gen_model(
        "positional_harmonic_gen",
        sample_rate=SR,
        n_harmonics=100,
        cond_dim=16,
        drone_names=["dregon", "michaels"],
        use_diff_noise=True,
    ).eval()
    rps, rel, names, initial_phases = _fixed_inputs()

    seed = 123
    torch.manual_seed(seed)
    out_opt = model(rps, rel, names, initial_phases=initial_phases)

    # Swap in the original implementations behind the same module globals the
    # generator resolves at call time, then re-run with the identical seed (the
    # diffuse-noise branch draws randn each forward).
    saved_phasors, saved_ola = dsp.freqs_to_phasors, dsp.overlap_add_50pct
    dsp.freqs_to_phasors = _ref_freqs_to_phasors
    dsp.overlap_add_50pct = lambda w, h: overlap_and_add(w, h)
    try:
        torch.manual_seed(seed)
        out_ref = model(rps, rel, names, initial_phases=initial_phases)
    finally:
        dsp.freqs_to_phasors, dsp.overlap_add_50pct = saved_phasors, saved_ola

    assert out_opt.shape == out_ref.shape == (8, 8, SR)
    assert torch.allclose(out_opt, out_ref, rtol=RTOL, atol=ATOL), (
        f"max abs diff {(out_opt - out_ref).abs().max().item():.3e}"
    )
