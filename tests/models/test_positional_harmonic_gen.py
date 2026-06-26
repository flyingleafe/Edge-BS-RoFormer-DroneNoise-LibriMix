"""Synthetic sanity checks for the position-aware noise generator.

Validates the physics of the propagation stage (fractional delay + 1/r
attenuation + summation), differentiability w.r.t. position, and the
emit/propagate composition + output shapes of
:class:`PositionalHarmonicNoiseGen`.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from models.generative import HarmonicNoiseGenNew, PositionalHarmonicNoiseGen
from models.generative.positional_harmonic_gen import (
    SPEED_OF_SOUND,
    fractional_delay,
    propagate,
)

SR = 16000


# ---------------------------------------------------------------------------
# fractional_delay
# ---------------------------------------------------------------------------


def test_fractional_delay_zero_is_identity():
    x = torch.randn(3, 4, 2048)
    tau = torch.zeros(3, 4)
    y = fractional_delay(x, tau, SR)
    assert torch.allclose(y, x, atol=1e-5)


def test_fractional_delay_integer_matches_roll():
    # A band-limited signal delayed by an integer #samples must equal a roll
    # (away from the wrap-around edges).
    t = torch.arange(4096) / SR
    x = torch.sin(2 * torch.pi * 220.0 * t) + 0.5 * torch.sin(2 * torch.pi * 440.0 * t)
    x = x[None]  # [1, T]
    k = 17
    tau = torch.tensor([k / SR])
    y = fractional_delay(x, tau, SR)
    rolled = torch.roll(x, shifts=k, dims=-1)
    # ignore the wrapped region at the start
    assert torch.allclose(y[:, k + 5 :], rolled[:, k + 5 :], atol=1e-3)


def test_fractional_delay_subsample_shifts_phase():
    # Half-sample delay of a single tone => known phase shift at that frequency.
    f0 = 1000.0
    t = torch.arange(8192) / SR
    x = torch.cos(2 * torch.pi * f0 * t)[None]
    half = torch.tensor([0.5 / SR])
    y = fractional_delay(x, half, SR)
    # expected: cos(2 pi f0 (t - 0.5/SR))
    expected = torch.cos(2 * torch.pi * f0 * (t - 0.5 / SR))[None]
    assert torch.allclose(y[:, 100:-100], expected[:, 100:-100], atol=1e-3)


# ---------------------------------------------------------------------------
# propagate
# ---------------------------------------------------------------------------


def test_propagate_single_source_matches_manual():
    # One rotor at distance r: output == (ref/r) * fractional_delay(src, r/c).
    src = torch.randn(2, 1, 4096)
    rel = torch.tensor([[[0.3, 0.0, 0.0]], [[0.25, 0.1, 0.0]]])  # [B=2, R=1, 3]
    ref_distance = 1.0
    out = propagate(src, rel, sample_rate=SR, ref_distance=ref_distance)  # [2, T]

    r = torch.linalg.vector_norm(rel, dim=-1)  # [2, 1]
    tau = r / SPEED_OF_SOUND
    manual = (ref_distance / r).unsqueeze(-1) * fractional_delay(src, tau, SR)
    manual = manual.sum(dim=-2)
    assert torch.allclose(out, manual, atol=1e-5)


def test_propagate_attenuation_follows_inverse_r():
    # Same source at two distances: amplitude ratio == inverse distance ratio.
    src = torch.randn(1, 1, 4096)
    near = propagate(src, torch.tensor([[[0.2, 0.0, 0.0]]]), sample_rate=SR)
    far = propagate(src, torch.tensor([[[0.6, 0.0, 0.0]]]), sample_rate=SR)
    # L2/energy is preserved by the phase-ramp delay (L1 is not), so compare RMS.
    ratio = near.norm() / far.norm()
    assert ratio.item() == pytest.approx(0.6 / 0.2, rel=1e-3)


def test_propagate_two_rotors_interfere():
    # Sum of two delayed copies != a single naive copy (delays create structure).
    src = torch.randn(1, 2, 4096)
    rel = torch.tensor([[[0.2, 0.0, 0.0], [0.4, 0.1, 0.0]]])  # [1, 2, 3]
    out = propagate(src, rel, sample_rate=SR)  # [1, T]
    assert out.shape == (1, 4096)
    # the combined signal differs from either rotor rendered alone
    solo0 = propagate(src[:, :1], rel[:, :1], sample_rate=SR)
    assert not torch.allclose(out, solo0, atol=1e-2)


def test_propagate_multi_observer_shape_and_consistency():
    src = torch.randn(2, 4, 4096)
    # [B=2, M=3, R=4, 3]
    rel = torch.randn(2, 3, 4, 3) * 0.2 + 0.3
    out = propagate(src, rel, sample_rate=SR)
    assert out.shape == (2, 3, 4096)
    # each observer channel must match the single-observer call on its slice
    for m in range(3):
        single = propagate(src, rel[:, m], sample_rate=SR)
        assert torch.allclose(out[:, m], single, atol=1e-5)


def test_propagate_fused_equals_time_domain_loop():
    # The frequency-domain rotor-sum (R fwd + M inv transforms) must equal the
    # naive per-(mic, rotor) time-domain sum of fractional_delay calls.
    src = torch.randn(2, 4, 4096)
    rel = torch.randn(2, 3, 4, 3) * 0.15 + 0.3  # [B=2, M=3, R=4, 3]
    fused = propagate(src, rel, sample_rate=SR)  # [2, 3, T]

    dist = torch.linalg.vector_norm(rel, dim=-1).clamp_min(1e-6)  # [2, 3, 4]
    ref = torch.zeros_like(fused)
    for mic in range(3):
        for rotor in range(4):
            tau = (dist[:, mic, rotor] / SPEED_OF_SOUND).unsqueeze(-1)  # [B, 1]
            amp = (1.0 / dist[:, mic, rotor]).unsqueeze(-1)  # [B, 1]
            delayed = fractional_delay(src[:, rotor], tau.squeeze(-1), SR)  # [B, T]
            ref[:, mic] += amp * delayed
    assert torch.allclose(fused, ref, atol=1e-4)


def test_propagate_differentiable_wrt_position():
    src = torch.randn(1, 2, 2048)
    rel = torch.tensor([[[0.3, 0.0, 0.1], [0.25, 0.1, 0.0]]], requires_grad=True)
    out = propagate(src, rel, sample_rate=SR)
    out.pow(2).mean().backward()
    assert rel.grad is not None
    assert torch.isfinite(rel.grad).all()
    assert rel.grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# PositionalHarmonicNoiseGen
# ---------------------------------------------------------------------------


def _det_model():
    # Deterministic emitter: disable the random broadband branch so emit() is
    # reproducible and forward == propagate(emit(...)).
    torch.manual_seed(0)
    emitter = HarmonicNoiseGenNew(
        n_harmonics=16, sample_rate=SR, n_oscillators=1, use_diff_noise=False
    )
    return PositionalHarmonicNoiseGen(emitter=emitter, sample_rate=SR)


def test_emit_shape():
    model = _det_model()
    rps = torch.full((2, 4, SR), 80.0)
    sources = model.emit(rps)
    assert sources.shape == (2, 4, SR)


def test_forward_single_point_shape():
    model = _det_model()
    rps = torch.full((2, 4, SR), 80.0)
    rel = torch.randn(2, 4, 3) * 0.1 + 0.3
    out = model(rps, rel)
    assert out.shape == (2, SR)


def test_forward_multi_observer_shape():
    model = _det_model()
    rps = torch.full((1, 4, SR), 80.0)
    rel = torch.randn(1, 8, 4, 3) * 0.1 + 0.3  # 8-mic array
    out = model(rps, rel)
    assert out.shape == (1, 8, SR)


def test_forward_composes_emit_and_propagate():
    # With the random branch off, forward must equal propagate(emit(...)).
    model = _det_model()
    rps = torch.full((1, 4, SR), 75.0)
    rel = torch.randn(1, 4, 3) * 0.1 + 0.3

    res = model(rps, rel, return_dict=True)
    expected = propagate(model.emit(rps), rel, sample_rate=SR, c=SPEED_OF_SOUND)
    assert torch.allclose(res["audio"], expected, atol=1e-5)


def test_per_rotor_sources_track_their_speed():
    # Each rotor's source should carry its own fundamental: a rotor spun faster
    # must place its spectral peak higher.
    model = _det_model()
    rps = torch.zeros(1, 2, SR)
    rps[0, 0] = 60.0
    rps[0, 1] = 120.0
    sources = model.emit(rps)  # [1, 2, T]

    freqs = np.fft.rfftfreq(SR, d=1.0 / SR)
    f0s = [60.0, 120.0]
    for r, f0 in enumerate(f0s):
        mag = np.abs(np.fft.rfft(sources[0, r].detach().numpy()))
        # Isolate the fundamental: search below the 2nd harmonic so an untrained
        # emitter's random per-harmonic gains can't make a higher harmonic win.
        band = (freqs > 10.0) & (freqs < 1.5 * f0)
        peak = freqs[band][np.argmax(mag[band])]
        assert peak == pytest.approx(f0, abs=3.0)
