"""Synthetic tests for the coupled Vold–Kalman order tracker.

Tests 1–5 from ``docs/vk-order-tracking-design.md`` §4: single-rotor wobble
capture from a biased init, twin-pair separation from a pair-mean init (the
stage-B/C killer), crossing tracks without identity swap, the capture basin
under the ``k_max``-growth schedule, and no hallucination on white noise.
All deterministic (seeded) and sized for speed (8–10 s @ 16 kHz, k_max 20).

Run:  pytest tests/test_vk_tracking.py
"""

import os
import sys
import time

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from data_processing.vk_tracking import VKConfig, vk_track  # noqa: E402

FS = 16000.0
K_MAX = 20


def synth_comb(
    t: np.ndarray, r_true_list: list[np.ndarray], snr_db: float, seed: int
) -> np.ndarray:
    """Sum of harmonic combs (k = 1..K_MAX, amps 1/sqrt(k), random phases) + noise."""
    rng = np.random.default_rng(seed)
    sig = np.zeros_like(t)
    for r_true in r_true_list:
        phase = 2 * np.pi * np.cumsum(r_true) / FS
        for k in range(1, K_MAX + 1):
            sig += (1.0 / np.sqrt(k)) * np.cos(k * phase + rng.uniform(0, 2 * np.pi))
    noise = rng.standard_normal(len(t))
    noise *= np.sqrt(np.mean(sig**2) / (10 ** (snr_db / 10)) / np.mean(noise**2))
    return sig + noise


def make_grid(dur: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Audio time grid, frame grid (32 ms), and an edge mask on the frame grid.

    The first/last 0.5 s are excluded from error metrics: zero-phase filtering
    and the second-difference prior both have transients at the boundaries.
    """
    t = np.arange(int(dur * FS)) / FS
    frame_times = np.arange(0, dur, 0.032)
    edge = (frame_times > 0.5) & (frame_times < dur - 0.5)
    return t, frame_times, edge


def make_cfg(**overrides) -> VKConfig:
    # couple_hz=20 keeps same-rotor adjacent harmonics (spacing ~45 Hz)
    # uncoupled while twin/crossing pairs (separation < 15 Hz) do couple.
    defaults: dict = dict(fs=FS, k_max=K_MAX, n_outer=12, couple_hz=20.0)
    defaults.update(overrides)
    return VKConfig(**defaults)


def test_single_rotor_wobble_capture():
    """Design test 1: ±2 rev/s wobble @ 0.3 Hz, SNR 10 dB, +1.5 rev/s biased
    constant init → recover within 0.05 rev/s RMS."""
    dur = 10.0
    t, frame_times, edge = make_grid(dur)
    r_true = 45.0 + 2.0 * np.sin(2 * np.pi * 0.3 * t)
    y = synth_comb(t, [r_true], snr_db=10.0, seed=0)
    r_init = np.full((1, len(frame_times)), 45.0 + 1.5)

    res = vk_track(y, r_init, frame_times, make_cfg())

    err = res.r_refined[0, edge] - np.interp(frame_times, t, r_true)[edge]
    rms = float(np.sqrt(np.mean(err**2)))
    assert rms < 0.05, f"RMS {rms:.4f} rev/s exceeds 0.05"
    # a locked comb must also be confident and mostly reconstructed
    assert float(np.median(res.confidence)) > 1.0
    assert res.residual_ratios[-1] < res.residual_ratios[0]


def test_twin_pair_from_pair_mean():
    """Design test 2: rotors 0.65 rev/s apart, both initialised at the pair
    mean → both recovered with |bias| < 0.1 rev/s (best permutation — twins
    are permutation-ambiguous from audio alone, cf. the project's PIT
    alignment convention)."""
    dur = 10.0
    t, frame_times, edge = make_grid(dur)
    speeds = (45.0, 45.65)
    y = synth_comb(t, [np.full_like(t, s) for s in speeds], snr_db=10.0, seed=1)
    r_init = np.full((2, len(frame_times)), np.mean(speeds))

    t0 = time.perf_counter()
    res = vk_track(y, r_init, frame_times, make_cfg())
    print(f"\n[twin pair] vk_track wall-clock: {time.perf_counter() - t0:.2f} s")

    pred = res.r_refined[:, edge]
    biases = []
    for perm in ((0, 1), (1, 0)):
        biases.append([float(np.mean(pred[i] - speeds[p])) for i, p in enumerate(perm)])
    best = min(biases, key=lambda b: sum(abs(v) for v in b))
    assert all(abs(b) < 0.1 for b in best), f"twin biases {best} (rev/s) exceed 0.1"


def test_crossing_tracks_keep_identity():
    """Design test 3: two rotors crossing in base speed, tracked through the
    crossing without identity swap (each closer to its own truth)."""
    dur = 10.0
    t, frame_times, edge = make_grid(dur)
    r1 = 40.0 + 1.25 * t  # 40 → 52.5 rev/s
    r2 = 52.0 - 1.25 * t  # 52 → 39.5 rev/s, crossing at ~4.8 s
    y = synth_comb(t, [r1, r2], snr_db=10.0, seed=3)
    r_init = np.stack([np.interp(frame_times, t, r1) + 0.3, np.interp(frame_times, t, r2) - 0.3])

    res = vk_track(y, r_init, frame_times, make_cfg())

    truths = [np.interp(frame_times, t, r)[edge] for r in (r1, r2)]
    pred = res.r_refined[:, edge]
    for i in range(2):
        own = float(np.sqrt(np.mean((pred[i] - truths[i]) ** 2)))
        other = float(np.sqrt(np.mean((pred[i] - truths[1 - i]) ** 2)))
        assert own < 0.1, f"rotor {i} RMS to own truth {own:.3f} exceeds 0.1"
        assert own < other, f"rotor {i} swapped identity (own {own:.3f} vs other {other:.3f})"


def test_capture_basin():
    """Design test 4: init offsets {0.5, 1, 2, 3} rev/s; with
    k_schedule='grow' the basin must extend to at least 2 rev/s."""
    dur = 8.0
    t, frame_times, edge = make_grid(dur)
    r_true = 45.0
    y = synth_comb(t, [np.full_like(t, r_true)], snr_db=10.0, seed=4)
    cfg = make_cfg()

    basin_edge = 0.0
    for offset in (0.5, 1.0, 2.0, 3.0):
        r_init = np.full((1, len(frame_times)), r_true + offset)
        res = vk_track(y, r_init, frame_times, cfg)
        max_err = float(np.max(np.abs(res.r_refined[0, edge] - r_true)))
        if max_err < 0.1:
            basin_edge = offset
        else:
            break
    print(f"\n[capture basin] edge: >= {basin_edge} rev/s")
    assert basin_edge >= 2.0, f"capture basin {basin_edge} rev/s below the required 2"


def test_white_noise_no_hallucination():
    """Design test 5: pure white noise — confidence ≈ 0 and the trajectory
    stays at the init (the gate must refuse to update a comb-less rotor)."""
    dur = 8.0
    _, frame_times, _ = make_grid(dur)
    rng = np.random.default_rng(2)
    y = rng.standard_normal(int(dur * FS))
    r_init = np.full((1, len(frame_times)), 45.0)

    res = vk_track(y, r_init, frame_times, make_cfg())

    drift = float(np.max(np.abs(res.r_refined - 45.0)))
    assert drift < 0.05, f"trajectory hallucinated {drift:.3f} rev/s from white noise"
    assert float(res.confidence.max()) < 0.2, (
        f"confidence {res.confidence.max():.3f} not ≈ 0 on white noise"
    )


def test_config_rejects_too_small_bandwidth():
    """The Tuma denominator positivity check must raise with the actual limit."""
    dur = 2.0
    _, frame_times, _ = make_grid(dur)
    y = np.zeros(int(dur * FS))
    with pytest.raises(ValueError, match="bandwidth too small"):
        vk_track(y, np.full((1, len(frame_times)), 45.0), frame_times, make_cfg(bw_hz=1e-9))


def test_torch_backend_matches_scipy():
    """``backend="torch"`` (CPU, complex128) reproduces the scipy path on the
    twin-pair scenario — the case that exercises every routed kernel: batched
    demod, cross-term demod, the coupled-group block-tridiagonal solve, and
    the phase-slope update. complex64 stays within the 1e-3 rev/s regression
    tolerance of the bench gate."""
    pytest.importorskip("torch")
    dur = 6.0
    t, frame_times, _ = make_grid(dur)
    speeds = (45.0, 45.65)
    y = synth_comb(t, [np.full_like(t, s) for s in speeds], snr_db=10.0, seed=1)
    r_init = np.full((2, len(frame_times)), np.mean(speeds))
    cfgs = {
        "scipy": make_cfg(n_outer=6),
        "torch": make_cfg(n_outer=6, backend="torch", device="cpu"),
        "torch64": make_cfg(n_outer=6, backend="torch", device="cpu", torch_dtype="complex64"),
    }
    ref = vk_track(y, r_init, frame_times, cfgs["scipy"]).r_refined
    got = vk_track(y, r_init, frame_times, cfgs["torch"]).r_refined
    mae128 = float(np.mean(np.abs(got - ref)))
    assert mae128 < 1e-6, f"torch/complex128 deviates from scipy: MAE {mae128:.2e} rev/s"
    got64 = vk_track(y, r_init, frame_times, cfgs["torch64"]).r_refined
    mae64 = float(np.mean(np.abs(got64 - ref)))
    assert mae64 < 1e-3, f"torch/complex64 deviates from scipy: MAE {mae64:.2e} rev/s"


def test_torch_backend_requires_fft_lp_mode():
    with pytest.raises(ValueError, match="lp_mode"):
        VKConfig(backend="torch", lp_mode="iir")
