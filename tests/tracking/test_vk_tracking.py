"""Synthetic tests for the coupled Vold–Kalman order tracker.

Tests 1–5 from ``docs/vk-order-tracking-design.md`` §4: single-rotor wobble
capture from a biased init, twin-pair separation from a pair-mean init (the
stage-B/C killer), crossing tracks without identity swap, the capture basin
under the ``k_max``-growth schedule, and no hallucination on white noise.
All deterministic (seeded) and sized for speed (8–10 s @ 16 kHz, k_max 20).

Run:  pytest tests/test_vk_tracking.py
"""

import time

import numpy as np
import pytest

from tracking.vk_tracking import VKConfig, vk_track

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


def wobble_fixture(dur: float = 10.0, snr_db: float = 10.0, seed: int = 0):
    """Design-test-1 fixture: single rotor, ±2 rev/s wobble, biased init."""
    t, frame_times, edge = make_grid(dur)
    r_true = 45.0 + 2.0 * np.sin(2 * np.pi * 0.3 * t)
    y = synth_comb(t, [r_true], snr_db=snr_db, seed=seed)
    r_init = np.full((1, len(frame_times)), 46.5)
    truth = np.interp(frame_times, t, r_true)
    return y, r_init, frame_times, edge, truth


def test_bw_adapt_off_is_inert():
    """``bw_adapt=False`` must leave results bit-identical to the default
    config regardless of the clamp knob (regression guard: at implementation
    time the off path was additionally verified byte-identical to the
    pre-change code on this fixture and the twin-pair fixture)."""
    y, r_init, frame_times, _, _ = wobble_fixture(dur=8.0)
    res_a = vk_track(y, r_init, frame_times, make_cfg())
    res_b = vk_track(y, r_init, frame_times, make_cfg(bw_adapt=False, bw_adapt_clamp=2.0))
    assert np.array_equal(res_a.r_refined, res_b.r_refined)
    assert np.array_equal(res_a.r_env, res_b.r_env)
    assert np.array_equal(res_a.envelopes.x, res_b.envelopes.x)
    assert res_a.residual_ratios == res_b.residual_ratios
    assert "bw_gain" not in res_a.extras and "bw_adapt_factors" not in res_a.extras
    # neutral adaptation (clamp = 1 pins every gain at 1) exercises the
    # per-track-rho^2 solver path and must reproduce the scalar path
    res_n = vk_track(y, r_init, frame_times, make_cfg(bw_adapt=True, bw_adapt_clamp=1.0))
    assert np.allclose(res_n.r_refined, res_a.r_refined, rtol=0, atol=1e-9)
    assert np.allclose(res_n.envelopes.x, res_a.envelopes.x, rtol=0, atol=1e-9)


def test_bw_adapt_converges_factors_lock():
    """``bw_adapt=True`` on a clean comb: still converges (error <= 1.2x the
    non-adapted run), factors > 1 early (band narrows during capture) and
    -> ~1 at lock, cumulative gain within the clamp."""
    y, r_init, frame_times, edge, truth = wobble_fixture()
    cfg = make_cfg()
    res0 = vk_track(y, r_init, frame_times, cfg)
    res1 = vk_track(y, r_init, frame_times, make_cfg(bw_adapt=True))
    rms0 = float(np.sqrt(np.mean((res0.r_refined[0, edge] - truth[edge]) ** 2)))
    rms1 = float(np.sqrt(np.mean((res1.r_refined[0, edge] - truth[edge]) ** 2)))
    assert rms1 < 0.05, f"adapted run failed to converge (RMS {rms1:.4f})"
    assert rms1 <= 1.2 * rms0, f"adapted RMS {rms1:.4f} worse than 1.2x baseline {rms0:.4f}"

    facs = res1.extras["bw_adapt_factors"]
    assert len(facs) == cfg.n_outer
    first = facs[0][np.isfinite(facs[0])]
    last = facs[-1][np.isfinite(facs[-1])]
    assert len(first) and len(last)
    assert float(np.median(first)) > 1.02, "capture-phase factors should exceed 1"
    assert np.all(np.abs(last - 1.0) < 0.05), f"factors not ~1 at lock: {last}"
    gain = res1.extras["bw_gain"]
    assert gain.shape == (1, cfg.k_max - cfg.k_min + 1)
    clamp = cfg.bw_adapt_clamp
    assert np.all(gain >= clamp**-2 - 1e-12) and np.all(gain <= clamp**2 + 1e-12)


def test_bw_adapt_clamp_respected_heavy_noise():
    """Adversarial fixture (comb at −10 dB SNR): noise-dominated bands push
    the factors up round after round — the cumulative gain must saturate at
    the clamp, never exceed it."""
    dur = 6.0
    t, frame_times, _ = make_grid(dur)
    y = synth_comb(t, [np.full_like(t, 45.0)], snr_db=-10.0, seed=7)
    r_init = np.full((1, len(frame_times)), 45.0)
    clamp = 1.2
    res = vk_track(y, r_init, frame_times, make_cfg(bw_adapt=True, bw_adapt_clamp=clamp))
    gain = res.extras["bw_gain"]
    assert np.all(np.isfinite(gain))
    assert np.all(gain >= clamp**-2 - 1e-12), f"gain below clamp floor: {gain.min()}"
    assert np.all(gain <= clamp**2 + 1e-12), f"gain above clamp ceiling: {gain.max()}"
    assert np.isclose(gain.max(), clamp**2), "clamp never engaged — fixture not adversarial"


def test_config_rejects_bad_bw_adapt_clamp():
    with pytest.raises(ValueError, match="bw_adapt_clamp"):
        VKConfig(bw_adapt_clamp=0.5)


def test_config_rejects_too_small_bandwidth():
    """The Tuma denominator positivity check must raise with the actual limit."""
    dur = 2.0
    _, frame_times, _ = make_grid(dur)
    y = np.zeros(int(dur * FS))
    with pytest.raises(ValueError, match="bandwidth too small"):
        vk_track(y, np.full((1, len(frame_times)), 45.0), frame_times, make_cfg(bw_hz=1e-9))
