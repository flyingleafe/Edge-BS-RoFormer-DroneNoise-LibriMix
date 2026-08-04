"""Tests for comb-spectral RPS trajectory refinement (``rps_refinement``).

Synthetic harmonic-comb signals (sum of ``a_k cos(k*phase + phi_k)`` with
``a_k = amp/k`` along an analytic rotor trajectory) exercise the whole stack:
clock-offset scan (stage A), constant-per-window ``delta`` grid (stage B),
joint spline refinement (stage C), the comb-contrast confidence gate, and the
coherent block-wise harmonic least-squares residual metric.

Runtimes are kept small by shortening durations (3-4 s) and dropping the Adam
iteration count to 150 in the shared config.
"""

from __future__ import annotations

import numpy as np

from tracking.rps_refinement import (
    RefineConfig,
    coarse_delta,
    comb_confidence,
    compute_logmag,
    estimate_clock_offset,
    harmonic_lsq_residual,
    refine_trajectories,
)

SR = 16000


def _cfg(**over: object) -> RefineConfig:
    """Shared config with a cheaper Adam budget than the 16 kHz default."""
    params: dict[str, object] = {"iters": 150}
    params.update(over)
    return RefineConfig(**params)  # type: ignore[arg-type]


def _synth_comb(
    rotors: list[np.ndarray],
    sr: int,
    rng: np.random.Generator,
    *,
    k_max: int = 30,
    noise_std: float = 0.01,
    amp: float = 0.5,
) -> np.ndarray:
    """Sum of harmonic combs (one per rotor sample-rate speed array) + white noise."""
    n = len(rotors[0])
    audio = np.zeros(n, dtype=np.float64)
    for r in rotors:
        phase = 2.0 * np.pi * np.cumsum(r) / sr
        for k in range(1, k_max + 1):
            phi = float(rng.uniform(0.0, 2.0 * np.pi))
            audio += (amp / k) * np.cos(k * phase + phi)
    audio += noise_std * rng.standard_normal(n)
    return audio.astype(np.float32)


def _sine_traj(base: float, dev: float, freq: float, phase: float, t: np.ndarray) -> np.ndarray:
    """``base + dev * sin(2 pi freq t + phase)`` on times ``t``."""
    return base + dev * np.sin(2.0 * np.pi * freq * t + phase)


# ─── 1. Stage A: clock-offset recovery ──────────────────────────────────────────


def test_estimate_clock_offset_recovers_injected_shift():
    rng = np.random.default_rng(0)
    cfg = _cfg()
    dur = 4.0
    n = int(dur * SR)
    t = np.arange(n, dtype=np.float64) / SR

    # Chirping audio trajectory r(t) = 75 + 5 sin(2 pi 0.4 t).
    r_audio = _sine_traj(75.0, 5.0, 0.4, 0.0, t)
    audio = _synth_comb([r_audio], SR, rng, noise_std=0.02)
    spec = compute_logmag(audio, cfg)

    # Telemetry (100 Hz) is the same trajectory delayed by +0.12 s: sampling
    # g(s) = r_audio(s - shift) means the best tau (telemetry lags audio) = shift.
    shift = 0.12
    motor_times = np.arange(-0.6, dur + 0.6, 0.01)
    motor_values = _sine_traj(75.0, 5.0, 0.4, 0.0, motor_times - shift)[None, :]

    tau_best, taus, scores = estimate_clock_offset(spec, motor_times, motor_values, cfg)
    assert scores.shape == taus.shape
    assert abs(tau_best - shift) < 0.01


# ─── 2. Stage B: constant per-rotor delta recovery ──────────────────────────────


def test_coarse_delta_recovers_constant_per_rotor_offsets():
    rng = np.random.default_rng(1)
    cfg = _cfg()
    dur = 4.0
    n = int(dur * SR)

    r_true = [72.0, 88.0]
    audio = _synth_comb([np.full(n, r_true[0]), np.full(n, r_true[1])], SR, rng, noise_std=0.02)
    spec = compute_logmag(audio, cfg)

    # Init corrupted away from truth; the recoverable delta is the correction.
    target = np.array([+0.8, -0.6])
    r_init = np.stack(
        [
            np.full(spec.n_frames, r_true[0] - target[0]),
            np.full(spec.n_frames, r_true[1] - target[1]),
        ]
    )

    delta, centers, _ = coarse_delta(spec, r_init, cfg)
    assert delta.shape[0] == 2
    assert centers.shape[0] == delta.shape[1]
    recovered = delta.mean(axis=1)
    assert np.all(np.abs(recovered - target) < 0.1)


# ─── 3. Stage C: joint refinement of a time-varying corruption ──────────────────


def test_refine_trajectories_reduces_time_varying_error():
    rng = np.random.default_rng(2)
    cfg = _cfg()
    dur = 4.0
    n = int(dur * SR)

    audio = _synth_comb([np.full(n, 75.0)], SR, rng, noise_std=0.01)
    spec = compute_logmag(audio, cfg)

    r_true = np.full((1, spec.n_frames), 75.0)
    # Slow, bounded (<= 1 rev/s) corruption: constant + sine.
    corruption = 0.6 + 0.4 * np.sin(2.0 * np.pi * 0.3 * spec.frame_times)
    r_init = r_true + corruption[None, :]

    result = refine_trajectories(spec, r_init, cfg)

    err_before = float(np.mean(np.abs(r_init - r_true)))
    err_after = float(np.mean(np.abs(result.r_refined - r_true)))
    assert err_after < 0.1
    assert err_after < err_before / 5.0


# ─── 4. Confidence gate: comb present vs. pure noise ─────────────────────────────


def test_comb_confidence_high_on_comb_near_zero_on_noise():
    rng = np.random.default_rng(3)
    cfg = _cfg()
    dur = 3.0
    n = int(dur * SR)

    r_samples = np.full(n, 75.0)
    comb_audio = _synth_comb([r_samples], SR, rng, noise_std=0.01)
    spec_comb = compute_logmag(comb_audio, cfg)
    r_frames = np.full((1, spec_comb.n_frames), 75.0)

    conf_comb, _ = comb_confidence(spec_comb, r_frames, cfg)
    assert conf_comb.mean() > 0.1

    noise_audio = rng.standard_normal(n).astype(np.float32)
    spec_noise = compute_logmag(noise_audio, cfg)
    r_frames_noise = np.full((1, spec_noise.n_frames), 75.0)
    conf_noise, _ = comb_confidence(spec_noise, r_frames_noise, cfg)
    assert np.abs(conf_noise).max() < 0.03


# ─── 5. Coherent harmonic least-squares residual ────────────────────────────────


def test_harmonic_lsq_residual_low_on_single_rotor_truth():
    rng = np.random.default_rng(4)
    cfg = _cfg()
    dur = 3.0
    n = int(dur * SR)

    audio = _synth_comb([np.full(n, 75.0)], SR, rng, noise_std=0.01)
    spec = compute_logmag(audio, cfg)
    r_frames = np.full((1, spec.n_frames), 75.0)

    res = harmonic_lsq_residual(audio, r_frames, spec.frame_times, cfg)
    assert res["residual_ratio"] < 0.05
    assert res["n_tracks"] > 0


def test_harmonic_lsq_residual_handles_crossing_rotors():
    rng = np.random.default_rng(5)
    cfg = _cfg()
    dur = 3.0
    n = int(dur * SR)
    t = np.arange(n, dtype=np.float64) / SR

    # Two rotors whose trajectories cross (same base, opposite-phase swings).
    r1 = _sine_traj(75.0, 2.0, 0.5, 0.0, t)
    r2 = _sine_traj(75.0, 2.0, 0.5, np.pi / 2.0, t)
    audio = _synth_comb([r1, r2], SR, rng, noise_std=0.01)
    spec = compute_logmag(audio, cfg)

    r_frames = np.stack(
        [
            _sine_traj(75.0, 2.0, 0.5, 0.0, spec.frame_times),
            _sine_traj(75.0, 2.0, 0.5, np.pi / 2.0, spec.frame_times),
        ]
    )
    res = harmonic_lsq_residual(audio, r_frames, spec.frame_times, cfg)
    # Joint solve must fit overlapping harmonics at the true trajectories.
    assert res["residual_ratio"] < 0.05


def test_harmonic_lsq_residual_penalises_corrupted_init():
    rng = np.random.default_rng(6)
    cfg = _cfg()
    dur = 3.0
    n = int(dur * SR)

    audio = _synth_comb([np.full(n, 75.0)], SR, rng, noise_std=0.01)
    spec = compute_logmag(audio, cfg)
    r_true = np.full((1, spec.n_frames), 75.0)
    r_corrupt = r_true + 0.7

    res_true = harmonic_lsq_residual(audio, r_true, spec.frame_times, cfg)
    res_corrupt = harmonic_lsq_residual(audio, r_corrupt, spec.frame_times, cfg)
    assert res_true["residual_ratio"] < res_corrupt["residual_ratio"]


# ─── 6. Shape / determinism sanity ──────────────────────────────────────────────


def test_refine_shapes_and_determinism():
    rng = np.random.default_rng(7)
    cfg = _cfg()
    dur = 3.0
    n = int(dur * SR)

    audio = _synth_comb([np.full(n, 75.0)], SR, rng, noise_std=0.01)
    spec = compute_logmag(audio, cfg)

    # Frame grid matches compute_logmag's grid metadata exactly.
    expected_times = np.arange(spec.n_frames) * cfg.hop_length / cfg.sample_rate
    np.testing.assert_allclose(spec.frame_times, expected_times)

    r_init = np.full((1, spec.n_frames), 75.5)
    res_a = refine_trajectories(spec, r_init, cfg)
    res_b = refine_trajectories(spec, r_init, cfg)

    assert res_a.r_refined.shape == r_init.shape
    assert res_a.r_coarse.shape == r_init.shape
    np.testing.assert_array_equal(res_a.frame_times, spec.frame_times)
    # Deterministic (CPU Adam from zero-initialised knots).
    np.testing.assert_allclose(res_a.r_refined, res_b.r_refined)
