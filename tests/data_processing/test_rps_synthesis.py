"""Tests for the OU control-mode RPS-trajectory synthesizer."""

from __future__ import annotations

import numpy as np
import pytest

from data_processing.rps_synthesis import (
    DEFAULT_CONFIG,
    DREGON_PROFILE,
    MICHAELS_PROFILE,
    MIXER,
    MODE_NAMES,
    NUM_ROTORS,
    OUModeParams,
    RPSSynthConfig,
    blend_profiles,
    fit_config,
    generate,
    generate_batch,
    generate_intermittent,
    generate_intermittent_batch,
    modes_from_rps,
    rps_from_modes,
    scaled_config,
)


def test_mixer_is_orthogonal():
    # B^T B = 4 I  -> columns orthogonal with squared norm 4.
    assert np.allclose(MIXER.T @ MIXER, NUM_ROTORS * np.eye(NUM_ROTORS))


def test_modes_rps_roundtrip():
    rng = np.random.default_rng(0)
    w = rng.normal(80.0, 5.0, size=(NUM_ROTORS, 200))
    assert np.allclose(rps_from_modes(modes_from_rps(w)), w)


def test_generate_shape_and_range():
    w = generate(3.0, 100.0, rng=0)
    assert w.shape == (NUM_ROTORS, 300)
    assert w.min() >= DEFAULT_CONFIG.rps_min
    assert w.max() <= DEFAULT_CONFIG.rps_max


def test_generate_is_deterministic_with_seed():
    a = generate(2.0, 50.0, rng=42)
    b = generate(2.0, 50.0, rng=42)
    assert np.array_equal(a, b)
    c = generate(2.0, 50.0, rng=43)
    assert not np.array_equal(a, c)


def test_common_mode_mean_matches_hover_level():
    # Long trajectory: rotor-mean should sit near the common-mode mean.
    w = generate(120.0, 100.0, rng=1)
    assert abs(w.mean() - DEFAULT_CONFIG.common.mean) < 1.0


def test_aggressiveness_scales_maneuver_spread():
    # Higher aggressiveness => larger spread between rotors (maneuver modes grow).
    fs, dur = 100.0, 60.0

    def cross_rotor_spread(agg: float) -> float:
        w = generate(dur, fs, aggressiveness=agg, rng=7)
        return float(np.mean(np.std(w, axis=0)))  # spread across rotors per frame

    gentle = cross_rotor_spread(0.3)
    normal = cross_rotor_spread(1.0)
    aggressive = cross_rotor_spread(3.0)
    assert gentle < normal < aggressive


def test_zero_aggressiveness_is_constant():
    w = generate(5.0, 100.0, aggressiveness=0.0, rng=0)
    # All dynamic stds vanish -> every rotor pinned to its mode-mean composite.
    assert np.allclose(w, w[:, :1], atol=1e-9)


def test_mean_reversion_lag1_autocorrelation():
    # Recover tau of an isolated mode from a long path via lag-1 autocorrelation.
    fs, dur = 200.0, 400.0
    tau_true = 0.5
    cfg = RPSSynthConfig(
        common=OUModeParams(mean=80.0, std=5.0, tau=tau_true),
        roll=OUModeParams(0.0, 0.0, 1.0),
        pitch=OUModeParams(0.0, 0.0, 1.0),
        yaw=OUModeParams(0.0, 0.0, 1.0),
    )
    w = generate(dur, fs, config=cfg, rng=3)
    common = modes_from_rps(w)[0]
    x = common - common.mean()
    rho1 = float(np.mean(x[:-1] * x[1:]) / np.var(x))
    tau_est = (1.0 / fs) / -np.log(rho1)
    assert tau_est == pytest.approx(tau_true, rel=0.15)


def test_rotor_correlation_is_strongly_positive():
    # The common mode dominates => all rotor pairs strongly positively correlated,
    # as observed in real DREGON/Michael's flights.
    w = generate(120.0, 100.0, rng=5)
    corr = np.corrcoef(w)
    off_diag = corr[~np.eye(NUM_ROTORS, dtype=bool)]
    assert off_diag.min() > 0.4


def test_mode_scales_targets_single_mode():
    fs, dur = 100.0, 120.0
    base = generate(dur, fs, rng=9)
    yaw_heavy = generate(dur, fs, mode_scales={"yaw": 5.0}, rng=9)
    yaw_std_base = modes_from_rps(base)[MODE_NAMES.index("yaw")].std()
    yaw_std_heavy = modes_from_rps(yaw_heavy)[MODE_NAMES.index("yaw")].std()
    assert yaw_std_heavy > 3.0 * yaw_std_base


def test_mode_scales_rejects_unknown_key():
    with pytest.raises(ValueError, match="unknown mode_scales"):
        generate(1.0, 50.0, mode_scales={"bogus": 2.0}, rng=0)


def test_generate_batch_shape_and_reproducibility():
    batch = generate_batch(8, 2.0, 50.0, rng=11)
    assert batch.shape == (8, NUM_ROTORS, 100)
    again = generate_batch(8, 2.0, 50.0, rng=11)
    assert np.array_equal(batch, again)
    # Trajectories within a batch differ from one another.
    assert not np.array_equal(batch[0], batch[1])


def test_scaled_config_multiplies_stds():
    cfg = scaled_config(DEFAULT_CONFIG, 2.0)
    assert cfg.yaw.std == pytest.approx(2.0 * DEFAULT_CONFIG.yaw.std)
    assert cfg.common.std == pytest.approx(2.0 * DEFAULT_CONFIG.common.std)
    assert cfg.yaw.mean == DEFAULT_CONFIG.yaw.mean  # means untouched


def test_fit_config_recovers_synthetic_params():
    # Generate from a known config, fit it back, expect rough agreement.
    fs, dur = 200.0, 300.0
    true = RPSSynthConfig(
        common=OUModeParams(mean=78.0, std=4.5, tau=0.8),
        roll=OUModeParams(mean=0.0, std=0.9, tau=0.5),
        pitch=OUModeParams(mean=0.0, std=0.8, tau=0.6),
        yaw=OUModeParams(mean=2.0, std=1.5, tau=1.0),
    )
    traces = [generate(dur, fs, config=true, rng=s) for s in range(4)]
    fitted = fit_config(traces, [1.0 / fs] * 4)
    assert fitted.common.mean == pytest.approx(true.common.mean, abs=0.5)
    assert fitted.common.std == pytest.approx(true.common.std, rel=0.2)
    assert fitted.yaw.mean == pytest.approx(true.yaw.mean, abs=0.5)
    assert fitted.yaw.tau == pytest.approx(true.yaw.tau, rel=0.3)


def test_fit_config_skips_takeoff_samples():
    # A trace that is mostly below the in-flight threshold but has a valid tail.
    fs = 100.0
    inflight = generate(60.0, fs, rng=0)
    ramp = np.linspace(0.0, 30.0, int(5 * fs))[None, :].repeat(NUM_ROTORS, axis=0)
    trace = np.concatenate([ramp, inflight], axis=1)
    cfg = fit_config([trace], [1.0 / fs])
    # Hover level recovered despite the leading sub-threshold ramp.
    assert cfg.common.mean == pytest.approx(DEFAULT_CONFIG.common.mean, abs=2.0)


# ---------------------------------------------------------------------------
# Intermittent ("pilot + airframe") model
# ---------------------------------------------------------------------------


def _active_fraction(w: np.ndarray, fs: float, abs_thresh: float = 0.8) -> float:
    """Fraction of frames where differential-mode activity exceeds an absolute std."""
    from scipy.ndimage import uniform_filter1d

    m = modes_from_rps(w)
    win = max(3, int(0.5 * fs))
    diff = m[1:]
    lm = uniform_filter1d(diff, win, axis=1, mode="nearest")
    lv = uniform_filter1d((diff - lm) ** 2, win, axis=1, mode="nearest")
    activity = np.sqrt(np.clip(lv, 0.0, None)).mean(axis=0)
    return float((activity > abs_thresh).mean())


def test_intermittent_shape_and_range():
    w = generate_intermittent(5.0, 100.0, profile=DREGON_PROFILE, rng=0)
    assert w.shape == (NUM_ROTORS, 500)
    assert w.min() >= DREGON_PROFILE.rps_min
    assert w.max() <= DREGON_PROFILE.rps_max


def test_intermittent_is_deterministic_with_seed():
    a = generate_intermittent(4.0, 50.0, profile=DREGON_PROFILE, rng=7)
    b = generate_intermittent(4.0, 50.0, profile=DREGON_PROFILE, rng=7)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, generate_intermittent(4.0, 50.0, profile=DREGON_PROFILE, rng=8))


def test_intermittent_is_mostly_steady():
    # The defining property: the drone holds steady most of the time (active
    # fraction well below half) rather than wandering continuously like OU.
    w = generate_intermittent(120.0, 100.0, profile=DREGON_PROFILE, rng=1)
    assert _active_fraction(w, 100.0) < 0.35


def test_intermittent_burstier_than_ou():
    # For matched overall spread, the intermittent model concentrates its motion
    # into bursts -> the distribution of per-frame |mode-velocity| is heavier
    # tailed (higher kurtosis) than continuous OU.
    fs = 100.0
    wi = generate_intermittent(200.0, fs, profile=DREGON_PROFILE, rng=2)
    wo = generate(200.0, fs, aggressiveness=1.0, rng=2)

    def velocity_kurtosis(w):
        v = np.diff(modes_from_rps(w)[1:], axis=1).ravel()
        v = v - v.mean()
        return float(np.mean(v**4) / (np.mean(v**2) ** 2 + 1e-12))

    assert velocity_kurtosis(wi) > velocity_kurtosis(wo)


def test_aggressiveness_scales_intermittent_spread():
    fs, dur = 100.0, 120.0

    def spread(agg):
        w = generate_intermittent(dur, fs, profile=DREGON_PROFILE, aggressiveness=agg, rng=5)
        return float(np.mean(np.std(w, axis=0)))

    assert spread(0.3) < spread(1.0) < spread(2.5)


def test_zero_aggressiveness_holds_trim():
    # No maneuvers -> only tiny cruise jitter about the trim composite.
    w = generate_intermittent(10.0, 100.0, profile=DREGON_PROFILE, aggressiveness=0.0, rng=0)
    trim_rps = rps_from_modes(np.array([[p.trim] for p in DREGON_PROFILE.modes]))
    assert np.allclose(w, trim_rps, atol=3.0)  # within a few rev/s of held trim


def test_blend_profiles_interpolates_motor_tau_and_trim():
    mid = blend_profiles(DREGON_PROFILE, MICHAELS_PROFILE, 0.5)
    assert mid.motor_tau == pytest.approx(
        0.5 * (DREGON_PROFILE.motor_tau + MICHAELS_PROFILE.motor_tau)
    )
    assert mid.common.trim == pytest.approx(
        0.5 * (DREGON_PROFILE.common.trim + MICHAELS_PROFILE.common.trim)
    )
    # Endpoints recover the originals.
    assert blend_profiles(t=0.0).motor_tau == DREGON_PROFILE.motor_tau
    assert blend_profiles(t=1.0).motor_tau == MICHAELS_PROFILE.motor_tau


def test_blend_profiles_rejects_out_of_range():
    with pytest.raises(ValueError, match="blend factor"):
        blend_profiles(t=1.5)


def test_drone_profile_knob_changes_hover_level():
    # Michael's hovers lower than DREGON; the blend knob should move the mean.
    dregon = generate_intermittent(60.0, 100.0, drone_profile=0.0, rng=3).mean()
    michaels = generate_intermittent(60.0, 100.0, drone_profile=1.0, rng=3).mean()
    assert dregon > michaels + 3.0


def test_larger_motor_tau_smooths_edges():
    # Isolate the airframe lag: two profiles identical except motor_tau, with
    # cruise jitter switched off (jitter is added after the lag, so it would
    # otherwise mask the smoothing).  Larger motor_tau -> smoother maneuver
    # response -> smaller mean |frame-to-frame change| of the common mode.
    import dataclasses

    no_jitter_modes = {
        name: dataclasses.replace(p, cruise_std=0.0)
        for name, p in zip(MODE_NAMES, DREGON_PROFILE.modes, strict=True)
    }
    base = dataclasses.replace(DREGON_PROFILE, **no_jitter_modes)
    snappy = dataclasses.replace(base, motor_tau=0.05)
    sluggish = dataclasses.replace(base, motor_tau=0.5)
    fs = 100.0

    def common_roughness(profile):
        w = generate_intermittent(120.0, fs, profile=profile, rng=6)
        return float(np.mean(np.abs(np.diff(modes_from_rps(w)[0]))))

    assert common_roughness(sluggish) < common_roughness(snappy)


def test_intermittent_rotor_correlation_positive():
    w = generate_intermittent(120.0, 100.0, profile=DREGON_PROFILE, rng=5)
    corr = np.corrcoef(w)
    off_diag = corr[~np.eye(NUM_ROTORS, dtype=bool)]
    assert off_diag.min() > 0.3


def test_intermittent_batch_shape_and_reproducibility():
    batch = generate_intermittent_batch(6, 3.0, 50.0, profile=DREGON_PROFILE, rng=11)
    assert batch.shape == (6, NUM_ROTORS, 150)
    again = generate_intermittent_batch(6, 3.0, 50.0, profile=DREGON_PROFILE, rng=11)
    assert np.array_equal(batch, again)
    assert not np.array_equal(batch[0], batch[1])


def test_intermittent_rejects_profile_and_drone_profile_together():
    with pytest.raises(ValueError, match="either profile or drone_profile"):
        generate_intermittent(1.0, 50.0, profile=DREGON_PROFILE, drone_profile=0.5, rng=0)


# ── Full-flight model ─────────────────────────────────────────────────────────


def test_full_flight_starts_and_ends_at_zero():
    from data_processing.rps_synthesis import generate_full_flight

    w = generate_full_flight(60.0, 100.0, profile=DREGON_PROFILE, rng=0)  # (4, M)
    assert w.shape == (NUM_ROTORS, 6000)
    assert (w >= 0.0).all()
    mean = w.mean(0)
    assert mean[:50].mean() < 1.0  # first 0.5 s: rotors off (ground)
    assert mean[-50:].mean() < 1.0  # last 0.5 s: rotors off (ground)


def test_full_flight_covers_all_regimes():
    from data_processing.rps_synthesis import generate_full_flight

    w = generate_full_flight(90.0, 100.0, profile=DREGON_PROFILE, rng=3)
    mean = w.mean(0)
    hover = DREGON_PROFILE.common.trim
    assert (mean < 1.0).any()  # zero (ground)
    assert ((mean > 0.30 * hover) & (mean < 0.6 * hover)).any()  # idle/warm-up plateau
    assert (mean > 0.85 * hover).any()  # hover/cruise
    assert mean.max() <= DREGON_PROFILE.rps_max


def test_full_flight_sampled_duration_and_too_short():
    from data_processing.rps_synthesis import generate_full_flight

    w = generate_full_flight(None, 50.0, drone_profile=0.5, rng=1)
    assert w.shape[0] == NUM_ROTORS and w.shape[1] > 50 * 20  # at least ~20 s total
    with pytest.raises(ValueError, match="too short"):
        generate_full_flight(4.0, 50.0, profile=DREGON_PROFILE, rng=0)


def test_full_flight_reproducible_and_profile_exclusive():
    from data_processing.rps_synthesis import generate_full_flight

    a = generate_full_flight(40.0, 50.0, profile=MICHAELS_PROFILE, rng=7)
    b = generate_full_flight(40.0, 50.0, profile=MICHAELS_PROFILE, rng=7)
    assert np.array_equal(a, b)
    with pytest.raises(ValueError, match="either profile or drone_profile"):
        generate_full_flight(40.0, 50.0, profile=DREGON_PROFILE, drone_profile=0.5, rng=0)
