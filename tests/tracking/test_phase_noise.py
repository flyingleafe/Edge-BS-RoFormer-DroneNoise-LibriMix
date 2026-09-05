"""WP18 — the rank-one-plus-diagonal moment estimator (``tracking.phase_noise``).

The measurement it backs claims to separate a harmonic-common term
``sigma_J^2`` from per-harmonic terms ``v_k`` out of one K x K covariance.
These tests construct data where both are known.
"""

from __future__ import annotations

import numpy as np
import pytest

from tracking import phase_noise as E


def _make(n=20000, k_n=12, sigma_j=0.3, seed=0):
    rng = np.random.default_rng(seed)
    v = 0.05 + 0.4 / np.arange(1, k_n + 1) ** 2  # a k-dependent diagonal
    j = rng.normal(0.0, sigma_j, n)
    x = j[None, :] + rng.normal(0.0, 1.0, (k_n, n)) * np.sqrt(v)[:, None]
    return x, v


def test_moment_fit_recovers_common_and_diagonal():
    x, v = _make()
    valid = np.ones(x.shape, dtype=bool)
    c, _ = E._pairwise_cov(x, valid)
    fit = E.fit_rank_one(c, np.full(x.shape[0], 6.0))
    assert fit["sigma_c2_mean"] == pytest.approx(0.09, rel=0.08)
    assert fit["sigma_c2_median"] == pytest.approx(0.09, rel=0.15)
    got = np.asarray(fit["v_k"])
    assert np.allclose(got, v, rtol=0.15, atol=0.01)
    # a genuine rank-one plus diagonal: uniform loading, small residual
    assert abs(fit["loading_beta"]) < 0.05
    assert fit["offdiag_resid_rel"] < 0.35
    assert fit["rank1_energy_frac"] > 0.9


def test_no_common_term_gives_zero_sigma():
    rng = np.random.default_rng(1)
    k_n = 12
    v = 0.05 + 0.4 / np.arange(1, k_n + 1) ** 2
    x = rng.normal(0.0, 1.0, (k_n, 20000)) * np.sqrt(v)[:, None]
    c, _ = E._pairwise_cov(x, np.ones(x.shape, dtype=bool))
    fit = E.fit_rank_one(c, np.full(k_n, 6.0))
    assert abs(fit["sigma_c2_mean"]) < 0.005
    assert np.allclose(np.asarray(fit["v_k"]), v, rtol=0.1, atol=0.01)


def test_pairwise_complete_matches_full_when_masked_at_random():
    x, v = _make(n=40000)
    rng = np.random.default_rng(3)
    valid = rng.random(x.shape) > 0.3  # per-harmonic dropout, as the twin gate does
    c, n = E._pairwise_cov(x, valid)
    fit = E.fit_rank_one(c, np.full(x.shape[0], 6.0))
    assert (n[~np.eye(x.shape[0], dtype=bool)] > 1000).all()
    assert fit["sigma_c2_mean"] == pytest.approx(0.09, rel=0.12)
    assert np.allclose(np.asarray(fit["v_k"]), v, rtol=0.2, atol=0.015)


def test_structured_common_term_is_flagged():
    """A common term whose loading falls as 1/k (a shared *phase*, not a shared
    delay) must NOT read as the uniform-loading model."""
    rng = np.random.default_rng(5)
    k_n, n = 12, 20000
    a = 1.0 / np.arange(1, k_n + 1)
    v = np.full(k_n, 0.05)
    x = (
        a[:, None] * rng.normal(0.0, 0.5, n)[None, :]
        + rng.normal(0, 1, (k_n, n)) * np.sqrt(v)[:, None]
    )
    c, _ = E._pairwise_cov(x, np.ones((k_n, n), dtype=bool))
    fit = E.fit_rank_one(c, np.full(k_n, 6.0))
    assert fit["loading_beta"] < -0.7  # recovers the 1/k loading
    assert fit["offdiag_resid_rel"] > 0.5  # and the constant-offdiag fit is bad


def test_kscaled_common_estimate_uses_wider_partners_only():
    """With a monotone band schedule the common power is cumulative in the
    band, so ``C_ij`` measures it up to ``min(B_i, B_j)``; the estimator must
    read harmonic k's common term off partners with a band at least as wide."""
    k_n, n = 10, 40000
    rng = np.random.default_rng(7)
    bands = np.arange(1, k_n + 1) * 1.0
    # common power grows with the band -> per-harmonic common contribution
    cum = 0.02 * bands
    inc = np.diff(np.concatenate(([0.0], cum)))
    comps = rng.normal(0.0, 1.0, (k_n, n)) * np.sqrt(inc)[:, None]
    # harmonic i sees components 0..i (everything inside its band)
    common = np.cumsum(comps, axis=0)
    v = np.full(k_n, 0.05)
    x = common + rng.normal(0, 1, (k_n, n)) * np.sqrt(v)[:, None]
    c, _ = E._pairwise_cov(x, np.ones((k_n, n), dtype=bool))
    fit = E.fit_rank_one(c, bands)
    # every harmonic but the widest (which has no wider partner and is
    # extrapolated, hence flagged) recovers its own diagonal
    assert np.allclose(np.asarray(fit["v_k"])[:-1], v[:-1], rtol=0.25, atol=0.01)
    assert fit["common_extrapolated"][-1] is True
    assert not any(fit["common_extrapolated"][:-1])
    got = np.asarray(fit["common_k"], dtype=float)
    assert np.allclose(got[:-1], cum[:-1], rtol=0.3, atol=0.012)


def test_brickwall_highpass_removes_slow_common_term():
    fs, n = 62.5, 4000
    t = np.arange(n) / fs
    slow = 2.0 * np.sin(2 * np.pi * 0.7 * t)
    fast = np.random.default_rng(9).normal(0, 0.2, n)
    y = E._brickwall(slow + fast, 4.0, fs, high=True)
    assert np.var(y) < 0.1 * np.var(slow)
    assert np.var(y) == pytest.approx(np.var(fast) * (1 - 4.0 / (fs / 2)), rel=0.25)


# ---------------------------------------------------------------------------
# The stochastic comb model's four readings: line width, line shape,
# cross-harmonic correlation, residual shape.


def _wiener_tone(gamma_hz, fs=500.0, dur_s=60.0, n_real=8, noise=1e-4, seed=11):
    """Complex envelopes of a tone whose phase is a Wiener process.

    Diffusion ``D = 4 pi gamma`` gives a Lorentzian line of half width
    ``gamma``: the phase increment variance over a lag is ``D lag``, so
    ``E[z(t+lag) conj z(t)] = exp(-D lag / 2) = exp(-2 pi gamma lag)``.
    """
    rng = np.random.default_rng(seed)
    n = int(round(dur_s * fs))
    d = 4.0 * np.pi * gamma_hz
    steps = rng.normal(0.0, np.sqrt(d / fs), (n_real, n))
    phi = np.cumsum(steps, axis=-1)
    z = np.exp(1j * phi)
    z += np.sqrt(noise / 2.0) * (rng.normal(size=z.shape) + 1j * rng.normal(size=z.shape))
    return z, fs


def test_wiener_tone_recovers_its_lorentzian_half_width():
    for gamma in (1.0, 4.0):
        z, fs = _wiener_tone(gamma)
        got = E.linewidth(z, fs, max_lag_s=1.0)
        assert not got["censored"]
        assert got["gamma_hz"] == pytest.approx(gamma, rel=0.20)
        # tau and gamma are the same statement, so they must agree exactly.
        assert got["tau_s"] == pytest.approx(1.0 / (2 * np.pi * got["gamma_hz"]), rel=1e-9)


def test_wiener_tone_reads_as_a_lorentzian_line():
    z, fs = _wiener_tone(2.0, dur_s=60.0)
    f, p = E.welch_envelope(z, fs)
    shape = E.fit_line_shape(f, p, hwhm0=2.0)
    assert shape["verdict"] == "lorentz"
    assert shape["log_resid_ratio"] > 0.05
    assert shape["lorentz"]["hwhm_hz"] == pytest.approx(2.0, rel=0.3)


def test_gaussian_jittered_tone_reads_as_a_gaussian_line():
    """A tone whose frequency is drawn from a Gaussian and then held — the
    inhomogeneous broadening whose line shape is Gaussian, not Lorentzian."""
    rng = np.random.default_rng(3)
    fs, dur_s, n_real, sigma_f = 500.0, 20.0, 128, 2.0
    n = int(round(dur_s * fs))
    t = np.arange(n) / fs
    f_c = rng.normal(0.0, sigma_f, n_real)
    z = np.exp(2j * np.pi * f_c[:, None] * t[None, :])
    z += 1e-2 * (rng.normal(size=z.shape) + 1j * rng.normal(size=z.shape))
    f, p = E.welch_envelope(z, fs)
    hwhm = sigma_f * np.sqrt(2.0 * np.log(2.0))
    shape = E.fit_line_shape(f, p, hwhm0=hwhm)
    assert shape["verdict"] == "gauss"
    assert shape["log_resid_ratio"] < 0.0


def test_linewidth_law_fit_recovers_gamma0_and_slope():
    k = np.arange(1, 31, dtype=float)
    gamma = 0.4 + 0.6 * k
    got = E.fit_linewidth_law(k, gamma)
    assert got["gamma0_hz"] == pytest.approx(0.4, abs=1e-9)
    assert got["slope_hz_per_k"] == pytest.approx(0.6, abs=1e-9)
    assert got["resid_rms_hz"] < 1e-9


def test_censored_coherence_time_is_reported_and_not_invented():
    """A tone with no phase noise never crosses exp(-1); the width is a BOUND."""
    fs, n = 500.0, 5000
    z = np.exp(2j * np.pi * 0.05 * np.arange(n) / fs)[None, :]
    got = E.linewidth(z, fs, max_lag_s=2.0)
    assert got["censored"]
    assert not np.isfinite(got["gamma_hz"])
    assert got["gamma_bound_hz"] == pytest.approx(1.0 / (2 * np.pi * 2.0), rel=1e-6)


# --- the cross-harmonic correlation, end to end through demod_rotor --------


def _comb_audio(mode, *, rate=60.0, k_max=12, dur_s=10.0, sr=16000, sigma=0.02, seed=17):
    """One rotor's comb whose harmonics share a shaft jitter, or do not.

    ``mode="shared"``: the SHAFT phase random-walks, so harmonic ``k`` carries
    ``k`` times that walk and every rate opinion is the same number.
    ``mode="independent"``: each harmonic carries its own walk.
    """
    rng = np.random.default_rng(seed)
    n = int(round(dur_s * sr))
    t = np.arange(n) / sr
    ks = np.arange(1, k_max + 1)
    walk_n = 1 if mode == "shared" else k_max
    steps = rng.normal(0.0, sigma / np.sqrt(sr), (walk_n, n))
    walk = np.cumsum(steps, axis=-1)  # revolutions
    x = np.zeros(n)
    for i, k in enumerate(ks):
        extra = walk[0] * k if mode == "shared" else walk[i]
        x += (1.0 / k) * np.cos(2 * np.pi * (k * rate * t + extra))
    x += 0.01 * rng.normal(size=n)
    ft = np.arange(0.0, dur_s, 0.032)
    return x[None, :], np.full((1, len(ft)), rate), ft


def _series(mode, arm):
    audio, r_ft, ft = _comb_audio(mode)
    dm = E.demod_rotor(audio, r_ft, ft, 0, k_max=12, b_wide=20.0)
    assert dm is not None
    return dm, E.arm_increments(dm, arm)


def test_shared_shaft_jitter_gives_rho_one_and_a_rank_one_covariance():
    arm = E.Arm("fixB1.5", "fixed", 1.5)
    dm, ser = _series("shared", arm)
    corr = E.cross_harmonic_correlation(ser)
    assert corr["n_keep"] >= 8
    assert np.nanmin(corr["rho_k"]) > 0.9
    fit = E.arm_covariance(dm, arm, series=ser)["cov"]["0"]
    assert fit["rank1_energy_frac"] > 0.9


def test_independent_per_harmonic_jitter_gives_rho_near_zero():
    arm = E.Arm("fixB1.5", "fixed", 1.5)
    dm, ser = _series("independent", arm)
    corr = E.cross_harmonic_correlation(ser)
    assert corr["n_keep"] >= 8
    assert abs(corr["rho_mean"]) < 0.15
    fit = E.arm_covariance(dm, arm, series=ser)["cov"]["0"]
    assert fit["rank1_energy_frac"] < 0.6


def test_arm_increments_leaves_arm_covariance_bit_identical():
    """The extraction is a refactor: passing the series in changes nothing."""
    arm = E.Arm("fixB3", "fixed", 3.0)
    dm, ser = _series("shared", arm)
    a = E.arm_covariance(dm, arm)
    b = E.arm_covariance(dm, arm, series=ser)
    assert a["cov"]["0"]["sigma_c2_mean"] == b["cov"]["0"]["sigma_c2_mean"]
    assert a["keep"] == b["keep"]
    assert a["snr"] == b["snr"]


# --- the residual shape ----------------------------------------------------


def test_residual_tail_stats_separates_a_cauchy_from_a_gaussian():
    rng = np.random.default_rng(23)
    g = E.residual_tail_stats(rng.normal(0.0, 1.0, 20000))
    assert g["verdict"] == "gauss"
    assert abs(g["excess_kurtosis"]) < 0.5
    c = E.residual_tail_stats(rng.standard_cauchy(20000))
    assert c["verdict"] == "cauchy"
    assert c["llr_per_sample"] > 0.1
    assert c["excess_kurtosis"] > 5.0


def test_shared_rate_opinion_removes_the_common_term():
    """With a shared shaft jitter the residual ``delta_k - c`` must collapse."""
    arm = E.Arm("fixB1.5", "fixed", 1.5)
    dm, ser = _series("shared", arm)
    cov = E.arm_covariance(dm, arm, series=ser)
    c, mask = E.shared_rate_opinion(ser, np.asarray(cov["v_k_used"]), np.asarray(cov["k_used"]))
    idx = np.where(ser.keep)[0]
    e = ser.delta[:, idx, :] - c[:, None, :]
    keep_frames = mask[None, None, :] & ser.valid[idx][None, :, :]
    assert np.nanstd(e[keep_frames]) < 0.5 * np.nanstd(ser.delta[:, idx, :][keep_frames])


def test_median_by_k_drops_the_harmonics_measured_once():
    """The law is fitted on medians, so a harmonic nobody agreed on is dropped."""
    k = np.array([8, 8, 8, 8, 10, 10, 10, 9, 1])
    g = np.array([0.2, 0.25, 0.22, 0.9, 0.3, 0.32, 0.28, 11.0, 10.0])
    ks, med, cnt = E.median_by_k(k, g, min_count=3)
    assert ks.tolist() == [8.0, 10.0]
    assert med[0] == pytest.approx(0.235, abs=1e-9)
    assert cnt.tolist() == [4, 3]


def test_theilsen_law_survives_an_outlier_that_breaks_least_squares():
    k = np.arange(8, 31, 2, dtype=float)
    gamma = 0.1 + 0.06 * k
    spoiled = gamma.copy()
    spoiled[0] = 11.0  # one decade-sized outlier at the lowest harmonic
    ols = E.fit_linewidth_law(k, spoiled)
    ts = E.fit_linewidth_law(k, spoiled, method="theilsen")
    assert ts["slope_hz_per_k"] == pytest.approx(0.06, rel=0.15)
    assert ols["slope_hz_per_k"] < 0.0  # the outlier flips the least-squares sign
    assert ts["slope_lo_hz_per_k"] <= ts["slope_hz_per_k"] <= ts["slope_hi_hz_per_k"]
    assert E.fit_linewidth_law(k, gamma, method="theilsen")["slope_hz_per_k"] == pytest.approx(
        0.06, rel=1e-6
    )


def test_gap_filter_is_a_no_op_without_a_band():
    x = np.random.default_rng(31).normal(size=(3, 200))
    valid = np.ones(x.shape, dtype=bool)
    assert E.gap_filter(x, valid, None, 62.5) is x


def test_smoothing_reveals_a_slow_shared_term_the_raw_rate_hides():
    """The shared SHAFT disturbance is slow, so ``rho_k`` is a curve in the
    smoothing bandwidth — the WP18 covariance at the raw frame rate does not
    see it, and that is a property of the estimator, not of the aircraft."""
    arm = E.Arm("fixB1.5", "fixed", 1.5)
    audio, r_ft, ft = _comb_audio("shared", sigma=0.004, seed=41)
    rng = np.random.default_rng(42)
    audio = audio + 0.30 * rng.normal(size=audio.shape)  # per-harmonic measurement noise
    dm = E.demod_rotor(audio, r_ft, ft, 0, k_max=12, b_wide=20.0)
    assert dm is not None
    ser = E.arm_increments(dm, arm)
    raw = E.cross_harmonic_correlation(ser)
    smoothed = E.cross_harmonic_correlation(ser, smooth_hz=0.25)
    mid = E.cross_harmonic_correlation(ser, smooth_hz=1.0)
    assert raw["rho_mean"] < mid["rho_mean"] < smoothed["rho_mean"]
    assert smoothed["rho_mean"] > raw["rho_mean"] + 0.2
    assert smoothed["smooth_hz"] == pytest.approx(0.25)
    assert raw["smooth_hz"] is None
