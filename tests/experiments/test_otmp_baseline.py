"""The optimal-transport multi-pitch baseline (Björkman & Elvander, TSP 2026).

Three things are worth pinning down, and they are the three the paper's method
actually rests on:

* the eq-(18) ground cost really does break the octave ambiguity that eq (17)
  cannot,
* the water-filling step of Prop. 2 solves its constrained sub-problem exactly
  (checked against a generic numerical optimizer), and
* the whole Bregman iteration recovers a fundamental from a clean signal.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from experiments.otmp_baseline.cost import (
    ground_cost,
    harmonic_cost_unnormalized,
    linear_grid,
    nearest_harmonic_order,
)
from experiments.otmp_baseline.estimate import (
    OTMPConfig,
    autocovariance,
    estimate_clip,
    estimate_frame,
    link_frames,
    simulated_config,
)
from experiments.otmp_baseline.solver import water_fill_log

# --------------------------------------------------------------------------
# grids and the ground cost
# --------------------------------------------------------------------------


def test_linear_grid_takes_a_step_or_a_count_but_not_both():
    np.testing.assert_allclose(linear_grid(1.0, 2.0, n=3), [1.0, 1.5, 2.0])
    np.testing.assert_allclose(linear_grid(1.0, 2.0, step=0.25), [1.0, 1.25, 1.5, 1.75, 2.0])
    with pytest.raises(ValueError):
        linear_grid(1.0, 2.0)
    with pytest.raises(ValueError):
        linear_grid(1.0, 2.0, step=0.1, n=5)


def test_exact_harmonics_cost_nothing_and_the_order_is_a_positive_integer():
    f0 = 197.0
    teeth = f0 * np.arange(1, 9, dtype=float)
    np.testing.assert_allclose(ground_cost(teeth, np.array([f0]))[:, 0], 0.0, atol=1e-24)
    # A frequency below the fundamental cannot be explained by k = 0.
    orders = nearest_harmonic_order(np.array([0.2 * f0]), np.array([f0]))
    assert orders[0, 0] == 1.0
    assert ground_cost(np.array([0.2 * f0]), np.array([f0]))[0, 0] == pytest.approx(0.64)


def test_ground_cost_prefers_the_fundamental_over_its_sub_octave():
    """The whole point of eq (18): a perturbed comb costs 4x under w0/2."""
    f0, offset = 197.0, 1.3
    teeth = f0 * np.arange(1, 9, dtype=float) + offset
    cost = ground_cost(teeth, np.array([f0, f0 / 2.0]))
    assert cost[:, 0].sum() < cost[:, 1].sum()
    np.testing.assert_allclose(cost[:, 1], 4.0 * cost[:, 0], rtol=1e-9)


def test_the_unnormalized_cost_of_eq_17_is_octave_blind():
    """The same comb, scored by eq (17): w0 and w0/2 are indistinguishable."""
    f0, offset = 197.0, 1.3
    teeth = f0 * np.arange(1, 9, dtype=float) + offset
    cost = harmonic_cost_unnormalized(teeth, np.array([f0, f0 / 2.0]))
    np.testing.assert_allclose(cost[:, 0], cost[:, 1], rtol=1e-9)
    np.testing.assert_allclose(cost[:, 0], offset**2, rtol=1e-9)


def test_the_normalized_cost_prefers_the_highest_consistent_fundamental():
    """A full comb of w0 is best explained by w0, not by w0/2 or w0/3."""
    f0 = 120.0
    rng = np.random.default_rng(0)
    teeth = f0 * np.arange(1, 11, dtype=float) + rng.uniform(-2.0, 2.0, 10)
    candidates = np.array([f0 / 3.0, f0 / 2.0, f0])
    sums = ground_cost(teeth, candidates).sum(axis=0)
    assert sums[2] < sums[1] < sums[0]


# --------------------------------------------------------------------------
# the water-filling step of Prop. 2
# --------------------------------------------------------------------------


def _brute_force_water_fill(log_b: np.ndarray, budget: float) -> np.ndarray:
    """Same sub-problem, solved by a generic constrained optimizer."""
    from scipy.optimize import minimize

    n = log_b.size
    shift = float(log_b.max())

    def obj(s):
        return float(np.sum(np.exp(log_b - shift - s)))

    def jac(s):
        return -np.exp(log_b - shift - s)

    res = minimize(
        obj,
        x0=np.full(n, budget / n),
        jac=jac,
        bounds=[(0.0, None)] * n,
        constraints=[{"type": "ineq", "fun": lambda s: budget - s.sum()}],
        method="SLSQP",
        options={"maxiter": 500, "ftol": 1e-14},
    )
    assert res.success, res.message
    return res.x


@pytest.mark.parametrize("budget", [0.5, 4.0, 30.0])
def test_water_filling_never_loses_to_a_generic_constrained_solver(budget):
    rng = np.random.default_rng(3)
    log_b = rng.normal(scale=3.0, size=(3, 8))
    got = water_fill_log(log_b, budget, max_active=8)

    assert np.all(got >= -1e-12)
    for row in range(log_b.shape[0]):
        want = _brute_force_water_fill(log_b[row], budget)
        shift = log_b[row].max()
        obj_got = np.sum(np.exp(log_b[row] - shift - got[row]))
        obj_want = np.sum(np.exp(log_b[row] - shift - want))
        assert obj_got <= obj_want + 1e-12
        # SLSQP stops well short of machine precision on this very flat
        # optimum, so agreement is asserted only coarsely here; the KKT test
        # below is what pins the closed form down exactly.
        np.testing.assert_allclose(got[row], want, atol=5e-2)


@pytest.mark.parametrize("budget", [0.5, 4.0, 30.0])
def test_water_filling_satisfies_the_kkt_conditions_of_its_sub_problem(budget):
    """Budget spent in full, one common water level over the active set.

    The sub-problem is strictly convex, so these conditions have a single
    solution — this is an exact check, not a comparison.
    """
    rng = np.random.default_rng(3)
    log_b = rng.normal(scale=3.0, size=(3, 8))
    got = water_fill_log(log_b, budget, max_active=8)

    for row in range(log_b.shape[0]):
        assert got[row].sum() == pytest.approx(budget, rel=1e-12)
        level = log_b[row] - got[row]
        active = got[row] > 0.0
        np.testing.assert_allclose(level[active], level[active][0], rtol=1e-12)
        assert np.all(log_b[row][~active] <= level[active][0] + 1e-12)


def test_water_filling_levels_the_active_set_and_leaves_the_rest_alone():
    log_b = np.array([[10.0, 9.0, 1.0, -5.0]])
    got = water_fill_log(log_b, budget=3.0, max_active=4)[0]
    assert got[2] == 0.0 and got[3] == 0.0  # below the water level
    # the two active entries end up at a common level
    assert (log_b[0, 0] - got[0]) == pytest.approx(log_b[0, 1] - got[1])
    assert got.sum() == pytest.approx(3.0)


def test_the_sorted_window_grows_when_it_saturates():
    """A window smaller than the true active set must be detected and grown."""
    log_b = np.zeros((2, 32))  # every entry active, whatever the budget
    got = water_fill_log(log_b, budget=8.0, max_active=2)
    np.testing.assert_allclose(got, 8.0 / 32.0)


def test_a_zero_budget_is_a_no_op():
    log_b = np.random.default_rng(0).normal(size=(2, 5))
    np.testing.assert_array_equal(water_fill_log(log_b, 0.0), np.zeros_like(log_b))


# --------------------------------------------------------------------------
# the data-fit term
# --------------------------------------------------------------------------


def test_the_fft_gram_matches_an_explicit_dictionary_product():
    """The Toeplitz/FFT shortcut must be exact, not approximate."""
    from experiments.otmp_baseline.solver import build_quadratic

    rng = np.random.default_rng(5)
    n_lags = 40
    freqs = np.linspace(0.05, 2.9, 97)
    r_hat = rng.normal(size=n_lags) + 1j * rng.normal(size=n_lags)
    quad = build_quadratic(r_hat, freqs)

    dictionary = np.exp(1j * np.outer(np.arange(n_lags), freqs))
    nu = rng.random(freqs.size)
    np.testing.assert_allclose(quad.matvec(nu), np.real(dictionary.conj().T @ dictionary) @ nu)
    np.testing.assert_allclose(quad.corr, np.real(dictionary.conj().T @ r_hat))
    # L = 2||A||^2 / T sets the step size; an underestimate would diverge.
    assert quad.op_norm_sq == pytest.approx(
        float(np.linalg.eigvalsh(np.real(dictionary.conj().T @ dictionary))[-1]), rel=1e-6
    )


def test_a_non_uniform_grid_falls_back_to_the_dense_gram():
    from experiments.otmp_baseline.solver import build_quadratic

    freqs = np.array([0.1, 0.15, 0.4, 0.9, 1.7])
    quad = build_quadratic(np.ones(12, dtype=complex), freqs)
    assert quad.kernel is None and quad.gram is not None
    dictionary = np.exp(1j * np.outer(np.arange(12), freqs))
    nu = np.arange(1.0, 6.0)
    np.testing.assert_allclose(quad.matvec(nu), np.real(dictionary.conj().T @ dictionary) @ nu)


# --------------------------------------------------------------------------
# analysis front end
# --------------------------------------------------------------------------


def test_autocovariance_of_a_complex_exponential_is_the_exponential():
    sr, n, freq = 8000.0, 4000, 313.0
    t = np.arange(n)
    z = np.exp(2j * np.pi * freq / sr * t)
    got = autocovariance(z, 64, unbiased=True)
    want = np.exp(2j * np.pi * freq / sr * np.arange(64))
    np.testing.assert_allclose(got, want, atol=1e-9)


# --------------------------------------------------------------------------
# the full estimator
# --------------------------------------------------------------------------


def _toy_config(**over) -> OTMPConfig:
    """A deliberately small grid so the solver runs in a second or two."""
    base = OTMPConfig(
        sample_rate=8000,
        frame_len=250,
        n_freq=400,
        pitch_lo_hz=60.0,
        pitch_hi_hz=300.0,
        n_pitch=49,  # 5 Hz steps
        n_pitches=1,
        eta=1e-1,
        zeta=1e1,
        eps=1e-6,
        beta=1.5e-2,
        debias_zeta=1e0,
        debias_beta=4.5e-2,
        max_iter=400,
        debias_max_iter=400,
        inner_iters=1,
    )
    return replace(base, **over) if over else base


def _harmonic_frame(f0: float, n_harm: int = 8, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(250)
    z = sum(
        np.exp(1j * (2 * np.pi * k * f0 / 8000.0 * t + rng.uniform(0, 2 * np.pi)))
        for k in range(1, n_harm + 1)
    )
    return np.asarray(z) / np.sqrt(np.mean(np.abs(z) ** 2))


@pytest.mark.parametrize("f0", [125.0, 200.0])
def test_a_noiseless_single_pitch_is_recovered(f0):
    est = estimate_frame(_harmonic_frame(f0), 8000, _toy_config())
    assert est.pitches_hz[0] == pytest.approx(f0, abs=5.0)  # one pitch-grid step
    assert est.masses[0] > 0.0


def test_the_estimator_does_not_fall_to_the_sub_octave():
    """The failure mode eq (18) exists to prevent, on a clean comb."""
    est = estimate_frame(_harmonic_frame(200.0), 8000, _toy_config())
    assert est.pitches_hz[0] > 150.0


def test_two_pitches_are_separated():
    cfg = _toy_config(n_pitches=2, min_sep_rel=0.05)
    frame = _harmonic_frame(125.0, seed=1) + _harmonic_frame(200.0, seed=2)
    est = estimate_frame(frame / np.sqrt(np.mean(np.abs(frame) ** 2)), 8000, cfg)
    found = np.sort(est.pitches_hz)
    np.testing.assert_allclose(found, [125.0, 200.0], atol=6.0)


def test_debiasing_concentrates_the_spectrum_on_the_active_pitches():
    cfg = _toy_config()
    biased = estimate_frame(_harmonic_frame(200.0), 8000, replace(cfg, debias=False))
    debiased = estimate_frame(_harmonic_frame(200.0), 8000, cfg)
    assert debiased.nu.sum() > biased.nu.sum()  # the l1 shrinkage is undone


def test_estimate_frame_rejects_a_mismatched_sample_rate():
    with pytest.raises(ValueError):
        estimate_frame(_harmonic_frame(200.0), 16000, _toy_config())


# --------------------------------------------------------------------------
# clip level
# --------------------------------------------------------------------------


def test_link_frames_reorders_for_continuity():
    # frame 1 is reported in the opposite order; linking must undo that
    pitches = np.array([[100.0, 201.0, 99.0], [200.0, 101.0, 202.0]])
    linked = link_frames(pitches)
    np.testing.assert_allclose(linked[0], [100.0, 101.0, 99.0])
    np.testing.assert_allclose(linked[1], [200.0, 201.0, 202.0])


def test_estimate_clip_returns_frame_centers_and_a_track_per_pitch():
    cfg = _toy_config(max_iter=40, debias_max_iter=40)
    clip = np.concatenate([_harmonic_frame(200.0, seed=s) for s in range(3)])
    times, pitches = estimate_clip(np.real(clip), 8000, cfg)
    assert pitches.shape == (cfg.n_pitches, 3)
    np.testing.assert_allclose(times, [125 / 8000, 375 / 8000, 625 / 8000])


# --------------------------------------------------------------------------
# the paper's simulation (slow)
# --------------------------------------------------------------------------


@pytest.mark.slow
def test_the_monte_carlo_self_test_stays_below_its_gross_error_bar():
    """Sec. VIII-A, a handful of draws. Guards against a silent regression.

    The paper reports 8-10 % gross error rate here; this implementation sits
    at 28 % over 50 draws, so the bar is set where a real regression would show
    but the known gap does not fail the suite. See the module docstring of
    ``experiments.otmp_baseline.simulation`` for what the gap is.
    """
    from experiments.otmp_baseline.simulation import run_monte_carlo

    cfg = simulated_config(max_iter=800, debias_max_iter=800, n_pitch=451)
    res = run_monte_carlo(6, cfg, snr_db=5.0, seed=1)
    assert res.ger <= 0.45
    assert np.median(res.deviations_cents) < 50.0
