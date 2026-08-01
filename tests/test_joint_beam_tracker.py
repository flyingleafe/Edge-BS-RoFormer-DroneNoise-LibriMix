"""Unit tests for the joint 4-rotor beam-search tracker.

Cheap and deterministic — no ``vk_track``, no real audio.  Four groups:

(a) the mode projection round-trips (``B`` orthogonal, ``B^T B = 4 I``);
(b) the OU transition cost is invariant to relabelling the rotors under the
    matching permutation — the property that FORCES one shared scale for
    roll/pitch/yaw, since rotor identity is arbitrary under PIT;
(c) the k-scaled emission bandwidth does what it claims: ``b0_rps = 0``
    reproduces point sampling exactly, and the capture radius in rev/s stops
    shrinking with harmonic index;
(d) on a synthetic window with known ground truth the tracker recovers a
    trajectory it is given as the ONLY candidate — a wiring check that scoring
    and backtracking agree.

Run:  pytest tests/test_joint_beam_tracker.py
"""

import itertools
import math
import os
import sys
from dataclasses import replace as dc_replace
from typing import Any

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from data_processing.joint_beam_tracker import (  # noqa: E402
    NUM_ROTORS,
    BeamCfg,
    EmissionCfg,
    OUPrior,
    _band_penalty,
    _mode_cost,
    comb_scores,
    joint_beam_track,
    union_emission,
)
from data_processing.rps_synthesis import MIXER, modes_from_rps, rps_from_modes  # noqa: E402

FRAME_S = 0.032


# --------------------------------------------------------------------------
# (a) mode projection


def test_mixer_is_orthogonal_with_norm_four():
    np.testing.assert_allclose(MIXER.T @ MIXER, 4.0 * np.eye(4), atol=1e-12)


def test_mode_projection_round_trips():
    rng = np.random.default_rng(0)
    w = rng.normal(80.0, 5.0, (NUM_ROTORS, 64))
    np.testing.assert_allclose(rps_from_modes(modes_from_rps(w)), w, atol=1e-10)


def test_single_rotor_move_splits_evenly_across_modes():
    """A move of delta on ONE rotor is delta/4 in every mode — the decomposition
    the whole transition prior is built on."""
    dw = np.zeros((NUM_ROTORS, 1))
    dw[0, 0] = 2.0
    np.testing.assert_allclose(modes_from_rps(dw)[:, 0], np.full(4, 0.5), atol=1e-12)
    dw_common = np.full((NUM_ROTORS, 1), 2.0)
    np.testing.assert_allclose(
        modes_from_rps(dw_common)[:, 0], np.array([2.0, 0.0, 0.0, 0.0]), atol=1e-12
    )


# --------------------------------------------------------------------------
# (b) permutation invariance of the transition cost


def _cost(prior: OUPrior, w_prev, w_new, mu_w):
    a_np, s_np = prior.coefficients()
    b = torch.as_tensor(MIXER, dtype=torch.float64) / NUM_ROTORS
    return float(
        _mode_cost(
            torch.as_tensor(w_prev, dtype=torch.float64),
            torch.as_tensor(w_new, dtype=torch.float64),
            torch.as_tensor(mu_w, dtype=torch.float64) @ b,
            torch.as_tensor(a_np, dtype=torch.float64),
            torch.as_tensor(s_np, dtype=torch.float64),
            prior.huber_knee,
        )
    )


@pytest.mark.parametrize("perm", list(itertools.permutations(range(NUM_ROTORS))))
def test_transition_cost_is_permutation_invariant(perm):
    """Relabelling the rotors consistently must not change the cost.

    Rotor identity is arbitrary under PIT, so a prior that is not invariant
    here would make the tracker's answer depend on a meaningless labelling.
    Invariance holds because a permutation maps the differential subspace to
    itself orthogonally — but ONLY when roll/pitch/yaw share one scale.
    """
    prior = OUPrior(tau_common=1.3)  # finite tau: exercises the mu term too
    rng = np.random.default_rng(7)
    w_prev = rng.normal(80.0, 3.0, NUM_ROTORS)
    w_new = w_prev + rng.normal(0.0, 0.5, NUM_ROTORS)
    mu_w = rng.normal(80.0, 3.0, NUM_ROTORS)
    p = list(perm)
    got = _cost(prior, w_prev[p], w_new[p], mu_w[p])
    assert got == pytest.approx(_cost(prior, w_prev, w_new, mu_w), rel=1e-10)


def test_per_mode_scales_would_break_invariance():
    """Guard on the reasoning, not the code: with DIFFERENT differential scales
    the cost is provably not invariant, which is why the dataclass exposes one.
    """
    b = MIXER.T / NUM_ROTORS
    rng = np.random.default_rng(3)
    dw = rng.normal(0.0, 1.0, NUM_ROTORS)
    s_bad = np.array([1.0, 0.3, 0.9, 2.0])
    costs = {
        round(float(np.sum(((b @ dw[list(p)]) / s_bad) ** 2)), 9)
        for p in itertools.permutations(range(NUM_ROTORS))
    }
    assert len(costs) > 1
    s_good = np.array([1.0, 0.5, 0.5, 0.5])
    costs_ok = {
        round(float(np.sum(((b @ dw[list(p)]) / s_good) ** 2)), 9)
        for p in itertools.permutations(range(NUM_ROTORS))
    }
    assert len(costs_ok) == 1


def test_random_walk_common_holds_any_level_for_free():
    """The asymmetric prior's whole point: a sustained COMMON offset costs
    nothing, a sustained single-rotor offset costs a fixed rate per frame."""
    prior = OUPrior()  # tau_common = inf
    r = prior.sustained_cost_rate(2.0)
    assert r["common"] == pytest.approx(0.0, abs=1e-12)
    assert r["single_rotor"] > 0.0
    assert math.isinf(r["ratio"])


def test_finite_tau_common_penalises_a_ramp():
    """And why it is infinite by default: a finite tau_common charges a takeoff.

    A real DREGON takeoff moves the common mode 54-57 rev/s away from where the
    window started.  Under a mean-reverting common mode holding that excursion
    costs ~0.3-0.5 per frame, i.e. 150-250 over a 500-frame 16 s window —
    comparable to the whole emission budget (lambda_e * 4 per frame at a
    normalised peak of 1), so the DP would rather drag the cruise plateau back
    towards the idle plateau it started from.  Under a random walk it is free.
    """
    n_frames = 500  # 16 s at 32 ms
    rw = OUPrior().sustained_cost_rate(56.0)["common"] * n_frames
    ou = OUPrior(tau_common=1.28).sustained_cost_rate(56.0)["common"] * n_frames
    assert rw == pytest.approx(0.0, abs=1e-9)
    assert ou > 100.0


# --------------------------------------------------------------------------
# (c) the k-scaled emission bandwidth


#: Spectral line width in bins.  A windowed sinusoid is not a delta on a bin
#: lattice — its main lobe spans ~2 bins — and that shape is the whole reason a
#: comb candidate loses its HIGH harmonics first: the k-th tooth of a candidate
#: off by `d` rev/s sits `k*d` Hz from the true line, so it falls off the lobe
#: at a k-dependent offset.  Depositing deltas at rounded bins (the obvious toy)
#: destroys exactly the effect under test.
LINE_SIGMA_BINS = 0.7


def _deposit(lm, t, f, bin_hz, amp=1.0):
    """Add a Gaussian spectral line centred at frequency ``f``."""
    x = f / bin_hz
    lo = max(0, int(np.floor(x - 3 * LINE_SIGMA_BINS)))
    hi = min(lm.shape[0] - 1, int(np.ceil(x + 3 * LINE_SIGMA_BINS)))
    if lo >= hi:
        return
    j = np.arange(lo, hi + 1)
    lm[j, t] += amp * np.exp(-((j - x) ** 2) / (2 * LINE_SIGMA_BINS**2))


#: Harmonic amplitude law: the 2-blade blade-pass emphasis of
#: ``rps_refine_lab.synth_window`` (even harmonics 1.6/k, odd 0.5/k).  The 1/k
#: DECAY is the load-bearing part: with flat harmonics a 4-rotor comb out to
#: k=16 puts ~64 equal lines into 200 bins, every candidate finds teeth
#: somewhere, and the score surface is junk regardless of the tracker.
def _amp(k: int) -> float:
    return (1.6 if k % 2 == 0 else 0.5) / k


def _toy_spec(bases, n_frames=8, n_f=1025, bin_hz=7.8125, k_max=12):
    lm = np.zeros((n_f, n_frames), dtype=np.float32)
    for b in bases:
        for k in range(1, k_max + 1):
            for t in range(n_frames):
                _deposit(lm, t, k * b, bin_hz, amp=_amp(k))
    return torch.from_numpy(lm), bin_hz


def test_b0_zero_reproduces_point_sampling():
    lm, bin_hz = _toy_spec([80.0])
    cfg0 = EmissionCfg(b0_rps=0.0, n_band=5)
    cfg1 = EmissionCfg(b0_rps=0.0, n_band=1)
    torch.testing.assert_close(
        comb_scores(lm, bin_hz, cfg0), comb_scores(lm, bin_hz, cfg1), rtol=0, atol=0
    )


def test_b0_zero_matches_the_lab_single_comb_score():
    """The emission must be the SAME contrast the existing chain uses, so the
    two are comparable: at b0 = 0 and uniform weights it reproduces
    ``rps_refine_lab._single_comb_scores`` on a zero base row."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
    import rps_refine_lab as lab  # noqa: PLC0415

    lm, bin_hz = _toy_spec([80.0, 84.0], n_frames=4)
    grid = np.arange(60.0, 100.0, 0.5)
    ref = lab._single_comb_scores(lm.numpy(), bin_hz, np.zeros(lm.shape[1]), grid)
    cfg = EmissionCfg(
        b0_rps=0.0,
        k_max=lab.M1_K_SCORE,
        k_weight="uniform",
        f_min=lab.COARSE_F_MIN,
        f_max=6000.0,
    )
    got = comb_scores(lm, bin_hz, cfg, torch.as_tensor(grid, dtype=torch.float32))
    np.testing.assert_allclose(got.numpy(), ref, atol=1e-5)


def test_k_scaled_band_equalises_the_capture_radius():
    """The point of ``B_k = k*B0``.

    Put a SINGLE line at ``k * 80`` Hz and score a candidate at ``80 + off``
    with ``k_max = k``: only harmonic ``k`` can find anything, so the score
    times ``k`` is exactly what that harmonic contributes.  With a fixed
    bandwidth (``b0 = 0``: the FFT bin) the k-th tooth of a candidate off by
    ``off`` rev/s is ``k*off`` Hz away, so it is lost once ``k*off`` exceeds the
    bin — the capture radius is ``bin/k``.  With ``B_k = k*b0`` it is ``b0``
    rev/s at every ``k``.
    """
    bin_hz, n_f = 7.8125, 4096
    off = 1.5  # rev/s

    def contribution(k: int, b0: float) -> float:
        lm = np.zeros((n_f, 2), dtype=np.float32)
        for t in range(2):
            _deposit(lm, t, k * 80.0, bin_hz)
        cfg = EmissionCfg(b0_rps=b0, k_max=k, k_weight="uniform", f_min=1.0)
        g = torch.tensor([80.0 + off], dtype=torch.float32)
        return float(comb_scores(torch.from_numpy(lm), bin_hz, cfg, g)[0, 0]) * k

    fixed = [contribution(k, 0.0) for k in (2, 8)]
    scaled = [contribution(k, off) for k in (2, 8)]
    assert fixed[0] > 0.7  # k=2: off by 3 Hz, still on the lobe
    assert fixed[1] < 0.25 * fixed[0]  # k=8: off by 12 Hz -> tooth lost
    assert scaled[1] > 0.9 * scaled[0]  # k-scaled band: recovered at both k
    # 0.775, not 1.0, is the ceiling: linear interpolation between bin
    # samples under-reads a peaked line even when centred on it.
    assert scaled[0] > 0.7


def test_k_weighting_favours_high_harmonics():
    ks = torch.arange(1, 9, dtype=torch.float64)
    w_k = (ks / ks.sum()).numpy()
    assert w_k[-1] > w_k[0]
    assert w_k.sum() == pytest.approx(1.0)


# --------------------------------------------------------------------------
# (d) end-to-end wiring on a synthetic window


def _synth_window(bases, n_frames=60, drift=0.0):
    """Whitened-spectrogram stand-in with four moving combs and known truth."""
    bin_hz, n_f, k_max = 7.8125, 1025, 12
    lm = np.full((n_f, n_frames), -0.05, dtype=np.float32)
    truth = np.zeros((NUM_ROTORS, n_frames))
    for i, b in enumerate(bases):
        truth[i] = b + drift * np.sin(2 * np.pi * np.arange(n_frames) / n_frames + i)
        for t in range(n_frames):
            for k in range(1, k_max + 1):
                _deposit(lm, t, k * truth[i, t], bin_hz, amp=_amp(k))
    st = np.arange(n_frames) * FRAME_S
    return lm, bin_hz, st, truth


def test_tracker_recovers_a_forced_single_candidate():
    """Scoring + backtracking wiring check.

    With the grid restricted to exactly the four true speeds there is only one
    admissible 4-subset, so the tracker MUST return it; anything else means the
    beam bookkeeping or the backtrack is wrong.
    """
    bases = [74.0, 81.0, 88.0, 95.0]
    lm, bin_hz, st, _ = _synth_window(bases, n_frames=40)
    emis = EmissionCfg(lo=74.0, hi=95.0, step=7.0, b0_rps=0.5)
    np.testing.assert_allclose(emis.grid(), bases, atol=1e-6)
    beam = BeamCfg(width=16, n_global=4, n_peaks=4, n_local=3, local_half_rps=8.0)
    traj, diag = joint_beam_track(lm, bin_hz, st, st, emis=emis, beam=beam)
    np.testing.assert_allclose(np.sort(traj.mean(axis=1)), bases, atol=1e-6)
    assert diag["beam_distinct_min"] >= 1


def _objective(lm, bin_hz, W, emis, beam, ou):
    """Total cost of a trajectory ``W`` (T, 4) under the tracker's OWN objective."""
    import data_processing.joint_beam_tracker as jbt  # noqa: PLC0415

    tab = jbt.comb_tables(torch.from_numpy(lm), bin_hz, emis)
    raw = jbt.comb_scores_from_tables(tab)
    nrm_a, nrm_b = jbt._norm_affine(raw, emis)
    g = torch.as_tensor(emis.grid(), dtype=torch.float32)
    b = torch.as_tensor(MIXER, dtype=torch.float32) / NUM_ROTORS
    a_np, s_np = ou.coefficients()
    a = torch.as_tensor(a_np, dtype=torch.float32)
    sv = torch.as_tensor(s_np, dtype=torch.float32)
    W = torch.as_tensor(W, dtype=torch.float32)
    mu = W[0] @ b
    tot = 0.0
    for t in range(W.shape[0]):
        u, mass = union_emission(tab, jbt._to_idx(W[t][None], g), t)
        tot += -beam.lambda_e * float(nrm_a[t] * u + mass * nrm_b[t])
        tot += float(_band_penalty(W[t][None], beam)[0])
        if t > 0:
            tot += float(jbt._mode_cost(W[t - 1], W[t], mu, a, sv, ou.huber_knee))
    return tot


def test_beam_finds_a_solution_at_least_as_good_as_the_truth():
    """Search correctness, stated so it cannot be confused with accuracy.

    A tracker can be wrong two ways: the SEARCH fails to find the best
    trajectory under its objective, or the OBJECTIVE prefers the wrong
    trajectory.  These need different fixes and the distinction is easy to lose,
    so test the first one directly: the returned path must cost no more than the
    ground-truth path, snapped to the same grid, under the tracker's own
    objective.

    On this synthetic it holds with room to spare (-713 vs -647), which says the
    remaining error is the objective's, not the beam's.  The specific reason
    here is a property of the toy rather than of the tracker: with four
    equal-amplitude combs and no sibling masking, the LOWEST rotor's half-tooth
    reference ((k-0.5)*c) lands on the higher rotors' teeth, so its contrast
    sits BELOW the per-frame median and the emission genuinely prefers a
    duplicate of a strong comb to the true weak one.  The real chain masks
    siblings in M1 for exactly this reason.  Accuracy is therefore measured on
    real windows (`scripts/jb_sweep.py --mode init`), not here.
    """
    bases = [72.0, 80.0, 88.0, 96.0]
    lm, bin_hz, st, truth = _synth_window(bases, n_frames=40, drift=1.5)
    emis = EmissionCfg(lo=66.0, hi=102.0, step=0.5)
    beam = BeamCfg(width=64, n_global=8, n_peaks=10, n_local=3)
    ou = OUPrior()
    traj, _ = joint_beam_track(lm, bin_hz, st, st, emis=emis, beam=beam, ou=ou)
    gt = np.round(truth.T / emis.step) * emis.step
    assert _objective(lm, bin_hz, traj.T, emis, beam, ou) <= _objective(
        lm, bin_hz, gt, emis, beam, ou
    )


def test_tracker_admits_an_unresolvable_twin_pair():
    """End-to-end counterpart: with two rotors 1 rev/s apart — below what a
    k <= 8 comb on 7.8 Hz bins can resolve — the tracker must still put TWO
    tracks in that band rather than refuse the pair and invent a rotor
    elsewhere."""
    bases = [74.0, 75.0, 82.0, 90.0]
    lm, bin_hz, st, _ = _synth_window(bases, n_frames=40)
    emis = EmissionCfg(lo=66.0, hi=102.0, step=0.5)
    beam = BeamCfg(width=64, n_global=8, n_peaks=12, n_local=3)
    traj, _ = joint_beam_track(lm, bin_hz, st, st, emis=emis, beam=beam)
    means = np.sort(traj.mean(axis=1))
    assert ((means > 72.0) & (means < 77.0)).sum() == 2, means
    assert abs(means[2] - 82.0) < 1.0 and abs(means[3] - 90.0) < 1.0


def test_score_trajectory_matches_the_reference_objective():
    """`score_trajectory` is the load-bearing measurement instrument of the
    objective-vs-search diagnosis, so it is pinned against this file's
    independent reimplementation of the same cost."""
    import data_processing.joint_beam_tracker as jbt  # noqa: PLC0415

    bases = [72.0, 80.0, 88.0, 96.0]
    lm, bin_hz, st, truth = _synth_window(bases, n_frames=30, drift=1.5)
    emis = EmissionCfg(lo=66.0, hi=102.0, step=0.5)
    beam, ou = BeamCfg(), OUPrior()
    gt = np.round(truth.T / emis.step) * emis.step  # (T, 4), on grid
    obj = jbt.build_objective(lm, bin_hz, ou=ou, emis=emis, beam=beam)
    got = jbt.score_trajectory(obj, gt.T)
    assert got["total"] == pytest.approx(_objective(lm, bin_hz, gt, emis, beam, ou), rel=1e-4)
    parts = got["emission"] + got["transition"] + got["band"]
    assert got["total"] == pytest.approx(parts, rel=1e-6)


def test_sorted_dedup_trades_objective_value_for_marginal_width():
    """Merging permuted states is NOT free, and this pins the trade it makes.

    The merge looks exact — emission and transition cost are both
    permutation-invariant — but `mu` is inherited from each state's own
    ancestry, so two states that are permutations of each other now can carry
    trims that are not, and merging discards a live hypothesis.  What it buys
    is a wider per-rotor marginal.  Both directions are asserted so that a
    future change cannot silently turn a measured trade into a free lunch."""
    bases = [74.0, 76.0, 82.0, 90.0]
    lm, bin_hz, st, _ = _synth_window(bases, n_frames=30, drift=1.0)
    emis = EmissionCfg(lo=66.0, hi=102.0, step=0.5)
    kw: dict[str, Any] = {"width": 64, "n_global": 8, "n_peaks": 12, "n_local": 3}
    _, d_on = joint_beam_track(lm, bin_hz, st, st, emis=emis, beam=BeamCfg(dedup_sorted=True, **kw))
    _, d_off = joint_beam_track(
        lm, bin_hz, st, st, emis=emis, beam=BeamCfg(dedup_sorted=False, **kw)
    )
    assert d_on["beam_marginal_min_mean"] > d_off["beam_marginal_min_mean"]
    # it costs objective value, but only marginally — a large regression here
    # means the merge is destroying hypotheses, not just permuted copies
    assert d_on["final_cost_best"] <= 0.99 * d_off["final_cost_best"]


def test_diversity_reserve_widens_the_per_rotor_marginal():
    """The reserve exists to stop a rotor's alternatives being crowded out by
    states that differ only in a rotor that is already doing well."""
    bases = [74.0, 76.0, 82.0, 90.0]
    lm, bin_hz, st, _ = _synth_window(bases, n_frames=30, drift=1.0)
    emis = EmissionCfg(lo=66.0, hi=102.0, step=0.5)
    kw: dict[str, Any] = {"width": 64, "n_global": 8, "n_peaks": 12, "n_local": 3}
    _, plain = joint_beam_track(lm, bin_hz, st, st, emis=emis, beam=BeamCfg(**kw))
    _, div = joint_beam_track(
        lm, bin_hz, st, st, emis=emis, beam=BeamCfg(diversity_reserve=32, **kw)
    )
    assert div["beam_marginal_min_mean"] > plain["beam_marginal_min_mean"]
    # and the beam still holds `width` states — the reserve reallocates, not adds
    assert div["beam_distinct_mean"] > 0


@pytest.mark.parametrize("pool", ["mean", "quantile", "frac_pos"])
def test_union_accounting_survives_the_pooling_change(pool):
    """The union's contract must hold under every pooling mode.

    `comb_mass` is what the affine normalisation uses to apply the per-frame
    median shift once per DISTINCT comb, so four coincident combs must count as
    one and a spread assignment must agree with the verified mean path.  Getting
    it wrong silently reintroduces a bias for or against degeneracy — the exact
    failure the union was built to remove.

    Bases are chosen with no small-integer relations.  A harmonically related
    set is a separate case, tested below.
    """
    import data_processing.joint_beam_tracker as jbt  # noqa: PLC0415

    bases = [71.0, 83.0, 91.0, 97.0]
    lm, bin_hz, _st, _ = _synth_window(bases, n_frames=8)
    emis = EmissionCfg(lo=60.0, hi=110.0, step=0.5, pool=pool)
    mean_emis = dc_replace(emis, pool="mean")
    tab = jbt.comb_tables(torch.from_numpy(lm), bin_hz, emis)
    tab_mean = jbt.comb_tables(torch.from_numpy(lm), bin_hz, mean_emis)
    g = torch.as_tensor(emis.grid(), dtype=torch.float32)
    spread = jbt._to_idx(torch.tensor([bases], dtype=torch.float32), g)
    degen = jbt._to_idx(torch.tensor([[bases[0]] * 4], dtype=torch.float32), g)
    s_spread, m_spread = jbt.union_emission(tab, spread, 4, emis)
    s_degen, m_degen = jbt.union_emission(tab, degen, 4, emis)
    assert float(m_degen) == pytest.approx(1.0)
    # Four DISTINCT combs explain a little less than four even here, because
    # some harmonics still collide — so the bar is agreement with the mean
    # path's accounting, not a bare 4.0.
    assert float(m_spread) == pytest.approx(
        float(jbt.union_emission(tab_mean, spread, 4, mean_emis)[1]), rel=1e-6
    )
    assert 3.0 < float(m_spread) < 4.0
    assert float(s_spread) > float(s_degen)


def test_quantile_pooling_is_sensitive_to_a_contaminated_half_tooth_reference():
    """A characterised weakness of quantile pooling, pinned so it stays known.

    On this toy the four combs are equal-amplitude, harmonically related and
    unmasked, so the LOWEST rotor's half-tooth reference ((k-0.5)*72) lands on
    the higher rotors' teeth and many of its per-tooth contrasts go negative.
    A weighted MEAN averages that away; a low quantile is driven by exactly
    those worst teeth, so the spread assignment scores BELOW the degenerate one
    (-0.010 vs +0.006) — the collapse the union exists to prevent.

    It does not fire on non-harmonic bases (the test above) and quantile
    pooling wins clearly on real windows (WP20: worst-rotor acquisition 0.448
    vs 0.383 for the best mean configuration), so this is a property of an
    unmasked harmonic toy rather than a reason to drop the pooling.  It is the
    reason sibling masking has to come BEFORE quantile pooling is trusted on a
    harmonically related rotor set.
    """
    import data_processing.joint_beam_tracker as jbt  # noqa: PLC0415

    bases = [72.0, 80.0, 88.0, 96.0]  # every base an integer multiple of 8
    lm, bin_hz, _st, _ = _synth_window(bases, n_frames=8)
    g_of = {}
    for pool in ("mean", "quantile"):
        emis = EmissionCfg(lo=60.0, hi=110.0, step=0.5, pool=pool)
        tab = jbt.comb_tables(torch.from_numpy(lm), bin_hz, emis)
        g = torch.as_tensor(emis.grid(), dtype=torch.float32)
        spread = jbt._to_idx(torch.tensor([bases], dtype=torch.float32), g)
        degen = jbt._to_idx(torch.tensor([[72.0] * 4], dtype=torch.float32), g)
        g_of[pool] = (
            float(jbt.union_emission(tab, spread, 4, emis)[0]),
            float(jbt.union_emission(tab, degen, 4, emis)[0]),
        )
    assert g_of["mean"][0] > g_of["mean"][1]  # the mean is robust here
    assert g_of["quantile"][0] < g_of["quantile"][1]  # the quantile is not
