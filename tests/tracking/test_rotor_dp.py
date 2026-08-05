"""Unit tests for the exact single-rotor Viterbi lattice tracker.

Cheap and deterministic — no real audio.  Three groups:

(a) the DP itself: exactness against brute-force enumeration (the load-bearing
    test — "exact" is the module's entire claim), the hard step band, and the
    Huber knee being what makes a linear ramp affordable;
(b) residual masking: claimed teeth leave the claimed comb at the floor and a
    disjoint comb untouched, and zero claims reproduce
    ``comb_scores_from_tables`` exactly;
(c) greedy peel: two well-separated combs are extracted as two distinct
    speeds, never the same comb twice.

Run:  pytest tests/test_rotor_dp.py
"""

import itertools

import numpy as np
import pytest
import torch

from tracking.joint_beam_tracker import (
    EmissionCfg,
    comb_scores_from_tables,
    comb_tables,
)
from tracking.rotor_dp import (
    LatticeCfg,
    greedy_peel,
    residual_scores,
    track_masked,
    viterbi_path,
)

# --------------------------------------------------------------------------
# (a) the DP itself


def _huber(z: float, knee: float) -> float:
    return 0.5 * z * z if z <= knee else knee * (z - 0.5 * knee)


def _path_cost(path, surf: np.ndarray, grid: np.ndarray, cfg: LatticeCfg) -> float:
    """The module's cost formula, reimplemented independently in numpy."""
    c = -cfg.lambda_e * sum(float(surf[path[t], t]) for t in range(surf.shape[1]))
    for t in range(1, surf.shape[1]):
        gap = float(abs(grid[path[t]] - grid[path[t - 1]]))
        if gap > cfg.max_step_rps + 1e-9:
            return float("inf")
        c += _huber(gap / cfg.s_rps, cfg.huber_knee)
    return float(c)


def test_viterbi_total_cost_matches_brute_force_enumeration():
    """The load-bearing test: on a problem small enough to enumerate (8^5
    paths), the banded DP must return exactly the global minimum of the cost
    it claims to minimise.  Everything else in the module rides on this."""
    rng = np.random.default_rng(0)
    d_n, t_n = 8, 5
    grid_np = 70.0 + 0.5 * np.arange(d_n)
    surf_np = rng.normal(0.0, 1.0, (d_n, t_n)).astype(np.float32)
    cfg = LatticeCfg(s_rps=0.4, huber_knee=1.5, max_step_rps=1.2, lambda_e=3.0)

    path, total = viterbi_path(
        torch.from_numpy(surf_np), torch.as_tensor(grid_np, dtype=torch.float32), cfg
    )

    brute = min(
        _path_cost(p, surf_np, grid_np, cfg) for p in itertools.product(range(d_n), repeat=t_n)
    )
    assert total == pytest.approx(brute, abs=1e-4)
    # and the returned path actually attains the returned cost
    assert _path_cost(path.tolist(), surf_np, grid_np, cfg) == pytest.approx(total, abs=1e-4)


def test_forbidden_jump_never_appears_even_when_the_surface_begs_for_it():
    """The band is a hard constraint, not a large-but-finite cost: a reward
    that only a > max_step_rps jump could reach must not induce one."""
    d_n, t_n = 41, 12
    grid = 60.0 + 0.5 * torch.arange(d_n, dtype=torch.float32)
    surf = torch.zeros((d_n, t_n))
    surf[0, :6] = 5.0
    surf[-1, 6:] = 500.0  # the surface begs for an instant 20 rev/s jump
    cfg = LatticeCfg(max_step_rps=1.0)
    path, _ = viterbi_path(surf, grid, cfg)
    steps = np.abs(np.diff(grid.numpy()[path.numpy()]))
    assert steps.max() <= 1.0 + 1e-6


def test_huber_knee_is_what_makes_a_linear_ramp_affordable():
    """A ridge moving 1 rev/s per frame is followed under the Huber cost, and
    NOT followed when the same innovation scale is charged quadratically all
    the way (knee pushed past the ramp's z) — i.e. the knee, not the scale, is
    what keeps takeoffs affordable.  Constructed so the per-frame numbers
    bracket the emission gain: gain 3 * 2.0 = 6; Huber cost of z = 4 at knee
    1.5 is 4.875 < 6; quadratic cost is 8 > 6."""
    d_n, t_n = 60, 20
    grid = 60.0 + 0.5 * torch.arange(d_n, dtype=torch.float32)
    ridge = 2 * np.arange(t_n)  # +1 rev/s per frame on a 0.5 grid
    surf = torch.zeros((d_n, t_n))
    surf[ridge, np.arange(t_n)] = 2.0

    hub = LatticeCfg(s_rps=0.25, huber_knee=1.5, max_step_rps=5.0, lambda_e=3.0)
    path_h, _ = viterbi_path(surf, grid, hub)
    np.testing.assert_array_equal(path_h.numpy(), ridge)

    quad = LatticeCfg(s_rps=0.25, huber_knee=1e6, max_step_rps=5.0, lambda_e=3.0)
    path_q, _ = viterbi_path(surf, grid, quad)
    g = grid.numpy()
    travel_h = abs(g[path_h[-1]] - g[path_h[0]])
    travel_q = abs(g[path_q[-1]] - g[path_q[0]])
    assert travel_h == pytest.approx((t_n - 1) * 1.0)  # the full 19 rev/s ramp
    assert travel_q < 0.5 * travel_h  # the quadratic path flattens


# --------------------------------------------------------------------------
# (b) residual masking

#: Gaussian line width in bins — a windowed sinusoid is not a delta (see the
#: joint-beam tests, where this shape is load-bearing for the same reason).
LINE_SIGMA_BINS = 0.7


def _deposit(lm: np.ndarray, t: int, f: float, bin_hz: float, amp: float = 1.0) -> None:
    x = f / bin_hz
    lo = max(0, int(np.floor(x - 3 * LINE_SIGMA_BINS)))
    hi = min(lm.shape[0] - 1, int(np.ceil(x + 3 * LINE_SIGMA_BINS)))
    if lo >= hi:
        return
    j = np.arange(lo, hi + 1)
    lm[j, t] += amp * np.exp(-((j - x) ** 2) / (2 * LINE_SIGMA_BINS**2))


def _amp(k: int) -> float:
    """2-blade blade-pass emphasis with 1/k decay (the lab's synth law)."""
    return (1.6 if k % 2 == 0 else 0.5) / k


def _two_comb_setup(n_frames: int = 6):
    """Toy spectrogram with combs at 70 and 90 rev/s + tables on a 60-100 grid.

    70 and 90 share NO rounded on-tooth bin up to k = 8 (7 k1 = 9 k2 first at
    k1 = 9), so claiming one comb must leave the other's scores untouched."""
    bin_hz, n_f = 7.8125, 1025
    bases = (70.0, 90.0)
    lm = np.zeros((n_f, n_frames), dtype=np.float32)
    for b in bases:
        for k in range(1, 13):
            for t in range(n_frames):
                _deposit(lm, t, k * b, bin_hz, amp=_amp(k))
    emis = EmissionCfg(lo=60.0, hi=100.0, step=0.5, k_max=8, pool="quantile", pool_q=0.25)
    grid = torch.as_tensor(emis.grid(), dtype=torch.float32)
    tab = comb_tables(torch.from_numpy(lm), bin_hz, emis, grid)
    return tab, emis, grid, bases


def _grid_index(grid: torch.Tensor, speed: float) -> int:
    return int(torch.argmin((grid - speed).abs()))


def test_residual_scores_with_no_claims_equals_comb_scores_from_tables():
    """R = 0 must be a bit-exact no-op, so the probe's raw_dp arm and the
    joint tracker read the same surface."""
    tab, emis, _grid, _ = _two_comb_setup()
    empty = torch.zeros((0, tab.v_on.shape[2]), dtype=torch.long)
    torch.testing.assert_close(
        residual_scores(tab, emis, empty), comb_scores_from_tables(tab, emis), rtol=0, atol=0
    )


def test_claiming_a_comb_floors_it_and_leaves_a_disjoint_comb_unchanged():
    """The masking contract, both directions: the claimed comb's own speed has
    every valid tooth excluded (zero survivors -> the per-frame floor, exactly
    min(scored) - 1), while a bin-disjoint comb keeps its score bit-exactly.
    Exact-bin masking (halfwidth 0) so bin-disjointness of the toy pair holds."""
    tab, emis, grid, (c1, c2) = _two_comb_setup()
    t_n = tab.v_on.shape[2]
    i1, i2 = _grid_index(grid, c1), _grid_index(grid, c2)

    empty = torch.zeros((0, t_n), dtype=torch.long)
    raw0 = residual_scores(tab, emis, empty, mask_halfwidth_bins=0)
    claimed = torch.full((1, t_n), i1, dtype=torch.long)
    raw1 = residual_scores(tab, emis, claimed, mask_halfwidth_bins=0)

    assert float(raw0[i1].mean()) > 0.05  # the comb is actually there
    # the disjoint comb: bit-exactly unchanged
    torch.testing.assert_close(raw1[i2], raw0[i2], rtol=0, atol=0)
    # the claimed speed: substantial drop, and exactly the documented floor
    assert float(raw1[i1].mean()) < float(raw0[i1].mean()) - 0.5
    for t in range(t_n):
        col = raw1[:, t]
        floor = float(col.min())
        assert float(raw1[i1, t]) == pytest.approx(floor)
        scored = col[col > floor + 1e-9]
        assert floor == pytest.approx(float(scored.min()) - 1.0, abs=1e-5)


def test_greedy_peel_extracts_two_separated_combs_without_collapsing():
    """Two well-separated combs, two peels: each within 0.5 rev/s of a truth,
    and never the same comb twice — the residual masking is what forbids the
    second pass from re-claiming the first comb."""
    tab, emis, grid, bases = _two_comb_setup(n_frames=12)
    out = greedy_peel(tab, emis, LatticeCfg(), n_rotors=2, grid=grid)
    means = np.sort(out["speeds"].mean(axis=1))
    np.testing.assert_allclose(means, np.sort(bases), atol=0.5)
    assert abs(means[1] - means[0]) > 5.0  # distinct combs, not one twice


def test_track_masked_reports_raw_support_along_the_path():
    """`support_raw_mean` is the honest support number: on a window with a
    real comb the first track's raw support is clearly positive, and a track
    forced onto a fully-claimed surface (both combs claimed) sits at or below
    the floor of the raw scores."""
    tab, emis, grid, (c1, c2) = _two_comb_setup()
    t_n = tab.v_on.shape[2]
    res = track_masked(tab, emis, LatticeCfg(), None, grid)
    assert res["support_raw_mean"] > 0.05
    assert res["speeds"].shape == (t_n,)
    both = torch.stack(
        [
            torch.full((t_n,), _grid_index(grid, c1), dtype=torch.long),
            torch.full((t_n,), _grid_index(grid, c2), dtype=torch.long),
        ]
    )
    res2 = track_masked(tab, emis, LatticeCfg(), both, grid)
    assert res2["support_raw_mean"] < res["support_raw_mean"]


def test_dilated_masking_kills_a_flank_impostor_but_spares_a_true_twin():
    """The first probe run's measured failure, reproduced and fixed in one toy:
    with exact-bin masking a candidate displaced 0.2 rev/s from a claimed comb
    keeps teeth on ADJACENT bins that read the claimed teeth's mainlobe flanks,
    so the claimed comb survives as its own flank (every oracle-masked DP
    reproduced the raw trajectory; the greedy peel collapsed to one comb).
    Dilating the claim by the Hann mainlobe halfwidth (+-2 bins at n_fft 4096)
    kills the flank at every k while a genuine 0.9 rev/s twin keeps its
    high-k teeth, whose bins clear the dilated set."""
    bin_hz, n_f, n_frames = 3.90625, 2049, 4
    lm = np.zeros((n_f, n_frames), dtype=np.float32)
    for b in (90.0, 90.9):
        for k in range(1, 21):
            for t in range(n_frames):
                _deposit(lm, t, k * b, bin_hz, amp=_amp(k))
    emis = EmissionCfg(lo=85.0, hi=95.0, step=0.1, k_max=16, pool="quantile", pool_q=0.25)
    grid = torch.as_tensor(emis.grid(), dtype=torch.float32)
    tab = comb_tables(torch.from_numpy(lm), bin_hz, emis, grid)
    i_base = _grid_index(grid, 90.0)
    i_flank = _grid_index(grid, 90.2)
    i_twin = _grid_index(grid, 90.9)

    empty = torch.zeros((0, n_frames), dtype=torch.long)
    raw0 = residual_scores(tab, emis, empty)
    claimed = torch.full((1, n_frames), i_base, dtype=torch.long)
    r_w0 = residual_scores(tab, emis, claimed, mask_halfwidth_bins=0)
    r_w2 = residual_scores(tab, emis, claimed, mask_halfwidth_bins=2)

    # the defect: exact-bin masking leaves the flank impostor with real score
    assert float(r_w0[i_flank].mean()) > float(r_w2[i_flank].mean()) + 0.02
    # the fix: at the mainlobe halfwidth the impostor loses its support...
    assert float(r_w2[i_flank].mean()) < float(raw0[i_flank].mean()) - 0.05
    # ...while the true twin keeps genuine support on its surviving teeth
    assert float(r_w2[i_twin].mean()) > float(r_w2[i_flank].mean()) + 0.02
    assert float(r_w2[i_twin].mean()) > 0.0


def test_min_surviving_teeth_floors_a_barely_cleared_impostor():
    """A dilated mask alone is not enough (second probe run): an impostor
    0.7 rev/s from a claimed comb clears the +-2-bin dilation on only its 3
    highest teeth, which read the claimed teeth's jitter-broadened skirts and
    still pool positive.  Requiring a minimum survivor count floors it, while
    a 0.9 rev/s twin keeps enough independent teeth to stay scoreable."""
    bin_hz, n_f, n_frames = 3.90625, 2049, 4
    lm = np.zeros((n_f, n_frames), dtype=np.float32)
    for b in (90.0, 90.9):
        for k in range(1, 21):
            for t in range(n_frames):
                _deposit(lm, t, k * b, bin_hz, amp=_amp(k))
    emis = EmissionCfg(lo=85.0, hi=95.0, step=0.1, k_max=16, pool="quantile", pool_q=0.25)
    grid = torch.as_tensor(emis.grid(), dtype=torch.float32)
    tab = comb_tables(torch.from_numpy(lm), bin_hz, emis, grid)
    i_base = _grid_index(grid, 90.0)
    i_imp = _grid_index(grid, 90.7)  # clears the dilation only at k >= 14
    i_twin = _grid_index(grid, 90.9)

    claimed = torch.full((1, n_frames), i_base, dtype=torch.long)
    r_any = residual_scores(tab, emis, claimed, mask_halfwidth_bins=2, min_surv_teeth=1)
    r_min4 = residual_scores(tab, emis, claimed, mask_halfwidth_bins=2, min_surv_teeth=4)

    def is_floor(r: torch.Tensor, i: int) -> bool:
        return all(float(r[i, t]) == pytest.approx(float(r[:, t].min())) for t in range(n_frames))

    # without the requirement the impostor keeps a live (non-floor) score
    assert not is_floor(r_any, i_imp)
    # with it, the impostor is floored while the twin stays scoreable
    assert is_floor(r_min4, i_imp)
    assert not is_floor(r_min4, i_twin)
    assert float(r_min4[i_twin].mean()) > float(r_min4[i_imp].mean()) + 0.02
