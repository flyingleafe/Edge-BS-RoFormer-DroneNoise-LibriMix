"""Acceptance tests of the telemetry refitter (issue 17 phase 6b).

The verification the issue asks for, in one place: on a SYNTHETIC comb whose
trajectory is known exactly, corrupt the truth the two ways DREGON's telemetry
is corrupted — a ``x1.005`` rate scale and the tachometer's 0.269 rev/s /
49.7 Hz staircase — and require the fitter to recover the truth. The bar is on
the SCALE, because that is the quantity the campaign is trying to measure:
recovered-versus-true scale error well under 0.1 %, against an injected 0.5 %.

The trajectory residual is reported but NOT used as a bar. The measurement is
that it does not shrink: the fitter's own per-frame noise on a cheap synthetic
window is 0.05-0.15 rev/s, larger than the staircase it replaces, so the
procedure buys the systematic scale and not the de-staircasing. That is issue
17's own warning ("the refined tracks carry two distinct corrections and they
must be reported separately") turned into a test.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from data_processing.rps_synthesis import synth_comb_window
from tracking import telemetry_refit as R
from tracking import top
from tracking.fitness import TACH_REFRESH_HZ, TACH_STEP_REV_S
from tracking.pipelines import make_peels
from tracking.top import get_rps, tracking_frame

SR = 16000
DUR_S = 4.0
FS_FT = 125.0  # fine enough to carry the 49.7 Hz staircase
#: Rotor means 85 / 67 / 97 / 91 rev/s — separated enough that no pair is a
#: twin, so the recovery is a test of the ladder rather than of the twin logic.
MODE_MEANS = (85.0, 3.0, -9.0, 6.0)


def _cfg(**over):
    """Fast settings: a low ceiling on both the ladder and the peel."""
    base: dict[str, Any] = dict(
        k_top=24,
        k_start_max=10,
        peel_k_max=20,
        f_max=7500.0,
        max_iters=4,
        min_iters=1,
        plateau_rel=0.0,
    )
    base.update(over)
    return R.RefitConfig(**base)


def staircase(r: np.ndarray, ft: np.ndarray) -> np.ndarray:
    """The DREGON tachometer's own corruption: quantise, then zero-order hold."""
    q = np.round(r / TACH_STEP_REV_S) * TACH_STEP_REV_S
    tick = np.floor(ft * TACH_REFRESH_HZ).astype(int)
    idx = np.searchsorted(tick, np.arange(tick[-1] + 1), side="left")
    held = np.empty_like(q)
    for j, start in enumerate(idx):
        stop = idx[j + 1] if j + 1 < len(idx) else q.shape[-1]
        held[:, start:stop] = q[:, start : start + 1]
    return held


@pytest.fixture(scope="module")
def window():
    """``(audio (2, T), ft (N,), r_true (4, N))`` from the project synthesizer."""
    w = synth_comb_window(
        0,
        mode_means=MODE_MEANS,
        aggressiveness=0.4,
        fc_hz=5.0,
        snr_db=6.0,
        dur=DUR_S,
        k_max=30,
        n_mic=2,
    )
    ft = np.arange(0.0, DUR_S, 1.0 / FS_FT)
    r_true = np.stack([np.interp(ft, w.t, w.r_true[i]) for i in range(w.r_true.shape[0])])
    return w.audio, ft, r_true


@pytest.fixture(scope="module")
def fits(window):
    """The fitter run once per corruption — the expensive fixture."""
    audio, ft, r_true = window
    cands = {
        "scale": r_true * 1.005,
        "staircase": staircase(r_true, ft),
        "both": staircase(r_true * 1.005, ft),
    }
    return {name: (c, R.refit_window(audio, c, ft, SR, cfg=_cfg())) for name, c in cands.items()}


# ---------------------------------------------------------------------------
# the acceptance criteria (synthetic recovery)


@pytest.mark.parametrize("case", ["scale", "staircase", "both"])
def test_recovers_the_true_scale(window, fits, case):
    """|recovered - true| scale error must sit well under 0.1 %."""
    _audio, ft, r_true = window
    _cand, res = fits[case]
    err = R.scale_summary(res.r_fit, r_true, ft, cfg=_cfg())
    assert abs(err["global_pct"]) < 0.02, f"{case}: global scale error {err['global_pct']} %"
    for i, v in enumerate(err["per_rotor_pct"]):
        assert abs(v) < 0.05, f"{case}: rotor {i} scale error {v} %"


@pytest.mark.parametrize("case", ["scale", "both"])
def test_reports_the_injected_scale(fits, case):
    """The report against the CORRUPTED carrier must read the injection back."""
    _cand, res = fits[case]
    # candidate = truth * 1.005, so fit - candidate is -0.5/1.005 = -0.4975 %.
    assert res.scale["global_pct"] == pytest.approx(-0.4975, abs=0.05)


def test_a_clean_carrier_is_left_alone(fits):
    """The staircase alone is zero-mean: the reported systematic must be ~0."""
    _cand, res = fits["staircase"]
    assert abs(res.scale["global_pct"]) < 0.05


def test_residual_rms_is_reported_and_does_not_beat_the_staircase(window, fits):
    """The de-staircasing is NOT what this procedure buys — measure, do not claim."""
    _audio, _ft, r_true = window
    cand, res = fits["staircase"]
    fit_rms = np.sqrt(np.mean((res.r_fit - r_true) ** 2, axis=1))
    init_rms = np.sqrt(np.mean((cand - r_true) ** 2, axis=1))
    assert np.all(fit_rms < 0.4), f"trajectory residual blew up: {fit_rms}"
    # Recorded as the finding, not asserted as an improvement.
    assert fit_rms.mean() > 0.3 * init_rms.mean()


def test_identity_survives(fits):
    """Rotor order and inter-rotor gaps: the twin-collapse failure mode."""
    for case, (_cand, res) in fits.items():
        assert res.identity["order_kept"], case
        for g in res.identity["gap_ratio"]:
            assert 0.9 < g < 1.1, f"{case}: gap moved by {g}"


# ---------------------------------------------------------------------------
# step 1: the carrier


def test_presmooth_removes_the_staircase(window):
    _audio, ft, r_true = window
    stair = staircase(r_true, ft)
    sm = R.presmooth(stair, ft, 5.0)
    assert np.sqrt(np.mean((sm - r_true) ** 2)) < np.sqrt(np.mean((stair - r_true) ** 2))
    assert np.allclose(R.presmooth(stair, ft, 0.0), stair)


def test_presmooth_is_the_carrier_the_fitter_starts_from(window, fits):
    _audio, ft, _r = window
    cand, res = fits["staircase"]
    assert np.allclose(res.r_init, R.presmooth(cand, ft, R.RefitConfig().smooth_cut_hz))
    assert np.allclose(res.r_raw, cand)


# ---------------------------------------------------------------------------
# step 2: the ladder


def test_the_ladder_is_never_flat_at_the_ceiling():
    """The failure mode the issue names: ``k_caps=(80, 80, 80)``."""
    cfg = R.RefitConfig()
    start = min(R.k_cap_for_error(cfg.e0_rev_s, cfg), cfg.k_start_max)
    assert start <= 20 < cfg.k_top


def test_k_cap_follows_the_wrap_rule():
    cfg = R.RefitConfig()
    for e in (2.0, 1.0, 0.5, 0.3):
        k = R.k_cap_for_error(e, cfg)
        assert 2 * np.pi * k * e / cfg.fs_env <= cfg.wrap_guard_rad + 1e-9
        assert k == cfg.k_top or 2 * np.pi * (k + 1) * e / cfg.fs_env > cfg.wrap_guard_rad


def test_advance_is_monotone_and_bounded():
    cfg = R.RefitConfig()
    k = 10
    for e in (0.5, 0.4, 0.001, 0.001):
        nxt = R.advance_k(k, e, cfg)
        assert nxt >= k
        assert nxt <= max(k, int(np.ceil(k * cfg.k_growth)))
        k = nxt
    assert k <= cfg.k_top
    # A worsening estimate holds the rung; it never steps back down.
    assert R.advance_k(60, 5.0, cfg) == 60


def test_the_fit_climbs(fits):
    for case, (_cand, res) in fits.items():
        ladder = res.k_ladder
        assert ladder == sorted(ladder), f"{case}: ladder went backwards {ladder}"
        assert ladder[0] < ladder[-1], f"{case}: ladder never climbed {ladder}"


# ---------------------------------------------------------------------------
# steps 3, 4, 6: the alternation, the LS peel, the twins


def test_the_alternation_peels_every_iteration(fits):
    """Each application re-peels at the current track (step 3), in LS mode (4)."""
    _cand, res = fits["scale"]
    assert res.iters, "no iteration was recorded"
    for rec in res.iters:
        assert rec["peel"]["mode"] == "ls"
        assert rec["peel"]["energy_ok"], rec
        assert rec["peel"]["e_resid_all_ratio"] < 1.0


def test_the_naive_arm_records_no_peel(window):
    _audio, ft, r_true = window
    audio, _, _ = window
    res = R.refit_window(audio, r_true * 1.005, ft, SR, cfg=_cfg(peel=False, max_iters=1))
    assert "peel" not in res.iters[0]


def test_twins_are_excluded_from_each_others_peel(window):
    """Step 6, verified on the existing peel rather than rebuilt.

    ``pair_audio[(lo, hi)]`` is the clip minus the NON-pair rotors only, so the
    two-tone observation that estimates a twin pair never has a sibling's
    reconstruction subtracted from it.
    """
    audio, ft, r_true = window
    clip = np.asarray(audio, dtype=np.float64)
    peel, pair, _diag = make_peels(clip, r_true, ft, SR, "ls", n_rotors=4, k_max=12, bw_hz=1.0)
    n = clip.shape[-1]
    removed_pair = clip - pair[(0, 1)]
    removed_0 = clip - peel[0]
    removed_1 = clip - peel[1]
    # What the pair peel removes is what BOTH single peels remove — i.e. rotors
    # 2 and 3 — and it removes strictly less than either single peel does.
    common = np.minimum(np.mean(removed_0**2), np.mean(removed_1**2))
    assert np.mean(removed_pair**2) < common
    assert removed_pair.shape[-1] == n


def test_pair_mode_is_wired_through(window):
    _audio, ft, r_true = window
    assert R.RefitConfig().pair_mode == "joint"
    res = R.refit_window(*_args(window, r_true), cfg=_cfg(max_iters=1))
    assert res.params["pair_mode"] == "joint"


def _args(window, cand):
    audio, ft, _ = window
    return audio, cand, ft, SR


# ---------------------------------------------------------------------------
# step 5: the stop


def test_a_loose_tolerance_converges_and_reports_it(window):
    res = R.refit_window(
        *_args(window, window[2] * 1.005), cfg=_cfg(tol_rev_s=10.0, max_iters=4, min_iters=1)
    )
    assert res.converged and res.stop_reason == "tolerance"
    assert len(res.iters) < 4


def test_the_cap_is_reported_as_a_non_convergence(fits):
    _cand, res = fits["scale"]
    assert res.stop_reason == "max_iters"
    assert not res.converged
    assert len(res.iters) == 4


def test_every_iteration_records_its_delta(fits):
    for case, (_cand, res) in fits.items():
        for rec in res.iters:
            assert {"iter", "k_cap", "delta_max", "delta_q", "delta_rms"} <= set(rec), case
            assert rec["delta_max"] >= rec["delta_q"] >= 0.0


def test_plateau_stops_the_alternation(window):
    res = R.refit_window(*_args(window, window[2] * 1.005), cfg=_cfg(plateau_rel=0.99, max_iters=4))
    assert res.stop_reason == "plateau"
    assert not res.converged


# ---------------------------------------------------------------------------
# the report and the seam


def test_scale_summary_is_exact_on_a_pure_scale(window):
    _audio, ft, r_true = window
    s = R.scale_summary(r_true * 1.005, r_true, ft)
    assert s["global_pct"] == pytest.approx(0.5, abs=1e-6)
    for v in s["per_rotor_pct"]:
        assert v == pytest.approx(0.5, abs=1e-6)


def test_order_and_gaps_reads_a_common_mode_shift_as_no_collapse():
    r = np.array([[86.0], [75.5], [85.7], [74.7]]) @ np.ones((1, 4))
    o0, g0 = R.order_and_gaps(r)
    o1, g1 = R.order_and_gaps(r * 0.99)
    assert o0 == o1
    assert np.allclose(np.asarray(g1) / np.asarray(g0), 0.99, atol=1e-6)


def test_report_is_json_serializable(fits):
    import json

    _cand, res = fits["scale"]
    json.loads(json.dumps(res.as_dict()))


def test_residual_block_flags_its_own_conditioning(fits):
    """A cruise-like carrier makes the 6a scale/offset split unidentifiable."""
    _cand, res = fits["scale"]
    cond = res.residual["pooled"]["design_cond"]
    assert cond is not None and cond > 10.0


def test_refit_stage_reads_rps_meas_and_logs(window):
    audio, ft, r_true = window
    frame = tracking_frame(
        audio, SR, rps=r_true, frame_times=ft, rps_meas=r_true * 1.005, dtype=np.float64
    )
    out = top.refit_stage(cfg=_cfg(max_iters=1))(frame)
    r_new, _ = get_rps(out)
    entry = out["meta"]["tracking"][-1]
    assert entry["stage"] == "telemetry_refit"
    assert entry["k_ladder"]
    # The carrier was rps_meas (the +0.5 % copy), not the frame's own rps.
    assert np.allclose(get_rps(out, "rps_meas")[0], r_true * 1.005)
    assert not np.allclose(r_new, r_true * 1.005)
