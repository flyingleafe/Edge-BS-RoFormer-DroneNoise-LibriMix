"""Unit tests for the importable core of ``scripts/joint_rescore.py``.

The solve itself is the tracking package's and is tested there; what is tested
here is what this driver ADDS — reading a step-5 arm's trajectory back, and the
window x hypothesis table it builds out of the unit rows.
"""

from __future__ import annotations

import json
from typing import Any

import joint_rescore as J
import numpy as np
import pytest


def _row(window: str, hyp: str, total: float, cells: int = 100, rms: float = 0.0) -> dict[str, Any]:
    obj = {
        "total": total,
        "data": total / 2.0,
        "rent": total / 2.0,
        "phase_priors": 0.0,
        "envelope_prior": 0.0,
        "n_cells": cells,
    }
    return {
        "window": window,
        "hypothesis": hyp,
        "objective": obj,
        "per_cell": {t: obj[t] / cells for t in J.TERMS},
        "k_hi": 40,
        "mean_rev_s": 80.0,
        "rms_vs_telemetry": rms,
        "residual_fraction": 0.5,
        "wall_s": 1.0,
    }


def test_load_hypothesis_reads_the_arm_and_the_telemetry(tmp_path) -> None:
    r_meas = np.full((4, 6), 80.0)
    raw = tmp_path / "raw"
    raw.mkdir()
    arm = np.arange(24, dtype=float).reshape(4, 6)
    J.arm_path(tmp_path, "REC__w01", "ours_full").write_text(json.dumps({"r_out": arm.tolist()}))

    assert np.array_equal(J.load_hypothesis(J.TELEMETRY, "REC__w01", r_meas, tmp_path), r_meas)
    assert np.array_equal(J.load_hypothesis("ours_full", "REC__w01", r_meas, tmp_path), arm)

    with pytest.raises(FileNotFoundError, match="no step-5 result"):
        J.load_hypothesis("multistart", "REC__w01", r_meas, tmp_path)
    # A trajectory on a different frame grid is not a hypothesis about THIS
    # window, and silently interpolating it would hide a mismatched campaign.
    J.arm_path(tmp_path, "REC__w02", "ours_full").write_text(
        json.dumps({"r_out": arm[:, :3].tolist()})
    )
    with pytest.raises(ValueError, match="against the window's grid"):
        J.load_hypothesis("ours_full", "REC__w02", r_meas, tmp_path)


#: The sidecar's own frame grid: 0.1 s frames over 40 s of a recording, in the
#: PUBLISHED recording's time reference (which is why it does not start at 0).
REFINED_FT = np.arange(0.0, 40.0, 0.1) + 5.48


def _write_sidecar(
    dst, rid: str = "REC", *, perm: tuple[int, ...] = (0, 1, 2, 3), bias: float = 0.6
) -> np.ndarray:
    """A synthetic refined-label sidecar whose trajectories are LINEAR in time.

    Linear on purpose: a linear interpolation onto any sub-grid is then exact,
    so the read-back can be asserted against the closed form instead of against
    a tolerance. ``r_refined`` is the telemetry plus ``bias``, which stands in
    for the measured scale correction, and ``perm`` permutes the rotor rows so
    the order guard can be exercised.
    """
    base = np.stack([80.0 + 5.0 * i + 0.2 * REFINED_FT for i in range(4)])
    np.savez(
        dst / f"{rid}.npz",
        ft=REFINED_FT,
        r_telemetry=base[list(perm)],
        r_refined=(base + bias)[list(perm)],
    )
    return base


def test_the_refined_hypothesis_is_read_at_the_window_s_own_frames(tmp_path) -> None:
    """The sidecar is a whole RECORDING; the window is a slice of one.

    They are joined by the window's recording-absolute ``start_s``, so the
    trajectory the rescore scores must be the sidecar sampled at
    ``start_s + ft`` — exactly, because the fixture's trajectories are linear.
    """
    start_s, ft = 20.0, np.arange(6) * 0.032
    base = _write_sidecar(tmp_path)
    want_tel = np.stack([80.0 + 5.0 * i + 0.2 * (start_s + ft) for i in range(4)])
    np.testing.assert_allclose(
        np.stack([np.interp(start_s + ft, REFINED_FT, row) for row in base]), want_tel
    )

    got = J.load_hypothesis(
        J.REFINED, "REC__w01", want_tel, tmp_path, start_s=start_s, ft=ft, label_dir=tmp_path
    )
    assert got.shape == (4, 6)
    np.testing.assert_allclose(got, want_tel + 0.6, atol=1e-9)


def test_the_refined_hypothesis_names_the_missing_sidecar(tmp_path) -> None:
    # It exists only where a recording has been refined and its .npz committed,
    # so the failure has to say which file is not there.
    with pytest.raises(FileNotFoundError, match="no refined-label sidecar"):
        J.load_hypothesis(
            J.REFINED,
            "OTHER__w00",
            np.full((4, 6), 80.0),
            tmp_path,
            start_s=20.0,
            ft=np.arange(6) * 0.032,
            label_dir=tmp_path,
        )


def test_the_refined_hypothesis_refuses_a_window_with_no_start(tmp_path) -> None:
    # An old prep cache reports start_s as nan, and a nan offset would read the
    # sidecar nowhere at all rather than fail.
    _write_sidecar(tmp_path)
    with pytest.raises(ValueError, match="no start_s"):
        J.load_hypothesis(
            J.REFINED,
            "REC__w01",
            np.full((4, 6), 80.0),
            tmp_path,
            start_s=float("nan"),
            ft=np.arange(6) * 0.032,
            label_dir=tmp_path,
        )


def test_a_permuted_sidecar_is_refused_instead_of_scored(tmp_path) -> None:
    """The guard that stops the verdict from being inverted silently.

    A sidecar whose rotor rows are in another order hands every rotor another
    rotor's trajectory. Nothing downstream can tell that from a large rate
    error, so it would be SCORED as one. The sidecar carries the telemetry it
    was initialized from, so the check is one more interpolation: sliced the
    same way, it must land on the window's own r_meas.
    """
    start_s, ft = 20.0, np.arange(6) * 0.032
    want_tel = np.stack([80.0 + 5.0 * i + 0.2 * (start_s + ft) for i in range(4)])
    _write_sidecar(tmp_path, "GOOD")
    _write_sidecar(tmp_path, "BAD", perm=(1, 0, 3, 2))

    J.load_hypothesis(
        J.REFINED, "GOOD__w01", want_tel, tmp_path, start_s=start_s, ft=ft, label_dir=tmp_path
    )
    with pytest.raises(ValueError, match="rotor order or the time reference"):
        J.load_hypothesis(
            J.REFINED, "BAD__w01", want_tel, tmp_path, start_s=start_s, ft=ft, label_dir=tmp_path
        )
    # And a window read at the WRONG place in the recording fails the same way,
    # because the same interpolation is what places it.
    with pytest.raises(ValueError, match="rotor order or the time reference"):
        J.load_hypothesis(
            J.REFINED, "GOOD__w01", want_tel, tmp_path, start_s=39.0, ft=ft, label_dir=tmp_path
        )


def test_refined_is_opt_in_and_never_packed(tmp_path) -> None:
    # The default hypotheses are unchanged: 'refined' exists only where a
    # sidecar is committed, so it cannot be a default. And it travels with the
    # checkout already, so a pack must not try to read it out of the arms.
    assert J.REFINED not in J.DEFAULT_HYPOTHESES
    assert J.refined_path("REC__w01").name == "REC.npz"
    assert J.refined_path("REC__w01").parent == J.REFINED_LABEL_DIR
    (tmp_path / "raw").mkdir()
    arm = np.arange(24, dtype=float).reshape(4, 6)
    J.arm_path(tmp_path, "REC__w01", "ours_full").write_text(json.dumps({"r_out": arm.tolist()}))
    pack = tmp_path / "pack.json"
    J.pack_hypotheses(tmp_path, ["REC__w01"], [J.TELEMETRY, J.REFINED, "ours_full"], pack)
    assert list(json.loads(pack.read_text())["windows"]["REC__w01"]) == ["ours_full"]


def test_pack_round_trips_the_arms(tmp_path) -> None:
    # results/ does not travel to a cluster, so the trajectories are packed into
    # one file — and a pack must read back exactly like the directory did.
    (tmp_path / "raw").mkdir()
    arm = np.arange(24, dtype=float).reshape(4, 6)
    for w in ("REC__w01", "REC__w02"):
        J.arm_path(tmp_path, w, "ours_full").write_text(json.dumps({"r_out": arm.tolist()}))
    pack = tmp_path / "pack.json"
    J.pack_hypotheses(tmp_path, ["REC__w01", "REC__w02"], [J.TELEMETRY, "ours_full"], pack)

    assert sorted(json.loads(pack.read_text())["windows"]) == ["REC__w01", "REC__w02"]
    # The telemetry is never packed: it comes from the prep window itself.
    assert list(json.loads(pack.read_text())["windows"]["REC__w01"]) == ["ours_full"]
    assert np.array_equal(J.read_arm(pack, "REC__w01", "ours_full"), arm)
    with pytest.raises(KeyError, match="in the pack"):
        J.read_arm(pack, "REC__w01", "multistart")


def test_summarize_ranks_by_the_per_cell_total() -> None:
    rows = [
        _row("w1", "telemetry", -100.0),
        _row("w1", "ours_full", -120.0, rms=0.4),
        _row("w1", "multistart", -110.0, rms=0.9),
    ]
    got = J.summarize(rows)["table"]["w1"]
    assert got["_ranking"] == ["ours_full", "multistart", "telemetry"]
    assert got["_best"] == "ours_full"
    assert got["_cells_agree"] is True
    # Negative = the audio prefers the hypothesis to the tachometer.
    assert got["_delta_vs_telemetry"]["ours_full"] == pytest.approx(-0.2)
    assert got["ours_full"]["total_per_cell"] == pytest.approx(-1.2)


def _marginal_row(
    window: str, hyp: str, total: float, marg: float, cells: int = 100
) -> dict[str, Any]:
    row = _row(window, hyp, total, cells)
    row["objective"]["total_marginal"] = marg
    row["objective"]["marginal_correction"] = marg - total
    row["per_cell"]["total_marginal"] = marg / cells
    row["per_cell"]["marginal_correction"] = (marg - total) / cells
    return row


def test_marginal_ranks_by_the_marginal_total_and_keeps_the_profiled_one() -> None:
    # The case the term exists for: the arm wins the PROFILED column by
    # absorbing more, and loses the MARGINAL one once that freedom is charged.
    rows = [
        _marginal_row("w1", "telemetry", -100.0, -100.0),
        _marginal_row("w1", "ours_full", -120.0, -90.0),
    ]
    got = J.summarize(rows)
    assert got["marginal"] is True
    win = got["table"]["w1"]
    assert win["_ranked_by"] == "total_marginal_per_cell"
    assert win["_ranking"] == ["telemetry", "ours_full"]
    assert win["_ranking_profiled"] == ["ours_full", "telemetry"]
    assert win["_best"] == "telemetry"
    # Positive = the marginal objective prefers the tachometer.
    assert win["_delta_vs_telemetry"]["ours_full"] == pytest.approx(0.1)
    J.print_table(got)


def test_a_missing_marginal_column_falls_back_to_the_profiled_ranking() -> None:
    # A run without --marginal, or a half-migrated results directory, must rank
    # by what every row actually carries rather than by a key some rows lack.
    rows = [_marginal_row("w1", "telemetry", -100.0, -100.0), _row("w1", "ours_full", -120.0)]
    got = J.summarize(rows)
    assert got["marginal"] is False
    assert got["table"]["w1"]["_ranked_by"] == "total_per_cell"
    assert got["table"]["w1"]["_ranking"] == ["ours_full", "telemetry"]


def test_the_marginal_flag_reaches_the_units() -> None:
    assert all(u.params["marginal"] is False for u in J.build_units(_args()))
    assert all(u.params["marginal"] is True for u in J.build_units(_args(marginal=True)))


def _h_row(window: str, hyp: str, total: float, total_h: float, cells: int = 100) -> dict[str, Any]:
    row = _row(window, hyp, total, cells)
    row["objective"]["total_h"] = total_h
    row["objective"]["data_h"] = total_h / 2.0
    row["objective"]["h_cells"] = cells // 4
    row["per_cell"]["total_h"] = total_h / cells
    row["per_cell"]["data_h"] = total_h / 2.0 / cells
    return row


def test_h_aware_ranks_by_the_h_total_and_keeps_the_profiled_one() -> None:
    # The case the term exists for: the fan wins the PROFILED column because
    # every hypothesis is charged alike for the line flanks, and loses the
    # H-aware one once the telemetry is allowed to EXPLAIN the humps.
    rows = [
        _h_row("w1", "telemetry", -100.0, -140.0),
        _h_row("w1", "ours_full", -120.0, -130.0),
    ]
    got = J.summarize(rows)
    assert got["h_aware"] is True and got["marginal"] is False
    win = got["table"]["w1"]
    assert win["_ranked_by"] == "total_h_per_cell"
    assert win["_ranking"] == ["telemetry", "ours_full"]
    assert win["_ranking_profiled"] == ["ours_full", "telemetry"]
    assert win["_best"] == "telemetry"
    assert win["telemetry"]["h_cells"] == 25
    J.print_table(got)


def test_a_missing_h_column_falls_back_to_the_profiled_ranking() -> None:
    rows = [_h_row("w1", "telemetry", -100.0, -140.0), _row("w1", "ours_full", -120.0)]
    got = J.summarize(rows)
    assert got["h_aware"] is False
    assert got["table"]["w1"]["_ranked_by"] == "total_per_cell"


def test_the_h_aware_flag_reaches_the_units() -> None:
    assert all(u.params["h_aware"] is False for u in J.build_units(_args()))
    assert all(u.params["h_aware"] is True for u in J.build_units(_args(h_aware=True)))


def test_summarize_flags_disagreeing_cell_counts() -> None:
    # Two hypotheses scored on different cell sets cannot be compared at all —
    # the table must say so rather than rank them.
    rows = [_row("w1", "telemetry", -100.0, cells=100), _row("w1", "ours_full", -120.0, cells=90)]
    assert J.summarize(rows)["table"]["w1"]["_cells_agree"] is False


def test_smoke_defaults_are_one_window_and_a_small_solve() -> None:
    ap_rows = J.build_units(
        _args(windows=J.DEFAULT_WINDOWS[0], hypotheses=",".join(J.DEFAULT_HYPOTHESES))
    )
    assert [u.uid for u in ap_rows] == [
        f"{J.DEFAULT_WINDOWS[0]}__{h}" for h in J.DEFAULT_HYPOTHESES
    ]
    assert all(u.params["k_trust"] == "3,12,80" and u.params["iters"] == 3 for u in ap_rows)


def _args(**over: Any) -> Any:
    import argparse

    base = {
        "windows": ",".join(J.DEFAULT_WINDOWS),
        "hypotheses": ",".join(J.DEFAULT_HYPOTHESES),
        "k_max": 40,
        "f_max": 6000.0,
        "mics": 8,
        "seconds": 0.0,
        "iters": J.DEFAULT_ITERS,
        "k_trust": ",".join(str(v) for v in J.DEFAULT_LADDER),
        "arms_dir": "results/fvk_arms",
        "prep_dir": "",
        "mem_budget_gb": 0.0,
    }
    return argparse.Namespace(**{**base, **over})
