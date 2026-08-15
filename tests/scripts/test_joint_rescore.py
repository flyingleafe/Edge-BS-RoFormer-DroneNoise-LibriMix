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
