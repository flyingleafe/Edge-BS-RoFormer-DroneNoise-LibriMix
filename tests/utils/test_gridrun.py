"""Tests for the restartable parallel unit-JSON harness (``utils.gridrun``).

Covers: (1) resume skips completed units (asserted through per-invocation
marker files); (2) a failing unit writes an ``.err`` file without killing the
pool, and is retried on the next (resumed) run; (3) the summary file +
``summarize`` callback; (4) the argparse mixin. All CPU-fast, no data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from utils.gridrun import (
    Unit,
    add_gridrun_args,
    gridrun_from_args,
    run_grid,
    unit_path,
)


def _marker_worker(unit: Unit) -> dict[str, Any]:
    """Record every invocation in a per-unit marker file, then succeed."""
    marker = Path(str(unit.params["marker_dir"])) / f"{unit.uid}.marker"
    marker.parent.mkdir(parents=True, exist_ok=True)
    n = int(marker.read_text()) + 1 if marker.exists() else 1
    marker.write_text(str(n))
    return {"uid": unit.uid, "value": unit.params["value"]}


def _flaky_worker(unit: Unit) -> dict[str, Any]:
    if unit.params.get("fail"):
        raise RuntimeError(f"boom in {unit.uid}")
    return {"uid": unit.uid}


def _units(tmp_path: Path, n: int) -> list[Unit]:
    return [
        Unit(uid=f"u{i:02d}", params={"marker_dir": str(tmp_path / "markers"), "value": i})
        for i in range(n)
    ]


def test_resume_skips_completed_units(tmp_path: Path) -> None:
    out = tmp_path / "grid"
    units = _units(tmp_path, 3)

    first = run_grid(units, _marker_worker, out, jobs=1, verbose=False)
    assert first.n_ok == 3 and first.n_skipped == 0 and first.n_failed == 0
    for u in units:
        assert unit_path(out, u.uid).exists()

    second = run_grid(units, _marker_worker, out, jobs=1, verbose=False)
    assert second.n_ok == 0 and second.n_skipped == 3
    # every unit ran exactly ONCE across both runs
    for u in units:
        marker = tmp_path / "markers" / f"{u.uid}.marker"
        assert marker.read_text() == "1"

    # resume=False re-runs everything
    third = run_grid(units, _marker_worker, out, jobs=1, resume=False, verbose=False)
    assert third.n_ok == 3 and third.n_skipped == 0
    for u in units:
        assert (tmp_path / "markers" / f"{u.uid}.marker").read_text() == "2"


def test_failed_unit_does_not_kill_the_pool(tmp_path: Path) -> None:
    out = tmp_path / "grid"
    units = [Unit(uid=f"u{i}", params={"fail": i == 1}) for i in range(4)]

    result = run_grid(units, _flaky_worker, out, jobs=2, verbose=False)
    assert result.n_ok == 3 and result.n_failed == 1
    assert result.exit_code == 1
    err = unit_path(out, "u1").with_suffix(".err")
    assert err.exists() and "boom in u1" in err.read_text()
    assert not unit_path(out, "u1").exists()

    # the failed unit is retried on a resumed run (its JSON is missing);
    # once it succeeds, the stale .err is cleared
    fixed = [Unit(uid=f"u{i}", params={}) for i in range(4)]
    retry = run_grid(fixed, _flaky_worker, out, jobs=2, verbose=False)
    assert retry.n_ok == 1 and retry.n_skipped == 3 and retry.n_failed == 0
    assert unit_path(out, "u1").exists() and not err.exists()


def test_summary_written_with_summarize_callback(tmp_path: Path) -> None:
    out = tmp_path / "grid"
    units = _units(tmp_path, 3)

    def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
        return {"n": len(rows), "values": sorted(r["value"] for r in rows)}

    result = run_grid(units, _marker_worker, out, jobs=1, summarize=summarize, verbose=False)
    assert result.summary == {"n": 3, "values": [0, 1, 2]}
    assert json.loads((out / "summary.json").read_text()) == result.summary

    # summarize sees ALL persisted units, including previously-completed ones
    more = units + [Unit(uid="u99", params={"marker_dir": str(tmp_path / "markers"), "value": 99})]
    again = run_grid(more, _marker_worker, out, jobs=1, summarize=summarize, verbose=False)
    assert again.summary["n"] == 4 and 99 in again.summary["values"]


def test_duplicate_uids_rejected(tmp_path: Path) -> None:
    units = [Unit(uid="same", params={}), Unit(uid="same", params={})]
    try:
        run_grid(units, _flaky_worker, tmp_path / "grid", jobs=1, verbose=False)
    except ValueError as exc:
        assert "duplicate" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("duplicate uids must raise")


def test_argparse_mixin(tmp_path: Path) -> None:
    parser = argparse.ArgumentParser()
    add_gridrun_args(parser, jobs=2)
    args = parser.parse_args(["--jobs", "1", "--no-resume"])
    assert args.jobs == 1 and args.no_resume

    out = tmp_path / "grid"
    units = _units(tmp_path, 2)
    run_grid(units, _marker_worker, out, jobs=1, verbose=False)
    result = gridrun_from_args(args, units, _marker_worker, out, verbose=False)
    # --no-resume re-ran the already-completed units
    assert result.n_ok == 2 and result.n_skipped == 0
    for u in units:
        assert (tmp_path / "markers" / f"{u.uid}.marker").read_text() == "2"
