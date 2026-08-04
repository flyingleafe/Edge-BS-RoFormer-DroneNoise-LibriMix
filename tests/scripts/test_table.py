"""Unit tests for the importable core of ``scripts/table.py``."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import table


def _perclip_csv(path: Path) -> None:
    path.write_text(
        "method,valid,clip_id,category,input_snr,si_sdr,pesq\n"
        "noisy,V,c1,drone,-15.0,-10.0,1.1\n"
        "noisy,V,c2,drone,-15.0,-12.0,1.3\n"
        "noisy,V,c3,drone,-5.0,-4.0,1.5\n"
        "model,V,c1,drone,-15.0,-2.0,1.9\n"
        "model,V,c2,drone,-15.0,0.0,2.1\n"
        "model,V,c3,drone,-5.0,4.0,2.5\n"
    )


def test_load_inputs_glob_and_concat(tmp_path: Path) -> None:
    _perclip_csv(tmp_path / "a.csv")
    _perclip_csv(tmp_path / "b.csv")
    df = table.load_inputs([str(tmp_path / "*.csv")])
    assert len(df) == 12


def test_pivot_rows_cols_and_baseline_deltas(tmp_path: Path) -> None:
    _perclip_csv(tmp_path / "a.csv")
    df = table.load_inputs([str(tmp_path / "a.csv")])
    grouped = table.aggregate(df, ["method"], "input_snr", ["si_sdr"], "mean")
    piv = table.pivot(grouped, ["method"], "input_snr", "si_sdr", baseline="noisy")
    # cell means
    assert piv.loc["noisy", -15.0] == pytest.approx(-11.0)
    assert piv.loc["model", -15.0] == pytest.approx(-1.0)
    # delta columns vs the noisy anchor, matched per column
    assert piv.loc["model", "d_-15.0"] == pytest.approx(10.0)
    assert piv.loc["model", "d_-5.0"] == pytest.approx(8.0)
    assert piv.loc["noisy", "d_-15.0"] == pytest.approx(0.0)


def test_pivot_without_cols_uses_metric_column(tmp_path: Path) -> None:
    _perclip_csv(tmp_path / "a.csv")
    df = table.load_inputs([str(tmp_path / "a.csv")])
    grouped = table.aggregate(df, ["method"], None, ["pesq"], "mean")
    piv = table.pivot(grouped, ["method"], None, "pesq")
    assert piv.loc["model", "pesq"] == pytest.approx((1.9 + 2.1 + 2.5) / 3)


def test_aggregate_count_mode(tmp_path: Path) -> None:
    _perclip_csv(tmp_path / "a.csv")
    df = table.load_inputs([str(tmp_path / "a.csv")])
    counts = table.aggregate(df, ["category"], "input_snr", [], "count")
    piv = table.pivot(counts, ["category"], "input_snr", "n")
    assert piv.loc["drone", -15.0] == 4
    assert piv.loc["drone", -5.0] == 2


def test_drop_silent_drops_only_globally_floored_clips() -> None:
    df = pd.DataFrame(
        {
            "method": ["noisy", "model", "noisy", "model"],
            "valid": ["V"] * 4,
            "clip_id": ["dead", "dead", "ok", "ok"],
            "si_sdr": [-80.0, -75.0, -80.0, 5.0],
        }
    )
    out = table.drop_silent(df)
    # "dead" is < -70 for EVERY method -> dropped; "ok" recovers under one
    # method -> kept (including its floored noisy row).
    assert set(out["clip_id"]) == {"ok"}
    assert len(out) == 2


def test_order_rows_puts_anchors_first(tmp_path: Path) -> None:
    _perclip_csv(tmp_path / "a.csv")
    df = table.load_inputs([str(tmp_path / "a.csv")])
    grouped = table.aggregate(df, ["method"], None, ["si_sdr"], "mean")
    piv = table.pivot(grouped, ["method"], None, "si_sdr")
    ordered = table.order_rows(piv, ["noisy"])
    assert list(ordered.index) == ["noisy", "model"]


def test_json_records_root_unnest_and_flatten() -> None:
    payload = {
        "groups": {
            "dregon": {
                "n_units": 3,
                "arms": {
                    "fixB6": {"alpha_signal": {"median": 2.0, "iqr": 0.5}},
                    "kscale0.25": {"alpha_signal": {"median": 1.5, "iqr": 0.4}},
                },
            }
        }
    }
    recs = table.json_records(
        payload, json_root="groups", index_name="group", unnest=[("arms", "arm")]
    )
    assert len(recs) == 2
    fix = next(r for r in recs if r["arm"] == "fixB6")
    assert fix["group"] == "dregon"
    assert fix["n_units"] == 3
    assert fix["alpha_signal.median"] == pytest.approx(2.0)


def test_to_markdown_shapes_a_pipe_table() -> None:
    piv = pd.DataFrame({"a": [1.0, float("nan")]}, index=pd.Index(["x", "y"], name="method"))
    md = table.to_markdown(piv, "demo")
    lines = md.strip().splitlines()
    assert lines[0] == "### demo"
    assert "| method | a |" in lines
    assert any("—" in ln for ln in lines)  # NaN renders as an em dash
