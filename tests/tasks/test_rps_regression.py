"""Regression tests for RPS prediction — golden-artifact verification.

Uses the *new* task-module evaluation pipeline (``src/tasks/rps_prediction.py``)
and asserts that inference matches the committed 10-sample golden within
tight tolerances."""

from __future__ import annotations

import itertools
import json

# Ensure src/ is on the path (pytest 9.x isolated-import behaviour).
from typing import Any

import numpy as np
import pytest

from tasks.rps_prediction import (
    EvalResult,
    evaluate,
    load_input_set,
    load_predictor,
)
from utils.paths import get_datasets_path, get_results_path

# ── Paths (relative to DATA_ROOT) ─────────────────────────────────────────

_GOLDEN_DIR = get_results_path("rps_predictor_comparison")
_GOLDEN_PER_SAMPLE = _GOLDEN_DIR / "val_inference/per_sample_metrics_10.json"
_GOLDEN_SIMPLE_CONV_CKPT = get_results_path("rps_exp_simple_conv/best_simple_conv.pt")
_DREGON_VALID = get_datasets_path("DREGON-LM/valid")

# Golden artifacts (checkpoint + dataset + per-sample json) are gitignored
# and machine-local; skip the whole module when they are absent instead of
# failing on fresh clones.
pytestmark = pytest.mark.skipif(
    not (
        _GOLDEN_SIMPLE_CONV_CKPT.is_file()
        and _DREGON_VALID.is_dir()
        and _GOLDEN_PER_SAMPLE.is_file()
    ),
    reason="golden artifacts (results/ + datasets/) not present on this machine",
)

# ── Test-sized subset ────────────────────────────────────────────────────

_N_SAMPLES = 10  # small for fast CPU regression smoke

_SPEC = f"simple_conv@{_GOLDEN_SIMPLE_CONV_CKPT}"

# ── Helpers ──────────────────────────────────────────────────────────────


def _run_inference() -> EvalResult:
    """Run task-module evaluation on the first N_SAMPLES of DREGON-LM valid."""
    predictor = load_predictor(_SPEC)
    samples = list(itertools.islice(load_input_set(str(_DREGON_VALID)), _N_SAMPLES))
    return evaluate(
        predictor,
        samples,
        model_spec="simple_conv",
        input_set_label="dregon-lm:valid:smoke",
        verbose=False,
    )


def load_golden_per_sample() -> list[dict[str, Any]]:
    with open(_GOLDEN_PER_SAMPLE) as f:
        return json.load(f)


# ── Artifact existence ──────────────────────────────────────────────────


def test_artifacts_exist():
    """Sanity: checkpoint and dataset are present on disk."""
    assert _GOLDEN_SIMPLE_CONV_CKPT.is_file(), f"Missing {_GOLDEN_SIMPLE_CONV_CKPT}"
    assert _DREGON_VALID.is_dir(), f"Missing {_DREGON_VALID}"


# ── Evaluation smoke test ────────────────────────────────────────────────


def test_eval_runs_and_produces_well_formed_output():
    """The evaluation pipeline runs end-to-end on a 10-sample subset
    and returns correctly structured per-sample and aggregate results."""
    result = _run_inference()

    # ── Aggregate checks ───────────────────────────────────────────────
    agg = result.aggregate
    assert agg["n_samples"] == _N_SAMPLES, f"expected {_N_SAMPLES} samples, got {agg['n_samples']}"
    assert isinstance(agg["mse"], float)
    assert isinstance(agg["mae_frame"], float)
    assert isinstance(agg["mae_clip"], float)
    assert isinstance(agg["rmse"], float)
    assert isinstance(agg["r2_mean"], float)
    assert agg["n_r2_valid"] == _N_SAMPLES, f"all {_N_SAMPLES} samples should have valid R²"

    # ── Per-sample checks ──────────────────────────────────────────────
    rows = result.per_sample
    assert len(rows) == _N_SAMPLES

    expected_keys = {"sample", "mse", "mae_frame", "mae_clip", "ss_tot", "r2", "input_snr"}
    for i, row in enumerate(rows):
        missing = expected_keys - set(row.keys())
        assert not missing, f"row {i} ({row.get('sample', '?')}) missing {missing}"
        assert isinstance(row["sample"], str), f"row {i}: sample not str"
        assert isinstance(row["mse"], float), f"row {i}: mse not float"
        assert isinstance(row["mae_frame"], float), f"row {i}: mae_frame not float"
        assert isinstance(row["mae_clip"], float), f"row {i}: mae_clip not float"
        assert isinstance(row["ss_tot"], float), f"row {i}: ss_tot not float"
        assert row["r2"] is not None, f"row {i}: r2 is None"
        assert isinstance(row["r2"], float), f"row {i}: r2 not float"
        assert isinstance(row["input_snr"], float), f"row {i}: input_snr not float"

        # Sanity: MSE ≥ 0, ss_tot > 0, R² ≤ 1.
        assert row["mse"] >= 0, f"row {i}: negative MSE {row['mse']}"
        assert row["ss_tot"] > 0, f"row {i}: non-positive ss_tot {row['ss_tot']}"
        assert row["r2"] <= 1.0 + 1e-6, f"row {i}: R² > 1 ({row['r2']})"

    # ── Per-sample numeric closeness ──────────────────────────────────
    golden_rows = load_golden_per_sample()
    golden_by_sample = {r["sample"]: r for r in golden_rows}
    assert len(golden_rows) == _N_SAMPLES, (
        f"golden has {len(golden_rows)} rows, expected {_N_SAMPLES}"
    )
    for row in rows:
        g = golden_by_sample[row["sample"]]
        for key in ("mse", "mae_frame", "mae_clip", "ss_tot", "r2"):
            assert np.isclose(row[key], g[key], rtol=1e-4, atol=1e-4), (
                f"{row['sample']}.{key}: got {row[key]:.6f}, golden {g[key]:.6f}"
            )


# ── Aggregate smoke test ─────────────────────────────────────────────────


def test_aggregate_structure():
    """Aggregate metrics are computed and have sensible ranges."""
    result = _run_inference()
    agg = result.aggregate

    # Structure.
    for k in (
        "n_samples",
        "n_r2_valid",
        "mse",
        "rmse",
        "mae_frame",
        "mae_clip",
        "r2_mean",
        "r2_median",
        "r2_std",
        "elapsed_s",
    ):
        assert k in agg, f"aggregate missing key {k!r}"

    # Sanity: monotonic relationships.
    assert agg["rmse"] == pytest.approx(np.sqrt(agg["mse"]), rel=1e-6), (
        f"RMSE {agg['rmse']:.6f} ≠ sqrt(MSE) = {np.sqrt(agg['mse']):.6f}"
    )
    assert agg["n_r2_valid"] <= agg["n_samples"]
    assert agg["elapsed_s"] > 0

    # ── Numeric closeness to golden aggregate ────────────────────────
    golden_rows = load_golden_per_sample()
    g_mse = float(np.mean([r["mse"] for r in golden_rows]))
    g_mae_frame = float(np.mean([r["mae_frame"] for r in golden_rows]))
    g_mae_clip = float(np.mean([r["mae_clip"] for r in golden_rows]))
    g_r2 = float(np.mean([r["r2"] for r in golden_rows]))
    assert np.isclose(agg["mse"], g_mse, rtol=1e-4, atol=1e-4), (
        f"agg MSE: got {agg['mse']:.6f}, golden {g_mse:.6f}"
    )
    assert np.isclose(agg["mae_frame"], g_mae_frame, rtol=1e-4, atol=1e-4), (
        f"agg MAE frame: got {agg['mae_frame']:.6f}, golden {g_mae_frame:.6f}"
    )
    assert np.isclose(agg["mae_clip"], g_mae_clip, rtol=1e-4, atol=1e-4), (
        f"agg MAE clip: got {agg['mae_clip']:.6f}, golden {g_mae_clip:.6f}"
    )
    assert np.isclose(agg["r2_mean"], g_r2, rtol=1e-4, atol=1e-4), (
        f"agg R²: got {agg['r2_mean']:.6f}, golden {g_r2:.6f}"
    )
