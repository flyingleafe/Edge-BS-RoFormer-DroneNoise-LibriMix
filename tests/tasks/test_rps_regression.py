"""Regression tests for RPS prediction — golden-artifact verification.

These tests run inference with the *existing* eval code (not the new task
module) and assert the output matches the committed golden files within
tight tolerances.  They are the acceptance gate for the refactor — the new
task module must pass them before any legacy code is removed.

Markers
-------
``slow`` — full DREGON-LM valid inference (600 samples, ~60 s on CPU).
These are skipped by default; run with ``pytest -m slow``.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from train_rps_predictor import DREGONRPSDataset, get_model

# ── Paths (relative to repo root) ────────────────────────────────────────

_REPO = Path(__file__).resolve().parents[2]
_GOLDEN_DIR = _REPO / "results/rps_predictor_comparison"
_GOLDEN_PER_SAMPLE = _GOLDEN_DIR / "val_inference/per_sample_metrics.json"
_GOLDEN_SIMPLE_CONV_CKPT = _GOLDEN_DIR / "best_simple_conv.pt"
_DREGON_VALID = _REPO / "datasets/DREGON-LM/valid"

# ── Constants from train_rps_predictor.py ────────────────────────────────

N_FFT = 2048
HOP = 512
MODEL_NAME = "simple_conv"
DEVICE = "cpu"


# ── Helpers ──────────────────────────────────────────────────────────────

def _run_inference(ckpt_path: Path, data_dir: Path) -> list[dict]:
    """Run eval_rps_val.py-style inference and return per-sample metrics."""
    ds = DREGONRPSDataset(str(data_dir), n_fft=N_FFT, hop_length=HOP)
    model = get_model(MODEL_NAME, n_fft=N_FFT, hop_length=HOP)
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt)
    model.eval()

    per_sample: list[dict] = []
    with torch.no_grad():
        for i, sample_dir in enumerate(ds.samples):
            audio, rps_target = ds[i]
            audio = audio.unsqueeze(0)  # (1, samples)
            rps_target_t = rps_target  # (4, T)

            rps_pred = model(audio).squeeze(0)  # (4, T_pred)

            T = min(rps_pred.shape[-1], rps_target_t.shape[-1])
            rps_pred = rps_pred[..., :T]
            rps_target_t = rps_target_t[..., :T]

            mse = F.mse_loss(rps_pred, rps_target_t).item()
            mae_frame = (rps_pred - rps_target_t).abs().mean().item()
            mae_clip = ((rps_pred - rps_target_t).mean(dim=-1).abs()).mean().item()
            ss_res = ((rps_pred - rps_target_t) ** 2).sum().item()
            ss_tot = ((rps_target_t - rps_target_t.mean()) ** 2).sum().item()
            r2 = (1.0 - ss_res / ss_tot) if ss_tot > 1e-6 else None

            sample_name = os.path.basename(sample_dir)
            per_sample.append({
                "sample": sample_name,
                "mse": mse,
                "mae_frame": mae_frame,
                "mae_clip": mae_clip,
                "ss_tot": ss_tot,
                "r2": r2,
            })

    return per_sample


def load_golden_per_sample() -> list[dict]:
    with open(_GOLDEN_PER_SAMPLE) as f:
        return json.load(f)


# ── Golden artifact existence ────────────────────────────────────────────

def test_golden_artifacts_exist():
    """Sanity: golden artefacts are present on disk."""
    assert _GOLDEN_PER_SAMPLE.is_file(), f"Missing {_GOLDEN_PER_SAMPLE}"
    assert _GOLDEN_SIMPLE_CONV_CKPT.is_file(), f"Missing {_GOLDEN_SIMPLE_CONV_CKPT}"
    assert _DREGON_VALID.is_dir(), f"Missing {_DREGON_VALID}"


def test_golden_per_sample_schema():
    """Each golden per-sample row has the expected keys."""
    rows = load_golden_per_sample()
    assert len(rows) == 600, f"expected 600 rows, got {len(rows)}"
    expected_keys = {"sample", "mse", "mae_frame", "mae_clip", "ss_tot", "r2"}
    for row in rows:
        assert expected_keys <= set(row.keys()), f"missing keys in {row['sample']}"


# ── Per-sample regression (the load-bearing test) ────────────────────────

@pytest.mark.slow
def test_per_sample_regression():
    """Full DREGON-LM valid inference must match golden per-sample metrics
    with rtol=1e-4, atol=1e-4."""
    golden = load_golden_per_sample()

    # Index by sample name for fast lookup.
    golden_by_sample = {row["sample"]: row for row in golden}

    new_rows = _run_inference(_GOLDEN_SIMPLE_CONV_CKPT, _DREGON_VALID)

    failures = []
    for row in new_rows:
        sid = row["sample"]
        g = golden_by_sample[sid]
        for key in ("mse", "mae_frame", "mae_clip", "ss_tot"):
            if not np.isclose(row[key], g[key], rtol=1e-4, atol=1e-4):
                failures.append(
                    f"  {sid}.{key}: got {row[key]:.6f}, golden {g[key]:.6f}"
                )
        # r2 may be None if degenerate; golden also may have None.
        if row["r2"] is None and g["r2"] is not None:
            failures.append(f"  {sid}.r2: got None, golden {g['r2']:.6f}")
        elif row["r2"] is not None and g["r2"] is None:
            failures.append(f"  {sid}.r2: got {row['r2']:.6f}, golden None")
        elif row["r2"] is not None and g["r2"] is not None:
            if not np.isclose(row["r2"], g["r2"], rtol=1e-4, atol=1e-4):
                failures.append(
                    f"  {sid}.r2: got {row['r2']:.6f}, golden {g['r2']:.6f}"
                )

    if failures:
        n = len(failures)
        msg = f"{n} per-sample metric mismatch(es) out of {len(new_rows)}:\n"
        msg += "\n".join(failures[:20])
        if n > 20:
            msg += f"\n  ... and {n - 20} more"
        pytest.fail(msg)


# ── Aggregate regression ─────────────────────────────────────────────────

@pytest.mark.slow
def test_aggregate_metrics():
    """Overall aggregate (MSE, R²) must match the golden within tolerance."""
    golden = load_golden_per_sample()
    new_rows = _run_inference(_GOLDEN_SIMPLE_CONV_CKPT, _DREGON_VALID)

    # Compute aggregate from golden.
    g_mse = float(np.mean([r["mse"] for r in golden]))
    g_mae_frame = float(np.mean([r["mae_frame"] for r in golden]))
    g_mae_clip = float(np.mean([r["mae_clip"] for r in golden]))
    g_r2_vals = [r["r2"] for r in golden if r["r2"] is not None]
    g_r2 = float(np.mean(g_r2_vals))

    # Compute aggregate from new.
    n_mse = float(np.mean([r["mse"] for r in new_rows]))
    n_mae_frame = float(np.mean([r["mae_frame"] for r in new_rows]))
    n_mae_clip = float(np.mean([r["mae_clip"] for r in new_rows]))
    n_r2_vals = [r["r2"] for r in new_rows if r["r2"] is not None]
    n_r2 = float(np.mean(n_r2_vals))

    assert np.isclose(n_mse, g_mse, rtol=1e-4, atol=1e-4), \
        f"MSE: got {n_mse:.6f}, golden {g_mse:.6f}"
    assert np.isclose(n_mae_frame, g_mae_frame, rtol=1e-4, atol=1e-4), \
        f"MAE frame: got {n_mae_frame:.6f}, golden {g_mae_frame:.6f}"
    assert np.isclose(n_mae_clip, g_mae_clip, rtol=1e-4, atol=1e-4), \
        f"MAE clip: got {n_mae_clip:.6f}, golden {g_mae_clip:.6f}"
    assert np.isclose(n_r2, g_r2, rtol=1e-4, atol=1e-4), \
        f"R²: got {n_r2:.6f}, golden {g_r2:.6f}"
    assert len(n_r2_vals) == len(g_r2_vals), \
        f"R² valid count: got {len(n_r2_vals)}, golden {len(g_r2_vals)}"
