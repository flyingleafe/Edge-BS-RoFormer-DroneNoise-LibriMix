"""Smoke tests for `plots.rps_prediction` plot functions."""

from __future__ import annotations

import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.figure
import numpy as np
import pytest
import tdseries as td

from data_processing.frames import with_meta
from tasks.rps_prediction import EvalResult

# ── Synthetic helpers ────────────────────────────────────────────────────


def _make_eval_result(model_spec: str, mse: float, r2: float, n: int = 10) -> EvalResult:
    """Build a minimal EvalResult with per-SNR stratification."""
    rows = []
    for i in range(n):
        snr = -30.0 + i * 3.0  # spread across SNR bins
        rows.append(
            {
                "sample": f"s{i:03d}",
                "mse": mse + np.random.uniform(-0.1, 0.1),
                "mae_frame": np.sqrt(mse) + np.random.uniform(-0.05, 0.05),
                "mae_clip": np.sqrt(mse) * 0.8,
                "ss_tot": 10.0,
                "r2": max(-1.0, min(1.0, r2 + np.random.uniform(-0.05, 0.05))),
                "input_snr": snr,
            }
        )
    r2_vals = [r["r2"] for r in rows]
    agg = {
        "n_samples": n,
        "n_r2_valid": n,
        "mse": float(np.mean([r["mse"] for r in rows])),
        "rmse": float(np.sqrt(np.mean([r["mse"] for r in rows]))),
        "mae_frame": float(np.mean([r["mae_frame"] for r in rows])),
        "mae_clip": float(np.mean([r["mae_clip"] for r in rows])),
        "r2_mean": float(np.mean(r2_vals)),
        "r2_median": float(np.median(r2_vals)),
        "r2_std": float(np.std(r2_vals)),
        "elapsed_s": 0.5,
    }
    return EvalResult(per_sample=rows, aggregate=agg, model_spec=model_spec)


def _make_frame() -> td.Frame:
    """Build a minimal Frame with audio + RPS entries."""
    sr = 16000.0
    audio = np.random.randn(16000).astype(np.float32)
    rps = np.random.uniform(100, 200, (4, 50)).astype(np.float64)
    dur = len(audio) / sr
    motor_sr = rps.shape[1] / dur
    motor_times = np.arange(rps.shape[1]) / motor_sr

    audio_series = td.uniform(audio, sr, dims=("time",), t_start=0.0)
    rps_series = td.events(motor_times, rps, dims=("rotor", "time"), t_start=0.0, t_end=dur)
    frame = td.Frame({"audio": audio_series, "rps": rps_series})
    return with_meta(frame, id="test_sample")


# ── plot_summary_metrics ─────────────────────────────────────────────────


def test_plot_summary_metrics_returns_figure():
    from plots.rps_prediction.summary_metrics import plot_summary_metrics

    results = [
        _make_eval_result("model_a", mse=2.0, r2=0.85),
        _make_eval_result("model_b", mse=1.5, r2=0.90),
    ]
    fig = plot_summary_metrics(results=results)
    assert isinstance(fig, matplotlib.figure.Figure)


def test_plot_summary_metrics_empty_raises():
    from plots.rps_prediction.summary_metrics import plot_summary_metrics

    with pytest.raises(ValueError, match="required"):
        plot_summary_metrics(results=[])


def test_plot_summary_metrics_length_mismatch_raises():
    from plots.rps_prediction.summary_metrics import plot_summary_metrics

    results = [_make_eval_result("a", 2.0, 0.8)]
    with pytest.raises(ValueError, match="length mismatch"):
        plot_summary_metrics(results=results, models=["a", "b"])


# ── plot_per_snr ─────────────────────────────────────────────────────────


def test_plot_per_snr_returns_figure():
    from plots.rps_prediction.per_snr import plot_per_snr

    results = [
        _make_eval_result("model_a", mse=2.0, r2=0.85, n=20),
        _make_eval_result("model_b", mse=1.5, r2=0.90, n=20),
    ]
    fig = plot_per_snr(results=results)
    assert isinstance(fig, matplotlib.figure.Figure)


def test_plot_per_snr_empty_raises():
    from plots.rps_prediction.per_snr import plot_per_snr

    with pytest.raises(ValueError, match="required"):
        plot_per_snr(results=[])


# ── plot_full_sequence ───────────────────────────────────────────────────


def test_plot_full_sequence_returns_figure():
    from plots.rps_prediction.full_sequence import plot_full_sequence

    sr = 16000.0
    audio = np.random.randn(32000).astype(np.float32)
    n_frames = len(audio) // 512 + 1
    rps_gt = np.random.uniform(100, 200, (4, n_frames)).astype(np.float32)
    rps_pred = rps_gt + np.random.randn(4, n_frames).astype(np.float32) * 2

    fig = plot_full_sequence(audio=audio, rps_gt=rps_gt, rps_pred=rps_pred, sr=sr)
    assert isinstance(fig, matplotlib.figure.Figure)


def test_plot_full_sequence_missing_input_raises():
    from plots.rps_prediction.full_sequence import plot_full_sequence

    with pytest.raises(ValueError, match="required"):
        plot_full_sequence()


# ── plot_sample_comparison ───────────────────────────────────────────────


def test_plot_sample_comparison_returns_figure():
    from plots.rps_prediction.sample_comparison import plot_sample_comparison

    frame = _make_frame()
    preds = {
        "model_a": np.random.uniform(100, 200, (4, 31)).astype(np.float32),
        "model_b": np.random.uniform(100, 200, (4, 31)).astype(np.float32),
    }
    fig = plot_sample_comparison(sample=frame, preds=preds)
    assert isinstance(fig, matplotlib.figure.Figure)


def test_plot_sample_comparison_no_args_raises():
    from plots.rps_prediction.sample_comparison import plot_sample_comparison

    with pytest.raises(ValueError, match="required"):
        plot_sample_comparison()


def test_plot_sample_comparison_bad_channel_raises():
    from plots.rps_prediction.sample_comparison import plot_sample_comparison

    frame = _make_frame()
    with pytest.raises(ValueError, match="channel"):
        plot_sample_comparison(sample=frame, channel="bogus")


# ── plot_training_curves ─────────────────────────────────────────────────


def test_plot_training_curves_returns_figure(tmp_path):
    from plots.rps_prediction.training_curves import plot_training_curves

    # Write a minimal training log CSV.
    log_path = tmp_path / "training_log.csv"
    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["epoch", "train_mse", "val_mse", "train_r2", "val_r2"]
        )
        writer.writeheader()
        for e in range(1, 6):
            writer.writerow(
                {
                    "epoch": str(e),
                    "train_mse": f"{10.0 / e:.4f}",
                    "val_mse": f"{12.0 / e:.4f}",
                    "train_r2": f"{0.5 + 0.1 * e:.4f}",
                    "val_r2": f"{0.4 + 0.1 * e:.4f}",
                }
            )

    fig = plot_training_curves(log_paths=[str(log_path)], labels=["test"])
    assert isinstance(fig, matplotlib.figure.Figure)


def test_plot_training_curves_empty_raises():
    from plots.rps_prediction.training_curves import plot_training_curves

    with pytest.raises(ValueError, match="required"):
        plot_training_curves(log_paths=[])


def test_plot_training_curves_length_mismatch_raises(tmp_path):
    from plots.rps_prediction.training_curves import plot_training_curves

    log_path = tmp_path / "log.csv"
    log_path.write_text("epoch,train_mse\n1,5.0\n")

    with pytest.raises(ValueError, match="length mismatch"):
        plot_training_curves(log_paths=[str(log_path)], labels=["a", "b"])
