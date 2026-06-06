# src/utils/plots/rps_prediction/training_curves.py
"""Training curves plot from training log CSV."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np


def plot_training_curves(
    *,
    log_paths: list[str] | None = None,
    labels: list[str] | None = None,
    ax=None,
    figsize: tuple[float, float] = (12, 5),
    **style,
) -> matplotlib.figure.Figure:
    """Plot train/val MSE and R² from one or more training logs.

    Parameters
    ----------
    log_paths : list of str
        Paths to training_log.csv files.
    labels : list of str
        Labels for the legend (must match log_paths length).
    """
    if not log_paths:
        raise ValueError("log_paths is required")
    if labels is None:
        labels = [Path(p).parent.name for p in log_paths]
    if len(labels) != len(log_paths):
        raise ValueError("length mismatch between log_paths and labels")

    colors = plt.cm.tab10(np.linspace(0, 1, len(log_paths)))

    fig, (ax_mse, ax_r2) = plt.subplots(1, 2, figsize=figsize)

    for log_path, label, color in zip(log_paths, labels, colors):
        data = _read_training_log(log_path)
        if not data:
            continue
        epochs = [d["epoch"] for d in data]

        # Train/val MSE
        train_mse = [d.get("train_mse") for d in data]
        val_mse = [d.get("val_mse") for d in data]
        if any(v is not None for v in train_mse):
            ax_mse.plot(epochs, train_mse, '-', color=color, alpha=0.5, linewidth=1)
        if any(v is not None for v in val_mse):
            ax_mse.plot(epochs, val_mse, 'o-', color=color, label=label,
                        markersize=3, linewidth=1.5)

        # Train/val R²
        train_r2 = [d.get("train_r2") for d in data]
        val_r2 = [d.get("val_r2") for d in data]
        if any(v is not None for v in train_r2):
            ax_r2.plot(epochs, train_r2, '-', color=color, alpha=0.5, linewidth=1)
        if any(v is not None for v in val_r2):
            ax_r2.plot(epochs, val_r2, 'o-', color=color, label=label,
                       markersize=3, linewidth=1.5)

    ax_mse.set_xlabel("Epoch")
    ax_mse.set_ylabel("MSE")
    ax_mse.set_title("Training / Validation MSE")
    ax_mse.grid(True, alpha=0.3)
    ax_mse.legend(fontsize=7)

    ax_r2.set_xlabel("Epoch")
    ax_r2.set_ylabel("R²")
    ax_r2.set_title("Training / Validation R²")
    ax_r2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def _read_training_log(path: str) -> list[dict]:
    """Read training_log.csv, return list of epoch dicts."""
    rows: list[dict] = []
    try:
        with open(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                d: dict = {}
                for k, v in row.items():
                    try:
                        d[k] = float(v)
                    except (ValueError, TypeError):
                        d[k] = v
                d["epoch"] = int(d.get("epoch", len(rows)))
                rows.append(d)
    except (FileNotFoundError, OSError):
        pass
    return rows
