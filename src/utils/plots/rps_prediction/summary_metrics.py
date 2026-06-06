# src/utils/plots/rps_prediction/summary_metrics.py
"""Bar chart of RMSE/MAE/R² across models from an EvalResult."""
from __future__ import annotations

import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np

from tasks.rps_prediction import EvalResult


def plot_summary_metrics(
    *,
    results: list[EvalResult] | None = None,
    models: list[str] | None = None,
    metric: str = "all",
    ax=None,
    figsize: tuple[float, float] = (14, 5),
    **style,
) -> matplotlib.figure.Figure:
    """Bar chart comparing aggregate metrics across models.

    Parameters
    ----------
    results : list of EvalResult
        One per model.
    models : list of str
        Model names (must align with ``results``).
    metric : str
        Which metric to plot: ``"mse"``, ``"mae"``, ``"r2"``, or ``"all"``
        (3 subplots).
    """

    if not results:
        raise ValueError("results is required")

    if models is None:
        models = [r.model_spec for r in results]
    if len(models) != len(results):
        raise ValueError(f"models ({len(models)}) and results ({len(results)}) length mismatch")

    labels = [m[:25] for m in models]  # truncate long specs
    colors = plt.cm.tab10(np.linspace(0, 1, len(models)))

    metrics = {
        "MSE": [r.aggregate["mse"] for r in results],
        "RMSE": [r.aggregate["rmse"] for r in results],
        "MAE (clip)": [r.aggregate["mae_clip"] for r in results],
        "R²": [r.aggregate["r2_mean"] for r in results],
    }

    fig, axes = plt.subplots(1, 4, figsize=figsize)
    for ax, (title, vals) in zip(axes, metrics.items()):
        bars = ax.bar(labels, vals, color=colors, alpha=0.85)
        ax.set_title(title)
        ax.tick_params(axis='x', rotation=45, labelsize=8)
        ax.grid(True, axis="y", alpha=0.3)
        # Add value labels on bars
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(vals) * 0.01,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    return fig
