# src/utils/plots/rps_prediction/per_snr.py
"""Per-SNR metric lines/bars comparing models across SNR bins."""

from __future__ import annotations

import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np

from tasks.rps_prediction import EvalResult


def plot_per_snr(
    *,
    results: list[EvalResult] | None = None,
    models: list[str] | None = None,
    metric: str = "mse",
    ax=None,
    figsize: tuple[float, float] = (10, 5),
    **style,
) -> matplotlib.figure.Figure:
    """Line/bar chart of a metric stratified by SNR bin across models.

    Parameters
    ----------
    results : list of EvalResult
        One per model.
    models : list of str
        Model names (must align with ``results``).
    metric : str
        ``"mse"``, ``"mae_frame"``, ``"mae_clip"``, or ``"r2"``.
    """
    if not results:
        raise ValueError("results is required")
    if models is None:
        models = [r.model_spec for r in results]

    metric_key = f"{metric}_mean"

    fig, ax = plt.subplots(figsize=figsize)
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(results), 10)))  # pyright: ignore[reportAttributeAccessIssue]

    for i, (result, model_name) in enumerate(zip(results, models)):
        per_snr = result.per_snr()
        overall_rows = [r for r in per_snr if r["snr_range"] != "Overall"]
        snr_labels = [r["snr_range"] for r in overall_rows]
        vals = [r.get(metric_key, 0) for r in overall_rows]
        ax.plot(
            snr_labels,
            vals,
            "o-",
            color=colors[i],
            label=model_name[:20],
            linewidth=2,
            markersize=6,
        )

    ax.set_xlabel("SNR range")
    ax.set_ylabel(metric.replace("_", " ").upper() if metric != "r2" else "R²")
    ax.set_title(f"Per-SNR {metric.upper()}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis="x", rotation=30)

    plt.tight_layout()
    return fig
