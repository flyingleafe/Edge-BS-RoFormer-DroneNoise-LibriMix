"""Thin typer CLI for ``make-plot``.

Dispatches to registered plot functions by dotted name::

    make-plot --type=rps_prediction.sample_comparison --sample=<path> ...
"""

from __future__ import annotations

import contextlib
import importlib
import json
from pathlib import Path

import typer

from plots import get_plot_fn, list_plot_types

app = typer.Typer(
    name="make-plot",
    help="Generate plots from evaluation results by plot type.",
    no_args_is_help=True,
)


@app.callback(invoke_without_command=True)
def main(
    plot_type: str = typer.Option(
        ...,
        "--type",
        "-t",
        help="Plot type dotted name (rps_prediction.sample_comparison, ...).",
    ),
    sample: str | None = typer.Option(
        None,
        "--sample",
        help="Path to a sample directory for per-sample plots.",
    ),
    results: str | None = typer.Option(
        None,
        "--results",
        help="Path to eval results JSON (from evaluate-rps) for result-based plots.",
    ),
    log_paths: list[str] | None = typer.Option(
        None,
        "--log",
        "-l",
        help="Path(s) to training_log.csv for training curves.",
    ),
    models: list[str] | None = typer.Option(
        None,
        "--model",
        "-m",
        help="Name(s) for the legend (repeatable).",
    ),
    output: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Output path (PNG or PDF).  Default: <type>.pdf",
    ),
    metric: str = typer.Option(
        "mse", "--metric", help="Metric for summary plots (mse, mae_frame, mae_clip, r2)."
    ),
    list_types: bool = typer.Option(
        False,
        "--list",
        help="List available plot types and exit.",
    ),
) -> None:
    """Dispatch to a registered plot function."""
    # Ensure plot sub-packages are imported (triggers registration).
    _import_plot_packages()

    if list_types:
        typer.echo("Available plot types:")
        for name in list_plot_types():
            typer.echo(f"  {name}")
        return

    fn = get_plot_fn(plot_type)
    kwargs: dict = {}

    # Build kwargs based on what the plot function needs.
    if sample:
        kwargs["sample_path"] = sample

    if results:
        with open(results) as f:
            data = json.load(f)
        # If the results JSON has a standard structure, unwrap it.
        if isinstance(data, dict) and "results" in data:
            from tasks.rps_prediction import EvalResult

            # Reconstruct EvalResult list (lightweight — metrics only).
            result_objects = []
            for i, agg in enumerate(data.get("results", [])):
                per_sample_data = (
                    data.get("per_sample", [[]])[i] if i < len(data.get("per_sample", [])) else []
                )
                r = EvalResult(
                    per_sample=per_sample_data,
                    aggregate=agg,
                    model_spec=data.get("models", ["?"])[i]
                    if i < len(data.get("models", ["?"]))
                    else "?",
                    input_set_label=data.get("input_set", ""),
                )
                result_objects.append(r)
            kwargs["results"] = result_objects

    if log_paths:
        kwargs["log_paths"] = log_paths

    if models:
        kwargs["models"] = models

    if metric and plot_type != "rps_prediction.sample_comparison":
        kwargs["metric"] = metric

    # Generate the figure.
    fig = fn(**kwargs)

    out = output or Path(f"{plot_type.replace('.', '_')}.pdf")
    fig.savefig(str(out), dpi=150, bbox_inches="tight")
    typer.echo(f"Wrote {out}")
    import matplotlib.pyplot as plt

    plt.close(fig)


def _import_plot_packages() -> None:
    """Import known plot sub-packages to trigger registration."""
    for mod_name in ["plots.rps_prediction", "plots.timeframe"]:
        with contextlib.suppress(ImportError):
            importlib.import_module(mod_name)


if __name__ == "__main__":
    main()
