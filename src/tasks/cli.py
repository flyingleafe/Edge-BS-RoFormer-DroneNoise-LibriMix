"""Thin typer CLI for ``evaluate-rps``.

All substance lives in ``tasks.rps_prediction``; this file is a command-line
veneer with argument parsing and output formatting.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer

from tasks.rps_prediction import (
    load_input_set,
    load_predictor,
    evaluate,
    EvalResult,
)

app = typer.Typer(
    name="evaluate-rps",
    help="Evaluate RPS prediction models on an input set.",
    no_args_is_help=True,
)

_OUTPUT_FORMATS = {"json", "tex"}


@app.callback(invoke_without_command=True)
def main(
    input_set: str = typer.Option(
        ..., "--input-set", "-i",
        help="Path to DREGON-LM dataset directory (e.g. datasets/DREGON-LM/valid).",
    ),
    models: list[str] = typer.Option(
        ..., "--model", "-m",
        help="Model spec: 'Type@/path/to/ckpt.pt' or classical name ('cepstral', 'hps', 'pyin', 'nmf', 'matched_filter'). Repeat for multi-model comparison.",
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o",
        help="Path for JSON metrics output.",
    ),
    tex: Optional[Path] = typer.Option(
        None, "--tex",
        help="Path for LaTeX table output (per-SNR stratification).",
    ),
    alignment: str = typer.Option(
        "stft_timestamps", "--alignment",
        help="GT alignment strategy: 'stft_timestamps' (canon) or 'shape_stretch' (legacy).",
    ),
    verbose: bool = typer.Option(
        True, "--verbose/--quiet",
        help="Print progress.",
    ),
) -> None:
    """Evaluate one or more RPS predictors on an input set."""
    if not models:
        typer.echo("Error: at least one --model is required.", err=True)
        raise typer.Exit(1)

    dataset_path = Path(input_set)
    if not dataset_path.is_dir():
        typer.echo(f"Error: input set not found: {input_set}", err=True)
        raise typer.Exit(1)

    results: list[EvalResult] = []
    for spec in models:
        if verbose:
            typer.echo(f"Loading predictor: {spec}")
        predictor = load_predictor(spec)

        if verbose:
            typer.echo(f"Loading samples from {input_set} ...")
        samples = list(load_input_set(str(dataset_path)))

        if verbose:
            typer.echo(f"Evaluating {spec} ({len(samples)} samples) ...")
        result = evaluate(
            predictor,
            samples,
            model_spec=spec,
            input_set_label=dataset_path.name,
            alignment=alignment,
            verbose=verbose,
        )
        results.append(result)

        if verbose:
            agg = result.aggregate
            typer.echo(
                f"  {spec}: MSE={agg['mse']:.4f}  RMSE={agg['rmse']:.3f}  "
                f"MAE/clip={agg['mae_clip']:.3f}  R²={agg['r2_mean']:.4f}  "
                f"({agg['elapsed_s']}s)"
            )

    # Save JSON output.
    if output:
        out_data = {
            "models": [r.model_spec for r in results],
            "input_set": str(dataset_path),
            "alignment": alignment,
            "results": [r.aggregate for r in results],
            "per_sample": [r.per_sample for r in results],
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w") as f:
            json.dump(out_data, f, indent=2)
        if verbose:
            typer.echo(f"Wrote {output}")

    # LaTeX table (per-SNR).
    if tex and results:
        _write_tex_table(tex, results)
        if verbose:
            typer.echo(f"Wrote {tex}")

    # Multi-model summary.
    if len(results) > 1 and verbose:
        typer.echo("\n" + "=" * 60)
        typer.echo(f"{'Model':<35} {'MSE':>8} {'RMSE':>8} {'MAE/clip':>9} {'R²':>9}")
        typer.echo("-" * 72)
        for r in results:
            a = r.aggregate
            typer.echo(
                f"{r.model_spec:<35} {a['mse']:8.2f} {a['rmse']:8.2f} "
                f"{a['mae_clip']:9.3f} {a['r2_mean']:9.4f}"
            )


def _write_tex_table(path: Path, results: list[EvalResult]) -> None:
    """Write a simple LaTeX per-SNR table for the first result."""
    if not results:
        return
    per_snr = results[0].per_snr()
    lines = [
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"SNR range & $n$ & MSE & MAE$_{\text{frame}}$ & MAE$_{\text{clip}}$ & $R^2$ \\",
        r"\midrule",
    ]
    for row in per_snr:
        lines.append(
            f"${row['snr_range']}$ & {row.get('n', 0)} & "
            f"{row.get('mse_mean', 0):.2f} & "
            f"{row.get('mae_frame_mean', 0):.2f} & "
            f"{row.get('mae_clip_mean', 0):.2f} & "
            f"{row.get('r2_mean', 0):.2f} \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
