#!/usr/bin/env python3
"""Append the FLY124 cross-drone evaluation to the 2026-06-15 report.

Reads the already-produced artifact ``results/fly124_simpleconvv2_eval/`` (the
DREGON-trained SimpleConvV2-8ch evaluated on stable in-flight slices of Michael's
FLY124 recording) and emits report assets, without touching existing content:

- ``assets/metrics_fly124.csv``        — regular vs PIT aggregate metrics
- ``assets/metrics_table_fly124.typ``  — Typst table for the new section
- ``assets/fly124_<sample>.png``       — representative per-slice figures (copied)
"""

from __future__ import annotations

import csv
import json
import shutil
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
ASSETS = Path(__file__).parent / "assets"
EVAL_DIR = PROJECT_ROOT / "results" / "fly124_simpleconvv2_eval"
EVAL_JSON = EVAL_DIR / "metrics.json"
PER_CHANNEL_CSV = EVAL_DIR / "per_channel_comparison.csv"

# Representative kept (stable in-flight) slices to show in the report.
SHOWN_SAMPLES = ["sample_00004", "sample_00006"]

_METRIC_KEYS = ["rmse", "mae_frame", "mae_clip", "r2_mean", "r2_median"]

_DS_IN, _DS_OUT = "DREGON-LM-V4", "FLY124-inflight"


def _load_metrics() -> dict:
    if not EVAL_JSON.is_file():
        raise FileNotFoundError(
            f"{EVAL_JSON} not found — run results/fly124_simpleconvv2_eval/run_eval.py first."
        )
    with open(EVAL_JSON) as f:
        return json.load(f)


def write_csv(metrics: dict) -> None:
    header = ["alignment", *_METRIC_KEYS, "n_samples", "n_rows"]
    rows = []
    for align in ("regular", "pit"):
        agg = metrics[align]
        rows.append(
            [align, *(f"{agg[k]:.4f}" for k in _METRIC_KEYS), agg["n_samples"], agg["n_rows"]]
        )
    lines = [",".join(map(str, header))] + [",".join(map(str, r)) for r in rows]
    (ASSETS / "metrics_fly124.csv").write_text("\n".join(lines) + "\n")
    print("Wrote assets/metrics_fly124.csv")


def write_table(metrics: dict) -> None:
    labels = {"regular": "Fixed-order", "pit": "PIT (oracle rotor match)"}
    lines = [
        "#figure(",
        "  placement: none,",
        "  table(",
        "    columns: (2fr, auto, auto, auto, auto, auto),",
        "    inset: 6pt,",
        "    align: (left + horizon, center + horizon, center + horizon, center + horizon, "
        "center + horizon, center + horizon),",
        "    table.header([*Alignment*], [*RMSE (Hz)*], [*MAE frame (Hz)*], [*MAE clip (Hz)*], "
        "[*$R^2$ mean*], [*$R^2$ median*]),",
        "    table.hline(),",
    ]
    for align in ("regular", "pit"):
        agg = metrics[align]
        lines.append(
            f"    [{labels[align]}], [{agg['rmse']:.2f}], [{agg['mae_frame']:.2f}], "
            f"[{agg['mae_clip']:.2f}], [{agg['r2_mean']:.3f}], [{agg['r2_median']:.3f}],"
        )
    n = metrics["pit"]
    lines.extend(
        [
            "  ),",
            f"  caption: [DREGON-trained SimpleConvV2 (8ch) on Michael's FLY124, "
            f"{n['n_samples']} stable in-flight 8 s slices ({n['n_rows']} sample×channel rows). "
            f"PIT = best of 24 rotor permutations per channel.],",
            ") <tab:fly124>",
        ]
    )
    (ASSETS / "metrics_table_fly124.typ").write_text("\n".join(lines))
    print("Wrote assets/metrics_table_fly124.typ")


def _read_per_channel() -> dict[str, dict[int, dict[str, float]]]:
    if not PER_CHANNEL_CSV.is_file():
        raise FileNotFoundError(f"{PER_CHANNEL_CSV} not found — run run_eval.py first.")
    out: dict[str, dict[int, dict[str, float]]] = defaultdict(dict)
    with open(PER_CHANNEL_CSV) as f:
        for row in csv.DictReader(f):
            out[row["dataset"]][int(row["channel"])] = {
                k: float(row[k]) for k in ("mae_frame", "rmse", "r2_median")
            }
    return out


def plot_per_channel_comparison() -> None:
    """Grouped per-channel barplot: in-domain DREGON-LM-V4 vs cross-drone FLY124."""
    data = _read_per_channel()
    chans = sorted(data[_DS_IN])
    x = np.arange(len(chans))
    width = 0.4

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    panels = [("mae_frame", "PIT frame MAE (Hz)"), ("rmse", "PIT RMSE (Hz)")]
    for ax, (metric, title) in zip(axes, panels, strict=True):
        in_dom = [data[_DS_IN][c][metric] for c in chans]
        cross = [data[_DS_OUT][c][metric] for c in chans]
        ax.bar(x - width / 2, in_dom, width, label="DREGON-LM-V4 (in-domain)", color="#1f77b4")
        ax.bar(x + width / 2, cross, width, label="FLY124 (cross-drone)", color="#d62728")
        ax.set_xticks(x)
        ax.set_xticklabels([f"ch{c}" for c in chans])
        ax.set_xlabel("microphone channel")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
        if metric == "mae_frame":
            ax.legend()
    fig.suptitle(
        "SimpleConvV2 (8ch): per-channel RPS error, in-domain vs cross-drone (PIT)",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(ASSETS / "fly124_vs_v4_per_channel.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Wrote assets/fly124_vs_v4_per_channel.png")


def copy_figures() -> None:
    for sid in SHOWN_SAMPLES:
        src = EVAL_DIR / "figures" / f"{sid}.png"
        if not src.is_file():
            raise FileNotFoundError(f"{src} not found — run run_eval.py first.")
        shutil.copyfile(src, ASSETS / f"fly124_{sid}.png")
        print(f"Copied assets/fly124_{sid}.png")


def main() -> None:
    ASSETS.mkdir(exist_ok=True)
    metrics = _load_metrics()
    write_csv(metrics)
    write_table(metrics)
    plot_per_channel_comparison()
    copy_figures()


if __name__ == "__main__":
    main()
