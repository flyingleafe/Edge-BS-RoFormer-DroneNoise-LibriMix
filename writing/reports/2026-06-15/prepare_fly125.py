#!/usr/bin/env python3
"""Append the FLY125-in-training cross-drone result to the 2026-06-15 report.

Reads ``results/fly125_simpleconvv2_eval/`` (two SimpleConvV2-8ch checkpoints —
DREGON-only vs DREGON+Michael's-FLY125 — each evaluated on DREGON-LM-V4/valid and
FLY124 in-flight) plus the wandb PIT-loss histories, and emits report assets
without touching existing content:

- ``assets/metrics_table_fly125.typ``  — 2x2 (model x dataset) PIT metrics table
- ``assets/fly125_per_channel.png``    — per-channel PIT MAE, both datasets
- ``assets/fly125_loss_curves.png``    — train/val PIT-loss curves for both runs
- ``assets/fly125_<sample>.png``       — new-model FLY124 per-slice figures (copied)
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
EVAL_DIR = PROJECT_ROOT / "results" / "fly125_simpleconvv2_eval"
EVAL_JSON = EVAL_DIR / "metrics.json"
PER_CHANNEL_CSV = EVAL_DIR / "per_channel_comparison.csv"

MODELS = ["DREGON-only", "DREGON+FLY125"]
DATASETS = ["DREGON-LM-V4", "FLY124-inflight"]
DS_LABEL = {"DREGON-LM-V4": "DREGON-LM-V4 (in-domain)", "FLY124-inflight": "FLY124 (cross-drone)"}
SHOWN_SAMPLES = ["sample_00004", "sample_00006"]
_COLORS = {"DREGON-only": "#d62728", "DREGON+FLY125": "#2ca02c"}


def _load_metrics() -> dict:
    if not EVAL_JSON.is_file():
        raise FileNotFoundError(f"{EVAL_JSON} not found — run run_eval.py first.")
    with open(EVAL_JSON) as f:
        return json.load(f)


def write_table(metrics: dict) -> None:
    lines = [
        "#figure(",
        "  placement: none,",
        "  table(",
        "    columns: (auto, auto, auto, auto, auto, auto),",
        "    inset: 6pt,",
        "    align: (left + horizon, left + horizon, center + horizon, center + horizon, "
        "center + horizon, center + horizon),",
        "    table.header([*Training set*], [*Eval set*], [*RMSE (Hz)*], [*MAE frame (Hz)*], "
        "[*$R^2$ mean*], [*$R^2$ median*]),",
        "    table.hline(),",
    ]
    for ds in DATASETS:
        for m in MODELS:
            agg = metrics[m][ds]["pit"]
            lines.append(
                f"    [{m}], [{DS_LABEL[ds]}], [{agg['rmse']:.2f}], [{agg['mae_frame']:.2f}], "
                f"[{agg['r2_mean']:.3f}], [{agg['r2_median']:.3f}],"
            )
        lines.append("    table.hline(),")
    lines.extend(
        [
            "  ),",
            "  caption: [SimpleConvV2 (8ch) PIT metrics (best of 24 rotor permutations per "
            "channel). Adding Michael's FLY125 to training collapses the cross-drone FLY124 "
            "error (RMSE 7.96 #sym.arrow.r 1.63 Hz) at the cost of a modest in-domain "
            "DREGON-LM-V4 regression (1.62 #sym.arrow.r 2.77 Hz). FLY124 = 9 stable in-flight "
            "8 s slices; DREGON-LM-V4 = 30 valid clips, each #sym.times 8 channels.],",
            ") <tab:fly125>",
        ]
    )
    (ASSETS / "metrics_table_fly125.typ").write_text("\n".join(lines))
    print("Wrote assets/metrics_table_fly125.typ")


def _read_per_channel() -> dict[tuple[str, str], dict[int, dict[str, float]]]:
    out: dict[tuple[str, str], dict[int, dict[str, float]]] = defaultdict(dict)
    with open(PER_CHANNEL_CSV) as f:
        for row in csv.DictReader(f):
            out[(row["model"], row["dataset"])][int(row["channel"])] = {
                k: float(row[k]) for k in ("mae_frame", "rmse", "r2_median")
            }
    return out


def plot_per_channel() -> None:
    """Per-channel PIT MAE: DREGON-only vs DREGON+FLY125, one panel per eval set."""
    data = _read_per_channel()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    width = 0.4
    for ax, ds in zip(axes, DATASETS, strict=True):
        chans = sorted(data[(MODELS[0], ds)])
        x = np.arange(len(chans))
        for i, m in enumerate(MODELS):
            vals = [data[(m, ds)][c]["mae_frame"] for c in chans]
            ax.bar(x + (i - 0.5) * width, vals, width, label=m, color=_COLORS[m])
        ax.set_xticks(x)
        ax.set_xticklabels([f"ch{c}" for c in chans])
        ax.set_xlabel("microphone channel")
        ax.set_ylabel("PIT frame MAE (Hz)")
        ax.set_title(DS_LABEL[ds])
        ax.grid(axis="y", alpha=0.3)
        ax.legend()
    fig.suptitle(
        "SimpleConvV2 (8ch): per-channel RPS error — effect of adding FLY125 to training (PIT)",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(ASSETS / "fly125_per_channel.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Wrote assets/fly125_per_channel.png")


def plot_loss_curves() -> None:
    """Train and val PIT-loss curves (as RMSE in Hz) for both runs."""

    def _load(name: str) -> dict[str, np.ndarray]:
        ep, tr, va = [], [], []
        with open(EVAL_DIR / f"losses_{name}.csv") as f:
            for row in csv.DictReader(f):
                ep.append(float(row["epoch"]))
                tr.append(float(row["train/mse"]))
                va.append(float(row["val/pit_mse"]))
        return {"epoch": np.array(ep), "train": np.sqrt(tr), "val": np.sqrt(va)}

    runs = {"DREGON-only": _load("dregon"), "DREGON+FLY125": _load("michaels")}
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    for ax, split, title in zip(
        axes,
        ("train", "val"),
        ("Train PIT loss (RMSE, Hz)", "Validation PIT loss (RMSE, Hz)"),
        strict=True,
    ):
        for m, d in runs.items():
            ax.plot(d["epoch"], d[split], label=m, color=_COLORS[m], lw=1.8)
            if split == "val":
                k = int(np.argmin(d["val"]))
                ax.scatter([d["epoch"][k]], [d["val"][k]], color=_COLORS[m], zorder=5, s=40)
                ax.annotate(
                    f"best ep {int(d['epoch'][k])}\n{d['val'][k]:.2f} Hz",
                    (d["epoch"][k], d["val"][k]),
                    textcoords="offset points",
                    xytext=(8, 8),
                    fontsize=8,
                    color=_COLORS[m],
                )
        ax.set_xlabel("epoch")
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend()
    axes[0].set_ylabel("PIT RMSE (Hz)")
    fig.suptitle(
        "SimpleConvV2 training — DREGON-only vs DREGON+FLY125 "
        "(each run on its own split; val sets differ)",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(ASSETS / "fly125_loss_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Wrote assets/fly125_loss_curves.png")


def copy_figures() -> None:
    for sid in SHOWN_SAMPLES:
        src = EVAL_DIR / "figures" / f"{sid}.png"
        if not src.is_file():
            raise FileNotFoundError(f"{src} not found — run run_eval.py first.")
        shutil.copyfile(src, ASSETS / f"fly125_{sid}.png")
        print(f"Copied assets/fly125_{sid}.png")

    # Vertical old-vs-new comparison on an in-domain DREGON-LM-V4 slice.
    src = EVAL_DIR / "figures" / "v4_sample_00012_compare.png"
    if not src.is_file():
        raise FileNotFoundError(f"{src} not found — run run_eval.py first.")
    shutil.copyfile(src, ASSETS / "fly125_v4_compare.png")
    print("Copied assets/fly125_v4_compare.png")


def main() -> None:
    ASSETS.mkdir(exist_ok=True)
    metrics = _load_metrics()
    write_table(metrics)
    plot_per_channel()
    plot_loss_curves()
    copy_figures()


if __name__ == "__main__":
    main()
