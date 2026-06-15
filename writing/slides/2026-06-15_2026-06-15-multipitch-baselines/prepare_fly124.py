#!/usr/bin/env python3
"""Append the FLY124 cross-drone evaluation to the 2026-06-15 slide deck.

From the already-produced artifact ``results/fly124_simpleconvv2_eval/`` this
copies the representative per-slice figures and renders the per-channel
in-domain-vs-cross-drone barplot into the deck's ``assets/`` (suffix
``fly124_``), without touching existing content. Headline numbers are written
inline in ``slides.typ``; see ``results/fly124_simpleconvv2_eval/metrics.json``.
"""

from __future__ import annotations

import csv
import shutil
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
ASSETS = Path(__file__).parent / "assets"
EVAL_DIR = PROJECT_ROOT / "results" / "fly124_simpleconvv2_eval"
PER_CHANNEL_CSV = EVAL_DIR / "per_channel_comparison.csv"

SHOWN_SAMPLES = ["sample_00004", "sample_00006"]
_DS_IN, _DS_OUT = "DREGON-LM-V4", "FLY124-inflight"


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
            raise FileNotFoundError(
                f"{src} not found — run results/fly124_simpleconvv2_eval/run_eval.py first."
            )
        shutil.copyfile(src, ASSETS / f"fly124_{sid}.png")
        print(f"Copied assets/fly124_{sid}.png")


def main() -> None:
    ASSETS.mkdir(exist_ok=True)
    plot_per_channel_comparison()
    copy_figures()


if __name__ == "__main__":
    main()
