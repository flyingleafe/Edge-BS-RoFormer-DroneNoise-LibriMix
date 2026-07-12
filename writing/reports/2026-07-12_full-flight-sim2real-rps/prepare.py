#!/usr/bin/env python3
"""Generate figures + the results table for the full-flight sim2real report.

Self-contained. Figures that depend only on code/known measurements are always
regenerated; the per-regime comparison + table read `assets/results.json`
(baseline is known; the E11 curriculum rows are filled as the runs finish).
"""

import json
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
ASSETS = pathlib.Path(__file__).resolve().parent / "assets"
ASSETS.mkdir(exist_ok=True)

BLUE, ORANGE, GREY = "#2c6fbb", "#e08a1e", "#888888"


def fig_fullflight():
    from data_processing import rps_synthesis as rs

    w = rs.generate_full_flight(70.0, 100.0, drone_profile=0.0, rng=3)  # (4, M)
    t = np.arange(w.shape[1]) / 100.0
    fig, ax = plt.subplots(figsize=(7.2, 2.7))
    ax.plot(t, w.mean(0), color=BLUE, lw=1.3)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("rotor speed (rev/s)")
    ax.set_ylim(-4, 95)
    for y, lab in [(0, "ground"), (36, "warm-up"), (80, "cruise")]:
        ax.axhline(y, color=GREY, ls=":", lw=0.7)
        ax.text(t[-1] * 1.005, y, lab, va="center", fontsize=8, color=GREY)
    ax.margins(x=0)
    fig.tight_layout()
    fig.savefig(ASSETS / "fullflight.png", dpi=150)
    plt.close(fig)


def fig_silence_fade():
    rps = np.array([0, 5, 10, 20, 40, 80])
    before = np.array([0.01201, 0.01072, 0.00973, 0.00970, 0.04400, 0.06897])
    after = np.array([0.00000, 0.00541, 0.00925, 0.00981, 0.04530, 0.07697])
    fig, ax = plt.subplots(figsize=(5.4, 2.9))
    ax.plot(rps, before, "o-", color=ORANGE, label="before (DC pedestal)")
    ax.plot(rps, after, "s-", color=BLUE, label="after (silence gate)")
    ax.set_xlabel("rotor speed (rev/s)")
    ax.set_ylabel("generator output RMS")
    ax.axvline(10, color=GREY, ls=":", lw=0.7)
    ax.text(10.5, ax.get_ylim()[1] * 0.9, "fade knee (10)", fontsize=7.5, color=GREY)
    ax.legend(frameon=False, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(ASSETS / "silence_fade.png", dpi=150)
    plt.close(fig)


def fig_trainval():
    """train vs val curve from the un-augmented E10 run (overfitting fingerprint)."""
    try:
        import wandb

        api = wandb.Api()
        r = api.runs(
            "flyingleafe/harmonic-noise-suppression",
            filters={"display_name": "e10_full_unigru128"},
            order="-created_at",
        )[0]
        h = r.history(keys=["epoch", "train/loss", "val/mse"], pandas=True).dropna(
            subset=["val/mse"]
        )
        ep = h["epoch"].astype(float).to_numpy()
        tr = np.sqrt(h["train/loss"].astype(float).clip(lower=0).to_numpy())
        va = np.sqrt(h["val/mse"].astype(float).to_numpy())
    except Exception as e:  # offline / no run: skip gracefully
        print(f"  (trainval: wandb unavailable, {e}); keeping any existing figure")
        return
    fig, ax = plt.subplots(figsize=(5.4, 2.9))
    ax.plot(ep, tr, "-", color=BLUE, label="train (synthetic)")
    ax.plot(ep, va, "-", color=ORANGE, label="validation (real)")
    ax.set_xlabel("epoch")
    ax.set_ylabel("error (rev/s, RMSE)")
    ax.legend(frameon=False, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(ASSETS / "trainval.png", dpi=150)
    plt.close(fig)


def fig_and_table_results():
    res = json.loads((ASSETS / "results.json").read_text())
    regimes = ["cruise", "warmup", "ground"]
    baseline = res["baseline"]  # {arch: {regime: mse, agg: ...}}
    curric = res.get("curriculum", {})
    archs = ["transformer", "uni_gru128", "scv2"]

    # grouped bars: for each regime, baseline vs curriculum (mean over archs)
    def regime_mean(d, reg):
        vals = [d[a][reg] for a in archs if a in d and d[a].get(reg) is not None]
        return float(np.mean(vals)) if vals else np.nan

    x = np.arange(len(regimes))
    bl = [regime_mean(baseline, r) for r in regimes]
    cu = [regime_mean(curric, r) for r in regimes]
    fig, ax = plt.subplots(figsize=(6.6, 3.0))
    ax.bar(x - 0.2, bl, 0.38, color=ORANGE, label="real-only + time-warp")
    ax.bar(x + 0.2, cu, 0.38, color=BLUE, label="full-flight curriculum")
    ax.set_xticks(x)
    ax.set_xticklabels([r.capitalize() for r in regimes])
    ax.set_ylabel("PIT-MSE (lower better)")
    ax.set_yscale("log")
    ax.legend(frameon=False, fontsize=8.5)
    for i, (b, c) in enumerate(zip(bl, cu, strict=False)):
        if not np.isnan(b):
            ax.text(i - 0.2, b, f"{b:.0f}", ha="center", va="bottom", fontsize=7.5)
        if not np.isnan(c):
            ax.text(i + 0.2, c, f"{c:.0f}", ha="center", va="bottom", fontsize=7.5)
    fig.tight_layout()
    fig.savefig(ASSETS / "regime_comparison.png", dpi=150)
    plt.close(fig)

    # Typst results table
    def row(name, d, a):
        if a not in d:
            return f"    [{name}], [--], [--], [--], [--],"
        e = d[a]

        def cell(k):
            return f"{e[k]:.1f}" if e.get(k) is not None else "--"

        return f"    [{name}], [{cell('cruise')}], [{cell('warmup')}], [{cell('ground')}], [{cell('agg')}],"

    lines = [
        "#figure(",
        "  table(",
        "    columns: 5, align: (left, right, right, right, right), stroke: 0.5pt,",
        "    table.header([Model], [Cruise], [Warm-up], [Ground], [All]),",
    ]
    labels = {"transformer": "Transformer", "uni_gru128": "Uni-GRU-128", "scv2": "SimpleConv-v2"}
    for a in archs:
        lines.append("    table.cell(colspan: 5)[" + f"#emph[{labels[a]}]" + "],")
        lines.append(row("real-only + warp", baseline, a))
        lines.append(row("full-flight curriculum", curric, a))
    lines += [
        "  ),",
        "  caption: [Per-regime and overall PIT-MSE on the full-envelope real validation "
        "set. Lower is better.],",
        ") <tab-results>",
        "",
    ]
    (ASSETS / "results_table.typ").write_text("\n".join(lines))


def main():
    fig_fullflight()
    fig_silence_fade()
    fig_trainval()
    fig_and_table_results()
    print("figures + table written to assets/")


if __name__ == "__main__":
    main()
