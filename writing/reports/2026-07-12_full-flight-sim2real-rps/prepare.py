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


GREEN = "#2e8b57"
CONDS = [
    ("baseline", "real-only, cruise-trained", ORANGE),
    ("curriculum", "sim full-flight curriculum", BLUE),
    ("real_fullflight", "real full-flight (min_rps=0)", GREEN),
]


def fig_tracking():
    """Mean predicted vs mean true RPS per regime, for the 3 best (transformer)
    models. The point: none collapses to a global mean — all track the regime."""
    # measured (mean_gt, mean_pred) per regime, transformer arch, from the sweep.
    gt = {"cruise": 78.9, "warmup": 34.5, "ground": 0.0}
    pred = {
        "baseline": {"cruise": 78.9, "warmup": 41.8, "ground": 48.0},
        "curriculum": {"cruise": 81.0, "warmup": 23.3, "ground": 12.6},
        "real_fullflight": {"cruise": 79.5, "warmup": 38.0, "ground": 14.7},
    }
    regimes = ["ground", "warmup", "cruise"]
    xg = [gt[r] for r in regimes]
    fig, ax = plt.subplots(figsize=(5.6, 3.1))
    ax.plot([-2, 85], [-2, 85], color=GREY, ls="--", lw=0.8, label="perfect (pred = true)")
    for key, lab, col in CONDS:
        yp = [pred[key][r] for r in regimes]
        ax.plot(xg, yp, "o-", color=col, lw=1.3, ms=5, label=lab)
    ax.set_xlabel("true mean rotor speed (rev/s)")
    ax.set_ylabel("predicted mean (rev/s)")
    ax.set_xlim(-4, 88)
    ax.set_ylim(-4, 88)
    ax.axhline(48.7, color=GREY, ls=":", lw=0.6)
    ax.text(2, 50.5, "global mean (~49)", fontsize=7, color=GREY)
    ax.legend(frameon=False, fontsize=7.8, loc="lower right")
    fig.tight_layout()
    fig.savefig(ASSETS / "tracking.png", dpi=150)
    plt.close(fig)


def fig_and_table_results():
    res = json.loads((ASSETS / "results.json").read_text())
    regimes = ["cruise", "warmup", "ground"]
    archs = ["transformer", "uni_gru128", "scv2"]

    def regime_mean(d, reg):
        vals = [d[a][reg] for a in archs if a in d and d[a].get(reg) is not None]
        return float(np.mean(vals)) if vals else np.nan

    # grouped bars: for each regime, the 3 conditions (mean over archs)
    x = np.arange(len(regimes))
    w = 0.26
    offs = [-w, 0.0, w]
    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    for (key, lab, col), off in zip(CONDS, offs, strict=False):
        vals = [regime_mean(res.get(key, {}), r) for r in regimes]
        ax.bar(x + off, vals, w, color=col, label=lab)
        for i, v in enumerate(vals):
            if not np.isnan(v):
                ax.text(i + off, v, f"{v:.0f}", ha="center", va="bottom", fontsize=6.8)
    ax.set_xticks(x)
    ax.set_xticklabels([r.capitalize() for r in regimes])
    ax.set_ylabel("PIT-MSE (lower better)")
    ax.set_yscale("log")
    ax.legend(frameon=False, fontsize=8.0)
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
        "    table.header([Training data], [Cruise], [Warm-up], [Ground], [All]),",
    ]
    labels = {"transformer": "Transformer", "uni_gru128": "Uni-GRU-128", "scv2": "SimpleConv-v2"}
    rlabel = {
        "baseline": "real-only (cruise)",
        "curriculum": "sim full-flight curriculum",
        "real_fullflight": "real full-flight",
    }
    for a in archs:
        lines.append("    table.cell(colspan: 5)[" + f"#emph[{labels[a]}]" + "],")
        for key in rlabel:
            lines.append(row(rlabel[key], res.get(key, {}), a))
    lines += [
        "  ),",
        "  caption: [Per-regime and overall PIT-MSE on the full-envelope real validation "
        "set (27 cruise / 6 warm-up / 4 ground clips). Lower is better; bold-worthy numbers "
        "discussed in text.],",
        ") <tab-results>",
        "",
    ]
    (ASSETS / "results_table.typ").write_text("\n".join(lines))


def main():
    fig_fullflight()
    fig_silence_fade()
    fig_trainval()
    fig_tracking()
    fig_and_table_results()
    print("figures + table written to assets/")


if __name__ == "__main__":
    main()
