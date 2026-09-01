#!/usr/bin/env python3
"""Slide 1's figure: the frequency-scaling probe, WITHOUT the phase-increment
readout curve.

This is `writing/papers/2026-08_wrapup/plot_freq_probe.py` (the wrap-up paper's
Fig. 3) with the "Kalman attention, R2 (hb_ckla)" series dropped, because the
supervisor deck argues only about augmentation and comb pre-training. The
response data are the paper's own, read from
`writing/papers/2026-08_wrapup/figures/freq_probe_full.json`.

The asset is checked in, so this script exists to REPRODUCE it, not to run on
every build. It refuses to overwrite an existing file unless called with
``--force``.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PAPER_FIGS = ROOT / "writing" / "papers" / "2026-08_wrapup" / "figures"
OUT = HERE / "assets" / "freq_probe_nophase.pdf"

CURVES = [
    ("no label-transforming augmentations", "#8c8c8c", "o", "-",
     "no label-transforming augmentation"),
    ("full-envelope real regime (R2)", "#c0392b", "s", "-",
     "augmentation only (R2)"),
    ("comb curriculum (R4)", "#1f5fa9", "^", "-",
     "augmentation + comb pre-training (R4)"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if OUT.exists() and not args.force:
        print(f"{OUT.name} exists; pass --force to rebuild")
        return

    a = json.loads((PAPER_FIGS / "freq_probe_full.json").read_text())
    x = 100.0 * (np.array(a["alphas"]) - 1.0)
    R = a["response"]

    fig, ax = plt.subplots(figsize=(3.45, 3.35))
    ax.plot(x, x, color="black", lw=1.0, ls=(0, (4, 3)), zorder=1,
            label="ideal response (1.00)")
    near = np.abs(x) <= 4.001
    for key, color, marker, ls, nice in CURVES:
        y = np.array(R[key])
        local = float((x[near] * y[near]).sum() / (x[near] * x[near]).sum())
        full = float((x * y).sum() / (x * x).sum())
        ax.plot(x, y, color=color, marker=marker, ms=3.2, lw=1.5, ls=ls,
                label=f"{nice} ({local:.2f})", zorder=3, clip_on=True)
        print(f"{key:42s} full {full:5.2f}  local {local:5.2f}")

    ax.text(-30.5, -38.5, "off scale: $-66$", fontsize=6.8, color="#c0392b",
            ha="left", va="bottom")
    ax.set_xlim(-32, 32)
    ax.set_ylim(-40.0, 36.0)
    ax.set_xlabel("frequency scaling of the input (%)", fontsize=8)
    ax.set_ylabel("change in predicted speed (%)", fontsize=8)
    ax.tick_params(labelsize=7.5)
    ax.axhline(0, color="#cccccc", lw=0.6, zorder=0)
    ax.axvline(0, color="#cccccc", lw=0.6, zorder=0)
    ax.grid(alpha=0.18, lw=0.5)
    leg = ax.legend(fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.20),
                    frameon=False, borderpad=0.2, handlelength=1.9,
                    labelspacing=0.3, title="slope near $\\alpha=1$ in brackets",
                    title_fontsize=6.5)
    leg._legend_box.align = "left"  # type: ignore[attr-defined]
    fig.tight_layout(pad=0.4)
    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
