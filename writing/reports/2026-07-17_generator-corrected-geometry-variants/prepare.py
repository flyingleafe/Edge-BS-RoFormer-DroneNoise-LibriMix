#!/usr/bin/env python3
"""Figures/tables for the corrected-geometry generator-variants report.

The heavy artefacts (the per-variant MSSTFT numbers in ``assets/
msstft_comparison.csv`` and the real-vs-generated spectrogram grid
``assets/spectrograms.png``) are produced by the reusable evaluator

    python scripts/eval_noise_gen_variants.py \
        --variants old_wronggeom v1_corrected v2_perrotor v3_wind \
        --val-samples 128 --out <this dir>/assets

(needs R2 creds in .env; loads the four checkpoints from R2 and scores them on
the corrected-geometry swapped valid set). This script only turns the committed
CSV into a comparison bar chart — no GPU, no network.
"""

from __future__ import annotations

import csv
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = pathlib.Path(__file__).parent
ASSETS = HERE / "assets"

# Display order + labels (OLD baseline first, then the three new variants).
ORDER = ["old_wronggeom", "v1_corrected", "v2_perrotor", "v3_wind"]
LABELS = {
    "old_wronggeom": "OLD\n(wrong geom)",
    "v1_corrected": "v1\ncorrected",
    "v2_perrotor": "v2\n+per-rotor",
    "v3_wind": "v3\n+wind",
}
# OLD is the baseline (grey); the three new variants share the accent family,
# with the winner (v2) highlighted.
COLORS = {
    "old_wronggeom": "#9aa0a6",
    "v1_corrected": "#5b8fb9",
    "v2_perrotor": "#2b7a52",
    "v3_wind": "#a86413",
}


def _read_csv() -> dict[str, dict[str, float]]:
    rows: dict[str, dict[str, float]] = {}
    with open(ASSETS / "msstft_comparison.csv") as f:
        for r in csv.DictReader(f):
            rows[r["variant"]] = {
                "mrstft": float(r["mrstft"]),
                "n_flight": float(r.get("n_flight", 0)),
            }
    return rows


def main() -> None:
    ASSETS.mkdir(exist_ok=True)
    data = _read_csv()
    names = [n for n in ORDER if n in data]
    labels = [LABELS[n] for n in names]
    colors = [COLORS[n] for n in names]
    x = range(len(names))
    n_flight = int(next(iter(data.values()))["n_flight"]) if data else 0

    fig, ax = plt.subplots(1, 1, figsize=(5.2, 3.8))
    mr = [data[n]["mrstft"] for n in names]
    ax.bar(x, mr, color=colors, width=0.62)
    ax.set_title("MR-STFT quality on free-flight clips (higher = better)", fontsize=10)
    ax.set_ylabel("mrstft  [0–100 rescaled]")
    lo, hi = min(mr), max(mr)
    ax.set_ylim(lo - 0.6, hi + 0.4)
    for xi, v in zip(x, mr):
        ax.text(xi, v + 0.03, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        f"Generator variants — free-flight subset (RPS≥45, n={n_flight}/variant)", fontsize=11
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = ASSETS / "msstft_bars.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
