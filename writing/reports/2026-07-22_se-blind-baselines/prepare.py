#!/usr/bin/env python3
"""Generate figures for the SE blind-baselines report from data/*.csv.

Data are the paper-matched Pass-A runs evaluated on SE-valid-drone at a uniform
25 mixtures/SNR-point subset (fair cross-model comparison), plus noisy/Wiener
anchors. See docs/experiments/f1-se-blind-baselines.md.
"""

from __future__ import annotations

import csv
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = pathlib.Path(__file__).parent
DATA = HERE / "data"
ASSETS = HERE / "assets"
SNRS = [-30, -25, -20, -15, -10, -5, 0]

# display name -> (csv, color, marker), in ranking order
MODELS = [
    ("MP-SENet", "mpsenet_a.csv", "#d62728", "o"),
    ("TF-GridNet", "tfgridnet_a.csv", "#1f77b4", "s"),
    ("Edge-BS-RoFormer", "edge_bs_rof_a.csv", "#2ca02c", "^"),
    ("DCUNet", "dcunet_a.csv", "#9467bd", "D"),
]


def _load(path: pathlib.Path) -> dict[float, dict[str, float]]:
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return {
        float(r["input_snr"]): {
            k: (float(r[k]) if r[k] not in ("", "nan") else float("nan"))
            for k in ("si_sdr", "sdr", "pesq", "estoi")
        }
        for r in rows
    }


def main() -> None:
    ASSETS.mkdir(exist_ok=True)
    with open(DATA / "anchors.csv") as f:
        an = list(csv.DictReader(f))
    noisy = {float(r["input_snr"]): float(r["si_sdr"]) for r in an if r["method"] == "noisy"}
    noisy_es = {float(r["input_snr"]): float(r["estoi"]) for r in an if r["method"] == "noisy"}
    models = {name: _load(DATA / fn) for name, fn, _c, _mk in MODELS}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.1))

    for name, _fn, color, mk in MODELS:
        d = models[name]
        ax1.plot(
            SNRS, [d[s]["si_sdr"] - noisy[s] for s in SNRS], marker=mk, color=color, label=name
        )
    ax1.axhline(0, color="grey", lw=0.8, ls="--")
    ax1.set_xlabel("input SNR (dB)")
    ax1.set_ylabel(r"$\Delta$ SI-SDR over noisy (dB)")
    ax1.set_title("(a) SI-SDR improvement")
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=8, loc="upper right")

    ax2.plot(SNRS, [noisy_es[s] for s in SNRS], marker="x", color="black", ls=":", label="noisy")
    for name, _fn, color, mk in MODELS:
        d = models[name]
        ax2.plot(SNRS, [d[s]["estoi"] for s in SNRS], marker=mk, color=color, label=name)
    ax2.set_xlabel("input SNR (dB)")
    ax2.set_ylabel("eSTOI")
    ax2.set_title("(b) Intelligibility (eSTOI)")
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=8, loc="upper left")

    fig.tight_layout()
    fig.savefig(ASSETS / "floor_drone.png", dpi=150, bbox_inches="tight")
    print("wrote", ASSETS / "floor_drone.png")


if __name__ == "__main__":
    main()
