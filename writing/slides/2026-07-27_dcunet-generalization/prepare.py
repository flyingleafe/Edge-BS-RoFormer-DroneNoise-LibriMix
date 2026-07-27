#!/usr/bin/env python3
"""Slide figures for the DCUNet-generalization deck.

Same per-clip CSVs as the report
(``writing/reports/2026-07-26_dcunet-generalization``), re-plotted for
projection: larger type, fewer series per panel, one idea per figure.
"""

from __future__ import annotations

import pathlib

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = pathlib.Path(__file__).parent
ASSETS = HERE / "assets"
ROOT = pathlib.Path(__file__).resolve().parents[3]
F2 = ROOT / "results" / "f2_perclip"

SEEN = "#1f77b4"
UNSEEN = "#d62728"
CTRL = "#7f7f7f"
MPSE = "#2ca02c"

plt.rcParams.update(
    {
        "font.size": 15,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.labelsize": 15,
        "axes.titlesize": 16,
        "legend.fontsize": 12,
        "figure.dpi": 170,
    }
)


def load(name: str) -> pd.DataFrame:
    d = pd.read_csv(F2 / name)
    return pd.DataFrame(d[d.groupby("clip_id")["si_sdr"].transform("max") > -70])


def bysnr(d: pd.DataFrame, cat: str | None = None) -> pd.DataFrame:
    if cat is not None:
        d = pd.DataFrame(d[d["category"] == cat])
    cols = [c for c in ("si_sdr", "estoi", "pesq", "pesq_nb", "gain_db", "corr") if c in d.columns]
    return pd.DataFrame(d.groupby("input_snr")[cols].mean()).sort_index()


def fig_seen_unseen() -> None:
    """The money slide: one model, two halves, only training exposure differs."""
    probe = load("f2_dcunet_avq_heldout__SE-valid-avq-split.csv")
    noisy = load("noisy__SE-valid-avq-split.csv")

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.3))
    for ax, metric, ylab in zip(
        axes, ("si_sdr", "estoi"), ("output SI-SDR (dB)", "eSTOI"), strict=True
    ):
        for cat, colour, name in [
            ("avq_ego_s1", SEEN, "noise it TRAINED on"),
            ("avq_ego_s2", UNSEEN, "noise it NEVER heard"),
        ]:
            g = bysnr(probe, cat)
            ax.plot(g.index, g[metric], color=colour, marker="o", ms=8, lw=3, label=name)
        g = bysnr(noisy, "avq_ego_s2")
        ax.plot(g.index, g[metric], color="black", lw=1.6, alpha=0.6, ls="--", label="do nothing")
        ax.set_xlabel("input SNR (dB)")
        ax.set_ylabel(ylab)
    axes[0].legend(loc="lower right", framealpha=0.95)
    axes[0].annotate(
        "12.9 dB",
        xy=(-14.3, -1.6),
        ha="left",
        fontsize=17,
        color=UNSEEN,
        weight="bold",
    )
    axes[0].annotate(
        "", xy=(-15, 3.6), xytext=(-15, -9.3), arrowprops=dict(arrowstyle="<->", color=UNSEEN, lw=2)
    )
    fig.tight_layout()
    fig.savefig(ASSETS / "seen_unseen.png", bbox_inches="tight")
    plt.close(fig)


def fig_control() -> None:
    """Same picture for the model that trained on ALL five recordings."""
    ctrl = load("f2_dcunet_avq_survey__SE-valid-avq-split.csv")
    probe = load("f2_dcunet_avq_heldout__SE-valid-avq-split.csv")

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.3), sharey=True)
    for ax, d, title in zip(
        axes,
        (ctrl, probe),
        ("trained on ALL 5 recordings", "trained on session 1 only"),
        strict=True,
    ):
        for cat, colour, name in [
            ("avq_ego_s1", SEEN, "session 1"),
            ("avq_ego_s2", UNSEEN, "session 2"),
        ]:
            g = bysnr(d, cat)
            ax.plot(g.index, g["si_sdr"], color=colour, marker="o", ms=8, lw=3, label=name)
        ax.set_title(title)
        ax.set_xlabel("input SNR (dB)")
    axes[0].set_ylabel("output SI-SDR (dB)")
    axes[0].legend(loc="upper left")
    for ax, label, colour in (
        (axes[0], "gap 0.3 dB", CTRL),
        (axes[1], "gap 12.9 dB", UNSEEN),
    ):
        ax.text(
            0.5,
            0.05,
            label,
            transform=ax.transAxes,
            ha="center",
            color=colour,
            fontsize=15,
            weight="bold",
        )
    fig.tight_layout()
    fig.savefig(ASSETS / "control.png", bbox_inches="tight")
    plt.close(fig)


def fig_ladder() -> None:
    """Widening the training pool destroys intelligibility, not energy."""
    noisy = bysnr(load("noisy__SE-valid-avq-survey.csv"))
    arms = [
        ("AVQ only (100%)", "f2_dcunet_avq_survey__SE-valid-avq-survey.csv", "#1f77b4"),
        ("+ all drone (14%)", "f2_dcunet_alldrone__SE-valid-avq-survey.csv", "#ff7f0e"),
        ("+ all harmonic (2%)", "f2_dcunet_allharmonic__SE-valid-avq-survey.csv", "#d62728"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.3))
    for label, csv_name, colour in arms:
        g = bysnr(load(csv_name))
        axes[0].plot(
            g.index, g["estoi"] - noisy["estoi"], marker="o", ms=8, color=colour, lw=3, label=label
        )
        axes[1].plot(
            g.index,
            g["si_sdr"] - noisy["si_sdr"],
            marker="o",
            ms=8,
            color=colour,
            lw=3,
            label=label,
        )
    axes[0].axhline(0, color="black", lw=1.5)
    axes[0].set_ylabel("ΔeSTOI  (speech recovered)")
    axes[0].set_title("intelligibility gain collapses")
    axes[1].set_ylabel("ΔSI-SDR (dB)  (energy removed)")
    axes[1].set_title("…energy gain survives")
    for ax in axes:
        ax.set_xlabel("input SNR (dB)")
    axes[0].legend(title="AVQ share of training", fontsize=11, title_fontsize=11)
    fig.tight_layout()
    fig.savefig(ASSETS / "ladder.png", bbox_inches="tight")
    plt.close(fig)


def fig_mpsenet() -> None:
    """The architecture control, on unseen noise only."""
    noisy = bysnr(load("noisy__SE-valid-avq-survey.csv"))
    noisy_s2 = bysnr(load("noisy__SE-valid-avq-split.csv"), "avq_ego_s2")
    mp = bysnr(load("f1_mpsenet_a__SE-valid-avq-survey.csv"))
    probe = bysnr(load("f2_dcunet_avq_heldout__SE-valid-avq-split.csv"), "avq_ego_s2")

    fig, ax = plt.subplots(figsize=(7.6, 4.3))
    ax.plot(
        mp.index,
        mp["estoi"] - noisy["estoi"],
        marker="*",
        ms=16,
        color=MPSE,
        lw=3,
        label="MP-SENet — never heard this drone",
    )
    ax.plot(
        probe.index,
        probe["estoi"] - noisy_s2["estoi"],
        marker="D",
        ms=9,
        color=UNSEEN,
        lw=3,
        label="DCUNet — same drone, unseen session",
    )
    ax.axhline(0, color="black", lw=1.5)
    ax.set_xlabel("input SNR (dB)")
    ax.set_ylabel("ΔeSTOI vs doing nothing")
    ax.legend(loc="upper left", fontsize=12)
    fig.tight_layout()
    fig.savefig(ASSETS / "mpsenet.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ASSETS.mkdir(exist_ok=True)
    fig_seen_unseen()
    fig_control()
    fig_ladder()
    fig_mpsenet()
    print("slide figures written to", ASSETS)


if __name__ == "__main__":
    main()
