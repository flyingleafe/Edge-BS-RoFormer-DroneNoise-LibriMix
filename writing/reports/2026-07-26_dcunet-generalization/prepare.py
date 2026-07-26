#!/usr/bin/env python3
"""Figures and tables for the DCUNet-generalization report.

Reads the per-clip eval CSVs written by ``scripts/eval_se_perclip.py`` (one row
per validation clip, with ``category``/``input_snr`` metadata) straight out of
``results/``, so every number in the report traces to a clip-level measurement
rather than a hand-copied summary.

Silent-reference clips are dropped: the F1 valid sets were built before the
silent-draw guard landed in ``src/data_processing/online_mixing.py``, and an
all-zero reference pins SI-SDR at the -80 dB floor for every method at once,
dragging a whole SNR group's mean by several dB.
"""

from __future__ import annotations

import pathlib
import re
from collections import Counter

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = pathlib.Path(__file__).parent
ASSETS = HERE / "assets"
ROOT = pathlib.Path(__file__).resolve().parents[3]
F1 = ROOT / "results" / "f1_perclip"
F2 = ROOT / "results" / "f2_perclip"

SEEN = "#1f77b4"
UNSEEN = "#d62728"
CTRL = "#7f7f7f"
MPSE = "#2ca02c"
DCU = "#9467bd"

plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 160})


def load(name: str, where: pathlib.Path = F2) -> pd.DataFrame:
    d = pd.read_csv(where / name)
    keep = d.groupby("clip_id")["si_sdr"].transform("max") > -70
    return pd.DataFrame(d[keep])


def bysnr(d: pd.DataFrame, cat: str | None = None) -> pd.DataFrame:
    if cat is not None:
        d = pd.DataFrame(d[d["category"] == cat])
    cols = [c for c in ("si_sdr", "estoi", "pesq", "pesq_nb", "gain_db", "corr") if c in d.columns]
    return pd.DataFrame(d.groupby("input_snr")[cols].mean()).sort_index()


# ── Figure 1: the seen/unseen experiment ────────────────────────────────────
def fig_seen_unseen() -> None:
    probe = load("f2_dcunet_avq_heldout__SE-valid-avq-split.csv")
    ctrl = load("f2_dcunet_avq_survey__SE-valid-avq-split.csv")
    noisy = load("noisy__SE-valid-avq-split.csv")

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9))
    for ax, metric, label in zip(
        axes, ("si_sdr", "estoi"), ("output SI-SDR (dB)", "eSTOI"), strict=True
    ):
        for cat, colour, style, name in [
            ("avq_ego_s1", SEEN, "-", "session 1 — SEEN in training"),
            ("avq_ego_s2", UNSEEN, "-", "session 2 — NEVER seen"),
        ]:
            g = bysnr(probe, cat)
            ax.plot(g.index, g[metric], style, color=colour, marker="o", lw=2, label=name)
        for cat, style, name in [
            ("avq_ego_s1", "--", "control (trained on all 5): session 1"),
            ("avq_ego_s2", ":", "control (trained on all 5): session 2"),
        ]:
            g = bysnr(ctrl, cat)
            ax.plot(g.index, g[metric], style, color=CTRL, marker="s", ms=4, lw=1.4, label=name)
        g = bysnr(noisy, "avq_ego_s2")
        ax.plot(g.index, g[metric], "-", color="black", lw=1, alpha=0.6, label="unprocessed input")
        ax.set_xlabel("input SNR (dB)")
        ax.set_ylabel(label)
    axes[0].legend(fontsize=7, loc="upper left")
    axes[0].set_title("(a) SI-SDR")
    axes[1].set_title("(b) intelligibility")
    fig.suptitle("Holding out 2 of 5 ego-noise recordings of the SAME drone", fontsize=11, y=1.005)
    fig.tight_layout()
    fig.savefig(ASSETS / "seen_unseen.png", bbox_inches="tight")
    plt.close(fig)


# ── Figure 2: the ladder — dilution of the in-domain recording ──────────────
def fig_ladder() -> None:
    noisy = bysnr(load("noisy__SE-valid-avq-survey.csv"))
    arms = [
        ("AVQ only\n(100%)", "f2_dcunet_avq_survey__SE-valid-avq-survey.csv", "#1f77b4"),
        ("+ all drone\n(~14%)", "f2_dcunet_alldrone__SE-valid-avq-survey.csv", "#ff7f0e"),
        ("+ all harmonic\n(~2%)", "f2_dcunet_allharmonic__SE-valid-avq-survey.csv", "#d62728"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.7))
    for label, csv_name, colour in arms:
        g = bysnr(load(csv_name))
        axes[0].plot(
            g.index, g["estoi"] - noisy["estoi"], marker="o", color=colour, lw=2, label=label
        )
        axes[1].plot(
            g.index, g["si_sdr"] - noisy["si_sdr"], marker="o", color=colour, lw=2, label=label
        )
    axes[0].axhline(0, color="black", lw=1)
    axes[0].set_ylabel("ΔeSTOI vs unprocessed")
    axes[0].set_title("(a) intelligibility gain collapses")
    axes[1].set_ylabel("ΔSI-SDR vs unprocessed (dB)")
    axes[1].set_title("(b) …while the energy gain survives")
    for ax in axes:
        ax.set_xlabel("input SNR (dB)")
        ax.legend(fontsize=8, title="AVQ share of training", title_fontsize=8)
    fig.suptitle(
        "Same model, same loss, same validation clips — only the training noise pool changes",
        fontsize=10,
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(ASSETS / "ladder.png", bbox_inches="tight")
    plt.close(fig)


# ── Figure 3: the control — MP-SENet on a drone it never heard ──────────────
def fig_control() -> None:
    noisy = bysnr(load("noisy__SE-valid-avq-survey.csv"))
    mp = bysnr(load("f1_mpsenet_a__SE-valid-avq-survey.csv"))
    probe_unseen = bysnr(load("f2_dcunet_avq_heldout__SE-valid-avq-split.csv"), "avq_ego_s2")
    noisy_s2 = bysnr(load("noisy__SE-valid-avq-split.csv"), "avq_ego_s2")

    fig, ax = plt.subplots(figsize=(5.6, 3.9))
    ax.plot(
        mp.index,
        mp["estoi"] - noisy["estoi"],
        marker="o",
        color=MPSE,
        lw=2,
        label="MP-SENet — never heard this drone",
    )
    ax.plot(
        probe_unseen.index,
        probe_unseen["estoi"] - noisy_s2["estoi"],
        marker="D",
        color=DCU,
        lw=2,
        label="DCUNet — same drone, unseen session",
    )
    ax.axhline(0, color="black", lw=1)
    ax.set_xlabel("input SNR (dB)")
    ax.set_ylabel("ΔeSTOI vs unprocessed")
    ax.set_title("Generalization to unseen rotor noise")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(ASSETS / "control.png", bbox_inches="tight")
    plt.close(fig)


# ── Figure 4: energy vs intelligibility, all conditions ─────────────────────
def fig_split() -> None:
    pts = []
    n_avq = bysnr(load("noisy__SE-valid-avq-survey.csv"))
    n_split = bysnr(load("noisy__SE-valid-avq-split.csv"), "avq_ego_s2")
    n_drone = bysnr(load("noisy__SE-valid-drone.csv"))
    n_harm = bysnr(load("noisy__SE-valid-harmonic.csv"))
    at = -15.0
    for name, csv_name, cat, anchor, colour, marker, off in [
        (
            "DCUNet — seen noise (survey protocol)",
            "f2_dcunet_avq_survey__SE-valid-avq-survey.csv",
            None,
            n_avq,
            SEEN,
            "o",
            (-152, -20),
        ),
        (
            "DCUNet — unseen session",
            "f2_dcunet_avq_heldout__SE-valid-avq-split.csv",
            "avq_ego_s2",
            n_split,
            UNSEEN,
            "D",
            (12, 10),
        ),
        (
            "DCUNet — broad drone pool",
            "f2_dcunet_alldrone__SE-valid-drone.csv",
            None,
            n_drone,
            "#ff7f0e",
            "s",
            (12, -16),
        ),
        (
            "DCUNet — broad harmonic pool",
            "f2_dcunet_allharmonic__SE-valid-harmonic.csv",
            None,
            n_harm,
            "#8c564b",
            "v",
            (-116, -30),
        ),
        (
            "MP-SENet — unseen drone",
            "f1_mpsenet_a__SE-valid-avq-survey.csv",
            None,
            n_avq,
            MPSE,
            "*",
            (-142, 12),
        ),
    ]:
        g = bysnr(load(csv_name), cat)
        pts.append(
            (
                name,
                g.loc[at, "si_sdr"] - anchor.loc[at, "si_sdr"],
                g.loc[at, "estoi"] - anchor.loc[at, "estoi"],
                colour,
                marker,
                off,
            )
        )

    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    ax.axhspan(-0.10, 0.06, color="#d62728", alpha=0.06, zorder=0)
    ax.text(0.4, 0.028, "no intelligibility gain", fontsize=7.5, color="#d62728", style="italic")
    for name, dsi, des, colour, marker, off in pts:
        ax.scatter(dsi, des, s=150 if marker == "*" else 90, c=colour, marker=marker, zorder=3)
        ax.annotate(name, (dsi, des), textcoords="offset points", xytext=off, fontsize=7.5)
    ax.axhline(0, color="black", lw=1)
    ax.set_xlabel("ΔSI-SDR vs unprocessed (dB)  —  noise ENERGY removed")
    ax.set_ylabel("ΔeSTOI vs unprocessed  —  SPEECH recovered")
    ax.set_title("At −15 dB input SNR: the two axes come apart")
    ax.set_xlim(-1, 22)
    ax.set_ylim(-0.10, 0.44)
    fig.tight_layout()
    fig.savefig(ASSETS / "energy_vs_intelligibility.png", bbox_inches="tight")
    plt.close(fig)


# ── Table: DN-LM leakage arithmetic ─────────────────────────────────────────
def table_leakage() -> None:
    d = ROOT / "data" / "drone_audio" / "Binary_Drone_Audio" / "yes_drone"
    files = sorted(p.name for p in d.glob("*.wav")) if d.is_dir() else []
    n = len(files) or 1332
    recordings = len(Counter(re.sub(r"_\d+_?\.wav$", "", f) for f in files)) if files else 257
    speech = 27901
    p_clip_seen = 1 - (1 - 1 / n) ** 6480
    p_utt_seen = 1 - (1 - 1 / speech) ** 6480
    rows = [
        ("distinct noise clips in the pool", f"{n}"),
        ("distinct underlying drone recordings", f"{recordings}"),
        ("training draws / validation draws", "6480 / 720"),
        (
            "validation clips whose NOISE clip is also in train",
            f"{720 * p_clip_seen:.0f} / 720  ({100 * p_clip_seen:.1f}%)",
        ),
        (
            "validation clips reusing an EXACT train utterance",
            f"{720 * p_utt_seen:.0f} / 720  ({100 * p_utt_seen:.1f}%)",
        ),
        ("speaker overlap between train and valid", "~100% (no speaker split)"),
    ]
    with open(ASSETS / "leakage.csv", "w") as f:
        f.write("quantity,value\n")
        for k, v in rows:
            f.write(f'"{k}","{v}"\n')


def main() -> None:
    ASSETS.mkdir(exist_ok=True)
    fig_seen_unseen()
    fig_ladder()
    fig_control()
    fig_split()
    table_leakage()
    print("figures + tables written to", ASSETS)


if __name__ == "__main__":
    main()
