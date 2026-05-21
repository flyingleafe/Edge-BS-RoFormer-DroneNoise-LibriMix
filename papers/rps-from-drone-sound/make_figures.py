#!/usr/bin/env python3
"""
Generate all figures for the paper from local results.

Outputs go to papers/rps-from-drone-sound/figures/.

Figures produced:
- fig_training_curves.pdf      — train/val MSE + R² vs epoch (SimpleConv)
- fig_qualitative_<id>.pdf     — spectrogram + ground-truth + predicted RPS for selected samples
- fig_qualitative_combined.pdf — three-panel composite of qualitative samples
- fig_highsnr_per_sample.pdf   — per-sample MSE on out-of-distribution high-SNR clips

Run from project root inside the nix dev shell:
    python papers/rps-from-drone-sound/make_figures.py
"""

import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf

# --- Plot style: clean, paper-friendly ----------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = Path(__file__).resolve().parent / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

ROTOR_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
ROTOR_LABELS = ["Rotor 1", "Rotor 2", "Rotor 3", "Rotor 4"]


# -----------------------------------------------------------------------------
# Fig. 1 — Training curves
# -----------------------------------------------------------------------------
def fig_training_curves() -> None:
    csv_path = ROOT / "results/rps_predictor/rps_predictor/training_log.csv"
    df = pd.read_csv(csv_path)

    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.4))

    # MSE — log scale, clip top so the catastrophic epoch 1 doesn't dominate
    ax = axes[0]
    ax.plot(df["epoch"], df["train_mse"], "-", lw=1.2, label="Train", color="#1f77b4")
    ax.plot(df["epoch"], df["val_mse"], "-", lw=1.2, label="Validation", color="#d62728")
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(r"MSE  $[(\mathrm{rev/s})^2]$")
    ax.set_title("(a) Training and validation MSE")
    ax.legend(frameon=False, loc="upper right")

    # R^2 — linear, mark best
    ax = axes[1]
    ax.plot(df["epoch"], df["r2"], "-", lw=1.2, color="#2ca02c")
    best = df.loc[df["val_mse"].idxmin()]
    ax.axhline(best["r2"], ls="--", lw=0.8, color="gray")
    ax.annotate(
        f"best $R^2={best['r2']:.3f}$ @ ep.{int(best['epoch'])}",
        xy=(best["epoch"], best["r2"]),
        xytext=(0.55, 0.18), textcoords="axes fraction",
        arrowprops=dict(arrowstyle="-", lw=0.6, color="gray"),
        fontsize=8,
    )
    ax.set_ylim(-0.2, 1.0)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(r"Validation $R^2$")
    ax.set_title("(b) Validation coefficient of determination")

    out = FIG_DIR / "fig_training_curves.pdf"
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print("wrote", out)


# -----------------------------------------------------------------------------
# Fig. 2 — Qualitative examples: spectrogram + GT/predicted RPS
# -----------------------------------------------------------------------------
def _load_sample(sample_id: str):
    base = ROOT / "results/rps_eval_specific_samples" / sample_id
    audio, sr = sf.read(base / "mixture.wav")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    gt_raw = np.load(base / "ground_truth_rps.npy")   # (4, T_gt)  native ~950 Hz
    pred   = np.load(base / "simple_conv_rps.npy")    # (4, T_pred) STFT grid
    # Linearly interpolate GT onto the prediction time grid so that
    # ground-truth and predicted traces are directly comparable.
    gt = np.zeros_like(pred)
    x_old = np.linspace(0, 1, gt_raw.shape[1])
    x_new = np.linspace(0, 1, pred.shape[1])
    for r in range(4):
        gt[r] = np.interp(x_new, x_old, gt_raw[r])
    return audio.astype(np.float32), sr, gt, pred


def fig_qualitative_examples(sample_ids=("sample_00000", "sample_00149", "sample_00599")) -> None:
    """Three-row composite: spectrogram on top, GT vs predicted RPS overlaid below."""
    n = len(sample_ids)
    fig, axes = plt.subplots(2, n, figsize=(7.1, 3.8),
                             gridspec_kw={"height_ratios": [1.0, 1.0], "hspace": 0.40, "wspace": 0.30})
    if n == 1:
        axes = np.array(axes).reshape(2, 1)

    for col, sid in enumerate(sample_ids):
        audio, sr, gt, pred = _load_sample(sid)
        duration = len(audio) / sr

        # Spectrogram
        ax = axes[0, col]
        n_fft, hop = 2048, 512
        spec = np.abs(np.fft.rfft(
            np.lib.stride_tricks.sliding_window_view(audio, n_fft)[::hop] *
            np.hanning(n_fft), axis=-1))
        log_mag = np.log1p(spec.T)
        ax.imshow(
            log_mag, origin="lower", aspect="auto",
            extent=[0, duration, 0, sr / 2 / 1000],
            cmap="magma",
        )
        ax.set_ylim(0, 4)  # focus on lowest 4 kHz where harmonics live
        ax.set_ylabel("freq [kHz]" if col == 0 else "")
        ax.set_title(f"sample {sid.split('_')[1]}")
        ax.set_xticklabels([])
        ax.grid(False)

        # RPS overlays
        ax = axes[1, col]
        t_gt = np.linspace(0, duration, gt.shape[1])
        t_pred = np.linspace(0, duration, pred.shape[1])
        for r in range(4):
            ax.plot(t_gt,   gt[r],   ":",  color=ROTOR_COLORS[r], lw=0.5, alpha=0.55,
                    label=f"GT R{r+1}" if col == 0 else None)
            ax.plot(t_pred, pred[r], "-",  color=ROTOR_COLORS[r], lw=0.5, alpha=0.75,
                    label=f"pred R{r+1}" if col == 0 else None)
        ax.set_xlabel("time [s]")
        if col == 0:
            ax.set_ylabel("rotor speed [rev/s]")
        ax.set_xlim(0, duration)

    # Compact legend below; reserve space so it does not overlap x-labels
    legend_handles = [
        plt.Line2D([0], [0], color="black", lw=0.5, ls=":",  alpha=0.55, label="ground truth"),
        plt.Line2D([0], [0], color="black", lw=0.5, ls="-",  alpha=0.75, label="predicted"),
    ] + [
        plt.Line2D([0], [0], color=ROTOR_COLORS[r], lw=2.0, label=ROTOR_LABELS[r]) for r in range(4)
    ]
    fig.subplots_adjust(bottom=0.20)
    fig.legend(handles=legend_handles, loc="lower center", ncol=6,
               bbox_to_anchor=(0.5, 0.0), frameon=False, fontsize=8)

    out = FIG_DIR / "fig_qualitative_combined.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


# -----------------------------------------------------------------------------
# Fig. 3 — High-SNR generalization: per-sample MSE
# -----------------------------------------------------------------------------
def fig_highsnr_per_sample() -> None:
    """Per-sample MSE on 10 out-of-distribution high-SNR free-flight clips.

    Reference horizontal line: held-out synthetic mixtures MSE = 5.15.
    The last sample (drone landing) is annotated as an outlier with
    very different acoustic conditions.
    """
    data = json.load(open(ROOT / "results/rps_high_snr_analysis.json"))
    samples = data["results"]
    times = [s["rel_time"] for s in samples]
    sc_mse = [s["simple_conv"]["mse"] for s in samples]
    dcu_mse = [s["dcunet"]["mse"] for s in samples]
    dcc_mse = [s["dccrn"]["mse"] for s in samples]

    fig, ax = plt.subplots(figsize=(6.8, 2.4))
    x = np.arange(len(samples))
    w = 0.27

    ax.bar(x - w, sc_mse, w, label="SimpleConv", color="#1f77b4")
    ax.bar(x, dcu_mse, w, label="DCUNet-enc", color="#ff7f0e")
    ax.bar(x + w, dcc_mse, w, label="DCCRN-enc", color="#2ca02c")

    # Reference line: synthetic-mixture MSE = 5.15
    ax.axhline(5.15, ls="--", lw=1.0, color="#444",
               label="SimpleConv held-out (synthetic), MSE = 5.15")

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{t:.0f}s" for t in times], rotation=0, fontsize=7)
    ax.set_xlabel("clip start time within DREGON free-flight $\\mathit{speech\\,high}$ room1")
    ax.set_ylabel(r"MSE  $[(\mathrm{rev/s})^2]$")
    ax.legend(frameon=False, loc="upper left", fontsize=7, ncol=2)

    # Annotate outlier
    outlier_idx = int(np.argmax(sc_mse))
    ax.annotate(
        "drone-landing regime\n(RPS$\\to$0)",
        xy=(outlier_idx, sc_mse[outlier_idx]),
        xytext=(outlier_idx - 2.8, sc_mse[outlier_idx] * 0.6),
        arrowprops=dict(arrowstyle="->", lw=0.7, color="gray"),
        fontsize=7, color="#555",
    )

    out = FIG_DIR / "fig_highsnr_per_sample.pdf"
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    fig_training_curves()
    fig_qualitative_examples()
    fig_highsnr_per_sample()
