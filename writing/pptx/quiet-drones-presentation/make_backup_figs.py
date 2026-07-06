#!/usr/bin/env python3
"""
Generate figures for the Quiet Drones Q&A / backup slides (after-submission work).

Produces four slide-ready PNGs in ./backup_figs:
  1. classical_mse_bar.png   - log-scale MSE: SimpleConv vs 5 classical methods
  2. classical_rps_overlay.png - one DREGON-LM clip: SimpleConv vs GT, NMF vs GT
  3. arch_scatter.png        - params vs validation R^2 for the 10 SimpleConv variants
  4. arch_fullseq_compare.png - 47 s free-flight: baseline vs BiGRU-v2 vs GT

Run from project root:
    python writing/pptx/quiet-drones-presentation/make_backup_figs.py
"""

import sys
from pathlib import Path
from typing import cast

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import torchaudio
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = Path(__file__).resolve().parent / "backup_figs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT))


import tdseries as td  # noqa: E402

from data_processing.frames import get_meta  # noqa: E402
from tasks.rps_prediction import load_input_set, load_predictor  # noqa: E402
from utils.paths import get_data_path, get_datasets_path, get_results_path  # noqa: E402

ROTOR_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
ROTOR_LABELS = ["Rotor 1", "Rotor 2", "Rotor 3", "Rotor 4"]
NAVY = "#333399"
TEAL = "#007A7A"
GRAY = "#9aa0a6"

SC_CKPT = str(get_results_path("rps_exp_simple_conv/best_simple_conv.pt"))
BIGRU_V2_CKPT = str(get_results_path("rps_exp_simple_conv_bigru_v2/best_simple_conv_bigru_v2.pt"))
DREGON_VALID = get_datasets_path("DREGON-LM/valid")
DREGON_DIR = get_data_path("DREGON")
RECORDING = "DREGON_free-flight_speech-high_room1"
SR = 16000
N_FFT = 2048
HOP = 512

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 15,
        "axes.titlesize": 17,
        "axes.labelsize": 15,
        "legend.fontsize": 13,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.6,
        "figure.dpi": 150,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.04,
    }
)


# ── 1. Classical-baseline MSE bar (from the classical-baselines report) ──────
def fig_classical_mse_bar():
    # Mean MSE on DREGON-LM test set (metrics.json, classical_baselines_report)
    data = [
        ("SimpleConv\n(learned CNN)", 8.44, NAVY),
        ("NMF", 617.17, GRAY),
        ("PYIN", 630.75, GRAY),
        ("HPS", 868.48, GRAY),
        ("Matched filter", 1343.50, GRAY),
        ("Cepstral", 1383.49, GRAY),
    ]
    labels = [d[0] for d in data]
    vals = [d[1] for d in data]
    colors = [d[2] for d in data]
    y = np.arange(len(data))[::-1]

    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    ax.barh(y, vals, color=colors, edgecolor="white", height=0.68)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xscale("log")
    ax.set_xlabel(r"mean MSE  [(rev/s)$^2$]  —  log scale")
    ax.set_xlim(4, 3000)
    ax.grid(axis="y", alpha=0)
    for yi, v in zip(y, vals):
        ax.text(v * 1.12, yi, f"{v:.0f}" if v >= 100 else f"{v:.1f}", va="center", fontsize=13)
    ax.set_title("Classical pitch trackers fail by 70–160×", color=NAVY, fontweight="bold")
    out = OUT_DIR / "classical_mse_bar.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print("wrote", out)


# ── 2. RPS overlay: SimpleConv vs GT (top), best classical NMF vs GT (bottom) ─
def fig_classical_rps_overlay(sid: str = "sample_00000"):
    samples = {get_meta(s, "id", ""): s for s in load_input_set(str(DREGON_VALID))}
    sample = samples[sid]
    audio_us = cast(td.Series, sample["audio"])
    audio = np.asarray(audio_us.data, dtype=np.float32)
    sr = cast(td.GridIndex, audio_us.tindex).sr
    dur = len(audio) / sr
    n_frames = len(audio) // HOP + 1
    frame_times = np.arange(n_frames) * HOP / sr
    gt = sample["rps"].interpolate(frame_times)  # (4, F)

    sc = load_predictor(f"simple_conv@{SC_CKPT}").predict(audio, sr=sr)
    nmf = load_predictor("nmf").predict(audio, sr=sr)

    fig, axes = plt.subplots(2, 1, figsize=(10.5, 6.0), gridspec_kw={"hspace": 0.32})
    for ax, pred, name in (
        (axes[0], sc, "SimpleConv (learned)"),
        (axes[1], nmf, "NMF (best classical)"),
    ):
        T = min(pred.shape[1], n_frames)
        for r in range(4):
            ax.plot(frame_times[:T], gt[r, :T], ":", color=ROTOR_COLORS[r], lw=1.4, alpha=0.65)
            ax.plot(frame_times[:T], pred[r, :T], "-", color=ROTOR_COLORS[r], lw=1.6, alpha=0.9)
        ax.set_xlim(0, dur)
        ax.set_ylabel("rotor speed [rev/s]")
        ax.set_title(
            name, fontsize=15, fontweight="bold", color=NAVY if "learned" in name else "#888"
        )
    axes[0].set_ylim(40, 100)
    axes[1].set_xlabel("time [s]")

    legend_handles = [
        Line2D([0], [0], color="black", lw=1.6, ls=":", alpha=0.65, label="ground truth"),
        Line2D([0], [0], color="black", lw=1.6, ls="-", alpha=0.9, label="prediction"),
    ] + [Line2D([0], [0], color=ROTOR_COLORS[r], lw=3.0, label=ROTOR_LABELS[r]) for r in range(4)]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=6,
        bbox_to_anchor=(0.5, -0.04),
        frameon=False,
    )
    out = OUT_DIR / "classical_rps_overlay.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print("wrote", out)


# ── 3. Params vs validation R^2 scatter for the 10 SimpleConv variants ───────
def fig_arch_scatter():
    # (name, params[M], R2, dx, dy, ha)  — offsets to avoid label overlap
    variants = [
        ("v2 (SE+Attn)", 1.50, 0.951, 0.06, 0.006, "left"),
        ("BiGRU-v2 (best)", 1.44, 0.948, 0.28, 0.0, "left"),
        ("BiGRU", 0.67, 0.945, 0.07, 0.0, "left"),
        ("TCN", 1.38, 0.936, 0.06, -0.012, "left"),
        ("MagPhase", 0.67, 0.917, 0.07, 0.0, "left"),
        ("AttnPool", 0.56, 0.860, 0.07, 0.0, "left"),
        ("Wide", 3.94, 0.847, -0.10, 0.004, "right"),
        ("MultiScale", 1.36, 0.840, 0.07, 0.0, "left"),
        ("Baseline", 0.54, 0.837, 0.07, -0.004, "left"),
        ("SE-Next", 1.41, 0.688, 0.07, 0.0, "left"),
    ]
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    for name, p, r2, dx, dy, ha in variants:
        if name == "Baseline":
            c, s, z = "#d62728", 180, 5
        elif "BiGRU-v2" in name:
            c, s, z = NAVY, 200, 6
        else:
            c, s, z = "#8a8fbf", 110, 3
        ax.scatter(p, r2, s=s, c=c, edgecolor="white", linewidth=1.2, zorder=z)
        ax.annotate(
            name,
            (p, r2),
            xytext=(p + dx, r2 + dy),
            ha=ha,
            va="center",
            fontsize=12,
            fontweight="bold" if c in (NAVY, "#d62728") else "normal",
            color=c if c in (NAVY, "#d62728") else "#444",
        )
    ax.set_xlabel("parameters [M]")
    ax.set_ylabel(r"validation $R^2$")
    ax.set_xlim(0.2, 4.6)
    ax.set_ylim(0.66, 0.97)
    ax.annotate(
        "+ BiGRU temporal head\n$R^2$: 0.84 → 0.95",
        xy=(0.62, 0.945),
        xytext=(1.9, 0.80),
        arrowprops=dict(arrowstyle="->", lw=1.4, color=NAVY),
        fontsize=13,
        color=NAVY,
        fontweight="bold",
    )
    ax.set_title(
        "10 architecture variants — temporal head is the big win", color=NAVY, fontweight="bold"
    )
    out = OUT_DIR / "arch_scatter.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print("wrote", out)


# ── 4. Full-sequence mean-RPS: baseline vs BiGRU-v2 vs GT ─────────────────────
def _load_full_sequence():
    rec_dir = DREGON_DIR / RECORDING
    audio_full, file_sr = torchaudio.load(str(rec_dir / f"{RECORDING}.wav"))
    audio_ts = scipy.io.loadmat(rec_dir / f"{RECORDING}_audiots.mat")["audio_timestamps"].flatten()
    motor_data = scipy.io.loadmat(rec_dir / f"{RECORDING}_motors.mat")["motor"][0, 0]
    measured = motor_data["measured"]
    motor_ts = motor_data["timestamps"].flatten()
    motor_sr_val = len(motor_ts) / (motor_ts[-1] - motor_ts[0])
    t0, t1 = motor_ts[0], motor_ts[-1]
    audio_crop = audio_full[
        :, int((t0 - audio_ts[0]) * file_sr) : int((t1 - audio_ts[0]) * file_sr)
    ]
    audio_mono = audio_crop.mean(dim=0, keepdim=True)
    if file_sr != SR:
        audio_mono = torchaudio.transforms.Resample(file_sr, SR)(audio_mono)
    peak = audio_mono.abs().max()
    if peak > 0:
        audio_mono = audio_mono / peak * 0.9
    audio = audio_mono.squeeze().numpy()
    rps_motor = measured.T.astype(np.float32)
    n_frames = len(audio) // HOP + 1
    stft_times = np.arange(n_frames) * HOP / SR
    motor_times = np.arange(rps_motor.shape[1]) / motor_sr_val
    gt = np.array([np.interp(stft_times, motor_times, rps_motor[r]) for r in range(4)])
    return audio, gt, stft_times


def fig_arch_fullseq_compare():
    audio, gt, t = _load_full_sequence()
    base = load_predictor(f"simple_conv@{SC_CKPT}").predict(audio, sr=SR)
    best = load_predictor(f"simple_conv_bigru_v2@{BIGRU_V2_CKPT}").predict(audio, sr=SR)
    T = min(gt.shape[1], base.shape[1], best.shape[1], len(t))
    gt_m = gt[:, :T].mean(0)
    base_m = base[:, :T].mean(0)
    best_m = best[:, :T].mean(0)
    tt = t[:T]

    fig, ax = plt.subplots(figsize=(11.0, 4.8))
    ax.plot(tt, gt_m, color="black", ls=":", lw=2.0, alpha=0.8, label="ground truth")
    ax.plot(
        tt,
        base_m,
        color="#d62728",
        lw=1.8,
        alpha=0.85,
        label="Baseline SimpleConv (in-flight MSE 24.4)",
    )
    ax.plot(tt, best_m, color=NAVY, lw=1.8, alpha=0.9, label="BiGRU-v2 (in-flight MSE 9.9)")
    ax.set_xlim(0, tt[-1])
    ax.set_xlabel("time [s]")
    ax.set_ylabel("mean rotor speed [rev/s]")
    ax.legend(loc="lower center", frameon=False, ncol=1, fontsize=12)
    ax.set_title(
        "Real 47 s free flight — best variant halves the in-flight error",
        color=NAVY,
        fontweight="bold",
    )
    out = OUT_DIR / "arch_fullseq_compare.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    fig_classical_mse_bar()
    fig_arch_scatter()
    fig_classical_rps_overlay()
    fig_arch_fullseq_compare()
