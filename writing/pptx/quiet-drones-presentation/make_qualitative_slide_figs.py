#!/usr/bin/env python3
"""
Generate per-sample qualitative figures for the Quiet Drones slide deck.

Splits the paper's 3-in-one `fig_qualitative_combined` into three larger,
standalone figures (one sample per slide): a log-STFT spectrogram on top and a
ground-truth vs. predicted rotor-speed overlay below.

Run from project root:
    python writing/pptx/quiet-drones-presentation/make_qualitative_slide_figs.py
"""

import sys
from pathlib import Path
from typing import cast

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = Path(__file__).resolve().parent / "qual_figs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT))


import tdseries as td  # noqa: E402

from data_processing.frames import get_meta  # noqa: E402
from tasks.rps_prediction import load_input_set, load_predictor  # noqa: E402
from utils.paths import get_datasets_path, get_results_path  # noqa: E402

ROTOR_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
ROTOR_LABELS = ["Rotor 1", "Rotor 2", "Rotor 3", "Rotor 4"]

CHECKPOINT = str(get_results_path("rps_exp_simple_conv/best_simple_conv.pt"))
DREGON_VALID = get_datasets_path("DREGON-LM/valid")
TARGET_SR = 16000
N_FFT = 2048
HOP = 512

SAMPLE_IDS = ("sample_00000", "sample_00149", "sample_00599")

# Slide-friendly style: larger fonts, clean axes.
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 16,
        "axes.titlesize": 18,
        "axes.labelsize": 16,
        "legend.fontsize": 15,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
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


def main():
    all_samples = {get_meta(s, "id", ""): s for s in load_input_set(str(DREGON_VALID))}
    pred = load_predictor(f"simple_conv@{CHECKPOINT}")

    for sid in SAMPLE_IDS:
        sample = all_samples[sid]
        audio_us = cast(td.Series, sample["audio"])
        audio = np.asarray(audio_us.data, dtype=np.float32)
        sr = cast(td.GridIndex, audio_us.tindex).sr
        dur = len(audio) / sr

        fig, axes = plt.subplots(
            2,
            1,
            figsize=(11.0, 6.2),
            gridspec_kw={"height_ratios": [1.0, 1.0], "hspace": 0.28},
        )

        # ── Spectrogram ───────────────────────────────────────────────
        ax = axes[0]
        spec = np.abs(
            np.fft.rfft(
                np.lib.stride_tricks.sliding_window_view(audio, N_FFT)[::HOP] * np.hanning(N_FFT),
                axis=-1,
            )
        )
        log_mag = np.log1p(spec.T)
        ax.imshow(
            log_mag,
            origin="lower",
            aspect="auto",
            extent=[0, dur, 0, sr / 2 / 1000],
            cmap="hot",
            vmin=np.percentile(log_mag, 2),
            vmax=np.percentile(log_mag, 99),
        )
        ax.set_ylim(0, 4)
        ax.set_ylabel("frequency [kHz]")
        ax.set_xticklabels([])
        ax.grid(False)

        # ── RPS overlay ───────────────────────────────────────────────
        ax = axes[1]
        n_frames = len(audio) // HOP + 1
        frame_times = np.arange(n_frames) * HOP / sr
        gt = sample["rps"].interpolate(frame_times)  # (4, F)
        pred_arr = pred.predict(audio, sr=sr)
        T = min(pred_arr.shape[1], n_frames)
        for r in range(4):
            ax.plot(
                frame_times[:T],
                gt[r, :T],
                ":",
                color=ROTOR_COLORS[r],
                lw=1.4,
                alpha=0.65,
            )
            ax.plot(
                frame_times[:T],
                pred_arr[r, :T],
                "-",
                color=ROTOR_COLORS[r],
                lw=1.6,
                alpha=0.9,
            )
        ax.set_xlabel("time [s]")
        ax.set_ylabel("rotor speed [rev/s]")
        ax.set_xlim(0, dur)

        legend_handles = [
            Line2D([0], [0], color="black", lw=1.6, ls=":", alpha=0.65, label="ground truth"),
            Line2D([0], [0], color="black", lw=1.6, ls="-", alpha=0.9, label="predicted"),
        ] + [
            Line2D([0], [0], color=ROTOR_COLORS[r], lw=3.0, label=ROTOR_LABELS[r]) for r in range(4)
        ]
        fig.legend(
            handles=legend_handles,
            loc="lower center",
            ncol=6,
            bbox_to_anchor=(0.5, -0.04),
            frameon=False,
        )

        out = OUT_DIR / f"qual_{sid}.png"
        fig.savefig(out, bbox_inches="tight", dpi=200)
        plt.close(fig)
        print("wrote", out)


if __name__ == "__main__":
    main()
