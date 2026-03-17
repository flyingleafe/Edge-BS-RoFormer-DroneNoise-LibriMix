#!/usr/bin/env python3
"""
Plot input spectrogram + target RPS + predicted RPS for RPS predictor samples.
Spectrograms use log-scale frequency axis. Can plot all .npz in a dir or specific samples.

Usage:
  # All samples → single figure
  direnv exec . python plot_rps_samples.py

  # Specific samples → one PNG per sample
  direnv exec . python plot_rps_samples.py --samples sample_00000 sample_00299 sample_00599 --output slides/2026-02-24/assets
"""

import argparse
import os
import glob

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

SAMPLES_DIR_DEFAULT = "results/rps_predictor/samples"
OUT_PATH_DEFAULT = "results/rps_predictor/sample_predictions.png"
ROTOR_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
ROTOR_LABELS = ["Rotor 1", "Rotor 2", "Rotor 3", "Rotor 4"]


def plot_one_sample(ax_spec, ax_tgt, ax_prd, data, name, freq_log_scale=True):
    """Fill three axes: spectrogram (log freq optional), target RPS, predicted RPS."""
    log_mag = data["log_mag"]          # (F, T)
    rps_tgt = data["rps_target"]       # (4, T)
    rps_prd = data["rps_pred"]         # (4, T)
    hop = int(data["hop_length"])
    sr = int(data["sample_rate"])

    # Larger font sizes for slides
    title_fs = 13
    sub_fs = 11
    label_fs = 11
    legend_fs = 9

    F, T = log_mag.shape
    time_edges = np.arange(T + 1) * hop / sr
    time_sec = (time_edges[:-1] + time_edges[1:]) / 2  # frame centers for RPS
    freq_edges = np.linspace(0, sr / 2, F + 1)

    # --- Spectrogram (log-scale frequency) ---
    im = ax_spec.pcolormesh(
        time_edges,
        freq_edges,
        log_mag,
        shading="flat",
        cmap="magma",
    )
    if freq_log_scale:
        ax_spec.set_yscale("symlog", linthresh=100, subs=[2, 3, 4, 5, 6, 7, 8, 9])
    ax_spec.set_ylabel("Freq (Hz)", fontsize=label_fs)
    ax_spec.set_title(f"{name} — Input Spectrogram", fontsize=title_fs, fontweight="bold")
    ax_spec.set_ylim(1, 4000)  # drone noise mostly < 4 kHz; avoid 0 for log scale
    ax_spec.tick_params(axis="both", labelsize=label_fs - 1)
    plt.colorbar(im, ax=ax_spec, pad=0.01, fraction=0.02, label="log(1+|X|)").ax.tick_params(labelsize=label_fs - 1)

    # --- Target RPS ---
    for r in range(4):
        ax_tgt.plot(time_sec, rps_tgt[r], color=ROTOR_COLORS[r],
                    label=ROTOR_LABELS[r], lw=1.2)
    ax_tgt.set_ylabel("RPS", fontsize=label_fs)
    ax_tgt.set_title("Target Motor Speeds", fontsize=sub_fs)
    ax_tgt.legend(loc="upper right", fontsize=legend_fs, ncol=4)
    ax_tgt.grid(True, alpha=0.3)
    ax_tgt.tick_params(axis="both", labelsize=label_fs - 1)

    # --- Predicted RPS (target as faint background) ---
    for r in range(4):
        ax_prd.plot(time_sec, rps_tgt[r], color=ROTOR_COLORS[r],
                    alpha=0.25, lw=1, ls="--")
        ax_prd.plot(time_sec, rps_prd[r], color=ROTOR_COLORS[r],
                    label=ROTOR_LABELS[r], lw=1.2)
    mae = np.abs(rps_prd - rps_tgt).mean()
    ax_prd.set_ylabel("RPS", fontsize=label_fs)
    ax_prd.set_title(f"Predicted Motor Speeds  (MAE={mae:.2f} RPS, dashed=target)",
                     fontsize=sub_fs)
    ax_prd.set_xlabel("Time (s)", fontsize=label_fs)
    ax_prd.legend(loc="upper right", fontsize=legend_fs, ncol=4)
    ax_prd.grid(True, alpha=0.3)
    ax_prd.tick_params(axis="both", labelsize=label_fs - 1)


def main():
    parser = argparse.ArgumentParser(description="Plot RPS predictor samples (spectrogram + target + predicted RPS)")
    parser.add_argument("--samples_dir", default=SAMPLES_DIR_DEFAULT,
                        help="Directory containing .npz sample files")
    parser.add_argument("--samples", nargs="*", default=None,
                        help="Sample IDs to plot (e.g. sample_00000 sample_00299). If omitted, plot all .npz")
    parser.add_argument("--output", default=OUT_PATH_DEFAULT,
                        help="Output path: file (if one figure) or directory (if one file per --samples)")
    parser.add_argument("--no_log_freq", action="store_true",
                        help="Use linear frequency axis for spectrogram instead of log scale")
    parser.add_argument("--per_sample", action="store_true",
                        help="With --output <dir>: write one PNG per .npz (use with or without --samples)")
    args = parser.parse_args()

    if args.samples:
        npz_files = [
            os.path.join(args.samples_dir, f"{s}.npz")
            for s in args.samples
        ]
        missing = [p for p in npz_files if not os.path.isfile(p)]
        if missing:
            raise FileNotFoundError(f"Missing .npz files: {missing}")
        output_dir = args.output
        os.makedirs(output_dir, exist_ok=True)
        one_per_sample = True
    else:
        npz_files = sorted(glob.glob(os.path.join(args.samples_dir, "*.npz")))
        if not npz_files:
            raise FileNotFoundError(f"No .npz files in {args.samples_dir}")
        one_per_sample = args.per_sample
        if one_per_sample:
            output_dir = args.output
            os.makedirs(output_dir, exist_ok=True)

    freq_log = not args.no_log_freq

    # 30% wider, same height (14 * 1.3, 4.2*3)
    fig_w, fig_h = 14 * 1.3, 4.2 * 3
    if one_per_sample:
        for path in npz_files:
            data = np.load(path)
            name = os.path.splitext(os.path.basename(path))[0]
            fig, (ax_spec, ax_tgt, ax_prd) = plt.subplots(3, 1, figsize=(fig_w, fig_h), sharex=True, gridspec_kw={"hspace": 0.45}, constrained_layout=True)
            plot_one_sample(ax_spec, ax_tgt, ax_prd, data, name, freq_log_scale=freq_log)
            out_path = os.path.join(output_dir, f"{name}.png")
            plt.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"Saved {out_path}")
    else:
        n = len(npz_files)
        fig = plt.figure(figsize=(fig_w, 4.2 * n))
        gs = GridSpec(n * 3, 1, hspace=0.45)
        for i, path in enumerate(npz_files):
            data = np.load(path)
            name = os.path.splitext(os.path.basename(path))[0]
            ax_spec = fig.add_subplot(gs[i * 3])
            ax_tgt = fig.add_subplot(gs[i * 3 + 1], sharex=ax_spec)
            ax_prd = fig.add_subplot(gs[i * 3 + 2], sharex=ax_spec)
            plot_one_sample(ax_spec, ax_tgt, ax_prd, data, name, freq_log_scale=freq_log)
        plt.suptitle("RPS Predictor — Validation Samples", fontsize=13, fontweight="bold", y=1.0)
        plt.tight_layout()
        plt.savefig(args.output, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
