#!/usr/bin/env python3
"""
Plot input spectrogram + target RPS + predicted RPS for each saved sample.
Each sample gets a 3-row subplot (spectrogram, target speeds, predicted speeds).
"""

import os
import glob

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

SAMPLES_DIR = "results/rps_predictor/samples"
OUT_PATH = "results/rps_predictor/sample_predictions.png"
ROTOR_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
ROTOR_LABELS = ["Rotor 1", "Rotor 2", "Rotor 3", "Rotor 4"]

npz_files = sorted(glob.glob(os.path.join(SAMPLES_DIR, "*.npz")))
n = len(npz_files)
assert n > 0, f"No .npz files in {SAMPLES_DIR}"

fig = plt.figure(figsize=(14, 4.2 * n))
gs = GridSpec(n * 3, 1, hspace=0.45)

for i, path in enumerate(npz_files):
    data = np.load(path)
    log_mag = data["log_mag"]          # (F, T)
    rps_tgt = data["rps_target"]       # (4, T)
    rps_prd = data["rps_pred"]         # (4, T)
    hop = int(data["hop_length"])
    sr = int(data["sample_rate"])
    name = os.path.splitext(os.path.basename(path))[0]

    T = rps_tgt.shape[1]
    time_sec = np.arange(T) * hop / sr

    # --- Row 1: spectrogram ---
    ax_spec = fig.add_subplot(gs[i * 3])
    im = ax_spec.imshow(
        log_mag, aspect="auto", origin="lower",
        extent=[0, time_sec[-1], 0, sr / 2],
        cmap="magma",
    )
    ax_spec.set_ylabel("Freq (Hz)")
    ax_spec.set_title(f"{name} — Input Spectrogram", fontsize=10, fontweight="bold")
    ax_spec.set_ylim(0, 4000)  # drone noise is mostly < 4 kHz
    fig.colorbar(im, ax=ax_spec, pad=0.01, fraction=0.02, label="log(1+|X|)")

    # --- Row 2: target RPS ---
    ax_tgt = fig.add_subplot(gs[i * 3 + 1], sharex=ax_spec)
    for r in range(4):
        ax_tgt.plot(time_sec, rps_tgt[r], color=ROTOR_COLORS[r],
                    label=ROTOR_LABELS[r], lw=1)
    ax_tgt.set_ylabel("RPS")
    ax_tgt.set_title("Target Motor Speeds", fontsize=9)
    ax_tgt.legend(loc="upper right", fontsize=7, ncol=4)
    ax_tgt.grid(True, alpha=0.3)

    # --- Row 3: predicted RPS (with target as faint background) ---
    ax_prd = fig.add_subplot(gs[i * 3 + 2], sharex=ax_spec)
    for r in range(4):
        ax_prd.plot(time_sec, rps_tgt[r], color=ROTOR_COLORS[r],
                    alpha=0.25, lw=1, ls="--")
        ax_prd.plot(time_sec, rps_prd[r], color=ROTOR_COLORS[r],
                    label=ROTOR_LABELS[r], lw=1)
    mae = np.abs(rps_prd - rps_tgt).mean()
    ax_prd.set_ylabel("RPS")
    ax_prd.set_title(f"Predicted Motor Speeds  (MAE={mae:.2f} RPS, dashed=target)",
                     fontsize=9)
    ax_prd.set_xlabel("Time (s)")
    ax_prd.legend(loc="upper right", fontsize=7, ncol=4)
    ax_prd.grid(True, alpha=0.3)

plt.suptitle("RPS Predictor — Validation Samples", fontsize=13, fontweight="bold", y=1.0)
plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
print(f"Saved to {OUT_PATH}")
plt.close()
