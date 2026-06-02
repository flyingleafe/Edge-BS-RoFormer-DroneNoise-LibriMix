#!/usr/bin/env python3
"""Generate side-by-side 3-panel plots for a V2 sample using pre-computed predictions."""

import os
import numpy as np
import torch
import torchaudio
import matplotlib.pyplot as plt

# ─── Config ─────────────────────────────────────────────────────────────────
SAMPLE_DIR = "results/rps_cross_eval/samples/v2_sample_00558"
OUT_DIR = "slides/2026-06-02-rps-progress/assets"
N_FFT = 2048
HOP_LENGTH = 512
SR = 16000
NUM_ROTORS = 4

# ─── Load audio ─────────────────────────────────────────────────────────────
audio, sr = torchaudio.load(os.path.join(SAMPLE_DIR, "mixture.wav"))
audio = audio[0]  # mono

# ─── Load ground truth RPS ──────────────────────────────────────────────────
rps_gt = torch.from_numpy(np.load(os.path.join(SAMPLE_DIR, "rps_target.npy")))  # (4, T)
n_frames = rps_gt.shape[1]
time = np.arange(n_frames) * HOP_LENGTH / SR

# ─── Load predictions ───────────────────────────────────────────────────────
predictions = {
    "old_simple_conv": np.load(os.path.join(SAMPLE_DIR, "rps_pred_old_simple_conv.npy")),
    "old_bigru_v2": np.load(os.path.join(SAMPLE_DIR, "rps_pred_old_bigru_v2.npy")),
    "v3_simple_conv": np.load(os.path.join(SAMPLE_DIR, "rps_pred_v3_simple_conv.npy")),
    "v3_bigru_v2": np.load(os.path.join(SAMPLE_DIR, "rps_pred_v3_bigru_v2.npy")),
}

# ─── Helper to build a 3-panel figure ───────────────────────────────────────
def build_3panel(models_dict, title_prefix, out_path):
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.patch.set_facecolor("white")

    # Panel 1: Spectrogram
    window = torch.hann_window(N_FFT)
    X = torch.stft(audio, n_fft=N_FFT, hop_length=HOP_LENGTH, window=window,
                   return_complex=True, normalized=True)
    Sxx = torch.abs(X).numpy()

    ax1 = axes[0]
    ax1.imshow(
        20 * np.log10(Sxx + 1e-10),
        aspect="auto", origin="lower", cmap="magma",
        extent=[time[0], time[-1], 0, SR // 2],
    )
    ax1.set_ylabel("Frequency (Hz)")
    ax1.set_title("Noisy Mixture Spectrogram")
    ax1.set_ylim(0, 4000)

    # Panel 2: SimpleConv
    ax2 = axes[1]
    colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3"]
    for r in range(NUM_ROTORS):
        ax2.plot(time, rps_gt[r].numpy(), color=colors[r], linestyle="--", alpha=0.7,
                 label=f"GT R{r+1}" if r == 0 else "")
        ax2.plot(time, models_dict["simple_conv"][r], color=colors[r], linestyle="-",
                 label=f"Pred R{r+1}" if r == 0 else "")
    ax2.set_ylabel("RPS (Hz)")
    ax2.set_title(f"{title_prefix} SimpleConv Predictions vs Ground Truth")
    ax2.legend(loc="upper right", ncol=2, fontsize=8)

    # Panel 3: BiGRU-v2
    ax3 = axes[2]
    for r in range(NUM_ROTORS):
        ax3.plot(time, rps_gt[r].numpy(), color=colors[r], linestyle="--", alpha=0.7)
        ax3.plot(time, models_dict["bigru_v2"][r], color=colors[r], linestyle="-")
    ax3.set_ylabel("RPS (Hz)")
    ax3.set_xlabel("Time (s)")
    ax3.set_title(f"{title_prefix} BiGRU-v2 Predictions vs Ground Truth")

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Saved figure to {out_path}")
    plt.close()


# ─── Generate both figures ──────────────────────────────────────────────────
build_3panel(
    {"simple_conv": predictions["old_simple_conv"], "bigru_v2": predictions["old_bigru_v2"]},
    "Old",
    os.path.join(OUT_DIR, "sample_comparison_v2_old.png"),
)

build_3panel(
    {"simple_conv": predictions["v3_simple_conv"], "bigru_v2": predictions["v3_bigru_v2"]},
    "V3",
    os.path.join(OUT_DIR, "sample_comparison_v2_v3.png"),
)
