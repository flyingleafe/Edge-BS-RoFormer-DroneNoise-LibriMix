#!/usr/bin/env python3
"""Generate 3-panel plot using V3 checkpoints on the SAME sample as the V1/V2 plot."""

import glob
import os

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import matplotlib.pyplot as plt

from models.rps_predictor import SimpleConv, SimpleConvBiGRUV2

# ─── Config ─────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_FFT = 2048
HOP_LENGTH = 512
NUM_ROTORS = 4
DATA_DIR = "datasets/DREGON-LM/valid"
SIMPLECONV_CKPT = "results/rps_predictor_v3/simple_conv/best_simple_conv.pt"
BIGRU_V2_CKPT = "results/rps_predictor_v3/simple_conv_bigru_v2/best_simple_conv_bigru_v2.pt"
OUT_PATH = "slides/2026-06-02-rps-progress/assets/sample_comparison_v3.png"
SAMPLE_NAME = "sample_00114"  # same sample as V1/V2 plot

# ─── Load the exact same sample ─────────────────────────────────────────────
sample_dir = os.path.join(DATA_DIR, SAMPLE_NAME)
print(f"Using sample: {sample_dir}")

audio, sr = torchaudio.load(os.path.join(sample_dir, "mixture.wav"))
audio = audio[0]

rps_gt = torch.from_numpy(np.load(os.path.join(sample_dir, "rps.npy"))).float()
n_frames = audio.shape[0] // HOP_LENGTH + 1
rps_gt = F.interpolate(
    rps_gt.unsqueeze(0), size=n_frames, mode="linear", align_corners=False
).squeeze(0)

# ─── Load V3 models ─────────────────────────────────────────────────────────
models = {
    "SimpleConv": SimpleConv(N_FFT, HOP_LENGTH, NUM_ROTORS).to(DEVICE),
    "BiGRU-v2": SimpleConvBiGRUV2(N_FFT, HOP_LENGTH, NUM_ROTORS).to(DEVICE),
}
checkpoints = {
    "SimpleConv": SIMPLECONV_CKPT,
    "BiGRU-v2": BIGRU_V2_CKPT,
}
for name, model in models.items():
    ckpt = torch.load(checkpoints[name], map_location=DEVICE, weights_only=True)
    model.load_state_dict(ckpt)
    model.eval()
    print(f"Loaded {name}")

# ─── Run inference ───────────────────────────────────────────────────────────
audio_batch = audio.unsqueeze(0).to(DEVICE)
predictions = {}
with torch.no_grad():
    for name, model in models.items():
        pred = model(audio_batch)
        predictions[name] = pred.squeeze(0).cpu().numpy()

# ─── Build 3-panel figure ───────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
fig.patch.set_facecolor("white")

time = np.arange(n_frames) * HOP_LENGTH / sr

# -- Panel 1: Spectrogram --
window = torch.hann_window(N_FFT)
X = torch.stft(audio, n_fft=N_FFT, hop_length=HOP_LENGTH, window=window, return_complex=True, normalized=True)
Sxx = torch.abs(X).numpy()

ax1 = axes[0]
im = ax1.imshow(
    20 * np.log10(Sxx + 1e-10),
    aspect="auto",
    origin="lower",
    cmap="magma",
    extent=[time[0], time[-1], 0, sr // 2],
)
ax1.set_ylabel("Frequency (Hz)")
ax1.set_title("Noisy Mixture Spectrogram")
ax1.set_ylim(0, 4000)

# -- Panel 2: SimpleConv RPS vs GT --
ax2 = axes[1]
colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3"]
for r in range(NUM_ROTORS):
    ax2.plot(time, rps_gt[r].numpy(), color=colors[r], linestyle="--", alpha=0.7, label=f"GT R{r+1}" if r == 0 else "")
    ax2.plot(time, predictions["SimpleConv"][r], color=colors[r], linestyle="-", label=f"Pred R{r+1}" if r == 0 else "")
ax2.set_ylabel("RPS (Hz)")
ax2.set_title("V3 SimpleConv Predictions vs Ground Truth")
ax2.legend(loc="upper right", ncol=2, fontsize=8)

# -- Panel 3: BiGRU-v2 RPS vs GT --
ax3 = axes[2]
for r in range(NUM_ROTORS):
    ax3.plot(time, rps_gt[r].numpy(), color=colors[r], linestyle="--", alpha=0.7)
    ax3.plot(time, predictions["BiGRU-v2"][r], color=colors[r], linestyle="-")
ax3.set_ylabel("RPS (Hz)")
ax3.set_xlabel("Time (s)")
ax3.set_title("V3 BiGRU-v2 Predictions vs Ground Truth")

plt.tight_layout()
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
plt.savefig(OUT_PATH, dpi=200, bbox_inches="tight", facecolor="white")
print(f"Saved figure to {OUT_PATH}")
