#!/usr/bin/env python3
"""Generate 3-panel plot: spectrogram + SimpleConv RPS + BiGRU-v2 RPS for one DREGON-LM valid sample."""

import glob
import os
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio

from utils.paths import get_datasets_path, get_results_path

PROJECT_ROOT = Path.cwd().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.rps_predictor import SimpleConv, SimpleConvBiGRUV2  # noqa: E402

# ─── Config ─────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_FFT = 2048
HOP_LENGTH = 512
NUM_ROTORS = 4
DATA_DIR = get_datasets_path("DREGON-LM/valid")
SIMPLECONV_CKPT = get_results_path("rps_exp_simple_conv/best_simple_conv.pt")
BIGRU_V2_CKPT = get_results_path("rps_exp_bigru_v2/best_simple_conv_bigru_v2.pt")
OUT_PATH = PROJECT_ROOT / "slides/2026-06-02-rps-progress/assets/sample_comparison_tmp.png"
SEED = 42

# ─── Pick a random valid sample ─────────────────────────────────────────────
random.seed(SEED)
samples = sorted(
    d
    for d in glob.glob(os.path.join(DATA_DIR, "sample_*"))
    if os.path.isfile(os.path.join(d, "mixture.wav")) and os.path.isfile(os.path.join(d, "rps.npy"))
)
# sample_dir = random.choice(samples)
sample_dir = DATA_DIR / "sample_00004"
print(f"Using sample: {sample_dir}")

# ─── Load audio & ground-truth RPS ──────────────────────────────────────────
audio, sr = torchaudio.load(os.path.join(sample_dir, "mixture.wav"))
audio = audio[0]  # mono

rps_gt = torch.from_numpy(np.load(os.path.join(sample_dir, "rps.npy"))).float()  # (4, rps_T)
n_frames = audio.shape[0] // HOP_LENGTH + 1
rps_gt = F.interpolate(
    rps_gt.unsqueeze(0), size=n_frames, mode="linear", align_corners=False
).squeeze(0)  # (4, n_frames)

# ─── Load models ────────────────────────────────────────────────────────────
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
audio_batch = audio.unsqueeze(0).to(DEVICE)  # (1, samples)
print(audio_batch.shape)
predictions = {}
with torch.no_grad():
    for name, model in models.items():
        pred = model(audio_batch)  # (1, 4, T)
        print(pred)
        predictions[name] = pred.squeeze(0).cpu().numpy()  # (4, T)

# ─── Build 3-panel figure ───────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
fig.patch.set_facecolor("white")

time = np.arange(n_frames) * HOP_LENGTH / sr  # seconds

# -- Panel 1: Spectrogram --
window = torch.hann_window(N_FFT)
X = torch.stft(
    audio, n_fft=N_FFT, hop_length=HOP_LENGTH, window=window, return_complex=True, normalized=True
)
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
    ax2.plot(
        time,
        rps_gt[r].numpy(),
        color=colors[r],
        linestyle="--",
        alpha=0.7,
        label=f"GT R{r + 1}" if r == 0 else "",
    )
    ax2.plot(
        time,
        predictions["SimpleConv"][r],
        color=colors[r],
        linestyle="-",
        label=f"Pred R{r + 1}" if r == 0 else "",
    )
ax2.set_ylabel("RPS (Hz)")
ax2.set_title("SimpleConv Predictions vs Ground Truth")
ax2.legend(loc="upper right", ncol=2, fontsize=8)

# -- Panel 3: BiGRU-v2 RPS vs GT --
ax3 = axes[2]
for r in range(NUM_ROTORS):
    ax3.plot(time, rps_gt[r].numpy(), color=colors[r], linestyle="--", alpha=0.7)
    ax3.plot(time, predictions["BiGRU-v2"][r], color=colors[r], linestyle="-")
ax3.set_ylabel("RPS (Hz)")
ax3.set_xlabel("Time (s)")
ax3.set_title("BiGRU-v2 Predictions vs Ground Truth")

plt.tight_layout()
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
plt.savefig(OUT_PATH, dpi=200, bbox_inches="tight", facecolor="white")
print(f"Saved figure to {OUT_PATH}")
