#!/usr/bin/env python3
"""Generate 3-panel plot using V4 (2.5% synth) checkpoints on v2_sample_00558."""

import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchaudio

from models.rps_predictor import SimpleConv, SimpleConvBiGRUV2
from utils.paths import get_results_path

# ─── Config ─────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_FFT = 2048
HOP_LENGTH = 512
NUM_ROTORS = 4
SR = 16000

# Same sample as slide 8
SAMPLE_DIR = str(get_results_path("rps_cross_eval/samples/v2_sample_00558"))
SIMPLECONV_CKPT = str(get_results_path("rps_predictor_v4_2.5pct/simple_conv/best_simple_conv.pt"))
BIGRU_V2_CKPT = str(
    get_results_path("rps_predictor_v4_2.5pct/simple_conv_bigru_v2/best_simple_conv_bigru_v2.pt")
)
OUT_PATH = "slides/2026-06-02-rps-progress/assets/sample_comparison_v4.png"

# ─── Load audio & ground-truth RPS ─────────────────────────────────────────
audio, sr = torchaudio.load(os.path.join(SAMPLE_DIR, "mixture.wav"))
audio = audio[0]

rps_gt = torch.from_numpy(np.load(os.path.join(SAMPLE_DIR, "rps_target.npy"))).float()
n_frames = rps_gt.shape[1]
time = np.arange(n_frames) * HOP_LENGTH / SR

# ─── Load V4 models ─────────────────────────────────────────────────────────
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

# ─── Run inference ─────────────────────────────────────────────────────────
audio_batch = audio.unsqueeze(0).to(DEVICE)
predictions = {}
with torch.no_grad():
    for name, model in models.items():
        pred = model(audio_batch)
        predictions[name] = pred.squeeze(0).cpu().numpy()

# ─── Build 3-panel figure ───────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
fig.patch.set_facecolor("white")

# Panel 1: Spectrogram
window = torch.hann_window(N_FFT)
X = torch.stft(
    audio, n_fft=N_FFT, hop_length=HOP_LENGTH, window=window, return_complex=True, normalized=True
)
Sxx = torch.abs(X).numpy()

ax1 = axes[0]
ax1.imshow(
    20 * np.log10(Sxx + 1e-10),
    aspect="auto",
    origin="lower",
    cmap="magma",
    extent=[time[0], time[-1], 0, SR // 2],
)
ax1.set_ylabel("Frequency (Hz)")
ax1.set_title("Noisy Mixture Spectrogram")
ax1.set_ylim(0, 4000)

# Panel 2: SimpleConv
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
ax2.set_title("V4 (2.5% synth) SimpleConv Predictions vs Ground Truth")
ax2.legend(loc="upper right", ncol=2, fontsize=8)

# Panel 3: BiGRU-v2
ax3 = axes[2]
for r in range(NUM_ROTORS):
    ax3.plot(time, rps_gt[r].numpy(), color=colors[r], linestyle="--", alpha=0.7)
    ax3.plot(time, predictions["BiGRU-v2"][r], color=colors[r], linestyle="-")
ax3.set_ylabel("RPS (Hz)")
ax3.set_xlabel("Time (s)")
ax3.set_title("V4 (2.5% synth) BiGRU-v2 Predictions vs Ground Truth")

plt.tight_layout()
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
plt.savefig(OUT_PATH, dpi=200, bbox_inches="tight", facecolor="white")
print(f"Saved figure to {OUT_PATH}")
