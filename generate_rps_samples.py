#!/usr/bin/env python3
"""
Generate input–prediction–target triples from the best RPS predictor checkpoint
on 5 validation samples.  Saves .npz files to results/rps_predictor/samples/.
"""

import os
import glob

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio

from train_rps_predictor import RPSPredictor

CHECKPOINT = "results/rps_predictor/best.pt"
DATA_DIR = "datasets/DREGON-LM/valid"
OUT_DIR = "results/rps_predictor/samples"
N_FFT = 2048
HOP = 512
DEVICE = "cpu"

# Pick 5 evenly-spaced samples from the validation set
sample_dirs = sorted(
    d for d in glob.glob(os.path.join(DATA_DIR, "sample_*"))
    if os.path.isfile(os.path.join(d, "mixture.wav"))
    and os.path.isfile(os.path.join(d, "rps.npy"))
)
indices = np.linspace(0, len(sample_dirs) - 1, 5, dtype=int)
selected = [sample_dirs[i] for i in indices]

# Load model
model = RPSPredictor(n_fft=N_FFT, hop_length=HOP)
model.load_state_dict(torch.load(CHECKPOINT, weights_only=True, map_location=DEVICE))
model.eval()

os.makedirs(OUT_DIR, exist_ok=True)

for idx, d in enumerate(selected):
    name = os.path.basename(d)

    # Audio
    audio, sr = torchaudio.load(os.path.join(d, "mixture.wav"))
    audio = audio[0]  # mono (samples,)

    # Target RPS, resampled to STFT frames
    rps_raw = np.load(os.path.join(d, "rps.npy"))  # (4, rps_T)
    rps_raw_t = torch.from_numpy(rps_raw).float()
    n_frames = audio.shape[0] // HOP + 1
    rps_target = F.interpolate(
        rps_raw_t.unsqueeze(0), size=n_frames, mode="linear", align_corners=False
    ).squeeze(0).numpy()  # (4, n_frames)

    # Prediction
    with torch.no_grad():
        rps_pred = model(audio.unsqueeze(0)).squeeze(0).numpy()  # (4, n_frames)

    # Spectrogram (for plotting)
    X = torch.stft(
        audio.unsqueeze(0), n_fft=N_FFT, hop_length=HOP,
        window=torch.hann_window(N_FFT), return_complex=True, normalized=True,
    )
    mag = X.abs().squeeze(0).numpy()  # (F, T)
    log_mag = np.log1p(mag)

    # Save
    out_path = os.path.join(OUT_DIR, f"{name}.npz")
    np.savez(
        out_path,
        audio=audio.numpy(),
        log_mag=log_mag,
        rps_target=rps_target,
        rps_pred=rps_pred,
        sample_rate=sr,
        n_fft=N_FFT,
        hop_length=HOP,
    )
    mae = np.abs(rps_pred - rps_target).mean()
    print(f"[{idx+1}/5] {name}: MAE={mae:.2f} RPS → {out_path}")

print(f"\nAll samples saved to {OUT_DIR}/")
