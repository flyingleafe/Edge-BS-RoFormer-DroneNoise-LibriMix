#!/usr/bin/env python3
"""Quick CPU test: tune aggressiveness + amplitude to match real data."""

# Diagnostic script: imports follow module setup — exempt from import-order lint.
# ruff: noqa: E402
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

SR = 16000
DUR_S = 1.0
N_SAMPLES = 20
N_HARM = 100
DEVICE = "cuda:0"
CKPT = "/gpfs/scratch/acw592/results/noise_gen_sweep/baseline/best_positional_harmonic_gen.pt"

from data_processing import rps_synthesis
from data_processing.dregon import get_geometry as dregon_geom
from data_processing.michaels import get_geometry as michaels_geom
from models.generative import PositionalHarmonicNoiseGen
from tasks.noise_generation import DroneCodebook, geometry_to_rel_pos

bundle = torch.load(CKPT, map_location=DEVICE, weights_only=False)
model = PositionalHarmonicNoiseGen(
    sample_rate=SR, n_harmonics=N_HARM, use_diff_noise=True, cond_dim=int(bundle["cond_dim"])
)
model.load_state_dict(bundle["model"])
model.to(DEVICE).eval()
codebook = DroneCodebook(int(bundle["cond_dim"]), names=list(bundle["drone_names"])).to(DEVICE)
codebook.load_state_dict(bundle["codebook"])


def gen(drone, n, aggressiveness, amp_scale=1.0):
    blend = {"dregon": 0.0, "michaels": 1.0}[drone]
    rng = np.random.default_rng(42)
    rps_np = rps_synthesis.generate_intermittent_batch(
        n, DUR_S, SR, drone_profile=blend, aggressiveness=aggressiveness, rng=rng
    )
    if drone == "michaels":
        mp, rp = michaels_geom()
    else:
        mp, rp = dregon_geom(Path("/gpfs/scratch/acw592/data/DREGON"))
    mp = np.array(mp)
    rp = np.array(rp)
    rel = (
        torch.from_numpy(geometry_to_rel_pos(mp, rp))
        .float()
        .to(DEVICE)
        .unsqueeze(0)
        .expand(n, -1, -1, -1)
    )
    z = codebook([drone] * n).to(DEVICE)
    rps_t = torch.from_numpy(rps_np).float().to(DEVICE)
    ip = (
        torch.from_numpy(rng.uniform(0, 2 * np.pi, size=(n, rp.shape[0], N_HARM)))
        .float()
        .to(DEVICE)
    )
    with torch.no_grad():
        audio = model(rps_t, rel, z, initial_phases=ip).cpu().numpy()
    return audio * amp_scale, rps_np


# Real reference values (from previous diagnostic)
REAL_RMS = {"dregon": 0.0548, "michaels": 0.0502}  # approximate
REAL_RPS_STD = {"dregon": 18.6, "michaels": 25.0}

# Test different aggressiveness values
for agg in [1.0, 5.0, 10.0, 20.0]:
    for drone in ["dregon", "michaels"]:
        audio, rps = gen(drone, N_SAMPLES, agg)
        rms = float(np.sqrt(np.mean(audio**2)))
        rps_std_all = rps.reshape(-1, rps.shape[-1]).std(axis=-1)
        rps_std = float(rps_std_all.mean())
        rps_range = float(np.percentile(rps, 95) - np.percentile(rps, 5))
        amp_ratio = rms / REAL_RMS[drone]
        rps_ratio = rps_std / REAL_RPS_STD[drone]
        print(
            f"agg={agg:4.0f} {drone:10s}  RMS={rms:.4f} (×{amp_ratio:.2f})  RPS σ={rps_std:.1f} (×{rps_ratio:.2f})  RPS range={rps_range:.1f}"
        )

# Now find a good amp_scale
print("\n=== With amplitude scaling ===")
for drone in ["dregon", "michaels"]:
    audio_raw, rps = gen(drone, N_SAMPLES, aggressiveness=5.0)
    # Compute per-channel RMS and find scaling per channel
    raw_rms = np.sqrt(np.mean(audio_raw**2, axis=-1))  # (N, C)
    mean_rms = float(raw_rms.mean())
    target_rms = REAL_RMS[drone]
    amp_scale = target_rms / max(mean_rms, 1e-10)
    scaled = audio_raw * amp_scale
    scaled_rms = float(np.sqrt(np.mean(scaled**2)))
    print(f"  {drone}: raw RMS={mean_rms:.4f}, scale={amp_scale:.3f}, scaled RMS={scaled_rms:.4f}")

# Best config
print("\n=== Best guess config ===")
BEST_AGG = 5.0
for drone in ["dregon", "michaels"]:
    audio_raw, rps = gen(drone, N_SAMPLES, BEST_AGG)
    mean_rms = float(np.sqrt(np.mean(audio_raw**2, axis=-1)).mean())
    amp_scale = REAL_RMS[drone] / max(mean_rms, 1e-10)
    rps_std_all = rps.reshape(-1, rps.shape[-1]).std(axis=-1)
    rps_std = float(rps_std_all.mean())
    print(
        f"  {drone}: aggressiveness={BEST_AGG}, amp_scale={amp_scale:.3f}, RPS σ={rps_std:.1f} (target {REAL_RPS_STD[drone]:.1f})"
    )

print("\nDone.")
