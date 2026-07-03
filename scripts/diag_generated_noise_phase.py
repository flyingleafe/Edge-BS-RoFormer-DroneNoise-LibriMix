#!/usr/bin/env python3
"""
Diagnostic part 2: phase randomization + extreme values.

Checks:
  - Zero-phase vs random-phase: amplitude/spectral difference
  - Whether generated audio contains extreme samples (clicks/pops)
  - Per-sample RMS consistency
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import torch

SR = 16000; DUR_S = 1.0; N_SAMPLES = 100; N_HARM = 100; DEVICE = "cuda:0"
CKPT = "/gpfs/scratch/acw592/results/noise_gen_sweep/baseline/best_positional_harmonic_gen.pt"

from models.generative import PositionalHarmonicNoiseGen
from tasks.noise_generation import DroneCodebook, geometry_to_rel_pos
from data_processing import rps_synthesis
from data_processing.dregon import get_geometry as dregon_geom
from data_processing.michaels import get_geometry as michaels_geom

bundle = torch.load(CKPT, map_location=DEVICE, weights_only=False)
model = PositionalHarmonicNoiseGen(sample_rate=SR, n_harmonics=N_HARM, use_diff_noise=True, cond_dim=int(bundle["cond_dim"]))
model.load_state_dict(bundle["model"]); model.to(DEVICE).eval()
codebook = DroneCodebook(int(bundle["cond_dim"]), names=list(bundle["drone_names"])).to(DEVICE)
codebook.load_state_dict(bundle["codebook"])

def gen(drone, n, random_phase):
    blend = {"dregon": 0.0, "michaels": 1.0}[drone]
    rng = np.random.default_rng(42)
    rps_np = rps_synthesis.generate_intermittent_batch(n, DUR_S, SR, drone_profile=blend, aggressiveness=1.0, rng=rng)
    if drone == "michaels":
        mp, rp = michaels_geom()
    else:
        mp, rp = dregon_geom(Path("/gpfs/scratch/acw592/data/DREGON"))
    mp = np.array(mp); rp = np.array(rp)
    rel = torch.from_numpy(geometry_to_rel_pos(mp, rp)).float().to(DEVICE).unsqueeze(0).expand(n,-1,-1,-1)
    z = codebook([drone]*n).to(DEVICE)
    rps_t = torch.from_numpy(rps_np).float().to(DEVICE)
    ip = None
    if random_phase:
        ip = torch.from_numpy(rng.uniform(0, 2*np.pi, size=(n, rp.shape[0], N_HARM))).float().to(DEVICE)
    with torch.no_grad():
        audio = model(rps_t, rel, z, initial_phases=ip).cpu().numpy()
    return audio

print("=== Generating with zero phase ===")
gz_d = gen("dregon", N_SAMPLES, False)
gz_m = gen("michaels", N_SAMPLES, False)

print("=== Generating with RANDOM phase ===")
gr_d = gen("dregon", N_SAMPLES, True)
gr_m = gen("michaels", N_SAMPLES, True)

# ---------- Compare zero vs random phase ----------
print("\n=== Phase comparison (per-sample RMS stats) ===")
for name, z_aud, r_aud in [("DREGON", gz_d, gr_d), ("Michael's", gz_m, gr_m)]:
    z_rms = np.sqrt(np.mean(z_aud**2, axis=(-2,-1)))  # per-sample RMS
    r_rms = np.sqrt(np.mean(r_aud**2, axis=(-2,-1)))
    print(f"\n{name}:")
    print(f"  Zero-phase RMS:   mean={z_rms.mean():.4f}  std={z_rms.std():.4f}  min={z_rms.min():.4f}  max={z_rms.max():.4f}")
    print(f"  Random-phase RMS: mean={r_rms.mean():.4f}  std={r_rms.std():.4f}  min={r_rms.min():.4f}  max={r_rms.max():.4f}")

    # Per-sample correlation (within each mic channel)
    z_flat = z_aud.reshape(N_SAMPLES, -1)  # (N, M*T)
    r_flat = r_aud.reshape(N_SAMPLES, -1)
    corr = np.array([np.corrcoef(z_flat[i], r_flat[i])[0,1] for i in range(N_SAMPLES)])
    print(f"  Per-sample correlation (zero vs random): mean={corr.mean():.3f}, std={corr.std():.3f}")

    # Check if random phase changes spectral shape
    def avg_spec(audio):
        specs = []
        for i in range(min(20, audio.shape[0])):
            for c in range(audio.shape[1]):
                x = torch.from_numpy(audio[i,c]).float()
                s = torch.stft(x, 2048, 512, window=torch.hann_window(2048), return_complex=True).abs().mean(-1).numpy()
                specs.append(s)
        return np.mean(specs, axis=0)

    z_spec = avg_spec(z_aud)
    r_spec = avg_spec(r_aud)
    spec_dist = float(np.mean(np.abs(20*np.log10(z_spec+1e-10) - 20*np.log10(r_spec+1e-10))))
    print(f"  Spectral distance (zero vs random): {spec_dist:.3f} dB")

# ---------- Extreme values ----------
print("\n=== Extreme values check (random-phase generated) ===")
for name, aud in [("DREGON", gr_d), ("Michael's", gr_m)]:
    abs_vals = np.abs(aud)
    p999 = float(np.percentile(abs_vals, 99.9))
    p9999 = float(np.percentile(abs_vals, 99.99))
    peak = float(np.max(abs_vals))
    rms = float(np.sqrt(np.mean(aud**2)))
    crest = peak / max(rms, 1e-10)
    n_outliers = int(np.sum(abs_vals > 10 * rms))
    print(f"\n{name}:")
    print(f"  RMS={rms:.4f}, Peak={peak:.4f}, Crest={crest:.1f}")
    print(f"  P99.9={p999:.4f}, P99.99={p9999:.4f}")
    print(f"  Samples > 10×RMS: {n_outliers} / {aud.size} ({100*n_outliers/aud.size:.4f}%)")

    # Check for NaN or inf directly
    has_nan = bool(np.any(np.isnan(aud)))
    has_inf = bool(np.any(np.isinf(aud)))
    print(f"  NaN present: {has_nan}, Inf present: {has_inf}")

    # Per-sample RMS variability
    per_sample_rms = np.sqrt(np.mean(aud**2, axis=(-2,-1)))
    print(f"  Per-sample RMS CV: {per_sample_rms.std()/max(per_sample_rms.mean(),1e-10):.3f}")

# ---------- Check what the RPS predictor actually sees ----------
print("\n=== What the RPS predictor sees ===")
# The online mixer normalizes? Let's check the values that would go into the network
# Without normalization, the raw audio RMS for generated vs real:
# Real DREGON RMS from previous run: ~0.05 (from the diagnostic stats)
# Generated DREGON RMS: ~0.03 (0.688 * 0.05 ≈ 0.034)
# But the issue is more about the training loss — let's check if there's a huge
# variance difference that could cause gradient explosion

for name, aud in [("DREGON", gr_d), ("Michael's", gr_m)]:
    per_sample_rms = np.sqrt(np.mean(aud**2, axis=(-2,-1)))
    print(f"  {name}: per-sample RMS mean={per_sample_rms.mean():.4f}, min={per_sample_rms.min():.4f}, max={per_sample_rms.max():.4f}")

print("\nDone.")
