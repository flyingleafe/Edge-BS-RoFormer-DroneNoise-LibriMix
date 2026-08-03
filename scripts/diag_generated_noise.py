#!/usr/bin/env python3
"""
Diagnostic: compare generated noise to real DREGON/Michael's recordings.

Checks:
  1. Amplitude distribution (RMS, peak, histogram)
  2. Spectral shape (average magnitude spectrum)
  3. RPS trajectory statistics (generated only, compared to real via simple
     mean/std stats from the TimeFrame metadata)
"""

# Diagnostic script: imports follow matplotlib-backend selection + module
# constants, and use loosely-typed OmegaConf / tdseries APIs — exempt from the
# import-order and dynamic-attribute lints.
# ruff: noqa: E402, SIM102
# pyright: reportAttributeAccessIssue=false, reportCallIssue=false, reportArgumentType=false
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
SR = 16000
DUR_S = 1.0
N_SAMPLES = 200
N_HARM = 100
DEVICE = "cuda:0"

# ---------------------------------------------------------------------------
# 1. Load noise-gen model
# ---------------------------------------------------------------------------
from data_processing import rps_synthesis
from data_processing.sources.dregon import get_geometry as dregon_geom
from data_processing.sources.dregon import load_dregon_timeframes
from data_processing.sources.michaels import get_geometry as michaels_geom
from data_processing.sources.michaels import load_michaels_timeframes
from models.generative import PositionalHarmonicNoiseGen
from tasks.noise_generation import DroneCodebook, geometry_to_rel_pos

CKPT = "/gpfs/scratch/acw592/results/noise_gen_sweep/baseline/best_positional_harmonic_gen.pt"
print(f"Loading checkpoint: {CKPT}")
bundle = torch.load(CKPT, map_location=DEVICE, weights_only=False)
cond_dim = int(bundle["cond_dim"])
model = PositionalHarmonicNoiseGen(
    sample_rate=SR, n_harmonics=N_HARM, use_diff_noise=True, cond_dim=cond_dim
)
model.load_state_dict(bundle["model"])
model.to(DEVICE).eval()
names = list(bundle["drone_names"])
print(f"Drone names: {names}")
codebook = DroneCodebook(cond_dim, names=names).to(DEVICE)
codebook.load_state_dict(bundle["codebook"])

_DRONE_PROFILE_BLEND = {"dregon": 0.0, "michaels": 1.0}


def generate_batch(drone: str, n: int) -> dict:
    blend = _DRONE_PROFILE_BLEND.get(drone, 0.0)
    rng = np.random.default_rng(42)
    rps_np = rps_synthesis.generate_intermittent_batch(
        n, DUR_S, SR, drone_profile=blend, aggressiveness=1.0, rng=rng
    )

    if drone == "michaels":
        mic_pos, rotor_pos = michaels_geom()
    else:
        mic_pos, rotor_pos = dregon_geom(Path("/gpfs/scratch/acw592/data/DREGON"))
    mic_pos = np.array(mic_pos)
    rotor_pos = np.array(rotor_pos)

    rel = torch.from_numpy(geometry_to_rel_pos(mic_pos, rotor_pos)).float().to(DEVICE)
    rel_b = rel.unsqueeze(0).expand(n, -1, -1, -1)
    z = codebook([drone] * n).to(DEVICE)
    rps_t = torch.from_numpy(rps_np).float().to(DEVICE)

    with torch.no_grad():
        audio = model(rps_t, rel_b, z).cpu().numpy()  # (n, M, T)
    return {"audio": audio, "rps": rps_np}


print("\n=== Generating ===")
gen_d = generate_batch("dregon", N_SAMPLES)
gen_m = generate_batch("michaels", N_SAMPLES)
print(f"DREGON: audio={gen_d['audio'].shape}, rps={gen_d['rps'].shape}")
print(f"Michael's: audio={gen_m['audio'].shape}, rps={gen_m['rps'].shape}")

# ---------------------------------------------------------------------------
# 2. Load real data (simple: extract raw audio from TimeFrames)
# ---------------------------------------------------------------------------
DATA = Path("/gpfs/scratch/acw592/data")
real_tfs_d = load_dregon_timeframes(DATA, splits=["in_flight_noise"], target_sr=SR, download=False)
real_tfs_m = load_michaels_timeframes(DATA, sr=SR)
print(f"\nReal DREGON tfs: {len(real_tfs_d)}, Michael's tfs: {len(real_tfs_m)}")


def extract_real_audio(tfs, n: int, min_rps: float = 30.0) -> np.ndarray:
    """Extract n random 1-second audio chunks from TimeFrames."""
    rng = np.random.default_rng(123)
    chunks = []
    for tf in tfs:
        if "audio" not in tf.tracks:
            continue
        audio = tf.tracks["audio"].samples  # (T,) or (C, T)
        if audio.ndim == 1:
            audio = audio[np.newaxis, :]
        # Skip low-RPS
        rps_track = tf.tracks.get("rps") or tf.tracks.get("motors_measured")
        if rps_track is not None:
            if rps_track.values is not None:
                mean_rps = float(np.mean(rps_track.values))
                if mean_rps < min_rps:
                    continue
        total = audio.shape[-1]
        n_possible = total // (SR * DUR_S)
        for _ in range(min(n_possible, 5)):
            if len(chunks) >= n:
                break
            start = int(rng.integers(0, max(1, total - SR * DUR_S)))
            chunks.append(audio[..., start : start + int(SR * DUR_S)])
        if len(chunks) >= n:
            break
    return np.stack(chunks[:n])


real_d_audio = extract_real_audio(real_tfs_d, N_SAMPLES)
real_m_audio = extract_real_audio(real_tfs_m, N_SAMPLES, min_rps=0.0)
print(f"Real DREGON audio: {real_d_audio.shape}")
print(f"Real Michael's audio: {real_m_audio.shape}")


# ---------------------------------------------------------------------------
# 3. Amplitude statistics
# ---------------------------------------------------------------------------
def amp_stats(audio, label):
    rms = float(np.sqrt(np.mean(audio**2)))
    peak = float(np.max(np.abs(audio)))
    rms_per_ch = np.sqrt(np.mean(audio**2, axis=-1)).mean(axis=0)  # per-channel mean
    crest = peak / max(rms, 1e-10)
    return {
        "label": label,
        "rms": rms,
        "peak": peak,
        "crest": crest,
        "rms_per_ch_min": float(rms_per_ch.min()),
        "rms_per_ch_max": float(rms_per_ch.max()),
    }


amp = [
    amp_stats(gen_d["audio"], "gen_dregon"),
    amp_stats(real_d_audio, "real_dregon"),
    amp_stats(gen_m["audio"], "gen_michaels"),
    amp_stats(real_m_audio, "real_michaels"),
]

print("\n=== AMPLITUDE ===")
print(f"{'Source':<20} {'RMS':>10} {'Peak':>10} {'Crest':>8} {'RMS_ch_min':>10} {'RMS_ch_max':>10}")
for a in amp:
    print(
        f"{a['label']:<20} {a['rms']:>10.4f} {a['peak']:>10.4f} {a['crest']:>8.2f} {a['rms_per_ch_min']:>10.4f} {a['rms_per_ch_max']:>10.4f}"
    )

print("\n=== RATIOS ===")
for gen_key, real_key, name in [
    ("gen_dregon", "real_dregon", "DREGON"),
    ("gen_michaels", "real_michaels", "Michael's"),
]:
    gen_val = next(a for a in amp if a["label"] == gen_key)
    real_val = next(a for a in amp if a["label"] == real_key)
    print(
        f"  {name}: gen/real RMS = {gen_val['rms'] / max(real_val['rms'], 1e-10):.3f}, "
        f"gen/real Peak = {gen_val['peak'] / max(real_val['peak'], 1e-10):.3f}"
    )


# ---------------------------------------------------------------------------
# 4. RPS statistics (generated only vs simple real stats)
# ---------------------------------------------------------------------------
def rps_stats(rps_arr, label):
    pooled = rps_arr.reshape(-1, rps_arr.shape[-1])  # (N*R, T)
    mean = float(pooled.mean())
    std = float(pooled.std())
    p5 = float(np.percentile(pooled, 5))
    p95 = float(np.percentile(pooled, 95))
    return {"label": label, "mean": mean, "std": std, "p5": p5, "p95": p95}


# For real data, just print available RPS ranges
print("\n=== RPS ===")
print(f"{'Source':<20} {'Mean':>10} {'Std':>10} {'P5':>10} {'P95':>10}")
for rps, label in [(gen_d["rps"], "gen_dregon"), (gen_m["rps"], "gen_michaels")]:
    s = rps_stats(rps, label)
    print(
        f"{s['label']:<20} {s['mean']:>10.2f} {s['std']:>10.2f} {s['p5']:>10.2f} {s['p95']:>10.2f}"
    )

print("\n  Real DREGON RPS ranges (from TimeFrame metadata):")
for tf in real_tfs_d:
    rps = tf.tracks.get("rps") or tf.tracks.get("motors_measured")
    if rps is not None and rps.values is not None:
        v = rps.values
        print(
            f"    Mean={np.mean(v):.1f}, std={np.std(v):.1f}, min={np.min(v):.1f}, max={np.max(v):.1f}"
        )

print("\n  Real Michael's RPS ranges:")
for tf in real_tfs_m:
    rps = tf.tracks.get("rps") or tf.tracks.get("motors_measured")
    if rps is not None and rps.values is not None:
        v = rps.values
        print(
            f"    Mean={np.mean(v):.1f}, std={np.std(v):.1f}, min={np.min(v):.1f}, max={np.max(v):.1f}"
        )


# ---------------------------------------------------------------------------
# 5. Spectral comparison
# ---------------------------------------------------------------------------
def avg_spectrum(audio, n_fft=2048, hop=512, max_samples=100):
    all_specs = []
    n = min(audio.shape[0], max_samples)
    for i in range(n):
        for c in range(audio.shape[1]):
            x = torch.from_numpy(audio[i, c]).float()
            spec = torch.stft(
                x, n_fft=n_fft, hop_length=hop, window=torch.hann_window(n_fft), return_complex=True
            )
            mag = spec.abs().mean(dim=-1).numpy()
            all_specs.append(mag)
    return np.mean(all_specs, axis=0)


print("\n=== SPECTRA ===")
specs = {}
for key, aud in [
    ("gen_dregon", gen_d["audio"]),
    ("real_dregon", real_d_audio),
    ("gen_michaels", gen_m["audio"]),
    ("real_michaels", real_m_audio),
]:
    specs[key] = avg_spectrum(aud)
    print(f"  {key}: shape={specs[key].shape}")


def spectral_distance(sg, sr):
    lg = 20 * np.log10(sg + 1e-10)
    lr = 20 * np.log10(sr + 1e-10)
    return float(np.mean(np.abs(lg - lr)))


print("\n=== SPECTRAL DISTANCE (dB) ===")
for gn, rn, nm in [
    ("gen_dregon", "real_dregon", "DREGON"),
    ("gen_michaels", "real_michaels", "Michael's"),
]:
    d = spectral_distance(specs[gn], specs[rn])
    print(f"  {nm}: {d:.2f} dB")

# Also check overall RMS per frequency
for gn, rn, nm in [
    ("gen_dregon", "real_dregon", "DREGON"),
    ("gen_michaels", "real_michaels", "Michael's"),
]:
    # Energy-weighted mean frequency
    freqs = np.linspace(0, SR / 2, len(specs[gn]))
    gen_energy = specs[gn] ** 2
    real_energy = specs[rn] ** 2
    gen_centroid = np.sum(freqs * gen_energy) / max(np.sum(gen_energy), 1e-10)
    real_centroid = np.sum(freqs * real_energy) / max(np.sum(real_energy), 1e-10)
    print(f"  {nm} spectral centroid: gen={gen_centroid:.1f} Hz, real={real_centroid:.1f} Hz")

# ---------------------------------------------------------------------------
# 6. Plots
# ---------------------------------------------------------------------------
out_dir = Path("/gpfs/scratch/acw592/results/noise_gen_diag")
out_dir.mkdir(parents=True, exist_ok=True)

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Row 0: amplitude histograms
for col, (gen_aud, real_aud, title) in enumerate(
    [
        (gen_d["audio"], real_d_audio, "DREGON"),
        (gen_m["audio"], real_m_audio, "Michael's"),
        (None, None, "Combined"),
    ]
):
    ax = axes[0, col]
    if gen_aud is not None:
        ax.hist(np.abs(gen_aud).flatten(), bins=200, alpha=0.5, density=True, label="Generated")
        ax.hist(np.abs(real_aud).flatten(), bins=200, alpha=0.5, density=True, label="Real")
        ax.set_xlabel("|amplitude|")
        ax.set_title(f"{title}: amplitude dist")
        ax.legend(fontsize=8)
    else:
        all_gen = np.abs(np.concatenate([gen_d["audio"].flatten(), gen_m["audio"].flatten()]))
        all_real = np.abs(np.concatenate([real_d_audio.flatten(), real_m_audio.flatten()]))
        ax.hist(all_gen, bins=200, alpha=0.5, density=True, label="Generated (all)")
        ax.hist(all_real, bins=200, alpha=0.5, density=True, label="Real (all)")
        ax.set_xlabel("|amplitude|")
        ax.set_title("Combined: amplitude dist")
        ax.legend(fontsize=8)

# Row 1: average spectra
freqs = np.linspace(0, SR / 2, len(specs["gen_dregon"]))
for col, (gk, rk, title) in enumerate(
    [
        ("gen_dregon", "real_dregon", "DREGON"),
        ("gen_michaels", "real_michaels", "Michael's"),
        (None, None, "Combined"),
    ]
):
    ax = axes[1, col]
    if gk is not None:
        ax.semilogy(freqs, specs[gk], alpha=0.7, linewidth=1, label="Generated")
        ax.semilogy(freqs, specs[rk], alpha=0.7, linewidth=1, label="Real")
        ax.set_xlabel("Freq (Hz)")
        ax.set_ylabel("Magnitude")
        ax.set_title(f"{title}: avg spectrum")
        ax.legend(fontsize=8)
    else:
        # Combined: average across all
        gen_all = np.mean([specs["gen_dregon"], specs["gen_michaels"]], axis=0)
        real_all = np.mean([specs["real_dregon"], specs["real_michaels"]], axis=0)
        ax.semilogy(freqs, gen_all, alpha=0.7, linewidth=1, label="Generated (avg)")
        ax.semilogy(freqs, real_all, alpha=0.7, linewidth=1, label="Real (avg)")
        ax.set_xlabel("Freq (Hz)")
        ax.set_ylabel("Magnitude")
        ax.set_title("Combined: avg spectrum")
        ax.legend(fontsize=8)

plt.tight_layout()
fig.savefig(out_dir / "diagnostic.png", dpi=150)
print(f"\nPlot saved to {out_dir / 'diagnostic.png'}")

# Save stats
with open(out_dir / "stats.txt", "w") as f:
    f.write("=== AMPLITUDE ===\n")
    f.write(f"{'Source':<20} {'RMS':>10} {'Peak':>10} {'Crest':>8}\n")
    for a in amp:
        f.write(f"{a['label']:<20} {a['rms']:>10.4f} {a['peak']:>10.4f} {a['crest']:>8.2f}\n")
    f.write("\n=== RPS ===\n")
    for rps, label in [(gen_d["rps"], "gen_dregon"), (gen_m["rps"], "gen_michaels")]:
        s = rps_stats(rps, label)
        f.write(
            f"{s['label']}: mean={s['mean']:.1f} std={s['std']:.1f} p5={s['p5']:.1f} p95={s['p95']:.1f}\n"
        )
    f.write("\n=== SPECTRAL DISTANCE ===\n")
    for gn, rn, nm in [
        ("gen_dregon", "real_dregon", "DREGON"),
        ("gen_michaels", "real_michaels", "Michael's"),
    ]:
        d = spectral_distance(specs[gn], specs[rn])
        f.write(f"{nm}: {d:.2f} dB\n")

print("Done.")
