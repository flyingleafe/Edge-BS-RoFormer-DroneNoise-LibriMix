#!/usr/bin/env python3
"""
Compute spectrum comparison: BEMT vs VLM vs real DREGON motor recordings.

For Task 1: compares spectral shape between BEMT and VLM simulations vs real
DREGON recordings across 12 motor/RPM combinations (Motors 1-4 × RPMs 50/70/90).

For Task 2: benchmarks BEMT vs VLM wall-clock time per call on single GPU.

Outputs:
  results/spectrum_comparison.json
  figures/fig_spectrum_comparison_m1r90_mic0.pdf
  figures/fig_real_motor1_rpms.pdf
"""
import os, sys, json, time
from pathlib import Path

import numpy as np
import scipy.signal
import soundfile as sf
import torch

# ── Project bootstrap ──────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from fwh_rotor_sim import Blade, Rotor, FWHRotorSolver
from fwh_rotor_sim.vlm import VortexLatticeSolver

# ── Parameters ────────────────────────────────────────────────────────────────
# APC 10×4.5 Thin Electric approximation (DREGON motor propeller)
CHORD_LERP = [
    (0.013, 0.010), (0.021, 0.030), (0.024, 0.030),
    (0.022, 0.020), (0.015, 0.005),
]
CHORD_R_FRACS = [0.0, 0.25, 0.50, 0.75, 1.0]
RADIUS     = 0.152
HUB_RADIUS = 0.02
NUM_BLADES = 2
NUM_RADIAL = 30
NFFT      = 4096
HOP       = 512
TARGET_SR = 16000


# ── Spectrum tools ────────────────────────────────────────────────────────────
def compute_spectrum(audio: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    """Averaged PSD (dB FS/Hz). Handles both real recordings (44.1 kHz) and
    low-rate simulations (~40 Hz at 200 samples / 5 s).

    Simulated signals are upsampled to NFFT×2 minimum before STFT so that
    frequency resolution is meaningful even at 40 Hz effective sample rate.
    """
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)   # average across DREGON 8 mics
    audio = audio.astype(np.float32)

    # Upsample if signal is shorter than NFFT (low-rate simulation case)
    if len(audio) < NFFT * 2:
        audio = scipy.signal.resample(audio, max(NFFT * 2, len(audio)))
    elif sr != TARGET_SR:
        # High-rate real recordings: resample to TARGET_SR
        audio = scipy.signal.resample_poly(audio, TARGET_SR, sr)

    win = scipy.signal.windows.hann(NFFT)
    f, _, Zxx = scipy.signal.stft(
        audio, fs=TARGET_SR, nperseg=NFFT, noverlap=NFFT - HOP, window=win)
    psd = np.mean(np.abs(Zxx) ** 2, axis=1)
    pdb = 10 * np.log10(psd + 1e-15)
    return f, pdb


def spectral_distance(freq1, psd1, freq2, psd2,
                      f_min=50.0, f_max=4000.0) -> dict:
    """Shape-only distance between two PSDs in [f_min, f_max] Hz.

    Both PSDs are recentred to zero mean before comparison so that absolute
    level differences (real recording in V vs simulated pressure in Pa)
    are discounted — only spectral SHAPE is compared.
    """
    f_common = np.linspace(f_min, f_max, 2048)
    p1 = np.interp(f_common, freq1, psd1, left=-120, right=-120)
    p2 = np.interp(f_common, freq2, psd2, left=-120, right=-120)

    p1n = p1 - np.mean(p1)
    p2n = p2 - np.mean(p2)

    mse_db = float(np.mean((p1n - p2n) ** 2))
    denom  = np.linalg.norm(p1n) * np.linalg.norm(p2n) + 1e-20
    corr   = float(np.dot(p1n, p2n) / denom)

    return {"mse_db": mse_db, "corr": corr, "rms_db": float(np.sqrt(mse_db))}


# ── Blade factory ─────────────────────────────────────────────────────────────
def make_blade():
    f_abs = [f * RADIUS for f in CHORD_R_FRACS]
    c_src = [c for _, c in CHORD_LERP]

    def c_fn(r):
        r_np = r.detach().cpu().numpy() if hasattr(r, "detach") else np.asarray(r)
        return torch.from_numpy(np.interp(r_np, f_abs, c_src)).to(
            r.dtype if hasattr(r, "dtype") else torch.float32)

    def t_fn(r):
        rv = r.detach().cpu().numpy() if hasattr(r, "detach") else np.asarray(r)
        return torch.tensor([15.0 * (1.0 - x / RADIUS) for x in rv],
                            dtype=r.dtype if hasattr(r, "dtype") else torch.float32)

    return Blade(radius=RADIUS, chord=c_fn, twist_deg=t_fn,
                 hub_radius=HUB_RADIUS, n_radial=NUM_RADIAL)


# ── Simulation ────────────────────────────────────────────────────────────────
def simulate(method: str, blade,
             x_obs: torch.Tensor, Omega: float,
             T: float = 5.0, n_t: int = 200) -> np.ndarray:
    """Simulate pressure via BEMT or VLM.

    n_t=200 (5 s at 40 Hz effective) is sufficient for spectral comparison:
    upsampling in compute_spectrum() maps this to 16 kHz for STFT,
    giving freq resolution ~3.9 Hz/bin — enough to resolve BPF harmonics.
    """
    t = torch.linspace(0, T, n_t)
    if method == "BEMT":
        solver = FWHRotorSolver(Rotor(blade=blade, num_blades=NUM_BLADES),
                               c0=343.0, rho0=1.225)
        with torch.no_grad():
            p = solver.compute_pressure(x_obs, t, torch.tensor(Omega))
    elif method == "VLM":
        vlm = VortexLatticeSolver(blade=blade, num_blades=NUM_BLADES)
        with torch.no_grad():
            Gamma, F_norm, F_tang = vlm.compute_circulation(torch.tensor(Omega))
            p = vlm.compute_pressure(x_obs, t, torch.tensor(Omega),
                                      Gamma, F_norm, F_tang)
    return p.cpu().numpy()


# ── Speed benchmark ───────────────────────────────────────────────────────────
def benchmark(blade, x_obs: torch.Tensor, n_runs: int = 50) -> dict:
    """Wall-clock time for one BEMT vs VLM call on single GPU."""
    Omega = 2 * np.pi * 80.0
    t = torch.linspace(0, 5.0, 200)
    x_o = x_obs.to(torch.float32)

    solver = FWHRotorSolver(Rotor(blade=blade, num_blades=NUM_BLADES),
                            c0=343.0, rho0=1.225)
    t0 = time.perf_counter()
    for _ in range(n_runs):
        solver.compute_pressure(x_o, t, torch.tensor(Omega))
    t_b = (time.perf_counter() - t0) / n_runs

    vlm = VortexLatticeSolver(blade=blade, num_blades=NUM_BLADES)
    Gamma, F_norm, F_tang = vlm.compute_circulation(torch.tensor(Omega))
    t0 = time.perf_counter()
    for _ in range(n_runs):
        vlm.compute_pressure(x_o, t, torch.tensor(Omega), Gamma, F_norm, F_tang)
    t_v = (time.perf_counter() - t0) / n_runs

    return {
        "bemt_s":  float(t_b),
        "vlm_s":   float(t_v),
        "ratio":   float(t_v / t_b),
        "bemt_hz": float(1.0 / t_b),
        "vlm_hz":  float(1.0 / t_v),
        "n_runs":  n_runs,
    }


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "serif", "font.size": 9,
        "axes.titlesize": 9, "axes.labelsize": 9,
        "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 150,
        "savefig.pad_inches": 0.02,
    })

    RESULT_DIR = Path(__file__).resolve().parents[1] / "results"
    RESULT_DIR.mkdir(exist_ok=True)
    FIG_DIR = Path(__file__).resolve().parents[2] / "papers" / "physical-modeling-report" / "figures"
    FIG_DIR.mkdir(exist_ok=True)

    DREGON_BASE = (ROOT /
                   "data/DREGON/DREGON_individual_motors_recordings/"
                   "DREGON_individual_motors_recordings")
    MOTOR_FILES = [(m, rpm) for m in range(1, 5) for rpm in [50, 70, 90]]

    # DREGON mic 0 position (representative; all 8 mics are within 0.2 m)
    mic0 = np.array([0.0420, 0.0615, -0.0410], dtype=np.float32)

    blade = make_blade()
    x_o  = torch.tensor(mic0, dtype=torch.float32)
    rows = []

    # ════════════════════════════════════════════════════════════════════════════
    # Task 1 — Spectrum comparison
    # ════════════════════════════════════════════════════════════════════════════
    print("\n=== Task 1: Spectrum comparison (BEMT / VLM vs real DREGON) ===\n")
    for motor, rpm in sorted(MOTOR_FILES):
        path = DREGON_BASE / f"Motor{motor}_{rpm}.wav"
        if not path.exists():
            print(f"  SKIP {path}")
            continue

        audio, sr = sf.read(str(path))
        clip = audio[-5 * sr:]          # last 5 s — near-steady-state
        rps   = float(rpm)
        Omega = 2 * np.pi * rps
        bpf   = rps * NUM_BLADES

        spec_real = compute_spectrum(clip, sr)
        p_b = simulate("BEMT", blade, x_o, Omega)
        p_v = simulate("VLM",  blade, x_o, Omega)
        sp_b = compute_spectrum(p_b, TARGET_SR)
        sp_v = compute_spectrum(p_v,  TARGET_SR)
        d_b = spectral_distance(*sp_b, *spec_real)
        d_v = spectral_distance(*sp_v, *spec_real)

        entry = {
            "motor": motor, "rpm": rpm, "rps": rps, "bpf": float(bpf),
            "bemt": d_b, "vlm": d_v,
            "bemt_mse_mean":  d_b["mse_db"],
            "vlm_mse_mean":   d_v["mse_db"],
            "bemt_corr_mean": d_b["corr"],
            "vlm_corr_mean":  d_v["corr"],
            "improvement":    d_b["mse_db"] - d_v["mse_db"],
            "spectra": {
                "real_bpf": float(bpf),
                "real": (spec_real[0].tolist(), spec_real[1].tolist()),
                "bemt":  (sp_b[0].tolist(), sp_b[1].tolist()),
                "vlm":   (sp_v[0].tolist(), sp_v[1].tolist()),
            },
        }
        rows.append(entry)
        print(f"  Motor{motor}_{rpm}  BPF={bpf:.0f} Hz  "
              f"BEMT mse={d_b['mse_db']:.2f} (r={d_b['corr']:.3f})  "
              f"VLM mse={d_v['mse_db']:.2f}  (r={d_v['corr']:.3f})  "
              f"Δ={entry['improvement']:+.2f}")

        # ── Figure: Motor1 @ 90 RPM ───────────────────────────────────────
        if motor == 1 and rpm == 90:
            fig, axes = plt.subplots(2, 1, figsize=(6.8, 4.5),
                                    gridspec_kw={"height_ratios": [1.2, 1.0],
                                                 "hspace": 0.35})
            for ax, xlim, title in zip(
                    axes,
                    [(50, 4000), (50, 2000)],
                    [f"Motor1_90 @ mic0 — spectrum, BPF = {bpf:.0f} Hz",
                     "zoomed 0–2 kHz (blade harmonics marked)"]):
                ax.plot(spec_real[0], spec_real[1],  "-",  lw=0.8, color="gray",
                       alpha=0.85, label="real recording")
                ax.plot(sp_b[0],   sp_b[1],          "--", lw=0.8, color="#1f77b4",
                       label="BEMT")
                ax.plot(sp_v[0],   sp_v[1],           ":",  lw=0.8, color="#ff7f0e",
                       label="VLM")
                for h in [1, 2, 3, 4]:
                    ax.axvline(bpf * h, ls=":", lw=0.5, color="#555", alpha=0.45)
                ax.set_xlim(xlim)
                ax.set_xlabel("frequency [Hz]")
                ax.set_ylabel("PSD [dB FS/Hz]")
                ax.set_title(title)
                ax.legend(frameon=False)

            plt.savefig(FIG_DIR / "fig_spectrum_comparison_m1r90_mic0.pdf",
                        bbox_inches="tight")
            plt.close()
            print(f"    → fig_spectrum_comparison_m1r90_mic0.pdf")

        # ── Figure: real motor spectra across 3 RPMs ──────────────────────
        if motor == 1 and rpm == 90:
            fig, ax = plt.subplots(figsize=(6.8, 2.5))
            for rpm_x, col in [(50, "#1f77b4"), (70, "#ff7f0e"), (90, "#d62728")]:
                fx = DREGON_BASE / f"Motor1_{rpm_x}.wav"
                ax_r, sr_r = sf.read(str(fx))
                sp = compute_spectrum(ax_r[-5 * sr_r:], sr_r)
                ax.plot(sp[0], sp[1], "-", color=col, lw=0.8,
                        label=f"real Motor1_{rpm_x}")
            ax.set_xlim(50, 4000)
            ax.set_xlabel("frequency [Hz]")
            ax.set_ylabel("PSD [dB FS/Hz]")
            ax.set_title("DREGON real recordings — Motor 1 — RPMs 50 / 70 / 90")
            ax.legend(frameon=False, ncol=3)
            plt.savefig(FIG_DIR / "fig_real_motor1_rpms.pdf", bbox_inches="tight")
            plt.close()
            print(f"    → fig_real_motor1_rpms.pdf")

    # ── Aggregation ────────────────────────────────────────────────────────────
    agg_by_rpm = {}
    for rpm in [50, 70, 90]:
        sub = [r for r in rows if r["rpm"] == rpm]
        if not sub:
            continue
        agg_by_rpm[rpm] = {
            "n": len(sub),
            "bemt_mse": {"mean": float(np.mean([r["bemt_mse_mean"]  for r in sub])),
                         "std":  float(np.std( [r["bemt_mse_mean"]  for r in sub]))},
            "vlm_mse":  {"mean": float(np.mean([r["vlm_mse_mean"]   for r in sub])),
                         "std":  float(np.std( [r["vlm_mse_mean"]   for r in sub]))},
            "bemt_corr":{"mean": float(np.mean([r["bemt_corr_mean"] for r in sub])),
                         "std":  float(np.std( [r["bemt_corr_mean"] for r in sub]))},
            "vlm_corr": {"mean": float(np.mean([r["vlm_corr_mean"]  for r in sub])),
                         "std":  float(np.std( [r["vlm_corr_mean"]  for r in sub]))},
            "improvement": float(
                np.mean([r["bemt_mse_mean"] for r in sub]) -
                np.mean([r["vlm_mse_mean"]  for r in sub])),
        }

    all_b  = [r["bemt_mse_mean"]  for r in rows]
    all_v  = [r["vlm_mse_mean"]   for r in rows]
    all_bc = [r["bemt_corr_mean"] for r in rows]
    all_vc = [r["vlm_corr_mean"]  for r in rows]
    agg_ov = {
        "n_total":    len(rows),
        "bemt_mse":   {"mean": float(np.mean(all_b)),  "std": float(np.std(all_b))},
        "vlm_mse":    {"mean": float(np.mean(all_v)),  "std": float(np.std(all_v))},
        "bemt_corr":  {"mean": float(np.mean(all_bc)), "std": float(np.std(all_bc))},
        "vlm_corr":   {"mean": float(np.mean(all_vc)), "std": float(np.std(all_vc))},
        "improvement": float(np.mean(all_b) - np.mean(all_v)),
    }

    # ════════════════════════════════════════════════════════════════════════════
    # Task 2 — Speed benchmark
    # ═════════════════════════════════════════════════════════════════════════════
    print("\n=== Task 2: Speed benchmark (single GPU) ===\n")
    timing = benchmark(blade, x_o, n_runs=50)
    print(f"  BEMT: {timing['bemt_s']*1000:.1f} ms/call  ({timing['bemt_hz']:.1f} calls/s)")
    print(f"  VLM:  {timing['vlm_s']*1000:.1f} ms/call  ({timing['vlm_hz']:.1f} calls/s)")
    print(f"  Ratio VLM/BEMT = {timing['ratio']:.2f}×  "
          f"({'faster' if timing['ratio'] < 1 else 'slower'})")

    # ── Save ──────────────────────────────────────────────────────────────────
    out = {
        "individual": rows,
        "agg_by_rpm": agg_by_rpm,
        "agg_overall": agg_ov,
        "timing":     timing,
    }
    with open(RESULT_DIR / "spectrum_comparison.json", "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\nResults → {RESULT_DIR / 'spectrum_comparison.json'}")

    # ── Summary table ───────────────────────────────────────────────────────
    print("\n=== Spectral distance summary ===")
    print(f"{'Quantifier':>16} | {'n':>3} | {'BEMT (mse / r)':>18} | {'VLM (mse / r)':>17} | {'Improvement':>12}")
    print("-" * 80)
    for rpm, s in sorted(agg_by_rpm.items()):
        print(f"RPM {rpm:>9} | {s['n']:>3} | "
              f"{s['bemt_mse']['mean']:>7.2f} / {s['bemt_corr']['mean']:.3f}"
              f" | {s['vlm_mse']['mean']:>7.2f} / {s['vlm_corr']['mean']:.3f}"
              f" | {s['improvement']:>+11.2f}")
    print("-" * 80)
    print(f"{'Overall':>16} | {agg_ov['n_total']:>3} | "
          f"{agg_ov['bemt_mse']['mean']:>7.2f} / {agg_ov['bemt_corr']['mean']:.3f}"
          f" | {agg_ov['vlm_mse']['mean']:>7.2f} / {agg_ov['vlm_corr']['mean']:.3f}"
          f" | {agg_ov['improvement']:>+11.2f}")

    print(f"\nSpeed (single GPU):")
    print(f"  BEMT: {timing['bemt_s']*1000:.1f} ms/call  ({timing['bemt_hz']:.1f} calls/s)")
    print(f"  VLM:  {timing['vlm_s']*1000:.1f} ms/call  ({timing['vlm_hz']:.1f} calls/s)")
    print(f"  VLM/BEMT = {timing['ratio']:.2f}×")

    verdict = "VLM wins on spectral accuracy" if agg_ov["improvement"] > 0 else "BEMT wins on spectral accuracy"
    print(f"\nConclusion: {verdict}: Δ = {agg_ov['improvement']:+.2f} dB "
          f"(BEMT r={agg_ov['bemt_corr']['mean']:.3f}, VLM r={agg_ov['vlm_corr']['mean']:.3f}).")

    return out


if __name__ == "__main__":
    main()
