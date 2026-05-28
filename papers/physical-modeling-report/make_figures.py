#!/usr/bin/env python3
"""
Generate figures for the BEMT vs VLM physical modeling report.

Reads pre-computed results from results/spectrum_comparison.json and
regenerates all PDF figures. Run from project root:

    python papers/physical-modeling-report/make_figures.py

Figures produced:
  fig_spectra_all_bpfs.pdf    — spectra for all 12 motor/RPM combos
  fig_mse_by_rpm.pdf          — bar chart: BEMT/VLM MSE by RPM
  fig_correlation_by_rpm.pdf  — bar chart: spectral correlation by RPM
  fig_speed_benchmark.pdf     — timing comparison BEMT vs VLM
"""
import json, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

plt.rcParams.update({
    "font.family": "serif", "font.size": 9,
    "axes.titlesize": 9, "axes.labelsize": 9,
    "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3,
    "figure.dpi": 150, "savefig.pad_inches": 0.02,
})

FIG_DIR    = ROOT / "papers" / "physical-modeling-report" / "figures"
JSON_PATH  = ROOT / "papers" / "physical-modeling-report" / "results" / "spectrum_comparison.json"


def plot_spectra_all(data, agg_by_rpm):
    """Grid of spectra: rows = RPMs, cols = motors."""
    motors = sorted(set(r["motor"] for r in data))
    rpms   = sorted(set(r["rpm"]  for r in data))

    fig, axes = plt.subplots(len(rpms), len(motors), figsize=(7.1, 5.5),
                             sharex=True, sharey="row",
                             gridspec_kw={"hspace": 0.35, "wspace": 0.25})
    if len(rpms) == 1:
        axes = np.array([axes])
    if len(motors) == 1:
        axes = axes.reshape(-1, 1)

    # Shared y-range across rows
    y_min, y_max = -5, 5  # normalised PSD centred at 0

    for row, rpm in enumerate(rpms):
        for col, motor in enumerate(motors):
            ax = axes[row, col]
            entry = next(
                (r for r in data if r["motor"] == motor and r["rpm"] == rpm), None)
            if entry is None:
                ax.set_visible(False)
                continue

            spec = entry["spectra"]
            bpf  = spec["real_bpf"]

            ax.plot(*spec["real"], "-",  lw=0.6, color="gray",    alpha=0.70, label="real")
            ax.plot(*spec["bemt"], "--", lw=0.6, color="#1f77b4", label="BEMT")
            ax.plot(*spec["vlm"],  ":",  lw=0.6, color="#ff7f0e", label="VLM")
            for h in [1, 2, 3, 4]:
                ax.axvline(bpf * h, ls=":", lw=0.5, color="#aaa", alpha=0.5)
            ax.set_xlim(50, 4000)
            ax.set_title(f"Motor{motor} @ {rpm} RPM  (BPF={bpf:.0f} Hz)",
                        fontsize=7.5, color="#333")
            if row == len(rpms) - 1:
                ax.set_xlabel("freq [Hz]", fontsize=8)
            if col == 0:
                ax.set_ylabel("PSD [dB]", fontsize=8)

    handles = [
        plt.Line2D([0], [0], color="gray",    ls="-",  lw=1,   label="real recording"),
        plt.Line2D([0], [0], color="#1f77b4", ls="--", lw=1,   label="BEMT"),
        plt.Line2D([0], [0], color="#ff7f0e", ls=":",  lw=1,   label="VLM"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3,
              bbox_to_anchor=(0.5, -0.02), frameon=False, fontsize=8)
    plt.suptitle("BEMT / VLM vs real spectra — all motor/RPM combinations",
                fontsize=10, y=1.01)
    out = FIG_DIR / "fig_spectra_all_bpfs.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close()
    print("Wrote", out)


def plot_mse_by_rpm(agg_by_rpm):
    """Bar chart: normalised MSE to real recording (lower = better)."""
    fig, ax = plt.subplots(figsize=(4.5, 2.8))
    rpms = [50, 70, 90]
    x    = np.arange(len(rpms))
    w    = 0.3

    def entry(r, key, stat):
        return agg_by_rpm[str(r)][key][stat]

    bemt_m = [entry(r, "bemt_mse", "mean") for r in rpms]
    bemt_s = [entry(r, "bemt_mse", "std")  for r in rpms]
    vlm_m  = [entry(r, "vlm_mse",  "mean") for r in rpms]
    vlm_s  = [entry(r, "vlm_mse",  "std")  for r in rpms]

    ax.bar(x - w/2, bemt_m, w, label="BEMT", color="#1f77b4",
           yerr=bemt_s, capsize=3, alpha=0.85)
    ax.bar(x + w/2, vlm_m,  w, label="VLM",  color="#ff7f0e",
           yerr=vlm_s,  capsize=3, alpha=0.65)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{r} RPM" for r in rpms])
    ax.set_xlabel("Rotor speed")
    ax.set_ylabel("Normalised MSE [dB]")
    ax.set_title("Spectral distance to real recording\n(lower = better match)")
    ax.legend(frameon=False)
    ax.set_ylim(30, 55)

    out = FIG_DIR / "fig_mse_by_rpm.pdf"
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close()
    print("Wrote", out)


def plot_correlation_by_rpm(agg_by_rpm):
    """Bar chart: spectral Pearson r (higher = better shape match)."""
    fig, ax = plt.subplots(figsize=(4.5, 2.8))
    rpms = [50, 70, 90]
    x    = np.arange(len(rpms))

    def entry(r, method, stat):
        return agg_by_rpm[str(r)][["bemt_corr", "vlm_corr"][["BEMT", "VLM"].index(method)]][stat]

    bemt_m = [agg_by_rpm[str(r)]["bemt_corr"]["mean"] for r in rpms]
    bemt_s = [agg_by_rpm[str(r)]["bemt_corr"]["std"]  for r in rpms]
    vlm_m  = [agg_by_rpm[str(r)]["vlm_corr"]["mean"]  for r in rpms]
    vlm_s  = [agg_by_rpm[str(r)]["vlm_corr"]["std"]   for r in rpms]

    ax.bar(x - 0.2, bemt_m, 0.4, label="BEMT", color="#1f77b4",
           yerr=bemt_s, capsize=3, alpha=0.85)
    ax.bar(x + 0.2, vlm_m,  0.4, label="VLM",  color="#ff7f0e",
           yerr=vlm_s,  capsize=3, alpha=0.65)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{r} RPM" for r in rpms])
    ax.set_xlabel("Rotor speed")
    ax.set_ylabel("Pearson r (PSD shape correlation)")
    ax.set_title("Spectral shape match to real recording\n(higher = better)")
    ax.legend(frameon=False)
    ax.set_ylim(0.68, 0.88)
    ax.axhline(0.80, ls="--", lw=0.7, color="#999", alpha=0.6)

    out = FIG_DIR / "fig_correlation_by_rpm.pdf"
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close()
    print("Wrote", out)


def plot_speed_benchmark(timing):
    """Horizontal bars: BEMT vs VLM wall-clock time (ms)."""
    fig, ax = plt.subplots(figsize=(4.5, 2.0))
    labels = ["BEMT", "VLM"]
    times  = [timing["bemt_s"] * 1000, timing["vlm_s"] * 1000]
    colors = ["#1f77b4", "#ff7f0e"]

    bars = ax.barh(labels, times, color=colors, alpha=0.85, height=0.5)
    for bar, t in zip(bars, times):
        ax.text(t + 30, bar.get_y() + bar.get_height() / 2,
                f"{t:.0f} ms", va="center", fontsize=8)

    ax.set_xlabel("Wall-clock time [ms per call]")
    ax.set_title(f"BEMT vs VLM speed benchmark (single GPU)\n"
                 f"VLM = {1 / timing['ratio']:.1f}× faster than BEMT")
    ax.set_xlim(0, None)

    out = FIG_DIR / "fig_speed_benchmark.pdf"
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close()
    print("Wrote", out)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    if not JSON_PATH.exists():
        print(f"WARNING: {JSON_PATH} not found — run compute_spectrum_comparison.py first")
        return

    with open(JSON_PATH) as f:
        results = json.load(f)

    data        = results["individual"]
    agg_by_rpm  = results.get("agg_by_rpm", {})
    agg_overall = results["agg_overall"]
    timing      = results["timing"]

    print("Generating report figures...")
    plot_spectra_all(data, agg_by_rpm)
    plot_mse_by_rpm(agg_by_rpm)
    plot_correlation_by_rpm(agg_by_rpm)
    plot_speed_benchmark(timing)
    print("Done.")


if __name__ == "__main__":
    main()
