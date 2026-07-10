#!/usr/bin/env python3
"""Generate figures for the 2026-07-06 noise-gen + GP-rotor-noise report.

Part 1 — Generated-noise augmentation (negative result, docs/experiments/
noise-generation-augmentation.md):
  assets/e3_smoothness_sweep.png       E3 2-D smoothness sweep table-as-heatmap
  assets/e4_aug_degradation.png        RPS prediction: baseline vs +generated noise
  assets/noise_gen_diagram.png         positional harmonic noise generator block diagram
                                       (copy of supervisor deck s2_model_diagram.png)
  assets/noise_gen_spec_dregon.png     spectrogram: real vs generated (DREGON)
  assets/noise_gen_spec_michaels.png   spectrogram: real vs generated (Michael's)
                                       (copied from supervisor deck s2_*)
Part 2 — GP rotor-noise model (work in progress, faithful Lee et al. 2026):
  assets/gp_overview.png               pipeline diagram (DWT -> Fourier-basis GP -> synth)
  assets/gp_v1_v2_rmse.png             generic-kernel vs faithful-Fourier-basis RMSE
  assets/gp_faithful_spectrum.png      real vs GP-generated spectra, Michael rec 1
  assets/gp_qd2026_tiers.png           QD2026 literature scan tier ranking

Run via `make figures` (sets PYTHONPATH to repo root).
"""

from __future__ import annotations

import json
import pathlib
import shutil
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle as _Rect

# ── bootstrap project paths (run from the repo root) ─────────────────────
REPORT_DIR = pathlib.Path(__file__).resolve().parent
ASSETS = REPORT_DIR / "assets"
ASSETS.mkdir(exist_ok=True)
ROOT = REPORT_DIR.parents[2]
sys.path.insert(0, str(ROOT / "src"))

SUPERVISOR_ASSETS = ROOT / "writing/slides/2026-06-30_supervisor-update/assets"


# ────────────────────────────────────────────────────────────────────────
# Part 1 — Generated-noise augmentation (negative result)
# ────────────────────────────────────────────────────────────────────────


def _copy_supervisor_assets():
    for src_name, dst_name in [
        ("s2_model_diagram.png", "noise_gen_diagram.png"),
        ("s2_spec_dregon.png", "noise_gen_spec_dregon.png"),
        ("s2_spec_michaels.png", "noise_gen_spec_michaels.png"),
    ]:
        src = SUPERVISOR_ASSETS / src_name
        dst = ASSETS / dst_name
        if src.exists():
            shutil.copy2(src, dst)


def _e3_smoothness_sweep():
    """Reproduce the E3 smoothness-sweep table from the experiment log."""
    rows = [
        # (harm_smooth, noise_smooth, best_val)
        (0.0, 0.0, 5.3554),  # baseline
        (1e-2, 0.0, 5.3581),  # harm too small
        (1e-1, 0.0, 5.3506),  # best
        (1.0, 0.0, 5.60),  # over-smooth
        (0.0, 1e-2, 5.601),  # noise 1e-2 actively hurts
        (0.0, 1.0, 5.3581),
        (1e-1, 10.0, 5.3581),  # combined example
    ]
    fig, ax = plt.subplots(figsize=(7.0, 2.7))
    xs = [f"h={r[0]:g}, n={r[1]:g}" for r in rows]
    vals = [r[2] for r in rows]
    best_i = int(np.argmin(vals))
    colors = ["#4af" if i == best_i else "#9aa" for i in range(len(vals))]
    ax.barh(xs, vals, color=colors)
    for i, v in enumerate(vals):
        ax.text(v + 0.001, i, f"{v:.4f}", va="center", fontsize=9)
    ax.set_xlim(5.3, 5.7)
    ax.set_xlabel("best validation spectral loss (↓ better)")
    ax.set_title("E3 — noise-gen smoothness sweep (best harm=1e-1, plateau within 0.008)")
    ax.axvline(5.3554, color="#888", lw=0.7, ls="--", label="baseline (no smoothness)")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(ASSETS / "e3_smoothness_sweep.png", dpi=130)
    plt.close(fig)


def _e4_aug_degradation():
    """E4 — augmenting RPS training with generated noise *degrades* results."""
    fig, ax = plt.subplots(figsize=(6.5, 3.0))
    models = ["uni-GRU128\n(causal)", "Transformer\n(global attn)"]
    base = [7.33, 8.46]
    aug = [9.29, 10.63]
    x = np.arange(len(models))
    w = 0.35
    b1 = ax.bar(x - w / 2, base, w, label="online-mix baseline", color="#9aa")
    b2 = ax.bar(x + w / 2, aug, w, label="+ generated noise", color="#e66")
    for bars in (b1, b2):
        for r in bars:
            ax.text(
                r.get_x() + r.get_width() / 2,
                r.get_height() + 0.1,
                f"{r.get_height():.2f}",
                ha="center",
                fontsize=9,
            )
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("val PIT MSE (↓ better)")
    ax.set_title("E4 — generated-noise augmentation degrades RPS prediction (+26%–27%)")
    ax.legend(loc="upper left", fontsize=9)
    ax.set_ylim(0, 12)
    for i, (bv, av) in enumerate(zip(base, aug)):
        ax.annotate(
            f"+{(av / bv - 1) * 100:.0f}%",
            xy=(i + w / 2, av),
            xytext=(i + w / 2 + 0.18, av + 0.8),
            fontsize=10,
            color="#c00",
            weight="bold",
        )
    fig.tight_layout()
    fig.savefig(ASSETS / "e4_aug_degradation.png", dpi=130)
    plt.close(fig)


# ────────────────────────────────────────────────────────────────────────
# Part 2 — GP rotor-noise model
# ────────────────────────────────────────────────────────────────────────


def _gp_overview():
    """Schematic of the GP pipeline (boxes + arrows drawn with matplotlib)."""
    fig, ax = plt.subplots(figsize=(10.0, 2.6))
    ax.set_axis_off()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    boxes = [
        (0.1, "audio\n(M, T)", "#cde"),
        (1.9, "DWT split\ndb4 L4", "#dfd"),
        (3.7, "phase align\nBPF ref", "#dfd"),
        (5.5, "lstsq on\nFourier basis\n(known BPFs)", "#fdd"),
        (7.6, "SVGP\nMatérn-5/2(y,z)\n⊙ IndexKernel", "#fdd"),
        (9.3, "synth:\n$\\mu_w\\!\\cdot\\!F$\n+ $\\sigma_b$ resid.", "#cde"),
    ]
    for x, t, c in boxes:
        ax.add_patch(_Rect((x, 1.0), 1.6, 1.2, facecolor=c, edgecolor="#444"))
        ax.text(x + 0.8, 1.6, t, ha="center", va="center", fontsize=8.5)
    for i in range(len(boxes) - 1):
        x0 = boxes[i][0] + 1.6
        x1 = boxes[i + 1][0]
        ax.annotate(
            "",
            xy=(x1, 1.6),
            xytext=(x0, 1.6),
            arrowprops=dict(arrowstyle="->", color="#444", lw=1.2),
        )
    ax.text(
        5.0,
        0.4,
        "Faithful Lee et al. (JASA 2026):  broadband → likelihood noise;  "
        "tonal → physics-injected Fourier design + GP-smoothed coefficients",
        ha="center",
        fontsize=9,
        style="italic",
        color="#444",
    )
    fig.tight_layout()
    fig.savefig(ASSETS / "gp_overview.png", dpi=130)
    plt.close(fig)


def _gp_v1_v2_rmse():
    fig, ax = plt.subplots(figsize=(6.0, 2.4))
    labels = [
        "v1  generic kernel\n(RBF⊙IndexKernel)",
        "v2  faithful Fourier-basis\n(phase-aligned, DWT-split, σ_b noise)",
    ]
    vals = [0.38, 0.10]
    colors = ["#999", "#4af"]
    ax.barh(labels, vals, color=colors)
    for i, v in enumerate(vals):
        ax.text(v + 0.005, i, f"{v:.2f}", va="center", fontsize=10)
    ax.set_xlabel("coefficient RMSE (held-out)")
    ax.set_xlim(0, 0.5)
    ax.set_title("GP rewrite — faithful construct cuts RMSE ~4×")
    fig.tight_layout()
    fig.savefig(ASSETS / "gp_v1_v2_rmse.png", dpi=130)
    plt.close(fig)


def _gp_faithful_spectrum():
    """Re-build the spectral comparison if outputs/gp_rotor_noise exists."""
    out = ROOT / "outputs/gp_rotor_noise"
    if not (out / "real_holdout.wav").exists():
        return
    import soundfile as sf

    r, sr = sf.read(str(out / "real_holdout.wav"))
    g, _ = sf.read(str(out / "generated.wav"))
    gn, _ = sf.read(str(out / "generated_noisy.wav"))
    m = json.loads((out / "fit_metrics.json").read_text())

    def stft(x):
        W = 2048
        x = np.asarray(x, np.float64).ravel()
        nf = x.shape[0] // W * W
        return 10 * np.log10(
            np.abs(np.fft.rfft((x[:nf].reshape(-1, W)) * np.hanning(W), axis=-1)) + 1e-12
        )

    f = np.fft.rfftfreq(2048, 1 / sr)
    Sr, Sg, Sgn = stft(r), stft(g), stft(gn)
    fig, ax = plt.subplots(2, 2, figsize=(11, 7))
    for a, S, t in [(ax[0, 0], Sr, "real (held-out)"), (ax[0, 1], Sg, "GP-generated (tonal mean)")]:
        a.imshow(
            S, aspect="auto", origin="lower", cmap="magma", extent=[f[0], f[-1], 0, S.shape[0]]
        )
        a.set_title(t)
        a.set_xlim(0, 4000)
        a.set_ylabel("frame")
    ax[1, 0].semilogx(f, Sr.mean(0), label="real", lw=1.2)
    ax[1, 0].semilogx(f, Sg.mean(0), label="GP mean", lw=1.0)
    ax[1, 0].semilogx(f, Sgn.mean(0), label="GP + broadband residual", lw=0.8, alpha=0.8)
    ax[1, 0].set_xlim(20, 8000)
    ax[1, 0].legend()
    ax[1, 0].set_xlabel("Hz")
    ax[1, 0].set_ylabel("dB")
    ax[1, 0].set_title("average spectra")
    ax[1, 1].bar(
        ["real", "gen", "gen+bb"],
        [
            10 * np.log10((Sr**2).mean() + 1e-12),
            10 * np.log10((Sg**2).mean() + 1e-12),
            10 * np.log10((Sgn**2).mean() + 1e-12),
        ],
        color=["#444", "#4af", "#f4a"],
    )
    ax[1, 1].set_ylabel("avg power (dB)")
    ax[1, 1].set_title("overall energy")
    fig.suptitle(
        f"Faithful GP (Lee et al. 2026) on Michael rec 1 — coeff RMSE {m['rmse_coeff']:.3f}",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(ASSETS / "gp_faithful_spectrum.png", dpi=130)
    plt.close(fig)


def _qd2026_tiers():
    """Quiet Drones 2026 literature scan — top relevant papers by tier."""
    papers = [
        # (tier, [43] index, short title, relevance)
        ("★★★", "43", "GP time-domain multi-rotor noise\n(uncertainty)", "closest peer"),
        ("★★★", "74", "RPS estimation from onboard audio\n(this project)", "own paper"),
        ("★★★", "51", "Drone simulation fidelity map\n(BEMT/VPM/FVM + FW-H)", "modeling"),
        ("★★★", "50", "Streamlined UAM noise prediction\n(URANS + FW-H)", "modeling"),
        ("★★★", "69", "Keynote: LBM low-Re rotor aeroacoustics\n(variable RPM)", "reference"),
        ("★★★", "39", "Propeller aeroacoustics under gusts\n(BPF±f_gust sidebands)", "mechanism"),
        ("★★★", "1", "Synchrophasing / stator clocking\n(~10 dB BPF reduction)", "phase prior"),
        ("★★★", "61", "Active noise cancelling quiet zone\nonboard DJI Mavic 3", "suppression"),
        ("★★", "28", "Tilting rotor noise (non-quasi-steady)", "modeling"),
        ("★★", "2", "Porous strut below propeller (BEM)", "modeling"),
        ("★★", "57", "Toroidal blade propeller (CFD)", "suppression"),
        ("★★", "56", "AeroFeathers (bio-fiber FDM)", "suppression"),
        ("★★", "62", "MVDR+SCDM, 8-mic drone audition", "adjacent"),
    ]
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.set_axis_off()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, len(papers) + 1)
    colors = {"★★★": "#bdf", "★★": "#fdd"}
    for i, (tier, idx, title, rel) in enumerate(papers):
        y = len(papers) - i
        ax.add_patch(_Rect((0, y - 0.4), 1.0, 0.8, facecolor=colors[tier], edgecolor="#888"))
        ax.text(0.5, y, tier, ha="center", va="center", fontsize=11)
        ax.text(1.2, y, f"[{idx}]", fontsize=9, va="center", weight="bold")
        ax.text(2.4, y, title, fontsize=8.4, va="center")
        ax.text(8.2, y, rel, fontsize=8.0, va="center", style="italic", color="#555")
    ax.set_title(
        "Quiet Drones 2026 proceedings scan (57 PDFs) — relevance to drone-noise modeling / suppression",
        fontsize=10.5,
        loc="left",
    )
    fig.tight_layout()
    fig.savefig(ASSETS / "gp_qd2026_tiers.png", dpi=130)
    plt.close(fig)


def main():
    try:
        _copy_supervisor_assets()
    except Exception as e:
        print("[warn] copy supervisor assets:", e)
    try:
        _e3_smoothness_sweep()
    except Exception as e:
        print("[warn] e3:", e)
    try:
        _e4_aug_degradation()
    except Exception as e:
        print("[warn] e4:", e)
    try:
        _gp_overview()
    except Exception as e:
        print("[warn] gp overview:", e)
    try:
        _gp_v1_v2_rmse()
    except Exception as e:
        print("[warn] v1v2:", e)
    try:
        _gp_faithful_spectrum()
    except Exception as e:
        print("[warn] faithful spectrum:", e)
    try:
        _qd2026_tiers()
    except Exception as e:
        print("[warn] qd tiers:", e)
    print("[prepare] done →", ASSETS)


if __name__ == "__main__":
    main()
