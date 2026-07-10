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
Part 2 — GP rotor-noise model (work in progress, Lee et al. 2026):
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
import matplotlib.ticker as mticker
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


def _e4_pit_curves():
    """E4 — train/val PIT loss curves fetched from wandb (flyingleafe/rps-prediction).

    Runs: transformer+gen pfr31em4 (m169le7r/d428l2db repeat seeds),
    uni-GRU128+gen bnu352hw, simple_conv_v2 (diverged) 60ua785e.
    Arrays are frozen here so `make figures` stays offline.
    """
    nan = float("nan")
    tf = {
        "e": list(range(1, 25)),
        "train": [264.672, 19.066, 15.153, 13.414, 11.441, 11.814, 10.242, 10.229, 9.88,
                  8.764, 8.453, 8.428, 7.357, 7.087, 6.797, 4.901, 4.9, 4.741, 4.536,
                  4.155, 4.394, 4.038, 3.468, 3.741],
        "val": [34.107, 18.929, 22.863, 19.948, 19.531, 13.604, 12.313, 12.082, 10.631,
                26.257, 16.627, 15.472, 11.948, 13.653, 29.364, 23.381, 39.756, 28.587,
                34.732, 22.122, 36.516, 41.454, 38.983, 43.638],
    }  # fmt: skip
    tf2 = {"e": list(range(1, 23)),
           "val": [89.83, 40.48, 26.35, 20.4, 20.258, 17.293, 15.658, 22.363, 18.215,
                   18.757, 19.122, 18.164, 17.535, 18.35, 18.175, 16.051, 16.078, 25.55,
                   20.276, 20.644, 19.406, 20.155]}  # fmt: skip
    tf3 = {"e": list(range(1, 26)),
           "val": [52.5, 37.528, 19.395, 26.582, 21.815, 20.028, 14.325, 19.834, 14.608,
                   12.464, 16.658, 19.001, 14.186, 16.716, 21.962, 32.34, 15.567, 21.735,
                   20.47, 22.649, 22.963, 19.139, 23.981, 23.434, 29.583]}  # fmt: skip
    gru = {
        "e": list(range(1, 26)),
        "train": [4176.055, 1972.192, 659.908, 124.632, 14.538, 10.801, 10.484, 7.97,
                  7.261, 10.191, 17.679, 10.062, nan, 8.451, 7.035, 5.762, 6.358, 6.065,
                  6.514, 5.596, 6.238, 6.588, 6.112, 6.048, 5.428],
        "val": [3029.95, 1291.728, 389.941, 61.832, 17.135, 10.7, 11.682, 13.592, 13.143,
                9.293, 15.961, 15.085, 11.926, 9.927, 13.417, 10.369, 10.546, 11.175,
                10.736, 10.532, 11.187, 10.256, 9.855, 9.87, 9.506],
    }  # fmt: skip
    divg = {"e": [1, 2, 3, 4], "val": [3034.075, 1291.943, 353.473, nan]}

    c_tr, c_vl, c_base, c_shade = "#2f6db3", "#d1495b", "#6b7280", "#f2c4c4"
    with plt.rc_context({"font.size": 10, "axes.grid": True, "grid.alpha": 0.25}):
        fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.2, 4.0))

        # Panel A — transformer overfit
        axL.axvspan(9, 24.5, color=c_shade, alpha=0.35, lw=0, zorder=0)
        for s in (tf2, tf3):
            axL.plot(s["e"], s["val"], color=c_vl, lw=1.0, alpha=0.30, zorder=1)
        axL.plot(
            tf["e"],
            tf["train"],
            "-o",
            color=c_tr,
            ms=3.2,
            lw=1.7,
            label="train PIT (memorising)",
            zorder=3,
        )
        axL.plot(
            tf["e"],
            tf["val"],
            "-o",
            color=c_vl,
            ms=3.2,
            lw=1.7,
            label="val PIT (3 seeds)",
            zorder=3,
        )
        axL.axhline(8.46, color=c_base, ls="--", lw=1.2, zorder=2)
        axL.set_yscale("log")
        axL.set_xlim(0.5, 25)
        axL.set_ylim(2.8, 300)
        axL.set_yticks([3, 5, 10, 20, 50, 100])
        axL.get_yaxis().set_major_formatter(mticker.ScalarFormatter())
        axL.set_xlabel("epoch")
        axL.set_ylabel("PIT MSE")
        axL.set_title("Transformer (global attn) + generated noise", fontsize=11, weight="bold")
        axL.annotate(
            "best val 10.6\n(epoch 9)",
            xy=(9, 10.631),
            xytext=(11.5, 5.6),
            fontsize=8.5,
            color=c_vl,
            arrowprops=dict(arrowstyle="->", color=c_vl, lw=1.0),
        )
        axL.annotate(
            "val 43.6",
            xy=(24, 43.638),
            xytext=(18.4, 90),
            fontsize=8.5,
            color=c_vl,
            weight="bold",
            arrowprops=dict(arrowstyle="->", color=c_vl, lw=1.0),
        )
        axL.annotate("train 3.7", xy=(23, 3.468), xytext=(18.6, 3.05), fontsize=8.5, color=c_tr)
        axL.text(
            16.7,
            200,
            "overfitting:\nval ↑ while train ↓",
            fontsize=8.2,
            color="#a03434",
            ha="center",
            style="italic",
        )
        axL.text(0.7, 8.9, "no-gen baseline 8.46", fontsize=7.8, color=c_base, va="top")
        axL.legend(loc="lower left", fontsize=8.0, framealpha=0.9)

        # Panel B — uni-GRU plateau + NaN divergence
        axR.plot(
            gru["e"],
            gru["train"],
            "-o",
            color=c_tr,
            ms=3.2,
            lw=1.7,
            label="uni-GRU train PIT",
            zorder=3,
        )
        axR.plot(
            gru["e"],
            gru["val"],
            "-o",
            color=c_vl,
            ms=3.2,
            lw=1.7,
            label="uni-GRU val PIT",
            zorder=3,
        )
        axR.plot(
            divg["e"],
            divg["val"],
            "--s",
            color="#8a8f98",
            ms=3.4,
            lw=1.4,
            label="simple-conv val (diverges)",
            zorder=2,
        )
        axR.plot(3.85, 353.473, marker="x", ms=11, mew=2.6, color="#111", zorder=4)
        axR.axhline(7.33, color=c_base, ls="--", lw=1.2, zorder=1)
        axR.set_yscale("log")
        axR.set_xlim(0.5, 25)
        axR.set_ylim(4.5, 6000)
        axR.set_yticks([5, 10, 30, 100, 1000, 5000])
        axR.get_yaxis().set_major_formatter(mticker.ScalarFormatter())
        axR.set_xlabel("epoch")
        axR.set_ylabel("PIT MSE")
        axR.set_title("uni-GRU (causal) + generated noise", fontsize=11, weight="bold")
        axR.annotate(
            "NaN — training\ndiverges (epoch 4)",
            xy=(3.85, 353.473),
            xytext=(5.2, 900),
            fontsize=8.5,
            color="#111",
            arrowprops=dict(arrowstyle="->", color="#111", lw=1.0),
        )
        axR.annotate(
            "val plateaus 9.5",
            xy=(25, 9.506),
            xytext=(15.5, 20),
            fontsize=8.5,
            color=c_vl,
            arrowprops=dict(arrowstyle="->", color=c_vl, lw=1.0),
        )
        axR.text(12.5, 6.6, "no-gen baseline 7.33", fontsize=7.8, color=c_base, va="top")
        axR.legend(loc="upper right", fontsize=8.0, framealpha=0.9)

        fig.suptitle(
            "Part I · E4 — generated-noise augmentation never beats the no-gen baseline",
            fontsize=12,
            weight="bold",
            y=1.00,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        fig.savefig(ASSETS / "e4_pit_curves.png", dpi=140)
        plt.close(fig)


def _noise_gen_diagram_positional():
    """Modified deep-generator diagram: per-rotor emit + drone-code FiLM + positional mix.

    Recasts the Stage-2 block diagram to foreground the three things that make it
    ``PositionalHarmonicNoiseGen``: (1) an external per-drone conditioning code z
    (FiLM), (2) one shared-weight single-rotor emitter applied per rotor (rotor
    axis folded into batch → rotors independent), (3) a propagate-and-mix stage
    that attenuates (1/r) + delays (r/c) each rotor's source to the mic and sums.
    """
    import matplotlib.patches as mp

    green, blue, yellow, purple, pink, wave = (
        "#cde6cd", "#cfe0f3", "#fdebc0", "#dcdcf0", "#fbc4ab", "#1f5fbf",
    )  # fmt: skip
    rotc = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    def rps_demo(n=600, rng=None):
        rng = rng or np.random.default_rng(1)
        base = np.linspace(0, 1, n)
        out = []
        for k in range(4):
            w = np.cumsum(rng.standard_normal(n)) * 0.6
            out.append(70 + 8 * np.sin(2 * np.pi * (base + k * 0.2)) + (w - w.mean()) * 0.8)
        return np.array(out)

    def wave_demo(n=4000, rng=None):
        rng = rng or np.random.default_rng(3)
        t = np.arange(n)
        env = 0.6 + 0.4 * np.sin(2 * np.pi * t / 900)
        return env * (rng.standard_normal(n) * 0.5 + 0.5 * np.sin(2 * np.pi * t / 40))

    fig, ax = plt.subplots(figsize=(13.2, 6.9))
    ax.set_xlim(0, 26)
    ax.set_ylim(0, 14)
    ax.axis("off")

    def rbox(cx, cy, w, h, text, fc, fs=9.5, ec="#444"):
        ax.add_patch(mp.FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                     boxstyle="round,pad=0.02,rounding_size=0.10", fc=fc, ec=ec, lw=1.2))  # fmt: skip
        ax.text(cx, cy, text, ha="center", va="center", fontsize=fs)

    def arrow(x0, y0, x1, y1, color="#333", lw=1.6, dashed=False, style="-|>"):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle=style, lw=lw, color=color,
                                    linestyle="--" if dashed else "-", shrinkA=0, shrinkB=0))  # fmt: skip

    def label(x, y, t, fs=9.0, color="#222", ha="center", weight="normal", style="normal"):
        ax.text(x, y, t, ha=ha, va="center", fontsize=fs, color=color, weight=weight, style=style)

    for x, t in [(3.0, "① Per-rotor inputs + drone code"),
                 (12.2, "② Emit  —  one shared emitter per rotor"),
                 (21.3, "③ Propagate & mix to the mic")]:  # fmt: skip
        label(x, 13.55, t, 11.5, "#1a1a1a", weight="bold")

    # stage 1 — inputs
    ax_rps = ax.inset_axes((0.9, 10.4, 4.3, 2.1), transform=ax.transData)
    for k, row in enumerate(rps_demo()):
        ax_rps.plot(row, lw=0.8, color=rotc[k])
    ax_rps.set_xticks([])
    ax_rps.set_yticks([])
    ax_rps.set_title(r"per-rotor RPS  $\mathrm{RPS}_i(t)$", fontsize=8.5, pad=2)
    for s in ax_rps.spines.values():
        s.set_edgecolor("#999")
    label(3.0, 9.85, "4 independent rotor speed channels", 8.5, "#555", style="italic")

    cw, cx0 = 0.62, 1.35
    codecols = ["#b8d8f0", "#f6c7a8", "#c7e6c0", "#e6c2c2", "#d9c9ec", "#f2e2b0"]
    for j in range(6):
        ax.add_patch(
            mp.Rectangle((cx0 + j * cw, 7.48), cw, 0.64, fc=codecols[j], ec="#555", lw=0.9)
        )
    label(
        3.0, 8.55, r"drone code  $z$   (DroneCodebook: id $\to z$)", 9.0, "#1a1a1a", weight="bold"
    )
    label(3.0, 6.95, "one code per drone, conditions every rotor", 8.2, "#555", style="italic")

    # stage 2 — per-rotor emitter (stacked = shared weights across rotors)
    for dx, dy in [(0.45, 0.45), (0.3, 0.3), (0.15, 0.15)]:
        ax.add_patch(mp.FancyBboxPatch((8.4 + dx, 5.7 + dy), 6.7, 5.4,
                     boxstyle="round,pad=0.02,rounding_size=0.12", fc="#eef2ee", ec="#9ab09a", lw=1.0))  # fmt: skip
    ax.add_patch(mp.FancyBboxPatch((8.4, 5.7), 6.7, 5.4,
                 boxstyle="round,pad=0.02,rounding_size=0.12", fc="#f6faf6", ec="#4a4", lw=1.5))  # fmt: skip
    label(11.55, 10.75, "single-rotor emitter · shared weights", 9.3, "#1a5a1a", weight="bold")
    label(15.5, 11.75, "× 4 rotors", 11, "#1a5a1a", weight="bold", ha="right")
    rbox(11.75, 9.7, 5.6, 0.75, "Causal Conv1d encoder", green, 9.2)
    arrow(11.75, 9.3, 10.4, 8.75)
    arrow(11.75, 9.3, 13.2, 8.75)
    rbox(10.2, 8.3, 2.7, 0.85, "harmonic\noscillator bank", purple, 8.6)
    rbox(13.3, 8.3, 2.7, 0.85, "filtered\nbroadband", purple, 8.6)
    ax.add_patch(mp.Circle((11.75, 7.05), 0.26, fc="white", ec="#333", lw=1.3))
    ax.text(11.75, 7.05, "+", ha="center", va="center", fontsize=13)
    arrow(10.2, 7.85, 11.55, 7.2)
    arrow(13.3, 7.85, 11.95, 7.2)
    arrow(11.75, 6.75, 11.75, 6.35)
    label(
        11.75,
        6.1,
        r"per-rotor source  $x_i(t)$  (radiated at the rotor)",
        8.8,
        "#1a1a1a",
        weight="bold",
    )
    label(
        11.75,
        5.25,
        "rotor axis folded into the batch  →  rotors modelled independently (no cross-rotor coupling)",
        8.0,
        "#555",
        style="italic",
    )

    arrow(5.25, 11.3, 8.35, 10.4, lw=1.7)
    label(6.7, 11.15, r"$\mathrm{RPS}_i(t)$", 8.5, "#333")
    arrow(5.05, 7.7, 8.35, 8.75, lw=1.7, color="#a33")
    label(7.15, 8.55, r"FiLM  ($\gamma,\beta$)", 8.6, "#a33", weight="bold")
    arrow(15.15, 8.3, 17.0, 8.3, lw=2.0)
    label(16.05, 8.7, r"4× $x_i(t)$", 8.6, "#333")

    # stage 3 — propagate & mix
    ax_geo = ax.inset_axes((17.4, 10.0, 3.0, 2.5), transform=ax.transData)
    rot = np.array([[-1, 1], [1, 1], [-1, -1], [1, -1]]) * 0.8
    mic = np.array([0.15, -0.05])
    for k, p in enumerate(rot):
        ax_geo.plot([p[0], mic[0]], [p[1], mic[1]], color="#bbb", lw=0.8, zorder=1)
        ax_geo.scatter(*p, s=60, color=rotc[k], zorder=2, edgecolor="#333", linewidth=0.5)
    ax_geo.scatter(*mic, s=70, marker="^", color="#222", zorder=3)
    ax_geo.text(mic[0] + 0.12, mic[1] - 0.28, "mic", fontsize=7, color="#222")
    ax_geo.text(0.0, 0.42, r"$r_i$", fontsize=8, color="#666")
    ax_geo.set_xlim(-1.4, 1.4)
    ax_geo.set_ylim(-1.4, 1.4)
    ax_geo.set_xticks([])
    ax_geo.set_yticks([])
    ax_geo.set_title("rotor & mic geometry", fontsize=8, pad=2)
    for s in ax_geo.spines.values():
        s.set_edgecolor("#999")

    rbox(
        21.3,
        8.55,
        5.0,
        1.2,
        r"per rotor:  attenuate $\times\, d_{\mathrm{ref}}/r_i$"
        + "\n"
        + r"delay $\;r_i/c\;$ (fractional)",
        yellow,
        9.0,
    )
    arrow(21.3, 7.95, 21.3, 7.3)
    ax.add_patch(mp.Circle((21.3, 6.9), 0.32, fc="white", ec="#333", lw=1.5))
    ax.text(21.3, 6.9, r"$\sum_i$", ha="center", va="center", fontsize=12)
    label(23.1, 6.9, "sum over\nrotors", 8.0, "#555", ha="left")
    arrow(21.3, 6.55, 21.3, 6.0)
    rbox(21.3, 5.55, 4.4, 0.8, r"observed noise  $y_m(t)$", blue, 9.5)
    label(
        21.3,
        4.8,
        "native multi-mic — rotors summed in the rfft domain",
        8.0,
        "#555",
        style="italic",
    )

    # bottom — loss (raised under the stages; generated sits under stage 3)
    ax_real = ax.inset_axes((8.2, 3.1, 3.4, 1.4), transform=ax.transData)
    ax_real.plot(wave_demo(rng=np.random.default_rng(9)), lw=0.4, color=wave)
    ax_real.axis("off")
    label(9.9, 2.75, "real recording", 8.6)
    ax_gen = ax.inset_axes((16.0, 3.1, 3.4, 1.4), transform=ax.transData)
    ax_gen.plot(wave_demo(), lw=0.4, color=wave)
    ax_gen.axis("off")
    label(17.7, 2.75, r"generated $y_m(t)$", 8.6)
    sk = 0.32
    poly = np.array([[12.3 + sk, 3.4], [15.1 + sk, 3.4], [15.1 - sk, 4.2], [12.3 - sk, 4.2]])
    ax.add_patch(mp.Polygon(poly, closed=True, fc=pink, ec="#a33", lw=1.1))
    label(13.7, 3.8, "multi-resolution\nspectral loss", 8.4)
    arrow(21.3, 5.15, 19.6, 4.5, lw=1.4)  # observed noise -> generated waveform
    arrow(15.95, 3.8, 15.2, 3.8, dashed=True, color="#a33", lw=1.2)  # generated -> loss
    arrow(12.5, 3.8, 11.65, 3.8, dashed=True, color="#a33", lw=1.2)  # loss -> real
    label(
        4.5,
        3.8,
        r"$y_m[t]=\sum_i \dfrac{d_{\mathrm{ref}}}{r_i}\,x_i\!\left(t-\dfrac{r_i}{c}\right)$",
        13,
        "#1a1a1a",
    )

    fig.tight_layout()
    fig.savefig(ASSETS / "noise_gen_diagram_positional.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ────────────────────────────────────────────────────────────────────────
# Part 2 — GP rotor-noise model
# ────────────────────────────────────────────────────────────────────────


def _gp_overview():
    """Schematic of the GP pipeline (boxes + arrows drawn with matplotlib)."""
    fig, ax = plt.subplots(figsize=(12.0, 3.6))
    ax.set_axis_off()
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4)
    bw, bh = 1.7, 1.45
    boxes = [
        (0.15, "audio\n(M, T)", "#cde"),
        (2.0, "DWT split\ndb4 L4", "#dfd"),
        (3.85, "phase align\nBPF ref", "#dfd"),
        (5.7, "least squares\nFourier basis\n(known BPFs)", "#fdd"),
        (8.1, "SVGP\nMatérn-5/2(y,z)\n⊙ IndexKernel", "#fdd"),
        (10.15, "synth:\n$\\mu_w \\cdot F$\n+ $\\sigma_b$ resid.", "#cde"),
    ]
    for x, t, c in boxes:
        ax.add_patch(_Rect((x, 1.3), bw, bh, facecolor=c, edgecolor="#444", lw=1.2))
        ax.text(x + bw / 2, 2.02, t, ha="center", va="center", fontsize=11.5)
    for i in range(len(boxes) - 1):
        x0 = boxes[i][0] + bw
        x1 = boxes[i + 1][0]
        ax.annotate(
            "",
            xy=(x1, 2.02),
            xytext=(x0, 2.02),
            arrowprops=dict(arrowstyle="->", color="#444", lw=1.4),
        )
    ax.text(
        6.0,
        0.5,
        "Lee et al. (JASA 2026):  broadband → likelihood noise;  "
        "tonal → physics-injected Fourier design + GP-smoothed coefficients",
        ha="center",
        fontsize=11,
        style="italic",
        color="#444",
    )
    fig.tight_layout()
    fig.savefig(ASSETS / "gp_overview.png", dpi=150)
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
    fig, ax = plt.subplots(2, 2, figsize=(9.5, 6.0))
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
        f"GP (Lee et al. 2026) on Michael rec 1 — coeff RMSE {m['rmse_coeff']:.3f}",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(ASSETS / "gp_faithful_spectrum.png", dpi=130)
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
        _e4_pit_curves()
    except Exception as e:
        print("[warn] e4 pit curves:", e)
    try:
        _noise_gen_diagram_positional()
    except Exception as e:
        print("[warn] noise-gen diagram:", e)
    try:
        _gp_overview()
    except Exception as e:
        print("[warn] gp overview:", e)
    try:
        _gp_faithful_spectrum()
    except Exception as e:
        print("[warn] faithful spectrum:", e)
    print("[prepare] done →", ASSETS)


if __name__ == "__main__":
    main()
