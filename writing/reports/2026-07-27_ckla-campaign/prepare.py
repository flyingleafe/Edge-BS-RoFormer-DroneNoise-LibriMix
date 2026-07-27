#!/usr/bin/env python3
"""Assets for the CKLA campaign report.

All figures here are synthesized directly from numbers recorded in
docs/ckla-design.md, docs/complex-ou-layer-exploration.md,
docs/experiments/ckla.md, docs/experiments/ckla-activation-analysis.md and
the commit log (no re-running of training/eval — those artifacts are
gitignored and the campaign is still mid-flight). Schematic figures (comb
spectrograms, recursion diagrams) are illustrative, built with the same
recipe as the 07-18/07-24 reports.
"""

from __future__ import annotations

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

HERE = pathlib.Path(__file__).parent
ASSETS = HERE / "assets"
ASSETS.mkdir(exist_ok=True)

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 150,
    }
)


# ---------------------------------------------------------------------------
# Fig 1 — augmentation dilution arithmetic
# ---------------------------------------------------------------------------
def make_aug_dilution(dest: pathlib.Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.0))

    # left: G6 bundle - pie-ish bar of transform families, freq_scale highlighted
    transforms = [
        "freq_scale",
        "spectral_recolor",
        "random_reverb",
        "tooth_dropout",
        "spec_mask",
        "floor_inject",
    ]
    ax = axes[0]
    p_hit = 0.7
    p_each = p_hit / len(transforms)
    colors = ["C3"] + ["0.75"] * (len(transforms) - 1)
    ax.barh(range(len(transforms)), [p_each] * len(transforms), color=colors)
    ax.set_yticks(range(len(transforms)))
    ax.set_yticklabels(transforms, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("P(applied to a given noise chunk)")
    ax.set_title("G6 bundle: 6-way choice, p=0.7", fontsize=9)
    ax.axvline(p_each, color="C3", lw=0.8, ls="--")
    ax.text(
        p_each + 0.003,
        0,
        f"freq_scale\n≈ 1 in {round(1 / p_each)}",
        color="C3",
        fontsize=8,
        va="center",
    )

    # right: solo policy
    ax = axes[1]
    ax.barh([0], [p_hit], color="C3")
    ax.barh([1], [0], color="0.75")
    ax.set_yticks([0])
    ax.set_yticklabels(["freq_scale"], fontsize=8)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("P(applied to a given noise chunk)")
    ax.set_title("g2_if_freqscale: solo, p=0.7", fontsize=9)
    ax.axvline(p_hit, color="C3", lw=0.8, ls="--")
    ax.text(
        p_hit + 0.02,
        0,
        f"freq_scale\n≈ 1 in {round(1 / p_hit)}",
        color="C3",
        fontsize=8,
        va="center",
    )
    ax.set_ylim(-0.5, 1.5)
    axes[1].set_yticks([0])

    fig.suptitle(
        "Bundling the only spacing-forcing transform with 5 others dilutes it ~6x", fontsize=10
    )
    fig.tight_layout()
    fig.savefig(dest, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig 2 — freq-scale spectrogram illustration + scale-response bars
# ---------------------------------------------------------------------------
def make_freqscale_illustration(dest: pathlib.Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.0), width_ratios=[1, 1, 0.9])

    def comb(f, f0, n, width, amp=1.0):
        y = np.zeros_like(f)
        for k in range(1, n + 1):
            y += amp / k * np.exp(-((f - k * f0) ** 2) / (2 * width**2))
        return y

    f = np.linspace(0, 800, 3000)
    f0 = 74.0

    ax = axes[0]
    ax.fill_between(f, 0, comb(f, f0, 10, 2.2), color="C0", alpha=0.85)
    ax.set_title("original: $f_0 = 74$ rev/s", fontsize=9)
    ax.set_xlabel("frequency (Hz)")
    ax.set_yticks([])
    ax.set_xlim(0, 800)

    ax = axes[1]
    ax.fill_between(f, 0, comb(f, f0 * 1.02, 10, 2.2), color="C3", alpha=0.85)
    ax.set_title(r"freq_scale $\alpha{=}1.02$: $f_0 = 75.5$ rev/s", fontsize=9)
    ax.set_xlabel("frequency (Hz)")
    ax.set_yticks([])
    ax.set_xlim(0, 800)

    ax = axes[2]
    labels = ["ideal\nresponse", "ckla_p1", "g2_if"]
    vals = [2.0, 0.03, 0.06]
    colors = ["0.6", "C3", "C0"]
    bars = ax.bar(labels, vals, color=colors)
    ax.set_ylabel("mean prediction shift (%)")
    ax.set_title(r"response to $\times1.02$ scale", fontsize=9)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.03, f"{v:.2f}%", ha="center", fontsize=8)
    ax.set_ylim(0, 2.4)

    fig.suptitle(
        "A genuine 2% comb-spacing shift moves neither trained model's prediction",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(dest, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig 3 — CKLA recursion / layer block diagram (schematic)
# ---------------------------------------------------------------------------
def make_ckla_block_diagram(dest: pathlib.Path) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 3.4))
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)

    def box(x, y, w, h, text, fc="white", fontsize=8.5):
        rect = Rectangle((x, y), w, h, fc=fc, ec="0.2", lw=1.0, zorder=2)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, zorder=3)

    def arrow(x0, y0, x1, y1):
        ax.annotate(
            "",
            xy=(x1, y1),
            xytext=(x0, y0),
            arrowprops=dict(arrowstyle="->", lw=1.1, color="0.2"),
            zorder=1,
        )

    box(0.1, 1.5, 1.3, 1.0, "conv(k=4)\n+ SiLU", fc="0.92")
    box(1.7, 1.5, 1.3, 1.0, "QK-norm\nsoftplus λv", fc="0.92")
    box(
        3.3,
        0.3,
        2.0,
        1.0,
        r"$\bar a_t=e^{-\gamma+i\omega_t}$" + "\n" + r"$\omega_t=\omega_0+s\cdot W_\omega h_t$",
        fc="#ffe3d9",
    )
    box(
        3.3,
        2.1,
        2.0,
        1.0,
        r"$\varphi_t=k_t^2\lambda v_t$" + "\n" + r"$\kappa_t=k_t\lambda v_t v_t$",
        fc="0.92",
    )
    box(
        5.7,
        1.2,
        2.2,
        1.6,
        "flat information\nrecursion (scan)\n"
        + r"$\mathrm{den}_t=|\bar a_t|^2+\bar p_t\lambda_{t-1}$"
        + "\n"
        + r"$\lambda_t,\ \eta_t$",
        fc="#dbe9ff",
    )
    box(8.3, 1.5, 1.5, 1.0, r"readout" + "\n" + r"$\mu=\eta/\lambda$", fc="0.92")

    arrow(1.4, 2.0, 1.7, 2.0)
    arrow(3.0, 2.0, 3.3, 2.6)
    arrow(3.0, 2.0, 3.3, 0.8)
    arrow(5.3, 2.6, 5.7, 2.2)
    arrow(5.3, 0.8, 5.7, 1.6)
    arrow(7.9, 2.0, 8.3, 2.0)

    ax.text(
        4.3,
        1.45,
        "the only complex piece:\nrotation multiplies $\\eta$,\nnot $\\lambda$",
        fontsize=7.5,
        ha="center",
        color="C3",
    )

    fig.suptitle(
        "ComplexKLALayer: real precision path untouched, rotation only on the information vector",
        fontsize=9.5,
    )
    fig.tight_layout()
    fig.savefig(dest, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig 4 — accumulator degeneration: lambda / gain trajectory (schematic,
# shape matches the reported numbers: t_sat ~7.2-7.7s of 8s, gain -> 1e-6)
# ---------------------------------------------------------------------------
def make_accumulator_degeneration(dest: pathlib.Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.0))
    t = np.linspace(0, 8, 400)

    ax = axes[0]
    for tsat, lbl, c in [(7.23, "layer 1", "C0"), (7.34, "layer 2", "C3")]:
        lam = 1 + (1e4 - 1) * (1 - np.exp(-t / (tsat / 4.5)))
        ax.plot(t, lam, label=lbl, color=c)
    ax.set_yscale("log")
    ax.set_xlabel("time within clip (s)")
    ax.set_ylabel(r"state precision $\lambda_t$")
    ax.axvspan(7.0, 8.0, color="0.9", zorder=0)
    ax.text(7.5, 2, "never\nsaturates", fontsize=7.5, ha="center", color="0.4")
    ax.legend(fontsize=8, frameon=False)
    ax.set_title(r"$\lambda_t$ climbs monotonically through 8 s", fontsize=9)

    ax = axes[1]
    for tsat, lbl, c in [(7.23, "layer 1", "C0"), (7.34, "layer 2", "C3")]:
        lam = 1 + (1e4 - 1) * (1 - np.exp(-t / (tsat / 4.5)))
        phi = 0.3
        gain = phi / lam
        ax.plot(t, gain, label=lbl, color=c)
    ax.set_yscale("log")
    ax.set_xlabel("time within clip (s)")
    ax.set_ylabel(r"Kalman gain $\varphi/\lambda$")
    ax.set_title("effective gain collapses to 1e-7..1e-4", fontsize=9)
    ax.legend(fontsize=8, frameon=False)

    fig.suptitle(
        "Accumulator degeneration: by mid-clip the head is a fixed clip-scale averager, not an adaptive filter",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(dest, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig 5 — rotation causal attribution (3-arm table as bars)
# ---------------------------------------------------------------------------
def make_rotation_attribution(dest: pathlib.Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    arms = ["intact", "rotation\nzeroed", "im-readout\nzeroed"]
    dregon = [3.481, 3.788, 3.647]
    fly124 = [1.174, 1.156, 1.165]

    x = np.arange(len(arms))
    w = 0.35
    ax.bar(x - w / 2, dregon, w, label="dregon_cruise", color="C0")
    ax.bar(x + w / 2, fly124, w, label="fly124_cruise", color="C3")
    ax.set_xticks(x)
    ax.set_xticklabels(arms, fontsize=8.5)
    ax.set_ylabel("PIT-MAE (rev/s), 12-clip subset")
    ax.legend(fontsize=8, frameon=False)
    for i, v in enumerate(dregon):
        ax.text(i - w / 2, v + 0.05, f"{v:.2f}", ha="center", fontsize=7.5)
    for i, v in enumerate(fly124):
        ax.text(i + w / 2, v + 0.05, f"{v:.2f}", ha="center", fontsize=7.5)
    ax.set_title("Rotation is causal on DREGON (+0.31, +9%), null on FLY124", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(dest, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig 6 — results ladder: full-envelope val
# ---------------------------------------------------------------------------
def make_results_ladder(dest: pathlib.Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.4), width_ratios=[1.3, 1])

    ax = axes[0]
    models = [
        "uni_gru128",
        "transformer\n(mag)",
        "transformer\n(IF, g2_if)",
        "CKLA\nbase",
        "CKLA\n+freqscale",
        "CKLA\n+pnoise",
    ]
    vals = [172.3, 72.7, 63.7, 85.2, 63.0, 44.8]
    colors = ["0.75", "0.75", "C0", "0.85", "#ffb08a", "C3"]
    bars = ax.bar(models, vals, color=colors)
    ax.set_ylabel("full-envelope val MSE")
    ax.set_title("Each lever fixes its measured pathology", fontsize=9.5)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 3, f"{v:.1f}", ha="center", fontsize=8)
    ax.tick_params(axis="x", labelsize=7.5)
    ax.axhline(63.7, color="C0", lw=0.8, ls="--", zorder=0)
    ax.text(5.5, 63.7 + 3, "matched\ntransformer", color="C0", fontsize=7, ha="right")

    ax = axes[1]
    pools = ["dregon_cruise", "fly124_cruise"]
    floor = [2.481, 2.33]
    ckla = [2.87, 1.39]
    x = np.arange(len(pools))
    w = 0.35
    ax.bar(x - w / 2, floor, w, label="g2_if floor", color="C0")
    ax.bar(x + w / 2, ckla, w, label="ckla_p1 best", color="C3")
    ax.set_xticks(x)
    ax.set_xticklabels(pools, fontsize=8.5)
    ax.set_ylabel("vk_eval PIT-MAE (rev/s)")
    ax.legend(fontsize=8, frameon=False)
    ax.set_title("P1 ledger: worse on DREGON, best-in-campaign on FLY124", fontsize=9)
    for i, v in enumerate(floor):
        ax.text(i - w / 2, v + 0.05, f"{v:.2f}", ha="center", fontsize=8)
    for i, v in enumerate(ckla):
        ax.text(i + w / 2, v + 0.05, f"{v:.2f}", ha="center", fontsize=8)

    fig.tight_layout()
    fig.savefig(dest, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig 7 — capture boundary comparison (P0b, schematic from reported numbers)
# ---------------------------------------------------------------------------
def make_capture_boundary(dest: pathlib.Path) -> None:
    fig, ax = plt.subplots(figsize=(6.6, 3.2))
    agg = [0.25, 0.5, 1, 2, 4]
    ckla_lock = [0.69, 0.60, 0.50, 0.20, 0.0]
    transformer_lock = [0.0, 0.0, 0.0, 0.0, 0.0]
    ax.plot(agg, ckla_lock, "o-", color="C3", label="CKLA (lock fraction)")
    ax.plot(agg, transformer_lock, "s--", color="C0", label="transformer (lock fraction)")
    ax.set_xlabel("drift aggressiveness")
    ax.set_ylabel("lock fraction (16 clips/cell)")
    ax.set_title("CKLA locks through drift, transformer never sustains lock", fontsize=9.5)
    ax.legend(fontsize=8.5, frameon=False)
    ax.set_ylim(-0.05, 0.85)
    fig.tight_layout()
    fig.savefig(dest, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    make_aug_dilution(ASSETS / "aug_dilution.png")
    make_freqscale_illustration(ASSETS / "freqscale_illustration.png")
    make_ckla_block_diagram(ASSETS / "ckla_block_diagram.png")
    make_accumulator_degeneration(ASSETS / "accumulator_degeneration.png")
    make_rotation_attribution(ASSETS / "rotation_attribution.png")
    make_results_ladder(ASSETS / "results_ladder.png")
    make_capture_boundary(ASSETS / "capture_boundary.png")
    print("wrote assets to", ASSETS)


if __name__ == "__main__":
    main()
