#!/usr/bin/env python3
"""Generate figures/tables for the decomposition-for-amplitude-targets deck.

Copies the already-built v3e decomposition spectrogram panels from the
scratchpad recipe (assets_recordings_v3e.py output) and builds two small
matplotlib figures (the linewidth law, the Whittle cost-of-a-cell curve)
from numbers already verified in the source docs.
"""

import pathlib
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

HERE = pathlib.Path(__file__).resolve().parent
ASSETS = HERE / "assets"
SCRATCH = pathlib.Path(
    "/tmp/claude-1000/-home-flyingleafe-Research-PhD-projects-harmonic-noise-suppression"
    "/5a88d51c-adfa-4ffa-951d-2f560860cb3c/scratchpad/split_demo3"
)

plt.rcParams.update(
    {
        "font.size": 13,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def copy_spectrogram_panels():
    ASSETS.mkdir(exist_ok=True)
    mapping = {
        "spec_free-flight_nosource_room1.jpg": "spec_dregon.jpg",
        "spec_FLY124.jpg": "spec_fly124.jpg",
        "spec_FLY125.jpg": "spec_fly125.jpg",
        "spec_perrotor_free-flight_nosource_room1.jpg": "spec_dregon_perrotor.jpg",
    }
    for src, dst in mapping.items():
        srcp = SCRATCH / src
        if srcp.exists():
            shutil.copy(srcp, ASSETS / dst)
        else:
            print(f"MISSING source spectrogram: {srcp}")


def crop_dregon_original_panel():
    """Crop just the top ('original (mic 1)') panel out of spec_dregon.jpg.

    Used on the "requirement" slide, before the comb/broadband terms are
    introduced -- showing only the messy raw audio, not the finished split.
    """
    src = ASSETS / "spec_dregon.jpg"
    if not src.exists():
        print(f"MISSING {src}, cannot crop original panel")
        return
    im = Image.open(src)
    w, _h = im.size
    crop = im.crop((0, 0, w, 300))
    crop.save(ASSETS / "spec_dregon_original.jpg", quality=92)


def fig_linewidth_law():
    """Order-averaged tooth contrast by band vs the 0.6*k Hz linewidth law."""
    bands = ["k1-9", "k10-24", "k25-49", "k50-80"]
    contrast = [6.76, 1.36, 0.13, 0.01]
    k_mid = [5, 17, 37, 65]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 4.0))

    ax1.bar(bands, contrast, color="#3b6fa0", width=0.6)
    ax1.set_ylabel("tooth contrast (dB)")
    ax1.set_title("Real tooth contrast drops with k")
    for i, v in enumerate(contrast):
        ax1.text(i, v + 0.15, f"{v:.2f}", ha="center", fontsize=11)
    ax1.axhline(0, color="black", lw=0.8)

    k = np.linspace(1, 80, 200)
    gamma = np.maximum(0.6 * k, 3.0)
    ax2.plot(k, gamma, color="#a03b3b", lw=2.5)
    ax2.scatter(k_mid, np.maximum(0.6 * np.array(k_mid), 3.0), color="#a03b3b", zorder=5)
    ax2.set_xlabel("harmonic order k")
    ax2.set_ylabel("Lorentzian half-width (Hz)")
    ax2.set_title(r"Linewidth law: $\gamma_k = \max(0.6k,\ \gamma_{\min})$")

    fig.tight_layout()
    fig.savefig(ASSETS / "linewidth_law.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_marginalization():
    """Marginalization result: floor + two Lorentzian lines summing to M(f)."""
    fig, ax = plt.subplots(figsize=(9.0, 4.4))
    f = np.linspace(0, 100, 1200)
    S = 1.4 + 0.004 * f  # smooth broadband floor
    lines = [(30.0, 6.0, 9.0), (62.0, 12.0, 5.5)]  # center, gamma, H
    H_L = np.zeros_like(f)
    for f0, gamma, H in lines:
        H_L += H / (1.0 + ((f - f0) / gamma) ** 2)
    M = S + H_L

    ax.plot(f, S, color="#4f9d5a", lw=2.2, ls="--", label="floor S(f)")
    ax.plot(f, M, color="#3b6fa0", lw=2.6, label="M(f) = S + sum H . L")
    ax.fill_between(f, S, M, color="#3b6fa0", alpha=0.12)
    for f0, _gamma, H in lines:
        ax.annotate(
            f"H = {H:.1f}",
            xy=(f0, S[np.argmin(np.abs(f - f0))] + H),
            xytext=(f0 + 8, S[np.argmin(np.abs(f - f0))] + H - 2.0),
            fontsize=10,
            arrowprops=dict(arrowstyle="->", color="#333333"),
        )
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("power")
    ax.set_ylim(0, max(M) * 1.28)
    ax.set_title("Marginal PSD: floor plus lines sum to M, one number per cell")
    ax.legend(frameon=False, fontsize=10, loc="upper left")
    fig.tight_layout()
    fig.savefig(ASSETS / "marginalization.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_tooth_def():
    """Tiny sketch defining 'tooth' = peak-minus-local-floor, in dB."""
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    f = np.linspace(-20, 20, 400)
    floor_db = -18.0
    peak_db = -8.0
    gamma = 3.0
    y = floor_db + (peak_db - floor_db) / (1.0 + (f / gamma) ** 2)
    ax.plot(f, y, color="#3b6fa0", lw=2.4)
    ax.axhline(floor_db, color="#4f9d5a", lw=1.6, ls="--")
    ax.annotate(
        "",
        xy=(0, peak_db),
        xytext=(0, floor_db),
        arrowprops=dict(arrowstyle="<->", color="#a03b3b", lw=1.6),
    )
    ax.text(1.0, (peak_db + floor_db) / 2, "tooth (dB)", color="#a03b3b", fontsize=11)
    ax.text(-19, floor_db + 0.6, "local floor", color="#4f9d5a", fontsize=10)
    ax.set_xlabel("frequency offset (Hz)")
    ax.set_ylabel("power (dB)")
    ax.set_xticks([])
    fig.tight_layout()
    fig.savefig(ASSETS / "tooth_def.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_whittle_cost():
    """The Whittle cost of one cell: U-curve + the two-bar line-cell example."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 4.0))

    P = 10.0  # observed power in the cell
    S = np.linspace(1.0, 30.0, 300)
    cost = P / S + np.log(S)
    ax1.plot(S, cost, color="#3b6fa0", lw=2.5)
    ax1.axvline(P, color="#a03b3b", ls="--", lw=1.5)
    ax1.scatter([P], [P / P + np.log(P)], color="#a03b3b", zorder=5)
    ax1.annotate(
        "minimum at\nclaimed power = observed power",
        xy=(P, P / P + np.log(P)),
        xytext=(14, 6.5),
        fontsize=10,
        arrowprops=dict(arrowstyle="->", color="#a03b3b"),
    )
    ax1.set_xlabel("claimed power S (a cell's model)")
    ax1.set_ylabel("cost = P/S + log S")
    ax1.set_title("The cost of one cell")

    S0 = 1.0
    cost_floor = P / S0 + np.log(S0)
    S_line = P * 0.98
    cost_line = P / S_line + np.log(S_line)
    labels = ["priced as floor\n(S = S0)", "priced as line\n(S = P)"]
    vals = [cost_floor, cost_line]
    bars = ax2.bar(labels, vals, color=["#a03b3b", "#3b6fa0"], width=0.55)
    for b, v in zip(bars, vals):
        ax2.text(b.get_x() + b.get_width() / 2, v + 0.15, f"{v:.1f}", ha="center", fontsize=12)
    ax2.set_ylabel("cost of the cell")
    ax2.set_title("One loud cell: floor vs. line pricing")

    fig.tight_layout()
    fig.savefig(ASSETS / "whittle_cost.png", dpi=200)
    plt.close(fig)


def fig_lorentzian_bumps():
    """Three-bump figure: Lorentzian half-width at k=5, 20, 40."""
    fig, ax = plt.subplots(figsize=(9.0, 3.6))
    f = np.linspace(-40, 40, 800)
    for k, color in zip([5, 20, 40], ["#3b6fa0", "#4f9d5a", "#a03b3b"]):
        gamma = max(0.6 * k, 3.0)
        y = 1.0 / (1.0 + (f / gamma) ** 2)
        ax.plot(f, y, color=color, lw=2.2, label=f"k={k}  (gamma={gamma:.0f} Hz)")
        ax.axvspan(-gamma, gamma, color=color, alpha=0.06)
    ax.set_xlabel("frequency offset from k*r(t) (Hz)")
    ax.set_ylabel("normalized line power")
    ax.set_title("Lorentzian line shape widens with harmonic order")
    ax.legend(frameon=False, fontsize=10)
    fig.tight_layout()
    fig.savefig(ASSETS / "lorentzian_bumps.png", dpi=200)
    plt.close(fig)


def fig_trajectory_margins():
    """Per-window J_v4/cell margins: refined vs. telemetry, and refined vs. the
    adversarial (multistart) fan, from the v4 rescore campaign.

    Source: results/joint_rescore_v4/summary.json (key total_v4_per_cell,
    lower = better fit). Margin = other_arm - refined, so positive means
    refined wins. Two panels because the fan margins span two orders of
    magnitude (0.004 to 1.3) while the telemetry margins are all small.
    """
    import json

    summary_path = pathlib.Path(
        "/home/flyingleafe/Research/PhD/projects/harmonic-noise-suppression"
        "/results/joint_rescore_v4/summary.json"
    )
    data = json.loads(summary_path.read_text())["table"]

    labels = {
        "FLY124__w04": "FLY124 w04",
        "FLY124__w05": "FLY124 w05",
        "free-flight_nosource_room1__w01": "DREGON w01",
        "free-flight_speech-low_room1__w01": "DREGON speech w01",
        "free-flight_whitenoise-low_room1__w01": "DREGON noise w01",
    }
    key = "total_v4_per_cell"
    names, tel_margins, fan_margins = [], [], []
    for win, label in labels.items():
        arms = data[win]
        rf = arms["refined"][key]
        names.append(label)
        tel_margins.append(arms["telemetry"][key] - rf)
        fan_margins.append(arms["multistart"][key] - rf)

    y = np.arange(len(names))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.6, 4.6), sharey=True)

    colors1 = ["#3b6fa0" if m >= 0 else "#a03b3b" for m in tel_margins]
    ax1.barh(y, tel_margins, color=colors1, height=0.5)
    ax1.axvline(0, color="black", lw=0.8)
    ax1.set_yticks(y, names, fontsize=12.5)
    ax1.invert_yaxis()
    span1 = max(tel_margins) - min(tel_margins)
    pad1 = span1 * 0.35
    ax1.set_xlim(min(tel_margins) - pad1, max(tel_margins) + pad1)
    for yi, m in zip(y, tel_margins):
        off = span1 * 0.05
        ax1.text(
            m + (off if m >= 0 else -off),
            yi,
            f"{m:+.3f}",
            va="center",
            ha="left" if m >= 0 else "right",
            fontsize=11,
        )
    ax1.set_xlabel("refined vs. telemetry\n(J_v4/cell, + = refined wins)", fontsize=11.5)
    ax1.tick_params(axis="x", labelsize=9.5)
    ax1.set_xticks([-0.015, -0.005, 0.005, 0.015])

    ax2.barh(y, fan_margins, color="#a03b3b", height=0.5)
    ax2.set_xscale("log")
    ax2.axvline(0.004, color="#555555", lw=0.8, ls=":")
    for yi, m in zip(y, fan_margins):
        ax2.text(m * 1.15, yi, f"{m:.3f}", va="center", ha="left", fontsize=11)
    ax2.set_xlabel(
        "refined vs. fan (multistart)\n(J_v4/cell, always > 0 = refined wins)", fontsize=11.5
    )
    ax2.set_xlim(0.002, 3.0)
    ax2.tick_params(axis="x", labelsize=10)

    fig.suptitle(
        "Refined labels vs. telemetry and the adversarial fan, by window", fontsize=13.5, y=1.0
    )
    fig.tight_layout()
    fig.savefig(ASSETS / "trajectory_margins.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_generator_ab():
    """Copy the two generator A/B figures from the scratchpad recipe.

    Both are built by ``scratchpad/make_gen_ab.py``, which re-synthesises each
    arm's audio for evaluation chunk 001 of ``free-flight_nosource_room1``
    (t = 8-12 s, 16 kHz) through ``scripts/eval_gen_comb_real.py`` and draws a
    fresh spectrogram beside the per-harmonic line spectrum from
    ``results/gen_comb_real/per_k.csv``. They are NOT rebuilt here: the old
    body cropped rows out of ``illustration_chunk001.png``, which stretched and
    mis-cut the panels."""
    ASSETS.mkdir(exist_ok=True)
    for fname in ("gen_ab_shared.png", "gen_ab_perrotor.png"):
        srcp = SCRATCH / fname
        if srcp.exists():
            shutil.copy(srcp, ASSETS / fname)
        else:
            print(f"MISSING generator A/B figure: {srcp}")


def fig_pipeline():
    """Small pipeline diagram for the closing formula-chain slide (comment 4):
    recording -> v4 fit -> (H, S) -> regression targets -> generator -> render.
    """
    stages = [
        "recording",
        "v4 fit",
        "(H, S)",
        "regression\ntargets",
        "generator",
        "render",
    ]
    fig, ax = plt.subplots(figsize=(11, 1.6))
    n = len(stages)
    xs = np.linspace(0.5, n - 0.5, n)
    for x, label in zip(xs, stages):
        ax.add_patch(
            plt.Rectangle(
                (x - 0.42, 0.15),
                0.84,
                0.7,
                fill=True,
                facecolor="#eef0f4",
                edgecolor="#333333",
                linewidth=1.0,
            )
        )
        ax.text(x, 0.5, label, ha="center", va="center", fontsize=11)
    for x0, x1 in zip(xs[:-1], xs[1:]):
        ax.annotate(
            "",
            xy=(x1 - 0.44, 0.5),
            xytext=(x0 + 0.44, 0.5),
            arrowprops=dict(arrowstyle="->", lw=1.3, color="#333333"),
        )
    ax.set_xlim(0, n)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(ASSETS / "pipeline.png", dpi=200)
    plt.close(fig)


def main():
    ASSETS.mkdir(exist_ok=True)
    copy_spectrogram_panels()
    crop_dregon_original_panel()
    fig_linewidth_law()
    fig_whittle_cost()
    fig_lorentzian_bumps()
    fig_marginalization()
    fig_tooth_def()
    fig_trajectory_margins()
    fig_generator_ab()
    fig_pipeline()


if __name__ == "__main__":
    main()
