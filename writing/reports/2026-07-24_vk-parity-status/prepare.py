#!/usr/bin/env python3
"""Assets for the VK-parity status report.

Most figures are cropped/regenerated from artifacts already produced by the
07-18 report/deck and the vk_tracking / vk_blind_sweep / rps_predictor_vk_eval
result trees — this script assembles them, it does not re-run experiments.
"""

from __future__ import annotations

import pathlib

import numpy as np

HERE = pathlib.Path(__file__).parent
ASSETS = HERE / "assets"
ROOT = HERE.resolve().parents[2]
RESULTS = ROOT / "results"


def make_vk_coupling_schematic(dest: pathlib.Path) -> None:
    """Two near-coincident rotor combs compete for the same spectral energy
    (explaining-away) — same recipe as the 07-18 deck."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 3.0))
    f = np.linspace(0, 500, 2000)

    def comb(f0, n, width, amp):
        y = np.zeros_like(f)
        for k in range(1, n + 1):
            y += amp / k * np.exp(-((f - k * f0) ** 2) / (2 * width**2))
        return y

    y1 = comb(74.0, 6, 3.0, 1.0)
    y2 = comb(75.5, 6, 3.0, 1.0)  # twin, nearly overlapping
    spec = y1 + y2 + 0.02

    ax.fill_between(f, 0, spec, color="0.85", label="observed |Y(f)|", zorder=1)
    ax.plot(f, y1, color="C0", lw=1.6, label=r"track $m$: $a_m(t)$")
    ax.plot(f, y2, color="C3", lw=1.6, label=r"track $m'$ (twin): $a_{m'}(t)$")

    for k in [3, 4, 5]:
        x0, x1 = k * 74.0, k * 75.5
        ax.annotate(
            "",
            xy=(x1, 0.55),
            xytext=(x0, 0.55),
            arrowprops=dict(arrowstyle="<->", color="black", lw=1.0),
        )
    ax.text(4 * 74.0, 0.65, "compete for the\nsame spectral energy", ha="center", fontsize=9)

    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("magnitude (a.u.)")
    ax.set_yticks([])
    ax.set_xlim(0, 480)
    ax.set_ylim(0, 1.15)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax.set_title(
        "Coupling term forces overlapping tracks to explain away shared energy", fontsize=10
    )
    fig.tight_layout()
    fig.savefig(dest, dpi=150)
    plt.close(fig)
    print(f"generated {dest.name}")


def make_dregon_blind_overlay(dest: pathlib.Path) -> None:
    """DREGON blind re-annotation: full figure (trajectories + error panel)."""
    import shutil

    src = RESULTS / "vk_tracking/blind_annotation/vit2dsp_free-flight_nosource_room1.png"
    shutil.copy2(src, dest)
    print(f"copied {dest.name}")


def make_fly124_blind_overlay(dest: pathlib.Path) -> None:
    """Build the FLY124 blind trajectory-overlay figure from the round-6 sweep
    npz (arm R, the load-bearing residual re-scan). `edge` is a KEEP mask."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    src = (
        RESULTS
        / "vk_blind_sweep_r6/omnirun-outputs/python-4a015c/results/vk_blind_sweep"
        / "FLY124-cruise__vit2dsp__R.npz"
    )
    d = np.load(src, allow_pickle=True)
    ft = d["ft"]
    edge = d["edge"].astype(bool)  # KEEP mask, not exclude
    measured = d["measured_sm"]
    final = d["stage_snaps"][-1]  # last ladder stage (post-refine, pre-guard-revert view)
    R = final.shape[0]

    fig, ax = plt.subplots(figsize=(9, 4.2))
    colors = ["C3", "C0", "C2", "C4"]
    for r in range(R):
        ax.plot(
            ft[edge],
            measured[r][edge],
            color="0.35",
            lw=1.0,
            ls="--",
            label="ground truth" if r == 0 else None,
        )
        ax.plot(ft[edge], final[r][edge], color=colors[r % 4], lw=1.7, label=f"blind rotor {r}")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("rev/s")
    ax.set_title("FLY124-cruise: blind annotation (arm R) vs ground-truth telemetry", fontsize=11)
    ax.legend(loc="upper right", fontsize=8, ncol=3)
    fig.tight_layout()
    fig.savefig(dest, dpi=150)
    plt.close(fig)
    print(f"generated {dest.name}")


def make_vk_speedup_bars(dest: pathlib.Path) -> None:
    """Before/after real-time-factor bars for the CPU fast-inference paths."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    labels = ["refine\n(telemetry-init)", "blind\n(no telemetry)"]
    before = [0.037, 0.34]
    after = [0.38, 0.95]
    x = np.arange(len(labels))
    w = 0.32
    ax.bar(x - w / 2, before, width=w, color="0.7", label="before (splu, full demod)")
    ax.bar(x + w / 2, after, width=w, color="C0", label="after (banded Cholesky + pruning)")
    for xi, v in zip(x - w / 2, before):
        ax.text(xi, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
    for xi, v in zip(x + w / 2, after):
        ax.text(xi, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
    ax.axhline(1.0, color="C3", ls="--", lw=1, label="real-time (rtf=1)")
    ax.set_xticks(x, labels)
    ax.set_ylabel("real-time factor (audio-s / wall-s)")
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title("~10x faster: 2.9x banded Cholesky x 1.7x pair pruning x fixes", fontsize=10)
    fig.tight_layout()
    fig.savefig(dest, dpi=150)
    plt.close(fig)
    print(f"generated {dest.name}")


def make_parity_bars(dest: pathlib.Path) -> None:
    """Bar chart: blind-VK precision reference vs the neural arms tried so
    far (phase A best-smoothing, phase B best-context), on both drones."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    groups = ["DREGON free-flight cruise\n(pooled, three recordings)", "FLY124 cruise\n(pooled)"]
    # blind VK: DREGON range 0.680-0.744, use midpoint marker with error bar;
    # FLY124: single number 1.027
    vk_vals = [0.71, 1.027]
    vk_err = [0.032, 0.0]
    phase_a = [2.62, 1.55]  # e12_transformer_best, best smoothing arm, ch0
    phase_b = [2.87, 1.90]  # best native-context arm (8s/4s, best smoothing)

    x = np.arange(len(groups))
    w = 0.25
    ax.bar(
        x - w,
        vk_vals,
        yerr=vk_err,
        width=w,
        color="C2",
        label="blind VK (precision reference)",
        capsize=3,
    )
    ax.bar(x, phase_a, width=w, color="C0", label="neural: phase A (test-time smoothing)")
    ax.bar(x + w, phase_b, width=w, color="C3", label="neural: phase B (4s/8s native context)")
    for xi, vals in zip([x - w, x, x + w], [vk_vals, phase_a, phase_b]):
        for xii, v in zip(xi, vals):
            ax.text(xii, v + 0.05, f"{v:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x, groups, fontsize=9)
    ax.set_ylabel("pooled PIT-MAE (rev/s, lower = better)")
    ax.set_ylim(0, max(phase_b) * 1.35)
    ax.set_title(
        "Neural parity gap: both training-side levers tried so far fall short", fontsize=10
    )
    ax.legend(fontsize=8, loc="upper right", ncol=1)
    fig.tight_layout()
    fig.savefig(dest, dpi=150)
    plt.close(fig)
    print(f"generated {dest.name}")


def make_alias_illustration(dest: pathlib.Path) -> None:
    """Illustrate the FLY124 seeding failure: an alias comb at (2/3)*91=60.7
    Hz (teeth k=3,6,9,... coincide with the true 91 rev/s rotor's even
    harmonics) can outscore a real weak comb in an uncapped scan."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 3.0))
    f = np.linspace(0, 1400, 3000)

    def comb(f0, n, width, amp, klist=None):
        y = np.zeros_like(f)
        ks = klist if klist is not None else range(1, n + 1)
        for k in ks:
            if k > n:
                continue
            y += amp / max(k, 1) * np.exp(-((f - k * f0) ** 2) / (2 * width**2))
        return y

    true_comb = comb(91.0, 15, 4.0, 1.0)
    alias_comb = comb(60.7, 20, 4.0, 0.9, klist=[3, 6, 9, 12, 15, 18])

    ax.fill_between(f, 0, true_comb + 0.02, color="C0", alpha=0.35, label="true comb: 91 rev/s")
    ax.plot(f, alias_comb, color="C3", lw=1.6, label="alias comb: 60.7 rev/s (teeth k=3,6,9,...)")
    ax.axvline(1200, color="0.3", ls="--", lw=1)
    ax.text(1210, 0.08, "scan_f_max\n= 1.2 kHz", fontsize=8, color="0.3", va="bottom")
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("magnitude (a.u.)")
    ax.set_yticks([])
    ax.set_xlim(0, 1400)
    ax.set_ylim(0, 1.15)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title(
        "60.7 ~ (2/3)*91: alias teeth coincide with the true rotor's even harmonics",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(dest, dpi=150)
    plt.close(fig)
    print(f"generated {dest.name}")


def main() -> None:
    ASSETS.mkdir(exist_ok=True)
    make_vk_coupling_schematic(ASSETS / "vk_coupling_schematic.png")
    make_dregon_blind_overlay(ASSETS / "vk_blind_dregon_full.png")
    make_fly124_blind_overlay(ASSETS / "vk_blind_fly124.png")
    make_vk_speedup_bars(ASSETS / "vk_speedup_bars.png")
    make_parity_bars(ASSETS / "vk_parity_bars.png")
    make_alias_illustration(ASSETS / "vk_alias_illustration.png")


if __name__ == "__main__":
    main()
