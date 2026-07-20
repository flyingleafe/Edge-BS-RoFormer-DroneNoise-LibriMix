#!/usr/bin/env python3
"""Assets for the DREGON-analysis / generator-design slide deck.

The deck reuses the figures already generated for the companion report
(``writing/reports/2026-07-18_dregon-analysis-and-generator-design``); this
script just copies them in so the deck is self-contained. Regenerate the source
figures there (``make figures``) before running this if the underlying data or
plots change.
"""

from __future__ import annotations

import pathlib
import shutil

import numpy as np

HERE = pathlib.Path(__file__).parent
ASSETS = HERE / "assets"
ROOT = HERE.resolve().parents[2]
REPORT_ASSETS = (
    ROOT / "writing" / "reports" / "2026-07-18_dregon-analysis-and-generator-design" / "assets"
)
RESULTS = ROOT / "results"

FIGURES = [
    "geo_propagation_phase.png",
    "geo_frame_alignment.png",
    "geo_summary.png",
    "fig_per_rotor.png",
    "fig_wind_schema.png",
]

# (source relative to results/, dest name in assets/)
RESULTS_FIGURES = [
    ("vk_tracking/blind_annotation/vit2dsp_free-flight_nosource_room1.png", "vk_blind_dregon.png"),
    ("jasa_gp/loudness.png", "jasa_gp_loudness.png"),
]


def make_vk_coupling_schematic(dest: pathlib.Path) -> None:
    """A slimmed 1-panel schematic of the VK coupling term: two near-coincident
    rotor combs compete for the same spectral energy (explaining-away)."""
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


def make_jasa_gp_eval_slim(dest: pathlib.Path) -> None:
    """Crop the dense 8x2 eval_V7.png panel grid down to the 2 clearest,
    highest-correlation mic rows (dense multi-panel is unreadable at slide
    size; regenerating a slimmed crop is clearer than shrinking it)."""
    from PIL import Image

    src = RESULTS / "jasa_gp" / "eval_V7.png"
    if not src.exists():
        raise SystemExit(f"Missing {src}")
    im = Image.open(src)
    w, h = im.size
    rh = h / 8
    # row 0: mic (-30,0) corr=0.79 ; row 6: mic (-30,50) corr=0.83 (best fit)
    row0 = im.crop((0, 0, w, int(rh)))
    row6 = im.crop((0, int(6 * rh), w, int(7 * rh)))
    # The right (spectrum) column is a flat line 0-500 Hz with all the
    # informative harmonic spikes crammed under ~300 Hz; crop it to that
    # band so it isn't mostly dead flat space next to the readable
    # time-domain overlays on the left.
    # add a little headroom so the rightmost x-tick label (e.g. "300") isn't
    # sliced off by the crop edge
    spec_w = int(w * 0.33)
    half = int(w * 0.5)

    def crop_row(row):
        left = row.crop((0, 0, half, row.height))
        right = row.crop((half, 0, half + spec_w, row.height))
        combined = Image.new("RGB", (left.width + right.width, row.height), "white")
        combined.paste(left, (0, 0))
        combined.paste(right, (left.width, 0))
        return combined

    row0 = crop_row(row0)
    row6 = crop_row(row6)
    out = Image.new("RGB", (row0.width, row0.height + row6.height), "white")
    out.paste(row0, (0, 0))
    out.paste(row6, (0, row0.height))
    out.save(dest)
    print(f"generated {dest.name}")


def main() -> None:
    ASSETS.mkdir(exist_ok=True)
    missing = []
    for name in FIGURES:
        src = REPORT_ASSETS / name
        if src.exists():
            shutil.copy2(src, ASSETS / name)
            print(f"copied {name}")
        else:
            missing.append(name)
    for rel, dest_name in RESULTS_FIGURES:
        src = RESULTS / rel
        if src.exists():
            shutil.copy2(src, ASSETS / dest_name)
            print(f"copied {dest_name}")
        else:
            missing.append(rel)
    if missing:
        raise SystemExit(
            "Missing source figures: "
            + ", ".join(missing)
            + f"\nRun `make figures` in {REPORT_ASSETS.parent} first."
        )

    make_vk_coupling_schematic(ASSETS / "vk_coupling_schematic.png")
    make_jasa_gp_eval_slim(ASSETS / "jasa_gp_eval_slim.png")


if __name__ == "__main__":
    main()
