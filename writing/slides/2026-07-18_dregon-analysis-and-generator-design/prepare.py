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

# spectrogram grid from the 07-17 generator-variants report (real vs
# old/v1/v2/v3, DREGON + Michael's) — shows the geometry-fix win directly
GENERATOR_VARIANTS_REPORT = (
    ROOT / "writing" / "reports" / "2026-07-17_generator-corrected-geometry-variants" / "assets"
)

# (source relative to results/, dest name in assets/)
RESULTS_FIGURES = [
    ("jasa_gp/loudness.png", "jasa_gp_loudness.png"),
]

# (source relative to repo root, dest name in assets/)
ROOT_FIGURES = [
    (
        "omnirun-outputs/python-fbd20f/results/gp_egonoise/dregon/overlay_R60_s0.png",
        "gp_dregon_overlay.png",
    ),
]


def make_four_way_spectrograms(dest: pathlib.Path) -> None:
    """Pull the real|CONA|deep|GP spectrogram grid straight out of the
    already-executed noise_four_way_comparison.ipynb (cell 6's cached PNG
    output) rather than re-running the notebook (which needs the CONA/GP/
    deep-generator checkpoints loaded)."""
    import base64
    import json

    nb_path = ROOT / "notebooks" / "noise_four_way_comparison.ipynb"
    nb = json.loads(nb_path.read_text())
    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        if "spectrogram grid" not in src:
            continue
        for out in cell.get("outputs", []):
            data = out.get("data", {}).get("image/png")
            if data:
                if isinstance(data, list):
                    data = "".join(data)
                dest.write_bytes(base64.b64decode(data))
                print(f"generated {dest.name} (extracted from notebook cache)")
                return
    raise SystemExit(
        "Could not find the spectrogram-grid cell output in "
        f"{nb_path} — re-run the notebook to regenerate it."
    )


def annotate_four_way_cona(dest: pathlib.Path) -> None:
    """The CONA panel's black region (t>0.65s, f>~200Hz) is a real dataset
    artefact (broadband truncation bug + tonal-only content), not a rendering
    glitch — label it so it doesn't read as broken."""
    from PIL import Image, ImageDraw, ImageFont

    im = Image.open(dest).convert("RGB")
    w, h = im.size
    pad = 46
    out = Image.new("RGB", (w, h + pad), "white")
    out.paste(im, (0, 0))
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 15)
    except OSError:
        font = ImageFont.load_default()
    # CONA is the 2nd of 4 panels, roughly [0.25, 0.5) of the width
    cx = int(w * 0.375)
    text = "tonal-only comb; broadband truncated at 0.65 s (dataset bug)"
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    draw.text((cx - tw / 2, h + 10), text, fill="#a83232", font=font)
    out.save(dest)
    print(f"annotated {dest.name} (CONA truncation note)")


def make_vk_blind_overlay_slim(dest: pathlib.Path) -> None:
    """Crop the blind-annotation overlay to just the RPS-trajectory panel
    (drop the illegible log-error panel below it) so legend/axis text reads
    at slide scale."""
    from PIL import Image

    src = RESULTS / "vk_tracking/blind_annotation/vit2dsp_free-flight_nosource_room1.png"
    im = Image.open(src)
    w, h = im.size  # 1800 x 1350
    # top RPS-overlay panel occupies roughly the top 49% of the figure
    crop = im.crop((0, 0, w, int(h * 0.49)))
    crop.save(dest)


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
    # Take the whole right half (not just a band) so the closed right spine
    # of the plot box is included, not cut mid-air (round-2 critique fix).
    spec_w = int(w * 0.5)
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
    # the source panels (eval_V7.png) carry no axis-unit labels, only bare
    # tick numbers — add them here so the crop is self-explanatory on a slide.
    from PIL import ImageDraw

    band_h = 34
    out = Image.new("RGB", (row0.width, row0.height + row6.height + band_h), "white")
    out.paste(row0, (0, 0))
    out.paste(row6, (0, row0.height))
    draw = ImageDraw.Draw(out)
    y = row0.height + row6.height + 4
    draw.text((row0.width * 0.22, y), "time (s)", fill="black")
    draw.text((row0.width * 0.74, y), "frequency (Hz) - spectral amplitude", fill="black")
    out.save(dest)
    print(f"generated {dest.name}")


def _geometry_panel(ax, mic, rotor_pos, title):
    import matplotlib.pyplot as plt  # noqa: F401 (ax already bound to a figure)

    ax.scatter(
        rotor_pos[:, 0], rotor_pos[:, 1], c="C3", marker="x", s=110, label="rotors", zorder=3
    )
    ax.scatter(mic[:, 0], mic[:, 1], c="C0", marker="o", s=55, label="mics", zorder=3)
    for i, p in enumerate(mic):
        ax.annotate(str(i), (p[0], p[1]), fontsize=8, xytext=(3, 3), textcoords="offset points")
    ax.set_title(title, fontsize=11)
    ax.set_aspect("equal")
    ax.axhline(0, color="0.85", lw=0.6, zorder=0)
    ax.axvline(0, color="0.85", lw=0.6, zorder=0)
    ax.set_xlabel("x (m)", fontsize=9)
    ax.set_ylabel("y (m)", fontsize=9)


def make_dregon_geometry_positions(dest: pathlib.Path) -> None:
    """Top-down (x,y) scatter of mic + rotor positions, before/after the
    DREGON 180 deg frame fix."""
    import sys

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sys.path.insert(0, str(ROOT))
    from src.data_processing import dregon

    dregon_dir = ROOT / "data" / "DREGON"
    mic_fixed, rotor = dregon.get_geometry(dregon_dir)
    # undo the 180 deg z-flip to recover the shipped ("wrong") frame
    mic_wrong = mic_fixed.copy()
    mic_wrong[:, 0] *= -1
    mic_wrong[:, 1] *= -1

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
    _geometry_panel(axes[0], mic_wrong, rotor, "shipped micPos.txt frame (WRONG)")
    _geometry_panel(axes[1], mic_fixed, rotor, "after 180deg z-flip (FIXED)")
    axes[0].legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(dest, dpi=150)
    plt.close(fig)
    print(f"generated {dest.name}")


def make_michaels_geometry_positions(dest: pathlib.Path) -> None:
    """Top-down (x,y) scatter of mic + rotor positions, wrong (vertical ring
    coded into the x-y plot) vs fixed (horizontal ring) for Michael's rig."""
    import sys

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sys.path.insert(0, str(ROOT))
    from src.data_processing import michaels

    mic_m, rotor_m = michaels.get_geometry()
    # the bug: ring coded as vertical (x-z plane) instead of horizontal (x-y)
    mic_m_wrong = mic_m.copy()
    mic_m_wrong[:, 1], mic_m_wrong[:, 2] = mic_m[:, 2] - mic_m[:, 2].mean(), mic_m[:, 1]

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
    _geometry_panel(
        axes[0],
        mic_m_wrong[:, [0, 1]],
        rotor_m[:, [0, 1]],
        "vertical-ring bug (x-z plotted as x-y)",
    )
    _geometry_panel(axes[1], mic_m[:, [0, 1]], rotor_m[:, [0, 1]], "horizontal ring (FIXED)")
    axes[0].legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(dest, dpi=150)
    plt.close(fig)
    print(f"generated {dest.name}")


def make_vk_speedup_bars(dest: pathlib.Path) -> None:
    """Before/after real-time-factor bars for the VK fast-path optimization."""
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


def make_rps_predictor_quadrant(dest: pathlib.Path) -> None:
    """Sketch: speed (x, log rtf) vs accuracy (y, lower err better) for
    VK now, neural predictor now, and the target region."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    # x: log10(rtf), higher = faster than real time
    pts = {
        "VK blind (today)": (np.log10(0.95), 0.68, "C0", (8, -12)),
        "VK refine (telemetry)": (np.log10(0.38), 0.604, "C0", (-10, 14)),
        "neural predictor (now)": (np.log10(50), 5.4, "C3", (-90, 10)),
        "target": (np.log10(20), 0.7, "0.3", (10, 6)),
    }
    for label, (x, y, c, off) in pts.items():
        marker = "*" if label == "target" else "o"
        s = 260 if label == "target" else 140
        ax.scatter([x], [y], c=c, marker=marker, s=s, zorder=3)
        ax.annotate(label, (x, y), fontsize=9, xytext=off, textcoords="offset points")
    ax.axvline(0, color="0.85", lw=0.8)
    ax.text(0.05, 6.0, "real time →", fontsize=8, color="0.5")
    ax.set_xlabel(r"speed: $\log_{10}$(real-time factor)  (right = faster)")
    ax.set_ylabel("pooled PIT error (rev/s, lower = better)")
    ax.set_title("Neural RPS prediction: fast but coarse; VK: accurate but slow", fontsize=10)
    ax.set_ylim(6.2, -0.3)
    fig.tight_layout()
    fig.savefig(dest, dpi=150)
    plt.close(fig)
    print(f"generated {dest.name}")


def annotate_generator_variants_grid(dest: pathlib.Path) -> None:
    """Overlay human row labels + the free-flight match score for each
    variant onto the copied real-vs-generated spectrogram grid, and pad the
    bottom so the "Time (s)" axis labels aren't clipped by the slide."""
    from PIL import Image, ImageDraw, ImageFont

    im = Image.open(dest).convert("RGB")
    w, h = im.size
    left_pad = 210
    bottom_pad = 40
    out = Image.new("RGB", (w + left_pad, h + bottom_pad), "white")
    out.paste(im, (left_pad, 0))
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    except OSError:
        font = font_small = ImageFont.load_default()

    # 5 rows: real, old, v1, v2, v3 — each occupies an equal vertical band
    # (the top title strip is small; rows start right after it).
    n_rows = 5
    top_margin = int(h * 0.045)
    row_h = (h - top_margin) / n_rows
    row_labels = ["REAL", "OLD\nwrong geom", "v1\ncorrected", "v2\nper-rotor", "v3\nwind"]
    # free-flight match scores (higher = better), from the scores table
    row_scores = [None, "score 4.51", "score 5.22", "score 4.82", "score 3.44"]

    for i, (label, score) in enumerate(zip(row_labels, row_scores)):
        cy = top_margin + row_h * (i + 0.5)
        lines = label.split("\n")
        ty = cy - 12 * len(lines)
        for line in lines:
            bbox = draw.textbbox((0, 0), line, font=font)
            tw = bbox[2] - bbox[0]
            draw.text((left_pad - 20 - tw, ty), line, fill="black", font=font)
            ty += 24
        if score:
            bbox = draw.textbbox((0, 0), score, font=font_small)
            tw = bbox[2] - bbox[0]
            draw.text(
                (left_pad - 20 - tw, cy + 12 * len(lines) + 2),
                score,
                fill="#1a6b3c",
                font=font_small,
            )

    out.save(dest)
    print(f"annotated {dest.name} (row labels + free-flight scores)")


def make_subembed_schema(dest: pathlib.Path) -> None:
    """Mini schema for per-rotor sub-embeddings: encoder -> z_drone ->
    [+ delta z_r] x4 -> per-rotor decoder, matching the fig_wind_schema.png
    visual style (grey/blue/orange boxes, black arrows)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    fig, ax = plt.subplots(figsize=(9.5, 3.4))
    ax.set_xlim(0, 10.5)
    ax.set_ylim(0, 4.2)
    ax.axis("off")

    def box(x, y, w, h, text, color, textcolor="white", fontsize=10):
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.05,rounding_size=0.08",
                facecolor=color,
                edgecolor="none",
            )
        )
        ax.text(
            x + w / 2,
            y + h / 2,
            text,
            ha="center",
            va="center",
            color=textcolor,
            fontsize=fontsize,
            fontweight="bold",
        )

    def arrow(x0, y0, x1, y1, style="-|>"):
        ax.add_patch(
            FancyArrowPatch(
                (x0, y0), (x1, y1), arrowstyle=style, color="black", lw=1.4, mutation_scale=14
            )
        )

    box(0.2, 1.5, 1.8, 1.0, "encoder", "#5f6b7a")
    box(2.5, 1.5, 1.7, 1.0, "$z_\\mathrm{drone}$", "#5f6b7a")

    # four per-rotor offset boxes, delta z highlighted in orange
    rotor_y = [3.4, 2.55, 1.7, 0.85]
    for i, y in enumerate(rotor_y):
        box(
            5.0,
            y - 0.35,
            1.9,
            0.7,
            f"$z_\\mathrm{{drone}} + \\delta z_{i + 1}$",
            "#c9752a",
            fontsize=9,
        )
        arrow(4.2, 2.0, 5.0, y)
        box(7.3, y - 0.35, 1.6, 0.7, f"decoder$_{i + 1}$", "#5f6b7a", fontsize=9)
        arrow(6.9, y, 7.3, y)

    arrow(2.0, 2.0, 2.5, 2.0)
    arrow(4.2, 2.0, 4.2, 2.0)  # placeholder (kept for clarity of flow)

    ax.text(
        5.95,
        4.0,
        r"$\delta z \in \mathbb{R}^{4 \times d}$, zero-init, $\lambda \Vert \delta z \Vert_2^2$",
        ha="center",
        fontsize=10,
        color="#c9752a",
        fontweight="bold",
    )
    ax.text(1.1, 0.9, "one shared\ncode", ha="center", fontsize=9, color="0.3")

    fig.tight_layout()
    fig.savefig(dest, dpi=150)
    plt.close(fig)
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
    for rel, dest_name in ROOT_FIGURES:
        src = ROOT / rel
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

    variant_grid_src = GENERATOR_VARIANTS_REPORT / "spectrograms.png"
    if variant_grid_src.exists():
        shutil.copy2(variant_grid_src, ASSETS / "generator_variants_spectrograms.png")
        print("copied generator_variants_spectrograms.png")
    else:
        raise SystemExit(f"Missing {variant_grid_src}")
    annotate_generator_variants_grid(ASSETS / "generator_variants_spectrograms.png")

    make_four_way_spectrograms(ASSETS / "four_way_spectrograms.png")
    annotate_four_way_cona(ASSETS / "four_way_spectrograms.png")
    make_vk_blind_overlay_slim(ASSETS / "vk_blind_dregon.png")
    make_vk_coupling_schematic(ASSETS / "vk_coupling_schematic.png")
    make_jasa_gp_eval_slim(ASSETS / "jasa_gp_eval_slim.png")
    make_dregon_geometry_positions(ASSETS / "dregon_geometry_positions.png")
    make_michaels_geometry_positions(ASSETS / "michaels_geometry_positions.png")
    make_vk_speedup_bars(ASSETS / "vk_speedup_bars.png")
    make_rps_predictor_quadrant(ASSETS / "rps_predictor_quadrant.png")
    make_subembed_schema(ASSETS / "fig_subembed_schema.png")


if __name__ == "__main__":
    main()
