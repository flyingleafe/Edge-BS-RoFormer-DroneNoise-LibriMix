#!/usr/bin/env python3
"""Figures for the DREGON-analysis / generator-design report.

Two computed figures (cheap CPU, a handful of Welch spectra):

  - ``fig_per_rotor.png`` — DREGON single-motor recordings. Each of the four
    rotors is recorded on its own; normalising out the propagation level (each
    spectrum divided by its own fundamental) leaves a *timbre* signature — the
    relative harmonic profile and broadband floor — and the four rotors' are
    visibly different. This is the empirical motivation for giving each rotor its
    own sub-embedding rather than one shared per-drone code.

  - ``fig_wind_schema.png`` — block schematic of the wind-wake channel
    (modules A/B/C: RPS->airspeed, wake flow field, flow->mic transduction),
    summed incoherently onto the coherent generator.

It also copies the reusable geometry figures from the Stage-0 calibration report
so this report is self-contained.

No GPU, no network — only local ``data/DREGON`` single-motor wavs + the geometry.
"""

from __future__ import annotations

import pathlib
import shutil

import matplotlib
import numpy as np
from scipy.signal import welch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch  # noqa: E402

HERE = pathlib.Path(__file__).parent
ASSETS = HERE / "assets"
ROOT = HERE.resolve().parents[2]  # writing/reports/<dir> -> repo root
DREGON_DIR = ROOT / "data" / "DREGON"
MOTORS_DIR = DREGON_DIR / "DREGON_individual_motors_recordings"
STAGE0_ASSETS = (
    ROOT / "writing" / "reports" / "2026-07-15_mic-array-geometry-calibration" / "assets"
)

# Rotor palette (four physically distinct motors).
ROTOR_COLORS = ["#2b7a52", "#5b8fb9", "#a8541b", "#8a4fb0"]
PHYS = "#3d6b8c"  # physics module fill
LEARN = "#b5651d"  # learned module fill
IO = "#5f6b7a"  # i/o box fill


# ---------------------------------------------------------------------------
# Figure 1 — per-rotor spectral signatures
# ---------------------------------------------------------------------------
def _nearest_mic(sig8: np.ndarray, mic_pos, rotor_xyz) -> int:
    """Nearest mic to this rotor (best SNR / near-field, isolates its source)."""
    try:
        d = np.linalg.norm(np.asarray(mic_pos) - np.asarray(rotor_xyz)[None, :], axis=1)
        return int(np.argmin(d))
    except Exception:
        # Fall back to the loudest channel.
        return int(np.argmax(sig8.std(axis=1)))


def _peak_near(f: np.ndarray, pxx: np.ndarray, fc: float, half: float = 12.0) -> float:
    m = (f >= fc - half) & (f <= fc + half)
    return float(pxx[m].max()) if m.any() else 1e-20


def per_rotor_figure(speed: int = 70, max_seconds: float = 15.0, n_harm: int = 12) -> None:
    import soundfile as sf

    from data_processing.dregon import get_geometry

    try:
        mic_pos, rotor_pos = get_geometry(DREGON_DIR)
    except Exception as exc:  # pragma: no cover - geometry optional
        print(f"[per-rotor] geometry unavailable ({exc}); using loudest channel")
        mic_pos = rotor_pos = None

    specs = {}  # rotor -> (freqs, psd_db_norm)
    harm_prof = {}  # rotor -> (n_harm,) dB relative to fundamental
    for r in range(1, 5):
        wav = MOTORS_DIR / f"Motor{r}_{speed}.wav"
        if not wav.exists():
            print(f"[per-rotor] missing {wav}")
            continue
        info = sf.info(str(wav))
        sr = info.samplerate
        frames = int(max_seconds * sr)
        audio, sr = sf.read(str(wav), frames=frames)
        x8 = np.asarray(audio, dtype=np.float64).T  # (8, N)
        rxyz = rotor_pos[r - 1] if rotor_pos is not None else None
        mic = _nearest_mic(x8, mic_pos, rxyz)
        x = x8[mic]

        nperseg = 8192
        f, pxx = welch(x, fs=sr, nperseg=nperseg, noverlap=nperseg // 2)
        pxx = np.maximum(pxx, 1e-20)

        f0_amp = _peak_near(f, pxx, float(speed))
        psd_db_norm = 10.0 * np.log10(pxx / f0_amp)
        specs[r] = (f, psd_db_norm)

        prof = np.array(
            [10.0 * np.log10(_peak_near(f, pxx, k * speed) / f0_amp) for k in range(1, n_harm + 1)]
        )
        harm_prof[r] = prof

    if not specs:
        raise SystemExit("[per-rotor] no single-motor recordings found; cannot build figure")

    # Spread across rotors of the harmonic profile (dB RMS), harmonics 2..n.
    prof_mat = np.vstack([harm_prof[r] for r in sorted(harm_prof)])  # (R, n_harm)
    spread_per_harm = prof_mat[:, 1:].std(axis=0)  # exclude fundamental (=0)
    rms_spread = float(np.sqrt((spread_per_harm**2).mean()))

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(10.2, 3.8))

    fmax = (n_harm + 0.5) * speed
    for r in sorted(specs):
        f, s = specs[r]
        m = f <= fmax
        axa.plot(f[m], s[m], color=ROTOR_COLORS[r - 1], lw=1.0, label=f"rotor {r}", alpha=0.9)
    for k in range(1, n_harm + 1):
        axa.axvline(k * speed, color="0.8", lw=0.6, zorder=0)
    axa.set_xlim(0, fmax)
    axa.set_ylim(-60, 5)
    axa.set_xlabel("Frequency (Hz)")
    axa.set_ylabel("PSD, normalised to fundamental (dB)")
    axa.set_title(f"(a) Single-motor spectra @ {speed} Hz, level-normalised", fontsize=10)
    axa.legend(fontsize=8, ncol=2, loc="upper right")
    axa.spines[["top", "right"]].set_visible(False)

    ks = np.arange(1, n_harm + 1)
    for r in sorted(harm_prof):
        axb.plot(
            ks, harm_prof[r], "-o", ms=3, color=ROTOR_COLORS[r - 1], lw=1.1, label=f"rotor {r}"
        )
    axb.set_xlabel(f"harmonic index $k$  (freq $= k \\times {speed}$ Hz)")
    axb.set_ylabel("harmonic level re. fundamental (dB)")
    axb.set_title("(b) Per-rotor harmonic profile", fontsize=10)
    axb.set_xticks(ks[::2])
    axb.spines[["top", "right"]].set_visible(False)
    axb.legend(fontsize=8, ncol=2, loc="lower left")
    axb.text(
        0.97,
        0.95,
        f"inter-rotor spread\n{rms_spread:.1f} dB RMS",
        transform=axb.transAxes,
        ha="right",
        va="top",
        fontsize=8.5,
        bbox=dict(boxstyle="round", fc="#fff4e6", ec="#a8541b", alpha=0.9),
    )

    fig.suptitle(
        "DREGON's four rotors are not one source: level-normalised timbre differs by rotor",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = ASSETS / "fig_per_rotor.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}  (inter-rotor harmonic spread {rms_spread:.1f} dB RMS)")

    # Persist the scalar for the report text.
    (ASSETS / "per_rotor_spread.txt").write_text(f"{rms_spread:.1f}\n")


# ---------------------------------------------------------------------------
# Figure 2 — wind-wake channel schematic
# ---------------------------------------------------------------------------
def _box(ax, xy, w, h, text, fc, ec="#333", fontsize: float = 9, tc="white", bold=True):
    x, y = xy
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.06",
            fc=fc,
            ec=ec,
            lw=1.3,
            mutation_scale=1.0,
        )
    )
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=tc,
        weight="bold" if bold else "normal",
        zorder=5,
    )


def _arrow(ax, p0, p1, color="#333", ls: str | tuple = "-"):
    ax.add_patch(
        FancyArrowPatch(
            p0,
            p1,
            arrowstyle="-|>",
            mutation_scale=14,
            lw=1.4,
            color=color,
            ls=ls,
            shrinkA=2,
            shrinkB=2,
            zorder=1,
        )
    )


def wind_schema_figure() -> None:
    fig, ax = plt.subplots(figsize=(10.6, 4.6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5.6)
    ax.axis("off")

    y = 3.2
    h = 1.0
    # Inputs.
    _box(ax, (0.15, y), 1.7, h, "RPS  $r(t)$\n[B, R, T]", IO, tc="white", fontsize=8.5)
    _box(ax, (0.15, 0.5), 1.7, h, "geometry\nmic / rotor pos", IO, tc="white", fontsize=8.5)

    # Module A.
    _box(ax, (2.35, y), 2.05, h, "A — quad\ndynamics\n(grey-box)", PHYS, fontsize=8.5)
    ax.text(3.37, y - 0.28, r"$V_{\mathrm{rel}}(t)$", ha="center", fontsize=8.5, color="#333")
    ax.text(3.37, y + h + 0.18, "physics", ha="center", fontsize=7.5, color=PHYS, style="italic")

    # Module B.
    _box(ax, (4.85, y), 2.35, h, "B — wake\nflow field\n(physics)", PHYS, fontsize=8.5)
    ax.text(6.02, y - 0.30, r"$U_m(t)=\sum_r v_i\, g$", ha="center", fontsize=8.5, color="#333")

    # Module C.
    _box(
        ax,
        (7.7, y),
        2.25,
        h,
        "C — flow$\\rightarrow$mic\ntransduction\n(learned)",
        LEARN,
        fontsize=8.5,
    )
    ax.text(8.82, y + h + 0.18, "learned", ha="center", fontsize=7.5, color=LEARN, style="italic")

    # Output + mix.
    _box(ax, (10.35, y), 1.5, h, "incoherent\nper-mic\n$y_{\\mathrm{wind}}$", IO, fontsize=8.5)

    # Sum node.
    ax.add_patch(Circle((11.1, 1.15), 0.28, fc="white", ec="#333", lw=1.4, zorder=4))
    ax.text(11.1, 1.15, "+", ha="center", va="center", fontsize=15, zorder=5)
    _box(
        ax,
        (7.7, 0.65),
        2.25,
        1.0,
        "coherent generator\n(emitter + $1/r$ + delay)",
        "#3f6f4a",
        fontsize=8.5,
    )
    ax.text(6.0, 0.05, "output field  $y[B, M, T]$", ha="center", fontsize=8.5, color="#333")

    # Arrows through the pipeline.
    _arrow(ax, (1.85, y + h / 2), (2.35, y + h / 2))
    _arrow(ax, (4.4, y + h / 2), (4.85, y + h / 2))
    _arrow(ax, (7.2, y + h / 2), (7.7, y + h / 2))
    _arrow(ax, (9.95, y + h / 2), (10.35, y + h / 2))
    # geometry feeds module B.
    _arrow(ax, (1.85, 1.0), (5.6, y), color=PHYS, ls=(0, (4, 2)))
    # wind + coherent -> sum.
    _arrow(ax, (11.1, y), (11.1, 1.43))
    _arrow(ax, (9.95, 1.15), (10.82, 1.15))
    _arrow(ax, (11.1, 0.87), (11.1, 0.35))
    ax.text(11.1, 0.15, "mix", ha="center", fontsize=8, color="#333")

    ax.text(
        0.15,
        5.25,
        "Physics (fixed, calibrated geometry) decides WHERE the air flows and HOW FAST; "
        "a small learned head decides only WHAT that flow does to a diaphragm.",
        fontsize=9.0,
        color="#333",
        style="italic",
    )
    ax.text(
        0.15,
        4.8,
        "Module A is skipped at hover ($V_{\\mathrm{rel}}=0$).  "
        "Per-mic noise is realised independently $\\Rightarrow$ spatially incoherent by construction.",
        fontsize=8.2,
        color="#666",
    )

    fig.tight_layout()
    out = ASSETS / "fig_wind_schema.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


# ---------------------------------------------------------------------------
# Copy reusable Stage-0 geometry figures
# ---------------------------------------------------------------------------
def copy_stage0() -> None:
    wanted = {
        "fig1_propagation_phase.png": "geo_propagation_phase.png",
        "fig3_frame_alignment.png": "geo_frame_alignment.png",
        "fig7_geometry_summary.png": "geo_summary.png",
    }
    for src_name, dst_name in wanted.items():
        src = STAGE0_ASSETS / src_name
        if src.exists():
            shutil.copy2(src, ASSETS / dst_name)
            print(f"copied {dst_name}")
        else:
            print(f"[stage0] missing {src}")


def main() -> None:
    ASSETS.mkdir(exist_ok=True)
    copy_stage0()
    wind_schema_figure()
    per_rotor_figure()


if __name__ == "__main__":
    main()
