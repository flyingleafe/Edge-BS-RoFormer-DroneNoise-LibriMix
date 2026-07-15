#!/usr/bin/env python3
"""Blind SPCup ego-noise annotation with the coupled Vold-Kalman tracker.

Successor of ``scripts/rps_refinement_spcup.py`` (whose blind base-speed scan +
rotor-count model selection is REUSED verbatim through its cached artifacts):
that pipeline's spline refinement was structurally unable to follow flight
dynamics — refined trajectories through the KU_Leuven maneuvers moved only
~±1.3 rev/s peak-to-peak. Here the same blind init (constant per-rotor speeds
from the predecessor's scan / model selection, read from
``results/rps_refinement/spcup/<rid>.npz``) seeds the two-phase VK recipe
validated on DREGON (``scripts/vk_validation.py``):

1. **Capture phase** — annealed schedule (``k_schedule="grow"``, n_outer=10,
   k_max=30, couple_hz=20): wide early bandwidth pulls the track from the
   constant init onto the real trajectory (basin >= 2 rev/s on DREGON), at the
   cost of a small systematic bias (~-0.4 rev/s measured on DREGON).
2. **Refine phase** — continue from the captured trajectories with the
   de-biasing config (``k_schedule="fixed"``, n_outer=5, k_min=6, k_max=30,
   bw_hz=1.5, max_step=0.3). k_min=6 is essential: merged low harmonics bias
   the frequency update (stage-D lesson).

Honest refusal: recordings whose VK confidence stays ~0 (no comb evidence —
the DREGON *success* band under this config is mean conf ~0.026-0.033, see
``results/vk_tracking/validation``) are reported REFUSED and their overlay
carries no track lines; the predecessor's ``comb_confidence`` refusal
(Idea_ssu, 0.018 < 0.02) is carried along as a triage prior.

Artifacts in ``results/vk_tracking/spcup/``: per-recording overlay PNG
(mic-averaged log-mag spectrogram, three zoom panels: 0-600 Hz with k=1..8,
a mid band around k~10, a high band around k~20-25, refined per-rotor tracks
as k*r_i(t) curves), per-recording NPZ (both phases' tracks, confidence,
residual ratios), and ``summary.csv`` (chosen R, base speeds, captured /
refused, residual before/after, p2p excursion vs the predecessor's).

Run: ``nice -n 10 python scripts/vk_spcup.py`` (CPU-only).
"""

from __future__ import annotations

import os

# Cap BLAS threads BEFORE numpy import: work is parallelised at the process
# level (one worker per recording); another eval shares this box.
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import csv  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.patheffects as patheffects  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402

from data_processing.rps_refinement import RefineConfig, compute_logmag  # noqa: E402
from data_processing.vk_tracking import VKConfig, VKResult, vk_track  # noqa: E402

# ── Constants ─────────────────────────────────────────────────────────────────
SR = 16000
FRAME_HOP_S = 0.032  # trajectory grid (predecessor's STFT hop)
EDGE_TRIM_S = 0.5  # p2p excursion excludes filter-transient edges
PRED_DIR = Path("results/rps_refinement/spcup")  # predecessor artifacts
SEG_DIR = PRED_DIR / "segments"
OUT = Path("results/vk_tracking/spcup")

# Two-phase recipe validated on DREGON (scripts/vk_validation.py MAIN_CFG +
# the capture-basin experiment): capture (annealed, biased) then refine
# (fixed narrow bands, k_min=6, de-biases).
CAPTURE_CFG = VKConfig(
    fs=float(SR),
    k_schedule="grow",
    n_outer=10,
    k_max=30,
    couple_hz=20.0,
)
REFINE_CFG = VKConfig(
    fs=float(SR),
    k_schedule="fixed",
    n_outer=5,
    k_min=6,
    k_max=30,
    bw_hz=1.5,
    max_step=0.3,
    couple_hz=20.0,
)

# Refusal gate on the refine-phase mean VK confidence. Calibration: every
# *validated success* on DREGON (all 5 room1 recordings, same REFINE_CFG)
# sits at mean conf 0.026-0.033. First run used 0.008 ("an order of magnitude
# below the success band") — too lenient: Idea_ssu (1-ch, conf 0.016,
# residual ratio 1.000 i.e. VK explains nothing) produced a visually
# hallucinated wandering track over a ridge-free spectrogram, while the
# visually-locked marginal cases (AGH ego-noise 0.022, Shout_COOEE 0.021)
# and every DREGON success sit at/above 0.021. Revised once to 0.02, which
# also matches the predecessor's comb_confidence refusal of the same
# recording (0.018 < 0.02).
REFUSE_CONF = 0.02
PRED_LOW_CONF = 0.02  # predecessor's comb_confidence refusal threshold

ROTOR_COLORS = ("#e41a1c", "#377eb8", "#4daf4a", "#984ea3")
PANEL_A_HZ = 600.0  # panel (a): 0..600 Hz, k=1..8 drawn
PANEL_A_KMAX = 8
BAND_MID_K = (8.5, 11.5)  # panel (b): k ~ 10
BAND_HIGH_K = (19.0, 26.0)  # panel (c): k ~ 20-25

# Same recording order as the predecessor (its TARGETS tuple).
TARGETS: tuple[str, ...] = (
    "Diagonal_Unloading__recordings__flight__square_10m",
    "AGH__ego-noise__mic_array__1",
    "KU_Leuven__SPCUP19_KU_Leuven_Team_1_recording",
    "Idea_ssu__free_flight_1",
    "Maverick__5",
    "Shout_COOEE__SPCUP19_Shout_COOEE_StaticSubmission1",
    "AGH__calibration__1",
)


# ── Predecessor artifact loading ─────────────────────────────────────────────
def load_segment(rid: str) -> dict[str, Any]:
    """Cached ``(C, T)`` 16 kHz segment + meta from the predecessor's cache."""
    with np.load(SEG_DIR / f"{rid}.npz", allow_pickle=False) as z:
        return {"audio": np.asarray(z["audio"], dtype=np.float64), **json.loads(str(z["meta"]))}


def load_blind_init(rid: str) -> dict[str, Any]:
    """Blind init from the predecessor's scan + model selection artifacts.

    ``base_speeds`` (median of the predecessor's refined tracks for its
    elbow-chosen R) are the constant per-rotor init speeds; the predecessor's
    own refined trajectories are kept for the p2p-excursion comparison.
    """
    with np.load(PRED_DIR / f"{rid}.npz", allow_pickle=False) as z:
        return {
            "chosen_r": int(z["chosen_r"]),
            "base_speeds": np.asarray(z["base_speeds"], dtype=np.float64),
            "pred_confidence": float(np.mean(z["confidence"])),
            "pred_r_refined": np.asarray(z["r_refined"], dtype=np.float64),
            "pred_frame_times": np.asarray(z["frame_times"], dtype=np.float64),
        }


# ── Per-recording driver ──────────────────────────────────────────────────────
def p2p_excursion(r: np.ndarray, t: np.ndarray) -> float:
    """Max over rotors of the peak-to-peak excursion (rev/s), edges trimmed."""
    m = (t >= t[0] + EDGE_TRIM_S) & (t <= t[-1] - EDGE_TRIM_S)
    if not m.any():
        m = np.ones_like(t, dtype=bool)
    return float(np.max(r[:, m].max(axis=1) - r[:, m].min(axis=1)))


def process(rid: str) -> dict[str, Any]:
    """Blind init -> capture-phase VK -> refine-phase VK; saves the NPZ."""
    seg = load_segment(rid)
    init = load_blind_init(rid)
    audio = seg["audio"]
    n_frames = int(audio.shape[-1] / SR / FRAME_HOP_S) + 1
    ft = np.arange(n_frames) * FRAME_HOP_S
    r0 = np.repeat(init["base_speeds"][:, None], n_frames, axis=1)

    t0 = time.time()
    cap: VKResult = vk_track(audio, r0, ft, CAPTURE_CFG)
    ref: VKResult = vk_track(audio, cap.r_refined, ft, REFINE_CFG)
    wall = time.time() - t0

    conf_mean = float(np.mean(ref.confidence))
    refused = conf_mean < REFUSE_CONF
    p2p_vk = p2p_excursion(ref.r_refined, ft)
    p2p_pred = p2p_excursion(init["pred_r_refined"], init["pred_frame_times"])

    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(
        OUT / f"{rid}.npz",
        frame_times=ft,
        r_init=init["base_speeds"],
        chosen_r=init["chosen_r"],
        r_capture=cap.r_refined,
        r_refined=ref.r_refined,
        confidence_capture=cap.confidence,
        confidence=ref.confidence,
        conf_times=ref.conf_times,
        residual_ratios_capture=np.asarray(cap.residual_ratios),
        residual_ratios_refine=np.asarray(ref.residual_ratios),
        max_deltas_capture=np.asarray(cap.max_deltas),
        max_deltas_refine=np.asarray(ref.max_deltas),
        refused=refused,
        pred_confidence=init["pred_confidence"],
    )

    return {
        "recording_id": rid,
        "drone": seg.get("drone"),
        "n_channels": int(audio.shape[0]),
        "seg_len_s": seg.get("seg_len_s"),
        "chosen_r": init["chosen_r"],
        "init_speeds": ", ".join(f"{v:.1f}" for v in init["base_speeds"]),
        "base_speeds_vk": ", ".join(f"{v:.1f}" for v in np.median(ref.r_refined, axis=1)),
        "pred_confidence": round(init["pred_confidence"], 4),
        "pred_refused": init["pred_confidence"] < PRED_LOW_CONF,
        "vk_confidence": round(conf_mean, 4),
        "refused": refused,
        "residual_capture_first": round(float(cap.residual_ratios[0]), 4),
        "residual_capture_last": round(float(cap.residual_ratios[-1]), 4),
        "residual_refine_last": round(float(ref.residual_ratios[-1]), 4),
        "p2p_vk": round(p2p_vk, 2),
        "p2p_pred": round(p2p_pred, 2),
        "wall_s": round(wall, 1),
    }


# ── Overlay figure ────────────────────────────────────────────────────────────
def _draw_tracks(
    ax: Axes, ft: np.ndarray, r: np.ndarray, k_lo: int, k_hi: int, f_lo: float, f_hi: float
) -> None:
    """k*r_i(t) curves for k in [k_lo, k_hi], nan-masked outside [f_lo, f_hi]."""
    halo = [patheffects.withStroke(linewidth=1.8, foreground="white", alpha=0.75)]
    for i in range(r.shape[0]):
        color = ROTOR_COLORS[i % len(ROTOR_COLORS)]
        for k in range(k_lo, k_hi + 1):
            track = k * r[i]
            visible = (track >= f_lo) & (track <= f_hi)
            if not visible.any():
                continue
            ax.plot(
                ft,
                np.where(visible, track, np.nan),
                color=color,
                lw=0.7,
                alpha=0.95,
                path_effects=halo,
            )


def make_overlay(rid: str, row: dict[str, Any]) -> None:
    """Three-panel zoomed spectrogram overlay for one recording."""
    seg = load_segment(rid)
    audio = seg["audio"].astype(np.float32)
    with np.load(OUT / f"{rid}.npz", allow_pickle=False) as z:
        ft = z["frame_times"]
        r = z["r_refined"]
        refused = bool(z["refused"])

    spec = compute_logmag(audio, RefineConfig(sample_rate=SR, device="cpu"))
    lm = spec.logmag.mean(dim=0).cpu().numpy()  # mic-averaged (F, N)
    freqs = np.arange(lm.shape[0]) * spec.bin_hz
    nyq = SR / 2.0
    r_med = float(np.median(r))

    bands = [
        ("(a) low band: k = 1..8", 0.0, PANEL_A_HZ, 1, PANEL_A_KMAX),
        (
            "(b) mid band: k ~ 10",
            BAND_MID_K[0] * r_med,
            min(BAND_MID_K[1] * r_med, nyq),
            int(np.floor(BAND_MID_K[0])),
            int(np.ceil(BAND_MID_K[1])),
        ),
        (
            "(c) high band: k ~ 20-25",
            BAND_HIGH_K[0] * r_med,
            min(BAND_HIGH_K[1] * r_med, nyq),
            int(np.floor(BAND_HIGH_K[0])),
            int(np.ceil(BAND_HIGH_K[1])),
        ),
    ]

    fig, axes = plt.subplots(
        3, 1, figsize=(13, 10), dpi=150, sharex=True, gridspec_kw={"height_ratios": (2.0, 1, 1)}
    )
    for ax, (title, f_lo, f_hi, k_lo, k_hi) in zip(axes, bands):
        fmask = (freqs >= f_lo) & (freqs <= f_hi)
        ax.pcolormesh(
            spec.frame_times,
            freqs[fmask],
            lm[fmask],
            shading="auto",
            cmap="magma",
            rasterized=True,
        )
        if refused:
            ax.text(
                0.5,
                0.5,
                f"REFUSED (VK conf {row['vk_confidence']:.3f} < {REFUSE_CONF})",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=13,
                color="white",
                bbox={"facecolor": "#b2182b", "alpha": 0.85, "pad": 6},
            )
        else:
            _draw_tracks(ax, ft, r, k_lo, k_hi, max(f_lo, 1.0), f_hi)
        ax.set_ylim(f_lo, f_hi)
        ax.set_ylabel("frequency (Hz)")
        ax.set_title(title, fontsize=10, loc="left")
    axes[-1].set_xlabel("time (s)")

    status = "REFUSED" if refused else "annotated"
    fig.suptitle(
        f"{rid}   ({row['drone']}, {row['n_channels']}ch)  |  VK two-phase, R={row['chosen_r']}, "
        f"init [{row['init_speeds']}] rev/s  |  conf={row['vk_confidence']:.3f} ({status})  |  "
        f"p2p {row['p2p_vk']:.1f} rev/s (predecessor {row['p2p_pred']:.1f})",
        fontsize=11,
    )
    fig.savefig(OUT / f"{rid}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=3) as pool:
        futs = {rid: pool.submit(process, rid) for rid in TARGETS}
        for rid in TARGETS:
            row = futs[rid].result()
            rows.append(row)
            print(
                f"[{row['wall_s']:>6.1f}s] {rid}: R={row['chosen_r']} "
                f"conf={row['vk_confidence']:.3f} refused={row['refused']} "
                f"resid {row['residual_capture_first']:.3f}->{row['residual_refine_last']:.3f} "
                f"p2p {row['p2p_vk']:.2f} (pred {row['p2p_pred']:.2f})",
                flush=True,
            )

    for row in rows:
        make_overlay(row["recording_id"], row)

    csv_cols = [
        "recording_id",
        "drone",
        "n_channels",
        "seg_len_s",
        "chosen_r",
        "init_speeds",
        "base_speeds_vk",
        "pred_confidence",
        "pred_refused",
        "vk_confidence",
        "refused",
        "residual_capture_first",
        "residual_capture_last",
        "residual_refine_last",
        "p2p_vk",
        "p2p_pred",
        "wall_s",
    ]
    with (OUT / "summary.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=csv_cols)
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row[c] for c in csv_cols})
    print(f"\nArtifacts written to {OUT}/ (summary.csv, per-recording .npz + .png)")


if __name__ == "__main__":
    main()
