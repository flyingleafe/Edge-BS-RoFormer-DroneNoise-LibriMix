#!/usr/bin/env python3
"""Generate figures/tables for the mic-array geometry calibration report.

CRITICAL environment notes (see workflow/... contract):
- The dataset (data/DREGON, data/recording_with_motor_speed) lives only in the
  MAIN checkout. This script is run with the main checkout's .venv Python.
- The corrected calibration code lives in THIS worktree. We put the worktree's
  `src` and `notebooks` FIRST on sys.path so `data_processing.michaels` and
  friends resolve to the fixed (horizontal-ring) geometry, not any stale copy
  a different sys.path order might pick up.
"""

from __future__ import annotations

import json
import pathlib
import sys
from typing import cast

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.axes3d import Axes3D
from PIL import Image

REPORT_DIR = pathlib.Path(__file__).resolve().parent
WORKTREE_ROOT = REPORT_DIR.parents[2]  # writing/reports/<name> -> worktree root
MAIN_ROOT = pathlib.Path("/home/flyingleafe/Research/PhD/projects/harmonic-noise-suppression")
SCRATCH = pathlib.Path(
    "/tmp/claude-1000/-home-flyingleafe-Research-PhD-projects-harmonic-noise-suppression"
    "/32381798-358b-4883-9b26-b7b518a64d96/scratchpad"
)

sys.path.insert(0, str(WORKTREE_ROOT / "src"))
sys.path.insert(0, str(WORKTREE_ROOT / "notebooks"))

import geom_calibration as gc  # noqa: E402
import stage0_rtf_utils as s0  # noqa: E402

from data_processing.dregon import _parse_mic_positions_txt  # noqa: E402
from data_processing.dregon import get_geometry as dregon_geometry  # noqa: E402
from data_processing.michaels import get_geometry as michaels_geometry  # noqa: E402

ASSETS = REPORT_DIR / "assets"
ASSETS.mkdir(exist_ok=True)

# NOTE on stability: data_processing.dregon.get_geometry() was changed mid-session
# (uncommitted, in this shared worktree) to internally apply the Section 3 frame
# correction, and geom_calibration.calibrate_dregon_positions' frame_correction_deg
# default moved from 183.0 to 0.0 in lockstep. To keep this script's numbers exactly
# reproducible regardless of further changes to that shared, uncommitted code, every
# figure below sources the RAW (as-shipped) mic positions itself, directly from
# micPos.txt via _parse_mic_positions_txt, and applies its own explicit rotate_z(...)
# correction rather than relying on get_geometry's internal (and, this session,
# moving-target) correction. rotor positions are untouched by the frame bug and are
# still read via get_geometry.
FRAME_CORRECTION_DEG = 183.0  # empirical best fit, see fig_frame_alignment


def _dregon_raw_mic_positions(dregon_dir: pathlib.Path) -> np.ndarray:
    """As-shipped mic positions, bypassing any correction baked into get_geometry."""
    mic_pos_path = pathlib.Path(dregon_dir) / "micPos.txt"
    if mic_pos_path.exists():
        return _parse_mic_positions_txt(mic_pos_path)
    import scipy.io

    return np.asarray(scipy.io.loadmat(str(pathlib.Path(dregon_dir) / "coordinates.mat"))["micPos"])


# Sanity check called out in the task contract: mis-loading the main-checkout's
# (uncorrected) copy of data_processing would show a *varying* mic z here.
_mic_check, _ = michaels_geometry()
assert np.allclose(_mic_check[:, 2], 0.33), (
    f"Loaded the WRONG data_processing.michaels (mic z not constant: {_mic_check[:, 2]}) "
    "-- check sys.path ordering (worktree src must come first)."
)

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "font.size": 10,
        "axes.titlesize": 10.5,
        "axes.labelsize": 10,
        "legend.fontsize": 8.5,
    }
)

COL_MEAS = "#1f6fb2"
COL_FF = "#e0761b"
COL_COH = "#5faa5f"
COL_BEFORE = "#b0413e"
COL_AFTER = "#1f6fb2"
COL_NOMINAL = "#888888"


def savefig(fig, name: str) -> None:
    path = ASSETS / name
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


# ---------------------------------------------------------------------------
# Figure 1 -- why position errors matter: phase-error / cancellation argument
# ---------------------------------------------------------------------------
def fig_propagation_phase() -> None:
    sr = 44100.0
    delay_err_samples = 15.0
    dt = delay_err_samples / sr  # seconds

    freqs = np.linspace(20.0, 2000.0, 2000)
    phase_deg = 360.0 * freqs * dt  # unwrapped phase error, degrees

    # Two-path interference magnitude if a model with this position error is
    # used to *cancel* the true signal (e.g. beamforming / notch): the
    # residual after subtracting a unit-amplitude replica shifted by the
    # phase error, relative to the true signal's amplitude.
    residual_after_cancel = np.abs(1.0 - np.exp(1j * np.radians(phase_deg)))

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.3))

    ax = axes[0]
    ax.plot(freqs, phase_deg, color=COL_MEAS, lw=1.8)
    for f_mark in (100.0, 1000.0):
        p = 360.0 * f_mark * dt
        ax.plot([f_mark], [p], "o", color=COL_BEFORE, ms=5, zorder=5)
        ax.annotate(
            f"{f_mark:.0f} Hz $\\to$ {p:.0f}$\\degree$",
            (f_mark, p),
            textcoords="offset points",
            xytext=(10, 6) if f_mark == 100 else (-88, 6),
            fontsize=8.5,
        )
    ax.axhline(180.0, color="k", lw=0.7, ls=":")
    ax.text(1550, 190, "$180\\degree$ = full cancellation", fontsize=8, va="bottom")
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("phase error (deg)")
    ax.set_title(
        f"(a) phase error from a {delay_err_samples:.0f}-sample\ndelay error, vs frequency"
    )
    ax.set_xlim(0, 2000)
    ax.set_ylim(0, 300)

    ax = axes[1]
    ax.plot(freqs, residual_after_cancel, color=COL_FF, lw=1.8)
    ax.axvspan(0, 300, color=COL_COH, alpha=0.12)
    ax.text(150, 1.45, "low $k$:\nstays\naligned", ha="center", fontsize=8, color="#2f6b2f")
    ax.axvspan(1150, 2000, color=COL_BEFORE, alpha=0.10)
    ax.text(
        1575,
        1.45,
        "high $k$:\nnear-total\ncancellation",
        ha="center",
        fontsize=8,
        color="#8a2f2c",
    )
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel(r"$|1 - e^{i\,\Delta\phi(f)}|$  (relative)")
    ax.set_title("(b) how badly a mistuned delay\ncancels the true field")
    ax.set_xlim(0, 2000)
    ax.set_ylim(0, 2.1)

    fig.suptitle(
        "A geometry (hence delay) error is a frequency-proportional corruption",
        y=1.04,
        fontsize=10.5,
    )
    fig.tight_layout()
    savefig(fig, "fig1_propagation_phase.png")


# ---------------------------------------------------------------------------
# Figure 2 -- RTF magnitude + coherence vs free-field, one DREGON rotor
# ---------------------------------------------------------------------------
def fig_rtf_coherence(dregon_dir: pathlib.Path) -> None:
    _, rotor_pos = dregon_geometry(dregon_dir)  # rotor positions are unaffected by the frame bug
    mic_corr = s0.rotate_z(_dregon_raw_mic_positions(dregon_dir), FRAME_CORRECTION_DEG)
    dist = s0.distance_matrix(mic_corr, rotor_pos)
    rotor_idx = 0  # Motor1
    ref = int(np.argmin(dist[rotor_idx]))

    x, sr = s0.load_motor(dregon_dir, motor_id=rotor_idx + 1, speed=70, max_seconds=15.0)
    freqs, rtf, coh = s0.estimate_rtf(x, sr, ref, nperseg=8192)
    rtf_ff = s0.freefield_rtf(freqs, dist[rotor_idx], ref)

    fmax = 3000.0
    sel = freqs <= fmax
    freqs = freqs[sel]
    rtf = rtf[:, sel]
    coh = coh[:, sel]
    rtf_ff = rtf_ff[:, sel]

    mics = [m for m in range(mic_corr.shape[0]) if m != ref]
    ncols, nrows = 4, 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(11.5, 5.4), sharex=True)
    axes = axes.ravel()
    for i, m in enumerate(mics):
        ax = axes[i]
        meas_db = 20 * np.log10(np.abs(rtf[m]) + 1e-9)
        ff_db = 20 * np.log10(np.abs(rtf_ff[m]) + 1e-9)
        ax.axvspan(400, 800, color=COL_COH, alpha=0.10, zorder=0)
        ax.plot(freqs, meas_db, color=COL_MEAS, lw=0.8, label="measured")
        ax.plot(freqs, ff_db, color=COL_FF, lw=1.6, label="free-field $1/r$")
        ax.set_title(f"mic{m}  (r={dist[rotor_idx, m]:.2f} m)", fontsize=9)
        ax.set_ylim(-45, 10)
        axc = ax.twinx()
        axc.plot(freqs, coh[m], color=COL_COH, lw=0.6, alpha=0.8)
        axc.set_ylim(0, 1)
        if i % ncols != ncols - 1:
            axc.set_yticklabels([])
        if i % ncols != 0:
            ax.set_yticklabels([])
    for j in range(len(mics), nrows * ncols):
        axes[j].axis("off")
    fig.text(0.5, 0.0, "frequency (Hz)", ha="center")
    fig.text(0.005, 0.5, "|RTF| (dB)", va="center", rotation="vertical")
    fig.text(0.995, 0.5, "coherence $\\gamma^2$", va="center", rotation=270)
    handles = [
        Line2D([0], [0], color=COL_MEAS, lw=1.2, label="measured $|RTF|$"),
        Line2D([0], [0], color=COL_FF, lw=1.6, label="free-field $1/r$"),
        Line2D(
            [0],
            [0],
            color=COL_COH,
            lw=1.2,
            label="coherence $\\gamma^2$ (shaded band = 400–800 Hz)",
        ),
    ]
    fig.legend(
        handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.06), frameon=False
    )
    fig.suptitle(
        f"Rotor 1, DREGON — measured vs. free-field RTF magnitude, per mic "
        f"(ref = mic{ref}, frame-corrected nominal geometry)",
        y=1.15,
        fontsize=10,
    )
    fig.tight_layout(rect=(0.015, 0.0, 0.985, 1.0))
    savefig(fig, "fig2_rtf_coherence.png")


# ---------------------------------------------------------------------------
# Figure 3 -- DREGON frame mismatch: alignment sweep + TDOA scatter
# ---------------------------------------------------------------------------
def fig_frame_alignment(dregon_dir: pathlib.Path) -> dict:
    _, rotor_pos = dregon_geometry(dregon_dir)  # rotor positions are unaffected by the frame bug
    mic_raw = _dregon_raw_mic_positions(dregon_dir)  # as-shipped, uncorrected
    x_by_rotor = {}
    sr = 0
    for r in range(4):
        x, sr = s0.load_motor(dregon_dir, motor_id=r + 1, speed=70, max_seconds=15.0)
        x_by_rotor[r] = x

    align = s0.align_mic_frame(x_by_rotor, mic_raw, rotor_pos, sr)

    measured = np.vstack([s0.measured_tdoa_row(x_by_rotor[r], 0) for r in range(4)])
    dist_shipped = s0.distance_matrix(mic_raw, rotor_pos)
    # Use the *empirically* best-fit rotation found by the sweep (matches panel a
    # exactly), not the internal library's nominal 180° flip (mic_corr) -- the two
    # differ by the few-degree sweep residual absorbed elsewhere (see Section 4).
    dist_corr = s0.distance_matrix(s0.rotate_z(mic_raw, align.best_degrees), rotor_pos)
    pred_shipped = np.vstack([s0.freefield_tdoa_row(dist_shipped[r], 0, sr) for r in range(4)])
    pred_corr = np.vstack([s0.freefield_tdoa_row(dist_corr[r], 0, sr) for r in range(4)])

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8))

    ax = axes[0]
    ax.plot(align.angles, align.corr_curve, color=COL_MEAS, lw=1.6)
    ax.axvline(align.best_degrees, color=COL_AFTER, ls="--", lw=1.0)
    ax.axhline(align.identity_corr, color=COL_BEFORE, ls=":", lw=1.0)
    ax.annotate(
        f"shipped (0$\\degree$): {align.identity_corr:.2f}",
        (0, align.identity_corr),
        xytext=(35, -8),
        textcoords="offset points",
        color=COL_BEFORE,
        fontsize=8.5,
    )
    ax.annotate(
        f"best: {align.best_degrees:.0f}$\\degree$ → {align.best_corr:.2f}",
        (align.best_degrees, align.best_corr),
        xytext=(-90, -18),
        textcoords="offset points",
        color=COL_AFTER,
        fontsize=8.5,
    )
    ax.set_xlabel("mic-frame $z$-rotation (deg)")
    ax.set_ylabel("corr(predicted, measured TDOA)")
    ax.set_title("(a) frame-alignment sweep")
    ax.set_xlim(0, 360)

    ax = axes[1]
    lims = (
        min(pred_shipped.min(), pred_corr.min(), measured.min()) - 1,
        max(pred_shipped.max(), pred_corr.max(), measured.max()) + 1,
    )
    ax.plot(lims, lims, color="k", lw=0.7, alpha=0.5)
    ax.scatter(
        pred_shipped.ravel(), measured.ravel(), s=16, color=COL_BEFORE, label="shipped", alpha=0.8
    )
    ax.scatter(
        pred_corr.ravel(),
        measured.ravel(),
        s=16,
        color=COL_AFTER,
        label=f"corrected {align.best_degrees:.0f}$\\degree$",
        alpha=0.8,
    )
    ax.set_xlabel("free-field predicted TDOA (samples)")
    ax.set_ylabel("measured GCC-PHAT TDOA (samples)")
    ax.set_title("(b) measured vs. predicted TDOA")
    ax.legend(frameon=False, loc="upper left")
    ax.set_xlim(*lims)
    ax.set_ylim(*lims)

    fig.suptitle("DREGON: shipped micPos/rotorsPos are frame-mismatched by ≈183°", y=1.03)
    fig.tight_layout()
    savefig(fig, "fig3_frame_alignment.png")
    return {
        "identity_corr": align.identity_corr,
        "best_degrees": align.best_degrees,
        "best_corr": align.best_corr,
    }


# ---------------------------------------------------------------------------
# Figure 4 -- bundle adjustment (DREGON) + synthetic control
# ---------------------------------------------------------------------------
def _synthetic_records(mic: np.ndarray, rotor: np.ndarray, freqs: np.ndarray) -> list:
    recs = []
    for r in range(rotor.shape[0]):
        d = np.linalg.norm(mic - rotor[r][None, :], axis=1)
        ref = int(np.argmin(d))
        ph = -2.0 * np.pi * freqs[None, :] * (d[:, None] - d[ref]) / gc.SPEED_OF_SOUND
        mag = (d[ref] / d)[:, None] * np.ones_like(ph)
        recs.append(
            gc.RotorBandRTF(
                rotor=r, ref=ref, freqs=freqs, meas_phase=ph, meas_mag=mag, coh=np.ones_like(ph)
            )
        )
    return recs


def _bundle_adjust_with_trace(records, mic_init, rotor_init, lam, iters, lr, log_every=25):
    mic_nom = torch.as_tensor(mic_init, dtype=torch.float64)
    rot_nom = torch.as_tensor(rotor_init, dtype=torch.float64)
    mic = mic_nom.clone().requires_grad_(True)
    opt = torch.optim.Adam([mic], lr=lr)
    trace_iter, trace_resid = [], []
    for it in range(iters):
        opt.zero_grad()
        tot, wtot = gc._phase_terms(records, mic, rot_nom, gc.SPEED_OF_SOUND, 0.0)
        loss = tot / torch.clamp(wtot, min=1e-12) + lam * ((mic - mic_nom) ** 2).sum()
        loss.backward()
        opt.step()
        if it % log_every == 0 or it == iters - 1:
            rms = (
                float(torch.sqrt(tot.detach() / torch.clamp(wtot.detach(), min=1e-12)))
                * 180.0
                / np.pi
            )
            trace_iter.append(it)
            trace_resid.append(rms)
    return mic.detach().numpy(), trace_iter, trace_resid


def _calibrate_dregon_positions_from_raw(
    dregon_dir: pathlib.Path, lam: float = 50.0, iters: int = 1500
) -> gc.CalibrationResult:
    """Self-contained re-implementation of gc.calibrate_dregon_positions that
    starts from our own explicit rotate_z(raw, FRAME_CORRECTION_DEG) nominal
    (see the module-level NOTE on stability) instead of depending on
    get_geometry's own (this-session, moving-target) internal correction."""
    _, rotor_pos = dregon_geometry(dregon_dir)
    mic_init = s0.rotate_z(_dregon_raw_mic_positions(dregon_dir), FRAME_CORRECTION_DEG)
    records, _, x_by_rotor, sr = gc.build_dregon_records(
        dregon_dir, mic_init, rotor_pos, speeds=(60, 70, 80), band=(400.0, 800.0), max_seconds=15.0
    )
    mic_opt, rotor_opt = gc.run_bundle_adjustment(
        records, mic_init, rotor_pos, lam=lam, iters=iters, lr=1e-3, refine_rotors=False
    )
    rotor_order = sorted(x_by_rotor)
    meas_tdoa = np.vstack([s0.measured_tdoa_row(x_by_rotor[r], 0) for r in rotor_order])
    return gc._assemble_result(
        records,
        mic_init,
        rotor_pos,
        mic_opt,
        rotor_opt,
        meas_tdoa,
        rotor_order,
        ref=0,
        sr=sr,
        coh_thr_hi=0.8,
    )


def fig_bundle_adjustment_and_synthetic(dregon_dir: pathlib.Path) -> gc.CalibrationResult:
    print("Running DREGON bundle adjustment (real audio, ~1500 Adam iters)...")
    result = _calibrate_dregon_positions_from_raw(dregon_dir, lam=50.0, iters=1500)

    print("Running synthetic-control recovery test...")
    rng = np.random.default_rng(0)
    mic_true = rng.standard_normal((8, 3)) * 0.05
    rotor = np.array([[0.2, 0.2, 0], [0.2, -0.2, 0], [-0.2, 0.2, 0], [-0.2, -0.2, 0]], float)
    freqs = np.linspace(400.0, 800.0, 60)
    recs = _synthetic_records(mic_true, rotor, freqs)
    mic_init = mic_true + rng.standard_normal((8, 3)) * 0.01
    mic_opt, trace_iter, trace_resid = _bundle_adjust_with_trace(
        recs, mic_init, rotor, lam=1e-4, iters=3000, lr=2e-3, log_every=20
    )
    _, proc_rmse_cm = gc.procrustes_align(mic_opt, mic_true)
    resid0 = trace_resid[0]
    resid1 = trace_resid[-1]

    fig = plt.figure(figsize=(11.0, 3.6))
    gs = fig.add_gridspec(1, 3, width_ratios=(1.0, 1.1, 1.2))

    ax = fig.add_subplot(gs[0])
    x = np.arange(2)
    width = 0.32
    ax.bar(
        x - width / 2,
        [result.resid_before_deg, result.resid_before_deg_hi],
        width,
        color=COL_BEFORE,
        label="before",
    )
    ax.bar(
        x + width / 2,
        [result.resid_after_deg, result.resid_after_deg_hi],
        width,
        color=COL_AFTER,
        label="after",
    )
    ax.set_xticks(x, ["all bins", "hi-coh"])
    ax.set_ylabel("phase resid. RMS (deg)")
    ax.set_title("(a) DREGON residual")
    ax.legend(frameon=False, fontsize=8)
    for xi, vals in zip(
        x,
        [
            (result.resid_before_deg, result.resid_after_deg),
            (result.resid_before_deg_hi, result.resid_after_deg_hi),
        ],
    ):
        for xoff, v in zip((-width / 2, width / 2), vals):
            ax.text(xi + xoff, v + 0.6, f"{v:.1f}", ha="center", fontsize=7.5)

    ax = fig.add_subplot(gs[1])
    order = np.argsort(-result.mic_delta_cm)
    ax.bar(np.arange(8), result.mic_delta_cm[order], color=COL_AFTER)
    ax.set_xticks(np.arange(8), [f"mic{m}" for m in order], rotation=45, fontsize=7.5)
    ax.set_ylabel("$|\\Delta|$ from nominal (cm)")
    ax.set_title("(b) DREGON per-mic move")
    ax.axhline(2.2, color="k", ls=":", lw=0.8)

    ax = fig.add_subplot(gs[2])
    ax.semilogy(trace_iter, trace_resid, color=COL_AFTER, lw=1.5)
    ax.axhline(resid0, color=COL_BEFORE, ls=":", lw=0.9)
    ax.set_xlabel("Adam iteration")
    ax.set_ylabel("phase resid. RMS (deg)")
    ax.set_title("(c) synthetic-control recovery")
    ax.text(
        0.97,
        0.92,
        f"{resid0:.2f}° → {resid1:.4f}°\nposition RMSE (gauge-fixed): {proc_rmse_cm:.2f} cm",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        bbox=dict(boxstyle="round", fc="white", ec="0.7"),
    )

    fig.suptitle(
        "Bundle adjustment: real DREGON refinement (a,b) and a synthetic ground-truth check (c)",
        y=1.05,
    )
    fig.tight_layout()
    savefig(fig, "fig4_bundle_adjustment.png")

    # Small Typst table of per-mic results for the report.
    lines = [
        "#figure(",
        "  table(",
        "    columns: 9,",
        "    align: center,",
        "    [*mic*], " + ", ".join(f"[{m}]" for m in range(8)) + ",",
        "    [$Delta$ (cm)], " + ", ".join(f"[{v:.2f}]" for v in result.mic_delta_cm) + ",",
        "  ),",
        "  caption: [Per-mic position change from the frame-corrected nominal after "
        "bundle adjustment (DREGON, coherence-weighted phase objective, "
        f"$lambda={50.0:g}$).],",
        ") <tab-dregon-deltas>",
        "",
    ]
    (ASSETS / "dregon_deltas_table.typ").write_text("\n".join(lines))
    print(f"wrote {ASSETS / 'dregon_deltas_table.typ'}")

    return result, dict(
        resid0=resid0,
        resid1=resid1,
        proc_rmse_cm=proc_rmse_cm,
        resid_before=result.resid_before_deg,
        resid_after=result.resid_after_deg,
        resid_before_hi=result.resid_before_deg_hi,
        resid_after_hi=result.resid_after_deg_hi,
        mag_err_before=result.mag_err_before_db,
        mag_err_after=result.mag_err_after_db,
        tdoa_before=result.tdoa_corr_before,
        tdoa_after=result.tdoa_corr_after,
    )


# ---------------------------------------------------------------------------
# Figure 5 -- Michael's: vertical-ring bug vs horizontal-ring photo correction
# ---------------------------------------------------------------------------
def fig_michaels_photo_correction(positions: dict) -> None:
    m = positions["michaels"]
    mic_orig = np.array(m["mic_original"])
    mic_ref = np.array(m["mic_refined"])
    rotor = np.array(m["rotor_original"])

    photo_path = MAIN_ROOT / "data/recording_with_motor_speed/Microphone_Array_Configuration.jpeg"
    img = Image.open(photo_path)
    # Crop to the array + measurement-label region (drop excess background grass
    # at the very bottom) to keep the panel legible at report size.
    w, h = img.size
    img_crop = img.crop((0, 0, w, int(h * 0.80)))

    fig = plt.figure(figsize=(11.5, 4.2))
    gs = fig.add_gridspec(1, 3, width_ratios=(1.0, 1.0, 0.85))

    ax = fig.add_subplot(gs[0])
    ax.scatter(
        mic_orig[:, 0],
        mic_orig[:, 1],
        color=COL_BEFORE,
        label="original (vertical-ring bug)",
        zorder=3,
    )
    ax.scatter(
        mic_ref[:, 0],
        mic_ref[:, 1],
        color=COL_AFTER,
        label="refined (horizontal, from photo)",
        zorder=3,
    )
    ax.scatter(
        rotor[:, 0], rotor[:, 1], color=COL_NOMINAL, marker="x", s=45, label="rotors", zorder=3
    )
    ax.set_xlabel("X, forward (m)")
    ax.set_ylabel("Y, left (m)")
    ax.set_title("(a) top view (X-Y)")
    ax.set_aspect("equal")
    ax.legend(frameon=False, fontsize=7.5, loc="upper left", bbox_to_anchor=(0.0, -0.18))

    ax = fig.add_subplot(gs[1])
    ax.scatter(mic_orig[:, 0], mic_orig[:, 2], color=COL_BEFORE, zorder=3)
    ax.scatter(mic_ref[:, 0], mic_ref[:, 2], color=COL_AFTER, zorder=3)
    ax.scatter(rotor[:, 0], rotor[:, 2], color=COL_NOMINAL, marker="x", s=45, zorder=3)
    ax.set_xlabel("X, forward (m)")
    ax.set_ylabel("Z, up (m)")
    ax.set_title("(b) side view (X-Z)")
    ax.set_aspect("equal")

    ax = fig.add_subplot(gs[2])
    ax.imshow(img_crop)
    ax.axis("off")
    ax.set_title("(c) the rig photo (top-down)")

    fig.suptitle(
        "Michael's array: the shipped code drew a VERTICAL ring "
        "(collapses to a line from above); the rig photo shows a HORIZONTAL one",
        y=1.04,
        fontsize=10,
    )
    fig.tight_layout()
    savefig(fig, "fig5_michaels_photo_correction.png")


# ---------------------------------------------------------------------------
# Figure 6 -- Michael's: rigid-ring degeneracy sweep
# ---------------------------------------------------------------------------
RING_SWEEP = [
    # lam, resid_deg, tilt_deg, radius_cm
    (0.5, 21.9, 81.0, 5.0),
    (2.0, 23.6, 74.0, 5.1),
    (10.0, 29.1, 47.0, 5.2),
    (50.0, 46.6, 8.0, 6.9),
    (200.0, 51.4, 1.5, 7.8),
]
NOMINAL_RESID = 53.4
UNCONSTRAINED_RESID = 16.5
UNCONSTRAINED_MOVE_M = 2.65
NOMINAL_RADIUS_CM = 8.25


def fig_michaels_degeneracy(anchored_ring_path: pathlib.Path) -> None:
    lam = np.array([r[0] for r in RING_SWEEP])
    resid = np.array([r[1] for r in RING_SWEEP])
    tilt = np.array([r[2] for r in RING_SWEEP])
    radius = np.array([r[3] for r in RING_SWEEP])

    fig = plt.figure(figsize=(11.0, 4.3))
    gs = fig.add_gridspec(1, 2, width_ratios=(1.15, 1.0))

    ax = fig.add_subplot(gs[0])
    ax.semilogx(lam, resid, "o-", color=COL_BEFORE, label="residual (deg)")
    ax.axhline(NOMINAL_RESID, color=COL_BEFORE, ls=":", lw=1.0)
    ax.set_ylim(18, 66)
    ax.text(
        0.6,
        NOMINAL_RESID + 3.0,
        f"nominal (rigid, no fit): {NOMINAL_RESID:.1f}°",
        fontsize=7.5,
        color=COL_BEFORE,
    )
    ax.set_xlabel("prior strength $\\lambda$ (anchored fit)")
    ax.set_ylabel("phase residual (deg)", color=COL_BEFORE)
    ax.tick_params(axis="y", colors=COL_BEFORE)

    ax2 = ax.twinx()
    ax2.semilogx(lam, tilt, "s-", color="#7a4fa8", label="tilt from horizontal (deg)")
    ax2.set_ylabel("tilt from horizontal (deg)", color="#7a4fa8")
    ax2.tick_params(axis="y", colors="#7a4fa8")

    ax3 = ax.twinx()
    ax3.spines["right"].set_position(("axes", 1.28))
    ax3.semilogx(lam, radius, "^-", color=COL_AFTER, label="ring radius (cm)")
    ax3.axhline(NOMINAL_RADIUS_CM, color=COL_AFTER, ls=":", lw=0.9)
    ax3.set_ylabel("ring radius (cm)", color=COL_AFTER)
    ax3.tick_params(axis="y", colors=COL_AFTER)
    ax.set_title(
        f"(a) anchored sweep: lowering residual only via tilt + shrink\n"
        f"(unconstrained: flies {UNCONSTRAINED_MOVE_M:.2f} m away, resid {UNCONSTRAINED_RESID:.1f}°)",
        fontsize=9,
    )

    # 3D visualisation: nominal horizontal ring vs the lambda=10 anchored fit
    d = json.loads(anchored_ring_path.read_text())
    mic_nom = np.array(d["mic_nominal"])
    mic_fit = np.array(d["mic_refined"])
    rotor = np.array(d["rotor"])

    ax3d = cast(Axes3D, fig.add_subplot(gs[1], projection="3d"))
    ax3d.plot(
        *np.vstack([mic_nom, mic_nom[:1]]).T,
        "-o",
        color=COL_NOMINAL,
        ms=4,
        label="nominal (horizontal, r=8.25cm)",
    )
    ax3d.plot(
        *np.vstack([mic_fit, mic_fit[:1]]).T,
        "-o",
        color=COL_BEFORE,
        ms=4,
        label="$\\lambda{=}10$ fit (tilt 47°, r=5.2cm)",
    )
    ax3d.scatter(
        rotor[:, 0],
        rotor[:, 1],
        rotor[:, 2],  # pyright: ignore[reportArgumentType]
        color="k",
        marker="x",
        s=40,
        label="rotors",
    )
    ax3d.set_xlabel("X")
    ax3d.set_ylabel("Y")
    ax3d.set_zlabel("Z")
    ax3d.set_title("(b) nominal vs. one anchored fit", fontsize=9)
    ax3d.legend(frameon=False, fontsize=7, loc="upper left", bbox_to_anchor=(-0.05, 1.05))
    ax3d.view_init(elev=18, azim=-60)

    fig.suptitle(
        "Michael's rigid-ring degeneracy: every relaxation trades physical plausibility for residual",
        y=1.03,
    )
    fig.tight_layout()
    savefig(fig, "fig6_michaels_degeneracy.png")

    lines = [
        "#figure(",
        "  table(",
        "    columns: 4,",
        "    align: center,",
        "    [*$lambda$*], [*resid (deg)*], [*tilt (deg)*], [*radius (cm)*],",
    ]
    for lam_v, r, t, rad in RING_SWEEP:
        lines.append(f"    [{lam_v:g}], [{r:.1f}], [{t:.0f}], [{rad:.1f}],")
    lines += [
        "  ),",
        "  caption: [Rigid-ring $lambda$-sweep (7 DOF: translate + 3-D rotate + radius, "
        "anchored to the nominal photo geometry). Nominal (no fit) residual "
        f"{NOMINAL_RESID:.1f}°; nominal radius {NOMINAL_RADIUS_CM:.2f} cm. Lower $lambda$ "
        "buys lower residual only by tilting the ring edge-on and shrinking it.],",
        ") <tab-ring-sweep>",
        "",
    ]
    (ASSETS / "ring_sweep_table.typ").write_text("\n".join(lines))
    print(f"wrote {ASSETS / 'ring_sweep_table.typ'}")


# ---------------------------------------------------------------------------
# Figure 7 -- summary: original vs refined 3D geometry, both drones
# ---------------------------------------------------------------------------
def fig_geometry_summary(positions: dict) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 8.2))

    for col, (name, label) in enumerate([("dregon", "DREGON"), ("michaels", "Michael's")]):
        d = positions[name]
        mic_o = np.array(d["mic_original"])
        mic_r = np.array(d["mic_refined"])
        rotor = np.array(d["rotor_refined"])

        ax = axes[0, col]
        ax.scatter(mic_o[:, 0], mic_o[:, 1], color=COL_BEFORE, label="original", zorder=3)
        ax.scatter(mic_r[:, 0], mic_r[:, 1], color=COL_AFTER, label="refined", zorder=3)
        ax.scatter(
            rotor[:, 0], rotor[:, 1], color=COL_NOMINAL, marker="x", s=40, label="rotors", zorder=3
        )
        ax.set_title(f"{label}: top view (X-Y)")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_aspect("equal")
        if col == 0:
            ax.legend(frameon=False, fontsize=8)

        ax = axes[1, col]
        ax.scatter(mic_o[:, 0], mic_o[:, 2], color=COL_BEFORE, zorder=3)
        ax.scatter(mic_r[:, 0], mic_r[:, 2], color=COL_AFTER, zorder=3)
        ax.scatter(rotor[:, 0], rotor[:, 2], color=COL_NOMINAL, marker="x", s=40, zorder=3)
        ax.set_title(f"{label}: side view (X-Z)")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Z (m)")
        ax.set_aspect("equal")

    fig.suptitle(
        "Original vs. refined geometry — DREGON (audio bundle adjustment) "
        "vs. Michael's (photo plane-correction only)",
        y=1.01,
    )
    fig.tight_layout()
    savefig(fig, "fig7_geometry_summary.png")


# ---------------------------------------------------------------------------
def main() -> None:
    dregon_dir = s0.find_dregon_dir()
    print(f"DREGON dir: {dregon_dir}")

    positions = json.loads((SCRATCH / "positions.json").read_text())

    fig_propagation_phase()
    fig_rtf_coherence(dregon_dir)
    frame_stats = fig_frame_alignment(dregon_dir)
    _, ba_stats = fig_bundle_adjustment_and_synthetic(dregon_dir)
    fig_michaels_photo_correction(positions)
    fig_michaels_degeneracy(SCRATCH / "michaels_ring_anchored.json")
    fig_geometry_summary(positions)

    stats = {**frame_stats, **ba_stats}
    (ASSETS / "stats.json").write_text(json.dumps(stats, indent=2))
    print("\n=== stats ===")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
