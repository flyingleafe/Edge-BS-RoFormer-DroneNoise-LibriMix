"""Acoustic-geometry self-calibration from single-source multichannel fields.

This module turns the Stage-0 relative-transfer-function (RTF) machinery into a
small *bundle adjustment*: given the measured multichannel field of one rotor
tone, optimise the microphone (and optionally rotor) positions in a small
neighbourhood of the nominal geometry so that the free-field ``1/r + delay``
model best explains the measured inter-mic phase in the coherent band.

Two datasets are supported:

* **DREGON** (:func:`calibrate_dregon_positions`) — single-motor constant-speed
  recordings give a clean single-source field per rotor. We start from the
  ``183°``-corrected nominal geometry (see ``stage0_rtf_utils.align_mic_frame``)
  and refine.

* **Michael's** (:func:`calibrate_michaels_positions`) — the four rotors run
  simultaneously, so we separate them by *frequency* using the per-rotor RPS
  telemetry: for each rotor we collect the TF bins that fall on one of its
  harmonics and on **no** other rotor's harmonic, and estimate a per-mic RTF
  from the cross-spectra at those bins. We additionally detect a channel
  indexing permutation / frame flip before refining.

Sign convention
---------------
There is a subtle sign trap. ``scipy.signal.csd(x, y)`` (used by
``stage0_rtf_utils.estimate_rtf``) returns ``E[Y · conj(X)]``, so for a pure
delay where mic ``m`` lags the reference by ``τ`` its phase is
``+2πf·τ`` — the **opposite** sign to ``stage0_rtf_utils.freefield_rtf``
(``-2πf·(r_m - r_ref)/c``) and to ``gcc_phat_tdoa``. Stage-0's TDOA validation
used ``gcc_phat`` and never compared the *phases* directly, so the mismatch was
latent. Here we compare phases, so the DREGON path **negates** the
``estimate_rtf`` phase to bring it into the free-field convention, while the
Michael's path builds the cross-spectrum manually as ``X_m · conj(X_ref)`` which
already carries the free-field sign. Everything downstream uses the free-field
convention ``phase = -2πf·(r_m - r_ref)/c``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import stage0_rtf_utils as s0
import torch
from scipy.signal import stft

SPEED_OF_SOUND = s0.SPEED_OF_SOUND


# ---------------------------------------------------------------------------
# Shared data model
# ---------------------------------------------------------------------------
@dataclass
class RotorBandRTF:
    """Measured single-rotor RTF restricted to a set of frequency bins.

    ``meas_phase`` is stored already in the **free-field convention**
    (``-2πf·(r_m - r_ref)/c`` for a pure delay), i.e. sign-corrected where
    necessary. ``coh`` is the magnitude-squared coherence used as the per-bin
    trust weight.
    """

    rotor: int
    ref: int
    freqs: np.ndarray  # (F,)
    meas_phase: np.ndarray  # (C, F) free-field convention
    meas_mag: np.ndarray  # (C, F) |RTF|
    coh: np.ndarray  # (C, F) in [0, 1]


@dataclass
class CalibrationResult:
    """Outcome of a bundle-adjustment run plus before/after validation."""

    mic_init: np.ndarray
    rotor_init: np.ndarray
    mic_opt: np.ndarray
    rotor_opt: np.ndarray
    mic_delta_cm: np.ndarray  # (C,) per-mic |Δ| from init, cm
    rotor_delta_cm: np.ndarray  # (R,) per-rotor |Δ| from init, cm
    mic_delta_procrustes_cm: np.ndarray  # (C,) residual after removing rigid motion
    resid_before_deg: float
    resid_after_deg: float
    resid_before_deg_hi: float  # high-coherence subset
    resid_after_deg_hi: float
    tdoa_corr_before: float
    tdoa_corr_after: float
    mag_err_before_db: float  # mean |dB| |RTF| error
    mag_err_after_db: float
    records: list[RotorBandRTF] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Core objective (torch, differentiable)
# ---------------------------------------------------------------------------
def _wrap(x: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(x), torch.cos(x))


def _phase_terms(
    records: list[RotorBandRTF],
    mic: torch.Tensor,
    rotor: torch.Tensor,
    c: float,
    coh_thr: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (weighted squared-phase-error sum, weight sum) over all records."""
    tot = mic.new_zeros(())
    wtot = mic.new_zeros(())
    for rc in records:
        r = rc.rotor
        ref = rc.ref
        d = torch.linalg.norm(mic - rotor[r][None, :], dim=1)  # (C,)
        f = torch.as_tensor(rc.freqs, dtype=mic.dtype)
        ph_ff = -2.0 * np.pi * f[None, :] * (d[:, None] - d[ref]) / c  # (C, F)
        mp = torch.as_tensor(rc.meas_phase, dtype=mic.dtype)
        w = torch.as_tensor(rc.coh, dtype=mic.dtype)
        if coh_thr > 0.0:
            w = torch.where(w >= coh_thr, w, torch.zeros_like(w))
        dphi = _wrap(mp - ph_ff)
        tot = tot + (w * dphi**2).sum()
        wtot = wtot + w.sum()
    return tot, wtot


def phase_residual_rms_deg(
    records: list[RotorBandRTF],
    mic_pos: np.ndarray,
    rotor_pos: np.ndarray,
    c: float = SPEED_OF_SOUND,
    coh_thr: float = 0.0,
) -> float:
    """Coherence-weighted RMS inter-mic phase residual (degrees) for a geometry."""
    mic = torch.as_tensor(mic_pos, dtype=torch.float64)
    rotor = torch.as_tensor(rotor_pos, dtype=torch.float64)
    tot, wtot = _phase_terms(records, mic, rotor, c, coh_thr)
    rms = torch.sqrt(tot / torch.clamp(wtot, min=1e-12))
    return float(rms) * 180.0 / np.pi


def run_bundle_adjustment(
    records: list[RotorBandRTF],
    mic_init: np.ndarray,
    rotor_init: np.ndarray,
    lam: float,
    iters: int = 1500,
    lr: float = 1e-3,
    refine_rotors: bool = False,
    c: float = SPEED_OF_SOUND,
    mag_weight: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Adam bundle adjustment of positions against the coherent-band phase.

    The prior ``lam·||θ - θ_init||²`` both regularises the (weakly constrained,
    at low frequency) absolute positions and *fixes the gauge*: relative delays
    determine geometry only up to a global rigid transform + reflection, and the
    anchor to the nominal geometry removes that freedom. ``mag_weight`` adds an
    optional small coherence-weighted ``|RTF|`` residual (``r_ref/r_m`` vs
    measured, in log domain).
    """
    mic_nom = torch.as_tensor(mic_init, dtype=torch.float64)
    rot_nom = torch.as_tensor(rotor_init, dtype=torch.float64)
    mic = mic_nom.clone().requires_grad_(True)
    params = [mic]
    if refine_rotors:
        rotor = rot_nom.clone().requires_grad_(True)
        params.append(rotor)
    else:
        rotor = rot_nom
    opt = torch.optim.Adam(params, lr=lr)
    for _ in range(iters):
        opt.zero_grad()
        tot, wtot = _phase_terms(records, mic, rotor, c, 0.0)
        loss = tot / torch.clamp(wtot, min=1e-12)
        if mag_weight > 0.0:
            loss = loss + mag_weight * _mag_terms(records, mic, rotor)
        loss = loss + lam * ((mic - mic_nom) ** 2).sum()
        if refine_rotors:
            loss = loss + lam * ((rotor - rot_nom) ** 2).sum()
        loss.backward()
        opt.step()
    return mic.detach().numpy(), rotor.detach().numpy()


def _mag_terms(records: list[RotorBandRTF], mic: torch.Tensor, rotor: torch.Tensor) -> torch.Tensor:
    tot = mic.new_zeros(())
    wtot = mic.new_zeros(())
    for rc in records:
        r = rc.rotor
        ref = rc.ref
        d = torch.linalg.norm(mic - rotor[r][None, :], dim=1)
        mag_ff = (d[ref] / d)[:, None]  # (C, 1)
        meas = torch.as_tensor(rc.meas_mag, dtype=mic.dtype)
        w = torch.as_tensor(rc.coh, dtype=mic.dtype)
        e = torch.log(meas + 1e-6) - torch.log(mag_ff + 1e-6)
        tot = tot + (w * e**2).sum()
        wtot = wtot + w.sum()
    return tot / torch.clamp(wtot, min=1e-12)


# ---------------------------------------------------------------------------
# Gauge handling — rigid Procrustes alignment to the nominal
# ---------------------------------------------------------------------------
def procrustes_align(
    src: np.ndarray, dst: np.ndarray, allow_reflection: bool = False
) -> tuple[np.ndarray, float]:
    """Rigidly align ``src`` onto ``dst`` (rotation + translation, no scale).

    Returns ``(src_aligned, rmse_cm)``. Used to strip the residual global rigid
    motion the weak low-frequency data + prior leave behind, so the *reported*
    per-point deltas reflect genuine shape change, not gauge drift.
    """
    cs = src.mean(0)
    cd = dst.mean(0)
    a = src - cs
    b = dst - cd
    u, _, vt = np.linalg.svd(a.T @ b)
    rot = u @ vt
    if not allow_reflection and np.linalg.det(rot) < 0:
        u[:, -1] *= -1.0
        rot = u @ vt
    aligned = a @ rot + cd
    rmse = float(np.sqrt(np.mean(np.sum((aligned - dst) ** 2, axis=1)))) * 100.0
    return aligned, rmse


# ---------------------------------------------------------------------------
# TDOA correlation metric (predicted geometry vs measured delays)
# ---------------------------------------------------------------------------
def tdoa_correlation(
    meas_tdoa: np.ndarray,
    mic_pos: np.ndarray,
    rotor_pos: np.ndarray,
    rotor_order: list[int],
    ref: int,
    sr: int,
) -> float:
    """Correlation between measured TDOA matrix and the free-field prediction."""
    dist = s0.distance_matrix(mic_pos, rotor_pos)
    pred = np.vstack([s0.freefield_tdoa_row(dist[r], ref, sr) for r in rotor_order])
    return float(np.corrcoef(meas_tdoa.ravel(), pred.ravel())[0, 1])


def _mean_abs_mag_err_db(
    records: list[RotorBandRTF], mic_pos: np.ndarray, rotor_pos: np.ndarray
) -> float:
    """Mean over rotors/mics of |coherence-weighted |RTF| error| in dB."""
    vals: list[float] = []
    for rc in records:
        dist = s0.distance_matrix(mic_pos, rotor_pos)[rc.rotor]
        rtf_ff = s0.freefield_rtf(rc.freqs, dist, rc.ref)
        rtf_meas = rc.meas_mag * np.exp(1j * rc.meas_phase)
        err = s0.coherence_weighted_mag_error(
            rtf_meas, rtf_ff, rc.coh, rc.freqs, band=(rc.freqs.min(), rc.freqs.max())
        )
        vals.append(float(np.mean(np.abs(err))))
    return float(np.mean(vals))


# ===========================================================================
# Q1 — DREGON position self-calibration
# ===========================================================================
def build_dregon_records(
    dregon_dir: Path,
    mic_nominal: np.ndarray,
    rotor_pos: np.ndarray,
    speeds: tuple[int, ...] = (60, 70, 80),
    band: tuple[float, float] = (400.0, 800.0),
    max_seconds: float = 15.0,
    nperseg: int = 8192,
) -> tuple[list[RotorBandRTF], list[int], dict[int, np.ndarray], int]:
    """Build band-limited RTF records for every rotor×speed single-motor file.

    The per-rotor reference mic is the nearest mic to that rotor in the nominal
    geometry. Returns ``(records, refs, x_by_rotor, sr)`` where ``x_by_rotor``
    holds one representative recording per rotor for the (broadband GCC-PHAT)
    TDOA validation.
    """
    dist = s0.distance_matrix(mic_nominal, rotor_pos)
    refs = [int(np.argmin(dist[r])) for r in range(rotor_pos.shape[0])]
    records: list[RotorBandRTF] = []
    x_by_rotor: dict[int, np.ndarray] = {}
    sr = 0
    for r in range(rotor_pos.shape[0]):
        for sp in speeds:
            x, sr = s0.load_motor(dregon_dir, r + 1, sp, max_seconds=max_seconds)
            if r not in x_by_rotor:
                x_by_rotor[r] = x
            freqs, rtf, coh = s0.estimate_rtf(x, sr, refs[r], nperseg=nperseg)
            sel = (freqs >= band[0]) & (freqs <= band[1])
            records.append(
                RotorBandRTF(
                    rotor=r,
                    ref=refs[r],
                    freqs=freqs[sel],
                    # sign fix: bring csd phase into the free-field convention
                    meas_phase=-np.angle(rtf[:, sel]),
                    meas_mag=np.abs(rtf[:, sel]),
                    coh=coh[:, sel],
                )
            )
    return records, refs, x_by_rotor, sr


def calibrate_dregon_positions(
    dregon_dir: Path | str | None = None,
    speeds: tuple[int, ...] = (60, 70, 80),
    band: tuple[float, float] = (400.0, 800.0),
    lam: float = 50.0,
    max_seconds: float = 15.0,
    refine_rotors: bool = False,
    iters: int = 1500,
    lr: float = 1e-3,
    frame_correction_deg: float = 0.0,
    coh_thr_hi: float = 0.8,
) -> CalibrationResult:
    """Self-calibrate DREGON mic (and optionally rotor) positions from audio.

    ``dregon.get_geometry`` now returns the 180°-z-frame-corrected mic array
    (the Stage-0 fix that turns the shipped anti-correlated TDOAs into +0.93 —
    see ``dregon._correct_mic_frame``), so the nominal is already in the right
    frame and ``frame_correction_deg`` defaults to 0. It is kept only as an
    override to re-probe the frame; the bundle adjustment refines the residual
    few-degree offset itself within a prior of strength ``lam``. (Historically
    this applied 183° on top of the *raw* shipped frame; the 3° beyond the true
    180° flip was sweep noise the refiner now absorbs.)
    """
    from data_processing.dregon import get_geometry

    dd = s0.find_dregon_dir(Path(dregon_dir)) if dregon_dir else s0.find_dregon_dir()
    mic_raw, rotor_pos = get_geometry(dd)
    mic_init = s0.rotate_z(mic_raw, frame_correction_deg)

    records, _, x_by_rotor, sr = build_dregon_records(
        dd, mic_init, rotor_pos, speeds=speeds, band=band, max_seconds=max_seconds
    )

    mic_opt, rotor_opt = run_bundle_adjustment(
        records, mic_init, rotor_pos, lam=lam, iters=iters, lr=lr, refine_rotors=refine_rotors
    )

    # Measured (broadband) TDOA matrix for validation, ref mic 0 across the array.
    rotor_order = sorted(x_by_rotor)
    meas_tdoa = np.vstack([s0.measured_tdoa_row(x_by_rotor[r], 0) for r in rotor_order])

    return _assemble_result(
        records,
        mic_init,
        rotor_pos,
        mic_opt,
        rotor_opt,
        meas_tdoa,
        rotor_order,
        ref=0,
        sr=sr,
        coh_thr_hi=coh_thr_hi,
    )


# ===========================================================================
# Q2 — Michael's: geometry from audio only (telemetry-gated separation)
# ===========================================================================
def find_data_root(start: Path | str | None = None) -> Path:
    """Walk upward to the checkout ``data/`` dir holding Michael's recordings."""
    here = Path(start).resolve() if start is not None else Path.cwd().resolve()
    for base in [here, *here.parents]:
        cand = base / "data" / "recording_with_motor_speed"
        if cand.is_dir():
            return base / "data"
    raise FileNotFoundError(f"Could not locate data/recording_with_motor_speed from {here}")


def _harmonic_comb(freqs: np.ndarray, f0_t: np.ndarray, fmax: float, tol_hz: float) -> np.ndarray:
    """Boolean ``(F, T)`` mask: bins within ``tol_hz`` of any harmonic ``k·f0_t``."""
    kmax = int(fmax // max(float(f0_t.max()), 1.0)) + 1
    mask = np.zeros((freqs.size, f0_t.size), dtype=bool)
    for k in range(1, kmax + 1):
        mask |= np.abs(freqs[:, None] - (k * f0_t)[None, :]) <= tol_hz
    return mask


def extract_michaels_rotor_rtfs(
    wav: np.ndarray,
    ts: np.ndarray,
    ms: np.ndarray,
    sr: int,
    windows: tuple[float, ...] = (40.0, 60.0, 80.0),
    win_seconds: float = 20.0,
    ref: int = 0,
    n_blades: int = 2,
    fmax: float = 2500.0,
    nperseg: int = 4096,
    hop: int = 1024,
    min_count: int = 10,
    tol_bins: float = 1.5,
) -> list[RotorBandRTF]:
    """Telemetry-gated per-rotor RTF estimation for Michael's simultaneous rotors.

    For each rotor ``i`` we accumulate cross-spectra (relative to ``ref``) over
    the TF bins that lie on one of rotor ``i``'s harmonics (``k · n_blades ·
    rps_i(t)``) and on **no** other rotor's harmonic. Cross-spectra are built as
    ``X_m · conj(X_ref)`` (free-field sign convention, no negation needed).
    Averaging over several ``windows`` improves the estimate and covers the case
    where two rotors momentarily share an rps.
    """
    n_rotor = ms.shape[0]
    n_mic = wav.shape[0]
    acc_num = [np.zeros((n_mic, nperseg // 2 + 1), dtype=np.complex128) for _ in range(n_rotor)]
    acc_smm = [np.zeros((n_mic, nperseg // 2 + 1)) for _ in range(n_rotor)]
    acc_srr = [np.zeros(nperseg // 2 + 1) for _ in range(n_rotor)]
    acc_cnt = [np.zeros(nperseg // 2 + 1) for _ in range(n_rotor)]
    freqs = np.array([])

    for t0 in windows:
        i0 = int(t0 * sr)
        i1 = int((t0 + win_seconds) * sr)
        if i1 > wav.shape[1]:
            continue
        seg = wav[:, i0:i1]
        freqs, tt, zxx = stft(seg, fs=sr, nperseg=nperseg, noverlap=nperseg - hop, axis=-1)
        abst = t0 + tt
        rps_f = np.array([np.interp(abst, ts, ms[r]) for r in range(n_rotor)])
        f0 = n_blades * rps_f
        tol_hz = tol_bins * float(freqs[1])
        combs = [_harmonic_comb(freqs, f0[r], fmax, tol_hz) for r in range(n_rotor)]
        xr = zxx[ref]
        for i in range(n_rotor):
            others = np.any([combs[j] for j in range(n_rotor) if j != i], axis=0)
            sel = combs[i] & ~others
            for m in range(n_mic):
                prod = np.where(sel, zxx[m] * np.conj(xr), 0.0)
                acc_num[i][m] += prod.sum(axis=1)
                acc_smm[i][m] += np.where(sel, np.abs(zxx[m]) ** 2, 0.0).sum(axis=1)
            acc_srr[i] += np.where(sel, np.abs(xr) ** 2, 0.0).sum(axis=1)
            acc_cnt[i] += sel.sum(axis=1)

    records: list[RotorBandRTF] = []
    for i in range(n_rotor):
        good = acc_cnt[i] >= min_count
        srr = acc_srr[i][good]
        rtf = acc_num[i][:, good] / srr[None, :]
        coh = np.abs(acc_num[i][:, good]) ** 2 / (acc_smm[i][:, good] * srr[None, :] + 1e-20)
        records.append(
            RotorBandRTF(
                rotor=i,
                ref=ref,
                freqs=freqs[good],
                meas_phase=np.angle(rtf),  # already free-field sign
                meas_mag=np.abs(rtf),
                coh=coh,
            )
        )
    return records


def michaels_tdoa_matrix(records: list[RotorBandRTF], sr: int) -> np.ndarray:
    """Per-rotor per-mic TDOA (samples) from coherence-weighted RTF phase slope.

    Fits ``phase ≈ -2πf·τ`` (free-field convention, zero intercept) per mic.
    """
    n_rotor = len(records)
    n_mic = records[0].meas_phase.shape[0]
    out = np.zeros((n_rotor, n_mic))
    for rc in records:
        for m in range(n_mic):
            ph = np.unwrap(rc.meas_phase[m])
            w = rc.coh[m]
            a = -2.0 * np.pi * rc.freqs
            tau = float(np.sum(w * a * ph) / (np.sum(w * a * a) + 1e-30))
            out[rc.rotor, m] = tau * sr
    return out


@dataclass
class PermutationResult:
    """Result of the mic-channel indexing / frame-flip detection.

    Michael's mic ring lies in the Y-Z plane (normal +X), so a channel *roll*
    corresponds to re-clocking the ring about its +X normal — an orientation
    gauge we *do* want to apply — while the residual body tilt is captured by a
    continuous ``rotate_z`` sweep. A *reflection* (``flip_*``) is the only
    handedness-changing relabeling and is applied only if the best reflection
    beats the best roll by ``flip_eps``. ``selected_perm`` / ``selected_rotation``
    define the oriented init used for refinement.
    """

    best_roll_name: str
    best_roll_score: float
    best_flip_name: str
    best_flip_score: float
    margin_flip_vs_roll: float
    flip_selected: bool
    selected_perm: np.ndarray
    selected_rotation_deg: float
    identity_score: float
    table: dict[str, float]


def _dihedral_perms(n: int) -> dict[str, np.ndarray]:
    """Ring relabelings: 8 rotations (rolls) + 8 reflected rolls (flips)."""
    base = np.arange(n)
    out: dict[str, np.ndarray] = {}
    for k in range(n):
        out[f"roll{k}"] = np.roll(base, k)
    for k in range(n):
        out[f"flip_roll{k}"] = np.roll(base[::-1], k)
    return out


def detect_mic_permutation(
    meas_tdoa: np.ndarray,
    mic_nominal: np.ndarray,
    rotor_pos: np.ndarray,
    sr: int,
    ref: int = 0,
    n_angles: int = 361,
    flip_eps: float = 0.02,
) -> PermutationResult:
    """Detect a channel-indexing permutation / frame flip against the nominal.

    For each dihedral relabeling of the mic ring we permute the *measured* TDOA
    columns and score the best correlation against the free-field TDOA over a
    continuous z-rotation sweep of the nominal array. Channel *rolls* re-clock
    the ring about its normal (an orientation gauge we adopt); a *reflection*
    (``flip_*``) is the only handedness-changing relabeling, selected only if the
    best reflection beats the best roll by ``flip_eps``. The chosen relabeling +
    residual rotation define the oriented init for refinement.
    """
    n_mic = mic_nominal.shape[0]
    rotor_order = list(range(rotor_pos.shape[0]))
    angles = np.linspace(0.0, 360.0, n_angles)

    def best_over_rotation(mt: np.ndarray) -> tuple[float, float]:
        best_c = -2.0
        best_a = 0.0
        for a in angles:
            mp = s0.rotate_z(mic_nominal, a)
            c = tdoa_correlation(mt, mp, rotor_pos, rotor_order, ref, sr)
            if c > best_c:
                best_c = c
                best_a = a
        return best_c, best_a

    table: dict[str, float] = {}
    rotations: dict[str, float] = {}
    perms = _dihedral_perms(n_mic)
    for name, p in perms.items():
        c, a = best_over_rotation(meas_tdoa[:, p])
        table[name] = c
        rotations[name] = a

    roll_names = [n for n in table if n.startswith("roll")]
    flip_names = [n for n in table if n.startswith("flip")]
    best_roll = max(roll_names, key=lambda k: table[k])
    best_flip = max(flip_names, key=lambda k: table[k])
    flip_selected = table[best_flip] > table[best_roll] + flip_eps
    chosen = best_flip if flip_selected else best_roll
    return PermutationResult(
        best_roll_name=best_roll,
        best_roll_score=table[best_roll],
        best_flip_name=best_flip,
        best_flip_score=table[best_flip],
        margin_flip_vs_roll=table[best_flip] - table[best_roll],
        flip_selected=flip_selected,
        selected_perm=perms[chosen],
        selected_rotation_deg=rotations[chosen],
        identity_score=table["roll0"],
        table=table,
    )


def calibrate_michaels_positions(
    data_root: Path | str | None = None,
    recording_index: int = 0,
    windows: tuple[float, ...] = (40.0, 60.0, 80.0),
    win_seconds: float = 20.0,
    lam: float = 20.0,
    n_blades: int = 2,
    fmax: float = 2500.0,
    iters: int = 1500,
    lr: float = 1e-3,
    refine_rotors: bool = False,
    coh_thr_hi: float = 0.7,
) -> tuple[CalibrationResult, PermutationResult]:
    """Full Michael's pipeline: gated RTFs → permutation detection → refinement.

    The nominal geometry (photo-estimated) is used both as the permutation-
    detection reference and as the bundle-adjustment prior anchor. The detected
    gross z-rotation orients the array before fine refinement.
    """
    from data_processing.michaels import (
        MICHAELS_FILES,
        _load_michaels_data_raw,
        get_geometry,
    )

    root = find_data_root(data_root) if data_root is None else Path(data_root)
    wav_rel, csv_rel, off, dil = MICHAELS_FILES[recording_index]
    wav, ts, ms, sr = _load_michaels_data_raw(root / wav_rel, root / csv_rel, off, dil, sr=None)
    mic_nominal, rotor_pos = get_geometry()

    records = extract_michaels_rotor_rtfs(
        wav,
        ts,
        ms,
        sr,
        windows=windows,
        win_seconds=win_seconds,
        n_blades=n_blades,
        fmax=fmax,
    )
    meas_tdoa = michaels_tdoa_matrix(records, sr)

    perm = detect_mic_permutation(meas_tdoa, mic_nominal, rotor_pos, sr)

    # Orient the nominal array by the detected relabeling (ring re-clocking, and
    # a reflection only if it genuinely wins) + residual z-rotation, then refine
    # within the prior anchored to this oriented frame. Scoring aligned
    # ``meas[:, perm]`` to ``rotate_z(nominal, a)``, so channel c sits at
    # position ``argsort(perm)[c]`` of the rotated nominal.
    inv = np.argsort(perm.selected_perm)
    mic_init = s0.rotate_z(mic_nominal, perm.selected_rotation_deg)[inv]

    mic_opt, rotor_opt = run_bundle_adjustment(
        records, mic_init, rotor_pos, lam=lam, iters=iters, lr=lr, refine_rotors=refine_rotors
    )

    rotor_order = list(range(rotor_pos.shape[0]))
    result = _assemble_result(
        records,
        mic_init,
        rotor_pos,
        mic_opt,
        rotor_opt,
        meas_tdoa,
        rotor_order,
        ref=0,
        sr=sr,
        coh_thr_hi=coh_thr_hi,
    )
    return result, perm


# ---------------------------------------------------------------------------
# Shared result assembly
# ---------------------------------------------------------------------------
def _assemble_result(
    records: list[RotorBandRTF],
    mic_init: np.ndarray,
    rotor_init: np.ndarray,
    mic_opt: np.ndarray,
    rotor_opt: np.ndarray,
    meas_tdoa: np.ndarray,
    rotor_order: list[int],
    ref: int,
    sr: int,
    coh_thr_hi: float,
) -> CalibrationResult:
    mic_delta = np.linalg.norm(mic_opt - mic_init, axis=1) * 100.0
    rotor_delta = np.linalg.norm(rotor_opt - rotor_init, axis=1) * 100.0
    aligned, _ = procrustes_align(mic_opt, mic_init)
    mic_delta_proc = np.linalg.norm(aligned - mic_init, axis=1) * 100.0

    return CalibrationResult(
        mic_init=mic_init,
        rotor_init=rotor_init,
        mic_opt=mic_opt,
        rotor_opt=rotor_opt,
        mic_delta_cm=mic_delta,
        rotor_delta_cm=rotor_delta,
        mic_delta_procrustes_cm=mic_delta_proc,
        resid_before_deg=phase_residual_rms_deg(records, mic_init, rotor_init),
        resid_after_deg=phase_residual_rms_deg(records, mic_opt, rotor_opt),
        resid_before_deg_hi=phase_residual_rms_deg(
            records, mic_init, rotor_init, coh_thr=coh_thr_hi
        ),
        resid_after_deg_hi=phase_residual_rms_deg(records, mic_opt, rotor_opt, coh_thr=coh_thr_hi),
        tdoa_corr_before=tdoa_correlation(meas_tdoa, mic_init, rotor_init, rotor_order, ref, sr),
        tdoa_corr_after=tdoa_correlation(meas_tdoa, mic_opt, rotor_opt, rotor_order, ref, sr),
        mag_err_before_db=_mean_abs_mag_err_db(records, mic_init, rotor_init),
        mag_err_after_db=_mean_abs_mag_err_db(records, mic_opt, rotor_opt),
        records=records,
    )
