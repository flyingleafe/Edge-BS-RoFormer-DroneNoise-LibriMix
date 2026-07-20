"""Can the coupled VK tracker do FULL BLIND RPS annotation? (DREGON room1 GT)

Tests whether ``data_processing.vk_tracking`` recovers correct rotor-speed
trajectories from a much worse starting point than telemetry — or from none at
all — on the 5 DREGON ``free-flight_*_room1`` recordings where
``motors_measured`` provides ground truth. Same 25 s mid-flight segments, tau
alignment, frame grid and edge masking as ``scripts/vk_validation.py`` (whose
telemetry-init refine-only reference is pooled 0.604 err / -0.075 bias).

Two-phase pipeline:
  * CAPTURE — annealed schedule: the validated basin-probe config
    ``replace(MAIN_CFG, k_schedule="grow", n_outer=12)`` (k_min=6, k_max=30,
    bw 1.5, max_step=0.3, couple_hz=20). See DOCUMENTED FIX #2 at
    ``CAPTURE_CFG`` — the literally-specified capture recipe with k_min=1
    defaults stalls from every non-telemetry init (archived under
    ``capture_v1_asspec/``).
  * REFINE — continue from the capture output with ``vk_validation.MAIN_CFG``
    (fixed schedule, k_min=6, k_max=30, bw 1.5, n_outer=5, max_step=0.3),
    which de-biases.

Conditions per recording (init for the capture phase):
  1. ``offset+2``  — telemetry command init shifted +2 rev/s on all rotors.
  2. ``offset+5``  — shifted +5 (beyond the single-phase basin).
  3. ``const-mean`` — all four rotors flat at the blind scan's best base speed.
  4. ``blind``     — full blind pipeline: base-speed comb scan -> per-peak
     init for R=4 -> capture -> refine.

DOCUMENTED FIX (the first failure hit, fixed before any tracking ran): the
predecessor's blind scan (``scripts/rps_refinement_spcup.py``, raw
``comb_score`` = mean log-magnitude along the comb) is envelope-dominated on
DREGON — low-frequency rumble makes SMALLER bases score higher (constant
37.15 rev/s scores 0.999 while the TRUE measured trajectory scores −0.29), so
the scan returns ~half the true base and its octave rule ("prefer smaller")
cements the error. Fix: **whiten** the log-mag spectrum (subtract a running
median over frequency, ~150 Hz window) before sampling the comb, so the score
measures line evidence above the local background. On the whitened scan the
top peak lands on the true fast pair (~86 rev/s) with the slow pair (~75)
next; the surviving subharmonic peak (43 ≈ 86/2) is rejected by a
harmonic-relation guard when picking the second peak for the R=4 init.

Scoring vs measured truth (raw + 0.25 s-smoothed), edge-masked, PIT-aligned
(4!-permutation search — blind conditions carry no rotor identity): pooled +
per-recording unsigned err and bias, per-rotor errors, and twin-pair
resolution (DREGON rotors fly as two tight pairs ~0.65 rev/s apart; report the
min pairwise track separation vs truth's and whether the 4 tracks match 4
distinct rotors).

Run:      nice -n 10 .venv/bin/python scripts/vk_blind_annotation.py
Trace:    ... scripts/vk_blind_annotation.py --trace <rid> <condition>
          (per-round trajectory evolution of a failing case, for diagnosis)

Artifacts (``results/vk_tracking/blind_annotation/``): ``scan_<rid>.npz``,
``<rid>__<cond>.npz``, ``summary.json``, ``blind_easy.png``/``blind_hard.png``,
``trace_<rid>_<cond>.png``. Per-run NPZs make the script resumable (existing
runs are not recomputed).
"""

from __future__ import annotations

import os

# Cap BLAS threads BEFORE numpy import (process-level parallelism instead).
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import argparse  # noqa: E402
import itertools  # noqa: E402
import json  # noqa: E402
import multiprocessing  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from dataclasses import asdict, replace  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

from vk_validation import (  # noqa: E402
    DREGON_TARGETS,
    MAIN_CFG,
    Prepared,
)
from vk_validation import (  # noqa: E402
    prepare_recording as _prepare_recording_uncached,
)

from data_processing.vk_tracking import (  # noqa: E402
    Envelopes,
    VKConfig,
    vk_envelopes,
    vk_reconstruct,
    vk_track,
)

SR = 16000
OUT_DIR = Path("results/vk_tracking/blind_annotation")

# Prep cache: loading + resampling a DREGON recording (load_timeframe at
# target_sr=16000) dominates worker startup; the resulting 25 s segment +
# telemetry + tau are tiny and deterministic per rid, so cache them as NPZ.
_PREP_CACHE_DIR = OUT_DIR / "prep_cache"
_PREP_CACHE_FIELDS = (
    "tau",
    "seg_lo",
    "seg_hi",
    "audio",
    "ft",
    "r_init",
    "r_meas",
    "r_meas_sm",
    "edge",
)


def prepare_recording(rid: str) -> Prepared:
    """Cached ``vk_validation.prepare_recording`` (NPZ under ``prep_cache/``)."""
    path = _PREP_CACHE_DIR / f"{rid}.npz"
    if path.exists():
        with np.load(path) as z:
            arrs = {k: z[k] for k in _PREP_CACHE_FIELDS}
        return Prepared(
            rid=rid,
            tau=float(arrs["tau"]),
            seg_lo=float(arrs["seg_lo"]),
            seg_hi=float(arrs["seg_hi"]),
            audio=arrs["audio"],
            ft=arrs["ft"],
            r_init=arrs["r_init"],
            r_meas=arrs["r_meas"],
            r_meas_sm=arrs["r_meas_sm"],
            edge=arrs["edge"].astype(bool),
        )
    prep = _prepare_recording_uncached(rid)
    _PREP_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        path, allow_pickle=False, **{k: np.asarray(getattr(prep, k)) for k in _PREP_CACHE_FIELDS}
    )
    return prep


# CAPTURE phase: annealed schedule (grow k_max + bandwidth), wide early bands.
#
# DOCUMENTED FIX #2 (capture config — the one allowed tracker-fix iteration):
# the first full run used the literally-specified
# ``VKConfig(fs, k_schedule="grow", n_outer=10, k_max=30, couple_hz=20)`` with
# library defaults (k_min=1, bw_hz=1.0, max_step=0.5) — and it stalled or
# actively repelled from EVERY non-telemetry init (5/5 jobs; e.g. offset+2 on
# nosource: init err 1.97 -> captured 1.92, per-round max deltas 0.21->0.05
# with mean rotor movement ~0; blind-from-near-truth got WORSE, 1.20 -> 1.50,
# results archived under ``capture_v1_asspec/``, per-round trajectory trace in
# ``trace_v1_*``). The capture-basin probe that validated +-2..3 rev/s
# recovery (results/vk_tracking/validation "probe (grow schedule): basin >=
# 2 rev/s") did NOT use those defaults — it used the grow schedule on top of
# the refine config: ``replace(MAIN_CFG, k_schedule="grow", n_outer=12)``,
# i.e. k_min=6 (low twin-merged harmonics excluded from the Fisher fusion),
# bw_hz=1.5, max_step=0.3. Fix = use exactly that validated capture config.
CAPTURE_CFG = replace(MAIN_CFG, k_schedule="grow", n_outer=12)
REFINE_CFG = MAIN_CFG  # fixed, k_min=6, k_max=30, bw 1.5, n_outer=5, step 0.3

# Wander-tracking fix experiment (coordinator-directed, blind condition only).
# Diagnosis: refine's evidence bandwidth is +-bw/(2k) ~ 0.125 rev/s at k=6 —
# the +-1.5 rev/s flight wander leaves the band and updates die, so blind
# tracks flatline at the scan mean. Arm A inserts a mid-band "track" phase
# between capture and refine: at k=6-12 a 7 Hz band reads detunings up to
# +-0.5 rev/s and re-centers every round — wide enough to follow the wander,
# high-enough k to reject twins.
TRACK_CFG = VKConfig(
    fs=float(SR),
    k_schedule="fixed",
    n_outer=8,
    k_min=6,
    k_max=12,
    bw_hz=7.0,
    max_step=0.5,
    couple_hz=20.0,
    update_gate=8.0,
)
FIX_TARGETS = ("free-flight_nosource_room1", "free-flight_speech-low_room1")
# Arm B: windowed whitened scan (the scan read the base speed correctly from
# nothing — run it per short window so it reads the wander too).
WSCAN_WIN_S = 2.0
WSCAN_HOP_S = 1.0

CONDITIONS = ("offset+2", "offset+5", "const-mean", "blind")
N_WORKERS = 5

# Blind base-speed scan (whitened; see the module docstring's documented fix).
SCAN_LO, SCAN_HI, SCAN_STEP = 30.0, 120.0, 0.05
SCAN_K_MAX = 40
WHITEN_HZ = 150.0  # running-median window (Hz) subtracted from the log-mag
OCTAVE_REL = 0.9  # prefer the half base when its score >= this fraction
HARM_GUARD = 1.5  # rev/s: 2nd init peak must not be a half/double of the 1st
PAIR_NUDGE = 0.5  # rev/s: 2 rotors per chosen peak at peak -/+ nudge
BLIND_OFFSETS = (-1.5, -0.5, 0.5, 1.5)  # fallback when only one peak exists

# Shared scan machinery now lives in data_processing.vk_blind_seeding (design
# §7 blind seeding v2); the historical constants above are wired through
# explicitly so the two places cannot drift apart.
from data_processing.vk_blind_seeding import SeedConfig as _SeedConfig  # noqa: E402

_SEED_CFG = _SeedConfig(
    scan_lo=SCAN_LO,
    scan_hi=SCAN_HI,
    scan_step=SCAN_STEP,
    k_scan=SCAN_K_MAX,
    whiten_hz=WHITEN_HZ,
    octave_rel=OCTAVE_REL,
    harm_guard=HARM_GUARD,
    pair_nudge=PAIR_NUDGE,
    blind_offsets=BLIND_OFFSETS,
)

# Reference points (results/vk_tracking/validation/summary.json).
REFERENCE = {
    "telemetry_refine_pooled": {"err": 0.604, "bias": -0.075},
    "telemetry_pooled": {"err": 0.609, "bias": -0.072},
}


# ---------------------------------------------------------------------------
# Blind initialisation (base-speed scan, cached per recording)


def _whitened_spec(prep: Prepared) -> tuple[np.ndarray, float, np.ndarray]:
    """``(F, N)`` channel-averaged whitened log-mag + ``(bin_hz, frame_times)``.

    The raw ``comb_score`` scan is envelope-dominated on DREGON (see the module
    docstring) — whitening subtracts a running median over frequency
    (``WHITEN_HZ`` window) so comb scores measure line evidence above the
    local background. Now a thin wrapper over
    ``data_processing.vk_blind_seeding.whitened_logmag`` (the scan machinery's
    shared home since blind seeding v2, design §7); ``_SEED_CFG`` reproduces
    this script's historical constants exactly.
    """
    from data_processing.vk_blind_seeding import whitened_logmag

    return whitened_logmag(prep.audio, float(SR), _SEED_CFG)


def _comb_scan(white_vec: np.ndarray, bin_hz: float, grid: np.ndarray) -> np.ndarray:
    """Whitened comb score of each constant base in ``grid`` (k=1..SCAN_K_MAX).

    Delegates to ``vk_blind_seeding.comb_scan`` (flat template = numerically
    identical to the historical implementation).
    """
    from data_processing.vk_blind_seeding import comb_scan

    return comb_scan(white_vec, bin_hz, grid, _SEED_CFG)


def _scan_peaks(grid: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Indices of local maxima (>=1.5 rev/s apart), fallback to the argmax."""
    from data_processing.vk_blind_seeding import scan_peaks

    return scan_peaks(grid, scores, _SEED_CFG)


def blind_scan(prep: Prepared) -> dict[str, Any]:
    """Whitened comb-score scan over constant base speeds; base + R=4 init."""
    white, bin_hz, _ = _whitened_spec(prep)
    grid = np.arange(SCAN_LO, SCAN_HI + SCAN_STEP / 2, SCAN_STEP)
    scores = _comb_scan(white.mean(axis=1), bin_hz, grid)
    idx_pk = _scan_peaks(grid, scores)
    order = np.argsort(scores[idx_pk])[::-1]
    peak_speeds, peak_scores = grid[idx_pk][order], scores[idx_pk][order]

    # Octave-down ambiguity: prefer the half base when its own peak scores
    # nearly as high (relative margin — a true base at b/2 implies the b comb
    # is its even-harmonic subset).
    base, octave = float(peak_speeds[0]), False
    half = base / 2.0
    if half >= grid[0]:
        d = np.abs(peak_speeds - half)
        j = int(np.argmin(d))
        if float(d[j]) <= 1.0 and float(peak_scores[j]) >= OCTAVE_REL * float(peak_scores[0]):
            base, octave = float(peak_speeds[j]), True

    # R=4 init: two rotors per peak on the top two harmonically-UNrelated peaks
    # (quadrotor = two tight pairs); a peak whose half/double coincides with the
    # primary (e.g. 43 ~= 86/2) is a subharmonic alias, not a second rotor pair.
    second = None
    for cand in peak_speeds[1:]:
        c = float(cand)
        if abs(c - base) <= 2.0 * PAIR_NUDGE:
            continue
        if abs(2.0 * c - base) < HARM_GUARD or abs(c - 2.0 * base) < HARM_GUARD:
            continue
        second = c
        break
    if second is None:
        init4 = base + np.asarray(BLIND_OFFSETS)
    else:
        init4 = np.sort(
            np.array(
                [base - PAIR_NUDGE, base + PAIR_NUDGE, second - PAIR_NUDGE, second + PAIR_NUDGE]
            )
        )
    return {
        "grid": grid,
        "scores": scores,
        "peak_speeds": peak_speeds,
        "peak_scores": peak_scores,
        "base": base,
        "octave": octave,
        "second": np.nan if second is None else second,
        "init4": np.clip(init4, SCAN_LO, SCAN_HI),
    }


def get_scan(rid: str) -> dict[str, Any]:
    """Load the cached blind scan for ``rid``, computing it once if missing."""
    path = OUT_DIR / f"scan_{rid}.npz"
    if path.exists():
        with np.load(path) as z:
            return {k: z[k] for k in z.files}
    prep = prepare_recording(rid)
    tic = time.perf_counter()
    scan = blind_scan(prep)
    print(
        f"[scan {rid}] base={scan['base']:.2f} octave={scan['octave']} "
        f"init4={np.round(scan['init4'], 2)} ({time.perf_counter() - tic:.0f}s)",
        flush=True,
    )
    np.savez(path, **scan)
    return scan


def build_init(prep: Prepared, cond: str, scan: dict[str, Any]) -> np.ndarray:
    """(4, N) capture-phase init for one condition."""
    n = len(prep.ft)
    if cond == "offset+2":
        return prep.r_init + 2.0
    if cond == "offset+5":
        return prep.r_init + 5.0
    if cond == "const-mean":
        return np.full((4, n), float(scan["base"]))
    if cond == "blind":
        return np.repeat(np.asarray(scan["init4"], dtype=np.float64)[:, None], n, axis=1)
    raise ValueError(f"unknown condition {cond!r}")


# ---------------------------------------------------------------------------
# Workers


def run_condition(rid: str, cond: str) -> str:
    """Worker: capture + refine one (recording, condition); save NPZ, return path."""
    path = OUT_DIR / f"{rid}__{cond}.npz"
    if path.exists():
        return str(path)
    prep = prepare_recording(rid)
    scan = get_scan(rid)
    r0 = build_init(prep, cond, scan)

    tic = time.perf_counter()
    cap = vk_track(prep.audio, r0, prep.ft, CAPTURE_CFG)
    wall_cap = time.perf_counter() - tic
    tic = time.perf_counter()
    ref = vk_track(prep.audio, cap.r_refined, prep.ft, REFINE_CFG)
    wall_ref = time.perf_counter() - tic
    print(
        f"[{rid} | {cond}] capture {wall_cap:.0f}s refine {wall_ref:.0f}s  "
        f"resid {cap.residual_ratios[0]:.3f}->{ref.residual_ratios[-1]:.3f}",
        flush=True,
    )
    np.savez(
        path,
        ft=prep.ft,
        edge=prep.edge,
        init=r0,
        command=prep.r_init,
        measured=prep.r_meas,
        measured_sm=prep.r_meas_sm,
        captured=cap.r_refined,
        refined=ref.r_refined,
        confidence=ref.confidence,
        cap_residual_ratios=np.array(cap.residual_ratios),
        cap_max_deltas=np.array(cap.max_deltas),
        ref_residual_ratios=np.array(ref.residual_ratios),
        ref_max_deltas=np.array(ref.max_deltas),
        tau=prep.tau,
        seg_bounds=np.array([prep.seg_lo, prep.seg_hi]),
        wall_capture_s=wall_cap,
        wall_refine_s=wall_ref,
        base=float(scan["base"]),
        init4=np.asarray(scan["init4"], dtype=np.float64),
    )
    return str(path)


def windowed_scan_traj(
    prep: Prepared, base0: float, max_jump: float = 1.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Arm B init: whitened scan per short window -> base-speed *trajectory*.

    Runs the whitened comb scan on ``WSCAN_WIN_S``-second windows (hop
    ``WSCAN_HOP_S``); each window's base is the local maximum nearest the
    previous window's value (continuity), seeded with the GLOBAL scan base
    ``base0`` (a window's own argmax may sit on the other rotor pair). A
    window whose nearest peak jumps more than ``max_jump`` rev/s keeps the
    previous value (rotors move < ~1 rev/s per hop; larger jumps are the scan
    hopping to the other pair or an alias). Returns ``(r_base on prep.ft,
    window centers, per-window bases)``.
    """
    white, bin_hz, st = _whitened_spec(prep)
    grid = np.arange(SCAN_LO, SCAN_HI + SCAN_STEP / 2, SCAN_STEP)
    centers: list[float] = []
    bases: list[float] = []
    prev = float(base0)
    t0 = 0.0
    while t0 + WSCAN_WIN_S <= float(st[-1]) + 1e-9:
        sel = (st >= t0) & (st < t0 + WSCAN_WIN_S)
        if int(sel.sum()) >= 4:
            scores = _comb_scan(white[:, sel].mean(axis=1), bin_hz, grid)
            cand = grid[_scan_peaks(grid, scores)]
            b = float(cand[np.argmin(np.abs(cand - prev))])
            if abs(b - prev) > max_jump:
                b = prev
            prev = b
            centers.append(t0 + WSCAN_WIN_S / 2.0)
            bases.append(b)
        t0 += WSCAN_HOP_S
    centers_a, bases_a = np.asarray(centers), np.asarray(bases)
    return np.interp(prep.ft, centers_a, bases_a), centers_a, bases_a


def run_arm(rid: str, arm: str) -> str:
    """Worker: one wander-tracking fix arm on the blind condition; saves NPZ.

    Arm A: continue from the SAME captured trajectories as the ladder's blind
    run (loaded from ``<rid>__blind.npz``) with the mid-band TRACK_CFG phase,
    then the standard refine. Arm B: windowed-scan trajectory init (per-rotor
    offsets = the global scan init's offsets around its base), short capture
    (n_outer=4), then the standard refine.
    """
    path = OUT_DIR / f"{rid}__blindfix{arm}.npz"
    if path.exists():
        return str(path)
    prep = prepare_recording(rid)
    scan = get_scan(rid)
    extra: dict[str, np.ndarray] = {}
    tic = time.perf_counter()
    if arm == "A":
        with np.load(OUT_DIR / f"{rid}__blind.npz") as z:
            r0 = z["init"]
            captured_prev = z["captured"]
        mid_res = vk_track(prep.audio, captured_prev, prep.ft, TRACK_CFG)
    else:
        r_base, wc, wb = windowed_scan_traj(prep, float(scan["base"]))
        offs = np.asarray(scan["init4"], dtype=np.float64) - float(scan["base"])
        r0 = r_base[None, :] + offs[:, None]
        mid_res = vk_track(prep.audio, r0, prep.ft, replace(CAPTURE_CFG, n_outer=4))
        extra = {"wscan_centers": wc, "wscan_bases": wb, "wscan_traj": r_base}
    wall_mid = time.perf_counter() - tic
    tic = time.perf_counter()
    ref = vk_track(prep.audio, mid_res.r_refined, prep.ft, REFINE_CFG)
    wall_ref = time.perf_counter() - tic
    print(
        f"[{rid} | fix{arm}] mid {wall_mid:.0f}s refine {wall_ref:.0f}s  "
        f"resid {mid_res.residual_ratios[0]:.3f}->{ref.residual_ratios[-1]:.3f}",
        flush=True,
    )
    arrays: dict[str, Any] = {
        "ft": prep.ft,
        "edge": prep.edge,
        "init": r0,
        "command": prep.r_init,
        "measured": prep.r_meas,
        "measured_sm": prep.r_meas_sm,
        "captured": mid_res.r_refined,
        "refined": ref.r_refined,
        "confidence": ref.confidence,
        "cap_residual_ratios": np.array(mid_res.residual_ratios),
        "cap_max_deltas": np.array(mid_res.max_deltas),
        "ref_residual_ratios": np.array(ref.residual_ratios),
        "ref_max_deltas": np.array(ref.max_deltas),
        "tau": prep.tau,
        "seg_bounds": np.array([prep.seg_lo, prep.seg_hi]),
        "wall_capture_s": wall_mid,
        "wall_refine_s": wall_ref,
        "base": float(scan["base"]),
        "init4": np.asarray(scan["init4"], dtype=np.float64),
        **extra,
    }
    np.savez(path, **arrays)
    return str(path)


# ---------------------------------------------------------------------------
# Scan-in-the-loop (coordinator's residual-guided mechanism)
#
# Any phase-slope update has visibility ~bw/(2k) rev/s — the blind init's
# per-rotor error (~1.0-1.2, pair means +- fixed offsets, wander missing)
# exceeds it at every twin-resolving k, which is why capture/track/refine all
# flatline. Here the *scan* moves inside the loop: each round the coupled VK
# envelope solve attributes energy to all rotors, and each rotor is rescanned
# locally (delta in +-2 rev/s) on the explaining-away residual — audio minus
# the OTHER rotors' reconstructions — where the twin's teeth are subtracted,
# making even low harmonics usable.

SCANLOOP_ROUNDS = 4
SCANLOOP_DELTA = 2.0  # rev/s: local scan half-range (also per-round clip)
SCANLOOP_DSTEP = 0.05  # rev/s: local scan grid step (parabolic-refined)
SCANLOOP_LAMBDA = 1.0  # window-grid D2 smoothing weight (~0.3-0.5 Hz traj bw)
SCANLOOP_ENV_CFG = VKConfig(
    fs=float(SR),
    k_min=1,
    k_max=30,
    bw_hz=1.5,
    couple_hz=20.0,
    k_schedule="fixed",
    n_outer=1,
)


def _recon_per_rotor(env: Envelopes, n_rotors: int, n_t: int) -> list[np.ndarray]:
    """Per-rotor reconstructions: element ``i`` uses rotor i's tracks only."""
    recons = []
    for i in range(n_rotors):
        x_i = env.x.copy()
        x_i[:, env.rotor != i] = 0.0
        recons.append(vk_reconstruct(replace(env, x=x_i), n_samples=n_t))
    return recons


def _local_comb_frame_scores(
    lm: np.ndarray, bin_hz: float, r_spec: np.ndarray, deltas: np.ndarray, ks: np.ndarray
) -> np.ndarray:
    """``(D, N)`` per-frame mean log-mag along the combs of ``r_spec + delta``.

    ``r_spec`` may be ``(N,)`` (one rotor) or ``(P, N)`` (a rigid multi-rotor
    template — e.g. a twin pair shifted together, separations frozen); the
    comb is then the union of all P rotors' teeth.
    """
    r2 = np.atleast_2d(r_spec)  # (P, N)
    n_f, n = lm.shape
    fmax = min(6000.0, (n_f - 1) * bin_hz)
    cols = np.arange(n)[None, :]
    out = np.empty((len(deltas), n))
    for di, d in enumerate(deltas):
        f = (ks[:, None, None] * (r2 + d)[None, :, :]).reshape(-1, n)  # (K*P, N)
        valid = (f >= 60.0) & (f <= fmax)
        idx = np.clip(f, 0.0, fmax) / bin_hz
        j = np.floor(idx).astype(int)
        frac = idx - j
        v = (1 - frac) * lm[j, cols] + frac * lm[np.minimum(j + 1, n_f - 1), cols]
        v = np.where(valid, v, np.nan)
        out[di] = np.nanmean(v, axis=0)
    return out


def _window_deltas(
    scores: np.ndarray,
    st: np.ndarray,
    deltas: np.ndarray,
    continuity: bool = True,
    win_s: float = WSCAN_WIN_S,
    hop_s: float = WSCAN_HOP_S,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-window best delta (parabolic-refined) + weight.

    ``continuity=True`` picks the local max nearest the previous window's
    value; ``False`` takes the global argmax — the pair-scan mechanism check
    showed continuity-from-0 glues the estimate to 0 on dense score surfaces
    while the global argmax tracks the true common-mode wander (corr ~0.8).
    """
    from scipy.signal import find_peaks

    centers: list[float] = []
    dbest: list[float] = []
    weights: list[float] = []
    prev = 0.0
    t0 = 0.0
    step = float(deltas[1] - deltas[0])
    while t0 + win_s <= float(st[-1]) + 1e-9:
        sel = (st >= t0) & (st < t0 + win_s)
        if int(sel.sum()) >= 4:
            sw = np.nanmean(scores[:, sel], axis=1)  # (D,)
            pk, _ = find_peaks(sw)
            if len(pk) == 0:
                pk = np.array([int(np.nanargmax(sw))])
            if continuity:
                gi = int(pk[np.argmin(np.abs(deltas[pk] - prev))])
            else:
                gi = int(np.nanargmax(sw))
            d0 = float(deltas[gi])
            if 0 < gi < len(deltas) - 1:  # parabolic refinement
                y0, y1, y2 = sw[gi - 1], sw[gi], sw[gi + 1]
                denom = y0 - 2.0 * y1 + y2
                if abs(denom) > 1e-12:
                    d0 += 0.5 * (y0 - y2) / denom * step
            centers.append(t0 + win_s / 2.0)
            dbest.append(float(np.clip(d0, deltas[0], deltas[-1])))
            weights.append(max(float(sw[gi] - np.nanmedian(sw)), 0.0))
            prev = dbest[-1]
        t0 += hop_s
    return np.asarray(centers), np.asarray(dbest), np.asarray(weights)


def _smooth_deltas(
    centers: np.ndarray,
    dvals: np.ndarray,
    weights: np.ndarray,
    ft: np.ndarray,
    lam: float = SCANLOOP_LAMBDA,
) -> np.ndarray:
    """Weighted D2-smoothed per-window deltas, interpolated onto ``ft``."""
    if len(centers) == 0:
        return np.zeros_like(ft)
    if len(centers) < 3:
        return np.interp(ft, centers, dvals)
    from scipy.sparse import diags_array
    from scipy.sparse.linalg import splu

    from data_processing.vk_tracking import _second_diff

    w = weights / max(float(weights.mean()), 1e-12)
    d2 = _second_diff(len(dvals))
    mat = (diags_array(w + 1e-3) + lam * (d2.T @ d2)).tocsc()
    sm = splu(mat).solve(w * dvals)
    return np.interp(ft, centers, sm)


def vk_track_scan(
    prep: Prepared, r0: np.ndarray, ks: np.ndarray, n_rounds: int = SCANLOOP_ROUNDS
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Alternate coupled envelope solves with per-rotor residual comb scans.

    Per round: (1) coupled VK envelope solve at the current trajectories
    (k 1..30, bw 1.5, couple 20); (2) per rotor ``i``, windowed local comb
    scan (harmonics ``ks``) of ``audio - reconstruction(other rotors)`` over
    ``r_i + delta``, delta in +-SCANLOOP_DELTA; (3) weighted-smoothed
    per-window deltas update ``r_i``. Returns the final trajectories and the
    per-round snapshots (element 0 = the init).
    """
    from data_processing.rps_refinement import RefineConfig, compute_logmag

    cfg_r = RefineConfig(sample_rate=SR, device="cpu")
    y = prep.audio
    n_t = y.shape[-1]
    n_rotors = r0.shape[0]
    t_aud = np.arange(n_t, dtype=np.float64) / float(SR)
    deltas = np.arange(-SCANLOOP_DELTA, SCANLOOP_DELTA + SCANLOOP_DSTEP / 2, SCANLOOP_DSTEP)
    r_cur = r0.copy()
    snaps = [r_cur.copy()]
    for _rd in range(n_rounds):
        r_aud = np.stack([np.interp(t_aud, prep.ft, r_cur[i]) for i in range(n_rotors)])
        env = vk_envelopes(y, r_aud, SCANLOOP_ENV_CFG)
        recons = _recon_per_rotor(env, n_rotors, n_t)
        recon_sum = np.sum(recons, axis=0)
        for i in range(n_rotors):
            resid_i = y - (recon_sum - recons[i])
            spec = compute_logmag(resid_i, cfg_r)
            lm = spec.logmag.cpu().numpy().mean(axis=0)  # (F, N) channel-mean
            st = np.asarray(spec.frame_times, dtype=np.float64)
            r_spec = np.interp(st, prep.ft, r_cur[i])
            fsc = _local_comb_frame_scores(lm, float(spec.bin_hz), r_spec, deltas, ks)
            centers, dvals, weights = _window_deltas(fsc, st, deltas)
            delta_ft = np.clip(
                _smooth_deltas(centers, dvals, weights, prep.ft),
                -SCANLOOP_DELTA,
                SCANLOOP_DELTA,
            )
            r_cur[i] = np.maximum(r_cur[i] + delta_ft, 0.0)
        snaps.append(r_cur.copy())
    return r_cur, snaps


def run_scanloop(rid: str, k_lo: int) -> str:
    """Worker: blind init -> scan-in-the-loop -> standard refine; saves NPZ."""
    cond = f"blindscanK{k_lo}"
    path = OUT_DIR / f"{rid}__{cond}.npz"
    if path.exists():
        return str(path)
    prep = prepare_recording(rid)
    scan = get_scan(rid)
    r0 = build_init(prep, "blind", scan)
    ks = np.arange(k_lo, SCANLOOP_ENV_CFG.k_max + 1)

    tic = time.perf_counter()
    r_scan, snaps = vk_track_scan(prep, r0, ks)
    wall_mid = time.perf_counter() - tic
    tic = time.perf_counter()
    ref = vk_track(prep.audio, r_scan, prep.ft, REFINE_CFG)
    wall_ref = time.perf_counter() - tic

    def pit_err(traj: np.ndarray) -> float:
        p = pit_perm(traj, prep.r_meas, prep.edge)
        return float(np.mean(np.abs(traj[list(p)][:, prep.edge] - prep.r_meas[:, prep.edge])))

    round_errs = [pit_err(s) for s in snaps] + [pit_err(ref.r_refined)]
    print(
        f"[{rid} | {cond}] scan {wall_mid:.0f}s refine {wall_ref:.0f}s  "
        f"PIT err per round: {[round(e, 3) for e in round_errs]}",
        flush=True,
    )
    arrays: dict[str, Any] = {
        "ft": prep.ft,
        "edge": prep.edge,
        "init": r0,
        "command": prep.r_init,
        "measured": prep.r_meas,
        "measured_sm": prep.r_meas_sm,
        "captured": r_scan,
        "refined": ref.r_refined,
        "confidence": ref.confidence,
        "cap_residual_ratios": np.zeros(0),
        "cap_max_deltas": np.zeros(0),
        "ref_residual_ratios": np.array(ref.residual_ratios),
        "ref_max_deltas": np.array(ref.max_deltas),
        "tau": prep.tau,
        "seg_bounds": np.array([prep.seg_lo, prep.seg_hi]),
        "wall_capture_s": wall_mid,
        "wall_refine_s": wall_ref,
        "base": float(scan["base"]),
        "init4": np.asarray(scan["init4"], dtype=np.float64),
        "scan_snaps": np.stack(snaps),
        "scan_round_errs": np.array(round_errs),
    }
    np.savez(path, **arrays)
    return str(path)


def scanloop_main() -> None:
    """Scan-in-the-loop experiment: k>=1 and k>=4 variants on FIX_TARGETS."""
    variants = (1, 4)
    jobs = [
        (rid, k)
        for rid in FIX_TARGETS
        for k in variants
        if not (OUT_DIR / f"{rid}__blindscanK{k}.npz").exists()
    ]
    if jobs:
        print(f"running {len(jobs)} scan-loop jobs on {len(jobs)} workers", flush=True)
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=len(jobs), mp_context=ctx) as pool:
            futs = [pool.submit(run_scanloop, rid, k) for rid, k in jobs]
            for f in futs:
                f.result()

    conds = ["blind"] + [f"blindscanK{k}" for k in variants]
    rows = [load_run(rid, cond) for rid in FIX_TARGETS for cond in conds]
    print_table(rows)
    pooled: dict[str, Any] = {}
    for cond in conds:
        sub = [r for r in rows if r["condition"] == cond]
        pooled[cond] = {
            "refined": pooled_over_recordings(sub, "refined"),
            "n_twins_resolved": sum(r["refined"]["twins_resolved"] for r in sub),
        }
        p = pooled[cond]["refined"]
        print(
            f"POOLED(2rec) {cond:<12} refined {p['err']:.3f} (bias {p['bias']:+.3f}, "
            f"vs smoothed {p['err_sm']:.3f}) | telemetry-refine ref 0.604",
            flush=True,
        )
    for rid in FIX_TARGETS:
        for k in variants:
            with np.load(OUT_DIR / f"{rid}__blindscanK{k}.npz") as z:
                errs = [round(float(v), 3) for v in z["scan_round_errs"]]
            print(f"round errs {rid} K{k}: init -> scans -> refine = {errs}", flush=True)
    best = min((f"blindscanK{k}" for k in variants), key=lambda c: pooled[c]["refined"]["err"])
    for rid in FIX_TARGETS:
        make_blind_plot(rid, f"scanloop_{best}_{rid}.png", cond=best)
    strip = lambda s: {k: v for k, v in s.items() if not k.startswith("_")}  # noqa: E731
    summary = {
        "env_config": asdict(SCANLOOP_ENV_CFG),
        "scan": {
            "rounds": SCANLOOP_ROUNDS,
            "delta": SCANLOOP_DELTA,
            "dstep": SCANLOOP_DSTEP,
            "lambda": SCANLOOP_LAMBDA,
            "win_s": WSCAN_WIN_S,
            "hop_s": WSCAN_HOP_S,
        },
        "best_variant": best,
        "rows": [
            {
                **{k: v for k, v in r.items() if k not in ("init", "captured", "refined")},
                "init": strip(r["init"]),
                "captured": strip(r["captured"]),
                "refined": strip(r["refined"]),
            }
            for r in rows
        ],
        "pooled": pooled,
    }
    with open(OUT_DIR / "scanloop_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"scan-loop artifacts: {OUT_DIR}/scanloop_summary.json + scanloop_{best}_<rid>.png")


# ---------------------------------------------------------------------------
# Pair-rigid windowed scan (coordinator's hierarchical mechanism)
#
# The wander is COMMON-MODE (rotors drift together): leave-one-out residuals
# are ~ the raw audio, so the per-rotor residual scan still sees the merged
# comb. Hierarchical decomposition r_i(t) = c_pair(t) + d_i(t): scan the PAIR
# TEMPLATE — both rotors' combs shifted rigidly, separations frozen — on the
# raw spectrum (the other pair is >= 8 rev/s away, outside the +-2 grid), then
# let the standard refine recover the small per-rotor offsets d_i (in-band
# once the common mode is tracked).

# v2 iteration (coordinator): the v1 mechanism check confirmed co-movement but
# the amplitude was attenuated ~4x — a 2 s window mean sinc-attenuates a 0.4 Hz
# wander component to ~25% and the 0.5 Hz smoother shaves more (v1 plateau at
# pooled 1.016, archived under pairscan_v1/). Hence: 1 s windows, 0.25 s hop
# (noisier per-window estimates, averaged by the weighted smoother), ~1 Hz
# trajectory bandwidth (lambda ~ 0.25 at the 4 Hz window-grid rate), and THREE
# re-centering rounds (geometric amplitude recovery: 0.5 -> 0.75 -> 0.875 of
# the true excursion at 50% capture per round) before the standard refine.
PAIRSCAN_ROUNDS = 3  # pair-scan rounds per iteration
PAIRSCAN_MAX_ITERS = 1  # single (pair-scan x3 -> refine) pass
PAIRSCAN_BAR = 0.75  # experiment-level early-stop on pooled PIT err (vs truth)
PAIRSCAN_WIN_S = 1.0
PAIRSCAN_HOP_S = 0.25
PAIRSCAN_LAMBDA = 0.25  # ~1 Hz trajectory bandwidth at the 4 Hz window grid

# v3 (coordinator): v2's residual decomposes into (a) pair separations frozen
# at the init's 1.0 rev/s guess (truth 1.36/1.58), (b) a leftover pair-mean
# bias, (c) >1 Hz wander the windowed scan cannot see. Hence: (1) a 2-D pair
# scan — (mean shift delta_c) x (separation s), template teeth at
# k*(c + delta_c +- s/2), s constant per pair per round (physically
# near-constant); (2) a MID-BAND phase round (k 6..10, bw 4 Hz — per-rotor
# tolerance bw/2k ~ 0.33 rev/s, in-band after the 2-D scan) to follow the
# 0.5-2 Hz wander; (3) the standard refine.
PAIR2D_ROUNDS = 2
PAIR2D_SEPS = np.arange(0.6, 2.4001, 0.1)  # coarse separation grid (rev/s)
PAIR2D_SEP_FINE = 0.02  # fine separation step (+-0.1 around the coarse argmax)
MIDBAND_CFG = VKConfig(
    fs=float(SR),
    k_schedule="fixed",
    n_outer=6,
    k_min=6,
    k_max=10,
    bw_hz=4.0,
    max_step=0.3,
    couple_hz=20.0,
    update_gate=8.0,
)


def vk_track_pair(
    prep: Prepared, r0: np.ndarray
) -> tuple[np.ndarray, list[tuple[str, np.ndarray]], list[tuple[int, int]], dict[str, Any], Any]:
    """Pair-rigid windowed scan + standard refine, up to PAIRSCAN_MAX_ITERS.

    Returns ``(final trajectories, [(stage label, snapshot)], pairs,
    first-round diagnostic {centers, dvals/weights per pair}, final VKResult)``.

    Deviations from the literal spec, forced by the mechanism pre-check on
    nosource: (1) the pair template is scored on the WHITENED spectrum (raw
    log-mag is envelope-dominated — established when the global scan failed);
    (2) per-window delta = global argmax, not continuity-linked (see
    ``_window_deltas``). With both, round-1 window deltas track the true
    pair-mean wander at corr ~0.75-0.81, MAE ~0.63.
    """
    lm, bin_hz, st = _whitened_spec(prep)  # (F, N) channel-mean whitened
    ks = np.arange(1, 31)
    deltas = np.arange(-SCANLOOP_DELTA, SCANLOOP_DELTA + SCANLOOP_DSTEP / 2, SCANLOOP_DSTEP)

    r_cur = r0.copy()
    order = np.argsort(r_cur.mean(axis=1))
    pairs = [(int(order[0]), int(order[1])), (int(order[2]), int(order[3]))]
    stages: list[tuple[str, np.ndarray]] = [("init", r_cur.copy())]
    diag: dict[str, Any] = {}

    def pit_err(traj: np.ndarray) -> float:
        p = pit_perm(traj, prep.r_meas, prep.edge)
        return float(np.mean(np.abs(traj[list(p)][:, prep.edge] - prep.r_meas[:, prep.edge])))

    ref = None
    for it in range(PAIRSCAN_MAX_ITERS):
        for rd in range(PAIRSCAN_ROUNDS):
            for pi, pair in enumerate(pairs):
                r_specs = np.stack([np.interp(st, prep.ft, r_cur[i]) for i in pair])
                fsc = _local_comb_frame_scores(lm, bin_hz, r_specs, deltas, ks)
                centers, dvals, weights = _window_deltas(
                    fsc,
                    st,
                    deltas,
                    continuity=False,
                    win_s=PAIRSCAN_WIN_S,
                    hop_s=PAIRSCAN_HOP_S,
                )
                if it == 0:  # mechanism-verification record, every round
                    diag[f"rd{rd}_centers"] = centers
                    diag[f"rd{rd}_dvals_p{pi}"] = dvals
                    diag[f"rd{rd}_weights_p{pi}"] = weights
                    diag[f"rd{rd}_r_before_p{pi}"] = np.stack([r_cur[i].copy() for i in pair])
                d_ft = np.clip(
                    _smooth_deltas(centers, dvals, weights, prep.ft, lam=PAIRSCAN_LAMBDA),
                    -SCANLOOP_DELTA,
                    SCANLOOP_DELTA,
                )
                for i in pair:
                    r_cur[i] = np.maximum(r_cur[i] + d_ft, 0.0)
            stages.append((f"it{it}.pairscan{rd}", r_cur.copy()))
        ref = vk_track(prep.audio, r_cur, prep.ft, REFINE_CFG)
        r_cur = ref.r_refined.copy()
        stages.append((f"it{it}.refine", r_cur.copy()))
        if pit_err(r_cur) <= PAIRSCAN_BAR:
            break
    assert ref is not None
    return r_cur, stages, pairs, diag, ref


def _pair2d_round(
    lm: np.ndarray,
    bin_hz: float,
    st: np.ndarray,
    ft: np.ndarray,
    r_pair: np.ndarray,
    deltas: np.ndarray,
    ks: np.ndarray,
) -> tuple[np.ndarray, float, dict[str, np.ndarray]]:
    """One 2-D (mean-shift x separation) scan round for one pair.

    For each candidate separation ``s`` the pair template ``k*(c + d +- s/2)``
    is scanned over ``d`` per window; ``s`` (constant per round — physically
    near-constant) is chosen by the total window score contrast, coarse grid
    then a +-0.1 fine pass. With ``s`` fixed, the per-window mean shifts are
    weight-smoothed onto ``ft``. Returns ``(c_new on ft, s, diag)``.
    """

    def scan_at_sep(s: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        c_spec = np.interp(st, ft, r_pair.mean(axis=0))
        r2 = np.stack([c_spec - s / 2.0, c_spec + s / 2.0])
        fsc = _local_comb_frame_scores(lm, bin_hz, r2, deltas, ks)
        return _window_deltas(
            fsc, st, deltas, continuity=False, win_s=PAIRSCAN_WIN_S, hop_s=PAIRSCAN_HOP_S
        )

    coarse = {float(s): scan_at_sep(float(s)) for s in PAIR2D_SEPS}
    s0 = max(coarse, key=lambda s: float(coarse[s][2].sum()))
    fine_grid = np.arange(s0 - 0.1, s0 + 0.1 + PAIR2D_SEP_FINE / 2, PAIR2D_SEP_FINE)
    fine = {
        float(s): (coarse[float(s)] if float(s) in coarse else scan_at_sep(float(s)))
        for s in fine_grid
    }
    s_best = max(fine, key=lambda s: float(fine[s][2].sum()))
    centers, dvals, weights = fine[s_best]
    d_ft = np.clip(
        _smooth_deltas(centers, dvals, weights, ft, lam=PAIRSCAN_LAMBDA),
        -SCANLOOP_DELTA,
        SCANLOOP_DELTA,
    )
    c_new = np.maximum(r_pair.mean(axis=0) + d_ft, 0.0)
    diag = {"centers": centers, "dvals": dvals, "weights": weights}
    return c_new, float(s_best), diag


def run_pair2d(rid: str) -> str:
    """Worker: blind init -> 2-D pair scan x2 -> mid-band phase -> refine."""
    cond = "blindpair2d"
    path = OUT_DIR / f"{rid}__{cond}.npz"
    if path.exists():
        return str(path)
    prep = prepare_recording(rid)
    scan = get_scan(rid)
    r0 = build_init(prep, "blind", scan)
    lm, bin_hz, st = _whitened_spec(prep)
    ks = np.arange(1, 31)
    deltas = np.arange(-SCANLOOP_DELTA, SCANLOOP_DELTA + SCANLOOP_DSTEP / 2, SCANLOOP_DSTEP)

    r_cur = r0.copy()
    order = np.argsort(r_cur.mean(axis=1))
    pairs = [(int(order[0]), int(order[1])), (int(order[2]), int(order[3]))]
    stages: list[tuple[str, np.ndarray]] = [("init", r_cur.copy())]
    seps_log: list[list[float]] = []
    diag: dict[str, Any] = {}

    tic = time.perf_counter()
    for rd in range(PAIR2D_ROUNDS):
        seps_rd = []
        for pi, pair in enumerate(pairs):
            r_pair = np.stack([r_cur[i] for i in pair])
            diag[f"rd{rd}_r_before_p{pi}"] = r_pair.copy()
            c_new, s_best, d = _pair2d_round(lm, bin_hz, st, prep.ft, r_pair, deltas, ks)
            r_cur[pair[0]] = c_new - s_best / 2.0
            r_cur[pair[1]] = c_new + s_best / 2.0
            seps_rd.append(s_best)
            diag[f"rd{rd}_centers"] = d["centers"]
            diag[f"rd{rd}_dvals_p{pi}"] = d["dvals"]
            diag[f"rd{rd}_weights_p{pi}"] = d["weights"]
        seps_log.append(seps_rd)
        stages.append((f"scan2d{rd}", r_cur.copy()))
    wall_scan = time.perf_counter() - tic

    tic = time.perf_counter()
    mid = vk_track(prep.audio, r_cur, prep.ft, MIDBAND_CFG)
    r_cur = mid.r_refined.copy()
    stages.append(("midband", r_cur.copy()))
    ref = vk_track(prep.audio, r_cur, prep.ft, REFINE_CFG)
    r_cur = ref.r_refined.copy()
    stages.append(("refine", r_cur.copy()))
    wall_vk = time.perf_counter() - tic

    def pit_err(traj: np.ndarray) -> tuple[float, float]:
        p = pit_perm(traj, prep.r_meas, prep.edge)
        a = traj[list(p)]
        return (
            float(np.mean(np.abs((a - prep.r_meas)[:, prep.edge]))),
            float(np.mean(np.abs((a - prep.r_meas_sm)[:, prep.edge]))),
        )

    labels = [s[0] for s in stages]
    errs = [pit_err(s[1]) for s in stages]
    print(
        f"[{rid} | {cond}] scan {wall_scan:.0f}s vk {wall_vk:.0f}s  seps {seps_log}  "
        + ", ".join(f"{lb}={e[0]:.3f}/{e[1]:.3f}" for lb, e in zip(labels, errs)),
        flush=True,
    )
    arrays: dict[str, Any] = {
        "ft": prep.ft,
        "edge": prep.edge,
        "init": r0,
        "command": prep.r_init,
        "measured": prep.r_meas,
        "measured_sm": prep.r_meas_sm,
        "captured": stages[PAIR2D_ROUNDS][1],  # after the last 2-D scan round
        "refined": r_cur,
        "confidence": ref.confidence,
        "cap_residual_ratios": np.zeros(0),
        "cap_max_deltas": np.zeros(0),
        "ref_residual_ratios": np.array(ref.residual_ratios),
        "ref_max_deltas": np.array(ref.max_deltas),
        "tau": prep.tau,
        "seg_bounds": np.array([prep.seg_lo, prep.seg_hi]),
        "wall_capture_s": wall_scan,
        "wall_refine_s": wall_vk,
        "base": float(scan["base"]),
        "init4": np.asarray(scan["init4"], dtype=np.float64),
        "stage_snaps": np.stack([s[1] for s in stages]),
        "stage_labels": np.array(labels),
        "stage_errs": np.array([e[0] for e in errs]),
        "stage_errs_sm": np.array([e[1] for e in errs]),
        "pair_seps": np.array(seps_log),
        "pairs": np.array(pairs),
        **{f"diag_{k}": v for k, v in diag.items()},
    }
    np.savez(path, **arrays)
    return str(path)


def pair2d_main() -> None:
    """2-D pair-scan + mid-band + refine experiment on FIX_TARGETS."""
    jobs = [rid for rid in FIX_TARGETS if not (OUT_DIR / f"{rid}__blindpair2d.npz").exists()]
    if jobs:
        print(f"running {len(jobs)} pair-2d jobs on {len(jobs)} workers", flush=True)
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=max(1, len(jobs)), mp_context=ctx) as pool:
            futs = [pool.submit(run_pair2d, rid) for rid in jobs]
            for f in futs:
                f.result()

    conds = ["blind", "blindpair", "blindpair2d"]
    rows = [load_run(rid, cond) for rid in FIX_TARGETS for cond in conds]
    print_table(rows)
    pooled: dict[str, Any] = {}
    for cond in conds:
        sub = [r for r in rows if r["condition"] == cond]
        pooled[cond] = {
            "refined": pooled_over_recordings(sub, "refined"),
            "n_twins_resolved": sum(r["refined"]["twins_resolved"] for r in sub),
        }
        p = pooled[cond]["refined"]
        print(
            f"POOLED(2rec) {cond:<12} refined {p['err']:.3f} err_sm {p['err_sm']:.3f} "
            f"(bias {p['bias']:+.3f}) | telemetry-refine 0.604 / err_sm ~0.350",
            flush=True,
        )
    stage_report: dict[str, Any] = {}
    spectra: dict[str, Any] = {}
    for rid in FIX_TARGETS:
        with np.load(OUT_DIR / f"{rid}__blindpair2d.npz") as z:
            labels = [str(v) for v in z["stage_labels"]]
            errs = [round(float(v), 3) for v in z["stage_errs"]]
            errs_sm = [round(float(v), 3) for v in z["stage_errs_sm"]]
            seps = z["pair_seps"].tolist()
            meas, edge = z["measured"], z["edge"].astype(bool)
        order = np.argsort(meas[:, edge].mean(axis=1))
        true_seps = [
            float(np.mean(meas[order[1], edge] - meas[order[0], edge])),
            float(np.mean(meas[order[3], edge] - meas[order[2], edge])),
        ]
        stage_report[rid] = {
            "stages": dict(zip(labels, zip(errs, errs_sm))),
            "pair_seps_per_round": seps,
            "true_pair_seps": [round(v, 3) for v in true_seps],
        }
        print(
            f"stages {rid}: "
            + ", ".join(f"{lb}={e}/{es}" for lb, e, es in zip(labels, errs, errs_sm))
        )
        print(f"  pair seps per round {seps} vs true {np.round(true_seps, 2)}")
        spectra[rid] = wander_error_spectrum(rid, cond="blindpair2d")
        print(
            f"  wander-error spectrum: rms {spectra[rid]['rms']:.3f}, peak "
            f"{spectra[rid]['peak_hz']:.2f} Hz, var frac <=0.5/1/2 Hz = "
            f"{spectra[rid]['var_frac_below_0.5hz']:.2f}/"
            f"{spectra[rid]['var_frac_below_1hz']:.2f}/"
            f"{spectra[rid]['var_frac_below_2hz']:.2f}",
            flush=True,
        )
        make_blind_plot(rid, f"pair2d_{rid}.png", cond="blindpair2d")
        make_pairscan_diag(rid, cond="blindpair2d", out_prefix="pair2d_diag")
    strip = lambda s: {k: v for k, v in s.items() if not k.startswith("_")}  # noqa: E731
    summary = {
        "pair2d": {
            "rounds": PAIR2D_ROUNDS,
            "sep_grid": [float(PAIR2D_SEPS[0]), float(PAIR2D_SEPS[-1]), 0.1],
            "sep_fine": PAIR2D_SEP_FINE,
            "win_s": PAIRSCAN_WIN_S,
            "hop_s": PAIRSCAN_HOP_S,
            "lambda": PAIRSCAN_LAMBDA,
        },
        "midband_config": asdict(MIDBAND_CFG),
        "stage_errs": stage_report,
        "wander_error_spectrum": spectra,
        "rows": [
            {
                **{k: v for k, v in r.items() if k not in ("init", "captured", "refined")},
                "init": strip(r["init"]),
                "captured": strip(r["captured"]),
                "refined": strip(r["refined"]),
            }
            for r in rows
        ],
        "pooled": pooled,
    }
    with open(OUT_DIR / "pair2d_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"pair-2d artifacts: {OUT_DIR}/pair2d_summary.json + pair2d_*.png")


# v4 (coordinator): v3's separation grid bounds were wrong (true pair
# separations 0.28/0.88 rev/s; the scan pinned at its 0.6 lower edge) and the
# iterate/re-center loop oscillates — scan round 0 was the best state ever
# reached. v4 replaces iteration with one globally-optimal pass: (1) Viterbi
# ridge tracking over the (window x delta_c) pair score lattice (node score =
# per-window median-normalized comb score, transition cost gamma*|d_delta|),
# NO iteration; (2) separation re-estimated AFTER c(t) is fixed, constant-s
# scan on k >= 12 teeth only (low k carries no split information and biases
# toward s=0); (3) mid-band + refine, STOPPING at the best-scoring stage.
VIT_GAMMA_MULTS = (2.0, 5.0, 10.0)  # x median window contrast; picked on nosource
VIT_SEP_GRID = np.arange(0.05, 1.5001, 0.02)
VIT_SEP_KMIN = 12

# v5 = the closing run (coordinator's direct diagnosis, my_surface_diag_wide):
# ROOT CAUSE of every plateau since v1 — the TRUE pair-mean wander is +-3..5
# rev/s (p2p ~8 on nosource) while every scan grid was +-2: the truth left the
# lattice and each path clipped at the edge. Fixes: delta_c grid +-6.0, gamma
# = 0.3 x surface contrast (2.0 over-smooths), and TWO mid-band phase stages
# (bw 6 then 4, k 6..10) before the final refine — the Viterbi path is locally
# within +-0.5..1 so the first phase stage starts wider.
VIT_DELTA = 6.0
VIT_GAMMA_MULT = 0.3
MIDBAND_CFGS = (
    replace(MIDBAND_CFG, bw_hz=6.0, n_outer=4),
    replace(MIDBAND_CFG, bw_hz=4.0, n_outer=4),
)


def _pair_surface(
    lm: np.ndarray,
    bin_hz: float,
    st: np.ndarray,
    ft: np.ndarray,
    r_pair: np.ndarray,
    deltas: np.ndarray,
    ks: np.ndarray,
    win_s: float = PAIRSCAN_WIN_S,
    hop_s: float = PAIRSCAN_HOP_S,
) -> tuple[np.ndarray, np.ndarray]:
    """``(window centers (W,), scores (W, D))`` of the pair template."""
    r_specs = np.stack([np.interp(st, ft, r_pair[j]) for j in range(r_pair.shape[0])])
    fsc = _local_comb_frame_scores(lm, bin_hz, r_specs, deltas, ks)  # (D, N)
    centers: list[float] = []
    rows: list[np.ndarray] = []
    t0 = 0.0
    while t0 + win_s <= float(st[-1]) + 1e-9:
        sel = (st >= t0) & (st < t0 + win_s)
        if int(sel.sum()) >= 4:
            centers.append(t0 + win_s / 2.0)
            rows.append(np.nanmean(fsc[:, sel], axis=1))
        t0 += hop_s
    return np.asarray(centers), np.stack(rows)


def _viterbi_ridge(surface: np.ndarray, deltas: np.ndarray, gamma: float) -> np.ndarray:
    """Max-sum DP over the (window, delta) lattice; returns the delta path."""
    s_norm = surface - np.median(surface, axis=1, keepdims=True)
    n_win, _ = s_norm.shape
    trans = gamma * np.abs(deltas[None, :] - deltas[:, None])  # (D_prev, D_cur)
    cost = s_norm[0].copy()
    ptr = np.zeros((n_win, len(deltas)), dtype=int)
    for w in range(1, n_win):
        m = cost[:, None] - trans
        ptr[w] = np.argmax(m, axis=0)
        cost = s_norm[w] + np.max(m, axis=0)
    path = np.empty(n_win, dtype=int)
    path[-1] = int(np.argmax(cost))
    for w in range(n_win - 1, 0, -1):
        path[w - 1] = ptr[w][path[w]]
    return deltas[path]


def _surface_contrast(surface: np.ndarray) -> float:
    """Median over windows of (max - median) node score — the gamma scale."""
    return float(np.median(np.max(surface, axis=1) - np.median(surface, axis=1)))


def _sep_scan(
    lm: np.ndarray, bin_hz: float, st: np.ndarray, ft: np.ndarray, c_traj: np.ndarray
) -> float:
    """Constant pair separation by scanning k>=VIT_SEP_KMIN teeth along c(t)."""
    ks_sep = np.arange(VIT_SEP_KMIN, 31)
    c_spec = np.interp(st, ft, c_traj)
    best_s, best_v = float(VIT_SEP_GRID[0]), -np.inf
    for s in VIT_SEP_GRID:
        r2 = np.stack([c_spec - s / 2.0, c_spec + s / 2.0])
        fsc = _local_comb_frame_scores(lm, bin_hz, r2, np.array([0.0]), ks_sep)
        v = float(np.nanmean(fsc))
        if v > best_v:
            best_s, best_v = float(s), v
    return best_s


def _vit_stage1(
    prep: Prepared,
    r0: np.ndarray,
    pairs: list[tuple[int, int]],
    lm: np.ndarray,
    bin_hz: float,
    st: np.ndarray,
    gamma_mult: float,
    diag: dict[str, Any] | None = None,
    win_s: float = PAIRSCAN_WIN_S,
    hop_s: float = PAIRSCAN_HOP_S,
    diag_prefix: str = "rd0",
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Viterbi pair-mean trajectories; returns updated tracks + per-pair c(t)."""
    deltas = np.arange(-VIT_DELTA, VIT_DELTA + SCANLOOP_DSTEP / 2, SCANLOOP_DSTEP)
    ks = np.arange(1, 31)
    r_new = r0.copy()
    c_trajs = []
    for pi, pair in enumerate(pairs):
        r_pair = np.stack([r0[i] for i in pair])
        centers, surface = _pair_surface(
            lm, bin_hz, st, prep.ft, r_pair, deltas, ks, win_s=win_s, hop_s=hop_s
        )
        gamma = gamma_mult * _surface_contrast(surface)
        ridge = _viterbi_ridge(surface, deltas, gamma)
        dc_ft = np.interp(prep.ft, centers, ridge)
        for i in pair:
            r_new[i] = np.maximum(r0[i] + dc_ft, 0.0)
        c_trajs.append(r_new[list(pair)].mean(axis=0))
        if diag is not None:
            diag[f"{diag_prefix}_centers"] = centers
            diag[f"{diag_prefix}_dvals_p{pi}"] = ridge
            diag[f"{diag_prefix}_weights_p{pi}"] = np.max(surface, axis=1) - np.median(
                surface, axis=1
            )
            diag[f"{diag_prefix}_r_before_p{pi}"] = r_pair.copy()
            diag[f"{diag_prefix}_surface_p{pi}"] = surface
            diag[f"{diag_prefix}_deltas"] = deltas
    return r_new, c_trajs


# v6 (coordinator): the magnitude template cannot resolve tight twins along an
# imperfect c(t) (v5 estimated s=0.95 vs true 0.28/0.88 and the wrong split
# poisoned the phase stages). BEAT-SPECTRUM separation instead: demodulate the
# audio at k*c(t) with a wide lowpass — the twin lines sit at +-k*s/2 around
# DC, so |z_k|^2 beats at exactly k*s Hz; fuse the per-k power spectra on the
# s = f/k axis (weighted by envelope power). Resolution ~1/(k*T), not STFT
# bins. Then a SECOND Viterbi pass with the corrected template (finer windows
# 0.5 s / 0.125 s — halves the slow-residual attenuation, the DP bridges the
# extra noise), then mid-band x2 + refine.
BEAT_KS = np.arange(12, 31)
BEAT_S_GRID = np.arange(0.10, 1.5001, 0.002)
VIT2_WIN_S = 0.5
VIT2_HOP_S = 0.125


def _beat_separation(
    audio: np.ndarray, c_traj: np.ndarray, ft: np.ndarray
) -> tuple[float, np.ndarray]:
    """Pair separation from the beat spectrum of wide-band envelopes at k*c(t).

    Returns ``(s_est, fused spectrum on BEAT_S_GRID)``. Each harmonic k is
    demodulated at the pair-mean phase with a ~45 Hz lowpass (fs_env=100,
    cutoff 0.45*fs_env — contains beats k*s <= 45 Hz); the mean-subtracted
    envelope power |z_k|^2 is Hann-windowed, its power spectrum mapped to
    s = f/k, unit-max normalized, and fused with envelope-power weights.
    """
    from data_processing.vk_tracking import demodulate

    n_t = audio.shape[-1]
    t_aud = np.arange(n_t, dtype=np.float64) / float(SR)
    c_aud = np.interp(t_aud, ft, c_traj)
    phase = 2.0 * np.pi * np.cumsum(c_aud) / float(SR)
    cfg = VKConfig(fs=float(SR), fs_env=100.0)
    z = demodulate(audio, BEAT_KS[:, None] * phase[None, :], cfg)  # (C, K, T_env)
    fs_env = float(SR) / max(1, int(round(SR / 100.0)))
    fused = np.zeros_like(BEAT_S_GRID)
    wsum = 0.0
    for ki, k in enumerate(BEAT_KS):
        e = (np.abs(z[:, ki]) ** 2).mean(axis=0)
        weight = float(e.mean())
        e = e - e.mean()
        pxx = np.abs(np.fft.rfft(e * np.hanning(len(e)))) ** 2
        f = np.fft.rfftfreq(len(e), 1.0 / fs_env)
        val = np.interp(BEAT_S_GRID, f / float(k), pxx, left=0.0, right=0.0)
        peak = float(val.max())
        if peak > 0:
            fused += weight * (val / peak)
            wsum += weight
    fused /= max(wsum, 1e-30)
    return float(BEAT_S_GRID[int(np.argmax(fused))]), fused


# v7 = the capstone (coordinator): the twin-crossing finding falsifies pair
# rigidity but prescribes the correct final model — a JOINT 2-rotor Viterbi
# per pair over the (delta1, delta2) lattice, centered on the stage-1 Viterbi
# pair-mean c(t) (so per-rotor deviations are small and the +-6 grid has wide
# margin). Union-of-teeth score counts shared bins once (merged states are not
# artificially favored); DP transitions are beam-limited to +-0.3 rev/s per
# rotor per hop and allow crossing through the diagonal.
VIT2D_DELTA = 6.0
VIT2D_STEP = 0.1
VIT2D_BEAM = 3  # grid steps (= 0.3 rev/s) per rotor per hop


def _tooth_cube(
    lm: np.ndarray,
    bin_hz: float,
    st: np.ndarray,
    ft: np.ndarray,
    c_traj: np.ndarray,
    deltas: np.ndarray,
    ks: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """``(window centers (W,), tooth values (W, K, D))`` along ``c(t)+delta``."""
    n_f, n = lm.shape
    fmax = min(6000.0, (n_f - 1) * bin_hz)
    cols = np.arange(n)[None, :]
    c_spec = np.interp(st, ft, c_traj)
    vals = np.empty((len(ks), len(deltas), n))
    for di, d in enumerate(deltas):
        f = ks[:, None] * (c_spec + d)[None, :]  # (K, N)
        valid = (f >= 60.0) & (f <= fmax)
        idx = np.clip(f, 0.0, fmax) / bin_hz
        j = np.floor(idx).astype(int)
        frac = idx - j
        v = (1 - frac) * lm[j, cols] + frac * lm[np.minimum(j + 1, n_f - 1), cols]
        vals[:, di, :] = np.where(valid, v, np.nan)
    centers: list[float] = []
    cube: list[np.ndarray] = []
    t0 = 0.0
    while t0 + PAIRSCAN_WIN_S <= float(st[-1]) + 1e-9:
        sel = (st >= t0) & (st < t0 + PAIRSCAN_WIN_S)
        if int(sel.sum()) >= 4:
            centers.append(t0 + PAIRSCAN_WIN_S / 2.0)
            cube.append(np.nanmean(vals[:, :, sel], axis=2))  # (K, D)
        t0 += PAIRSCAN_HOP_S
    return np.asarray(centers), np.stack(cube)


def _pair_score_2d(cube_w: np.ndarray, ks: np.ndarray, bin_hz: float) -> np.ndarray:
    """``(D, D)`` union-of-teeth score from per-harmonic tooth values ``(K, D)``.

    For harmonic k, the two teeth ``k*(c+d_i)`` and ``k*(c+d_j)`` merge into
    one spectral bin when ``k*|d_i - d_j| < bin_hz`` — merged teeth contribute
    once, distinct teeth twice. SUM form, not mean: whitened log-mag is ~0 on
    background, so the union score is a matched-filter energy sum — a mean
    normalization makes the diagonal (both rotors on the one strongest comb)
    beat any (strong + weaker) split, which is exactly the twin collapse this
    stage exists to avoid (verified: mean form collapses to the diagonal 100%
    of windows, sum form 0%).
    """
    n_k, n_d = cube_w.shape
    gap = np.abs(np.arange(n_d)[None, :] - np.arange(n_d)[:, None]).astype(np.float64)
    score = np.zeros((n_d, n_d))
    for ki in range(n_k):
        v = np.nan_to_num(cube_w[ki], nan=0.0)
        a = v[:, None] + v[None, :]
        merged = gap * VIT2D_STEP * float(ks[ki]) < bin_hz
        score += np.where(merged, 0.5 * a, a)
    return score


def _joint_viterbi(s2: np.ndarray, gamma: float) -> tuple[np.ndarray, np.ndarray]:
    """Max-sum DP over the (window, delta1, delta2) lattice; beam +-VIT2D_BEAM.

    Returns the two delta paths ``(W,), (W,)`` (unordered rotor identities —
    crossing through the diagonal is allowed and costs only the step size).
    """
    n_w, n_d, _ = s2.shape
    s_norm = s2 - np.median(s2.reshape(n_w, -1), axis=1)[:, None, None]
    offs = [
        (a, b)
        for a in range(-VIT2D_BEAM, VIT2D_BEAM + 1)
        for b in range(-VIT2D_BEAM, VIT2D_BEAM + 1)
    ]
    cost = s_norm[0].copy()
    ptrs = np.zeros((n_w, n_d, n_d), dtype=np.uint8)
    neg = -1e18
    for w in range(1, n_w):
        best = np.full((n_d, n_d), neg)
        bidx = np.zeros((n_d, n_d), dtype=np.uint8)
        for oi, (a, b) in enumerate(offs):
            shifted = np.full((n_d, n_d), neg)
            src_i = slice(max(0, -a), n_d - max(0, a))
            dst_i = slice(max(0, a), n_d - max(0, -a))
            src_j = slice(max(0, -b), n_d - max(0, b))
            dst_j = slice(max(0, b), n_d - max(0, -b))
            shifted[dst_i, dst_j] = cost[src_i, src_j] - gamma * VIT2D_STEP * (abs(a) + abs(b))
            upd = shifted > best
            best[upd] = shifted[upd]
            bidx[upd] = oi
        ptrs[w] = bidx
        cost = s_norm[w] + best
    flat = int(np.argmax(cost))
    i, j = flat // n_d, flat % n_d
    path = np.empty((n_w, 2), dtype=int)
    path[-1] = (i, j)
    for w in range(n_w - 1, 0, -1):
        a, b = offs[int(ptrs[w][i, j])]
        i, j = i - a, j - b
        path[w - 1] = (i, j)
    return path[:, 0].astype(float), path[:, 1].astype(float)


def run_vit2d(rid: str) -> str:
    """Worker: blind init -> Viterbi c(t) -> joint 2-rotor Viterbi -> midband
    (bw 6) -> refine; stop-at-best reporting."""
    cond = "blindvit2d"
    path = OUT_DIR / f"{rid}__{cond}.npz"
    if path.exists():
        return str(path)
    prep = prepare_recording(rid)
    scan = get_scan(rid)
    r0 = build_init(prep, "blind", scan)
    lm, bin_hz, st = _whitened_spec(prep)
    ks = np.arange(1, 31)
    deltas = np.arange(-VIT2D_DELTA, VIT2D_DELTA + VIT2D_STEP / 2, VIT2D_STEP)

    r_cur = r0.copy()
    order = np.argsort(r_cur.mean(axis=1))
    pairs = [(int(order[0]), int(order[1])), (int(order[2]), int(order[3]))]
    stages: list[tuple[str, np.ndarray]] = [("init", r_cur.copy())]

    tic = time.perf_counter()
    r_cur, c_trajs = _vit_stage1(prep, r_cur, pairs, lm, bin_hz, st, VIT_GAMMA_MULT)
    stages.append(("viterbi_c", r_cur.copy()))
    extras: dict[str, Any] = {"vit2d_deltas": deltas}
    for pi, pair in enumerate(pairs):
        centers, cube = _tooth_cube(lm, bin_hz, st, prep.ft, c_trajs[pi], deltas, ks)
        s2 = np.stack([_pair_score_2d(cube[w], ks, bin_hz) for w in range(cube.shape[0])])
        contrast = float(
            np.median(np.max(s2.reshape(s2.shape[0], -1), axis=1))
            - np.median(np.median(s2.reshape(s2.shape[0], -1), axis=1))
        )
        gamma = VIT_GAMMA_MULT * contrast
        d1_idx, d2_idx = _joint_viterbi(s2, gamma)
        d1 = np.interp(prep.ft, centers, deltas[d1_idx.astype(int)])
        d2 = np.interp(prep.ft, centers, deltas[d2_idx.astype(int)])
        r_cur[pair[0]] = np.maximum(c_trajs[pi] + d1, 0.0)
        r_cur[pair[1]] = np.maximum(c_trajs[pi] + d2, 0.0)
        extras[f"vit2d_centers_p{pi}"] = centers
        extras[f"vit2d_s2_p{pi}"] = s2.astype(np.float32)
        extras[f"vit2d_path_p{pi}"] = np.stack(
            [deltas[d1_idx.astype(int)], deltas[d2_idx.astype(int)]]
        )
        extras[f"vit2d_c_p{pi}"] = c_trajs[pi]
    stages.append(("vit2d", r_cur.copy()))
    wall_scan = time.perf_counter() - tic

    tic = time.perf_counter()
    mid = vk_track(prep.audio, r_cur, prep.ft, MIDBAND_CFGS[0])  # bw 6, k 6..10
    stages.append(("midband_bw6", mid.r_refined.copy()))
    ref = vk_track(prep.audio, mid.r_refined, prep.ft, REFINE_CFG)
    stages.append(("refine", ref.r_refined.copy()))
    wall_vk = time.perf_counter() - tic

    def pit_errs(traj: np.ndarray) -> tuple[float, float]:
        p = pit_perm(traj, prep.r_meas, prep.edge)
        a = traj[list(p)]
        return (
            float(np.mean(np.abs((a - prep.r_meas)[:, prep.edge]))),
            float(np.mean(np.abs((a - prep.r_meas_sm)[:, prep.edge]))),
        )

    labels = [s[0] for s in stages]
    errs = [pit_errs(s[1]) for s in stages]
    best_idx = int(np.argmin([e[1] for e in errs]))
    print(
        f"[{rid} | {cond}] scan {wall_scan:.0f}s vk {wall_vk:.0f}s  "
        + ", ".join(f"{lb}={e[0]:.3f}/{e[1]:.3f}" for lb, e in zip(labels, errs))
        + f"  BEST={labels[best_idx]}",
        flush=True,
    )
    arrays: dict[str, Any] = {
        "ft": prep.ft,
        "edge": prep.edge,
        "init": r0,
        "command": prep.r_init,
        "measured": prep.r_meas,
        "measured_sm": prep.r_meas_sm,
        "captured": stages[2][1],  # after the joint 2-D Viterbi stage
        "refined": stages[best_idx][1],  # stop-at-best
        "confidence": ref.confidence,
        "cap_residual_ratios": np.zeros(0),
        "cap_max_deltas": np.zeros(0),
        "ref_residual_ratios": np.array(ref.residual_ratios),
        "ref_max_deltas": np.array(ref.max_deltas),
        "tau": prep.tau,
        "seg_bounds": np.array([prep.seg_lo, prep.seg_hi]),
        "wall_capture_s": wall_scan,
        "wall_refine_s": wall_vk,
        "base": float(scan["base"]),
        "init4": np.asarray(scan["init4"], dtype=np.float64),
        "stage_snaps": np.stack([s[1] for s in stages]),
        "stage_labels": np.array(labels),
        "stage_errs": np.array([e[0] for e in errs]),
        "stage_errs_sm": np.array([e[1] for e in errs]),
        "best_stage": np.array(labels[best_idx]),
        "pairs": np.array(pairs),
        **extras,
    }
    np.savez(path, **arrays)
    return str(path)


def make_vit2d_heatmaps(rid: str) -> None:
    """(delta1, delta2) score heatmaps: one separated window, one crossing."""
    with np.load(OUT_DIR / f"{rid}__blindvit2d.npz") as z:
        arrs = {k: z[k] for k in z.files}
    ft, edge = arrs["ft"], arrs["edge"].astype(bool)
    meas, init, pairs = arrs["measured"], arrs["init"], arrs["pairs"]
    deltas = arrs["vit2d_deltas"]
    p = pit_perm(init, meas, edge)
    inv = np.empty(4, dtype=int)
    for truth_row, track_row in enumerate(list(p)):
        inv[track_row] = truth_row
    fig, ax = plt.subplots(2, 2, figsize=(12, 11))
    for pi in range(2):
        pair = [int(v) for v in pairs[pi]]
        centers = arrs[f"vit2d_centers_p{pi}"]
        s2 = arrs[f"vit2d_s2_p{pi}"]
        dp = arrs[f"vit2d_path_p{pi}"]  # (2, W)
        c_traj = arrs[f"vit2d_c_p{pi}"]
        t1 = np.interp(centers, ft, meas[inv[pair[0]]] - c_traj)
        t2 = np.interp(centers, ft, meas[inv[pair[1]]] - c_traj)
        gap = np.abs(t1 - t2)
        for ci, (wi, tag) in enumerate(
            ((int(np.argmax(gap)), "separated"), (int(np.argmin(gap)), "crossing"))
        ):
            a = ax[pi][ci]
            sn = s2[wi] - np.median(s2[wi])
            a.pcolormesh(deltas, deltas, sn.T, shading="auto", cmap="magma")
            a.plot([t1[wi]], [t2[wi]], "w*", ms=14, label="truth")
            a.plot([t2[wi]], [t1[wi]], "w*", ms=8, alpha=0.5)
            a.plot(
                [dp[0, wi]],
                [dp[1, wi]],
                marker="o",
                color="cyan",
                ms=9,
                mfc="none",
                mew=2,
                ls="none",
                label="DP state",
            )
            a.axline((0, 0), slope=1, color="gray", lw=0.5)
            a.set_xlabel("delta_1 (rev/s)")
            a.set_ylabel("delta_2 (rev/s)")
            a.set_title(f"pair {pi}, {tag} window (t={centers[wi]:.1f}s, |gap|={gap[wi]:.2f})")
            a.legend(fontsize=8, loc="upper left")
    fig.suptitle(f"{rid}: joint (delta1, delta2) pair score surfaces")
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"vit2d_heatmap_{rid}.png", dpi=150)
    plt.close(fig)


def vit2d_main() -> None:
    """Capstone: joint 2-rotor Viterbi per pair on FIX_TARGETS."""
    jobs = [rid for rid in FIX_TARGETS if not (OUT_DIR / f"{rid}__blindvit2d.npz").exists()]
    if jobs:
        print(f"running {len(jobs)} vit2d jobs on {len(jobs)} workers", flush=True)
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=max(1, len(jobs)), mp_context=ctx) as pool:
            futs = [pool.submit(run_vit2d, rid) for rid in jobs]
            for f in futs:
                f.result()

    conds = ["blind", "blindpair", "blindvit2d"]
    rows = [load_run(rid, cond) for rid in FIX_TARGETS for cond in conds]
    print_table(rows)
    pooled: dict[str, Any] = {}
    for cond in conds:
        sub = [r for r in rows if r["condition"] == cond]
        pooled[cond] = {
            "refined": pooled_over_recordings(sub, "refined"),
            "n_twins_resolved": sum(r["refined"]["twins_resolved"] for r in sub),
        }
        p = pooled[cond]["refined"]
        print(
            f"POOLED(2rec) {cond:<12} refined {p['err']:.3f} err_sm {p['err_sm']:.3f} "
            f"(bias {p['bias']:+.3f}) | telemetry-refine 0.604 / err_sm ~0.350",
            flush=True,
        )
    stage_report: dict[str, Any] = {}
    for rid in FIX_TARGETS:
        with np.load(OUT_DIR / f"{rid}__blindvit2d.npz") as z:
            labels = [str(v) for v in z["stage_labels"]]
            errs = [round(float(v), 3) for v in z["stage_errs"]]
            errs_sm = [round(float(v), 3) for v in z["stage_errs_sm"]]
            best_stage = str(z["best_stage"])
        stage_report[rid] = {
            "stages": {lb: [e, es] for lb, e, es in zip(labels, errs, errs_sm)},
            "best_stage": best_stage,
        }
        print(
            f"stages {rid} (best={best_stage}): "
            + ", ".join(f"{lb}={e}/{es}" for lb, e, es in zip(labels, errs, errs_sm))
        )
        make_blind_plot(rid, f"vit2d_{rid}.png", cond="blindvit2d")
        make_vit2d_heatmaps(rid)
    summary = {
        "vit2d": {
            "delta": VIT2D_DELTA,
            "step": VIT2D_STEP,
            "beam": VIT2D_BEAM,
            "gamma_mult": VIT_GAMMA_MULT,
            "center": "stage-1 Viterbi pair-mean c(t)",
            "win_s": PAIRSCAN_WIN_S,
            "hop_s": PAIRSCAN_HOP_S,
        },
        "stage_errs": stage_report,
        "pooled": pooled,
    }
    with open(OUT_DIR / "vit2d_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"vit2d artifacts: {OUT_DIR}/vit2d_summary.json + vit2d_*.png")


# v8 = spatial contrast (coordinator's final axis): every scan so far averaged
# the 8 mics, discarding per-rotor identity. The DREGON array is on the drone;
# per-mic rotor distances differ by up to 1.47x (~3.4 dB in 1/r^2 power), and
# the speed-twin pairs are DIAGONAL rotor pairs (fast = rotors 0&2, slow =
# 1&3), which the corner mics separate best. Rotor-specific score surfaces
# S_r = sum_m w_mr S_m with w_mr ∝ 1/d(mic_m, rotor_r)^2; the joint 2-rotor DP
# then scores S_a(delta1) + S_b(delta2) — swapping delta1<->delta2 changes the
# score, which is what can break twin symmetry. Track->physical-rotor
# assignment uses the PIT identity (experiment-level GT use, documented; the
# blind equivalent is a cheap 2x2 assignment scan).


def _whitened_spec_multi(prep: Prepared) -> tuple[np.ndarray, float, np.ndarray]:
    """Per-channel whitened log-mag ``(C, F, N)`` + ``(bin_hz, frame_times)``."""
    from scipy.ndimage import median_filter

    from data_processing.rps_refinement import RefineConfig, compute_logmag

    cfg = RefineConfig(sample_rate=SR, device="cpu")
    spec = compute_logmag(prep.audio, cfg)
    lm = spec.logmag.cpu().numpy()  # (C, F, N)
    bin_hz = float(spec.bin_hz)
    win = int(round(WHITEN_HZ / bin_hz)) | 1
    white = lm - median_filter(lm, size=(1, win, 1))
    st = np.asarray(spec.frame_times, dtype=np.float64)
    return white, bin_hz, st


def _rotor_mic_weights(prep_dir: str = "data/DREGON") -> np.ndarray:
    """``(8 mics, 4 rotors)`` weights ∝ 1/d^2, normalized per rotor."""
    from data_processing.dregon import get_geometry

    mic, rot = get_geometry(Path(prep_dir))
    d = np.linalg.norm(mic[:, None, :] - rot[None, :, :], axis=2)
    w = 1.0 / d**2
    return w / w.sum(axis=0, keepdims=True)


def _pair_score_2d_spatial(
    cube_a: np.ndarray, cube_b: np.ndarray, ks: np.ndarray, bin_hz: float
) -> np.ndarray:
    """``(D, D)`` union score with rotor-SPECIFIC tooth values (sum form).

    ``score(d1, d2) = sum_k v_a(k, d1) + v_b(k, d2)``, with merged teeth
    (``k*|d1-d2| < bin_hz``) counted once as the mean of the two rotors'
    values. Asymmetric in (d1, d2) — the spatial weighting is what
    distinguishes the two rotor identities.
    """
    n_k, n_d = cube_a.shape
    gap = np.abs(np.arange(n_d)[None, :] - np.arange(n_d)[:, None]).astype(np.float64)
    score = np.zeros((n_d, n_d))
    for ki in range(n_k):
        va = np.nan_to_num(cube_a[ki], nan=0.0)
        vb = np.nan_to_num(cube_b[ki], nan=0.0)
        a = va[:, None] + vb[None, :]
        merged = gap * VIT2D_STEP * float(ks[ki]) < bin_hz
        score += np.where(merged, 0.5 * a, a)
    return score


def run_vit2dsp(rid: str) -> str:
    """Worker: blind init -> Viterbi c(t) -> SPATIAL joint 2-rotor Viterbi ->
    midband (bw 6) -> refine; stop-at-best reporting."""
    cond = "blindvit2dsp"
    path = OUT_DIR / f"{rid}__{cond}.npz"
    if path.exists():
        return str(path)
    prep = prepare_recording(rid)
    scan = get_scan(rid)
    r0 = build_init(prep, "blind", scan)
    lm_avg, bin_hz, st = _whitened_spec(prep)
    lm_multi, _, _ = _whitened_spec_multi(prep)
    weights = _rotor_mic_weights()  # (8, 4)
    ks = np.arange(1, 31)
    deltas = np.arange(-VIT2D_DELTA, VIT2D_DELTA + VIT2D_STEP / 2, VIT2D_STEP)

    r_cur = r0.copy()
    order = np.argsort(r_cur.mean(axis=1))
    pairs = [(int(order[0]), int(order[1])), (int(order[2]), int(order[3]))]
    # Track -> physical rotor assignment (PIT vs measured; experiment-level).
    p = pit_perm(r0, prep.r_meas, prep.edge)
    inv = np.empty(4, dtype=int)
    for truth_row, track_row in enumerate(list(p)):
        inv[track_row] = truth_row
    print(f"[{rid}] track->physical rotor map: {inv.tolist()}", flush=True)
    print(f"[{rid}] mic weights per rotor (cols):\n{np.round(weights, 3)}", flush=True)

    stages: list[tuple[str, np.ndarray]] = [("init", r_cur.copy())]
    tic = time.perf_counter()
    r_cur, c_trajs = _vit_stage1(prep, r_cur, pairs, lm_avg, bin_hz, st, VIT_GAMMA_MULT)
    stages.append(("viterbi_c", r_cur.copy()))
    extras: dict[str, Any] = {"vit2d_deltas": deltas, "mic_weights": weights, "phys_map": inv}
    for pi, pair in enumerate(pairs):
        rot_a, rot_b = int(inv[pair[0]]), int(inv[pair[1]])
        lm_a = np.tensordot(weights[:, rot_a], lm_multi, axes=(0, 0))  # (F, N)
        lm_b = np.tensordot(weights[:, rot_b], lm_multi, axes=(0, 0))
        centers, cube_a = _tooth_cube(lm_a, bin_hz, st, prep.ft, c_trajs[pi], deltas, ks)
        _, cube_b = _tooth_cube(lm_b, bin_hz, st, prep.ft, c_trajs[pi], deltas, ks)
        s2 = np.stack(
            [
                _pair_score_2d_spatial(cube_a[w], cube_b[w], ks, bin_hz)
                for w in range(cube_a.shape[0])
            ]
        )
        flat = s2.reshape(s2.shape[0], -1)
        contrast = float(np.median(np.max(flat, axis=1)) - np.median(np.median(flat, axis=1)))
        d1_idx, d2_idx = _joint_viterbi(s2, VIT_GAMMA_MULT * contrast)
        d1 = np.interp(prep.ft, centers, deltas[d1_idx.astype(int)])
        d2 = np.interp(prep.ft, centers, deltas[d2_idx.astype(int)])
        r_cur[pair[0]] = np.maximum(c_trajs[pi] + d1, 0.0)
        r_cur[pair[1]] = np.maximum(c_trajs[pi] + d2, 0.0)
        extras[f"vit2d_centers_p{pi}"] = centers
        extras[f"vit2d_s2_p{pi}"] = s2.astype(np.float32)
        extras[f"vit2d_path_p{pi}"] = np.stack(
            [deltas[d1_idx.astype(int)], deltas[d2_idx.astype(int)]]
        )
        extras[f"vit2d_c_p{pi}"] = c_trajs[pi]
        # Per-rotor 1-D surfaces for the branch diagnostic (sum over k).
        extras[f"vit2d_s1a_p{pi}"] = np.nansum(cube_a, axis=1).astype(np.float32)  # (W, D)
        extras[f"vit2d_s1b_p{pi}"] = np.nansum(cube_b, axis=1).astype(np.float32)
        extras[f"vit2d_rots_p{pi}"] = np.array([rot_a, rot_b])
    stages.append(("vit2dsp", r_cur.copy()))
    wall_scan = time.perf_counter() - tic

    tic = time.perf_counter()
    mid = vk_track(prep.audio, r_cur, prep.ft, MIDBAND_CFGS[0])
    stages.append(("midband_bw6", mid.r_refined.copy()))
    ref = vk_track(prep.audio, mid.r_refined, prep.ft, REFINE_CFG)
    stages.append(("refine", ref.r_refined.copy()))
    wall_vk = time.perf_counter() - tic

    def pit_errs(traj: np.ndarray) -> tuple[float, float]:
        pp = pit_perm(traj, prep.r_meas, prep.edge)
        a = traj[list(pp)]
        return (
            float(np.mean(np.abs((a - prep.r_meas)[:, prep.edge]))),
            float(np.mean(np.abs((a - prep.r_meas_sm)[:, prep.edge]))),
        )

    labels = [s[0] for s in stages]
    errs = [pit_errs(s[1]) for s in stages]
    best_idx = int(np.argmin([e[1] for e in errs]))
    print(
        f"[{rid} | {cond}] scan {wall_scan:.0f}s vk {wall_vk:.0f}s  "
        + ", ".join(f"{lb}={e[0]:.3f}/{e[1]:.3f}" for lb, e in zip(labels, errs))
        + f"  BEST={labels[best_idx]}",
        flush=True,
    )
    arrays: dict[str, Any] = {
        "ft": prep.ft,
        "edge": prep.edge,
        "init": r0,
        "command": prep.r_init,
        "measured": prep.r_meas,
        "measured_sm": prep.r_meas_sm,
        "captured": stages[2][1],
        "refined": stages[best_idx][1],
        "confidence": ref.confidence,
        "cap_residual_ratios": np.zeros(0),
        "cap_max_deltas": np.zeros(0),
        "ref_residual_ratios": np.array(ref.residual_ratios),
        "ref_max_deltas": np.array(ref.max_deltas),
        "tau": prep.tau,
        "seg_bounds": np.array([prep.seg_lo, prep.seg_hi]),
        "wall_capture_s": wall_scan,
        "wall_refine_s": wall_vk,
        "base": float(scan["base"]),
        "init4": np.asarray(scan["init4"], dtype=np.float64),
        "stage_snaps": np.stack([s[1] for s in stages]),
        "stage_labels": np.array(labels),
        "stage_errs": np.array([e[0] for e in errs]),
        "stage_errs_sm": np.array([e[1] for e in errs]),
        "best_stage": np.array(labels[best_idx]),
        "pairs": np.array(pairs),
        **extras,
    }
    np.savez(path, **arrays)
    return str(path)


def make_vit2dsp_branch_diag(rid: str) -> None:
    """Branch preference through crossings: does S_a prefer rotor a's branch?

    For each pair, plot ``S_r(w, truth_own) - S_r(w, truth_twin)`` for both
    rotor-specific surfaces; a real spatial signature keeps rotor a's curve
    positive (prefers its own branch) where the twins are separated and
    through crossings.
    """
    with np.load(OUT_DIR / f"{rid}__blindvit2dsp.npz") as z:
        arrs = {k: z[k] for k in z.files}
    ft = arrs["ft"]
    meas, pairs, inv = arrs["measured"], arrs["pairs"], arrs["phys_map"]
    deltas = arrs["vit2d_deltas"]
    fig, ax = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for pi in range(2):
        pair = [int(v) for v in pairs[pi]]
        centers = arrs[f"vit2d_centers_p{pi}"]
        c_traj = arrs[f"vit2d_c_p{pi}"]
        s1a, s1b = arrs[f"vit2d_s1a_p{pi}"], arrs[f"vit2d_s1b_p{pi}"]  # (W, D)
        t_own = np.interp(centers, ft, meas[int(inv[pair[0]])] - c_traj)
        t_twin = np.interp(centers, ft, meas[int(inv[pair[1]])] - c_traj)

        def at(surf: np.ndarray, dvals: np.ndarray) -> np.ndarray:
            idx = np.clip(
                np.round((dvals - deltas[0]) / VIT2D_STEP).astype(int), 0, len(deltas) - 1
            )
            return surf[np.arange(len(dvals)), idx]

        pref_a = at(s1a, t_own) - at(s1a, t_twin)
        pref_b = at(s1b, t_twin) - at(s1b, t_own)
        gap = np.abs(t_own - t_twin)
        a = ax[pi]
        a.plot(centers, pref_a, label=f"S_r{int(inv[pair[0]])} own-minus-twin")
        a.plot(centers, pref_b, label=f"S_r{int(inv[pair[1]])} own-minus-twin")
        a2 = a.twinx()
        a2.plot(centers, gap, "k:", lw=0.8, alpha=0.6)
        a2.set_ylabel("|true gap| (rev/s)")
        a.axhline(0, color="gray", lw=0.6)
        a.set_ylabel("branch preference (score)")
        a.set_title(f"pair {pi} (physical rotors {[int(inv[i]) for i in pair]})")
        a.legend(fontsize=8)
    ax[1].set_xlabel("segment time (s)")
    fig.suptitle(f"{rid}: rotor-specific surface branch preference (spatial contrast test)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"vit2dsp_branch_{rid}.png", dpi=150)
    plt.close(fig)


def vit2dsp_main() -> None:
    """Spatial-contrast finale on FIX_TARGETS."""
    jobs = [rid for rid in FIX_TARGETS if not (OUT_DIR / f"{rid}__blindvit2dsp.npz").exists()]
    if jobs:
        print(f"running {len(jobs)} vit2d-spatial jobs on {len(jobs)} workers", flush=True)
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=max(1, len(jobs)), mp_context=ctx) as pool:
            futs = [pool.submit(run_vit2dsp, rid) for rid in jobs]
            for f in futs:
                f.result()

    conds = ["blindpair", "blindvit2d", "blindvit2dsp"]
    rows = [load_run(rid, cond) for rid in FIX_TARGETS for cond in conds]
    print_table(rows)
    pooled: dict[str, Any] = {}
    for cond in conds:
        sub = [r for r in rows if r["condition"] == cond]
        pooled[cond] = {
            "refined": pooled_over_recordings(sub, "refined"),
            "n_twins_resolved": sum(r["refined"]["twins_resolved"] for r in sub),
        }
        pp = pooled[cond]["refined"]
        print(
            f"POOLED(2rec) {cond:<14} refined {pp['err']:.3f} err_sm {pp['err_sm']:.3f} "
            f"(bias {pp['bias']:+.3f}) | telemetry-refine 0.604 / err_sm ~0.350",
            flush=True,
        )
    stage_report: dict[str, Any] = {}
    for rid in FIX_TARGETS:
        with np.load(OUT_DIR / f"{rid}__blindvit2dsp.npz") as z:
            labels = [str(v) for v in z["stage_labels"]]
            errs = [round(float(v), 3) for v in z["stage_errs"]]
            errs_sm = [round(float(v), 3) for v in z["stage_errs_sm"]]
            best_stage = str(z["best_stage"])
        stage_report[rid] = {
            "stages": {lb: [e, es] for lb, e, es in zip(labels, errs, errs_sm)},
            "best_stage": best_stage,
        }
        print(
            f"stages {rid} (best={best_stage}): "
            + ", ".join(f"{lb}={e}/{es}" for lb, e, es in zip(labels, errs, errs_sm))
        )
        make_blind_plot(rid, f"vit2dsp_{rid}.png", cond="blindvit2dsp")
        make_vit2dsp_branch_diag(rid)
    summary = {
        "vit2dsp": {
            "delta": VIT2D_DELTA,
            "step": VIT2D_STEP,
            "beam": VIT2D_BEAM,
            "gamma_mult": VIT_GAMMA_MULT,
            "weights": "1/d^2 mic-rotor, normalized per rotor",
            "assignment": "track->physical rotor via PIT (experiment-level GT)",
        },
        "stage_errs": stage_report,
        "pooled": pooled,
    }
    with open(OUT_DIR / "vit2dsp_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"vit2dsp artifacts: {OUT_DIR}/vit2dsp_summary.json + vit2dsp_*.png")


# v9 = the last knob (coordinator): the phase stages' trajectory smoother
# (traj_lambda 1e4 ~ 0.3 Hz bandwidth) may low-pass away exactly the 0.5-2 Hz
# per-rotor differential band the mid-band envelopes (bw 6) can otherwise see
# — which would explain the flat midband/refine stages even from good
# spatial-DP tracks. Sweep it; a flat sweep completes the floor proof.
LSWEEP_MID = (1e4, 1e3, 1e2, 10.0)
LSWEEP_REF = (1e4, 1e3)


def run_lambda_combo(rid: str, lam_mid: float, source_cond: str = "blindvit2dsp") -> dict[str, Any]:
    """Worker: midband(bw6, lam_mid) from the ``source_cond`` tracks + refines."""
    prep = prepare_recording(rid)
    with np.load(OUT_DIR / f"{rid}__{source_cond}.npz") as z:
        cap = z["captured"]  # tracks after the spatial joint-DP stage

    def pit_errs(traj: np.ndarray) -> list[float]:
        pp = pit_perm(traj, prep.r_meas, prep.edge)
        a = traj[list(pp)]
        return [
            float(np.mean(np.abs((a - prep.r_meas)[:, prep.edge]))),
            float(np.mean(np.abs((a - prep.r_meas_sm)[:, prep.edge]))),
        ]

    mid = vk_track(prep.audio, cap, prep.ft, replace(MIDBAND_CFGS[0], traj_lambda=lam_mid))
    out: dict[str, Any] = {
        "recording": rid,
        "lambda_mid": lam_mid,
        "start": pit_errs(cap),
        "midband": pit_errs(mid.r_refined),
    }
    for lam_ref in LSWEEP_REF:
        ref = vk_track(prep.audio, mid.r_refined, prep.ft, replace(REFINE_CFG, traj_lambda=lam_ref))
        out[f"refine_lam{lam_ref:g}"] = pit_errs(ref.r_refined)
    print(
        f"[{rid} | lam_mid={lam_mid:g}] start {out['start'][0]:.3f}/{out['start'][1]:.3f} "
        f"-> mid {out['midband'][0]:.3f}/{out['midband'][1]:.3f} -> "
        + ", ".join(
            f"ref(l{lr:g}) {out[f'refine_lam{lr:g}'][0]:.3f}/{out[f'refine_lam{lr:g}'][1]:.3f}"
            for lr in LSWEEP_REF
        ),
        flush=True,
    )
    return out


def lambda_sweep_main(source_cond: str = "blindvit2dsp") -> None:
    """Final exhibit: does relaxing traj_lambda unlock the differential band?"""
    jobs = [(rid, lm) for rid in FIX_TARGETS for lm in LSWEEP_MID]
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=len(jobs), mp_context=ctx) as pool:
        futs = [pool.submit(run_lambda_combo, rid, lm, source_cond) for rid, lm in jobs]
        rows = [f.result() for f in futs]
    with open(OUT_DIR / "lambda_sweep_final.json", "w") as f:
        json.dump(rows, f, indent=2)
    print("\nSWEEP SUMMARY (err/err_sm):")
    for rid in FIX_TARGETS:
        for r in sorted((r for r in rows if r["recording"] == rid), key=lambda r: -r["lambda_mid"]):
            print(
                f"{rid[:34]:<36} lam_mid={r['lambda_mid']:>7g}  "
                f"mid {r['midband'][0]:.3f}/{r['midband'][1]:.3f}  "
                + "  ".join(
                    f"ref(l{lr:g}) {r[f'refine_lam{lr:g}'][0]:.3f}/{r[f'refine_lam{lr:g}'][1]:.3f}"
                    for lr in LSWEEP_REF
                )
            )
    print(f"artifacts: {OUT_DIR}/lambda_sweep_final.json")


# v10 (coordinator, URGENT user input): DREGON's mic positions are JUMBLED
# relative to channel order (found by another agent) — the v8 geometric 1/d^2
# weights were scrambled, so the 0.688 gain came from per-rotor surface
# diversity, not correct spatial assignment; correct assignment may have
# unexploited headroom. Replacement: DATA-DRIVEN AFFINITY, immune to the
# indexing bug and fully blind (needs no physical rotor identity at all) —
# A[m, r] = mean whitened comb score of TRACK r's teeth (k 6..30) in mic m,
# softmax-normalized per track over mics. Impact analysis of the bug on
# earlier runs: all scan/score paths averaged mics (immune); vk_track's
# envelope solve and frequency update never use positions and fuse channels
# with data-driven Fisher weights (immune); ONLY v8's geometric weighting was
# affected.
AFF_KS = np.arange(6, 31)
AFF_ITERS = 2  # (tracks -> affinity -> DP) rounds


def _track_mic_affinity(
    lm_multi: np.ndarray,
    bin_hz: float,
    st: np.ndarray,
    ft: np.ndarray,
    tracks: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """``(A raw (M, R), softmax weights (M, R))`` comb affinity per mic/track."""
    n_c, n_f, n = lm_multi.shape
    fmax = min(6000.0, (n_f - 1) * bin_hz)
    cols = np.arange(n)[None, :]
    a = np.zeros((n_c, tracks.shape[0]))
    for r in range(tracks.shape[0]):
        r_spec = np.interp(st, ft, tracks[r])
        f = AFF_KS[:, None] * r_spec[None, :]
        valid = (f >= 60.0) & (f <= fmax)
        idx = np.clip(f, 0.0, fmax) / bin_hz
        j = np.floor(idx).astype(int)
        frac = idx - j
        for m in range(n_c):
            v = (1 - frac) * lm_multi[m][j, cols] + frac * lm_multi[m][
                np.minimum(j + 1, n_f - 1), cols
            ]
            a[m, r] = float(np.nanmean(np.where(valid, v, np.nan)))
    w = np.zeros_like(a)
    for r in range(tracks.shape[0]):
        col = a[:, r]
        tau = max(float(col.std()), 1e-9)
        e = np.exp((col - col.max()) / tau)
        w[:, r] = e / e.sum()
    return a, w


def run_vit2daff(rid: str) -> str:
    """Worker: measured-affinity spatial joint DP from the vit2dsp state."""
    cond = "blindvit2daff"
    path = OUT_DIR / f"{rid}__{cond}.npz"
    if path.exists():
        return str(path)
    prep = prepare_recording(rid)
    with np.load(OUT_DIR / f"{rid}__blindvit2dsp.npz") as z:
        tracks = z["captured"].copy()  # per-rotor tracks after the v8 DP
        pairs_arr = z["pairs"]
        init = z["init"]
    pairs = [
        (int(pairs_arr[0][0]), int(pairs_arr[0][1])),
        (int(pairs_arr[1][0]), int(pairs_arr[1][1])),
    ]
    lm_multi, bin_hz, st = _whitened_spec_multi(prep)
    ks = np.arange(1, 31)
    deltas = np.arange(-VIT2D_DELTA, VIT2D_DELTA + VIT2D_STEP / 2, VIT2D_STEP)

    stages: list[tuple[str, np.ndarray]] = [("vit2dsp_start", tracks.copy())]
    extras: dict[str, Any] = {"vit2d_deltas": deltas}
    tic = time.perf_counter()
    for it in range(AFF_ITERS):
        a_raw, w = _track_mic_affinity(lm_multi, bin_hz, st, prep.ft, tracks)
        extras[f"affinity_raw_it{it}"] = a_raw
        extras[f"affinity_w_it{it}"] = w
        if it == 0:
            geo = _rotor_mic_weights()
            print(f"[{rid}] measured affinity (raw):\n{np.round(a_raw, 4)}", flush=True)
            print(f"[{rid}] measured weights (softmax):\n{np.round(w, 3)}", flush=True)
            print(f"[{rid}] (jumbled) geometric prediction:\n{np.round(geo, 3)}", flush=True)
        for pair in pairs:
            c_traj = tracks[list(pair)].mean(axis=0)
            lm_a = np.tensordot(w[:, pair[0]], lm_multi, axes=(0, 0))
            lm_b = np.tensordot(w[:, pair[1]], lm_multi, axes=(0, 0))
            centers, cube_a = _tooth_cube(lm_a, bin_hz, st, prep.ft, c_traj, deltas, ks)
            _, cube_b = _tooth_cube(lm_b, bin_hz, st, prep.ft, c_traj, deltas, ks)
            s2 = np.stack(
                [
                    _pair_score_2d_spatial(cube_a[wi], cube_b[wi], ks, bin_hz)
                    for wi in range(cube_a.shape[0])
                ]
            )
            flat = s2.reshape(s2.shape[0], -1)
            contrast = float(np.median(np.max(flat, axis=1)) - np.median(np.median(flat, axis=1)))
            d1_idx, d2_idx = _joint_viterbi(s2, VIT_GAMMA_MULT * contrast)
            d1 = np.interp(prep.ft, centers, deltas[d1_idx.astype(int)])
            d2 = np.interp(prep.ft, centers, deltas[d2_idx.astype(int)])
            tracks[pair[0]] = np.maximum(c_traj + d1, 0.0)
            tracks[pair[1]] = np.maximum(c_traj + d2, 0.0)
        stages.append((f"aff_dp{it}", tracks.copy()))
    wall_scan = time.perf_counter() - tic

    tic = time.perf_counter()
    mid = vk_track(prep.audio, tracks, prep.ft, MIDBAND_CFGS[0])
    stages.append(("midband_bw6", mid.r_refined.copy()))
    ref = vk_track(prep.audio, mid.r_refined, prep.ft, REFINE_CFG)
    stages.append(("refine", ref.r_refined.copy()))
    wall_vk = time.perf_counter() - tic

    def pit_errs(traj: np.ndarray) -> tuple[float, float]:
        pp = pit_perm(traj, prep.r_meas, prep.edge)
        aa = traj[list(pp)]
        return (
            float(np.mean(np.abs((aa - prep.r_meas)[:, prep.edge]))),
            float(np.mean(np.abs((aa - prep.r_meas_sm)[:, prep.edge]))),
        )

    labels = [s[0] for s in stages]
    errs = [pit_errs(s[1]) for s in stages]
    best_idx = int(np.argmin([e[1] for e in errs]))
    print(
        f"[{rid} | {cond}] scan {wall_scan:.0f}s vk {wall_vk:.0f}s  "
        + ", ".join(f"{lb}={e[0]:.3f}/{e[1]:.3f}" for lb, e in zip(labels, errs))
        + f"  BEST={labels[best_idx]}",
        flush=True,
    )
    arrays: dict[str, Any] = {
        "ft": prep.ft,
        "edge": prep.edge,
        "init": init,
        "command": prep.r_init,
        "measured": prep.r_meas,
        "measured_sm": prep.r_meas_sm,
        "captured": stages[AFF_ITERS][1],  # after the last affinity DP
        "refined": stages[best_idx][1],
        "confidence": ref.confidence,
        "cap_residual_ratios": np.zeros(0),
        "cap_max_deltas": np.zeros(0),
        "ref_residual_ratios": np.array(ref.residual_ratios),
        "ref_max_deltas": np.array(ref.max_deltas),
        "tau": prep.tau,
        "seg_bounds": np.array([prep.seg_lo, prep.seg_hi]),
        "wall_capture_s": wall_scan,
        "wall_refine_s": wall_vk,
        "base": 0.0,
        "init4": np.zeros(4),
        "stage_snaps": np.stack([s[1] for s in stages]),
        "stage_labels": np.array(labels),
        "stage_errs": np.array([e[0] for e in errs]),
        "stage_errs_sm": np.array([e[1] for e in errs]),
        "best_stage": np.array(labels[best_idx]),
        "pairs": np.array(pairs),
        **extras,
    }
    np.savez(path, **arrays)
    return str(path)


def vit2daff_main() -> None:
    """Measured-affinity finale + the traj_lambda sweep from its tracks."""
    jobs = [rid for rid in FIX_TARGETS if not (OUT_DIR / f"{rid}__blindvit2daff.npz").exists()]
    if jobs:
        print(f"running {len(jobs)} affinity-DP jobs on {len(jobs)} workers", flush=True)
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=max(1, len(jobs)), mp_context=ctx) as pool:
            futs = [pool.submit(run_vit2daff, rid) for rid in jobs]
            for f in futs:
                f.result()

    conds = ["blindvit2dsp", "blindvit2daff"]
    rows = [load_run(rid, cond) for rid in FIX_TARGETS for cond in conds]
    print_table(rows)
    pooled: dict[str, Any] = {}
    for cond in conds:
        sub = [r for r in rows if r["condition"] == cond]
        pooled[cond] = {"refined": pooled_over_recordings(sub, "refined")}
        pp = pooled[cond]["refined"]
        print(
            f"POOLED(2rec) {cond:<15} refined {pp['err']:.3f} err_sm {pp['err_sm']:.3f} "
            f"(bias {pp['bias']:+.3f}) | telemetry-refine 0.604 / err_sm ~0.350",
            flush=True,
        )
    stage_report: dict[str, Any] = {}
    for rid in FIX_TARGETS:
        with np.load(OUT_DIR / f"{rid}__blindvit2daff.npz") as z:
            labels = [str(v) for v in z["stage_labels"]]
            errs = [round(float(v), 3) for v in z["stage_errs"]]
            errs_sm = [round(float(v), 3) for v in z["stage_errs_sm"]]
            best_stage = str(z["best_stage"])
            aff = z["affinity_w_it0"].tolist()
        stage_report[rid] = {
            "stages": {lb: [e, es] for lb, e, es in zip(labels, errs, errs_sm)},
            "best_stage": best_stage,
            "affinity_weights_it0": aff,
        }
        print(
            f"stages {rid} (best={best_stage}): "
            + ", ".join(f"{lb}={e}/{es}" for lb, e, es in zip(labels, errs, errs_sm))
        )
        make_blind_plot(rid, f"vit2daff_{rid}.png", cond="blindvit2daff")
    with open(OUT_DIR / "vit2daff_summary.json", "w") as f:
        json.dump({"stage_errs": stage_report, "pooled": pooled}, f, indent=2)
    print(f"vit2daff artifacts: {OUT_DIR}/vit2daff_summary.json + vit2daff_*.png")
    # The final lambda sweep, from the affinity-DP tracks.
    lambda_sweep_main(source_cond="blindvit2daff")


def run_vit(rid: str, gamma_mult: float) -> str:
    """Worker: blind init -> Viterbi c(t) -> separation -> mid-band -> refine."""
    cond = "blindvit"
    path = OUT_DIR / f"{rid}__{cond}.npz"
    if path.exists():
        return str(path)
    prep = prepare_recording(rid)
    scan = get_scan(rid)
    r0 = build_init(prep, "blind", scan)
    lm, bin_hz, st = _whitened_spec(prep)

    r_cur = r0.copy()
    order = np.argsort(r_cur.mean(axis=1))
    pairs = [(int(order[0]), int(order[1])), (int(order[2]), int(order[3]))]
    diag: dict[str, Any] = {}
    stages: list[tuple[str, np.ndarray]] = [("init", r_cur.copy())]

    tic = time.perf_counter()
    r_cur, c_trajs = _vit_stage1(prep, r_cur, pairs, lm, bin_hz, st, gamma_mult, diag)
    stages.append(("viterbi", r_cur.copy()))
    seps = []
    for pi, pair in enumerate(pairs):
        s = _sep_scan(lm, bin_hz, st, prep.ft, c_trajs[pi])
        r_cur[pair[0]] = np.maximum(c_trajs[pi] - s / 2.0, 0.0)
        r_cur[pair[1]] = c_trajs[pi] + s / 2.0
        seps.append(s)
    stages.append(("sep", r_cur.copy()))
    wall_scan = time.perf_counter() - tic

    tic = time.perf_counter()
    for mcfg in MIDBAND_CFGS:
        mid = vk_track(prep.audio, r_cur, prep.ft, mcfg)
        r_cur = mid.r_refined.copy()
        stages.append((f"midband_bw{mcfg.bw_hz:g}", r_cur.copy()))
    ref = vk_track(prep.audio, r_cur, prep.ft, REFINE_CFG)
    stages.append(("refine", ref.r_refined.copy()))
    wall_vk = time.perf_counter() - tic

    def pit_errs(traj: np.ndarray) -> tuple[float, float]:
        p = pit_perm(traj, prep.r_meas, prep.edge)
        a = traj[list(p)]
        return (
            float(np.mean(np.abs((a - prep.r_meas)[:, prep.edge]))),
            float(np.mean(np.abs((a - prep.r_meas_sm)[:, prep.edge]))),
        )

    labels = [s[0] for s in stages]
    errs = [pit_errs(s[1]) for s in stages]
    # Best stage by err_sm (the user's primary metric); the coordinator's
    # stop-at-best rule — later phase stages may corrupt a good scan state.
    best_idx = int(np.argmin([e[1] for e in errs]))
    print(
        f"[{rid} | {cond}] scan {wall_scan:.0f}s vk {wall_vk:.0f}s gamma_mult {gamma_mult} "
        f"seps {np.round(seps, 2).tolist()}  "
        + ", ".join(f"{lb}={e[0]:.3f}/{e[1]:.3f}" for lb, e in zip(labels, errs))
        + f"  BEST={labels[best_idx]}",
        flush=True,
    )
    arrays: dict[str, Any] = {
        "ft": prep.ft,
        "edge": prep.edge,
        "init": r0,
        "command": prep.r_init,
        "measured": prep.r_meas,
        "measured_sm": prep.r_meas_sm,
        "captured": stages[1][1],  # after the Viterbi stage
        "refined": stages[-1][1],  # final stage (full pipeline output)
        "confidence": ref.confidence,
        "cap_residual_ratios": np.zeros(0),
        "cap_max_deltas": np.zeros(0),
        "ref_residual_ratios": np.array(ref.residual_ratios),
        "ref_max_deltas": np.array(ref.max_deltas),
        "tau": prep.tau,
        "seg_bounds": np.array([prep.seg_lo, prep.seg_hi]),
        "wall_capture_s": wall_scan,
        "wall_refine_s": wall_vk,
        "base": float(scan["base"]),
        "init4": np.asarray(scan["init4"], dtype=np.float64),
        "stage_snaps": np.stack([s[1] for s in stages]),
        "stage_labels": np.array(labels),
        "stage_errs": np.array([e[0] for e in errs]),
        "stage_errs_sm": np.array([e[1] for e in errs]),
        "best_stage": np.array(labels[best_idx]),
        "gamma_mult": gamma_mult,
        "pair_seps": np.array(seps),
        "pairs": np.array(pairs),
        **{f"diag_{k}": v for k, v in diag.items()},
    }
    np.savez(path, **arrays)
    return str(path)


def make_vit_heatmap(rid: str) -> None:
    """(surface + truth + Viterbi path) heatmap per pair — the mechanism check."""
    with np.load(OUT_DIR / f"{rid}__blindvit.npz") as z:
        arrs = {k: z[k] for k in z.files}
    ft, edge = arrs["ft"], arrs["edge"].astype(bool)
    meas, init, pairs = arrs["measured"], arrs["init"], arrs["pairs"]
    p = pit_perm(init, meas, edge)
    inv = np.empty(4, dtype=int)
    for truth_row, track_row in enumerate(list(p)):
        inv[track_row] = truth_row
    centers = arrs["diag_rd0_centers"]
    deltas = arrs["diag_rd0_deltas"]
    fig, ax = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    for pi in range(2):
        pair = [int(v) for v in pairs[pi]]
        surface = arrs[f"diag_rd0_surface_p{pi}"]  # (W, D)
        path = arrs[f"diag_rd0_dvals_p{pi}"]
        r_before = arrs[f"diag_rd0_r_before_p{pi}"]
        true_dc = np.stack(
            [np.interp(centers, ft, meas[inv[i]] - r_before[j]) for j, i in enumerate(pair)]
        ).mean(axis=0)
        s_norm = surface - np.median(surface, axis=1, keepdims=True)
        ax[pi].pcolormesh(centers, deltas, s_norm.T, shading="auto", cmap="magma")
        ax[pi].plot(centers, true_dc, "w-", lw=1.6, label="true pair-mean wander")
        ax[pi].plot(centers, path, "c--", lw=1.4, label="Viterbi path")
        ax[pi].set_ylabel("delta_c (rev/s)")
        ax[pi].set_title(f"{rid} pair {pi} (rotors {pair}): score surface + truth + path")
        ax[pi].legend(fontsize=8, loc="upper right")
    ax[1].set_xlabel("segment time (s)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"vit_heatmap_{rid}.png", dpi=150)
    plt.close(fig)


def vit_main() -> None:
    """Closing run: wide-grid Viterbi -> separation -> 2x mid-band -> refine."""
    best_mult = VIT_GAMMA_MULT  # coordinator-prescribed (0.1-0.3 both work)
    jobs = [rid for rid in FIX_TARGETS if not (OUT_DIR / f"{rid}__blindvit.npz").exists()]
    if jobs:
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=max(1, len(jobs)), mp_context=ctx) as pool:
            futs = [pool.submit(run_vit, rid, best_mult) for rid in jobs]
            for f in futs:
                f.result()

    conds = ["blind", "blindpair", "blindpair2d", "blindvit"]
    rows = [load_run(rid, cond) for rid in FIX_TARGETS for cond in conds]
    print_table(rows)
    pooled: dict[str, Any] = {}
    for cond in conds:
        sub = [r for r in rows if r["condition"] == cond]
        pooled[cond] = {
            "refined": pooled_over_recordings(sub, "refined"),
            "n_twins_resolved": sum(r["refined"]["twins_resolved"] for r in sub),
        }
        p = pooled[cond]["refined"]
        print(
            f"POOLED(2rec) {cond:<12} refined {p['err']:.3f} err_sm {p['err_sm']:.3f} "
            f"(bias {p['bias']:+.3f}) | telemetry-refine 0.604 / err_sm ~0.350",
            flush=True,
        )
    stage_report: dict[str, Any] = {}
    spectra: dict[str, Any] = {}
    for rid in FIX_TARGETS:
        with np.load(OUT_DIR / f"{rid}__blindvit.npz") as z:
            labels = [str(v) for v in z["stage_labels"]]
            errs = [round(float(v), 3) for v in z["stage_errs"]]
            errs_sm = [round(float(v), 3) for v in z["stage_errs_sm"]]
            seps = [round(float(v), 3) for v in z["pair_seps"]]
            best_stage = str(z["best_stage"])
            meas, edge = z["measured"], z["edge"].astype(bool)
        order = np.argsort(meas[:, edge].mean(axis=1))
        true_seps = [
            float(np.mean(meas[order[1], edge] - meas[order[0], edge])),
            float(np.mean(meas[order[3], edge] - meas[order[2], edge])),
        ]
        stage_report[rid] = {
            "stages": {lb: [e, es] for lb, e, es in zip(labels, errs, errs_sm)},
            "best_stage": best_stage,
            "est_pair_seps": seps,
            "true_pair_seps": [round(v, 3) for v in true_seps],
        }
        print(
            f"stages {rid} (best={best_stage}): "
            + ", ".join(f"{lb}={e}/{es}" for lb, e, es in zip(labels, errs, errs_sm))
        )
        print(f"  est pair seps {seps} vs true {np.round(true_seps, 2)}")
        spectra[rid] = wander_error_spectrum(rid, cond="blindvit")
        print(
            f"  wander-error spectrum (best stage): rms {spectra[rid]['rms']:.3f}, peak "
            f"{spectra[rid]['peak_hz']:.2f} Hz, var frac <=0.5/1/2 Hz = "
            f"{spectra[rid]['var_frac_below_0.5hz']:.2f}/"
            f"{spectra[rid]['var_frac_below_1hz']:.2f}/"
            f"{spectra[rid]['var_frac_below_2hz']:.2f}",
            flush=True,
        )
        make_blind_plot(rid, f"vit_{rid}.png", cond="blindvit")
        make_pairscan_diag(rid, cond="blindvit", out_prefix="vit_diag")
        make_vit_heatmap(rid)
    strip = lambda s: {k: v for k, v in s.items() if not k.startswith("_")}  # noqa: E731
    summary = {
        "viterbi": {
            "delta": VIT_DELTA,
            "gamma_mult_chosen": best_mult,
            "sep_grid": [float(VIT_SEP_GRID[0]), float(VIT_SEP_GRID[-1]), 0.02],
            "sep_kmin": VIT_SEP_KMIN,
            "win_s": PAIRSCAN_WIN_S,
            "hop_s": PAIRSCAN_HOP_S,
        },
        "midband_config": asdict(MIDBAND_CFG),
        "stage_errs": stage_report,
        "wander_error_spectrum": spectra,
        "rows": [
            {
                **{k: v for k, v in r.items() if k not in ("init", "captured", "refined")},
                "init": strip(r["init"]),
                "captured": strip(r["captured"]),
                "refined": strip(r["refined"]),
            }
            for r in rows
        ],
        "pooled": pooled,
    }
    with open(OUT_DIR / "vit_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"viterbi artifacts: {OUT_DIR}/vit_summary.json + vit_*.png")


def run_pairscan(rid: str) -> str:
    """Worker: blind init -> pair-rigid scan pipeline; saves NPZ."""
    cond = "blindpair"
    path = OUT_DIR / f"{rid}__{cond}.npz"
    if path.exists():
        return str(path)
    prep = prepare_recording(rid)
    scan = get_scan(rid)
    r0 = build_init(prep, "blind", scan)

    tic = time.perf_counter()
    r_fin, stages, pairs, diag, ref = vk_track_pair(prep, r0)
    wall = time.perf_counter() - tic

    def pit_err(traj: np.ndarray) -> float:
        p = pit_perm(traj, prep.r_meas, prep.edge)
        return float(np.mean(np.abs(traj[list(p)][:, prep.edge] - prep.r_meas[:, prep.edge])))

    labels = [s[0] for s in stages]
    errs = [pit_err(s[1]) for s in stages]
    print(
        f"[{rid} | {cond}] {wall:.0f}s  PIT err per stage: "
        + ", ".join(f"{lb}={e:.3f}" for lb, e in zip(labels, errs)),
        flush=True,
    )
    # "captured" = the last pair-scan stage before the final refine.
    cap_idx = max(i for i, lb in enumerate(labels) if "pairscan" in lb)
    arrays: dict[str, Any] = {
        "ft": prep.ft,
        "edge": prep.edge,
        "init": r0,
        "command": prep.r_init,
        "measured": prep.r_meas,
        "measured_sm": prep.r_meas_sm,
        "captured": stages[cap_idx][1],
        "refined": r_fin,
        "confidence": ref.confidence,
        "cap_residual_ratios": np.zeros(0),
        "cap_max_deltas": np.zeros(0),
        "ref_residual_ratios": np.array(ref.residual_ratios),
        "ref_max_deltas": np.array(ref.max_deltas),
        "tau": prep.tau,
        "seg_bounds": np.array([prep.seg_lo, prep.seg_hi]),
        "wall_capture_s": wall,
        "wall_refine_s": 0.0,
        "base": float(scan["base"]),
        "init4": np.asarray(scan["init4"], dtype=np.float64),
        "stage_snaps": np.stack([s[1] for s in stages]),
        "stage_labels": np.array(labels),
        "stage_errs": np.array(errs),
        "pairs": np.array(pairs),
        **{f"diag_{k}": v for k, v in diag.items()},
    }
    np.savez(path, **arrays)
    return str(path)


def make_pairscan_diag(
    rid: str, cond: str = "blindpair", out_prefix: str = "pairscan_diag"
) -> None:
    """Mechanism check: per-window pair delta-hat vs the true pair-mean wander.

    Columns = first and last scan round of the ``cond`` run.
    """
    with np.load(OUT_DIR / f"{rid}__{cond}.npz") as z:
        arrs = {k: z[k] for k in z.files}
    ft, edge = arrs["ft"], arrs["edge"].astype(bool)
    meas = arrs["measured"]
    init = arrs["init"]
    pairs = arrs["pairs"]
    p = pit_perm(init, meas, edge)  # rotor identity at the diagnostic's stage
    # pit_perm returns the row order of `init` matching meas rows; invert it so
    # inv[track_row] = the truth row matched to that track row.
    inv = np.empty(4, dtype=int)
    for truth_row, track_row in enumerate(list(p)):
        inv[track_row] = truth_row
    rounds = sorted(
        {int(k.split("_")[1][2:]) for k in arrs if k.startswith("diag_rd") and "centers" in k}
    )
    show = [rounds[0], rounds[-1]] if len(rounds) > 1 else rounds
    fig, ax = plt.subplots(2, len(show), figsize=(6.5 * len(show), 8), sharex=True, squeeze=False)
    for ci, rd in enumerate(show):
        centers = arrs[f"diag_rd{rd}_centers"]
        for pi in range(2):
            pair = [int(v) for v in pairs[pi]]
            dvals = arrs[f"diag_rd{rd}_dvals_p{pi}"]
            weights = arrs[f"diag_rd{rd}_weights_p{pi}"]
            r_before = arrs[f"diag_rd{rd}_r_before_p{pi}"]  # (2, N) at scan time
            true_dc = np.stack(
                [np.interp(centers, ft, meas[inv[i]] - r_before[j]) for j, i in enumerate(pair)]
            ).mean(axis=0)
            a = ax[pi][ci]
            a.plot(centers, true_dc, "k-", lw=1.4, label="true pair-mean wander")
            sc = a.scatter(
                centers, dvals, c=weights, cmap="viridis", s=22, zorder=5, label="pair-scan delta"
            )
            a.axhline(0, color="gray", lw=0.6)
            a.set_ylabel("delta_c (rev/s)")
            a.set_title(f"pair {pi} (rotors {pair}), scan round {rd + 1}")
            a.legend(fontsize=8)
            fig.colorbar(sc, ax=a, label="weight")
        ax[1][ci].set_xlabel("segment time (s)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{out_prefix}_{rid}.png", dpi=150)
    plt.close(fig)


def wander_error_spectrum(rid: str, cond: str = "blindpair") -> dict[str, float]:
    """PSD of the rotor-mean signed tracking error — which band stays untracked."""
    from scipy.signal import welch

    with np.load(OUT_DIR / f"{rid}__{cond}.npz") as z:
        ft, edge = z["ft"], z["edge"].astype(bool)
        meas, ref = z["measured"], z["refined"]
    p = pit_perm(ref, meas, edge)
    e = (ref[list(p)] - meas)[:, edge].mean(axis=0)
    e = e - e.mean()
    fs = 1.0 / float(ft[1] - ft[0])
    f, pxx = welch(e, fs=fs, nperseg=min(len(e), 256))
    total = float(np.trapezoid(pxx, f))
    return {
        "peak_hz": float(f[int(np.argmax(pxx))]),
        "var_frac_below_0.5hz": float(np.trapezoid(pxx[f <= 0.5], f[f <= 0.5]) / max(total, 1e-12)),
        "var_frac_below_1hz": float(np.trapezoid(pxx[f <= 1.0], f[f <= 1.0]) / max(total, 1e-12)),
        "var_frac_below_2hz": float(np.trapezoid(pxx[f <= 2.0], f[f <= 2.0]) / max(total, 1e-12)),
        "rms": float(np.sqrt(np.mean(e**2))),
    }


def pairscan_main() -> None:
    """Pair-rigid scan experiment on FIX_TARGETS (blind init)."""
    jobs = [rid for rid in FIX_TARGETS if not (OUT_DIR / f"{rid}__blindpair.npz").exists()]
    if jobs:
        print(f"running {len(jobs)} pair-scan jobs on {len(jobs)} workers", flush=True)
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=max(1, len(jobs)), mp_context=ctx) as pool:
            futs = [pool.submit(run_pairscan, rid) for rid in jobs]
            for f in futs:
                f.result()

    conds = ["blind", "blindscanK1", "blindpair"]
    rows = [load_run(rid, cond) for rid in FIX_TARGETS for cond in conds]
    print_table(rows)
    pooled: dict[str, Any] = {}
    for cond in conds:
        sub = [r for r in rows if r["condition"] == cond]
        pooled[cond] = {
            "refined": pooled_over_recordings(sub, "refined"),
            "n_twins_resolved": sum(r["refined"]["twins_resolved"] for r in sub),
        }
        p = pooled[cond]["refined"]
        print(
            f"POOLED(2rec) {cond:<12} refined {p['err']:.3f} (bias {p['bias']:+.3f}, "
            f"vs smoothed {p['err_sm']:.3f}) | telemetry-refine ref 0.604",
            flush=True,
        )
    stage_report = {}
    spectra = {}
    for rid in FIX_TARGETS:
        with np.load(OUT_DIR / f"{rid}__blindpair.npz") as z:
            labels = [str(v) for v in z["stage_labels"]]
            errs = [round(float(v), 3) for v in z["stage_errs"]]
        stage_report[rid] = dict(zip(labels, errs))
        print(f"stages {rid}: " + ", ".join(f"{lb}={e}" for lb, e in zip(labels, errs)))
        spectra[rid] = wander_error_spectrum(rid)
        print(
            f"wander-error spectrum {rid}: rms {spectra[rid]['rms']:.3f}, "
            f"peak {spectra[rid]['peak_hz']:.2f} Hz, var frac <=0.5/1/2 Hz = "
            f"{spectra[rid]['var_frac_below_0.5hz']:.2f}/"
            f"{spectra[rid]['var_frac_below_1hz']:.2f}/"
            f"{spectra[rid]['var_frac_below_2hz']:.2f}",
            flush=True,
        )
        make_blind_plot(rid, f"pairscan_{rid}.png", cond="blindpair")
        make_pairscan_diag(rid)
    strip = lambda s: {k: v for k, v in s.items() if not k.startswith("_")}  # noqa: E731
    summary = {
        "pairscan": {
            "rounds": PAIRSCAN_ROUNDS,
            "max_iters": PAIRSCAN_MAX_ITERS,
            "bar": PAIRSCAN_BAR,
            "delta": SCANLOOP_DELTA,
            "dstep": SCANLOOP_DSTEP,
            "win_s": PAIRSCAN_WIN_S,
            "hop_s": PAIRSCAN_HOP_S,
            "lambda": PAIRSCAN_LAMBDA,
            "k": [1, 30],
        },
        "stage_errs": stage_report,
        "wander_error_spectrum": spectra,
        "rows": [
            {
                **{k: v for k, v in r.items() if k not in ("init", "captured", "refined")},
                "init": strip(r["init"]),
                "captured": strip(r["captured"]),
                "refined": strip(r["refined"]),
            }
            for r in rows
        ],
        "pooled": pooled,
    }
    with open(OUT_DIR / "pairscan_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"pair-scan artifacts: {OUT_DIR}/pairscan_summary.json + pairscan_*.png")


# ---------------------------------------------------------------------------
# Metrics (main process, recomputed from NPZ -> resumable)


def pit_perm(traj: np.ndarray, gt: np.ndarray, edge: np.ndarray) -> tuple[int, ...]:
    """Best rotor permutation of ``traj`` rows onto ``gt`` rows (min masked MAE)."""
    best, best_p = np.inf, tuple(range(gt.shape[0]))
    for p in itertools.permutations(range(gt.shape[0])):
        c = float(np.mean(np.abs(traj[list(p)][:, edge] - gt[:, edge])))
        if c < best:
            best, best_p = c, p
    return best_p


def min_pair_sep(traj: np.ndarray, edge: np.ndarray) -> float:
    """Min over rotor pairs of the mean absolute separation (masked)."""
    seps = [
        float(np.mean(np.abs(traj[i, edge] - traj[j, edge])))
        for i in range(traj.shape[0])
        for j in range(i + 1, traj.shape[0])
    ]
    return min(seps)


def pit_stats(traj: np.ndarray, z: dict[str, np.ndarray]) -> dict[str, Any]:
    """PIT-aligned metrics of one trajectory set vs measured truth."""
    meas, meas_sm, edge = z["measured"], z["measured_sm"], z["edge"].astype(bool)
    p = pit_perm(traj, meas, edge)
    a = traj[list(p)]
    d, d_sm = (a - meas)[:, edge], (a - meas_sm)[:, edge]
    err_rotor = np.mean(np.abs(d), axis=1)
    truth_sep = min_pair_sep(meas, edge)
    track_sep = min_pair_sep(a, edge)
    return {
        "perm": list(p),
        "err": float(np.mean(np.abs(d))),
        "bias": float(np.mean(d)),
        "err_sm": float(np.mean(np.abs(d_sm))),
        "bias_sm": float(np.mean(d_sm)),
        "err_rotor": [float(v) for v in err_rotor],
        "bias_rotor": [float(v) for v in np.mean(d, axis=1)],
        "min_pair_sep_tracks": track_sep,
        "min_pair_sep_truth": truth_sep,
        # 4 distinct trajectories matching the 4 true rotors: every rotor's
        # PIT error below the truth's tightest pair gap AND the tracks keep
        # at least half of that gap between themselves (no twin collapse).
        "twins_resolved": bool(np.all(err_rotor < truth_sep) and track_sep > 0.5 * truth_sep),
        "_d": d,
        "_d_sm": d_sm,
    }


def load_run(rid: str, cond: str) -> dict[str, Any]:
    path = OUT_DIR / f"{rid}__{cond}.npz"
    with np.load(path) as z:
        arrs = {k: z[k] for k in z.files}
    row: dict[str, Any] = {
        "recording": rid,
        "condition": cond,
        "wall_capture_s": round(float(arrs["wall_capture_s"]), 1),
        "wall_refine_s": round(float(arrs["wall_refine_s"]), 1),
        "tau": round(float(arrs["tau"]), 4),
        "base": round(float(arrs["base"]), 2),
        "init4": [round(float(v), 2) for v in arrs["init4"]],
        "init": pit_stats(arrs["init"], arrs),
        "captured": pit_stats(arrs["captured"], arrs),
        "refined": pit_stats(arrs["refined"], arrs),
        "cap_residual_ratios": [round(float(v), 4) for v in arrs["cap_residual_ratios"]],
        "ref_residual_ratios": [round(float(v), 4) for v in arrs["ref_residual_ratios"]],
        "mean_confidence": float(arrs["confidence"].mean()),
    }
    return row


def pooled_over_recordings(rows: list[dict[str, Any]], stage: str) -> dict[str, float]:
    d = np.concatenate([r[stage]["_d"] for r in rows], axis=1)
    d_sm = np.concatenate([r[stage]["_d_sm"] for r in rows], axis=1)
    return {
        "err": float(np.mean(np.abs(d))),
        "bias": float(np.mean(d)),
        "err_sm": float(np.mean(np.abs(d_sm))),
        "bias_sm": float(np.mean(d_sm)),
    }


# ---------------------------------------------------------------------------
# Figures


def make_blind_plot(rid: str, fname: str, cond: str = "blind") -> None:
    """Refined (PIT-aligned) vs measured truth for a blind-family condition."""
    with np.load(OUT_DIR / f"{rid}__{cond}.npz") as z:
        arrs = {k: z[k] for k in z.files}
    ft, edge = arrs["ft"], arrs["edge"].astype(bool)
    meas = arrs["measured"]
    p_cap = pit_perm(arrs["captured"], meas, edge)
    p_ref = pit_perm(arrs["refined"], meas, edge)
    cap, ref = arrs["captured"][list(p_cap)], arrs["refined"][list(p_ref)]
    init = arrs["init"]  # blind constants; identity order is meaningless — plot flat

    fig, ax = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    colors = ("#e41a1c", "#377eb8", "#4daf4a", "#984ea3")
    for i in range(4):
        ax[0].plot(ft, meas[i], "k-", lw=1.0, alpha=0.8, label="measured (GT)" if i == 0 else None)
        ax[0].plot(
            ft,
            ref[i],
            color=colors[i],
            lw=1.3,
            label=f"blind refined r{i}" if i < 4 else None,
        )
        ax[0].plot(
            ft,
            init[i],
            ls=":",
            color=colors[i],
            lw=0.9,
            alpha=0.6,
            label="blind init" if i == 0 else None,
        )
    ax[0].set_ylabel("rev/s")
    ax[0].set_title(f"{rid}: {cond} annotation, refined vs truth (PIT-aligned)")
    ax[0].legend(fontsize=8, ncol=3)

    for i in range(4):
        ax[1].plot(
            ft,
            np.abs(cap[i] - meas[i]),
            ls="--",
            color=colors[i],
            lw=0.8,
            alpha=0.6,
            label="after capture" if i == 0 else None,
        )
        ax[1].plot(
            ft,
            np.abs(ref[i] - meas[i]),
            color=colors[i],
            lw=1.0,
            label="after refine" if i == 0 else None,
        )
    ax[1].set_xlabel("segment time (s)")
    ax[1].set_ylabel("|error| (rev/s)")
    ax[1].set_yscale("log")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / fname, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Per-round trace (failure diagnosis)


def run_trace(rid: str, cond: str) -> None:
    """Re-run one condition round-by-round, snapshotting trajectories per round.

    Emulates each outer round of ``vk_track`` by a 1-round fixed-schedule call
    with that round's annealed (k_max, bw, lambda). Near-exact: the only
    deviation is the per-group ``sep_bw_factor`` clamp floor, which reads
    ``cfg.bw_hz`` (here the round's bw instead of the final one).
    """
    from data_processing.vk_tracking import _bw_schedule, _k_schedule, _stride

    prep = prepare_recording(rid)
    scan = get_scan(rid)
    r_cur = build_init(prep, cond, scan)
    snaps: list[np.ndarray] = [r_cur.copy()]
    labels: list[str] = ["init"]
    for phase, cfg in (("capture", CAPTURE_CFG), ("refine", REFINE_CFG)):
        ks = _k_schedule(cfg)
        _, fs_env = _stride(cfg)
        bws = _bw_schedule(cfg, fs_env)
        lams = (
            np.geomspace(10.0 * cfg.traj_lambda, cfg.traj_lambda, cfg.n_outer)
            if cfg.n_outer > 1
            else [cfg.traj_lambda]
        )
        for rd in range(cfg.n_outer):
            cfg_rd = replace(
                cfg,
                n_outer=1,
                k_schedule="fixed",
                k_max=int(ks[rd]),
                bw_hz=float(bws[rd]),
                traj_lambda=float(lams[rd]),
            )
            tic = time.perf_counter()
            res = vk_track(prep.audio, r_cur, prep.ft, cfg_rd)
            r_cur = res.r_refined
            snaps.append(r_cur.copy())
            labels.append(f"{phase} rd{rd} (k<={ks[rd]}, bw={bws[rd]:.2f})")
            p = pit_perm(r_cur, prep.r_meas, prep.edge)
            err = float(np.mean(np.abs(r_cur[list(p)][:, prep.edge] - prep.r_meas[:, prep.edge])))
            print(f"[trace {labels[-1]}] PIT err {err:.3f} ({time.perf_counter() - tic:.0f}s)")

    np.savez(
        OUT_DIR / f"trace_{rid}__{cond}.npz",
        snaps=np.stack(snaps),
        labels=np.array(labels),
        ft=prep.ft,
        measured=prep.r_meas,
        edge=prep.edge,
    )

    # Plot: per-round PIT error + rotor-mean trajectories across rounds.
    fig, ax = plt.subplots(2, 1, figsize=(12, 9))
    errs = []
    for s in snaps:
        p = pit_perm(s, prep.r_meas, prep.edge)
        errs.append(float(np.mean(np.abs(s[list(p)][:, prep.edge] - prep.r_meas[:, prep.edge]))))
    ax[0].plot(range(len(errs)), errs, "o-")
    ax[0].set_xticks(range(len(errs)))
    ax[0].set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
    ax[0].set_ylabel("pooled PIT |err| (rev/s)")
    ax[0].set_yscale("log")
    ax[0].set_title(f"trace {rid} | {cond}: error vs outer round")

    cmap = plt.get_cmap("viridis")
    for si, s in enumerate(snaps):
        col = cmap(si / max(1, len(snaps) - 1))
        for i in range(4):
            ax[1].plot(prep.ft, s[i], color=col, lw=0.7, alpha=0.7)
    for i in range(4):
        ax[1].plot(prep.ft, prep.r_meas[i], "k-", lw=1.2, alpha=0.9)
    ax[1].set_xlabel("segment time (s)")
    ax[1].set_ylabel("rev/s")
    ax[1].set_title("trajectories per round (viridis: dark=init -> light=final; black=truth)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"trace_{rid}__{cond}.png", dpi=150)
    plt.close(fig)
    print(f"trace saved: {OUT_DIR}/trace_{rid}__{cond}.npz/.png")


# ---------------------------------------------------------------------------


def print_table(rows: list[dict[str, Any]]) -> None:
    width = 128
    print("\n" + "=" * width)
    print(
        "BLIND ANNOTATION: capture+refine vs measured GT, PIT-aligned "
        "(reference: telemetry-refine pooled err 0.604 / bias -0.075)"
    )
    print("=" * width)
    print(
        f"{'recording':<34}{'condition':<12}{'init_err':>9}{'cap_err':>9}{'ref_err':>9}"
        f"{'ref_bias':>9}{'ref_err_sm':>11}{'minsep_t':>9}{'minsep_gt':>10}{'twins':>7}"
        f"{'wall_s':>8}"
    )
    print("-" * width)
    for r in rows:
        print(
            f"{r['recording']:<34}{r['condition']:<12}"
            f"{r['init']['err']:>9.3f}{r['captured']['err']:>9.3f}{r['refined']['err']:>9.3f}"
            f"{r['refined']['bias']:>9.3f}{r['refined']['err_sm']:>11.3f}"
            f"{r['refined']['min_pair_sep_tracks']:>9.3f}"
            f"{r['refined']['min_pair_sep_truth']:>10.3f}"
            f"{'YES' if r['refined']['twins_resolved'] else 'no':>7}"
            f"{r['wall_capture_s'] + r['wall_refine_s']:>8.0f}"
        )
    print("-" * width)


def fix_main() -> None:
    """Wander-tracking fix experiment: arms A/B on FIX_TARGETS, blind only."""
    jobs = [
        (rid, arm)
        for rid in FIX_TARGETS
        for arm in ("A", "B")
        if not (OUT_DIR / f"{rid}__blindfix{arm}.npz").exists()
    ]
    if jobs:
        print(f"running {len(jobs)} fix-arm jobs on {len(jobs)} workers", flush=True)
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=len(jobs), mp_context=ctx) as pool:
            futs = [pool.submit(run_arm, rid, arm) for rid, arm in jobs]
            for f in futs:
                f.result()

    conds = ("blind", "blindfixA", "blindfixB")
    rows = [load_run(rid, cond) for rid in FIX_TARGETS for cond in conds]
    print_table(rows)
    pooled: dict[str, Any] = {}
    for cond in conds:
        sub = [r for r in rows if r["condition"] == cond]
        pooled[cond] = {
            "init": pooled_over_recordings(sub, "init"),
            "captured": pooled_over_recordings(sub, "captured"),
            "refined": pooled_over_recordings(sub, "refined"),
            "n_twins_resolved": sum(r["refined"]["twins_resolved"] for r in sub),
        }
        p = pooled[cond]["refined"]
        print(
            f"POOLED(2rec) {cond:<10} refined {p['err']:.3f} (bias {p['bias']:+.3f}, "
            f"vs smoothed {p['err_sm']:.3f}) | telemetry-refine ref 0.604",
            flush=True,
        )
    best = min(("blindfixA", "blindfixB"), key=lambda c: pooled[c]["refined"]["err"])
    for rid in FIX_TARGETS:
        make_blind_plot(rid, f"fix_{best}_{rid}.png", cond=best)
    strip = lambda s: {k: v for k, v in s.items() if not k.startswith("_")}  # noqa: E731
    summary = {
        "track_config": asdict(TRACK_CFG),
        "arm_b": {"win_s": WSCAN_WIN_S, "hop_s": WSCAN_HOP_S, "capture_n_outer": 4},
        "best_arm": best,
        "rows": [
            {
                **{k: v for k, v in r.items() if k not in ("init", "captured", "refined")},
                "init": strip(r["init"]),
                "captured": strip(r["captured"]),
                "refined": strip(r["refined"]),
            }
            for r in rows
        ],
        "pooled": pooled,
    }
    with open(OUT_DIR / "fix_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"fix artifacts: {OUT_DIR}/fix_summary.json + fix_{best}_<rid>.png", flush=True)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", nargs=2, metavar=("RID", "COND"), default=None)
    parser.add_argument("--fix", action="store_true", help="run the wander-tracking fix arms")
    parser.add_argument(
        "--scanloop", action="store_true", help="run the residual scan-in-the-loop experiment"
    )
    parser.add_argument(
        "--pairscan", action="store_true", help="run the pair-rigid windowed-scan experiment"
    )
    parser.add_argument(
        "--pair2d", action="store_true", help="run the 2-D pair scan + mid-band + refine"
    )
    parser.add_argument("--vit", action="store_true", help="run the Viterbi-pass experiment")
    parser.add_argument(
        "--vit2d", action="store_true", help="run the joint 2-rotor Viterbi capstone"
    )
    parser.add_argument(
        "--vit2d-spatial",
        action="store_true",
        help="run the spatial-contrast joint Viterbi finale",
    )
    parser.add_argument(
        "--vit2d-affinity",
        action="store_true",
        help="run the measured-affinity joint DP + lambda sweep",
    )
    parser.add_argument(
        "--lambda-sweep",
        nargs="?",
        const="blindvit2dsp",
        default=None,
        metavar="SOURCE_COND",
        help="sweep midband/refine traj_lambda from a saved run's tracks",
    )
    args = parser.parse_args()
    if args.trace:
        run_trace(args.trace[0], args.trace[1])
        return
    if args.fix:
        fix_main()
        return
    if args.scanloop:
        scanloop_main()
        return
    if args.pairscan:
        pairscan_main()
        return
    if args.pair2d:
        pair2d_main()
        return
    if args.vit:
        vit_main()
        return
    if args.vit2d:
        vit2d_main()
        return
    if args.vit2d_spatial:
        vit2dsp_main()
        return
    if args.vit2d_affinity:
        vit2daff_main()
        return
    if args.lambda_sweep:
        lambda_sweep_main(source_cond=args.lambda_sweep)
        return

    # Blind scans first (serial, cached): const-mean/blind inits need them, and
    # precomputing keeps torch out of the vk workers.
    for rid in DREGON_TARGETS:
        get_scan(rid)

    jobs = [
        (rid, cond)
        for rid in DREGON_TARGETS
        for cond in CONDITIONS
        if not (OUT_DIR / f"{rid}__{cond}.npz").exists()
    ]
    tic = time.perf_counter()
    if jobs:
        print(f"running {len(jobs)} (recording, condition) jobs on {N_WORKERS} workers")
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=N_WORKERS, mp_context=ctx) as pool:
            futs = [pool.submit(run_condition, rid, cond) for rid, cond in jobs]
            for f in futs:
                f.result()
    wall_total = time.perf_counter() - tic

    rows = [load_run(rid, cond) for rid in DREGON_TARGETS for cond in CONDITIONS]
    print_table(rows)

    pooled: dict[str, Any] = {}
    for cond in CONDITIONS:
        sub = [r for r in rows if r["condition"] == cond]
        pooled[cond] = {
            "init": pooled_over_recordings(sub, "init"),
            "captured": pooled_over_recordings(sub, "captured"),
            "refined": pooled_over_recordings(sub, "refined"),
            "n_twins_resolved": sum(r["refined"]["twins_resolved"] for r in sub),
        }
        p = pooled[cond]["refined"]
        print(
            f"POOLED {cond:<12} init {pooled[cond]['init']['err']:.3f} -> "
            f"capture {pooled[cond]['captured']['err']:.3f} -> refined {p['err']:.3f} "
            f"(bias {p['bias']:+.3f}, vs smoothed {p['err_sm']:.3f}), "
            f"twins resolved {pooled[cond]['n_twins_resolved']}/5"
        )

    strip = lambda s: {k: v for k, v in s.items() if not k.startswith("_")}  # noqa: E731
    summary = {
        "capture_config": asdict(CAPTURE_CFG),
        "refine_config": asdict(REFINE_CFG),
        "reference": REFERENCE,
        "conditions": list(CONDITIONS),
        "rows": [
            {
                **{k: v for k, v in r.items() if k not in ("init", "captured", "refined")},
                "init": strip(r["init"]),
                "captured": strip(r["captured"]),
                "refined": strip(r["refined"]),
            }
            for r in rows
        ],
        "pooled": pooled,
        "wall_total_s": round(wall_total, 1),
    }
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    make_blind_plot("free-flight_nosource_room1", "blind_easy.png")
    # hard case: worst blind refined err among the other recordings
    hard = max(
        (r for r in rows if r["condition"] == "blind" and r["recording"] != DREGON_TARGETS[0]),
        key=lambda r: r["refined"]["err"],
    )
    make_blind_plot(hard["recording"], "blind_hard.png")
    print(f"\nArtifacts written to {OUT_DIR}/ (wall {wall_total:.0f}s)")


if __name__ == "__main__":
    main()
