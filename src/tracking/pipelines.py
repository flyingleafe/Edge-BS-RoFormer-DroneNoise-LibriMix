"""The canonical tracking ladders: frozen configs, vit2dsp, peeled alternation.

Two ladders live here, both calibrated and both frozen:

1. The blind RPS-annotation ladder that ``scripts/vk_blind_annotation.py``
   validated (DREGON pooled err_sm 0.688, condition ``blindvit2dsp``)::

       blind init -> Viterbi pair-mean c(t) -> SPATIAL joint 2-rotor Viterbi
       (per-rotor mic mixes) -> mid-band VK (bw 6) -> VK refine

2. The FLAGSHIP peeled alternation (``docs/experiments/beat-vk.md``), which
   starts from that ladder's output and iterates::

       peel (VK envelopes at the current track -> per-harmonic least-squares
       re-fit -> subtract the OTHER rotors' combs) -> one pi_kalman pass

   :func:`make_peels` is the peel, :func:`pi_kalman_arm_stage` is one
   application as a Stage, and :func:`peel_alternation` iterates it.

Two layers live here:

- The array core (:func:`vit2dsp_pipeline` and its helpers
  :func:`vit_stage1`, :func:`tooth_cube`, :func:`pair_score_2d_spatial`,
  :func:`joint_viterbi`, :func:`apply_guard`) — moved verbatim from the
  script; the science is unchanged.
- The frame adapter (:func:`vit2dsp_stage`) — a ``tracking.stages`` Stage
  that runs the ladder on a tracking frame and seeds itself via
  :func:`tracking.stages.blind_seed_stage` when the frame has no ``"rps"``
  entry.

FROZEN CONFIG REGISTRY: ``CAPTURE_CFG`` / ``REFINE_CFG`` / ``TRACK_CFG`` /
``MIDBAND_CFG`` / ``MIDBAND_CFGS`` / ``SEED_CFG`` and the ladder constants
below are the calibrated blind-annotation ladder configs. Changing any value
invalidates the published annotations (and every number derived from them) —
treat them as data, not knobs. ``tests/tracking/test_pipelines.py``
spot-checks the values against the published calibration.

What stays in ``scripts/vk_blind_annotation.py``: recording preparation and
GT scoring (``Prepared``, PIT metrics — they load DREGON data), the mic-
geometry weights (``_rotor_mic_weights`` imports ``data_processing``), and
every superseded experiment arm. Purity rule: this module imports only
numpy/scipy/torch (via the tracking cores) and ``tracking.*`` — enforced by
the ``tracking stays pure`` import-linter contract.
"""

from __future__ import annotations

import dataclasses
import time
from dataclasses import dataclass, replace
from typing import Any, Protocol

import numpy as np
import tdseries as td

from tracking.phase_increment_tracker import pi_kalman_refine
from tracking.stages import (
    DEFAULT_HOP_S,
    Stage,
    blind_seed_stage,
    get_audio,
    get_rps,
    with_rps,
)
from tracking.vk_blind_seeding import SeedConfig, logmag_spectrogram, whitened_logmag
from tracking.vk_blind_seeding import stage_guard as _stage_guard_fn
from tracking.vk_tracking import (
    VKConfig,
    VKResult,
    ls_project_envelopes,
    vk_envelopes,
    vk_reconstruct,
    vk_track,
)

__all__ = [
    "ARMS",
    "CAPTURE_CFG",
    "DEFAULT_PEEL_MODE",
    "LADDER_N_ROTORS",
    "LadderInput",
    "MIDBAND_CFG",
    "MIDBAND_CFGS",
    "PAIRSCAN_HOP_S",
    "PAIRSCAN_WIN_S",
    "PEEL_BW_HZ",
    "PEEL_K_MAX",
    "PEEL_MODES",
    "PI_BAND_HZ",
    "PI_N_ITER",
    "PI_PAIR_MODE",
    "PI_VARIANTS",
    "REFINE_CFG",
    "SEED_CFG",
    "SR",
    "TRACK_CFG",
    "VIT2D_BEAM",
    "VIT2D_DELTA",
    "VIT2D_STEP",
    "VIT_DELTA",
    "VIT_DSTEP",
    "VIT_GAMMA_MULT",
    "apply_guard",
    "joint_viterbi",
    "local_comb_frame_scores",
    "make_peels",
    "pair_score_2d_spatial",
    "pair_surface",
    "peel_alternation",
    "pi_kalman_arm_stage",
    "surface_contrast",
    "tooth_cube",
    "vit2dsp_pipeline",
    "vit2dsp_stage",
    "vit_stage1",
    "viterbi_lattice",
    "viterbi_ridge",
    "whitened_logmag_multi",
]

#: The sample rate every frozen config below is calibrated at.
SR = 16000

#: The ladder is a two-twin-pair (quadrotor) construction throughout.
LADDER_N_ROTORS = 4


# ---------------------------------------------------------------------------
# frozen config registry (calibrated — see the module docstring)


# REFINE: the validated de-biasing config (``scripts/vk_validation.py``'s
# MAIN_CFG — fixed schedule, k_min=6 excludes the twin-merged low harmonics
# from the Fisher fusion, narrow bands, tight step).
REFINE_CFG = VKConfig(
    fs=float(SR),
    couple_hz=20.0,
    n_outer=5,
    k_min=6,
    k_max=30,
    k_schedule="fixed",
    bw_hz=1.5,
    max_step=0.3,
)

# CAPTURE: the annealed grow schedule on top of the refine config — the
# validated capture-basin recipe (+-2..3 rev/s recovery). The literally
# specified capture config with k_min=1 library defaults stalls from every
# non-telemetry init (vk_blind_annotation DOCUMENTED FIX #2).
CAPTURE_CFG = replace(REFINE_CFG, k_schedule="grow", n_outer=12)

# TRACK: the mid-band wander-tracking phase between capture and refine — at
# k=6-12 a 7 Hz band reads detunings up to +-0.5 rev/s and re-centers every
# round (wide enough to follow the +-1.5 rev/s flight wander, high-enough k
# to reject twins).
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

# MIDBAND: the phase stage after the DP scans (k 6..10 — per-rotor tolerance
# bw/2k ~ 0.33 rev/s at bw 4).
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

# The vit2dsp ladder runs the wide (bw 6) element first; MIDBAND_CFGS[1] is
# the second (bw 4) phase stage of the superseded v5/v6 arms.
MIDBAND_CFGS = (
    replace(MIDBAND_CFG, bw_hz=6.0, n_outer=4),
    replace(MIDBAND_CFG, bw_hz=4.0, n_outer=4),
)

# SEED: the validated blind-scan knobs (whitened comb scan, octave rule,
# harmonic-relation guard, R=4 init nudges) — written out in full even where
# a value equals the ``SeedConfig`` default, so the calibration is pinned
# here rather than in the dataclass defaults.
SEED_CFG = SeedConfig(
    scan_lo=30.0,
    scan_hi=120.0,
    scan_step=0.05,
    k_scan=40,
    whiten_hz=150.0,
    octave_rel=0.9,
    harm_guard=1.5,
    pair_nudge=0.5,
    blind_offsets=(-1.5, -0.5, 0.5, 1.5),
)

# PEEL / pi_kalman: the flagship alternation's frozen settings
# (docs/experiments/beat-vk.md). 1 Hz envelope bandwidth ~ 1 s coherence,
# inside the measured 0.5-1.5 s tau_k window at k = 8-40.
PEEL_BW_HZ = 1.0
PEEL_K_MAX = 40
#: Peel subtraction modes (issue #17 step 4):
#: ``"open"`` subtracts the VK reconstruction as solved (the 2026-08-04
#: flagship, whose mis-phased components could INJECT energy); ``"ls"``
#: re-fits each harmonic's complex gain onto the clip per time block first
#: (:func:`tracking.ls_project_envelopes`), so one component cannot.
PEEL_MODES = ("open", "ls")
DEFAULT_PEEL_MODE = "ls"

#: Protocol pi_kalman settings (the 0.641 row; findings.md "Iterated
#: pi_kalman: mechanism findings").
PI_N_ITER = 3
PI_BAND_HZ = 6.0
PI_PAIR_MODE = "joint"
#: Bandwidth-and-admission revision rows (docs/experiments/beat-vk.md): extra
#: :func:`tracking.pi_kalman_refine` kwargs per variant. ``k_anneal``/``full``
#: thread the annealed per-rotor ``band_b0`` across applications.
PI_VARIANTS: dict[str, dict[str, Any]] = {
    "protocol": {},
    "k_scaled": {"band_mode": "k_scaled"},
    "k_anneal": {"band_mode": "k_scaled", "band_anneal": "posterior"},
    "full": {
        "band_mode": "k_scaled",
        "band_anneal": "posterior",
        "lowk_gate": "consistency",
        "probe_mode": "clean",
    },
}
#: The two alternation arms: ``peeled`` (the flagship) and ``naive`` (plain
#: re-application, the comparison arm).
ARMS = ("naive", "peeled")

# Ladder constants (calibrated with the configs above).
PAIRSCAN_WIN_S = 1.0  # pair-template scan window (s)
PAIRSCAN_HOP_S = 0.25  # pair-template scan hop (s)
VIT_DELTA = 6.0  # rev/s: stage-1 Viterbi delta half-range (true pair-mean
# wander is +-3..5 rev/s p2p ~8 — a +-2 grid clips at the edge)
VIT_DSTEP = 0.05  # rev/s: stage-1 Viterbi delta-grid step (historically the
# script's SCANLOOP_DSTEP)
VIT_GAMMA_MULT = 0.3  # gamma = this x surface contrast (2.0 over-smooths)
VIT2D_DELTA = 6.0  # rev/s: joint 2-rotor DP delta half-range
VIT2D_STEP = 0.1  # rev/s: joint 2-rotor DP delta-grid step
VIT2D_BEAM = 3  # grid steps (= 0.3 rev/s) per rotor per hop


class LadderInput(Protocol):
    """What :func:`vit2dsp_pipeline` needs from a prepared recording.

    Structural (duck) type: the scripts' ``Prepared`` dataclass satisfies it,
    and :func:`vit2dsp_stage` builds a minimal stand-in from a frame.
    """

    @property
    def audio(self) -> np.ndarray:
        """``(C, T)`` segment audio at the pipeline's sample rate."""
        ...

    @property
    def ft(self) -> np.ndarray:
        """``(N,)`` trajectory frame times, seconds, audio-relative."""
        ...


@dataclass(frozen=True)
class _Segment:
    """Minimal :class:`LadderInput` (the frame adapter's stand-in)."""

    audio: np.ndarray
    ft: np.ndarray


# ---------------------------------------------------------------------------
# ladder core (moved verbatim from scripts/vk_blind_annotation.py)


def local_comb_frame_scores(
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


def pair_surface(
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
    fsc = local_comb_frame_scores(lm, bin_hz, r_specs, deltas, ks)  # (D, N)
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


def viterbi_lattice(surface: np.ndarray, grid: np.ndarray, gamma: float) -> np.ndarray:
    """Max-sum DP over a ``(n_steps, D)`` lattice; returns the ``grid`` path.

    THE dense L1-transition Viterbi of the blind ladders: emission
    ``surface[t, d]``, transition cost ``gamma * |grid[d] - grid[d']|``. Every
    blind scan (the pair-mean ridge, the coarse full-range pass) uses this
    one; ``tracking.rotor_dp.viterbi_path`` is the different, banded-Huber
    torch lattice of the DP tracker.
    """
    n_steps, _ = surface.shape
    trans = gamma * np.abs(grid[None, :] - grid[:, None])  # (D_prev, D_cur)
    cost = surface[0].copy()
    ptr = np.zeros((n_steps, len(grid)), dtype=int)
    for w in range(1, n_steps):
        m = cost[:, None] - trans
        ptr[w] = np.argmax(m, axis=0)
        cost = surface[w] + np.max(m, axis=0)
    path = np.empty(n_steps, dtype=int)
    path[-1] = int(np.argmax(cost))
    for w in range(n_steps - 1, 0, -1):
        path[w - 1] = ptr[w][path[w]]
    return grid[path]


def viterbi_ridge(surface: np.ndarray, deltas: np.ndarray, gamma: float) -> np.ndarray:
    """:func:`viterbi_lattice` on a per-window median-centered surface."""
    return viterbi_lattice(surface - np.median(surface, axis=1, keepdims=True), deltas, gamma)


def surface_contrast(surface: np.ndarray) -> float:
    """Median over windows of (max - median) node score — the gamma scale."""
    return float(np.median(np.max(surface, axis=1) - np.median(surface, axis=1)))


def vit_stage1(
    ft: np.ndarray,
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
    """Viterbi pair-mean trajectories; returns updated tracks + per-pair c(t).

    ``ft`` is the trajectory frame-time grid (the script passed its
    ``Prepared`` here and used only ``prep.ft``).
    """
    deltas = np.arange(-VIT_DELTA, VIT_DELTA + VIT_DSTEP / 2, VIT_DSTEP)
    ks = np.arange(1, 31)
    r_new = r0.copy()
    c_trajs = []
    for pi, pair in enumerate(pairs):
        r_pair = np.stack([r0[i] for i in pair])
        centers, surface = pair_surface(
            lm, bin_hz, st, ft, r_pair, deltas, ks, win_s=win_s, hop_s=hop_s
        )
        gamma = gamma_mult * surface_contrast(surface)
        ridge = viterbi_ridge(surface, deltas, gamma)
        dc_ft = np.interp(ft, centers, ridge)
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


def tooth_cube(
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


def pair_score_2d_spatial(
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


def joint_viterbi(s2: np.ndarray, gamma: float) -> tuple[np.ndarray, np.ndarray]:
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


def whitened_logmag_multi(
    audio: np.ndarray, fs: float, cfg: SeedConfig | None = None
) -> tuple[np.ndarray, float, np.ndarray]:
    """Per-channel whitened log-mag ``(C, F, N)`` + ``(bin_hz, frame_times)``.

    The per-channel sibling of :func:`tracking.vk_blind_seeding.whitened_logmag`
    (which channel-averages) — the spatial DP stage mixes channels per rotor.
    """
    white, _, bin_hz, st = logmag_spectrogram(audio, float(fs), cfg or SEED_CFG)
    return white, bin_hz, st


def apply_guard(
    label: str,
    r_prev: np.ndarray,
    r_new: np.ndarray,
    white: np.ndarray,
    bin_hz: float,
    spec_times: np.ndarray,
    ft: np.ndarray,
    cfg: SeedConfig | None = None,
    *,
    enabled: bool = True,
    log: dict[str, Any] | None = None,
) -> np.ndarray:
    """Blind per-track stage guard: revert the tracks a stage damaged.

    Runs :func:`tracking.vk_blind_seeding.stage_guard` on the before/after
    trajectories against the whitened spectrogram ``white`` and returns the
    guarded trajectories — a track that a stage re-captured onto an occupied
    comb, or whose comb confidence collapsed, reverts to ``r_prev``. When
    ``log`` is given, the reverted indices are recorded under
    ``guard_reverted_<label>``. ``enabled=False`` returns ``r_new`` unchanged
    (the validated guard-less ladder default).
    """
    if not enabled:
        return r_new
    guarded, reverted, gdiag = _stage_guard_fn(
        r_prev, r_new, white, bin_hz, spec_times, ft, cfg or SEED_CFG
    )
    if log is not None:
        log[f"guard_reverted_{label}"] = np.array(reverted, dtype=np.int64)
    if reverted:
        print(f"[stage_guard | {label}] reverted {gdiag['reasons']}", flush=True)
    return guarded


def vit2dsp_pipeline(
    prep: LadderInput,
    r0: np.ndarray,
    weights: np.ndarray,
    phys_map: np.ndarray,
    midband_cfg: VKConfig | None = None,
    refine_cfg: VKConfig | None = None,
    stage_guard: bool = False,
    sr: float = float(SR),
) -> tuple[list[tuple[str, np.ndarray]], VKResult, dict[str, Any], float, float]:
    """The spatial-DP ladder from an arbitrary 4-track blind init.

    Viterbi pair-mean c(t) -> SPATIAL joint 2-rotor Viterbi (per-rotor
    1/d^2 mic mixes) -> midband (bw 6) -> refine. Extracted from the
    validated ``run_vit2dsp`` worker (blindvit2dsp, DREGON pooled err_sm
    0.688) so other callers (the §7 blind-seeding sweep) can compose their
    own seeding with this ladder. ``weights``: (n_mics, 4) per-rotor mic
    weights; ``phys_map``: (4,) track -> physical rotor (weights column).
    ``stage_guard=True`` applies the blind per-track guard
    (``vk_blind_seeding.stage_guard``) after every stage: a track that a
    stage re-captured onto an occupied comb, or whose comb confidence
    collapsed, is reverted to its pre-stage trajectory (the r4 FLY124
    failure: viterbi_c tracked all four rotors at pooled 1.03, then the
    joint-DP pulled the weak 82.4 track onto the 91 comb). Default False =
    the validated guard-less behaviour.
    Returns ``(stages, final VKResult, extras, wall_scan_s, wall_vk_s)``.
    """
    lm_avg, bin_hz, st = whitened_logmag(prep.audio, float(sr), SEED_CFG)
    lm_multi, _, _ = whitened_logmag_multi(prep.audio, float(sr), SEED_CFG)
    ks = np.arange(1, 31)
    deltas = np.arange(-VIT2D_DELTA, VIT2D_DELTA + VIT2D_STEP / 2, VIT2D_STEP)
    r_cur = r0.copy()
    order = np.argsort(r_cur.mean(axis=1))
    pairs = [(int(order[0]), int(order[1])), (int(order[2]), int(order[3]))]

    stages: list[tuple[str, np.ndarray]] = [("init", r_cur.copy())]
    guard_log: dict[str, Any] = {}

    def _guard(label: str, r_prev: np.ndarray, r_new: np.ndarray) -> np.ndarray:
        return apply_guard(
            label,
            r_prev,
            r_new,
            lm_avg,
            bin_hz,
            st,
            prep.ft,
            SEED_CFG,
            enabled=stage_guard,
            log=guard_log,
        )

    tic = time.perf_counter()
    r_prev = r_cur.copy()
    r_cur, c_trajs = vit_stage1(prep.ft, r_cur, pairs, lm_avg, bin_hz, st, VIT_GAMMA_MULT)
    r_cur = _guard("viterbi_c", r_prev, r_cur)
    stages.append(("viterbi_c", r_cur.copy()))
    extras: dict[str, Any] = {
        "vit2d_deltas": deltas,
        "mic_weights": weights,
        "phys_map": phys_map,
        "pairs": np.array(pairs),
    }
    # Pair means from the (possibly guard-reverted) stage-1 output — equals
    # vit_stage1's own c_trajs when no track was reverted.
    c_trajs = [r_cur[list(pair)].mean(axis=0) for pair in pairs]
    r_prev = r_cur.copy()
    for pi, pair in enumerate(pairs):
        rot_a, rot_b = int(phys_map[pair[0]]), int(phys_map[pair[1]])
        lm_a = np.tensordot(weights[:, rot_a], lm_multi, axes=(0, 0))  # (F, N)
        lm_b = np.tensordot(weights[:, rot_b], lm_multi, axes=(0, 0))
        centers, cube_a = tooth_cube(lm_a, bin_hz, st, prep.ft, c_trajs[pi], deltas, ks)
        _, cube_b = tooth_cube(lm_b, bin_hz, st, prep.ft, c_trajs[pi], deltas, ks)
        s2 = np.stack(
            [
                pair_score_2d_spatial(cube_a[w], cube_b[w], ks, bin_hz)
                for w in range(cube_a.shape[0])
            ]
        )
        flat = s2.reshape(s2.shape[0], -1)
        contrast = float(np.median(np.max(flat, axis=1)) - np.median(np.median(flat, axis=1)))
        d1_idx, d2_idx = joint_viterbi(s2, VIT_GAMMA_MULT * contrast)
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
    r_cur = _guard("vit2dsp", r_prev, r_cur)
    stages.append(("vit2dsp", r_cur.copy()))
    wall_scan = time.perf_counter() - tic

    tic = time.perf_counter()
    mid = vk_track(prep.audio, r_cur, prep.ft, midband_cfg or MIDBAND_CFGS[0])
    r_mid = _guard("midband_bw6", r_cur, mid.r_refined)
    stages.append(("midband_bw6", r_mid.copy()))
    ref = vk_track(prep.audio, r_mid, prep.ft, refine_cfg or REFINE_CFG)
    r_ref = _guard("refine", r_mid, ref.r_refined)
    stages.append(("refine", r_ref.copy()))
    wall_vk = time.perf_counter() - tic
    extras.update(guard_log)
    return stages, ref, extras, wall_scan, wall_vk


# ---------------------------------------------------------------------------
# stage adapter


def vit2dsp_stage(
    *,
    weights: np.ndarray | None = None,
    phys_map: np.ndarray | None = None,
    midband_cfg: VKConfig | None = None,
    refine_cfg: VKConfig | None = None,
    stage_guard: bool = False,
    seed_cfg: SeedConfig | None = None,
    hop_s: float = DEFAULT_HOP_S,
    name: str = "vit2dsp",
) -> Stage:
    """The vit2dsp ladder as a ``tracking.stages`` Stage.

    Runs :func:`vit2dsp_pipeline` on the frame's audio. The pipeline seeds
    itself: when the frame has no ``"rps"`` entry, the ladder init comes from
    :func:`tracking.stages.blind_seed_stage` (4 rotors, ``seed_cfg`` —
    default :data:`SEED_CFG` — grid hop ``hop_s``), which appends its own
    ``<name>_seed`` log entry; an existing ``"rps"`` entry (4 tracks) is used
    as the init instead.

    ``weights`` is the (n_mics, 4) per-rotor mic-weight matrix and
    ``phys_map`` the (4,) track -> physical-rotor map of the pipeline. Blind
    defaults: uniform weights (every rotor hears the plain channel mean — the
    spatial contrast degenerates to the non-spatial union score) and the
    identity map. ``midband_cfg`` / ``refine_cfg`` default to the frozen
    ``MIDBAND_CFGS[0]`` / ``REFINE_CFG`` (the validated ladder); pass reduced
    overrides explicitly instead of editing the frozen values.

    The ladder runs as one bespoke unit rather than as composed
    ``vk_stage`` / ``guarded`` sub-stages: the validated pipeline fixes the
    twin pairs at init and shares one whitened spectrogram across all stages
    (per-stage recomputation would change both the cost and — via mid-ladder
    pair re-assignment — the science).
    """

    def run(frame):
        f = frame
        if "rps" not in f:
            f = blind_seed_stage(
                LADDER_N_ROTORS, seed_cfg or SEED_CFG, hop_s=hop_s, name=f"{name}_seed"
            )(f)
        audio, sr_f = get_audio(f)
        r0, times = get_rps(f)
        if r0.shape[0] != LADDER_N_ROTORS:
            raise ValueError(
                f"vit2dsp is a {LADDER_N_ROTORS}-track (two twin pairs) ladder, "
                f"got {r0.shape[0]} rps tracks"
            )
        mcfg = midband_cfg or MIDBAND_CFGS[0]
        rcfg = refine_cfg or REFINE_CFG
        for label, cfg in (("midband_cfg", mcfg), ("refine_cfg", rcfg)):
            if abs(cfg.fs - sr_f) > 1e-6:
                raise ValueError(f"{label}.fs={cfg.fs} does not match the frame audio rate {sr_f}")
        t0 = float(f["audio"].t_start)
        seg = _Segment(audio=audio, ft=times - t0)
        n_mics = audio.shape[0]
        w = (
            np.asarray(weights, dtype=np.float64)
            if weights is not None
            else np.full((n_mics, LADDER_N_ROTORS), 1.0 / n_mics)
        )
        pm = np.asarray(phys_map, dtype=int) if phys_map is not None else np.arange(LADDER_N_ROTORS)
        stage_snaps, ref, extras, wall_scan, wall_vk = vit2dsp_pipeline(
            seg,
            r0,
            w,
            pm,
            midband_cfg=mcfg,
            refine_cfg=rcfg,
            stage_guard=stage_guard,
            sr=sr_f,
        )
        conf = ref.confidence
        info: dict[str, Any] = {
            "stages": [lb for lb, _ in stage_snaps],
            "confidence_mean": float(conf.mean()) if conf.size else float("nan"),
            "residual_ratios": [float(v) for v in ref.residual_ratios],
            "guard_reverted": {
                k[len("guard_reverted_") :]: [int(v) for v in np.asarray(arr).ravel()]
                for k, arr in extras.items()
                if k.startswith("guard_reverted_")
            },
            "wall_scan_s": float(wall_scan),
            "wall_vk_s": float(wall_vk),
        }
        return with_rps(f, stage_snaps[-1][1], times, stage=name, info=info)

    return run


# ---------------------------------------------------------------------------
# the flagship peeled alternation


def make_peels(
    clip: np.ndarray,
    r_ft: np.ndarray,
    ft: np.ndarray,
    sr: float,
    peel_mode: str = DEFAULT_PEEL_MODE,
    *,
    n_rotors: int = LADDER_N_ROTORS,
    bw_hz: float = PEEL_BW_HZ,
    k_max: int = PEEL_K_MAX,
) -> tuple[dict[int, np.ndarray], dict[tuple[int, int], np.ndarray], dict[str, Any]]:
    """Return ``(peel_audio, pair_audio, diag)`` for one alternation step.

    ``peel_audio[i]`` = the audio minus the OTHER rotors' coherent comb
    reconstructions; ``pair_audio[(lo, hi)]`` = the audio minus the NON-pair
    rotors' reconstructions (for the joint twin observations). ``diag``
    carries the energy bookkeeping for the peel sanity gate. Both mappings
    go straight into :func:`tracking.pi_kalman_refine`'s peel seam.

    With ``peel_mode="ls"`` the envelopes are first re-projected onto the clip
    (per harmonic, per 0.25 s block, per channel), so what is subtracted is the
    least-squares fit of each modelled harmonic to the audio rather than the
    VK solve's own amplitude and phase. Which reconstruction goes into which
    peel is UNCHANGED — in particular a rotor never sees its own comb
    subtracted, and twins are only ever peeled of the non-pair rotors, because
    a sibling's fit tracks the target itself.
    """
    if peel_mode not in PEEL_MODES:
        raise ValueError(f"unknown peel_mode {peel_mode!r}; valid: {list(PEEL_MODES)}")
    cfg = VKConfig(fs=float(sr), bw_hz=bw_hz, k_max=k_max, f_max=6000.0, n_outer=1)
    t_aud = np.arange(clip.shape[-1]) / sr
    r_aud = np.vstack([np.interp(t_aud, ft, r_ft[r]) for r in range(n_rotors)])
    env = vk_envelopes(clip, r_aud, cfg)
    ls_diag: dict[str, Any] | None = None
    if peel_mode == "ls":
        env, ls_diag = ls_project_envelopes(clip, env)
    n_t = clip.shape[-1]
    recon: dict[int, np.ndarray] = {}
    for rot in range(n_rotors):
        x_mask = env.x.copy()
        x_mask[:, env.rotor != rot, :] = 0.0
        recon[rot] = vk_reconstruct(dataclasses.replace(env, x=x_mask), n_samples=n_t)
    e_audio = float(np.mean(clip**2))
    peel_audio: dict[int, np.ndarray] = {}
    diag: dict[str, Any] = {
        "bw_hz": bw_hz,
        "mode": peel_mode,
        "e_audio": e_audio,
        "per_rotor": [],
        **({"ls": ls_diag} if ls_diag is not None else {}),
    }
    for rot in range(n_rotors):
        others = sum(recon[j] for j in range(n_rotors) if j != rot)
        peeled = clip - others
        peel_audio[rot] = peeled.astype(np.float32)
        diag["per_rotor"].append(
            {
                "rotor": rot,
                "e_removed_frac": round(float(np.mean(others**2)) / e_audio, 5),
                "e_resid_ratio": round(float(np.mean(peeled**2)) / e_audio, 5),
            }
        )
    resid_all = clip - sum(recon[j] for j in range(n_rotors))
    diag["e_resid_all_ratio"] = round(float(np.mean(resid_all**2)) / e_audio, 5)
    diag["recon_energy_frac"] = [
        round(float(np.mean(recon[j] ** 2)) / e_audio, 5) for j in range(n_rotors)
    ]
    # The gate: a correctly-phased peel removes energy. A residual above the
    # window energy (or a per-rotor residual above it) means the peel is
    # mis-phased and would INJECT interference — flag, never average over.
    # Under peel_mode="ls" each harmonic on its own cannot inject; a SUM of
    # independently-fitted harmonics still can, so the gate stays.
    diag["energy_ok"] = bool(
        diag["e_resid_all_ratio"] < 1.0 and all(d["e_resid_ratio"] < 1.0 for d in diag["per_rotor"])
    )
    pair_audio: dict[tuple[int, int], np.ndarray] = {}
    for lo in range(n_rotors):
        for hi in range(n_rotors):
            if lo == hi:
                continue
            nonpair = sum(
                (recon[j] for j in range(n_rotors) if j not in (lo, hi)),
                np.zeros_like(clip),
            )
            pair_audio[(lo, hi)] = (clip - nonpair).astype(np.float32)
    return peel_audio, pair_audio, diag


def pi_kalman_arm_stage(
    *,
    peel: bool = True,
    peel_mode: str = DEFAULT_PEEL_MODE,
    peel_bw_hz: float = PEEL_BW_HZ,
    peel_k_max: int = PEEL_K_MAX,
    n_rotors: int = LADDER_N_ROTORS,
    name: str | None = None,
    **pi_kwargs: Any,
) -> Stage:
    """ONE application of the alternation as a Stage.

    ``peel=True`` (the flagship ``peeled`` arm) runs :func:`make_peels` at the
    frame's current trajectories and hands the per-rotor / per-pair residuals
    to :func:`tracking.pi_kalman_refine` through its peel seam; ``peel=False``
    is the ``naive`` arm — the same pass on the unmodified clip. Both log the
    same entry shape: peel/pi wall times, the per-rotor step statistics, the
    peel diagnostics (peeled arm only) and ``band_b0_final`` when the variant
    anneals its trust region. ``pi_kwargs`` go to the core.

    ``peel_bw_hz`` / ``peel_k_max`` are the peel's own geometry, defaulting to
    the frozen flagship values :data:`PEEL_BW_HZ` / :data:`PEEL_K_MAX`. They are
    exposed because the peel's cost is independent of the tracker's harmonic
    cap, so a caller running a short or cheap window (``tracking.telemetry_refit``,
    its tests) must be able to shrink it without touching the frozen constants.
    """

    def run(frame: td.Frame) -> td.Frame:
        audio, sr = get_audio(frame)
        r_cur, times = get_rps(frame)
        t0 = float(frame["audio"].t_start)
        clip = np.asarray(audio, dtype=np.float64)
        ft = times - t0
        peel_diag: dict[str, Any] | None = None
        seam: dict[str, Any] = {}
        tic = time.perf_counter()
        if peel:
            peel_audio, pair_audio, peel_diag = make_peels(
                clip,
                r_cur,
                ft,
                sr,
                peel_mode,
                n_rotors=n_rotors,
                bw_hz=peel_bw_hz,
                k_max=peel_k_max,
            )
            seam = {"peel_audio": peel_audio, "pair_audio": pair_audio}
        wall_peel = time.perf_counter() - tic
        tic = time.perf_counter()
        r_next, pi_diag = pi_kalman_refine(clip, r_cur, ft, sr=int(round(sr)), **seam, **pi_kwargs)
        wall_pi = time.perf_counter() - tic
        step = r_next - r_cur
        b0_final = pi_diag.get("band_b0_final")
        info: dict[str, Any] = {
            "wall_peel_s": round(wall_peel, 1),
            "wall_pi_s": round(wall_pi, 1),
            "step_rms": [round(float(np.sqrt(np.mean(step[r] ** 2))), 4) for r in range(len(step))],
            "step_mean": [round(float(np.mean(step[r])), 4) for r in range(len(step))],
            **({"band_b0_final": b0_final} if b0_final is not None else {}),
            **({"peel": peel_diag} if peel_diag is not None else {}),
        }
        return with_rps(
            frame, r_next, times, stage=name or ("peeled" if peel else "naive"), info=info
        )

    return run


def peel_alternation(
    frame: td.Frame,
    n_apps: int,
    *,
    arm: str = "peeled",
    peel_mode: str = DEFAULT_PEEL_MODE,
    pi_variant: str = "protocol",
    band_b0: float | None = None,
    n_rotors: int = LADDER_N_ROTORS,
    tag: str = "",
    verbose: bool = True,
) -> list[td.Frame]:
    """Iterate :func:`pi_kalman_arm_stage` ``n_apps`` times from ``frame``.

    Returns the ``n_apps + 1`` frames of the alternation, ``[0]`` being the
    input (the init) — so a caller reads trajectory ``i`` with
    :func:`tracking.get_rps` and application ``i``'s diagnostics off the last
    ``meta["tracking"]`` entry. ``pi_variant`` selects a :data:`PI_VARIANTS`
    row; annealed variants carry the per-rotor ``band_b0`` posterior across
    applications (that carry is the reason this is a driver and not a plain
    :func:`tracking.pipeline` composition). ``band_b0`` overrides the initial
    k-scaled band scale (rev/s) of that row.
    """
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; valid: {list(ARMS)}")
    if pi_variant not in PI_VARIANTS:
        raise KeyError(f"unknown pi_variant {pi_variant!r}; known: {sorted(PI_VARIANTS)}")
    pi_kwargs = dict(PI_VARIANTS[pi_variant])
    if band_b0 is not None:
        pi_kwargs["band_b0"] = float(band_b0)
    frames = [frame]
    for app in range(1, n_apps + 1):
        stage = pi_kalman_arm_stage(
            peel=arm == "peeled",
            peel_mode=peel_mode,
            n_rotors=n_rotors,
            n_iter=PI_N_ITER,
            pair_mode=PI_PAIR_MODE,
            band_hz=PI_BAND_HZ,
            **pi_kwargs,
        )
        frames.append(stage(frames[-1]))
        info = frames[-1]["meta"]["tracking"][-1]
        b0_final = info.get("band_b0_final")
        if pi_kwargs.get("band_anneal") == "posterior" and b0_final is not None:
            pi_kwargs["band_b0"] = tuple(b0_final)  # trust region carries over
        if verbose:
            peel_diag = info.get("peel")
            print(
                f"  [{tag}/{arm}] app {app}: peel {info['wall_peel_s']:.0f}s "
                f"pi {info['wall_pi_s']:.0f}s"
                + (
                    f" resid_all {peel_diag['e_resid_all_ratio']:.3f} ok={peel_diag['energy_ok']}"
                    if peel_diag
                    else ""
                ),
                flush=True,
            )
    return frames
