"""The ladder ARRAY CORES and the frozen config registry.

Everything here computes; nothing here is a Stage. The stages that wire these
cores into a ``td.Frame -> td.Frame`` ladder, and the named recipes that
compose them, live in :mod:`tracking.top` — read that module first.

Three cores live here, all calibrated and all frozen:

1. The blind RPS-annotation ladder that ``scripts/vk_blind_annotation.py``
   validated (DREGON pooled err_sm 0.688, condition ``blindvit2dsp``) —
   :func:`vit2dsp_pipeline` and its helpers :func:`vit_stage1`,
   :func:`tooth_cube`, :func:`pair_score_2d_spatial`, :func:`joint_viterbi`,
   :func:`apply_guard`::

       blind init -> Viterbi pair-mean c(t) -> SPATIAL joint 2-rotor Viterbi
       (per-rotor mic mixes) -> mid-band VK (bw 6) -> VK refine

2. The ``blind_fullrange`` coarse pass (:func:`coarse_init` and its
   :class:`CoarseConfig`) — the BPF octave check plus the full-range
   frame-rate Viterbi with the energy-timed takeoff bridge, which puts a
   takeoff or warmup ramp inside the ladder's reachable state space.

3. The FLAGSHIP peel (``docs/experiments/beat-vk.md``) — :func:`make_peels`:
   VK envelopes at the current track, a per-harmonic least-squares re-fit,
   then subtract the OTHER rotors' combs.

Every comb reading in this module goes through ONE tooth sampler,
:func:`comb_teeth`.

FROZEN CONFIG REGISTRY: ``CAPTURE_CFG`` / ``REFINE_CFG`` / ``TRACK_CFG`` /
``MIDBAND_CFG`` / ``MIDBAND_CFGS`` / ``SEED_CFG``, the ladder constants and
:class:`CoarseConfig`'s defaults are the calibrated blind-annotation ladder
configs. Changing any value invalidates the published annotations (and every
number derived from them) — treat them as data, not knobs.
``tests/tracking/test_pipelines.py`` spot-checks the values against the
published calibration.

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
    "CoarseConfig",
    "Segment",
    "apply_guard",
    "bpf_octave_ratio",
    "coarse_frame_scores",
    "coarse_init",
    "coarse_spectrogram",
    "comb_teeth",
    "energy_bridge",
    "joint_viterbi",
    "local_comb_frame_scores",
    "make_peels",
    "pair_score_2d_spatial",
    "pair_surface",
    "surface_contrast",
    "tooth_cube",
    "vit2dsp_pipeline",
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
class Segment:
    """Minimal :class:`LadderInput` — one window's audio and frame grid.

    What the frame stages of :mod:`tracking.top` hand to these cores.
    """

    audio: np.ndarray
    ft: np.ndarray


# ---------------------------------------------------------------------------
# ladder core (moved verbatim from scripts/vk_blind_annotation.py)


#: Top of every comb reading: above this the whitened lines carry no usable
#: rotor evidence at the resolutions the ladders run at.
COMB_F_MAX = 6000.0
#: Bottom of the ladder's comb reading — the seed scan's floor.
COMB_F_MIN = 60.0


def comb_teeth(
    lm: np.ndarray,
    bin_hz: float,
    f: np.ndarray,
    *,
    f_min: float = COMB_F_MIN,
    f_max: float = COMB_F_MAX,
    pos_only: bool = False,
) -> np.ndarray:
    """``(M, N)`` interpolated spectrogram value at each tooth frequency.

    THE tooth sampler of the ladders — every comb reading in this module goes
    through it, so "the value on a tooth" means one thing.

    ``lm`` is ``(F, N)``, ``f`` is either ``(M,)`` (teeth that do not move over
    the window — the coarse pass's constant-c templates) or ``(M, N)`` (one
    frequency per tooth per frame). The value is linearly interpolated between
    the two neighbouring bins. A tooth outside ``[f_min, f_max]`` reads
    ``NaN``, so a caller reduces with ``np.nanmean`` and never has to build a
    per-delta mask of its own. ``pos_only`` clips each value at zero BEFORE
    the caller's reduction (the half-tooth null of the coarse contrast: a
    whitening dip must not be counted as evidence AGAINST a comb).
    """
    n_f, n = lm.shape
    fmax = min(f_max, (n_f - 1) * bin_hz)
    fa = np.asarray(f, dtype=np.float64)
    valid = (fa >= f_min) & (fa <= fmax)
    idx = np.clip(fa, 0.0, fmax) / bin_hz
    j = np.floor(idx).astype(int)
    frac = idx - j
    hi = np.minimum(j + 1, n_f - 1)
    if fa.ndim == 1:
        v = (1 - frac)[:, None] * lm[j] + frac[:, None] * lm[hi]
        valid = valid[:, None]
    else:
        cols = np.arange(n)[None, :]
        v = (1 - frac) * lm[j, cols] + frac * lm[hi, cols]
    if pos_only:
        v = np.maximum(v, 0.0)
    return np.where(valid, v, np.nan)


def local_comb_frame_scores(
    lm: np.ndarray, bin_hz: float, r_spec: np.ndarray, deltas: np.ndarray, ks: np.ndarray
) -> np.ndarray:
    """``(D, N)`` per-frame mean log-mag along the combs of ``r_spec + delta``.

    ``r_spec`` may be ``(N,)`` (one rotor) or ``(P, N)`` (a rigid multi-rotor
    template — e.g. a twin pair shifted together, separations frozen); the
    comb is then the union of all P rotors' teeth.
    """
    r2 = np.atleast_2d(r_spec)  # (P, N)
    n = lm.shape[1]
    out = np.empty((len(deltas), n))
    for di, d in enumerate(deltas):
        f = (ks[:, None, None] * (r2 + d)[None, :, :]).reshape(-1, n)  # (K*P, N)
        out[di] = np.nanmean(comb_teeth(lm, bin_hz, f), axis=0)
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
    n = lm.shape[1]
    c_spec = np.interp(st, ft, c_traj)
    vals = np.empty((len(ks), len(deltas), n))
    for di, d in enumerate(deltas):
        f = ks[:, None] * (c_spec + d)[None, :]  # (K, N)
        vals[:, di, :] = comb_teeth(lm, bin_hz, f)
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
# blind_fullrange: the coarse full-range pass (ramp-following, octave-corrected)
#
# Two diagnosed failure modes of the blind arms on the frozen protocol:
#
# 1. RAMP WINDOWS (DREGON w0s: takeoff ~15 -> 80 rev/s inside the 16 s
#    window). The seeder's hypothesis class is constant bases from the
#    TIME-AVERAGED spectrum (cruise-dominated) and the ladder's Viterbi
#    stages only track +-6 rev/s around them (VIT2D_DELTA) — the ramp is
#    outside the tracked state space entirely (blind_KR MAE 15.4-23.1).
# 2. FLY124 WARMUP windows (true shaft ~31-41 rev/s). The warmup spectrum
#    contains ONLY even shaft harmonics (blade-pass lines of the 2-blade
#    props: 62.5/72.3/82 Hz + their multiples; measured, zero odd-line
#    energy), so the scan's octave-up promotion commits to the 2x bases and
#    MAE rails at 33-36. Pure comb evidence CANNOT resolve this octave; the
#    discriminator is physical: for a candidate base b, if the line AT b is
#    STRONGER than the line at 2b, then b is itself the blade-pass comb and
#    the shaft is b/2 (at cruise the BPF line 2b dominates the shaft line b:
#    measured ratios v(b)/v(2b) <= 0.93 on every cruise window vs >= 1.87 on
#    every warmup window; threshold 1.4).
#
# blind_fullrange therefore prepends to blind_KR:
#   (a) the BPF octave check above (:func:`bpf_octave_ratio`) — median ratio
#       over unique seed bases >= ``halve_ratio`` halves ALL bases (and drops
#       the K-gate, which was calibrated on the rejected bases);
#   (b) a coarse slope-tolerant Viterbi c(t) over an fft2048 whitened
#       spectrogram at the native 32 ms frame rate (window-averaged surfaces
#       smear a 30 rev/s-per-second ramp into invisibility; at 2048/32 ms the
#       k<=8 comb sweep is ~1 bin per frame), scoring the RIGID additive
#       union template r0(c) = c + (bases - median(bases)) with a
#       positive-half-tooth contrast (:func:`coarse_frame_scores`: on-teeth
#       mean minus max(0, .) mean at (k-0.5) teeth — penalizes sub-multiple
#       aliases without the whitening-dip artifact that a signed contrast
#       has), per-frame soft-normalized so weak-evidence ramp frames still
#       express their preference. Grid: full 12-120 rev/s (floor 12 excludes
#       the low-c GCD-alias zone where k<=8 teeth all fall into LF rumble) —
#       or, for HALVED windows, restricted to median +- 16 rev/s: in the
#       BPF-only regime full-range magnitude evidence is structurally
#       octave-attracted, and +-16 still covers the warmup ramps;
#   (c) TWO TRUST GATES on the DP path (the first full 15-window run showed
#       the coarse DP must not override a good constant seed): a STEADY gate
#       — path span (p98 - p2) < ``span_min`` means there is no ramp to
#       track, use the exact blind_KR constant init (removes coarse wobble
#       on every steady window); and a DISTRUST gate — |median(path) -
#       median(bases)| > ``med_shift_max`` means the DP abandoned the
#       seed structure (FLY124 w3/w4: asymmetric seeds — a dup pair at 74 +
#       singles 82.7/92.35 — let the DP park the tight pair on the dominant
#       91.5 comb, shifting c by +17 and turning MAE 1.18 into 15.6; on
#       every well-behaved window the shift is <= 1.1, on the broken ones
#       16-17), fall back to the constant init;
#   (d) an ENERGY-TIMED TAKEOFF BRIDGE (:func:`energy_bridge`): through the
#       middle of a takeoff ramp the narrowband evidence vanishes under the
#       broadband spool-up whoosh (the DP times the low->high transition
#       ~1.5 s late, or idles on an alias when a masker buries the idle
#       comb), but acoustic power tracks rps steeply. When the DP path
#       contains a > 20 rev/s two-plateau jump AND the window has a >=
#       ``bridge_idle_min_s`` low-energy idle phase (the takeoff-from-idle
#       signature; without it the bridge must stay off — it mangled FLY124
#       w2's maneuver window when keyed on energy alone), the pre-cruise path
#       is rebuilt from the ``energy_band`` (50-200 Hz rotor rumble —
#       monotone in rps even under the speech / white-noise masker
#       recordings, where the first-run 2-6 kHz band was flooded) profile:
#       idle frames -> c_lo from a constant-c re-scan of the idle frames
#       restricted to <= ``bridge_idle_c_frac`` * c_hi (the DP's own low
#       plateau is junk exactly when a masker hides the idle comb),
#       transition frames -> power-law c_lo * (c_hi/c_lo)^alpha, then a
#       catch-up hold at c_hi (median DP path over sustained-high-energy
#       frames) until the DP path rejoins it.
#
# Ladder init: r0[i](t) = base_i + (coarse_c(t) - median(coarse_c)), clamped
# at 0 — anchored on the SEED bases (not on the path), so any residual
# constant DP offset cancels; gated windows reduce to blind_KR's constant
# init exactly. The standard vit2dsp ladder runs on top, unchanged.
# Measured init PIT-MAE vs raw telemetry (recorded blind_KR FINAL MAE in
# parens): nosource w0 3.45 (15.4), speech w0 2.82 (16.8), whitenoise w0
# 4.32 (23.1), FLY124 w0 3.96 (35.8), w1 1.73 (33.2), w2 5.4-class (5.36);
# every steady window gated to the exact blind_KR init.


@dataclass(frozen=True)
class CoarseConfig:
    """The coarse full-range pass, calibrated (see the block comment above).

    The one variant the campaign runs is ``blind_fullrange_2xwin``:
    ``CoarseConfig(nfft=4096, hop=1024, gamma=0.2)`` — 2x finer in frequency
    (3.9 Hz bins, so the k<=8 twin-separation threshold halves), 2x coarser in
    time. The transition penalty is HALVED rather than the per-hop allowance
    doubled: ``gamma`` is a cost per rev/s of ``|dc|`` per hop, so at a 2x hop
    the same physical ramp pays 2x ``|dc|`` per hop while contributing half as
    many evidence frames. Ramp-machinery caveat (not adapted, by design): the
    bridge's second-based thresholds adapt through the frame period
    automatically, but ``smooth_frames`` and ``energy_smooth_frames`` are
    frame counts, so their spans double.
    """

    #: 7.8 Hz bins, 0.128 s window: a 30 rev/s-per-second ramp sweeps ~1 bin
    #: per k<=8 tooth per frame (8192 smears it over ~26 bins).
    nfft: int = 2048
    hop: int = 512
    #: Transition cost per rev/s of |dc| per 32 ms frame.
    gamma: float = 0.4
    #: The full grid, rev/s. The floor excludes the low-c GCD-alias zone.
    lo: float = 12.0
    hi: float = 120.0
    step: float = 0.5
    #: Low harmonics: wide basins, coarse evidence.
    k_max: int = 8
    #: Below the seed's 60 Hz floor — keeps the k1/k2 teeth of warmup/ramp
    #: bases in band (the whitened floor is ~0 there).
    f_min: float = 20.0
    #: Light time smoothing of the per-frame node scores.
    smooth_frames: int = 3
    #: Soft floor (x global median contrast) on the per-frame
    #: (score - median) / (peak - median) normalization.
    norm_soft: float = 0.3
    #: BPF octave check threshold on median v(b) / v(2b).
    halve_ratio: float = 1.4
    #: Line-strength readout half-width around b / 2b.
    line_half_hz: float = 1.5
    #: +- grid half-range around median(bases) when halved.
    restrict: float = 16.0
    #: Halved grid only: use k up to f_top / c (band-matched tooth count).
    adaptive_f_top: float = 360.0
    adaptive_k_cap: int = 24
    #: Steady gate: a path span (p98 - p2) below this is no ramp (steady
    #: windows measure <= 6.5; the smallest true ramp, FLY124 w0's warmup
    #: spool-up, measures 10.0).
    span_min: float = 8.0
    #: Distrust gate: |median(path) - median(bases)| above this means the DP
    #: abandoned the seeds (<= 1.1 good, 16-17 broken).
    med_shift_max: float = 5.0
    #: Bridge energy band: rotor rumble — monotone in rps and the
    #: strongest-contrast band on every recording, including the speech and
    #: white-noise masker ones (2-6 kHz is flooded by the white-noise source).
    energy_band: tuple[float, float] = (50.0, 200.0)
    energy_smooth_frames: int = 11
    #: rev/s: minimum two-plateau jump to re-time.
    bridge_jump_min: float = 20.0
    #: alpha >= 0.9 must hold this long to anchor c_hi.
    bridge_sustain_s: float = 0.5
    #: Minimum low-energy (alpha <= 0.1) idle phase before the spool-up for
    #: the bridge to engage at all (the takeoff-from-idle signature).
    bridge_idle_min_s: float = 1.0
    #: Idle re-scan restricted to c <= this * c_hi (excludes the c_hi/2
    #: sub-multiple attractor).
    bridge_idle_c_frac: float = 0.45
    #: rev/s: catch-up hold until the DP path is this close.
    bridge_rejoin_tol: float = 5.0
    #: Minimum log-energy gap between plateaus to trust.
    bridge_min_contrast: float = 0.5
    #: Frame period fallback when the spectrogram has a single frame.
    frame_s: float = 0.032


def coarse_spectrogram(
    audio: np.ndarray, cfg: CoarseConfig | None = None, sr: float = float(SR)
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """Short-FFT spectrogram for the coarse pass.

    Returns ``(whitened (F, N), bin_hz, frame_times (N,), energy (N,))`` —
    channel-mean whitened log-mag (running-median-over-frequency subtracted,
    the same whitening as the seed scan but at ``cfg.nfft``) plus the
    channel-mean RAW log-mag averaged over ``cfg.energy_band``, which is the
    bridge's timing signal.
    """
    use = cfg or CoarseConfig()
    white, raw, bin_hz, st = logmag_spectrogram(
        audio, float(sr), SEED_CFG, n_fft=use.nfft, hop_length=use.hop
    )
    freqs = np.arange(white.shape[1]) * bin_hz
    band = (freqs >= use.energy_band[0]) & (freqs <= use.energy_band[1])
    energy = raw.mean(axis=0)[band].mean(axis=0)
    return white.mean(axis=0), bin_hz, st, energy


def bpf_octave_ratio(
    audio: np.ndarray, bases: np.ndarray, cfg: CoarseConfig | None = None, sr: float = float(SR)
) -> float:
    """Median over unique seed bases of the line strength ``v(b) / v(2b)``.

    Lines are read off the 8192-FFT whitened time mean (the seed scan's
    resolution — warmup lines are narrow and 2048 washes them out), taking the
    maximum within ``+-cfg.line_half_hz``. A ratio at or above
    ``cfg.halve_ratio`` says the seed committed to the blade-pass comb and the
    shaft is half of it — see the block comment above for the physics.
    """
    use = cfg or CoarseConfig()
    lm8, bin8, _ = whitened_logmag(audio, float(sr), SEED_CFG)
    vec = lm8.mean(axis=1)

    def line(f: float) -> float:
        lo = max(0, int(np.floor((f - use.line_half_hz) / bin8)))
        hi = min(len(vec) - 1, int(np.ceil((f + use.line_half_hz) / bin8)))
        return float(vec[lo : hi + 1].max())

    uniq: list[float] = []
    for b in np.sort(np.asarray(bases, dtype=np.float64)):
        if all(abs(float(b) - u) > 1.0 for u in uniq):
            uniq.append(float(b))
    return float(np.median([line(b) / max(line(2.0 * b), 1e-6) for b in uniq]))


def coarse_frame_scores(
    lm: np.ndarray,
    bin_hz: float,
    offsets: np.ndarray,
    c_grid: np.ndarray,
    adaptive_k: bool,
    cfg: CoarseConfig | None = None,
) -> np.ndarray:
    """``(D, N)`` per-frame union-comb CONTRAST of the template ``c + offsets``.

    The sibling of :func:`local_comb_frame_scores`, and different from it in
    three ways that the coarse pass needs: the template is constant over the
    window (``c`` is the scanned state, not a trajectory), the floor is
    ``cfg.f_min`` rather than the seed's 60 Hz, and the score is a CONTRAST —
    the mean whitened value on the teeth ``k (c + offset)`` minus the mean
    POSITIVE value on the half-teeth ``(k - 0.5) (c + offset)``, which
    penalizes a sub-multiple alias. ``adaptive_k`` (the halved/restricted grid
    only) uses ``k`` up to ``cfg.adaptive_f_top / c``, so every ``c`` is scored
    on a comparable band.
    """
    use = cfg or CoarseConfig()
    n = lm.shape[1]

    def comb_mean(freqs: np.ndarray, pos_only: bool) -> np.ndarray:
        v = comb_teeth(lm, bin_hz, freqs, f_min=use.f_min, pos_only=pos_only)
        if not np.any(np.isfinite(v)):
            return np.zeros(n)
        return np.nanmean(v, axis=0)

    out = np.empty((len(c_grid), n))
    off_arr = np.asarray(offsets, dtype=np.float64)
    for ci, c in enumerate(c_grid):
        r = float(c) + off_arr
        k_max = use.k_max
        if adaptive_k:
            k_max = int(
                np.clip(
                    np.floor(use.adaptive_f_top / max(float(c), 1.0)),
                    use.k_max,
                    use.adaptive_k_cap,
                )
            )
        ks = np.arange(1, k_max + 1, dtype=np.float64)
        out[ci] = comb_mean((ks[:, None] * r[None, :]).ravel(), False) - comb_mean(
            ((ks - 0.5)[:, None] * r[None, :]).ravel(), True
        )
    return out


def energy_bridge(
    path: np.ndarray,
    fsc: np.ndarray,
    c_grid: np.ndarray,
    energy: np.ndarray,
    frame_s: float,
    cfg: CoarseConfig | None = None,
) -> tuple[np.ndarray, str]:
    """Rebuild the pre-cruise part of a takeoff window from the energy profile.

    See item (d) of the block comment: this requires a two-plateau DP path
    jumping more than ``cfg.bridge_jump_min``, usable energy contrast, a
    sustained high-energy (cruise) run AND an idle phase of at least
    ``cfg.bridge_idle_min_s``. Then idle frames get ``c_lo`` from a restricted
    constant-c re-scan of the idle frames (the DP's own low plateau is junk
    when a masker hides the idle comb), transition frames get the power-law
    energy mapping, and ``c_hi`` is held until the DP path rejoins it. Any
    unmet requirement returns the path unchanged, with the reason.
    """
    from scipy.ndimage import median_filter

    use = cfg or CoarseConfig()
    if float(path.max() - path.min()) < use.bridge_jump_min:
        return path, "no-op"
    cmid = float(path.max() + path.min()) / 2.0
    c_lo_p = float(np.median(path[path < cmid]))
    c_hi_p = float(np.median(path[path >= cmid]))
    if c_hi_p - c_lo_p < use.bridge_jump_min:
        return path, "no-op"
    e_sm = median_filter(energy, size=use.energy_smooth_frames)
    e_lo = float(np.percentile(e_sm, 2))
    e_hi = float(np.percentile(e_sm, 90))
    if e_hi - e_lo < use.bridge_min_contrast:
        return path, "no-contrast"
    alpha = np.clip((e_sm - e_lo) / (e_hi - e_lo), 0.0, 1.0)
    n_sus = max(1, int(round(use.bridge_sustain_s / frame_s)))
    t_hi0 = None
    run = 0
    for i, m in enumerate(alpha >= 0.9):
        run = run + 1 if m else 0
        if run >= n_sus:
            t_hi0 = i - n_sus + 1
            break
    if t_hi0 is None:
        return path, "no-hi-run"
    c_hi = float(np.median(path[alpha >= 0.9]))
    idle = np.zeros(len(path), dtype=bool)
    idle[:t_hi0] = alpha[:t_hi0] <= 0.1
    if float(idle.sum()) * frame_s < use.bridge_idle_min_s:
        return path, "no-idle"
    sel = c_grid <= use.bridge_idle_c_frac * c_hi
    s_idle = fsc[:, idle].mean(axis=1)
    c_lo = float(c_grid[sel][int(np.argmax(s_idle[sel]))])
    out = path.copy()
    out[idle] = c_lo
    trans = np.zeros(len(path), dtype=bool)
    trans[:t_hi0] = (alpha[:t_hi0] > 0.1) & (alpha[:t_hi0] < 0.9)
    a_resc = np.clip((alpha - 0.1) / 0.8, 0.0, 1.0)
    out[trans] = c_lo * (c_hi / c_lo) ** a_resc[trans]
    t = t_hi0
    while t < len(path) and abs(float(path[t]) - c_hi) > use.bridge_rejoin_tol:
        out[t] = c_hi
        t += 1
    return out, (
        f"bridge hi0={t_hi0 * frame_s:.2f}s catchup->{t * frame_s:.2f}s "
        f"c_lo={c_lo:.1f} c_hi={c_hi:.1f} idle={float(idle.sum()) * frame_s:.1f}s"
    )


def coarse_init(
    prep: LadderInput,
    bases: np.ndarray,
    *,
    cfg: CoarseConfig | None = None,
    sr: float = float(SR),
) -> tuple[np.ndarray, np.ndarray, bool, dict[str, Any]]:
    """The ``blind_fullrange`` ladder init (mechanism: the block comment above).

    ``bases`` are the seed's constant bases. Returns
    ``(r0 (R, N), the effective bases, halved, coarse diagnostics)``. The
    effective bases differ from the input only when the BPF octave check
    halves them — in which case the caller must also drop the seed's auto
    update gate, since the K calibration ran on the rejected 2x bases.
    """
    use = cfg or CoarseConfig()
    bases = np.sort(np.asarray(bases, dtype=np.float64))
    ratio = bpf_octave_ratio(prep.audio, bases, use, sr)
    halved = ratio >= use.halve_ratio
    if halved:
        bases = bases / 2.0
    med = float(np.median(bases))
    offsets = bases - med
    if halved:
        lo = max(use.lo, med - use.restrict)
        hi = min(use.hi, med + use.restrict)
    else:
        lo, hi = use.lo, use.hi
    c_grid = np.arange(lo, hi + use.step / 2, use.step)

    lm2, bin2, st2, energy = coarse_spectrogram(prep.audio, use, sr)
    fsc = coarse_frame_scores(lm2, bin2, offsets, c_grid, halved, use)
    kern = np.ones(use.smooth_frames) / use.smooth_frames
    fsc = np.apply_along_axis(lambda r: np.convolve(r, kern, mode="same"), 1, fsc)
    med_f = np.median(fsc, axis=0, keepdims=True)
    peak_f = fsc.max(axis=0, keepdims=True)
    glob = float(np.median(peak_f - med_f))
    s = (fsc - med_f) / np.maximum(peak_f - med_f, use.norm_soft * glob)
    path = viterbi_lattice(s.T, c_grid, use.gamma)  # (D, N) scores -> (N, D) lattice
    frame_s = float(st2[1] - st2[0]) if len(st2) > 1 else use.frame_s

    # Trust gates (item (c)): a coarse path only overrides the constant
    # blind_KR init when it tracks a real ramp AND kept the seed structure.
    span = float(np.percentile(path, 98) - np.percentile(path, 2))
    shift = abs(float(np.median(path)) - med)
    if span < use.span_min or shift > use.med_shift_max:
        mode = "const-steady" if span < use.span_min else "const-distrust"
        coarse = np.full(len(prep.ft), med)
        r0 = np.repeat(bases[:, None], len(prep.ft), axis=1)
        bridge_info = "gated"
    else:
        mode = "coarse"
        path, bridge_info = energy_bridge(path, fsc, c_grid, energy, frame_s, use)
        coarse = np.interp(prep.ft, st2, path)
        # Anchor on the SEED bases: residual constant DP offsets cancel.
        r0 = np.maximum(bases[:, None] + (coarse - float(np.median(path)))[None, :], 0.0)
    diag = {
        "coarse_c": coarse,
        "coarse_nfft": use.nfft,
        "coarse_hop": use.hop,
        "coarse_gamma": use.gamma,
        "coarse_bpf_ratio": ratio,
        "coarse_halved": halved,
        "coarse_grid": (float(lo), float(hi)),
        "coarse_bridge": bridge_info,
        "coarse_mode": mode,
        "coarse_span": span,
        "coarse_shift": shift,
    }
    return r0, bases, halved, diag


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
