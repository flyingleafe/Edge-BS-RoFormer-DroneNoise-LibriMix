"""THE top-level tracking module: the frame plumbing, every stage, every recipe.

Read this file to know what the tracking stack can do. It holds three things
and nothing else:

1. **The plumbing** — :data:`Stage`, :func:`pipeline`, and the frame accessors
   (:func:`tracking_frame`, :func:`get_audio`, :func:`get_rps`,
   :func:`with_rps`, :func:`with_meta`).
2. **The stage vocabulary** — one small factory per stage, each with its
   config beside it. Every stage is a ``td.Frame -> td.Frame`` callable.
3. **The recipes** — every shipped variant of the algorithm, written here as a
   named composition of those stages. A driver script calls a recipe; it never
   assembles a ladder of its own.

This module WIRES. The array cores stay where they are
(:mod:`tracking.vk_tracking`, :mod:`tracking.phase_increment_tracker`,
:mod:`tracking.pipelines`, ...) and nothing here implements signal processing.

The stage vocabulary
--------------------

===========================  ==========================  ==========================
Stage                        Config                      What it does
===========================  ==========================  ==========================
:func:`blind_seed_stage`     :class:`SeedConfig`         blind comb scan -> constant ``rps`` init
:func:`coarse_init_stage`    :class:`CoarseConfig`       full-range Viterbi c(t) -> time-varying init
:func:`vit2dsp_stage`        :class:`Vit2dspConfig`      the calibrated blind-annotation ladder
:func:`vk_stage`             :class:`VKConfig`           coupled Vold-Kalman order tracking
:func:`peel_stage`           :class:`PeelConfig`         subtract the other rotors' combs -> a seam in meta
:func:`pi_kalman_stage`      :class:`PiConfig`           phase-increment Kalman refinement (eats the seam)
:func:`warp_stage`           kwargs                      iterated time-warp IF refinement
:func:`refine_coherent_stage` :class:`RefineConfig`      coherent phase-slope refinement
:func:`presmooth_stage`      ``cut_hz``                  low-pass the trajectory (5 Hz staircase removal)
:func:`scale_stage`          ``factor``                  multiply the trajectory by a constant
:func:`shift_stage`          ``tau``                     read the trajectory ``tau`` seconds later
:func:`fitness_stage`        :class:`FitnessConfig`      score the trajectory (does not change it)
:func:`fvk_stage`            :class:`FVKConfig`          score it by F_VK (profiled coupled-VK residual)
:func:`fvk_refine_stage`     :class:`FVKConfig`          L-BFGS on F_VK under a k-annealing schedule
:func:`decompose_stage`      :class:`FVKConfig`          split the audio into per-harmonic tracks + a residual
:func:`refit_stage`          :class:`RefitConfig`        the whole telemetry refit as one stage
:func:`guarded`              :class:`SeedConfig`         wrap a stage with the blind per-track guard
===========================  ==========================  ==========================

The recipes
-----------

============================  =========================================================
Recipe                        Composition
============================  =========================================================
:func:`vit2dsp`               blind seed -> the calibrated ladder
:func:`blind_fullrange`       blind seed (K, R) -> coarse full-range init -> the ladder
:func:`flagship`              ``n_apps`` x (peel -> pi_kalman)
:func:`peel_alternation`      :func:`flagship` one application at a time, all frames kept
:func:`refit_stage`           presmooth -> coarse-to-fine (peel -> pi_kalman) to convergence
:func:`judge`                 a candidate stage -> :func:`fitness_stage` under one control
============================  =========================================================

The frame contract
------------------

- ``"audio"``: ``(mic, time)`` float32 Series on a ``GridIndex`` at the audio
  sample rate. :func:`tracking_frame` accepts a mono ``(T,)`` array and stores
  it as ``(1, T)``.
- ``"rps"``: ``(rotor, time)`` float64 Series on a ``StampIndex`` at the
  trajectory frame times — the *current candidate trajectories*. A stage that
  changes the trajectory replaces this entry and appends one
  ``{"stage": name, ...}`` dict to the ``"tracking"`` list inside the
  invariant ``"meta"`` sub-Frame.
- ``"rps_meas"``: optional reference trajectories, never touched by a stage.

A stage that does NOT change the trajectory (:func:`peel_stage`) leaves a
**seam** in ``meta`` instead of a log entry, and the stage that consumes the
seam records what it consumed. That is what keeps one application of the
flagship one log entry, however it is composed.

Purity rule: this module imports only ``numpy``, ``tdseries`` and
``tracking.*`` (enforced by the ``tracking stays pure`` import-linter
contract).
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

import numpy as np
import tdseries as td

if TYPE_CHECKING:  # the decomposition core is imported lazily inside its stage
    from tracking.decompose import BandwidthSchedule

from tracking.fitness import FitnessConfig, Holdout, score_window
from tracking.fitness_vk import (
    FVKConfig,
    FVKStage,
    fvk_score,
    optimize_trajectory,
)
from tracking.phase_increment_tracker import pi_kalman_refine
from tracking.pipelines import (
    ARMS,
    DEFAULT_PEEL_MODE,
    LADDER_N_ROTORS,
    MIDBAND_CFGS,
    PEEL_BW_HZ,
    PEEL_K_MAX,
    PI_BAND_HZ,
    PI_N_ITER,
    PI_PAIR_MODE,
    PI_VARIANTS,
    REFINE_CFG,
    SEED_CFG,
    CoarseConfig,
    Segment,
    coarse_init,
    make_peels,
    vit2dsp_pipeline,
)
from tracking.rps_refinement import RefineConfig, refine_coherent
from tracking.telemetry_refit import RefitConfig, presmooth
from tracking.vk_blind_seeding import (
    SeedConfig,
    SeedResult,
    blind_seed,
    stage_guard,
    whitened_logmag,
)
from tracking.vk_tracking import VKConfig, vk_track
from tracking.warp_refinement import iter_warp_refine

__all__ = [
    "DEFAULT_HOP_S",
    "PI_PROTOCOL",
    "CoarseConfig",
    "FVKConfig",
    "FVKStage",
    "PeelConfig",
    "PiConfig",
    "Stage",
    "Vit2dspConfig",
    "blind_fullrange",
    "blind_seed_stage",
    "coarse_init_stage",
    "decompose_stage",
    "fitness_stage",
    "flagship",
    "fvk_refine_stage",
    "fvk_stage",
    "get_audio",
    "get_rps",
    "guarded",
    "judge",
    "peel_alternation",
    "peel_stage",
    "pi_kalman_arm_stage",
    "pi_kalman_stage",
    "pipeline",
    "presmooth_stage",
    "refine_coherent_stage",
    "refit_stage",
    "scale_stage",
    "shift_stage",
    "tracking_frame",
    "vit2dsp",
    "vit2dsp_stage",
    "vk_stage",
    "warp_stage",
    "with_meta",
    "with_rps",
]

#: A tracking stage: consumes a frame, returns a frame with ``"rps"``
#: replaced and one diagnostics dict appended to ``meta["tracking"]``.
Stage = Callable[[td.Frame], td.Frame]

#: Default trajectory frame hop (seconds) for stages that create the grid
#: (:func:`blind_seed_stage`) — the evaluation-grid convention of
#: ``scripts/vk_validation.py`` (``FRAME_HOP_S``, the predecessor's STFT hop).
DEFAULT_HOP_S = 0.032

#: The meta key the peel seam travels in (see :func:`peel_stage`).
PEEL_SEAM = "peel_seam"


# ---------------------------------------------------------------------------
# 1. the plumbing: frame construction, accessors, composition


def _rps_series(r: np.ndarray, frame_times: np.ndarray) -> td.Series:
    """``(R, N)`` trajectories at ``frame_times`` -> a ``(rotor, time)`` Series.

    Stored on a ``StampIndex`` (the michaels-frames events convention), which
    accepts any frame grid — uniform or not — at nanosecond-tick resolution.
    """
    r2 = np.atleast_2d(np.asarray(r, dtype=np.float64))
    ft = np.asarray(frame_times, dtype=np.float64)
    if r2.ndim != 2:
        raise ValueError(f"rps must be (R, N), got shape {np.asarray(r).shape}")
    if r2.shape[-1] != len(ft):
        raise ValueError(f"rps has {r2.shape[-1]} frames but frame_times has {len(ft)}")
    return td.events(ft, r2, dims=("rotor", "time"))


def tracking_frame(
    audio: np.ndarray,
    sr: float | int | tuple[int, int],
    *,
    rps: np.ndarray | None = None,
    frame_times: np.ndarray | None = None,
    rps_meas: np.ndarray | None = None,
    meta: Mapping[str, Any] | None = None,
    dtype: Any = np.float32,
) -> td.Frame:
    """Build the canonical tracking frame from raw arrays.

    ``audio`` is ``(T,)`` or ``(C, T)`` at ``sr`` (stored as ``dtype``
    ``(mic, time)`` — float32 by convention; pass ``np.float64`` to keep a
    float64 signal exactly, which :func:`get_audio` then returns unchanged).
    A mono input becomes ``(1, T)``. ``sr`` must be exact —
    an int, an integral float, or an ``(num, den)`` rational tuple
    (``tdseries`` rejects non-integral float rates). ``rps`` / ``rps_meas``
    are ``(R, N)`` rev/s on the ``frame_times`` grid (``frame_times`` is
    required when either is given). ``meta`` seeds the invariant ``"meta"``
    sub-Frame; stage diagnostics accumulate under its ``"tracking"`` key.
    """
    a = np.asarray(audio, dtype=dtype)
    if a.ndim == 1:
        a = a[None, :]
    if a.ndim != 2:
        raise ValueError(f"audio must be (T,) or (C, T), got shape {np.asarray(audio).shape}")
    entries: dict[str, Any] = {"audio": td.uniform(a, sr, dims=("mic", "time"))}
    if (rps is not None or rps_meas is not None) and frame_times is None:
        raise ValueError("frame_times is required when rps or rps_meas is given")
    if rps is not None and frame_times is not None:
        entries["rps"] = _rps_series(rps, frame_times)
    if rps_meas is not None and frame_times is not None:
        entries["rps_meas"] = _rps_series(rps_meas, frame_times)
    entries["meta"] = td.Frame(dict(meta or {}))
    return td.Frame(entries)


def get_audio(frame: td.Frame) -> tuple[np.ndarray, float]:
    """``frame["audio"]`` as ``((C, T) float32, sample_rate)``.

    A mono ``(time,)`` entry is returned as ``(1, T)``. A float64 entry keeps
    its precision (every core widens to float64 anyway); anything else becomes
    float32, the storage convention.
    """
    series = frame["audio"]
    idx = series.tindex
    if not isinstance(idx, td.GridIndex):
        raise TypeError(f"'audio' must be uniformly sampled, got {type(idx).__name__}")
    raw = np.asarray(series.data)
    data = raw if raw.dtype == np.float64 else np.asarray(raw, dtype=np.float32)
    if data.ndim == 1:
        data = data[None, :]
    return data, float(idx.sr)


def get_rps(frame: td.Frame, entry: str = "rps") -> tuple[np.ndarray, np.ndarray]:
    """``frame[entry]`` as ``((R, N) float64, frame times in absolute seconds)``.

    Accepts trajectories on either a ``StampIndex`` (the convention
    :func:`tracking_frame` writes) or a ``GridIndex`` (e.g. an STFT-grid
    ``rps`` entry from a dataset frame).
    """
    series = frame[entry]
    r = np.atleast_2d(np.asarray(series.data, dtype=np.float64))
    idx = series.tindex
    if isinstance(idx, td.GridIndex):
        times = idx.sample_times()
    elif isinstance(idx, td.StampIndex):
        times = idx.abs_stamps
    else:
        raise TypeError(
            f"{entry!r} must sit on a GridIndex or StampIndex, got {type(idx).__name__}"
        )
    return r, np.asarray(times, dtype=np.float64)


def _meta_entries(frame: td.Frame) -> dict[str, Any]:
    """The ``"meta"`` sub-Frame's entries as a fresh dict (``{}`` if absent)."""
    if "meta" not in frame:
        return {}
    meta = frame["meta"]
    return {key: meta[key] for key in meta}


def with_meta(frame: td.Frame, **entries: Any) -> td.Frame:
    """Return ``frame`` with ``entries`` set in the invariant ``"meta"`` sub-Frame.

    The seam mechanism: a stage that produces something for a LATER stage
    (:func:`peel_stage`'s residual audio, an annealed trust region) leaves it
    here. Frames are immutable, so the sub-Frame is rebuilt; the input frame is
    never mutated. ``None`` REMOVES an entry (how a seam is consumed).
    """
    meta = _meta_entries(frame)
    for key, value in entries.items():
        if value is None:
            meta.pop(key, None)
        else:
            meta[key] = value
    return frame.with_entry("meta", td.Frame(meta))


def with_rps(
    frame: td.Frame,
    r: np.ndarray,
    frame_times: np.ndarray,
    *,
    stage: str,
    info: Mapping[str, Any],
) -> td.Frame:
    """Replace ``"rps"`` and append ``{"stage": stage, **info}`` to the log.

    The diagnostics log is the ``"tracking"`` list inside the invariant
    ``"meta"`` sub-Frame. Frames are immutable, so the list and the sub-Frame
    are rebuilt (copied) — the input frame and any frame sharing its meta are
    never mutated.
    """
    out = frame.with_entry("rps", _rps_series(r, frame_times))
    meta = _meta_entries(frame)
    history = list(meta.get("tracking", []))
    history.append({"stage": stage, **dict(info)})
    meta["tracking"] = history
    return out.with_entry("meta", td.Frame(meta))


def pipeline(*stages: Stage) -> Stage:
    """Left-to-right composition: ``pipeline(a, b)(frame) == b(a(frame))``."""

    def run(frame: td.Frame) -> td.Frame:
        for stage in stages:
            frame = stage(frame)
        return frame

    return run


def _core_inputs(frame: td.Frame) -> tuple[np.ndarray, float, np.ndarray, np.ndarray, float]:
    """``(audio, sr, r, frame_times, t0)`` for handing to an array core.

    ``frame_times`` stay absolute (for :func:`with_rps`); the cores get
    ``frame_times - t0`` with ``t0`` the audio entry's ``t_start``, so a
    time-sliced frame refines correctly against its own audio slice.
    """
    audio, sr = get_audio(frame)
    r, times = get_rps(frame)
    t0 = float(frame["audio"].t_start)
    return audio, sr, r, times, t0


def _log(frame: td.Frame) -> list[Mapping[str, Any]]:
    """The stage log so far (``meta["tracking"]``), oldest first."""
    if "meta" not in frame or "tracking" not in frame["meta"]:
        return []
    return list(frame["meta"]["tracking"])


def _last_log(frame: td.Frame, key: str) -> Any:
    """The most recent log entry's value of ``key``, or ``None``."""
    for entry in reversed(_log(frame)):
        if key in entry:
            return entry[key]
    return None


# ---------------------------------------------------------------------------
# 2. the stage vocabulary


def blind_seed_stage(
    n_rotors: int = LADDER_N_ROTORS,
    cfg: SeedConfig | None = None,
    arms: Iterable[str] = (),
    *,
    hop_s: float = DEFAULT_HOP_S,
    name: str = "blind_seed",
) -> Stage:
    """Blind seeding -> a constant-trajectory ``"rps"`` init.

    Runs :func:`tracking.blind_seed` on the frame's audio (the core
    channel-averages the whitened spectrogram internally, so all mics
    contribute) and writes ``SeedResult.bases`` as constant trajectories on a
    fresh uniform grid with hop ``hop_s`` (default: the
    ``scripts/vk_validation.py`` evaluation grid). Any existing ``"rps"``
    entry is replaced. Diagnostics: the bases plus a scalar summary of
    ``SeedResult.diagnostics`` (the full scan arrays stay out of the frame).
    """

    def run(frame: td.Frame) -> td.Frame:
        audio, sr = get_audio(frame)
        result: SeedResult = blind_seed(audio, float(sr), n_rotors, cfg, arms)
        t0 = float(frame["audio"].t_start)
        duration = audio.shape[-1] / sr
        ft = t0 + np.arange(0.0, duration - hop_s / 2, hop_s)
        r = np.tile(result.bases[:, None], (1, len(ft)))
        diag = result.diagnostics
        info: dict[str, Any] = {
            "bases": [float(b) for b in result.bases],
            "arms": list(diag.get("arms", [])),
            "primary": diag.get("primary"),
            "octave": diag.get("octave"),
            "n_candidates": len(result.candidates),
            "accepted_bases": [float(c["base"]) for c in result.candidates if c.get("accepted")],
            "update_gate": result.update_gate,
            "bw_hz": result.bw_hz,
        }
        return with_rps(frame, r, ft, stage=name, info=info)

    return run


def coarse_init_stage(cfg: CoarseConfig | None = None, *, name: str = "coarse_init") -> Stage:
    """Full-range coarse Viterbi c(t) -> a time-varying ``"rps"`` init.

    Wraps :func:`tracking.pipelines.coarse_init`: a BPF octave check on the
    current (constant) bases, then a 12-120 rev/s frame-rate Viterbi pass with
    the energy-timed takeoff bridge, so a warmup or takeoff ramp inside the
    window is reachable by the ladder that follows. The stage keeps the frame's
    rps grid and re-anchors the bases onto the coarse path; the trust gates can
    return the constant init unchanged (``coarse_mode`` says which).

    The current ``"rps"`` must be the constant bases of
    :func:`blind_seed_stage` — this stage reads them off row means.
    """
    use = cfg or CoarseConfig()

    def run(frame: td.Frame) -> td.Frame:
        audio, sr, r, times, t0 = _core_inputs(frame)
        seg = Segment(audio=audio, ft=times - t0)
        bases = r.mean(axis=1)
        gate = _last_log(frame, "update_gate")
        r0, bases_out, halved, diag = coarse_init(seg, bases, cfg=use, sr=float(sr))
        info: dict[str, Any] = {
            "bases": [round(float(b), 3) for b in bases_out],
            "halved": bool(halved),
            # The K calibration ran on the rejected 2x bases, so a halved seed
            # drops the auto update gate (beatvk_vk_arms.fullrange_init).
            "update_gate": None if halved else gate,
            **{k: v for k, v in diag.items() if k != "coarse_c"},
        }
        return with_rps(frame, r0, times, stage=name, info=info)

    return run


@dataclass(frozen=True)
class Vit2dspConfig:
    """The calibrated blind-annotation ladder's knobs.

    ``weights`` / ``phys_map`` are the per-rotor mic-weight matrix
    ``(n_mics, 4)`` and the track -> physical-rotor map (both data-bound, so a
    blind caller leaves them ``None``: uniform weights and the identity map).
    ``splice_update_gate`` takes the seed's auto ``update_gate`` off the stage
    log and splices it into the two VK configs — the ``blind_KR`` behaviour.
    """

    weights: Any = None
    phys_map: Any = None
    midband_cfg: VKConfig | None = None
    refine_cfg: VKConfig | None = None
    stage_guard: bool = False
    seed_cfg: SeedConfig | None = None
    hop_s: float = DEFAULT_HOP_S
    splice_update_gate: bool = False


def vit2dsp_stage(
    cfg: Vit2dspConfig | None = None, *, name: str = "vit2dsp", **kwargs: Any
) -> Stage:
    """The vit2dsp ladder (:func:`tracking.vit2dsp_pipeline`) as one Stage.

    Runs on the frame's audio and current ``"rps"``; when the frame has no
    ``"rps"`` entry the ladder seeds itself via :func:`blind_seed_stage`
    (4 rotors, ``cfg.seed_cfg`` — default :data:`tracking.SEED_CFG`), which
    appends its own ``<name>_seed`` log entry.

    The ladder runs as one bespoke unit rather than as composed ``vk_stage`` /
    ``guarded`` sub-stages: the validated pipeline fixes the twin pairs at init
    and shares one whitened spectrogram across all its internal stages
    (per-stage recomputation would change both the cost and — via mid-ladder
    pair re-assignment — the science).
    """
    use = cfg or Vit2dspConfig()
    if kwargs:
        from dataclasses import replace as _replace

        use = _replace(use, **kwargs)

    def run(frame: td.Frame) -> td.Frame:
        f = frame
        if "rps" not in f:
            f = blind_seed_stage(
                LADDER_N_ROTORS, use.seed_cfg or SEED_CFG, hop_s=use.hop_s, name=f"{name}_seed"
            )(f)
        audio, sr_f = get_audio(f)
        r0, times = get_rps(f)
        if r0.shape[0] != LADDER_N_ROTORS:
            raise ValueError(
                f"vit2dsp is a {LADDER_N_ROTORS}-track (two twin pairs) ladder, "
                f"got {r0.shape[0]} rps tracks"
            )
        mcfg = use.midband_cfg or MIDBAND_CFGS[0]
        rcfg = use.refine_cfg or REFINE_CFG
        if use.splice_update_gate:
            gate = _last_log(f, "update_gate")
            if gate is not None:
                from dataclasses import replace as _replace

                mcfg = _replace(mcfg, update_gate=float(gate))
                rcfg = _replace(rcfg, update_gate=float(gate))
        for label, vcfg in (("midband_cfg", mcfg), ("refine_cfg", rcfg)):
            if abs(vcfg.fs - sr_f) > 1e-6:
                raise ValueError(f"{label}.fs={vcfg.fs} does not match the frame audio rate {sr_f}")
        t0 = float(f["audio"].t_start)
        seg = Segment(audio=audio, ft=times - t0)
        n_mics = audio.shape[0]
        w = (
            np.asarray(use.weights, dtype=np.float64)
            if use.weights is not None
            else np.full((n_mics, LADDER_N_ROTORS), 1.0 / n_mics)
        )
        pm = (
            np.asarray(use.phys_map, dtype=int)
            if use.phys_map is not None
            else np.arange(LADDER_N_ROTORS)
        )
        stage_snaps, ref, extras, wall_scan, wall_vk = vit2dsp_pipeline(
            seg,
            r0,
            w,
            pm,
            midband_cfg=mcfg,
            refine_cfg=rcfg,
            stage_guard=use.stage_guard,
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


def vk_stage(cfg: VKConfig, *, name: str = "vk") -> Stage:
    """Coupled VK order tracking (:func:`tracking.vk_track`) on ``"rps"``.

    ``cfg.fs`` must match the frame's audio rate. Diagnostics: mean windowed
    confidence, the per-round residual ratios and max deltas, ``n_outer``.
    """

    def run(frame: td.Frame) -> td.Frame:
        audio, sr, r, times, t0 = _core_inputs(frame)
        if abs(cfg.fs - sr) > 1e-6:
            raise ValueError(f"cfg.fs={cfg.fs} does not match the frame audio rate {sr}")
        result = vk_track(audio, r, times - t0, cfg)
        conf = result.confidence
        info: dict[str, Any] = {
            "confidence_mean": float(conf.mean()) if conf.size else float("nan"),
            "residual_ratios": [float(v) for v in result.residual_ratios],
            "max_deltas": [float(v) for v in result.max_deltas],
            "n_outer": int(cfg.n_outer),
        }
        return with_rps(frame, result.r_refined, times, stage=name, info=info)

    return run


@dataclass(frozen=True)
class PeelConfig:
    """The peel's own geometry (:func:`tracking.make_peels`).

    Defaults are the frozen flagship values (``docs/experiments/beat-vk.md``):
    a 1 Hz envelope bandwidth (~1 s coherence, inside the measured 0.5-1.5 s
    tau_k window at k = 8-40) and 40 harmonics. ``mode="ls"`` re-fits each
    harmonic's complex gain onto the clip per time block before subtracting,
    so one component cannot inject energy; ``"open"`` subtracts the VK solve as
    it stands (the 2026-08-04 flagship).

    The peel's cost is independent of the tracker's harmonic cap, so a caller
    on a short or cheap window can shrink ``k_max`` without touching the frozen
    tracker settings.
    """

    mode: str = DEFAULT_PEEL_MODE
    bw_hz: float = PEEL_BW_HZ
    k_max: int = PEEL_K_MAX
    n_rotors: int = LADDER_N_ROTORS


def peel_stage(cfg: PeelConfig | None = None, *, name: str = "peel") -> Stage:
    """Peel the OTHER rotors' combs out of the audio -> a seam in ``meta``.

    Runs :func:`tracking.make_peels` at the frame's current trajectories and
    leaves ``{"peel_audio", "pair_audio", "diag", "wall_s"}`` under
    ``meta[PEEL_SEAM]``. The trajectory is untouched and NO log entry is
    appended: the stage that consumes the seam (:func:`pi_kalman_stage`)
    records the peel's diagnostics in its own entry, so one application of the
    flagship stays one log entry however it is composed.
    """
    use = cfg or PeelConfig()

    def run(frame: td.Frame) -> td.Frame:
        audio, sr, r_cur, times, t0 = _core_inputs(frame)
        clip = np.asarray(audio, dtype=np.float64)
        tic = time.perf_counter()
        peel_audio, pair_audio, diag = make_peels(
            clip,
            r_cur,
            times - t0,
            sr,
            use.mode,
            n_rotors=use.n_rotors,
            bw_hz=use.bw_hz,
            k_max=use.k_max,
        )
        wall = time.perf_counter() - tic
        seam = {
            "stage": name,
            "peel_audio": peel_audio,
            "pair_audio": pair_audio,
            "diag": diag,
            "wall_s": wall,
        }
        return with_meta(frame, **{PEEL_SEAM: seam})

    return run


@dataclass(frozen=True)
class PiConfig:
    """:func:`tracking.pi_kalman_refine` settings for :func:`pi_kalman_stage`.

    ``variant`` selects a :data:`tracking.PI_VARIANTS` row (the
    bandwidth-and-admission revision rows of ``docs/experiments/beat-vk.md``).
    Every other field is ``None`` = "the core's own default", so a bare
    ``PiConfig()`` is the plain refiner; :data:`PI_PROTOCOL` is the flagship
    row. ``extra`` passes anything else straight through.
    """

    variant: str = "protocol"
    n_iter: int | None = None
    band_hz: float | None = None
    pair_mode: str | None = None
    band_b0: float | Sequence[float] | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)

    def kwargs(self) -> dict[str, Any]:
        """The merged keyword arguments of one ``pi_kalman_refine`` call."""
        if self.variant not in PI_VARIANTS:
            raise KeyError(f"unknown pi variant {self.variant!r}; known: {sorted(PI_VARIANTS)}")
        out: dict[str, Any] = dict(PI_VARIANTS[self.variant])
        for name in ("n_iter", "band_hz", "pair_mode", "band_b0"):
            value = getattr(self, name)
            if value is not None:
                out[name] = value
        out.update(dict(self.extra))
        return out


#: The flagship's protocol row (the 0.641 result; ``findings.md`` "Iterated
#: pi_kalman: mechanism findings").
PI_PROTOCOL = PiConfig(n_iter=PI_N_ITER, band_hz=PI_BAND_HZ, pair_mode=PI_PAIR_MODE)


def pi_kalman_stage(
    cfg: PiConfig | None = None,
    *,
    name: str = "pi_kalman",
    diagnostics: bool = True,
    **kwargs: Any,
) -> Stage:
    """Phase-increment Kalman refinement (:func:`tracking.pi_kalman_refine`).

    Consumes the :func:`peel_stage` seam when the frame carries one: the
    per-rotor / per-pair residual audio goes in through the core's
    ``peel_audio`` / ``pair_audio`` arguments and the seam is cleared, so a
    second application re-peels rather than reusing a stale residual.

    An annealed variant (``band_anneal="posterior"``) picks its trust region up
    from the most recent ``band_b0_final`` in the stage log, which is how the
    posterior carries across applications without a driver holding it.

    ``kwargs`` override ``cfg``'s. ``diagnostics=False`` keeps the core's own
    diagnostics dict out of the log entry (the flagship's per-application
    record, which is serialized into every run cache).
    """
    call = (cfg or PiConfig()).kwargs()
    call.update(kwargs)

    def run(frame: td.Frame) -> td.Frame:
        audio, sr, r_cur, times, t0 = _core_inputs(frame)
        clip = np.asarray(audio, dtype=np.float64)
        seam = _meta_entries(frame).get(PEEL_SEAM)
        use = dict(call)
        if use.get("band_anneal") == "posterior":
            carried = _last_log(frame, "band_b0_final")
            if carried is not None:
                use["band_b0"] = tuple(carried)
        peel_kwargs = (
            {"peel_audio": seam["peel_audio"], "pair_audio": seam["pair_audio"]} if seam else {}
        )
        tic = time.perf_counter()
        r_next, pi_diag = pi_kalman_refine(
            clip, r_cur, times - t0, sr=int(round(sr)), **peel_kwargs, **use
        )
        wall_pi = time.perf_counter() - tic
        step = r_next - r_cur
        b0_final = pi_diag.get("band_b0_final")
        info: dict[str, Any] = {
            "wall_peel_s": round(float(seam["wall_s"]) if seam else 0.0, 1),
            "wall_pi_s": round(wall_pi, 1),
            "step_rms": [round(float(np.sqrt(np.mean(step[r] ** 2))), 4) for r in range(len(step))],
            "step_mean": [round(float(np.mean(step[r])), 4) for r in range(len(step))],
            **({"band_b0_final": b0_final} if b0_final is not None else {}),
            **({"peel": seam["diag"]} if seam else {}),
            **({"diagnostics": pi_diag} if diagnostics else {}),
        }
        out = with_rps(frame, r_next, times, stage=name, info=info)
        return with_meta(out, **{PEEL_SEAM: None}) if seam else out

    return run


def warp_stage(name: str = "warp", **kwargs: Any) -> Stage:
    """Iterated time-warp refinement (:func:`tracking.iter_warp_refine`).

    ``kwargs`` are passed through to the core (``rounds``, ``rungs``,
    ``max_step``, ...). Diagnostics: the core's JSON-serializable dict.
    """

    def run(frame: td.Frame) -> td.Frame:
        audio, sr, r, times, t0 = _core_inputs(frame)
        r_new, diag = iter_warp_refine(audio, r, times - t0, int(round(sr)), **kwargs)
        return with_rps(frame, r_new, times, stage=name, info={"diagnostics": diag})

    return run


def refine_coherent_stage(
    cfg: RefineConfig | None = None, *, name: str = "stage_d", **kwargs: Any
) -> Stage:
    """Coherent phase-slope refinement (:func:`tracking.refine_coherent`).

    ``cfg`` defaults to ``RefineConfig(sample_rate=<frame audio rate>)``; a
    given ``cfg`` must match the frame's audio rate. ``kwargs`` pass through
    to the core (``k_min``, ``k_max``, ``bandwidth_hz``, ``n_iter``, ...).
    The core returns no diagnostics — the log entry records the parameters.
    """

    def run(frame: td.Frame) -> td.Frame:
        audio, sr, r, times, t0 = _core_inputs(frame)
        rcfg = cfg if cfg is not None else RefineConfig(sample_rate=int(round(sr)))
        if rcfg.sample_rate != int(round(sr)):
            raise ValueError(
                f"cfg.sample_rate={rcfg.sample_rate} does not match the frame audio rate {sr}"
            )
        r_new = refine_coherent(audio, r, times - t0, rcfg, **kwargs)
        info: dict[str, Any] = {"params": dict(kwargs)}
        return with_rps(frame, r_new, times, stage=name, info=info)

    return run


# --- trajectory-candidate stages (the fitness judge's inputs) ---------------


def presmooth_stage(cut_hz: float = 5.0, *, source: str = "rps", name: str = "presmooth") -> Stage:
    """Low-pass the trajectory at ``cut_hz`` (:func:`tracking.presmooth`).

    Step 1 of the telemetry refit: the tachometer's 0.269 rev/s / 49.7 Hz
    staircase is measurement noise. ``source`` names the entry the trajectory
    is read from (``"rps_meas"`` smooths the untouched telemetry into
    ``"rps"``). ``cut_hz <= 0`` is the identity.
    """

    def run(frame: td.Frame) -> td.Frame:
        r, times = get_rps(frame, source)
        out = presmooth(r, times, cut_hz)
        return with_rps(
            frame, out, times, stage=name, info={"cut_hz": float(cut_hz), "from": source}
        )

    return run


def scale_stage(factor: float, *, source: str = "rps", name: str = "scale") -> Stage:
    """Multiply the trajectory by a constant rate scale."""

    def run(frame: td.Frame) -> td.Frame:
        r, times = get_rps(frame, source)
        return with_rps(
            frame,
            r * float(factor),
            times,
            stage=name,
            info={"factor": float(factor), "from": source},
        )

    return run


def shift_stage(tau: float, *, source: str = "rps", name: str = "shift") -> Stage:
    """Read the trajectory ``tau`` seconds LATER, i.e. move the trace earlier.

    Positive ``tau`` is the correction for a telemetry trace that runs LATE.
    The shift is an interpolation on the frame grid, never a roll of samples,
    so ``tau`` is free of the grid. ``np.interp`` clamps at the ends.
    """

    def run(frame: td.Frame) -> td.Frame:
        r, times = get_rps(frame, source)
        out = np.stack([np.interp(times + float(tau), times, row) for row in r])
        return with_rps(frame, out, times, stage=name, info={"tau_s": float(tau), "from": source})

    return run


# --- scoring / whole-procedure stages ---------------------------------------


def fitness_stage(
    reference_entry: str = "rps_meas",
    *,
    cfg: FitnessConfig = FitnessConfig(),
    holdouts: Sequence[Holdout] | None = None,
    control: str = "none",
    name: str = "fitness",
) -> Stage:
    """Score the frame's current ``rps`` (:func:`tracking.score_window`).

    The trajectory is NOT changed; the stage appends a ``{"stage": "fitness",
    ...}`` entry, so it can be dropped anywhere into a ladder to record how the
    fit moved. The reference (which pins the bands, the block grid and the
    admission gate) is read from ``reference_entry`` — the frame's untouched
    ``"rps_meas"`` by default, which is what "fixed degrees of freedom" means.
    """
    from dataclasses import replace as _replace

    def run(frame: td.Frame) -> td.Frame:
        audio, sr = get_audio(frame)
        r, ft = get_rps(frame)
        ref, _ = get_rps(frame, reference_entry)
        t0 = float(ft[0]) if ft.size else 0.0
        info = score_window(
            audio,
            ft - t0,
            r,
            ref,
            cfg=_replace(cfg, sr=int(round(sr))),
            holdouts=holdouts,
            control=control,
            n_boot=0,
        )
        return with_rps(frame, r, ft, stage=name, info=info)

    return run


def fvk_stage(
    cfg: FVKConfig | None = None,
    *,
    reference_entry: str = "rps_meas",
    k_hi: int | None = None,
    rho_scale: float = 1.0,
    name: str = "fvk",
) -> Stage:
    """Score the frame's current ``rps`` by F_VK (:func:`tracking.fvk_score`).

    The profiled coupled-VK residual — the differentiable sibling of
    :func:`fitness_stage`, and like it the trajectory is NOT changed: the stage
    only appends a ``{"stage": "fvk", ...}`` entry, so it can be dropped
    anywhere into a ladder. ``reference_entry`` pins the harmonic cap (the fixed
    degrees of freedom); it falls back to the frame's own ``"rps"`` when the
    frame carries no reference.
    """
    use = cfg or FVKConfig()

    def run(frame: td.Frame) -> td.Frame:
        audio, sr = get_audio(frame)
        r, ft = get_rps(frame)
        ref = get_rps(frame, reference_entry)[0] if reference_entry in frame else r
        t0 = float(frame["audio"].t_start)
        info = fvk_score(
            audio,
            sr,
            r,
            ft - t0,
            replace(use, sr=int(round(sr))),
            reference=ref,
            k_hi=k_hi,
            rho_scale=rho_scale,
        )
        return with_rps(frame, r, ft, stage=name, info=info)

    return run


def fvk_refine_stage(
    cfg: FVKConfig | None = None,
    *,
    schedule: Sequence[FVKStage] | None = None,
    knot_s: float = 0.25,
    smooth_lambda: float | str = 1.0,
    lr: float = 1.0,
    reference_entry: str = "rps_meas",
    name: str = "fvk_refine",
) -> Stage:
    """Refine ``rps`` by L-BFGS on F_VK (:func:`tracking.optimize_trajectory`).

    The continuous step the VK literature never took: the frame's current
    trajectory is the init, a coarse cubic-spline basis is the parameterization,
    and the ``k_max`` annealing schedule is the continuation. The per-rung loss
    trace and argmin movement land in the log entry — that is the
    continuation-validity reading, so a driver never has to instrument the loop.

    ``smooth_lambda`` takes ``"auto"`` (:func:`tracking.auto_smooth_lambda`),
    which is what a frame outside the cruise regime needs — the default 1.0 is
    cruise-calibrated and pins a takeoff ramp in place.
    """
    use = cfg or FVKConfig()

    def run(frame: td.Frame) -> td.Frame:
        audio, sr, r, times, t0 = _core_inputs(frame)
        ref = get_rps(frame, reference_entry)[0] if reference_entry in frame else r
        r_out, diag = optimize_trajectory(
            audio,
            sr,
            r,
            times - t0,
            replace(use, sr=int(round(sr))),
            schedule=None if schedule is None else tuple(schedule),
            knot_s=knot_s,
            smooth_lambda=smooth_lambda,
            lr=lr,
            reference=ref,
        )
        return with_rps(frame, r_out, times, stage=name, info=diag)

    return run


def decompose_stage(
    cfg: FVKConfig | None = None,
    *,
    k_hi: int | None = None,
    rho_scale: float = 1.0,
    bw_schedule: BandwidthSchedule | None = None,
    reference_entry: str = "rps_meas",
    name: str = "decompose",
) -> Stage:
    """Split the frame's audio into per-harmonic tracks plus a residual.

    One coupled Vold-Kalman solve at the frame's CURRENT trajectory
    (:mod:`tracking.decompose`), so the split follows whatever produced that
    trajectory — a blind ladder, a refit, or the L-BFGS refiner. Like
    :func:`fitness_stage` the trajectory is NOT changed; the products travel as
    a seam, ``meta["decompose"]``::

        {"envelopes": Envelopes, "phase": (R, T), "recon": (C, T),
         "track_energy": (M,)}

    and the ENERGY LEDGER (track / residual / cross-term fractions, per-band
    shares) is the diagnostics entry, so a caller that only wants the reading
    never touches the seam. The residual is ``audio - recon`` by definition, so
    the decomposition is exact and only the SPLIT is estimated.

    ``k_hi`` pins the harmonic set; it defaults to the cap of
    ``reference_entry`` (falling back to the frame's own ``"rps"``), which is
    what makes two windows of one recording stitchable track by track.
    ``bw_schedule`` is the v2 linewidth-matched per-track bandwidth
    (:class:`tracking.decompose.BandwidthSchedule`); ``None`` keeps the flat v1
    band, under which the comb LEAKS into the residual above about ``k`` 10. Read
    :func:`tracking.decompose.group_plan` before running this on a long window:
    the coupled group is the whole comb and costs about
    ``1e-4 k_hi^2 window_s`` GB.
    """
    from tracking.decompose import (
        energy_ledger,
        reconstruct,
        shaft_phase,
        solve_window,
    )
    from tracking.fitness_vk import k_cap, to_audio_grid

    use = cfg or FVKConfig()

    def run(frame: td.Frame) -> td.Frame:
        audio, sr, r, times, t0 = _core_inputs(frame)
        conf = replace(use, sr=int(round(sr)))
        n_t = int(audio.shape[-1])
        r_audio = to_audio_grid(r, times - t0, n_t, conf.sr)
        ref = get_rps(frame, reference_entry)[0] if reference_entry in frame else r
        cap = int(k_hi) if k_hi is not None else k_cap(conf, ref)
        tic = time.perf_counter()
        env = solve_window(
            audio, r_audio, conf, k_hi=cap, rho_scale=rho_scale, bw_schedule=bw_schedule
        )
        phase = shaft_phase(r_audio, conf.sr)
        recon, track_energy = reconstruct(env.x, env.k, env.rotor, phase, conf.stride)
        y = np.asarray(audio, dtype=np.float64)[: env.x.shape[0]]
        info: dict[str, Any] = {
            "k_hi": cap,
            "n_tracks": int(len(env.k)),
            "n_env": int(env.x.shape[-1]),
            "fs_env": float(env.fs_env),
            "rho_scale": float(rho_scale),
            "bw_schedule": bw_schedule.as_dict() if bw_schedule is not None else None,
            "wall_s": round(time.perf_counter() - tic, 3),
            **energy_ledger(y, recon, track_energy, env.k),
        }
        out = with_meta(
            frame,
            decompose={
                "envelopes": env,
                "phase": phase,
                "recon": recon,
                "track_energy": track_energy,
            },
        )
        return with_rps(out, r, times, stage=name, info=info)

    return run


def refit_stage(
    *,
    cfg: RefitConfig | None = None,
    reference_entry: str = "rps_meas",
    name: str = "telemetry_refit",
) -> Stage:
    """RECIPE: fit the rotors to the harmonics (GitHub issue 17, steps 1-6)::

        presmooth(5 Hz) -> repeat(peel -> pi_kalman at a k that climbs with
        the measured error) until max |dr| < tol

    One iteration is :func:`pi_kalman_arm_stage`, the alternation above. The
    loop is data-dependent — each rung's harmonic cap comes from the last
    iteration's update — so it is a driver, and its convergence bookkeeping and
    its readings live in :func:`tracking.refit_window`.

    The carrier is the frame's ``reference_entry`` (``"rps_meas"`` by default,
    the untouched telemetry) — NOT the frame's current ``"rps"``, because the
    procedure is defined as a refinement OF the measurement. The stage replaces
    ``"rps"`` with the fit and appends the report (minus the trajectories) as
    its diagnostics entry.
    """
    from tracking.telemetry_refit import refit_window

    use = cfg or RefitConfig()

    def run(frame: td.Frame) -> td.Frame:
        audio, sr = get_audio(frame)
        ref, ft = get_rps(frame, reference_entry)
        t0 = float(frame["audio"].t_start)
        res = refit_window(audio, ref, ft - t0, sr, cfg=use)
        return with_rps(frame, res.r_fit, ft, stage=name, info=res.as_dict())

    return run


def guarded(inner: Stage, *, cfg: SeedConfig | None = None, name: str | None = None) -> Stage:
    """Wrap ``inner`` with the blind per-track stage guard.

    Runs ``inner``, then :func:`tracking.stage_guard` on the before/after
    trajectories against the whitened spectrogram of the frame's audio, and
    reverts the per-rotor trajectories the guard vetoes — a track that a stage
    re-captured onto an occupied comb, or whose comb confidence collapsed, goes
    back to its pre-stage trajectory. The guard requires ``inner`` to keep the
    rps grid (same shape and frame times). Diagnostics: reverted track indices,
    per-track confidences before/after, and the revert reasons.
    """

    def run(frame: td.Frame) -> td.Frame:
        r_before, times_before = get_rps(frame)
        out = inner(frame)
        r_after, times_after = get_rps(out)
        if r_after.shape != r_before.shape or not np.allclose(times_after, times_before):
            raise ValueError("guarded() requires the inner stage to keep the rps frame grid")
        audio, sr = get_audio(frame)
        t0 = float(frame["audio"].t_start)
        scfg = cfg if cfg is not None else SeedConfig()
        white, bin_hz, spec_times = whitened_logmag(audio, float(sr), scfg)
        r_guarded, reverted, diag = stage_guard(
            r_before, r_after, white, bin_hz, spec_times, times_before - t0, scfg
        )
        info: dict[str, Any] = {"reverted": [int(i) for i in reverted], **diag}
        return with_rps(out, r_guarded, times_after, stage=name or "guard", info=info)

    return run


# ---------------------------------------------------------------------------
# 3. the recipes — every shipped variant of the algorithm, in a few lines each


def vit2dsp(cfg: Vit2dspConfig | None = None, *, n_rotors: int = LADDER_N_ROTORS) -> Stage:
    """The calibrated blind-annotation ladder (DREGON pooled err_sm 0.688)::

    blind seed -> Viterbi pair-mean c(t) -> spatial joint 2-rotor Viterbi
    -> mid-band VK (bw 6) -> VK refine

    The four ladder steps after the seed are one unit — see
    :func:`vit2dsp_stage` for why they are not composed here.
    """
    use = cfg or Vit2dspConfig()
    return pipeline(
        blind_seed_stage(n_rotors, use.seed_cfg or SEED_CFG, hop_s=use.hop_s),
        vit2dsp_stage(use),
    )


def blind_fullrange(
    cfg: Vit2dspConfig | None = None,
    *,
    coarse: CoarseConfig | None = None,
    n_rotors: int = LADDER_N_ROTORS,
) -> Stage:
    """The ``blind_fullrange`` arm: a takeoff/warmup ramp is reachable::

    blind seed (arms K, R) -> coarse full-range Viterbi c(t) -> the ladder

    The seed's auto ``update_gate`` is spliced into the ladder's VK configs
    (the ``blind_KR`` behaviour), unless the coarse stage's octave check
    halved the bases — the K calibration ran on the rejected 2x bases.
    """
    from dataclasses import replace as _replace

    use = _replace(cfg or Vit2dspConfig(), splice_update_gate=True)
    return pipeline(
        blind_seed_stage(n_rotors, use.seed_cfg or SEED_CFG, arms=("K", "R"), hop_s=use.hop_s),
        coarse_init_stage(coarse),
        vit2dsp_stage(use),
    )


def flagship(
    n_apps: int,
    *,
    peel: PeelConfig | None = None,
    pi: PiConfig = PI_PROTOCOL,
    name: str | None = None,
) -> Stage:
    """The FLAGSHIP peeled alternation: ``n_apps`` x (peel -> pi_kalman).

    ``peel=None`` is the ``naive`` comparison arm — the same pi_kalman passes
    on the unmodified clip. An annealed ``pi`` variant carries its trust region
    across applications through the stage log, so this is a plain composition
    and not a driver.
    """
    stages: list[Stage] = []
    for _ in range(n_apps):
        if peel is not None:
            stages.append(peel_stage(peel))
        stages.append(pi_kalman_stage(pi, name=name or _arm_name(peel), diagnostics=False))
    return pipeline(*stages)


def _arm_name(peel: PeelConfig | None) -> str:
    return "peeled" if peel is not None else "naive"


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
    """ONE application of the alternation — :func:`flagship` with ``n_apps=1``.

    Kept as the array-level entry point every campaign driver already calls
    (``scripts/beatvk_flagship.py``, :mod:`tracking.telemetry_refit`,
    ``scripts/tracking_ref.py``): ``pi_kwargs`` go straight to the core, and
    the log entry is the single per-application record — peel and pi wall
    times, the per-rotor step statistics and the peel diagnostics.
    """
    cfg = (
        PeelConfig(mode=peel_mode, bw_hz=peel_bw_hz, k_max=peel_k_max, n_rotors=n_rotors)
        if peel
        else None
    )
    return flagship(1, peel=cfg, pi=PiConfig(extra=pi_kwargs), name=name or _arm_name(cfg))


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
    """:func:`flagship` one application at a time, keeping every frame.

    Returns the ``n_apps + 1`` frames of the alternation, ``[0]`` being the
    input (the init) — so a caller reads trajectory ``i`` with :func:`get_rps`
    and application ``i``'s diagnostics off the last ``meta["tracking"]``
    entry. ``pi_variant`` selects a :data:`tracking.PI_VARIANTS` row and
    ``band_b0`` overrides its initial k-scaled band scale (rev/s).
    """
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; valid: {list(ARMS)}")
    peel = PeelConfig(mode=peel_mode, n_rotors=n_rotors) if arm == "peeled" else None
    from dataclasses import replace as _replace

    pi = _replace(PI_PROTOCOL, variant=pi_variant, band_b0=band_b0)
    step = flagship(1, peel=peel, pi=pi)
    frames = [frame]
    for app in range(1, n_apps + 1):
        frames.append(step(frames[-1]))
        if verbose:
            info = frames[-1]["meta"]["tracking"][-1]
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


def judge(
    candidate: Stage,
    *,
    control: str = "none",
    cfg: FitnessConfig = FitnessConfig(),
    holdouts: Sequence[Holdout] | None = None,
    reference_entry: str = "rps_meas",
) -> Stage:
    """Score a CANDIDATE trajectory at fixed degrees of freedom::

    candidate -> fitness

    ``candidate`` is any stage that writes ``"rps"`` — a composition of
    :func:`presmooth_stage` / :func:`scale_stage` / :func:`shift_stage` off
    ``"rps_meas"`` is the campaign's one-parameter family, and
    :func:`refit_stage` is the fitted one. The band, the block grid and the
    admission gate are pinned to ``reference_entry``, so the carrier is the
    only input that changes. ``control`` runs one of the section-B nulls
    (:data:`tracking.CONTROLS`).
    """
    return pipeline(
        candidate,
        fitness_stage(reference_entry, cfg=cfg, holdouts=holdouts, control=control),
    )
