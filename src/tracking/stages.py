"""TimeFrame stage API: ``td.Frame -> td.Frame`` adapters over the tracking cores.

Every tracking stage is a callable ``Stage: td.Frame -> td.Frame`` (plan doc
``docs/refactor-2026-08-plan.md`` §3.2). The frame conventions:

- ``"audio"``: ``(mic, time)`` float32 Series on a ``GridIndex`` at the audio
  sample rate. :func:`tracking_frame` accepts a mono ``(T,)`` array and stores
  it as ``(1, T)``.
- ``"rps"``: ``(rotor, time)`` float64 Series on a ``StampIndex`` at the
  trajectory frame times — the *current candidate trajectories*. Stages
  replace this entry and append their diagnostics to the ``"tracking"`` list
  inside the invariant ``"meta"`` sub-Frame (one ``{"stage": name, ...}``
  dict per stage, in application order).
- ``"rps_meas"``: optional reference trajectories, same convention as
  ``"rps"``; never touched by stages.

The array cores (:func:`tracking.vk_track`, :func:`tracking.blind_seed`,
:func:`tracking.pi_kalman_refine`, :func:`tracking.iter_warp_refine`,
:func:`tracking.refine_coherent`, ...) stay as they are — the adapters here
only move arrays in and out of the frame. Multichannel convention: every
core accepts ``(T,)`` or ``(C, T)`` audio (channels capped at 8 inside the
cores), so the full ``(mic, time)`` entry is passed straight through;
:func:`blind_seed` / :func:`stage_guard` channel-average internally via
``whitened_logmag``. Frame times handed to the cores are made relative to
the audio entry's ``t_start``, so stages also work on time-sliced frames.

Purity rule: this module imports only ``numpy``, ``tdseries`` and
``tracking.*`` (enforced by the ``tracking stays pure`` import-linter
contract).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any

import numpy as np
import tdseries as td

from tracking.phase_increment_tracker import pi_kalman_refine
from tracking.rps_refinement import RefineConfig, refine_coherent
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
    "Stage",
    "blind_seed_stage",
    "get_audio",
    "get_rps",
    "guarded",
    "pi_kalman_stage",
    "pipeline",
    "refine_coherent_stage",
    "tracking_frame",
    "vk_stage",
    "warp_stage",
    "with_rps",
]

#: A tracking stage: consumes a frame, returns a frame with ``"rps"``
#: replaced and one diagnostics dict appended to ``meta["tracking"]``.
Stage = Callable[[td.Frame], td.Frame]

#: Default trajectory frame hop (seconds) for stages that create the grid
#: (:func:`blind_seed_stage`) — the evaluation-grid convention of
#: ``scripts/vk_validation.py`` (``FRAME_HOP_S``, the predecessor's STFT hop).
DEFAULT_HOP_S = 0.032


# ---------------------------------------------------------------------------
# frame construction / accessors


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


# ---------------------------------------------------------------------------
# stage adapters


def blind_seed_stage(
    n_rotors: int,
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


def pi_kalman_stage(name: str = "pi_kalman", **kwargs: Any) -> Stage:
    """Phase-increment Kalman refinement (:func:`tracking.pi_kalman_refine`).

    ``kwargs`` are passed through to the core (``n_iter``, ``band_hz``,
    ``k_max``, ...). Diagnostics: the core's JSON-serializable dict.
    """

    def run(frame: td.Frame) -> td.Frame:
        audio, sr, r, times, t0 = _core_inputs(frame)
        r_new, diag = pi_kalman_refine(audio, r, times - t0, int(round(sr)), **kwargs)
        return with_rps(frame, r_new, times, stage=name, info={"diagnostics": diag})

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


def guarded(inner: Stage, *, cfg: SeedConfig | None = None, name: str | None = None) -> Stage:
    """Wrap ``inner`` with the blind per-track stage guard.

    Runs ``inner``, then :func:`tracking.stage_guard` on the before/after
    trajectories against the whitened spectrogram of the frame's audio, and
    reverts the per-rotor trajectories the guard vetoes — mirroring how
    ``scripts/vk_blind_annotation.py`` applies the guard after every ladder
    stage. The guard requires ``inner`` to keep the rps grid (same shape and
    frame times). Diagnostics: reverted track indices, per-track confidences
    before/after, and the revert reasons.
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
