#!/usr/bin/env python3
"""Decompose a drone recording into per-harmonic tracks plus a broadband residual.

The decomposition is the coupled Vold-Kalman solve of :mod:`tracking.fitness_vk`
run on the REFINED rotor-speed labels (``scripts/refine_dregon_rps.py``), window
by window, then stitched into one full-length envelope bank::

    x(t) = sum_{rotor, k} Re[ a_{rotor,k}(t) e^{j k phi_rotor(t)} ] + residual(t)

The sum is EXACT by construction, because the residual is DEFINED as what the
track sum does not explain. What is estimated is the split, and that split is a
maximum-likelihood one: the VK cost is a penalized least squares, which is the
MAP estimate of the envelopes under a Gaussian residual and a per-track
second-difference (bandwidth) prior. So a track carries the energy the model can
explain at its own carrier and inside its own band, and the residual carries the
rest.

What the instrument measures
----------------------------
Three questions, one pass:

Where does the energy sit
    The ledger in ``report.json``: total, the track sum, the residual, and the
    cross term (the tracks are not orthogonal, so the three do not add up to the
    total by themselves). Track energy is also split into the harmonic bands
    ``k1-9 / k10-24 / k25-49 / k50-80``.

Is the phase model per-harmonic or per-shaft
    Each track's envelope phase is the phase ERROR of that harmonic against the
    label-driven carrier. The pi-kalman model treats the per-harmonic drifts as
    INDEPENDENT; a shaft-jitter model predicts that harmonic ``k`` sees ``k``
    times ONE common phase, that is a rank-one increment covariance. The report
    gives, per rotor, the drift standard deviation against ``k`` and the top
    eigenvalue share of the correlation matrix of the increments across ``k``.

What is left in the residual that is not broadband
    ``report.json`` -> ``residual_tones``: per ~8 s segment, the strongest ten
    tonal peaks of the residual below 2 kHz, each with its distance to the
    nearest rotor order. On ``free-flight_nosource_room1`` these sit at
    NON-INTEGER orders that drift independently of the rotor speed — foreign
    quasi-stationary tones (structural / aerodynamic resonances), not comb
    leakage — and a smooth-PSD noise model cannot represent them. Measurement
    only: nothing here removes a tone.

How prior-dependent the weak tracks are
    ``--bw-sweep`` re-solves one mid-recording window on two prior axes and
    reports how the per-band mean amplitude and drift move: the requested
    ``bw_rps`` (0.5, 1, 2, 4) and ``rho_scale`` (0.25, 4). The second axis is
    there because the first one is largely INERT — a coupled group clamps every
    track to ``max(VKConfig.bw_hz, 6 * line separation)``, and a dense comb
    floors that at 1 Hz, so the ``bw_rps`` arms mostly measure the clamp. Each
    arm reports the bandwidth it actually got.

    That clamp is what ``--bw-schedule bw0,slope,capfrac,absmax`` (v2) defeats:
    it sets each track's band THROUGH the selectivity, after the clamp, to
    ``clip(bw0 + slope * k, base, min(capfrac * separation, absmax))`` Hz. A
    real line is not 1 Hz wide at every harmonic — the shaft jitter widens
    harmonic ``k`` by ``k`` times the rate error — and the flat band left about
    72 % of the comb's order contrast in the residual at k10-24. Empty (the
    default) is the flat v1 band, call for call.

Conventions a consumer must know
--------------------------------
``t_env`` in ``envelopes.npz`` is seconds from the audio ``t_start`` of the
PUBLISHED (untrimmed) frame — the same reference the refined-label sidecar uses.
The loader trims each frame to the audio-telemetry overlap (5.48 s on
``free-flight_nosource_room1``) and the trim (``t0_offset_s``) is added back
here.

Every window is re-referenced to ONE global shaft phase before the stitch. The
solver's own ``phase`` starts at the window (``phase = 2 pi cumsum(r) / fs``), so
window ``W`` that starts at audio sample ``a0`` carries the constant offset
``Phi(a0 - 1)``; the stitch multiplies each track by ``exp(-j k Phi(a0 - 1))``.
Without that, two overlapping windows hold the same physical track at two
different phase origins and the cross-fade cancels them.

Windows start on a multiple of the envelope stride, so every window's envelope
grid is a slice of one global envelope grid and no resampling is needed.

How much memory one window costs
--------------------------------
Read :func:`group_plan` before sizing a job. Coupling is transitive, so at
``k_hi`` 62 the whole comb is ONE banded system and a 16 s window needs 6.3 GB
per worker — the solve is memory-bound, not compute-bound, and the cost grows
as ``k_hi^2`` times the window length. ``--mem-budget-gb`` (8 GB by default)
refuses a window that does not fit, so a unit fails with the arithmetic instead
of the pool being killed. A run at the default settings is a CLUSTER job.

Run::

    # smoke: one 4 s window, k 20, 2 mics, solve + stitch — about ten seconds
    PYTHONPATH=src python scripts/vk_decompose.py --smoke

    # the full recording (8 mics, k 80) on a CPU node
    omnirun submit --backend uni-cpu --gpus 0 --cpus 4 --mem 32 --time 4h \\
      --name vk-decompose --outputs "results/vk_decompose/**" \\
      --env PYTHONPATH=src -- \\
      python scripts/vk_decompose.py --mode solve --jobs 2 --bw-sweep
    omnirun pull vk-decompose
    PYTHONPATH=src python scripts/vk_decompose.py --mode stitch

    # the v2 configuration: linewidth-matched bands, and 32 kHz so that k 80 is
    # reachable at all (see docs/experiments/vk-decomposition.md § v2). One
    # coupling group of 320 tracks needs 7.9 GB per worker at 12 s.
    python scripts/vk_decompose.py --mode all --jobs 2 --out results/vk_decompose_v2 \\
      --sr 32000 --f-max 8000 --k-max 80 --window-s 12 --hop-s 9 \\
      --mem-budget-gb 9 --bw-schedule 3,0,1.5,3

Outputs: the gridrun units under ``<out>/raw/`` (one JSON plus one ``.npz`` of
complex envelopes each, 26 MB per window at 8 mics and ``k_hi`` 62) and, per
recording, ``<out>/<recording_id>/{envelopes.npz, residual.npz, report.json,
bw_sweep.json}``. The envelope bank is ``(mic, track, env frame)`` float32
twice: 100 MB for 8 mics over the 64 s of ``free-flight_nosource_room1``.
"""

from __future__ import annotations

import os

# Cap BLAS threads BEFORE numpy import (process-level parallelism instead) —
# the shared harness convention (utils.gridrun re-asserts it).
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# The array core is tracking.decompose — the solve, the phase re-reference, the
# cross-fade stitch and every statistic below come from there. What is left in
# this file is the DATA (the generator's own loader), the unit harness and the
# file formats.
import tracking.decompose as D  # noqa: E402
from tracking.decompose import (  # noqa: E402
    DEFAULT_BANDS as BANDS,
)
from tracking.decompose import (  # noqa: E402
    BandwidthSchedule,
    band_name,
    band_summary,
    drift_increments,
    energy_ledger,
    fade_weights,
    interp_rps,
    per_track_stats,
    phase_model_report,
    rank_one_share,
    reconstruct,
    reference_mic,
    residual_tones,
    shaft_phase,
    to_audio_grid,
    track_bands,
    welch_psd,
)
from tracking.joint_decompose import (  # noqa: E402
    JointConfig,
    corrected_phase,
    global_rate_correction,
    joint_solve_window,
    order_cell_profile,
    theta_rate,
    whitened_flatness,
    window_extra_phase,
)
from utils.gridrun import Unit, add_gridrun_args, gridrun_from_args, unit_path  # noqa: E402

__all__ = [  # the importable core the tests read; the CLI is main()
    "BANDS",
    "BandwidthSchedule",
    "joint_config",
    "band_name",
    "band_summary",
    "drift_increments",
    "energy_ledger",
    "fade_weights",
    "frame_grid",
    "fvk_config",
    "group_plan",
    "interp_rps",
    "per_track_stats",
    "phase_model_report",
    "rank_one_share",
    "reconstruct",
    "reference_mic",
    "shaft_phase",
    "solve_window",
    "to_audio_grid",
    "track_bands",
    "welch_psd",
    "window_bounds",
    "window_span",
]

OUT_DEFAULT = "results/vk_decompose"
#: The refined labels this decomposition is conditioned on.
LABEL_DIR_DEFAULT = "src/data_processing/refined_labels"
#: The generator's own noise source, loaded with the generator's own arguments.
FRAMES_SPEC = "frames:DREGON-frames"
RPS_KEY = "motors_measured"
SPLITS = ["in_flight_noise"]
#: Default sample rate — the v1 rate, kept as the default so an unqualified run
#: reproduces v1. ``--sr 32000`` is the v2 configuration (see ``--f-max``).
SR = 16000
#: Default modelling ceiling in Hz. It is only ever half of the cap: the
#: geometry also holds every line under ``0.375 * sr``, so 8 kHz needs 32 kHz
#: audio to mean anything.
F_MAX = 6000.0
#: The frozen evaluation frame grid (``tracking.protocols.BEATVK.hop_s``) — the
#: grid the refined labels sit on, and the grid windows are cut on.
HOP_S = 0.032
#: Below this mean rate a window has no usable comb and is not decomposed.
IDLE_REV_S = 20.0
#: ``tracking.vk_tracking._MAX_CHANNELS`` — the solver's own channel clamp. It is
#: exactly the DREGON array size, so all 8 microphones go into ONE solve and no
#: channel batching is necessary.
MAX_MICS = 8


# ---------------------------------------------------------------------------
# data


def frame_grid(n_t: int, sr: int) -> np.ndarray:
    """The uniform ``HOP_S`` frame grid of a recording, in relative seconds."""
    return D.frame_grid(n_t, sr, HOP_S)


def published_audio_starts(spec: str) -> dict[str, int]:
    """``{recording id: audio t_start in ticks}`` of the UNTRIMMED frames.

    The time reference of every output. ``load_published_noise_sources`` trims
    each frame to the audio-telemetry overlap, so the trimmed frame's own
    ``t_start`` would put the results 5.48 s early for a consumer that reads the
    published recording. Resampling keeps ``t_start_ticks``, so this pass reads
    the raw audio entry and does no resample.
    """
    from data_processing.frames import meta_dict
    from data_processing.noise_rps_dataset import _parse_frames_spec
    from data_processing.streams import iter_published_frames

    name, version = _parse_frames_spec(spec)
    starts: dict[str, int] = {}
    splits = None if "michaels" in spec else SPLITS
    for tf in iter_published_frames(name, version, splits=splits):
        rid = meta_dict(tf).get("recording_id")
        if rid and "audio" in tf:
            starts[str(rid)] = int(tf["audio"].t_start_ticks)
    return starts


def load_recordings(spec: str, label_dir: str | Path, sr: int = SR) -> list[dict[str, Any]]:
    """Every surviving noise recording, with the REFINED labels applied.

    The loader is the generator's (``load_published_noise_sources`` with the
    generator's arguments), so the audio decomposed here is the audio training
    sees. ``rps_override_dir`` replaces the telemetry values with the refined
    trajectory before the overlap trim, which is where the sidecar's times are
    defined.

    ``sr`` is the rate the loader resamples to, and it is a REAL knob of the
    measurement: a harmonic is modelled only while its line stays under
    ``min(f_max, 0.45 sr)``, so 16 kHz alone caps DREGON near ``k`` 75 at its
    rate peaks whatever ``f_max`` says. The v2 configuration runs at 32 kHz so
    the 8 kHz ceiling is the only cap left.

    ``ft`` is LOCAL: seconds from the trimmed frame's audio ``t_start``.
    ``t0_offset_s`` carries the trim, so every output can be written against the
    published recording.
    """
    import tdseries as td

    from data_processing.frames import meta_dict
    from data_processing.noise_rps_dataset import load_published_noise_sources

    # The dataset profile follows the spec: Michael's frames carry the already
    # recalibrated generic ``rps`` track (no refined sidecar exists, none is
    # needed), DREGON carries ``motors_measured`` plus the sidecar override.
    if "michaels" in spec:
        origin, rps_key, splits, override = "michaels", "rps", None, None
    else:
        origin, rps_key, splits, override = "dregon", RPS_KEY, SPLITS, label_dir
    starts = published_audio_starts(spec)
    recs: list[dict[str, Any]] = []
    for src in load_published_noise_sources(
        spec, int(sr), origin=origin, rps_key=rps_key, splits=splits, rps_override_dir=override
    ):
        frame = src.frame
        meta = meta_dict(frame)
        rid = str(meta.get("recording_id") or "")
        if not rid:
            raise KeyError(f"published frame has no meta.recording_id (keys: {sorted(meta)})")
        if rid not in starts:
            raise KeyError(f"{rid}: no untrimmed audio t_start — the two loader passes disagree")
        audio_s = frame["audio"]
        audio = np.atleast_2d(np.asarray(audio_s.data, dtype=np.float32))
        rps_s = frame[src.rps_key]
        # Tick-exact relative seconds: both indexes expose integer ticks, and
        # the absolute epoch (~1e18 ticks) does not survive float subtraction.
        t0 = int(audio_s.tindex.t_start_ticks)
        ticks = np.asarray(rps_s.tindex.abs_stamps_ticks, dtype=np.int64)
        stamps = (ticks - t0) / float(td.TICKS_PER_SECOND)
        n_t = int(audio.shape[-1])
        ft = frame_grid(n_t, int(sr))
        r_ref = interp_rps(np.asarray(rps_s.data), stamps, ft)
        recs.append(
            {
                "recording_id": rid,
                "audio": audio,
                "ft": ft,
                "r_ref": r_ref,
                # The audio-rate carrier rate. Every window slices THIS array,
                # so a window's carrier is a slice of the global one and the
                # phase re-reference below is exact.
                "r_audio": to_audio_grid(r_ref, ft, n_t, int(sr)),
                "t0_offset_s": (t0 - starts[rid]) / float(td.TICKS_PER_SECOND),
                "rps_key": rps_key,
                "sr": int(sr),
            }
        )
    if not recs:
        raise RuntimeError(f"{spec}: no recording with an {RPS_KEY} track survived loading")
    return recs


#: Per-process recording cache, keyed by ``(recording id, sample rate)``. Pool
#: workers are reused across units, so each process decodes the dataset once;
#: under a fork start method it inherits the parent's copy and decodes nothing.
#: The rate is part of the key because it is part of the AUDIO — two rates are
#: two different decodes of one recording.
_RECORDINGS: dict[tuple[str, int], dict[str, Any]] = {}


def cache_recordings(recs: list[dict[str, Any]], sr: int = SR) -> None:
    """Fill the per-process cache — called before the pool forks."""
    _RECORDINGS.update({(str(r["recording_id"]), int(sr)): r for r in recs})


def get_recording(rid: str, spec: str, label_dir: str | Path, sr: int = SR) -> dict[str, Any]:
    key = (str(rid), int(sr))
    if key not in _RECORDINGS:
        cache_recordings(load_recordings(spec, label_dir, int(sr)), int(sr))
    if key not in _RECORDINGS:
        raise KeyError(f"recording {rid!r} not in {spec}")
    return _RECORDINGS[key]


def window_bounds(n_frames: int, window_s: float, hop_s: float) -> list[tuple[int, int]]:
    """Window frame ranges over a whole recording, the last one right-aligned."""
    return D.window_bounds(n_frames, window_s, hop_s, HOP_S)


def window_span(ft: Any, i0: int, i1: int, n_t: int, stride: int, sr: int = SR) -> tuple[int, int]:
    """Audio sample range of one window, snapped to the envelope stride."""
    return D.window_span(ft, i0, i1, n_t, stride, int(sr), HOP_S)


# ---------------------------------------------------------------------------
# the solve unit


def fvk_config(
    k_max: int,
    *,
    mics: int = MAX_MICS,
    bw_rps: float = 1.0,
    sr: int = SR,
    f_max: float = F_MAX,
) -> Any:
    """THE measurement geometry — one construction, so every solve agrees.

    ``f_max`` is the modelling ceiling, but never the only one:
    :func:`tracking.decompose.solve_config` holds it under ``0.375 sr``, so the
    SAMPLE RATE caps the harmonic set whenever it is the smaller of the two.
    """
    return D.solve_config(k_max, sr=sr, mics=mics, bw_rps=bw_rps, f_max=f_max)


def solve_window(
    audio: Any,
    r_audio: Any,
    k_hi: int,
    k_max: int,
    bw_rps: float,
    mics: int,
    *,
    sr: int = SR,
    f_max: float = F_MAX,
    rho_scale: float = 1.0,
    bw_schedule: BandwidthSchedule | None = None,
) -> Any:
    """``(config, envelopes)`` of one coupled VK solve of one window.

    The harmonic set is capped from the RECORDING's reference trajectory (see
    :func:`recording_k_hi`), so every window of a recording holds the identical
    ``(rotor, harmonic)`` track set and the windows can be stitched track by
    track. ``bw_schedule`` is the v2 linewidth-matched per-track bandwidth
    (``--bw-schedule``); ``None`` is the flat v1 band.
    """
    cfg = fvk_config(k_max, mics=mics, bw_rps=bw_rps, sr=sr, f_max=f_max)
    return cfg, D.solve_window(
        audio,
        r_audio,
        cfg,
        k_hi=k_hi,
        mics=mics,
        rho_scale=rho_scale,
        bw_schedule=bw_schedule,
    )


def group_plan(r_audio: Any, k_hi: int, cfg: Any) -> dict[str, Any]:
    """Coupling-group partition of one window, and the memory it will cost.

    THE memory model of this script — read :func:`tracking.decompose.group_plan`
    before a job is sized. The short version: coupling is TRANSITIVE, the whole
    comb is ONE banded system, and the cost is about ``1e-4 k_hi^2 window_s`` GB
    per worker (6.3 GB at the default ``k_hi`` 62 and 16 s windows, which is
    what a local three-worker run could not hold).
    """
    return D.group_plan(r_audio, k_hi, cfg)


def recording_k_hi(r_ref: Any, k_max: int, *, sr: int = SR, f_max: float = F_MAX) -> int:
    """The harmonic cap of a WHOLE recording, from its refined labels.

    ``tracking.fitness_vk.k_cap`` reads the maximum rate of the reference it is
    given. Giving it the whole recording (and not the window) is what keeps the
    track set identical across windows, which the stitch needs; it is also the
    safe direction, because the cap of the fastest window is the smallest one.
    """
    from tracking.fitness_vk import k_cap

    return int(k_cap(fvk_config(k_max, sr=sr, f_max=f_max), np.asarray(r_ref)))


def joint_config(params: dict[str, Any]) -> JointConfig:
    """The v3 :class:`JointConfig` from the JSON-safe unit parameters.

    One construction, so a worker and the stitch cannot disagree about the arm
    that produced a window. Every field is carried as a scalar or a
    comma-separated string, because it has to survive a JSON round trip through
    the unit table and the report's provenance.
    """
    parts = [float(v) for v in str(params.get("bw_psi", "0.6,8,1.5")).split(",")]
    slope, cap = parts[0], parts[1]
    floor = parts[2] if len(parts) > 2 else 1.5
    ladder = tuple(int(v) for v in str(params.get("k_trust", "3,12,80")).split(",") if v.strip())
    return JointConfig(
        iters=int(params.get("iters", 3)),
        k_trust=ladder,
        bw_theta_hz=float(params.get("bw_theta", 1.5)),
        bw_psi_slope=slope,
        bw_psi_max=cap,
        bw_psi_min=floor,
        whiten=bool(params.get("whiten", True)),
    )


def solve_worker(unit: Unit) -> dict[str, Any]:
    """One unit: one window (``kind`` = ``window``) or one bandwidth (``bw``).

    A window unit writes its complex envelopes to ``<out>/raw/<uid>.npz`` and
    records the path in its JSON — a complex bank does not belong in JSON. A
    bandwidth unit reports band statistics only and writes no array.
    """
    p = dict(unit.params)
    sr, f_max = int(p.get("sr", SR)), float(p.get("f_max", F_MAX))
    rec = get_recording(str(p["recording"]), str(p["spec"]), str(p["label_dir"]), sr)
    i0, i1 = int(p["i0"]), int(p["i1"])
    mics = min(int(p["mics"]), int(rec["audio"].shape[0]))
    stride = int(p["stride"])
    n_t = int(rec["audio"].shape[-1])
    a0, a1 = window_span(rec["ft"], i0, i1, n_t, stride, sr)
    offset = float(rec["t0_offset_s"])
    r_win = np.asarray(rec["r_audio"])[:, a0:a1]
    mean_rev_s = float(r_win.mean())

    out: dict[str, Any] = {
        "kind": str(p.get("kind", "window")),
        "recording": str(p["recording"]),
        "i0": i0,
        "i1": i1,
        "a0": a0,
        "a1": a1,
        "start_s": round(a0 / float(sr) + offset, 6),
        "end_s": round(a1 / float(sr) + offset, 6),
        "mean_rev_s": round(mean_rev_s, 4),
        "mics": mics,
        "sr": sr,
        "f_max": f_max,
        "k_max": int(p["k_max"]),
        "bw_rps": float(p["bw_rps"]),
        "rho_scale": float(p.get("rho_scale", 1.0)),
        "bw_schedule": str(p.get("bw_schedule", "")),
    }
    if mean_rev_s < IDLE_REV_S:
        return {**out, "used": False, "reason": "idle"}

    k_hi = recording_k_hi(rec["r_ref"], int(p["k_max"]), sr=sr, f_max=f_max)
    cfg = fvk_config(int(p["k_max"]), mics=mics, bw_rps=float(p["bw_rps"]), sr=sr, f_max=f_max)
    plan = group_plan(r_win, k_hi, cfg)
    budget = float(p["mem_budget_gb"])
    if budget > 0 and float(plan["banded_gb"]) > budget:
        # Fail this unit with the arithmetic, instead of letting the operating
        # system kill the pool. gridrun turns the exception into one .err file.
        raise MemoryError(
            f"the coupled group needs {plan['banded_gb']} GB (max group "
            f"{plan['max_group']} tracks, {plan['n_env']} envelope frames) against a "
            f"--mem-budget-gb of {budget}. Decrease --window-s or --k-max, or raise the "
            "budget on a node that can hold it."
        )
    tic = time.perf_counter()
    joint = bool(p.get("joint", False)) and out["kind"] != "bw"
    jres = None
    if joint:
        # v3: the alternation. It replaces the single solve and returns the
        # EFFECTIVE envelope (g e^{j psi}) against the CORRECTED carrier, so
        # everything below reads it exactly as it reads a v2 bank.
        jres = joint_solve_window(
            rec["audio"][:, a0:a1],
            r_win,
            cfg,
            k_hi=k_hi,
            mics=mics,
            jcfg=joint_config(p),
            bw_schedule=BandwidthSchedule.parse(str(p.get("bw_schedule", ""))),
            rho_scale=float(p.get("rho_scale", 1.0)),
            t_start_s=a0 / float(sr) + offset,
        )
        env = jres.env
    else:
        _, env = solve_window(
            rec["audio"][:, a0:a1],
            r_win,
            k_hi,
            int(p["k_max"]),
            float(p["bw_rps"]),
            mics,
            sr=sr,
            f_max=f_max,
            rho_scale=float(p.get("rho_scale", 1.0)),
            bw_schedule=BandwidthSchedule.parse(str(p.get("bw_schedule", ""))),
        )
    wall = time.perf_counter() - tic
    out.update(
        {
            "used": True,
            "reason": "ok",
            "k_hi": int(k_hi),
            "group_plan": plan,
            "n_tracks": int(len(env.k)),
            "n_env": int(env.x.shape[-1]),
            "fs_env": float(env.fs_env),
            "wall_s": round(wall, 2),
            # The bandwidth the solver ACTUALLY used per band — the schedule as
            # achieved, not as requested (see the "bw" arm's comment below).
            "bw_track_hz_by_band": band_summary(env.bw_track, env.k),
        }
    )

    ref = reference_mic(rec["audio"][:mics, a0:a1], int(p["ref_mic"]))
    out["ref_mic"] = ref
    if out["kind"] == "bw":
        # The prior-sensitivity arm: statistics only, on this window alone and
        # on the reference microphone, so the arms are comparable without any
        # stitch.
        amp = np.abs(env.x[ref])
        pherr = np.unwrap(np.angle(env.x[ref].astype(np.complex128)), axis=-1)
        mask = np.ones(amp.shape[-1], dtype=bool)
        mean, cv, drift = per_track_stats(amp, pherr, mask, env.fs_env)
        return {
            **out,
            "amp_mean_by_band": band_summary(mean, env.k),
            "amp_cv_by_band": band_summary(cv, env.k),
            "drift_std_rad_s_by_band": band_summary(drift, env.k),
            # The bandwidth the solver ACTUALLY used, which is NOT the one asked
            # for. A coupled group clamps every track's band to
            # ``max(VKConfig.bw_hz, 6 * separation)``, and a dense comb has
            # adjacent lines a fraction of a hertz apart, so the clamp floors
            # the whole group at 1 Hz and the ``bw_rps`` arms collapse onto each
            # other. ``rho_scale`` is applied AFTER the clamp, which is why the
            # sweep carries that second axis. Read this before reading the arms.
            "bw_track_hz_by_band": band_summary(env.bw_track, env.k),
        }

    npz = unit_path(p["out"], unit.uid).with_suffix(".npz")
    npz.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, Any] = {
        "x": np.asarray(env.x, dtype=np.complex64),
        "valid": np.asarray(env.valid, dtype=bool),
        "rotor": np.asarray(env.rotor, dtype=np.int64),
        "k": np.asarray(env.k, dtype=np.int64),
        "bw_track": np.asarray(env.bw_track, dtype=np.float64),
    }
    if jres is not None:
        # The v3 extras. ``dr`` is the gauge-free form of ``theta`` (its time
        # derivative, in rev/s) and it is what the stitch carries across windows
        # — a phase has an arbitrary constant per window, a rate does not.
        arrays.update(
            theta=np.asarray(jres.theta_env, dtype=np.float64),
            dr=np.asarray(theta_rate(jres.theta_env, float(env.fs_env)), dtype=np.float64),
            psi=np.asarray(jres.psi, dtype=np.float32),
            psd_freq=np.asarray(jres.psd.freq, dtype=np.float64),
            psd_t=np.asarray(jres.psd.t_block, dtype=np.float64),
            psd_log_s=np.asarray(jres.psd.log_s, dtype=np.float32),
        )
        out["joint"] = {
            "config": joint_config(p).__dict__ | {"k_trust": list(joint_config(p).k_trust)},
            "iterations": jres.iterations,
        }
    np.savez(npz, allow_pickle=False, **arrays)
    # A cheap per-window capture check on ONE channel: the fraction of that
    # channel's energy the track sum explains. The full ledger is the stitch's,
    # which is why this reconstructs one channel and not the array.
    #
    # The carrier must be the solver's OWN phase. On the joint path that phase
    # carries the shaft correction, and rebuilding the plain label phase here
    # instead scores the bank against a carrier it was never fitted to — which
    # read as a NEGATIVE r2 on nearly every production window while the stitched
    # ledger was healthy.
    phase_w = np.asarray(env.phase) if joint else shaft_phase(r_win, sr)
    recon0, _ = reconstruct(env.x[ref : ref + 1], env.k, env.rotor, phase_w, stride)
    y0 = np.asarray(rec["audio"][ref, a0:a1], dtype=np.float64)
    e_y = float((y0**2).sum())
    resid = float(((y0 - recon0[0]) ** 2).sum())
    return {
        **out,
        "npz": str(npz.relative_to(Path(p["out"]))),
        "r2_ref_mic": round(1.0 - resid / max(e_y, 1e-30), 6),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Pooled unit table — counts, capture, wall time, and the bandwidth arms."""

    def mean(vals: list[float]) -> float | None:
        v = np.asarray(vals, dtype=np.float64)
        v = v[np.isfinite(v)]
        return round(float(v.mean()), 6) if v.size else None

    wins = [r for r in rows if r.get("kind", "window") == "window"]
    used = [r for r in wins if r.get("used")]
    return {
        "n_units": len(rows),
        "n_windows": len(wins),
        "n_used": len(used),
        "n_idle": sum(1 for r in wins if r.get("reason") == "idle"),
        "n_bw_units": sum(1 for r in rows if r.get("kind") == "bw"),
        "r2_ref_mic": mean([float(r["r2_ref_mic"]) for r in used if "r2_ref_mic" in r]),
        "wall_s": mean([float(r["wall_s"]) for r in rows if "wall_s" in r]),
        "recordings": sorted({str(r["recording"]) for r in rows}),
    }


# ---------------------------------------------------------------------------
# stitch


def read_rows(out: Path) -> list[dict[str, Any]]:
    raw = out / "raw"
    return [json.loads(p.read_text()) for p in sorted(raw.glob("*.json"))] if raw.is_dir() else []


def stitch_envelopes(
    rows: list[dict[str, Any]],
    out: Path,
    phi: Any,
    stride: int,
    ramp: int,
    *,
    r_audio: Any = None,
    sr: int = SR,
) -> dict[str, Any]:
    """Load the window ``.npz`` banks of one recording and stitch them.

    The file I/O of :func:`tracking.decompose.stitch_bank` — the phase
    re-reference and the cross-fade are that function's. The harmonic-set check
    is here because it is a check on what the UNITS wrote, not on the arrays.

    A JOINT (v3) window carries its own shaft correction, so before the stitch
    the windows have to be brought onto ONE carrier. That is done in the only
    gauge-free currency there is — the rate. Each window's ``dr`` (rev/s) is
    cross-faded into one global rate correction, the corrected phase is its
    integral, and each window's bank is rotated by the difference between its
    own carrier and that global one (:func:`window_extra_phase`). The rotation
    is slow by construction, and ``theta_stitch_max_rate_hz`` reports how fast
    the fastest track's rotation actually is, so a caller can see whether it
    stayed inside the 100 Hz envelope grid.
    """
    used = sorted((r for r in rows if r.get("used") and "npz" in r), key=lambda r: int(r["a0"]))
    if not used:
        raise SystemExit("no usable window unit — run --mode solve first")

    with np.load(out / used[0]["npz"]) as first:
        k = np.asarray(first["k"], dtype=np.int64)
        bw_track = np.asarray(first["bw_track"], dtype=np.float64)
        joint = "dr" in first

    windows: list[dict[str, Any]] = []
    for r in used:
        with np.load(out / r["npz"]) as data:
            if not np.array_equal(np.asarray(data["k"], dtype=np.int64), k):
                raise SystemExit(
                    f"{r['npz']}: harmonic set differs from the first window — the windows "
                    "were solved at different k_hi and cannot be stitched"
                )
            w = {
                "a0": int(r["a0"]),
                "x": np.asarray(data["x"], dtype=np.complex64),
                "valid": np.asarray(data["valid"], dtype=bool),
                "rotor": np.asarray(data["rotor"], dtype=np.int64),
                "k": k,
            }
            if joint:
                w["dr"] = np.asarray(data["dr"], dtype=np.float64)
                w["theta"] = np.asarray(data["theta"], dtype=np.float64)
            windows.append(w)

    extra: dict[str, Any] = {}
    if not joint:
        return {
            **D.stitch_bank(windows, phi, stride, ramp),
            "bw_track": bw_track,
            "windows": used,
            "phi": np.asarray(phi),
            "joint": False,
        }

    a_min = min(int(w["a0"]) for w in windows)
    a_max = max(int(w["a0"]) + int(w["x"].shape[-1]) * stride for w in windows)
    dr_g = global_rate_correction(windows, stride, a_min, a_max, ramp)
    r_corr, phi_t = corrected_phase(r_audio, dr_g, sr, stride, a_min, a_max)
    rot = np.asarray(windows[0]["rotor"], dtype=np.int64)
    max_rate = 0.0
    max_rate_raw = 0.0
    for w in windows:
        e_w = window_extra_phase(
            w["theta"], phi, phi_t, int(w["a0"]), stride, int(w["x"].shape[-1])
        )
        w["x"] = w["x"] * np.exp(1j * k[None, :, None] * e_w[rot][None, :, :]).astype(np.complex64)
        if e_w.shape[-1] > 1:
            # Weighted by the cross-fade: a rotation at a window EDGE is applied
            # where that window contributes almost nothing to the stitch, so the
            # unweighted maximum overstates what reaches the bank. Both are
            # reported, and the weighted one is the number to read.
            step = np.abs(np.diff(e_w, axis=-1)) * float(sr) / stride / (2.0 * np.pi)
            fade = fade_weights(int(e_w.shape[-1]), min(int(ramp), int(e_w.shape[-1]) // 2))
            pair = np.minimum(fade[:-1], fade[1:])[None, :]
            max_rate = max(max_rate, float((step * pair).max()) * float(k.max()))
            max_rate_raw = max(max_rate_raw, float(step.max()) * float(k.max()))
    extra = {
        "dr_global": dr_g,
        "r_corrected": r_corr,
        "phi": phi_t,
        "theta_stitch_max_rate_hz": round(max_rate, 3),
        "theta_stitch_max_rate_hz_raw": round(max_rate_raw, 3),
        "joint": True,
    }
    return {
        **D.stitch_bank(windows, phi_t, stride, ramp),
        "bw_track": bw_track,
        "windows": used,
        **extra,
    }


def _flatness_report(
    residual: Any, sr: int, st: dict[str, Any], out: Path, wins: list[dict[str, Any]]
) -> dict[str, Any] | None:
    """Whitened-residual flatness against the joint floor model, if there is one.

    The floor is per WINDOW, so the reading uses the middle used window's
    surface — the point of the check is the SHAPE of the whitened residual, and
    the shape is what a smooth floor holds still across a recording.
    """
    from tracking.joint_decompose import SmoothPSD

    used = sorted((r for r in wins if r.get("used") and "npz" in r), key=lambda r: int(r["a0"]))
    if not used:
        return None
    with np.load(out / used[len(used) // 2]["npz"]) as data:
        if "psd_log_s" not in data:
            return None
        psd = SmoothPSD(
            freq=np.asarray(data["psd_freq"], dtype=np.float64),
            t_block=np.asarray(data["psd_t"], dtype=np.float64),
            log_s=np.asarray(data["psd_log_s"], dtype=np.float64),
        )
    return whitened_flatness(residual, sr, psd)


def _order_cell_bands(audio: Any, sr: int, r_audio: Any, k_hi: int) -> dict[str, Any]:
    """Band table of :func:`order_cell_profile`, without the plotting arrays."""
    prof = order_cell_profile(audio, sr, r_audio, k_max=int(k_hi))
    return {
        nm: {kk: vv for kk, vv in d.items() if kk not in ("offsets", "cell")}
        for nm, d in prof["bands"].items()
    }


def stitch(
    out: Path,
    spec: str,
    label_dir: str,
    params: dict[str, Any],
    only: set[str] | None = None,
) -> list[Path]:
    """Combine the window units of each recording into one decomposition."""
    rows = read_rows(out)
    if not rows:
        raise SystemExit(f"{out}/raw is empty — run --mode solve first")
    stride = int(params["stride"])
    sr = int(params.get("sr", SR))
    sched = BandwidthSchedule.parse(str(params.get("bw_schedule", "")))
    written: list[Path] = []

    ids = {str(r["recording"]) for r in rows}
    for rid in sorted(ids if only is None else ids & only):
        got = [r for r in rows if r["recording"] == rid]
        wins = [r for r in got if r.get("kind", "window") == "window"]
        rec = get_recording(rid, spec, label_dir, sr)
        offset = float(rec["t0_offset_s"])
        phi = shaft_phase(rec["r_audio"], sr)
        ramp = max(0, int(round((params["window_s"] - params["hop_s"]) * params["fs_env"])))
        st = stitch_envelopes(
            wins, out, phi, stride, ramp, r_audio=rec["r_audio"], sr=sr
        )
        # The joint stitch replaces the carrier with the CORRECTED one; the v2
        # stitch hands back the same phi it was given, so one line covers both.
        phi_use = np.asarray(st["phi"])

        a_min, a_max, n_env = int(st["a_min"]), int(st["a_max"]), int(st["n_env"])
        x = st["x"]
        k, rotor = st["k"], st["rotor"]
        t_env = (a_min + np.arange(n_env) * stride) / float(sr) + offset
        audio = np.asarray(rec["audio"][: x.shape[0], a_min:a_max], dtype=np.float64)
        recon, track_energy = reconstruct(x, k, rotor, phi_use[:, a_min:a_max], stride)

        rec_dir = out / rid
        rec_dir.mkdir(parents=True, exist_ok=True)
        amp = np.empty(x.shape, dtype=np.float32)
        pherr = np.empty(x.shape, dtype=np.float32)
        for c in range(x.shape[0]):
            amp[c] = np.abs(x[c])
            pherr[c] = np.unwrap(np.angle(x[c].astype(np.complex128)), axis=-1).astype(np.float32)
        np.savez(
            rec_dir / "envelopes.npz",
            allow_pickle=False,
            t_env=t_env,
            rotor=rotor,
            k=k,
            amp=amp,
            phase_err=pherr,
            valid=st["valid"],
            bw_track=st["bw_track"],
            fs_env=np.float64(params["fs_env"]),
            stride=np.int64(stride),
            sample_rate=np.int64(sr),
            t0_offset_s=np.float64(offset),
            span_samples=np.asarray([a_min, a_max], dtype=np.int64),
            recording_id=np.array(rid),
            spec=np.array(spec),
            rps_key=np.array(str(rec.get("rps_key", RPS_KEY))),
            label_dir=np.array(str(label_dir)),
            time_reference=np.array(
                "seconds from the published recording's audio t_start (the full frame, "
                "before the loader's telemetry-overlap trim of t0_offset_s)"
            ),
        )

        residual = (audio - recon).astype(np.float32)
        f_psd, psd_res = welch_psd(residual, sr)
        _, psd_org = welch_psd(audio, sr)
        np.savez(
            rec_dir / "residual.npz",
            allow_pickle=False,
            residual=residual,
            freq_hz=f_psd,
            psd_residual=psd_res,
            psd_original=psd_org,
            sample_rate=np.int64(sr),
            t_start_s=np.float64(a_min / float(sr) + offset),
            span_samples=np.asarray([a_min, a_max], dtype=np.int64),
            recording_id=np.array(rid),
        )

        rate = np.asarray(rec["r_audio"])[:, a_min:a_max:stride].mean(axis=0)[:n_env]
        mask = st["covered"] & (rate > IDLE_REV_S)
        resynth = float(np.abs(audio - (recon + residual)).max())
        ref = reference_mic(audio, int(params["ref_mic"]))
        report: dict[str, Any] = {
            "recording_id": rid,
            "source": {
                "spec": spec,
                "rps_key": str(rec.get("rps_key", RPS_KEY)),
                "label_dir": str(label_dir),
                "sample_rate": sr,
                "splits": SPLITS,
            },
            "time_reference": (
                "seconds from the published recording's audio t_start (the full frame, "
                "before the loader's telemetry-overlap trim of t0_offset_s)"
            ),
            "t0_offset_s": round(offset, 6),
            "params": {**params, "idle_rev_s": IDLE_REV_S},
            "span_s": [
                round(a_min / float(sr) + offset, 6),
                round(a_max / float(sr) + offset, 6),
            ],
            "n_windows": len(wins),
            "n_used": len(st["windows"]),
            "n_idle": sum(1 for r in wins if r.get("reason") == "idle"),
            "n_tracks": int(len(k)),
            "n_env": n_env,
            "k_hi": int(k.max()),
            "mics": int(x.shape[0]),
            "energy": energy_ledger(audio, recon, track_energy, k),
            "resynthesis_max_abs": resynth,
            "flatness": _flatness_report(residual, sr, st, out, wins),
            "bw_schedule": (sched.as_dict() if sched is not None else None),
            "bw_track_hz_by_band": band_summary(st["bw_track"], k),
            # What is left in the residual that a smooth-PSD noise model cannot
            # represent. Measurement only: nothing here removes a tone.
            "residual_tones": residual_tones(
                residual,
                sr,
                np.asarray(rec["r_audio"])[:, a_min:a_max],
                t_start_s=a_min / float(sr) + offset,
            ),
            "phase_reference_max_dev_rad": D.phase_reference_deviation(
                rec["r_audio"], phi, int(st["windows"][0]["a0"]), sr
            ),
            # THE acceptance instrument, on the stitched residual and on the
            # ORIGINAL for reference. Read ``excess_db`` (absolute comb power
            # left, comparable between the two) before ``depth_db`` (a ratio,
            # which rises as the residual approaches the floor).
            "order_cell": {
                "residual": _order_cell_bands(
                    residual, sr, np.asarray(rec["r_audio"])[:, a_min:a_max], int(k.max())
                ),
                "original": _order_cell_bands(
                    audio, sr, np.asarray(rec["r_audio"])[:, a_min:a_max], int(k.max())
                ),
            },
            "ref_mic": ref,
            "phase_model": phase_model_report(
                amp[ref], pherr[ref], st["valid"], rotor, k, mask, float(params["fs_env"])
            ),
            "windows": [
                {kk: vv for kk, vv in r.items() if kk != "npz"}
                for r in sorted(wins, key=lambda r: int(r["a0"]))
            ],
        }
        if st.get("joint"):
            dr_g = np.asarray(st["dr_global"])
            report["joint"] = {
                "theta_stitch_max_rate_hz": st["theta_stitch_max_rate_hz"],
                "theta_stitch_max_rate_hz_raw": st["theta_stitch_max_rate_hz_raw"],
                "rate_correction_rms_rev_s": [
                    round(float(np.sqrt(np.mean(row**2))), 5) for row in dr_g
                ],
                "rate_correction_mean_rev_s": [round(float(row.mean()), 5) for row in dr_g],
                "rate_correction_pct_of_rate": [
                    round(float(100.0 * row.mean() / max(float(rr.mean()), 1e-9)), 4)
                    for row, rr in zip(dr_g, np.asarray(rec["r_audio"])[:, a_min:a_max], strict=False)
                ],
            }
            np.savez(
                rec_dir / "joint.npz",
                allow_pickle=False,
                t_env=t_env,
                dr_global=dr_g,
                r_corrected=np.asarray(st["r_corrected"])[:, a_min:a_max:stride][:, :n_env],
                r_labels=np.asarray(rec["r_audio"])[:, a_min:a_max:stride][:, :n_env],
                sample_rate=np.int64(sr),
                recording_id=np.array(rid),
                note=np.array(
                    "dr_global is the stitched shaft-rate correction in rev/s on the "
                    "envelope grid; r_corrected = r_labels + dr_global is the carrier the "
                    "envelopes are referenced to. Per-window psi and the per-microphone "
                    "smooth floor stay in the unit .npz files under raw/."
                ),
            )
        (rec_dir / "report.json").write_text(json.dumps(report, indent=1))
        written.append(rec_dir / "report.json")

        bw_rows = sorted(
            (r for r in got if r.get("kind") == "bw"),
            key=lambda r: (float(r["rho_scale"]), float(r["bw_rps"])),
        )
        if bw_rows:
            (rec_dir / "bw_sweep.json").write_text(
                json.dumps(
                    {
                        "recording_id": rid,
                        "note": (
                            "one mid-recording window re-solved on two prior axes: the "
                            "requested bw_rps, and rho_scale. Compare an arm against its "
                            "bw_track_hz_by_band first — a bw_rps arm that did not move the "
                            "achieved bandwidth was absorbed by the per-group clamp"
                        ),
                        "arms": bw_rows,
                    },
                    indent=1,
                )
            )
        led = report["energy"]
        print(
            f"[stitch] {rid}: {report['n_used']}/{report['n_windows']} windows, "
            f"{report['n_tracks']} tracks, tracks {led['track_fraction']:.3f} / residual "
            f"{led['residual_fraction']:.3f} of total -> {rec_dir}",
            flush=True,
        )
    return written


# ---------------------------------------------------------------------------
# CLI


def build_units(recs: list[dict[str, Any]], args: Any, common: dict[str, Any]) -> list[Unit]:
    """One unit per (recording, window), plus the bandwidth arms when asked."""
    units: list[Unit] = []
    for rec in recs:
        rid = rec["recording_id"]
        bounds = window_bounds(int(rec["ft"].size), args.window_s, args.hop_s)
        flight = [
            (i0, i1) for i0, i1 in bounds if float(rec["r_ref"][:, i0:i1].mean()) > IDLE_REV_S
        ]
        if args.smoke:
            if not flight:
                raise SystemExit(f"--smoke: {rid} has no window above {IDLE_REV_S} rev/s")
            bounds = [flight[len(flight) // 2]]
        units += [
            Unit(f"{rid}__f{i0:06d}", {"recording": rid, "i0": i0, "i1": i1, **common})
            for i0, i1 in bounds
        ]
        if args.bw_sweep and flight:
            # The middle FLIGHT window: the first seconds are the takeoff ramp.
            i0, i1 = flight[len(flight) // 2]
            arms = [(float(bw), 1.0) for bw in args.bw_grid]
            arms += [(float(common["bw_rps"]), float(rho)) for rho in args.rho_grid]
            units += [
                Unit(
                    f"{rid}__bw{bw:g}_r{rho:g}__f{i0:06d}",
                    {
                        "recording": rid,
                        "i0": i0,
                        "i1": i1,
                        **{**common, "kind": "bw", "bw_rps": bw, "rho_scale": rho},
                    },
                )
                for bw, rho in arms
            ]
        if args.smoke:
            break
    return units


def parse_floats(text: str) -> list[float]:
    return [float(v) for v in text.split(",") if v.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--mode", default="all", choices=("solve", "stitch", "all"))
    ap.add_argument("--recording", default="", help="comma-separated ids (default: all surviving)")
    ap.add_argument("--window-s", type=float, default=16.0)
    ap.add_argument("--hop-s", type=float, default=12.0)
    ap.add_argument("--k-max", type=int, default=80)
    ap.add_argument("--mics", type=int, default=MAX_MICS, help=f"microphones, {MAX_MICS} maximum")
    ap.add_argument(
        "--sr",
        type=int,
        default=SR,
        help=(
            f"sample rate the loader resamples to (default {SR}, the v1 rate). It caps the "
            "harmonic set too: no line above 0.375 * sr is modelled, so k 80 on DREGON needs "
            "--sr 32000"
        ),
    )
    ap.add_argument(
        "--f-max",
        type=float,
        default=F_MAX,
        help=f"modelling ceiling in Hz (default {F_MAX:g}); the v2 configuration is 8000",
    )
    ap.add_argument("--bw-rps", type=float, default=1.0, help="k-scaled VK bandwidth, rev/s")
    ap.add_argument(
        "--bw-schedule",
        default="",
        metavar="bw0,slope,capfrac,absmax",
        help=(
            "v2 linewidth-matched per-track bandwidth: bw_k = clip(bw0 + slope * k, base, "
            "min(capfrac * line separation, absmax)) Hz. Empty (the default) keeps the flat "
            "v1 band, under which the comb leaks into the residual above about k 10"
        ),
    )
    ap.add_argument(
        "--joint",
        action="store_true",
        help=(
            "v3: the JOINT decomposition — alternate the whitened VK solve, the shaft/track "
            "phase split and the masked smooth-floor fit (tracking.joint_decompose). Off by "
            "default, and off IS the v2 path, call for call"
        ),
    )
    ap.add_argument("--iters", type=int, default=3, help="--joint: alternation rounds")
    ap.add_argument(
        "--k-trust",
        default="3,12,80",
        help=(
            "--joint: the annealing ladder, one trustable harmonic cap per iteration. It "
            "starts low because the ENVELOPE BAND, not the phase unwrap, is what limits "
            "which harmonics can measure the shaft (see JointConfig.k_trust)"
        ),
    )
    ap.add_argument(
        "--bw-psi",
        default="0.6,8,1.5",
        metavar="slope,max[,min]",
        help=(
            "--joint: per-track phase-correction bandwidth, clip(slope * k, min, max) Hz. The "
            "min is what serves a strong LOW harmonic, whose true linewidth is wider than "
            "0.6 * k"
        ),
    )
    ap.add_argument(
        "--bw-theta", type=float, default=1.5, help="--joint: shaft-correction bandwidth, Hz"
    )
    ap.add_argument(
        "--no-whiten",
        action="store_true",
        help="--joint: skip the noise whitening (block A runs on the unweighted misfit)",
    )
    ap.add_argument(
        "--mem-budget-gb",
        type=float,
        default=8.0,
        help="refuse a window whose coupled group needs more than this (0 = no limit)",
    )
    ap.add_argument(
        "--ref-mic",
        type=int,
        default=-1,
        help="microphone the per-track tables are read on (-1 = the loudest one)",
    )
    ap.add_argument(
        "--bw-sweep",
        action="store_true",
        help="add the prior-sensitivity arms on one mid-recording window",
    )
    ap.add_argument("--bw-grid", default="0.5,1,2,4", help="requested bw_rps arms")
    ap.add_argument(
        "--rho-grid",
        default="0.25,4",
        help="rho_scale arms — the prior axis that survives the per-group bandwidth clamp",
    )
    ap.add_argument("--spec", default=FRAMES_SPEC)
    ap.add_argument("--label-dir", default=LABEL_DIR_DEFAULT)
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument(
        "--smoke", action="store_true", help="one 4 s flight window at k_max 20 on 2 microphones"
    )
    add_gridrun_args(ap, jobs=4)
    args = ap.parse_args()

    if args.smoke:
        args.window_s, args.hop_s = 4.0, 4.0
        args.k_max, args.mics = 20, 2
    if args.joint:
        # Fail on a malformed ladder here, in the CLI, and not in a worker.
        joint_config(
            {
                "iters": args.iters,
                "k_trust": args.k_trust,
                "bw_psi": args.bw_psi,
                "bw_theta": args.bw_theta,
            }
        )
        if args.iters < 1:
            raise SystemExit(f"--iters {args.iters}: the alternation needs at least one round")
    if args.sr <= 0:
        raise SystemExit(f"--sr {args.sr}: the sample rate must be positive")
    if args.mics > MAX_MICS:
        raise SystemExit(
            f"--mics {args.mics}: the VK solver clamps at {MAX_MICS} channels "
            "(tracking.vk_tracking._MAX_CHANNELS), which is the whole DREGON array"
        )
    args.bw_grid = parse_floats(args.bw_grid)
    args.rho_grid = parse_floats(args.rho_grid)
    # Parse once, here, so a malformed schedule fails the CLI and not a worker.
    sched = BandwidthSchedule.parse(args.bw_schedule)
    out = Path(args.out)
    fs_env = 100.0  # tracking.fitness_vk.FVKConfig.fs_env
    stride = max(1, int(round(args.sr / fs_env)))
    params = {
        "window_s": float(args.window_s),
        "hop_s": float(args.hop_s),
        "k_max": int(args.k_max),
        "mics": int(args.mics),
        "sr": int(args.sr),
        "f_max": float(args.f_max),
        "bw_rps": float(args.bw_rps),
        # Carried as the CLI string: it must survive a JSON round trip through
        # the unit parameters and the report's provenance.
        "bw_schedule": sched.text() if sched is not None else "",
        "ref_mic": int(args.ref_mic),
        "mem_budget_gb": float(args.mem_budget_gb),
        "fs_env": fs_env,
        "stride": stride,
        "joint": bool(args.joint),
        "iters": int(args.iters),
        "k_trust": str(args.k_trust),
        "bw_psi": str(args.bw_psi),
        "bw_theta": float(args.bw_theta),
        "whiten": not bool(args.no_whiten),
    }
    wanted = {v.strip() for v in args.recording.split(",") if v.strip()}

    if args.mode in ("solve", "all"):
        recs = load_recordings(args.spec, args.label_dir, int(args.sr))
        if wanted:
            recs = [r for r in recs if r["recording_id"] in wanted]
            if not recs:
                raise SystemExit(f"no recording of {sorted(wanted)} in {args.spec}")
        # Warm the cache BEFORE the pool forks: workers inherit the decoded
        # recordings and open no R2 connection (concurrent per-worker streams
        # caused SSL failures and killed the pool on the cluster).
        cache_recordings(recs, int(args.sr))
        common = {
            "spec": args.spec,
            "label_dir": args.label_dir,
            "out": str(out),
            "kind": "window",
            **params,
        }
        units = build_units(recs, args, common)
        # Size the job BEFORE it runs: one solve holds one banded group, and
        # that group is the whole comb (see group_plan).
        probe = recs[0]
        pk = recording_k_hi(probe["r_ref"], int(args.k_max), sr=int(args.sr), f_max=args.f_max)
        plan = group_plan(
            np.asarray(probe["r_audio"])[:, : int(round(args.window_s * args.sr))],
            pk,
            fvk_config(
                int(args.k_max),
                mics=args.mics,
                bw_rps=args.bw_rps,
                sr=int(args.sr),
                f_max=args.f_max,
            ),
        )
        print(
            f"[vk_decompose] {len(units)} units -> {out} at {args.sr} Hz, f_max {args.f_max:g}\n"
            f"[vk_decompose] k_hi {pk}, {plan['n_tracks']} tracks in {plan['n_groups']} coupling "
            f"group(s), largest {plan['max_group']}: {plan['banded_gb']} GB per worker, "
            f"{round(plan['banded_gb'] * args.jobs, 2)} GB for --jobs {args.jobs}",
            flush=True,
        )
        res = gridrun_from_args(args, units, solve_worker, out, summarize=summarize)
        if res.n_failed:
            raise SystemExit(res.exit_code)

    if args.mode in ("stitch", "all"):
        stitch(out, args.spec, args.label_dir, params, only=wanted or None)


if __name__ == "__main__":
    main()
