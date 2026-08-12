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

How prior-dependent the weak tracks are
    ``--bw-sweep`` re-solves one mid-recording window on two prior axes and
    reports how the per-band mean amplitude and drift move: the requested
    ``bw_rps`` (0.5, 1, 2, 4) and ``rho_scale`` (0.25, 4). The second axis is
    there because the first one is largely INERT — a coupled group clamps every
    track to ``max(VKConfig.bw_hz, 6 * line separation)``, and a dense comb
    floors that at 1 Hz, so the ``bw_rps`` arms mostly measure the clamp. Each
    arm reports the bandwidth it actually got.

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

from utils.gridrun import Unit, add_gridrun_args, gridrun_from_args, unit_path  # noqa: E402

OUT_DEFAULT = "results/vk_decompose"
#: The refined labels this decomposition is conditioned on.
LABEL_DIR_DEFAULT = "src/data_processing/refined_labels"
#: The generator's own noise source, loaded with the generator's own arguments.
FRAMES_SPEC = "frames:DREGON-frames"
RPS_KEY = "motors_measured"
SPLITS = ["in_flight_noise"]
SR = 16000
#: The frozen evaluation frame grid (``tracking.protocols.BEATVK.hop_s``) — the
#: grid the refined labels sit on, and the grid windows are cut on.
HOP_S = 0.032
#: Below this mean rate a window has no usable comb and is not decomposed.
IDLE_REV_S = 20.0
#: ``tracking.vk_tracking._MAX_CHANNELS`` — the solver's own channel clamp. It is
#: exactly the DREGON array size, so all 8 microphones go into ONE solve and no
#: channel batching is necessary.
MAX_MICS = 8
#: Harmonic bands the energy ledger and the bandwidth sweep report against.
BANDS: tuple[tuple[int, int], ...] = ((1, 9), (10, 24), (25, 49), (50, 80))
#: Welch segment of the residual and original spectra.
NPERSEG = 4096


def band_name(lo: int, hi: int) -> str:
    return f"k{lo}-{hi}"


# ---------------------------------------------------------------------------
# data


def frame_grid(n_t: int, sr: int) -> np.ndarray:
    """The uniform ``HOP_S`` frame grid of a recording, in relative seconds.

    Same construction as ``tracking.protocols.slice_window``, so a window here
    and a protocol window agree frame for frame.
    """
    return np.arange(0.0, n_t / float(sr) - HOP_S / 2, HOP_S)


def interp_rps(vals: Any, stamps: Any, ft: Any) -> np.ndarray:
    """``(R, M)`` telemetry at ``stamps`` -> ``(R, N)`` on ``ft``.

    ``noise_rps_dataset.upsample_rps_to_audio_rate`` in float64: the same
    duplicate-stamp drop and the same clip against extrapolation, but the
    carrier must not carry a float32 rounding staircase.
    """
    ts = np.asarray(stamps, dtype=np.float64)
    _, uniq = np.unique(ts, return_index=True)
    uniq = np.sort(uniq)
    ts = ts[uniq]
    v = np.asarray(vals, dtype=np.float64)[:, uniq]
    q = np.clip(np.asarray(ft, dtype=np.float64), ts[0], ts[-1])
    return np.stack([np.interp(q, ts, row) for row in v])


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
    for tf in iter_published_frames(name, version, splits=SPLITS):
        rid = meta_dict(tf).get("recording_id")
        if rid and "audio" in tf:
            starts[str(rid)] = int(tf["audio"].t_start_ticks)
    return starts


def load_recordings(spec: str, label_dir: str | Path) -> list[dict[str, Any]]:
    """Every surviving noise recording, with the REFINED labels applied.

    The loader is the generator's (``load_published_noise_sources`` with the
    generator's arguments), so the audio decomposed here is the audio training
    sees. ``rps_override_dir`` replaces the telemetry values with the refined
    trajectory before the overlap trim, which is where the sidecar's times are
    defined.

    ``ft`` is LOCAL: seconds from the trimmed frame's audio ``t_start``.
    ``t0_offset_s`` carries the trim, so every output can be written against the
    published recording.
    """
    import tdseries as td

    from data_processing.frames import meta_dict
    from data_processing.noise_rps_dataset import load_published_noise_sources

    starts = published_audio_starts(spec)
    recs: list[dict[str, Any]] = []
    for src in load_published_noise_sources(
        spec, SR, origin="dregon", rps_key=RPS_KEY, splits=SPLITS, rps_override_dir=label_dir
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
        ft = frame_grid(n_t, SR)
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
                "r_audio": to_audio_grid(r_ref, ft, n_t, SR),
                "t0_offset_s": (t0 - starts[rid]) / float(td.TICKS_PER_SECOND),
            }
        )
    if not recs:
        raise RuntimeError(f"{spec}: no recording with an {RPS_KEY} track survived loading")
    return recs


#: Per-process recording cache. Pool workers are reused across units, so each
#: process decodes the dataset once; under a fork start method it inherits the
#: parent's copy and decodes nothing.
_RECORDINGS: dict[str, dict[str, Any]] = {}


def get_recording(rid: str, spec: str, label_dir: str | Path) -> dict[str, Any]:
    if rid not in _RECORDINGS:
        _RECORDINGS.update({r["recording_id"]: r for r in load_recordings(spec, label_dir)})
    if rid not in _RECORDINGS:
        raise KeyError(f"recording {rid!r} not in {spec}")
    return _RECORDINGS[rid]


def to_audio_grid(r: Any, ft: Any, n_t: int, sr: int) -> np.ndarray:
    """``(R, N)`` rates on ``ft`` -> ``(R, T)`` at audio rate."""
    r2 = np.atleast_2d(np.asarray(r, dtype=np.float64))
    t_audio = np.arange(n_t, dtype=np.float64) / float(sr)
    return np.stack([np.interp(t_audio, np.asarray(ft, dtype=np.float64), row) for row in r2])


def shaft_phase(r_audio: Any, sr: int) -> np.ndarray:
    """``(R, T)`` fundamental shaft phase in radians.

    ``tracking.vk_tracking.vk_envelopes`` computes exactly this
    (``2 pi cumsum(r) / fs``) on the window it is given, so a window that starts
    at sample ``a0`` differs from this global phase by the constant
    ``Phi(a0 - 1)``. That constant is what :func:`stitch_envelopes` removes.
    """
    return (
        2.0
        * np.pi
        * np.cumsum(np.atleast_2d(np.asarray(r_audio, dtype=np.float64)), axis=-1)
        / float(sr)
    )


def window_bounds(n_frames: int, window_s: float, hop_s: float) -> list[tuple[int, int]]:
    """Window frame ranges over a whole recording, the last one right-aligned."""
    w = max(1, int(round(window_s / HOP_S)))
    step = max(1, int(round(hop_s / HOP_S)))
    if n_frames <= w:
        return [(0, n_frames)]
    starts = list(range(0, n_frames - w + 1, step))
    if starts[-1] + w < n_frames:
        starts.append(n_frames - w)
    return [(s, s + w) for s in starts]


def window_span(ft: Any, i0: int, i1: int, n_t: int, stride: int) -> tuple[int, int]:
    """Audio sample range of one window, snapped to the envelope stride.

    Both ends land on a multiple of ``stride``, so the window's envelope grid is
    a slice of the recording's global envelope grid. Without the snap the two
    grids are offset by a fraction of a knot (a 0.032 s frame is 3.2 knots at
    16 kHz and 100 Hz) and the stitch would have to resample.
    """
    ftv = np.asarray(ft, dtype=np.float64)
    a0 = (int(round(float(ftv[i0]) * SR)) // stride) * stride
    a1_raw = min(n_t, int(round((float(ftv[i1 - 1]) + HOP_S) * SR)))
    a1 = a0 + ((a1_raw - a0) // stride) * stride
    return a0, a1


# ---------------------------------------------------------------------------
# reconstruction and per-track statistics


def reconstruct(
    x: Any,
    k: Any,
    rotor: Any,
    phase: Any,
    stride: int,
    *,
    knots_per_chunk: int = 1024,
) -> tuple[np.ndarray, np.ndarray]:
    """``(C, M, N_env)`` envelopes -> the ``(C, T)`` track sum and per-track energy.

    Track ``m`` is ``Re[a_m(t) e^{j k_m phi(t)}]`` with ``a_m`` linearly
    interpolated from the envelope grid onto the audio grid, real and imaginary
    parts separately, and held constant beyond the last knot — the rule
    :func:`tracking.vk_reconstruct` uses. ``phase`` must already be the phase
    the envelopes are referenced to.

    The work is chunked in time and the tracks are accumulated one at a time:
    the whole ``(C, M, T)`` bank is hundreds of gigabytes at a realistic
    ``k_hi``, and only the sum and the per-track energies are wanted.
    """
    xa = np.asarray(x)
    ph = np.atleast_2d(np.asarray(phase, dtype=np.float64))
    n_ch, n_tracks, n_env = xa.shape
    n_t = int(ph.shape[-1])
    recon = np.zeros((n_ch, n_t), dtype=np.float32)
    energy = np.zeros(n_tracks, dtype=np.float64)
    if n_tracks == 0 or n_env == 0 or n_t == 0:
        return recon, energy

    ramp = np.arange(stride, dtype=np.float32) / np.float32(stride)
    ks = np.asarray(k).astype(np.float64)
    rot = np.asarray(rotor).astype(int)
    for j0 in range(0, n_env, knots_per_chunk):
        j1 = min(n_env, j0 + knots_per_chunk)
        s0, s1 = j0 * stride, min(n_t, j1 * stride)
        if s1 <= s0:
            break
        # One extra knot when there is one: the last knot of the chunk needs the
        # difference to its successor, not a hold.
        jend = min(j1 + 1, n_env)
        n_out = s1 - s0
        for m in range(n_tracks):
            vals = xa[:, m, j0:jend]
            up_r = _upsample_knots(np.real(vals).astype(np.float32), ramp, n_out)
            up_i = _upsample_knots(np.imag(vals).astype(np.float32), ramp, n_out)
            arg = ks[m] * ph[rot[m], s0:s1]
            comp = up_r * np.cos(arg).astype(np.float32) - up_i * np.sin(arg).astype(np.float32)
            recon[:, s0:s1] += comp
            energy[m] += float(np.square(comp, dtype=np.float64).sum())
    return recon, energy


def _upsample_knots(vals: Any, ramp: Any, n_out: int) -> np.ndarray:
    """``(C, J)`` knots -> ``(C, n_out)`` linear upsample on the uniform grid."""
    v = np.asarray(vals)
    if v.shape[-1] > 1:
        diffs = np.diff(v, axis=-1)
        up = (v[:, :-1, None] + diffs[:, :, None] * ramp).reshape(v.shape[0], -1)
    else:
        up = np.zeros((v.shape[0], 0), dtype=v.dtype)
    if up.shape[-1] < n_out:  # hold the last knot beyond the grid
        tail = np.repeat(v[:, -1:], n_out - up.shape[-1], axis=-1)
        up = np.concatenate([up, tail], axis=-1)
    return up[:, :n_out]


def reference_mic(audio: Any, ref_mic: int) -> int:
    """Which microphone the per-track statistics are read on.

    ``ref_mic < 0`` selects the channel with the most energy on the span. The
    DREGON array is not uniform — on one cruise window channels 1 and 4 hold
    378 and 347 units of energy against 26 to 84 for the other six — so a fixed
    channel 0 would report the phase and amplitude tables at 6 times less
    signal-to-noise ratio than the array can give.
    """
    y = np.asarray(audio, dtype=np.float64)
    if ref_mic >= 0:
        return min(int(ref_mic), int(y.shape[0]) - 1)
    return int(np.argmax((y**2).sum(axis=-1)))


def track_bands(k: Any) -> dict[str, np.ndarray]:
    """``{band name: boolean track mask}`` over the ``(M,)`` harmonic indices."""
    ks = np.asarray(k).astype(int)
    return {band_name(lo, hi): (ks >= lo) & (ks <= hi) for lo, hi in BANDS}


def drift_increments(phase_err: Any, fs_env: float) -> np.ndarray:
    """``(M, N-1)`` time derivative of the per-track phase error, in rad/s.

    The statistic every phase-model test is built from. An increment larger than
    ``pi`` means the unwrap itself is ambiguous, so the report carries the
    maximum increment beside the standard deviations.
    """
    p = np.asarray(phase_err, dtype=np.float64)
    return np.diff(p, axis=-1) * float(fs_env)


def rank_one_share(increments: Any) -> dict[str, Any]:
    """Top-eigenvalue share and mean pairwise correlation of drift increments.

    ``increments`` is ``(K, N)`` — one row per harmonic of ONE rotor. A shaft
    jitter model says every harmonic sees ``k`` times the same phase, so the
    correlation matrix is rank one and the share is 1. Independent per-harmonic
    drift (the pi-kalman model) gives a share near ``1 / K``.
    """
    d = np.asarray(increments, dtype=np.float64)
    n_k, n_t = d.shape
    if n_k < 2 or n_t < n_k + 2:
        return {"lambda1_share": None, "mean_corr": None, "n_k": int(n_k), "n_frames": int(n_t)}
    sd = d.std(axis=-1)
    keep = sd > 0
    if int(keep.sum()) < 2:
        return {"lambda1_share": None, "mean_corr": None, "n_k": int(n_k), "n_frames": int(n_t)}
    corr = np.corrcoef(d[keep])
    lam = np.sort(np.linalg.eigvalsh(corr))[::-1]
    off = corr[~np.eye(corr.shape[0], dtype=bool)]
    return {
        "lambda1_share": round(float(lam[0] / max(lam.sum(), 1e-30)), 6),
        "mean_corr": round(float(off.mean()), 6),
        "n_k": int(keep.sum()),
        "n_frames": int(n_t),
    }


def per_track_stats(
    amp: Any, phase_err: Any, mask: Any, fs_env: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(mean amplitude, amplitude CV, drift std)`` per track, on masked frames.

    ``amp`` / ``phase_err`` are ``(M, N)`` — ONE microphone. The mask selects the
    frames that are covered and not idle; drift uses the frame PAIRS inside it.
    """
    a = np.asarray(amp, dtype=np.float64)
    m = np.asarray(mask, dtype=bool)
    mean = a[:, m].mean(axis=-1) if m.any() else np.zeros(a.shape[0])
    std = a[:, m].std(axis=-1) if m.any() else np.zeros(a.shape[0])
    cv = std / np.maximum(mean, 1e-30)
    inc = drift_increments(phase_err, fs_env)
    pair = m[:-1] & m[1:]
    drift = inc[:, pair].std(axis=-1) if pair.any() else np.zeros(a.shape[0])
    return mean, cv, drift


def band_summary(values: Any, k: Any) -> dict[str, float | None]:
    """Band means of a per-track quantity — ``None`` for a band with no track."""
    v = np.asarray(values, dtype=np.float64)
    return {
        name: (round(float(v[sel].mean()), 8) if sel.any() else None)
        for name, sel in track_bands(k).items()
    }


# ---------------------------------------------------------------------------
# the solve unit


def fvk_config(k_max: int, *, mics: int = MAX_MICS, bw_rps: float = 1.0, sr: int = SR) -> Any:
    """THE measurement geometry — one construction, so every solve agrees.

    ``f_max`` is the campaign's 6 kHz ceiling, held below three quarters of the
    Nyquist frequency so a lower sample rate (the tests) keeps a modelled
    harmonic inside the band instead of on top of it.
    """
    from tracking.fitness_vk import FVKConfig

    return FVKConfig(
        sr=int(sr),
        k_max=int(k_max),
        f_max=min(6000.0, 0.375 * float(sr)),
        max_channels=int(mics),
        bw_rps=float(bw_rps),
    )


def solve_window(
    audio: Any,
    r_audio: Any,
    k_hi: int,
    k_max: int,
    bw_rps: float,
    mics: int,
    *,
    sr: int = SR,
    rho_scale: float = 1.0,
) -> Any:
    """One coupled VK solve of one window — the whole numerical content.

    The validity mask is disabled and the harmonic set is capped from the
    RECORDING's reference trajectory (see :func:`recording_k_hi`), so every
    window of a recording holds the identical ``(rotor, harmonic)`` track set
    and the windows can be stitched track by track.
    """
    from tracking.fitness_vk import solve_envelopes

    cfg = fvk_config(k_max, mics=mics, bw_rps=bw_rps, sr=sr)
    y = np.ascontiguousarray(np.asarray(audio, dtype=np.float64)[:mics])
    return cfg, solve_envelopes(
        y,
        np.asarray(r_audio, dtype=np.float64),
        cfg,
        k_hi=int(k_hi),
        rho_scale=float(rho_scale),
    )


def group_plan(r_audio: Any, k_hi: int, cfg: Any) -> dict[str, Any]:
    """Coupling-group partition of one window, and the memory it will cost.

    THE memory model of this script, and it must be read before a job is sized.
    :func:`tracking.vk_tracking._coupling_groups` is a union-find over the pairs
    whose lines come within ``couple_hz`` (50 Hz here), so coupling is
    TRANSITIVE: at ``k_hi`` 62 the four rotors put 248 lines into 0 to 5.7 kHz,
    a mean spacing of 23 Hz, and the chain merges 244 of them into ONE group.

    One group of ``g`` tracks is solved as a Hermitian banded system of ``g``
    times ``n_env`` unknowns with ``2 g`` superdiagonals, and the factorization
    holds a second copy, so

        bytes = 2 (2 g + 1) g n_env 16 ,   approximately 1e-4 k_hi^2 window_s GB

    with ``g`` about ``4 k_hi`` and ``n_env`` about ``100 window_s``. Channels
    are right-hand sides and cost nothing here. The default full-recording
    configuration (``k_hi`` 62, 16 s windows) therefore needs 6.3 GB per WORKER,
    which is what a local three-worker run could not hold.
    """
    from tracking.vk_tracking import _coupling_groups, _track_table, env_stride

    vk = cfg.vk_config(int(k_hi))
    stride, fs_env = env_stride(vk)
    r = np.atleast_2d(np.asarray(r_audio, dtype=np.float64))
    rot, ks = _track_table(int(r.shape[0]), vk.k_min, int(k_hi))
    f = ks[:, None].astype(np.float64) * r[:, ::stride][rot]
    valid = f <= min(vk.f_max, 0.45 * vk.fs)
    couple = fs_env / 2.0 if vk.couple_hz is None else float(vk.couple_hz)
    groups = _coupling_groups(f, valid, couple)
    g = max((len(x) for x in groups), default=0)
    n_env = int(f.shape[-1])
    return {
        "n_tracks": int(len(ks)),
        "n_groups": len(groups),
        "max_group": int(g),
        "n_env": n_env,
        "banded_gb": round(2.0 * (2 * g + 1) * g * n_env * 16 / 1e9, 3),
    }


def recording_k_hi(r_ref: Any, k_max: int) -> int:
    """The harmonic cap of a WHOLE recording, from its refined labels.

    ``tracking.fitness_vk.k_cap`` reads the maximum rate of the reference it is
    given. Giving it the whole recording (and not the window) is what keeps the
    track set identical across windows, which the stitch needs; it is also the
    safe direction, because the cap of the fastest window is the smallest one.
    """
    from tracking.fitness_vk import k_cap

    return int(k_cap(fvk_config(k_max), np.asarray(r_ref)))


def solve_worker(unit: Unit) -> dict[str, Any]:
    """One unit: one window (``kind`` = ``window``) or one bandwidth (``bw``).

    A window unit writes its complex envelopes to ``<out>/raw/<uid>.npz`` and
    records the path in its JSON — a complex bank does not belong in JSON. A
    bandwidth unit reports band statistics only and writes no array.
    """
    p = dict(unit.params)
    rec = get_recording(str(p["recording"]), str(p["spec"]), str(p["label_dir"]))
    i0, i1 = int(p["i0"]), int(p["i1"])
    mics = min(int(p["mics"]), int(rec["audio"].shape[0]))
    stride = int(p["stride"])
    n_t = int(rec["audio"].shape[-1])
    a0, a1 = window_span(rec["ft"], i0, i1, n_t, stride)
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
        "start_s": round(a0 / float(SR) + offset, 6),
        "end_s": round(a1 / float(SR) + offset, 6),
        "mean_rev_s": round(mean_rev_s, 4),
        "mics": mics,
        "k_max": int(p["k_max"]),
        "bw_rps": float(p["bw_rps"]),
        "rho_scale": float(p.get("rho_scale", 1.0)),
    }
    if mean_rev_s < IDLE_REV_S:
        return {**out, "used": False, "reason": "idle"}

    k_hi = recording_k_hi(rec["r_ref"], int(p["k_max"]))
    cfg = fvk_config(int(p["k_max"]), mics=mics, bw_rps=float(p["bw_rps"]))
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
    _, env = solve_window(
        rec["audio"][:, a0:a1],
        r_win,
        k_hi,
        int(p["k_max"]),
        float(p["bw_rps"]),
        mics,
        rho_scale=float(p.get("rho_scale", 1.0)),
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
    np.savez(
        npz,
        allow_pickle=False,
        x=np.asarray(env.x, dtype=np.complex64),
        valid=np.asarray(env.valid, dtype=bool),
        rotor=np.asarray(env.rotor, dtype=np.int64),
        k=np.asarray(env.k, dtype=np.int64),
        bw_track=np.asarray(env.bw_track, dtype=np.float64),
    )
    # A cheap per-window capture check on ONE channel: the fraction of that
    # channel's energy the track sum explains. The full ledger is the stitch's,
    # which is why this reconstructs one channel and not the array.
    phase_w = shaft_phase(r_win, SR)
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


def fade_weights(n_win: int, ramp: int) -> np.ndarray:
    """Linear cross-fade weights over one window's envelope frames.

    The floor keeps the weight positive everywhere, so a frame only one window
    covers (the two ends of a recording) still resolves to that window.
    """
    idx = np.arange(n_win, dtype=np.float64)
    if ramp <= 0:
        return np.ones(n_win)
    rise = np.minimum(idx, idx[::-1]) + 1.0
    return np.clip(rise / (ramp + 1.0), 1e-3, 1.0)


def stitch_envelopes(
    rows: list[dict[str, Any]], out: Path, phi: Any, stride: int, ramp: int
) -> dict[str, Any]:
    """Cross-fade the window envelopes of one recording onto one global grid.

    Every window is first re-referenced to the global shaft phase: the solver's
    ``phase`` starts at the window, so track ``m`` of the window that starts at
    sample ``a0`` must be multiplied by ``exp(-j k_m Phi_rotor(a0 - 1))``. Then
    real and imaginary parts are cross-faded with linear ramps over the overlap.

    Returns the stitched bank plus the span it covers.
    """
    used = sorted((r for r in rows if r.get("used") and "npz" in r), key=lambda r: int(r["a0"]))
    if not used:
        raise SystemExit("no usable window unit — run --mode solve first")
    a_min = min(int(r["a0"]) for r in used)
    a_max = max(int(r["a1"]) for r in used)
    e0, e1 = a_min // stride, a_max // stride
    n_env = e1 - e0

    first = np.load(out / used[0]["npz"])
    rotor = np.asarray(first["rotor"], dtype=np.int64)
    k = np.asarray(first["k"], dtype=np.int64)
    bw_track = np.asarray(first["bw_track"], dtype=np.float64)
    n_ch, n_tracks = int(first["x"].shape[0]), int(first["x"].shape[1])
    first.close()

    num = np.zeros((n_ch, n_tracks, n_env), dtype=np.complex64)
    den = np.zeros(n_env, dtype=np.float64)
    valid = np.zeros((n_tracks, n_env), dtype=bool)
    for r in used:
        with np.load(out / r["npz"]) as data:
            x = np.asarray(data["x"], dtype=np.complex64)
            v = np.asarray(data["valid"], dtype=bool)
            if not np.array_equal(np.asarray(data["k"], dtype=np.int64), k):
                raise SystemExit(
                    f"{r['npz']}: harmonic set differs from the first window — the windows "
                    "were solved at different k_hi and cannot be stitched"
                )
        a0 = int(r["a0"])
        j0 = a0 // stride - e0
        n_w = int(x.shape[-1])
        # The window's own phase origin, removed. Phi(-1) is 0 by definition.
        shift = np.zeros(int(np.max(rotor)) + 1) if a0 == 0 else np.asarray(phi)[:, a0 - 1]
        x *= np.exp(-1j * k[None, :, None] * shift[rotor][None, :, None]).astype(np.complex64)
        w = fade_weights(n_w, min(ramp, n_w // 2))
        num[:, :, j0 : j0 + n_w] += x * w[None, None, :].astype(np.complex64)
        den[j0 : j0 + n_w] += w
        valid[:, j0 : j0 + n_w] |= v

    covered = den > 0.0
    num /= np.maximum(den, 1e-12).astype(np.float32)[None, None, :]
    valid &= covered[None, :]
    return {
        "x": num,
        "valid": valid,
        "rotor": rotor,
        "k": k,
        "bw_track": bw_track,
        "covered": covered,
        "a_min": a_min,
        "a_max": a_max,
        "env_i0": e0,
        "n_env": n_env,
        "windows": used,
    }


def check_phase_reference(rec: dict[str, Any], phi: Any, a0: int) -> float:
    """Maximum radians by which the window phase and the global phase disagree.

    The re-reference of :func:`stitch_envelopes` assumes
    ``phase_window(t) = Phi(t) - Phi(a0 - 1)``. That is an identity only while
    the window's carrier is the same array the global phase was built from, so
    it is measured on one window instead of assumed.
    """
    a1 = min(int(np.asarray(phi).shape[-1]), a0 + 4 * SR)
    local = shaft_phase(np.asarray(rec["r_audio"])[:, a0:a1], SR)
    shift = 0.0 if a0 == 0 else np.asarray(phi)[:, a0 - 1 : a0]
    return float(np.abs(local - (np.asarray(phi)[:, a0:a1] - shift)).max())


def energy_ledger(audio: Any, recon: Any, track_energy: Any, k: Any) -> dict[str, Any]:
    """Total / track / residual / cross-term energy, and the per-band shares.

    The tracks are not orthogonal — neighbouring harmonics of two rotors overlap
    inside their bands — so the three parts do not add up to the total by
    themselves. The cross term is what is left, and its size is the honest
    statement of how much of the decomposition is interference between tracks.
    """
    y = np.asarray(audio, dtype=np.float64)
    rec = np.asarray(recon, dtype=np.float64)
    total = float((y**2).sum())
    resid = float(((y - rec) ** 2).sum())
    tracks = float(np.asarray(track_energy, dtype=np.float64).sum())
    e_k = np.asarray(track_energy, dtype=np.float64)
    shares = {
        name: round(float(e_k[sel].sum() / max(tracks, 1e-30)), 6)
        for name, sel in track_bands(k).items()
    }
    return {
        "total": total,
        "per_channel_total": [round(float(v), 6) for v in (y**2).sum(axis=-1)],
        "per_channel_residual": [round(float(v), 6) for v in ((y - rec) ** 2).sum(axis=-1)],
        "tracks": tracks,
        "residual": resid,
        "cross_term": total - tracks - resid,
        "track_fraction": round(tracks / max(total, 1e-30), 6),
        "residual_fraction": round(resid / max(total, 1e-30), 6),
        "band_share_of_tracks": shares,
    }


def phase_model_report(
    amp0: Any, pherr0: Any, valid: Any, rotor: Any, k: Any, mask: Any, fs_env: float
) -> dict[str, Any]:
    """The per-rotor phase and amplitude tables, on the reference microphone.

    Two readings of the same increments: the drift standard deviation against
    ``k`` (does the drift grow with the harmonic, as a shaft model says) and the
    rank-one share (do the harmonics drift together, as a shaft model says).

    ``max_abs_step_rad`` is the guard on both: the phase error is unwrapped, so
    a step at or above ``pi`` radians makes the unwrap ambiguous and the drift
    of that harmonic is then a lower bound, not a measurement.
    """
    mask = np.asarray(mask, dtype=bool) & np.asarray(valid, dtype=bool).all(axis=0)
    mean, cv, drift = per_track_stats(amp0, pherr0, mask, fs_env)
    inc = drift_increments(pherr0, fs_env)
    pair = mask[:-1] & mask[1:]
    rot = np.asarray(rotor, dtype=int)
    ks = np.asarray(k, dtype=int)
    per_rotor: dict[str, Any] = {}
    for rr in sorted(set(rot.tolist())):
        sel = rot == rr
        order = np.argsort(ks[sel])
        idx = np.flatnonzero(sel)[order]
        per_rotor[str(rr)] = {
            "k": [int(v) for v in ks[idx]],
            "drift_std_rad_s": [round(float(v), 5) for v in drift[idx]],
            "amp_mean": [round(float(v), 8) for v in mean[idx]],
            "amp_cv": [round(float(v), 5) for v in cv[idx]],
            "rank_one": rank_one_share(inc[np.ix_(idx, np.flatnonzero(pair))]),
        }
    return {
        "n_frames": int(mask.sum()),
        "max_abs_step_rad": round(
            float(np.abs(inc[:, pair]).max() / fs_env) if pair.any() else 0.0, 5
        ),
        "max_abs_drift_rad_s": round(float(np.abs(inc[:, pair]).max()) if pair.any() else 0.0, 4),
        "drift_std_rad_s_by_band": band_summary(drift, k),
        "amp_mean_by_band": band_summary(mean, k),
        "amp_cv_by_band": band_summary(cv, k),
        "per_rotor": per_rotor,
    }


def welch_psd(audio: Any, sr: int) -> tuple[np.ndarray, np.ndarray]:
    from scipy.signal import welch

    f, p = welch(np.asarray(audio, dtype=np.float64), fs=float(sr), nperseg=NPERSEG, axis=-1)
    return np.asarray(f, dtype=np.float64), np.asarray(p, dtype=np.float64)


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
    written: list[Path] = []

    ids = {str(r["recording"]) for r in rows}
    for rid in sorted(ids if only is None else ids & only):
        got = [r for r in rows if r["recording"] == rid]
        wins = [r for r in got if r.get("kind", "window") == "window"]
        rec = get_recording(rid, spec, label_dir)
        offset = float(rec["t0_offset_s"])
        phi = shaft_phase(rec["r_audio"], SR)
        ramp = max(0, int(round((params["window_s"] - params["hop_s"]) * params["fs_env"])))
        st = stitch_envelopes(wins, out, phi, stride, ramp)

        a_min, a_max, n_env = int(st["a_min"]), int(st["a_max"]), int(st["n_env"])
        x = st["x"]
        k, rotor = st["k"], st["rotor"]
        t_env = (a_min + np.arange(n_env) * stride) / float(SR) + offset
        audio = np.asarray(rec["audio"][: x.shape[0], a_min:a_max], dtype=np.float64)
        recon, track_energy = reconstruct(x, k, rotor, phi[:, a_min:a_max], stride)

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
            sample_rate=np.int64(SR),
            t0_offset_s=np.float64(offset),
            span_samples=np.asarray([a_min, a_max], dtype=np.int64),
            recording_id=np.array(rid),
            spec=np.array(spec),
            rps_key=np.array(RPS_KEY),
            label_dir=np.array(str(label_dir)),
            time_reference=np.array(
                "seconds from the published recording's audio t_start (the full frame, "
                "before the loader's telemetry-overlap trim of t0_offset_s)"
            ),
        )

        residual = (audio - recon).astype(np.float32)
        f_psd, psd_res = welch_psd(residual, SR)
        _, psd_org = welch_psd(audio, SR)
        np.savez(
            rec_dir / "residual.npz",
            allow_pickle=False,
            residual=residual,
            freq_hz=f_psd,
            psd_residual=psd_res,
            psd_original=psd_org,
            sample_rate=np.int64(SR),
            t_start_s=np.float64(a_min / float(SR) + offset),
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
                "rps_key": RPS_KEY,
                "label_dir": str(label_dir),
                "sample_rate": SR,
                "splits": SPLITS,
            },
            "time_reference": (
                "seconds from the published recording's audio t_start (the full frame, "
                "before the loader's telemetry-overlap trim of t0_offset_s)"
            ),
            "t0_offset_s": round(offset, 6),
            "params": {**params, "idle_rev_s": IDLE_REV_S},
            "span_s": [
                round(a_min / float(SR) + offset, 6),
                round(a_max / float(SR) + offset, 6),
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
            "bw_track_hz_by_band": band_summary(st["bw_track"], k),
            "phase_reference_max_dev_rad": float(
                check_phase_reference(rec, phi, int(st["windows"][0]["a0"]))
            ),
            "ref_mic": ref,
            "phase_model": phase_model_report(
                amp[ref], pherr[ref], st["valid"], rotor, k, mask, float(params["fs_env"])
            ),
            "windows": [
                {kk: vv for kk, vv in r.items() if kk != "npz"}
                for r in sorted(wins, key=lambda r: int(r["a0"]))
            ],
        }
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
    ap.add_argument("--bw-rps", type=float, default=1.0, help="k-scaled VK bandwidth, rev/s")
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
    if args.mics > MAX_MICS:
        raise SystemExit(
            f"--mics {args.mics}: the VK solver clamps at {MAX_MICS} channels "
            "(tracking.vk_tracking._MAX_CHANNELS), which is the whole DREGON array"
        )
    args.bw_grid = parse_floats(args.bw_grid)
    args.rho_grid = parse_floats(args.rho_grid)
    out = Path(args.out)
    fs_env = 100.0  # tracking.fitness_vk.FVKConfig.fs_env
    stride = max(1, int(round(SR / fs_env)))
    params = {
        "window_s": float(args.window_s),
        "hop_s": float(args.hop_s),
        "k_max": int(args.k_max),
        "mics": int(args.mics),
        "bw_rps": float(args.bw_rps),
        "ref_mic": int(args.ref_mic),
        "mem_budget_gb": float(args.mem_budget_gb),
        "fs_env": fs_env,
        "stride": stride,
    }
    wanted = {v.strip() for v in args.recording.split(",") if v.strip()}

    if args.mode in ("solve", "all"):
        recs = load_recordings(args.spec, args.label_dir)
        if wanted:
            recs = [r for r in recs if r["recording_id"] in wanted]
            if not recs:
                raise SystemExit(f"no recording of {sorted(wanted)} in {args.spec}")
        # Warm the cache BEFORE the pool forks: workers inherit the decoded
        # recordings and open no R2 connection (concurrent per-worker streams
        # caused SSL failures and killed the pool on the cluster).
        _RECORDINGS.update({r["recording_id"]: r for r in recs})
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
        pk = recording_k_hi(probe["r_ref"], int(args.k_max))
        plan = group_plan(
            np.asarray(probe["r_audio"])[:, : int(round(args.window_s * SR))],
            pk,
            fvk_config(int(args.k_max), mics=args.mics, bw_rps=args.bw_rps),
        )
        print(
            f"[vk_decompose] {len(units)} units -> {out}\n"
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
