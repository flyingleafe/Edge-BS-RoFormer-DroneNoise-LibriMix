#!/usr/bin/env python3
"""Refine the DREGON generator-training telemetry with the L-BFGS F_VK refiner.

The generator trains on DREGON ``in_flight_noise`` recordings whose rotor-speed
labels are the raw tachometer telemetry, and that telemetry carries a
scale-like error of 0.35-0.85 % (``docs/experiments/telemetry-fitness.md`` § 6d).
Phase 7 measured what that costs: a constant 0.5 % label bias alone takes 8.6 dB
off the generated line energy at harmonics 50-80. This driver replaces the
labels of the whole recording with the trajectory that
:func:`tracking.fitness_vk.optimize_trajectory` finds — window by window, then
stitched — and writes a committed sidecar per recording.

The data path is the GENERATOR's, not a protocol's: recordings come from
``data_processing.noise_rps_dataset.load_published_noise_sources`` with exactly
the arguments generator training uses, and the audio and telemetry refined here
are the ones training sees. The sidecar's times are seconds from the audio
``t_start`` of the PUBLISHED recording — the loader trims each frame to the
audio-telemetry overlap (5.48 s on ``free-flight_nosource_room1``) and the trim
is added back at stitch time, so a consumer can apply the labels to the
untrimmed frame. Every epoch conversion runs in integer ticks: the published
frames sit at ~1e18 ticks, which float64 cannot subtract without loss.

Three things this driver adds on top of the refiner:

``smooth_lambda`` is rescaled per window
    The refiner's log-domain prior is ``sum (Delta log r)^2`` on the decimated
    envelope grid, and its default weight 1.0 is calibrated for a cruise window
    (prior ~0.8). On a takeoff ramp the same prior reads 244, so a fixed weight
    swamps the data term and the window cannot move at all. Each unit measures
    the prior of its own telemetry init (``p0``) and uses
    ``lambda = 1.0 if p0 <= 0.5 else 0.5 / p0``, so the prior never exceeds half
    the (normalized, order-1) data term.

Two acceptance guards
    A window falls back to its telemetry when the F_VK objective did not improve
    or when the refinement moved a rotor by more than ``MAX_MOVE_REV_S``. At
    cruise the expected movement is the known -0.6 % scale, about 0.4 rev/s.

Idle windows are not refined
    Below ``IDLE_REV_S`` there is no comb to fit; the telemetry passes through.

Run::

    # smoke: one 4 s cruise window, k_max 10, one channel — about a minute
    PYTHONPATH=src python scripts/refine_dregon_rps.py --smoke

    # full run on a cluster (refine only), then stitch locally after a pull
    omnirun submit --backend uni-cpu --gpus 0 --cpus 16 --mem 64 --time 24h \
      --env PYTHONPATH=src -- \
      python scripts/refine_dregon_rps.py --mode refine --jobs 6
    PYTHONPATH=src python scripts/refine_dregon_rps.py --mode stitch

Outputs: the gridrun units under ``<out>/raw/``, ``<out>/summary.json``, and per
recording ``src/data_processing/refined_labels/<recording_id>.{npz,report.json}``.
The npz files are small and are meant to be committed.
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from utils.gridrun import Unit, add_gridrun_args, gridrun_from_args  # noqa: E402

OUT_DEFAULT = "results/refine_dregon_rps"
#: Committed sidecars. Small (one float64 array per recording at 31.25 Hz).
LABEL_DIR_DEFAULT = "src/data_processing/refined_labels"
#: The generator's own noise source, loaded with the generator's own arguments.
FRAMES_SPEC = "frames:DREGON-frames"
RPS_KEY = "motors_measured"
SPLITS = ["in_flight_noise"]
SR = 16000
#: The frozen evaluation frame grid (``tracking.protocols.BEATVK.hop_s``), kept
#: here so a sidecar lands on the same grid every campaign reads.
HOP_S = 0.032
#: Below this mean rate a window has no usable comb — telemetry passes through.
IDLE_REV_S = 20.0
#: A refinement that moves a rotor more than this is not a label correction.
MAX_MOVE_REV_S = 3.0
#: Windows above this mean rate are the cruise pool the report pools over.
CRUISE_REV_S = 45.0
#: The k_max ladder the schedule is cut out of (``fitness_vk.DEFAULT_SCHEDULE``).
K_LADDER = (5, 10, 20, 40, 80)


# ---------------------------------------------------------------------------
# data


def frame_grid(n_t: int, sr: int) -> Any:
    """The uniform ``HOP_S`` frame grid of a recording, in relative seconds.

    Same construction as ``tracking.protocols.slice_window`` so a sidecar and a
    protocol window agree frame for frame.
    """
    import numpy as np

    return np.arange(0.0, n_t / float(sr) - HOP_S / 2, HOP_S)


def interp_rps(vals: Any, stamps: Any, ft: Any) -> Any:
    """``(R, M)`` telemetry at ``stamps`` -> ``(R, N)`` on ``ft``.

    ``noise_rps_dataset.upsample_rps_to_audio_rate`` in float64: the same
    duplicate-stamp drop and the same clip against extrapolation, but the
    refiner's init must not carry a float32 rounding staircase.
    """
    import numpy as np

    ts = np.asarray(stamps, dtype=np.float64)
    _, uniq = np.unique(ts, return_index=True)
    uniq = np.sort(uniq)
    ts = ts[uniq]
    v = np.asarray(vals, dtype=np.float64)[:, uniq]
    q = np.clip(np.asarray(ft, dtype=np.float64), ts[0], ts[-1])
    return np.stack([np.interp(q, ts, row) for row in v])


def published_audio_starts(spec: str) -> dict[str, int]:
    """``{recording id: audio t_start in ticks}`` of the UNTRIMMED frames.

    The sidecar's time reference. ``load_published_noise_sources`` trims each
    frame to the audio-telemetry overlap (5.48 s into DREGON's
    ``free-flight_nosource_room1``), so the trimmed frame's own ``t_start``
    would put the labels 5.48 s early for a consumer that applies them to the
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


def load_recordings(spec: str) -> list[dict[str, Any]]:
    """Every surviving generator noise recording, as plain arrays.

    The loader is the generator's (``load_published_noise_sources`` with the
    generator's arguments), and the frames it RETURNS are used — it time-slices
    the frame onto the audio-telemetry overlap, so taking the frame back from it
    is what makes these arrays byte-identical to the training ones.

    ``ft`` is LOCAL: seconds from the trimmed frame's audio ``t_start``, which
    is what the refiner needs. ``t0_offset_s`` carries the trim, so the sidecar
    can be written against the published recording (see :func:`stitch`).
    """
    import numpy as np
    import tdseries as td

    from data_processing.frames import meta_dict
    from data_processing.noise_rps_dataset import load_published_noise_sources

    starts = published_audio_starts(spec)
    recs: list[dict[str, Any]] = []
    for src in load_published_noise_sources(
        spec, SR, origin="dregon", rps_key=RPS_KEY, splits=SPLITS
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
        ft = frame_grid(int(audio.shape[-1]), SR)
        recs.append(
            {
                "recording_id": rid,
                "audio": audio,
                "ft": ft,
                "r_tel": interp_rps(np.asarray(rps_s.data), stamps, ft),
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


def get_recording(rid: str, spec: str) -> dict[str, Any]:
    if rid not in _RECORDINGS:
        _RECORDINGS.update({r["recording_id"]: r for r in load_recordings(spec)})
    if rid not in _RECORDINGS:
        raise KeyError(f"recording {rid!r} not in {spec}")
    return _RECORDINGS[rid]


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


# ---------------------------------------------------------------------------
# the refinement unit


def make_schedule(k_max: int, spec: list[list[int]] | None) -> Any:
    """The continuation schedule: the ``K_LADDER`` rungs up to ``k_max``."""
    from tracking.fitness_vk import FVKStage

    if spec:
        return tuple(FVKStage(int(k), 1.0, int(n)) for k, n in spec)
    rungs = [k for k in K_LADDER if k < k_max] + [int(k_max)]
    return tuple(FVKStage(k) for k in rungs)


def init_prior(r_init: Any, ft: Any, n_t: int, cfg: Any) -> float:
    """The refiner's own smoothness prior at the telemetry init.

    Reads through the refiner's private helpers on purpose: the rescale is only
    correct if it measures the SAME quantity the optimizer adds to its loss —
    the log-domain second difference on the decimated envelope grid.
    """
    import torch

    from tracking.fitness_vk import _smoothness, _to_audio_grid

    r_audio = _to_audio_grid(r_init, ft, n_t, cfg.sr)
    return float(_smoothness(torch.from_numpy(r_audio), cfg.stride))


def scale_shift_pct(refined: Any, init: Any) -> list[float]:
    """Per-rotor mean rate shift, in percent of the telemetry."""
    import numpy as np

    a = np.asarray(init, dtype=np.float64)
    b = np.asarray(refined, dtype=np.float64)
    ratio = b / np.maximum(np.abs(a), 1e-9)
    return [round(float(100.0 * (row.mean() - 1.0)), 5) for row in ratio]


def refine_worker(unit: Unit) -> dict[str, Any]:
    """One (recording, window) unit: L-BFGS on F_VK from the telemetry init."""
    import numpy as np

    from tracking.fitness_vk import FVKConfig, fvk_score, optimize_trajectory

    p = dict(unit.params)
    rec = get_recording(str(p["recording"]), str(p["spec"]))
    i0, i1 = int(p["i0"]), int(p["i1"])
    ft, r_tel = rec["ft"], rec["r_tel"]
    n_t = int(rec["audio"].shape[-1])

    a0 = int(round(float(ft[i0]) * SR))
    a1 = min(n_t, int(round((float(ft[i1 - 1]) + HOP_S) * SR)))
    channels = int(p["channels"])
    audio = np.ascontiguousarray(rec["audio"][:channels, a0:a1], dtype=np.float64)
    ft_w = np.asarray(ft[i0:i1], dtype=np.float64) - float(ft[i0])
    r_init = np.asarray(r_tel[:, i0:i1], dtype=np.float64)
    mean_rev_s = float(r_init.mean())

    # Frame indices are LOCAL (they index the trimmed grid the stitch uses);
    # the reported seconds carry the trim, so they read on the recording.
    offset = float(rec["t0_offset_s"])
    out: dict[str, Any] = {
        "recording": str(p["recording"]),
        "i0": i0,
        "i1": i1,
        "start_s": round(float(ft[i0]) + offset, 6),
        "end_s": round(float(ft[i1 - 1]) + HOP_S + offset, 6),
        "n_frames": int(i1 - i0),
        "mean_rev_s": round(mean_rev_s, 4),
        "k_max": int(p["k_max"]),
        "channels": channels,
    }
    if mean_rev_s < IDLE_REV_S:
        return {**out, "used": False, "reason": "idle", "r_window": r_init.tolist()}

    cfg = FVKConfig(sr=SR, k_max=int(p["k_max"]), max_channels=channels)
    p0 = init_prior(r_init, ft_w, int(audio.shape[-1]), cfg)
    lam = 1.0 if p0 <= 0.5 else 0.5 / p0

    # The reference is the telemetry for the score and for the refiner, so the
    # harmonic cap and the cell set are the window's and not the candidate's.
    before = fvk_score(audio, float(SR), r_init, ft_w, cfg, reference=r_init)
    tic = time.perf_counter()
    r_ref, diag = optimize_trajectory(
        audio,
        float(SR),
        r_init,
        ft_w,
        cfg,
        schedule=make_schedule(int(p["k_max"]), p.get("schedule")),
        knot_s=float(p["knot_s"]),
        smooth_lambda=lam,
        lr=float(p["lr"]),
    )
    wall = time.perf_counter() - tic
    after = fvk_score(audio, float(SR), r_ref, ft_w, cfg, reference=r_init)

    move_max = float(np.abs(r_ref - r_init).max())
    used, reason = True, "ok"
    if after["objective"] >= before["objective"]:
        used, reason = False, "no_improvement"
    elif move_max > MAX_MOVE_REV_S:
        used, reason = False, "move_too_large"
    r_window = r_ref if used else r_init

    return {
        **out,
        "used": used,
        "reason": reason,
        "smooth_lambda": round(lam, 9),
        "prior_init": round(p0, 6),
        "scale_pct_per_rotor": scale_shift_pct(r_ref, r_init),
        "move_max": round(move_max, 5),
        "objective_before": round(float(before["objective"]), 8),
        "objective_after": round(float(after["objective"]), 8),
        "r2_before": round(float(before["r2"]), 6),
        "r2_after": round(float(after["r2"]), 6),
        "wall_s": round(wall, 2),
        "stages": [
            {k: s[k] for k in ("k_max", "n_evals", "loss_start", "loss_end", "wall_s")}
            for s in diag["stages"]
        ],
        "k_cap": int(diag["k_cap"]),
        "r_window": np.asarray(r_window, dtype=np.float64).tolist(),
    }


# ---------------------------------------------------------------------------
# stitch


def read_rows(out: Path) -> list[dict[str, Any]]:
    raw = out / "raw"
    return [json.loads(p.read_text()) for p in sorted(raw.glob("*.json"))] if raw.is_dir() else []


def fade_weights(n_win: int, ramp: int) -> Any:
    """Linear cross-fade weights over a window's frames.

    The floor keeps the weight positive everywhere, so a frame that only one
    window covers (the two ends of a recording) still resolves to that window.
    """
    import numpy as np

    idx = np.arange(n_win, dtype=np.float64)
    if ramp <= 0:
        return np.ones(n_win)
    rise = np.minimum(idx, idx[::-1]) + 1.0
    return np.clip(rise / (ramp + 1.0), 1e-3, 1.0)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Pooled unit table — counts, cruise scale shift, objective move, wall."""
    import numpy as np

    def mean(vals: list[float]) -> float | None:
        v = np.asarray(vals, dtype=np.float64)
        v = v[np.isfinite(v)]
        return round(float(v.mean()), 6) if v.size else None

    used = [r for r in rows if r.get("used")]
    cruise = [r for r in used if float(r.get("mean_rev_s", 0.0)) > CRUISE_REV_S]
    per_rec: dict[str, Any] = {}
    for rid in sorted({str(r["recording"]) for r in rows}):
        got = [r for r in rows if r["recording"] == rid]
        per_rec[rid] = {
            "n_windows": len(got),
            "n_used": sum(1 for r in got if r.get("used")),
            "reasons": sorted({str(r.get("reason", "")) for r in got if not r.get("used")}),
        }
    return {
        "n_units": len(rows),
        "n_used": len(used),
        "per_recording": per_rec,
        "cruise_scale_pct": mean([float(np.mean(r["scale_pct_per_rotor"])) for r in cruise]),
        "d_objective": mean(
            [
                float(r["objective_after"] - r["objective_before"])
                for r in used
                if "objective_after" in r
            ]
        ),
        "wall_s": mean([float(r["wall_s"]) for r in rows if "wall_s" in r]),
    }


def stitch(out: Path, label_dir: Path, spec: str, params: dict[str, Any]) -> list[Path]:
    """Combine the window units of each recording into one full-length label set.

    The sidecar's ``ft`` is seconds from the PUBLISHED recording's audio
    ``t_start`` — the loader's overlap trim (``t0_offset_s``) is added back
    here, because a consumer applies the labels to the untrimmed frame. It
    therefore starts at ``t0_offset_s`` and covers the telemetry span only.
    """
    import numpy as np

    rows = read_rows(out)
    if not rows:
        raise SystemExit(f"{out}/raw is empty — run --mode refine first")
    label_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for rid in sorted({str(r["recording"]) for r in rows}):
        got = sorted((r for r in rows if r["recording"] == rid), key=lambda r: int(r["i0"]))
        rec = get_recording(rid, spec)
        ft = np.asarray(rec["ft"], dtype=np.float64)
        r_tel = np.asarray(rec["r_tel"], dtype=np.float64)
        num = np.zeros_like(r_tel)
        den = np.zeros(r_tel.shape[-1], dtype=np.float64)
        ramp = max(0, int(round((params["window_s"] - params["hop_s"]) / HOP_S)))
        for r in got:
            i0, i1 = int(r["i0"]), int(r["i1"])
            arr = np.asarray(r["r_window"], dtype=np.float64)
            w = fade_weights(i1 - i0, min(ramp, (i1 - i0) // 2))
            num[:, i0:i1] += arr * w[None, :]
            den[i0:i1] += w
        # A frame no window covered keeps its telemetry (windows tile the whole
        # recording, so this is a guard and not a path).
        r_ref = np.where(den > 0.0, num / np.maximum(den, 1e-12), r_tel)

        npz = label_dir / f"{rid}.npz"
        offset = float(rec["t0_offset_s"])
        np.savez(
            npz,
            allow_pickle=False,
            ft=ft + offset,
            t0_offset_s=np.float64(offset),
            r_telemetry=r_tel,
            r_refined=r_ref,
            window_used=np.asarray([bool(r["used"]) for r in got]),
            window_starts=np.asarray([int(r["i0"]) for r in got], dtype=np.int64),
            k_max=np.int64(params["k_max"]),
            channels=np.int64(params["channels"]),
            window_s=np.float64(params["window_s"]),
            hop_s=np.float64(params["hop_s"]),
            hop_frame_s=np.float64(HOP_S),
            sample_rate=np.int64(SR),
        )
        cruise = [
            r for r in got if r.get("used") and float(r.get("mean_rev_s", 0.0)) > CRUISE_REV_S
        ]
        report = {
            "recording_id": rid,
            "source": {"spec": spec, "rps_key": RPS_KEY, "splits": SPLITS, "sample_rate": SR},
            "time_reference": (
                "seconds from the published recording's audio t_start (the full frame, "
                "before the loader's telemetry-overlap trim of t0_offset_s)"
            ),
            "t0_offset_s": round(offset, 6),
            "params": {
                **params,
                "hop_frame_s": HOP_S,
                "idle_rev_s": IDLE_REV_S,
                "max_move_rev_s": MAX_MOVE_REV_S,
            },
            "n_frames": int(ft.size),
            "n_windows": len(got),
            "n_used": sum(1 for r in got if r["used"]),
            "cruise_scale_pct": (
                round(float(np.mean([np.mean(r["scale_pct_per_rotor"]) for r in cruise])), 5)
                if cruise
                else None
            ),
            "max_abs_delta": round(float(np.abs(r_ref - r_tel).max()), 6),
            "windows": [{k: v for k, v in r.items() if k != "r_window"} for r in got],
        }
        (label_dir / f"{rid}.report.json").write_text(json.dumps(report, indent=1))
        written.append(npz)
        print(
            f"[stitch] {rid}: {report['n_used']}/{report['n_windows']} windows used, "
            f"cruise scale {report['cruise_scale_pct']} %, max |delta| "
            f"{report['max_abs_delta']} rev/s -> {npz}",
            flush=True,
        )
    return written


# ---------------------------------------------------------------------------
# CLI


def parse_schedule(text: str) -> list[list[int]] | None:
    """``"5:3,10:5"`` -> ``[[5, 3], [10, 5]]`` (``""`` = the k_max ladder)."""
    if not text.strip():
        return None
    out = []
    for part in text.split(","):
        k, _, n = part.partition(":")
        out.append([int(k), int(n or 20)])
    return out


def build_units(recs: list[dict[str, Any]], args: Any, common: dict[str, Any]) -> list[Unit]:
    """One unit per (recording, window), smoke-restricted when asked."""
    units: list[Unit] = []
    for rec in recs:
        rid = rec["recording_id"]
        bounds = window_bounds(int(rec["ft"].size), args.window_s, args.hop_s)
        if args.smoke:
            cruise = [
                (i0, i1) for i0, i1 in bounds if float(rec["r_tel"][:, i0:i1].mean()) > CRUISE_REV_S
            ]
            if not cruise:
                raise SystemExit(f"--smoke: {rid} has no window above {CRUISE_REV_S} rev/s")
            # The MIDDLE cruise candidate: DREGON's first seconds are the
            # takeoff ramp, which is above 45 rev/s but is not cruise.
            bounds = [cruise[len(cruise) // 2]]
        units += [
            Unit(f"{rid}__f{i0:06d}", {"recording": rid, "i0": i0, "i1": i1, **common})
            for i0, i1 in bounds
        ]
        if args.smoke:
            break
    return units


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--mode", default="all", choices=("refine", "stitch", "all"))
    ap.add_argument("--window-s", type=float, default=16.0)
    ap.add_argument("--hop-s", type=float, default=12.0)
    ap.add_argument("--k-max", type=int, default=40)
    ap.add_argument(
        "--channels",
        type=int,
        default=4,
        help="microphones the refiner sees (k_max 40 x 8 mics is ~12.6 GB of autograd)",
    )
    ap.add_argument("--knot-s", type=float, default=0.25)
    ap.add_argument("--lr", type=float, default=1.0)
    ap.add_argument("--schedule", default="", help="'k:iters,...' (default: the k_max ladder)")
    ap.add_argument("--spec", default=FRAMES_SPEC)
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--label-dir", default=LABEL_DIR_DEFAULT)
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="one 4 s cruise window of the first recording at k_max 10, 1 channel",
    )
    add_gridrun_args(ap, jobs=4)
    args = ap.parse_args()

    if args.smoke:
        args.window_s, args.hop_s = 4.0, 4.0
        args.k_max, args.channels = 10, 1
        args.schedule = args.schedule or "5:20"
    out = Path(args.out)
    label_dir = Path(args.label_dir)
    if not label_dir.is_absolute():
        label_dir = ROOT / label_dir
    params = {
        "window_s": float(args.window_s),
        "hop_s": float(args.hop_s),
        "k_max": int(args.k_max),
        "channels": int(args.channels),
        "knot_s": float(args.knot_s),
        "lr": float(args.lr),
        "schedule": parse_schedule(args.schedule),
    }

    if args.mode in ("refine", "all"):
        recs = load_recordings(args.spec)
        units = build_units(recs, args, {"spec": args.spec, **params})
        print(f"[refine_dregon_rps] {len(units)} units -> {out}", flush=True)
        res = gridrun_from_args(args, units, refine_worker, out, summarize=summarize)
        if res.n_failed:
            raise SystemExit(res.exit_code)

    if args.mode in ("stitch", "all"):
        stitch(out, label_dir, args.spec, params)


if __name__ == "__main__":
    main()
