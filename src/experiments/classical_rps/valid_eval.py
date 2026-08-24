"""Protocol evaluation of the classical RPS baselines on the frozen valid split.

Scores each of the five estimators of :mod:`experiments.classical_rps.predictors`
over every (clip, channel) of ``DREGON-LM-V4-michaels-valid-full``, with the
per-frame Hungarian (PIT) convention of ``results/m3cur_regime_probe/regime_probe.py``
— the same protocol the modern checkpoints and the OT multi-pitch baseline use,
so the numbers are directly comparable.

Frames are grouped by the TARGET speeds:

* zero: the maximum over the rotors is less than 1 rev/s (stopped rotors)
* flight: the mean over the rotors is 45 rev/s or more (mid-flight)
* low: all other frames (warm-up plus the take-off and landing ramps)

One gridrun unit is one (method, clip, channel) triple. Each unit JSON carries
per-regime sums of ``|err|`` and ``err**2`` plus the element counts, so the
aggregation in ``summary.json`` is exact and does not depend on the unit sizes.

Run::

    PYTHONPATH=src python -m experiments.classical_rps.valid_eval \\
        --out results/classical_valid_eval --jobs 8
"""

from __future__ import annotations

import argparse
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from experiments.classical_rps.predictors import CLASSICAL_TRACKERS
from utils.gridrun import Unit, add_gridrun_args, gridrun_from_args

DATASET = "dload:DREGON-LM-V4-michaels-valid-full"
N_FFT = 2048
HOP_LENGTH = 512
SAMPLE_RATE = 16000
N_CHANNELS = 8
REGIMES = ("zero", "low", "flight")


def _dataset(channel: int):
    from data_processing.frame_datasets import DregonLMFrameDataset

    return DregonLMFrameDataset(
        data_dir=DATASET,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        sample_rate=SAMPLE_RATE,
        channel=channel,
    )


def _frame_groups(target: np.ndarray) -> np.ndarray:
    """Regime label per frame from the (4, F) target track."""
    mx, mn = target.max(0), target.mean(0)
    groups = np.full(target.shape[1], "low", dtype=object)
    groups[mx < 1.0] = "zero"
    groups[mn >= 45.0] = "flight"
    return groups


def _pit_err(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Per-frame Hungarian matching: (4, F), (4, F) -> |err| of shape (4, F)."""
    n_rotors, n_frames = target.shape
    out = np.empty((n_rotors, n_frames))
    for i in range(n_frames):
        cost = np.abs(pred[:, None, i] - target[None, :, i])
        ri, ci = linear_sum_assignment(cost)
        out[:, i] = np.abs(pred[ri, i] - target[ci, i])
    return out


def _score_unit(unit: Unit) -> dict[str, Any]:
    method = str(unit.params["method"])
    clip_idx = int(unit.params["clip"])
    channel = int(unit.params["channel"])
    tracker = CLASSICAL_TRACKERS[method]

    frame = _dataset(channel)[clip_idx]
    audio = np.asarray(frame["mixture"].data, dtype=np.float32).reshape(-1)
    target = np.asarray(frame["rps"].data, dtype=np.float64)

    pred = np.asarray(tracker(audio), dtype=np.float64)
    n_frames = min(pred.shape[-1], target.shape[-1])
    err = _pit_err(pred[:, :n_frames], target[:, :n_frames])
    groups = _frame_groups(target)[:n_frames]

    pools: dict[str, dict[str, float]] = {}
    for regime in REGIMES:
        sel = groups == regime
        if not sel.any():
            continue
        vals = err[:, sel].ravel()
        pools[regime] = {
            "sum_abs": float(np.abs(vals).sum()),
            "sum_sq": float((vals**2).sum()),
            "n": int(vals.size),
        }
    return {
        "method": method,
        "clip": clip_idx,
        "channel": channel,
        "n_frames": int(n_frames),
        "pools": pools,
    }


def _summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    acc: dict[str, dict[str, dict[str, float]]] = {}
    for res in results:
        per_method = acc.setdefault(str(res["method"]), {})
        for regime, stats in res["pools"].items():
            for pool in (regime, "all"):
                cur = per_method.setdefault(pool, {"sum_abs": 0.0, "sum_sq": 0.0, "n": 0.0})
                cur["sum_abs"] += float(stats["sum_abs"])
                cur["sum_sq"] += float(stats["sum_sq"])
                cur["n"] += float(stats["n"])
    return {
        method: {
            pool: {
                "mae": cur["sum_abs"] / cur["n"] if cur["n"] else float("nan"),
                "mse": cur["sum_sq"] / cur["n"] if cur["n"] else float("nan"),
                "n": int(cur["n"]),
            }
            for pool, cur in sorted(pools.items())
        }
        for method, pools in sorted(acc.items())
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="results/classical_valid_eval")
    ap.add_argument(
        "--methods",
        default="all",
        help=f"'all' or a comma list of {', '.join(sorted(CLASSICAL_TRACKERS))}",
    )
    ap.add_argument("--channels", default="all", help="'all' or a comma list of channel indices")
    ap.add_argument("--clips", type=int, default=None, help="cap the number of clips (debug)")
    add_gridrun_args(ap, jobs=8)
    args = ap.parse_args()

    methods = sorted(CLASSICAL_TRACKERS) if args.methods == "all" else args.methods.split(",")
    unknown = [m for m in methods if m not in CLASSICAL_TRACKERS]
    if unknown:
        ap.error(f"unknown method(s): {', '.join(unknown)}")
    channels = (
        list(range(N_CHANNELS))
        if args.channels == "all"
        else [int(c) for c in args.channels.split(",")]
    )
    n_clips = len(_dataset(channels[0]))
    if args.clips is not None:
        n_clips = min(n_clips, int(args.clips))

    units = [
        Unit(
            uid=f"{m}__clip{c:03d}_ch{ch}",
            params={"method": m, "clip": c, "channel": ch},
        )
        for m in methods
        for c in range(n_clips)
        for ch in channels
    ]
    print(
        f"[classical_valid_eval] {len(units)} units "
        f"({len(methods)} methods x {n_clips} clips x {len(channels)} ch)"
    )
    result = gridrun_from_args(args, units, _score_unit, args.out, summarize=_summarize)
    print(
        f"[classical_valid_eval] ok={result.n_ok} "
        f"skipped={result.n_skipped} failed={result.n_failed}"
    )
    for method, pools in result.summary.items():
        for pool, stats in pools.items():
            print(
                f"  {method:15s} {pool:7s} MAE {stats['mae']:7.2f}  "
                f"MSE {stats['mse']:9.1f}  n={stats['n']}"
            )
    return 0 if result.n_failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
