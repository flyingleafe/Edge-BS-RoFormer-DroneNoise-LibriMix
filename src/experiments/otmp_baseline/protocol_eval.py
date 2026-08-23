"""Full-protocol evaluation of the OT multi-pitch baseline.

Scores :func:`estimate.adapted_drone_config` (or a named preset) over every
(clip, channel) of the frozen full-envelope validation split, with per-frame
Hungarian PIT matching against the telemetry targets — the same convention as
``results/m3cur_regime_probe/regime_probe.py``. One gridrun unit per
(clip, channel); resumable.

Run::

    python -m experiments.otmp_baseline.protocol_eval \
        --out results/otmp_protocol --jobs 8 [--channels 0]
"""

from __future__ import annotations

import argparse
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from experiments.otmp_baseline.estimate import (
    OTMPConfig,
    adapted_drone_config,
    drone_config,
    estimate_clip,
)
from utils.gridrun import Unit, add_gridrun_args, gridrun_from_args

DATASET = "dload:DREGON-LM-V4-michaels-valid-full"
PRESETS = {"adapted": adapted_drone_config, "paper": drone_config}


def _frame_targets(
    rps: np.ndarray, times_s: np.ndarray, frame_s: float, hop_s: float
) -> np.ndarray:
    """Mean target per OT frame: (4, T') from the (4, F) STFT-grid track."""
    n = rps.shape[1]
    grid = np.arange(n) * hop_s  # target frame times (STFT hop grid)
    out = np.empty((rps.shape[0], len(times_s)))
    for i, t in enumerate(times_s):
        sel = (grid >= t - frame_s / 2) & (grid < t + frame_s / 2)
        out[:, i] = rps[:, sel].mean(axis=1) if sel.any() else np.nan
    return out


def _regime(target_col: np.ndarray) -> str:
    if float(target_col.max()) < 1.0:
        return "zero"
    if float(target_col.mean()) >= 45.0:
        return "flight"
    return "low"


def _score_unit(unit: Unit) -> dict[str, Any]:
    from data_processing.frame_datasets import DregonLMFrameDataset

    clip_idx = int(unit.params["clip"])
    channel = int(unit.params["channel"])
    cfg: OTMPConfig = PRESETS[str(unit.params["preset"])]()

    ds = DregonLMFrameDataset(
        data_dir=DATASET, n_fft=2048, hop_length=512, sample_rate=16000, channel=channel
    )
    frame = ds[clip_idx]
    x = np.asarray(frame["mixture"].data, dtype=np.float64)
    rps = np.asarray(frame["rps"].data, dtype=np.float64)

    times, pitches = estimate_clip(x, cfg.sample_rate, cfg)
    frame_s = cfg.frame_len / cfg.sample_rate
    targets = _frame_targets(rps, times, frame_s, 512 / 16000)

    rows = []
    for i in range(len(times)):
        t = targets[:, i]
        if not np.isfinite(t).all():
            continue
        p = pitches[:, i]
        cost = np.abs(p[:, None] - t[None, :])
        ri, ci = linear_sum_assignment(cost)
        err = np.abs(p[ri] - t[ci])
        rows.append(
            {
                "t": float(times[i]),
                "regime": _regime(t),
                "mae": float(err.mean()),
                "mse": float((err**2).mean()),
            }
        )
    return {"clip": clip_idx, "channel": channel, "frames": rows}


def _summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    pools: dict[str, list[dict[str, float]]] = {}
    for res in results:
        for row in res["frames"]:
            pools.setdefault(row["regime"], []).append(row)
            pools.setdefault("all", []).append(row)
    return {
        pool: {
            "mae": float(np.mean([r["mae"] for r in rows])),
            "mse": float(np.mean([r["mse"] for r in rows])),
            "n_frames": len(rows),
        }
        for pool, rows in sorted(pools.items())
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="results/otmp_protocol")
    ap.add_argument("--preset", default="adapted", choices=sorted(PRESETS))
    ap.add_argument("--channels", default="all", help="'all' or a comma list of channel indices")
    ap.add_argument("--clips", type=int, default=None, help="cap the number of clips (debug)")
    add_gridrun_args(ap, jobs=8)
    args = ap.parse_args()

    from data_processing.frame_datasets import DregonLMFrameDataset

    n_clips = len(
        DregonLMFrameDataset(
            data_dir=DATASET, n_fft=2048, hop_length=512, sample_rate=16000, channel=0
        )
    )
    if args.clips is not None:
        n_clips = min(n_clips, int(args.clips))
    channels = (
        list(range(8)) if args.channels == "all" else [int(c) for c in args.channels.split(",")]
    )

    units = [
        Unit(
            uid=f"clip{c:03d}_ch{ch}",
            params={"clip": c, "channel": ch, "preset": args.preset},
        )
        for c in range(n_clips)
        for ch in channels
    ]
    print(f"[otmp_protocol] {len(units)} units ({n_clips} clips x {len(channels)} ch)")
    result = gridrun_from_args(args, units, _score_unit, args.out, summarize=_summarize)
    print(f"[otmp_protocol] ok={result.n_ok} skipped={result.n_skipped} failed={result.n_failed}")
    for pool, stats in result.summary.items():
        print(
            f"  {pool:8s} MAE {stats['mae']:7.2f}  MSE {stats['mse']:9.1f}  n={stats['n_frames']}"
        )
    return 0 if result.n_failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
