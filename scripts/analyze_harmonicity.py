#!/usr/bin/env python3
"""Measure harmonicity across the published external noise datasets.

The "analysis stage" of the external-dataset effort (see
``docs/external-datasets-plan.md``): stream each dataset's ``tdframe-v1`` frames
from R2, run :func:`data_processing.harmonicity.measure_harmonicity` on the
audio, and emit (a) a per-sample table and (b) a per-dataset summary — the
evidence for *how harmonic* each rotating-source corpus is and thus how useful
it is for ultra-low-SNR harmonic-noise suppression.

Harmonicity is CPU-heavy per clip, so this is meant for the CPU cluster; by
default it samples ``--max-per-dataset`` shuffled clips per dataset (enough for
distribution stats). Pass ``--all`` to score every sample.

    python scripts/analyze_harmonicity.py --max-per-dataset 1000 \
        --out results/harmonicity

Outputs ``<out>/per_sample.csv`` and ``<out>/summary.csv``.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np

from data_processing import streams
from data_processing.frames import get_meta
from data_processing.harmonicity import measure_harmonicity

DEFAULT_DATASETS = [
    "MIMII",
    "MIMII-DG",
    "AeroSonicDB",
    "DroneAudioSet",
    "drone-detection-samples",
    "HornBase",
    "KAIST-rotating-acoustic",
    "HUSTmotor",
]

METRIC_COLS = [
    "f0_hz",
    "harmonic_energy_ratio",
    "harmonic_to_noise_db",
    "n_prominent_harmonics",
    "spectral_flatness",
]


def _meta_group_get(frame: Any, group: str, key: str) -> Any:
    if "meta" not in frame:
        return None
    meta = frame["meta"]
    if group not in meta:
        return None
    sub = meta[group]
    if key not in sub:
        return None
    return sub[key]


def analyze_dataset(
    name: str, max_samples: int | None, seed: int, shuffle_buffer: int = 256
) -> list[dict[str, Any]]:
    """Score up to ``max_samples`` shuffled clips of one published dataset.

    ``shuffle_buffer`` is kept small on purpose: dload also shuffles *shard*
    order, so a modest item buffer still samples representatively across the
    whole dataset — while the default 4096 would buffer that many decoded frames
    before emitting one, which for 8-ch MIMII (~5 MB/frame) is ~21 GB and OOMs
    the job. 256 frames caps it at ~1.3 GB even for the multichannel sets.
    """
    ds = streams.DloadFrameDataset(
        name,
        shuffle=seed,
        shuffle_buffer=shuffle_buffer,
        take=max_samples,
        decoder=streams.decode_tdframe,
    )
    rows: list[dict[str, Any]] = []
    for frame in ds:
        if "audio" not in frame:
            continue
        audio = frame["audio"]
        data = np.asarray(audio.data, dtype=np.float32)
        sr = int(audio.tindex.sr)
        h = measure_harmonicity(data, sr)
        rows.append(
            {
                "dataset": name,
                "recording_id": get_meta(frame, "recording_id"),
                "category": _meta_group_get(frame, "system", "category"),
                "observation": _meta_group_get(frame, "observation", "type"),
                **h.as_dict(),
            }
        )
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-dataset aggregate (median/mean of the metrics over finite values)."""
    out: dict[str, Any] = {"dataset": rows[0]["dataset"], "n": len(rows)}
    out["category"] = rows[0]["category"]
    for col in METRIC_COLS:
        vals = np.array([r[col] for r in rows], dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            out[f"{col}_median"] = out[f"{col}_mean"] = float("nan")
            continue
        out[f"{col}_median"] = float(np.median(vals))
        out[f"{col}_mean"] = float(np.mean(vals))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="*", default=DEFAULT_DATASETS)
    parser.add_argument("--max-per-dataset", type=int, default=1000)
    parser.add_argument("--all", action="store_true", help="score every sample (ignore the cap)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--shuffle-buffer",
        type=int,
        default=256,
        help="dload item shuffle buffer (small on purpose — see analyze_dataset)",
    )
    parser.add_argument("--out", default="results/harmonicity")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = None if args.all else int(args.max_per_dataset)

    all_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for name in args.datasets:
        print(f"Analyzing {name} (cap={cap}) ...", flush=True)
        try:
            rows = analyze_dataset(name, cap, args.seed, shuffle_buffer=args.shuffle_buffer)
        except Exception as exc:  # noqa: BLE001 - one bad dataset shouldn't sink the run
            print(f"  {name}: FAILED ({type(exc).__name__}: {exc})", flush=True)
            continue
        if not rows:
            print(f"  {name}: no samples", flush=True)
            continue
        all_rows.extend(rows)
        s = summarize(rows)
        summaries.append(s)
        print(
            f"  {name}: n={s['n']} "
            f"harm_ratio_med={s['harmonic_energy_ratio_median']:.3f} "
            f"flatness_med={s['spectral_flatness_median']:.3f} "
            f"f0_med={s['f0_hz_median']:.1f}Hz",
            flush=True,
        )

    if all_rows:
        per_sample = out_dir / "per_sample.csv"
        with open(per_sample, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(all_rows[0].keys()))
            w.writeheader()
            w.writerows(all_rows)
        print(f"wrote {per_sample} ({len(all_rows)} rows)")
    if summaries:
        summary = out_dir / "summary.csv"
        with open(summary, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(summaries[0].keys()))
            w.writeheader()
            w.writerows(summaries)
        print(f"wrote {summary} ({len(summaries)} datasets)")


if __name__ == "__main__":
    main()
