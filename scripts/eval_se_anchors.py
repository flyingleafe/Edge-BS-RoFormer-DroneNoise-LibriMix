#!/usr/bin/env python3
"""Compute the noisy-input and Wiener anchors for the F1 SE valid sets.

Every F1 floor table carries two trivial anchors (``docs/se-baselines-plan.md``
§ "Metrics & reporting"): the **noisy input** (mixture vs clean target — the
"do nothing" floor) and a **Wiener** filter (a weak classical denoiser every
trained model must beat at ≥ −10 dB). This scores both on a published SE valid
set (``SE-valid-drone`` / ``SE-valid-harmonic``) per SNR (and per category for
the harmonic set), with the same SI-SDR / SDR / PESQ / eSTOI metrics ``eval.py``
uses, and writes a tidy CSV under ``results/se_anchors/``.

    python scripts/eval_se_anchors.py --dataset SE-valid-drone
    python scripts/eval_se_anchors.py --dataset SE-valid-harmonic --by-category
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.signal import wiener

from data_processing.frame_datasets import SEValidFrameDataset
from data_processing.frames import get_meta
from metrics.separation import pesq, sdr, si_sdr, stoi

SR = 16000


def _metrics(ref: np.ndarray, est: np.ndarray) -> dict[str, float]:
    ref = np.asarray(ref, dtype=np.float32).reshape(-1)
    est = np.asarray(est, dtype=np.float32).reshape(-1)
    out = {
        "si_sdr": float(si_sdr(ref[None, :], est[None, :])),  # (channel, samples)
        "sdr": float(np.asarray(sdr(ref[None, None, :], est[None, None, :])).reshape(-1)[0]),
    }
    try:
        out["pesq"] = float(pesq(ref, est, SR))
    except Exception:
        out["pesq"] = float("nan")
    try:
        out["estoi"] = float(stoi(ref, est, SR, extended=True))
    except Exception:
        out["estoi"] = float("nan")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Noisy + Wiener anchors for an SE valid set.")
    ap.add_argument("--dataset", required=True, help="SE-valid-drone | SE-valid-harmonic")
    ap.add_argument("--by-category", action="store_true", help="also group rows by meta.category")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ds = SEValidFrameDataset(args.dataset, sample_rate=SR)
    # group_key -> method -> metric -> list
    acc: dict[tuple, dict[str, dict[str, list[float]]]] = defaultdict(
        lambda: {"noisy": defaultdict(list), "wiener": defaultdict(list)}
    )
    for i in range(len(ds)):
        fr = ds[i]
        mixture = np.asarray(fr["mixture"].data, dtype=np.float32).reshape(-1)
        target = np.asarray(fr["target"].data, dtype=np.float32).reshape(-1)
        snr = float(get_meta(fr, "input_snr"))
        cat = str(get_meta(fr, "category", "all")) if args.by_category else "all"
        key = (cat, snr)
        wien = wiener(mixture).astype(np.float32)
        wien = np.nan_to_num(wien, nan=0.0, posinf=0.0, neginf=0.0)
        for method, est in (("noisy", mixture), ("wiener", wien)):
            for m, v in _metrics(target, est).items():
                acc[key][method][m].append(v)

    metrics = ["si_sdr", "sdr", "pesq", "estoi"]
    out_path = Path(args.out or f"results/se_anchors/{args.dataset}.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["category", "input_snr", "method", "n", *metrics])
        for cat, snr in sorted(acc):
            for method in ("noisy", "wiener"):
                vals = acc[(cat, snr)][method]
                n = len(vals["si_sdr"])
                row = [cat, snr, method, n]
                row += [f"{float(np.nanmean(vals[m])):.4f}" if vals[m] else "nan" for m in metrics]
                w.writerow(row)
    print(f"wrote {out_path} ({len(acc)} groups)")


if __name__ == "__main__":
    main()
