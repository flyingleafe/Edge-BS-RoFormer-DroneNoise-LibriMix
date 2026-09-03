"""Dump per-frame RPS predictions of many checkpoints on named validation sets.

One aggregate number per (model, set) is what training monitors, and it hides
the SHAPE of the error. Measured on `salv2_hppnet_comb_nomix`: a mean of 0.46
rev/s is 168 near-exact clips and 88 failures that carry 76% of the error. A
mean cannot separate "imprecise everywhere" from "exact, with rare failures",
and the two call for different fixes. This CLI keeps every prediction, so the
distribution -- percentiles, per-flight failures, per-rotor structure, aliases,
stopped rotors -- is read off afterwards on any machine, without a GPU and
without the models.

    python scripts/rps_dump.py \\
        --sets comb=conf/data/salv2_comb_nomix.yaml,real=conf/data/m3cur_s2.yaml \\
        --experiments salv2_hppnet_comb_nomix,m3mixv2_scv2 --device cuda

A set is a name from `experiments.rps_bench.PARTS` (`comb`, `stoch`, `real`,
`comb_speech`, `stoch_speech`) or the `valid` block of a Hydra data yaml,
built exactly as training builds it, with optional `key=value` overrides after
colons -- a smoke is `comb=conf/data/salv2_comb_nomix.yaml:n=8`. The set
builder and the readout live in `experiments.rps_bench`, which the notebook
browser shares.

Layout under --out:

    <set>/_gt.npz     rps (N, R, T) rev/s on the label grid; n_t (N,)
    <set>/_meta.json  one dict per frame (sample/recording id, channel, ...)
    <set>/<exp>.npz   pred (N, R, T_pred) rev/s; n_t (N,); metric (N,)

`metric` is the MONITORED per-frame PIT MAE -- `RPSMetric("mae_frame")` for a
regressor, `LayerPeakRPSMetric` for a salience port -- so its mean over a
model's own validation set reproduces the W&B number, which is the check that
the dump is the model the run selected. A salience port's layers are read by
the same peak + parabola readout that metric uses, not by the CRF, and the raw
layers are not kept (4 layers x 300 bins x 250 frames x hundreds of clips x
dozens of models).

Arrays are NaN-padded to the longest frame of the set; `n_t` holds the true
length. Existing `<exp>.npz` files are skipped, so a killed job resumes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import zoo  # noqa: E402
from data_processing.frames import meta_dict  # noqa: E402
from experiments.rps_bench import Readout, build_set, parse_sets  # noqa: E402
from metrics._common import get_array  # noqa: E402


def pad_stack(arrs: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """``[(R, T_i)]`` -> ``(N, R, max T)`` NaN-padded, plus ``(N,)`` lengths."""
    n_t = np.array([a.shape[-1] for a in arrs], dtype=np.int64)
    out = np.full((len(arrs), arrs[0].shape[0], int(n_t.max())), np.nan, dtype=np.float32)
    for i, a in enumerate(arrs):
        out[i, :, : a.shape[-1]] = a
    return out, n_t


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--sets", required=True, help="comma-separated name=data_yaml[:key=value...] items"
    )
    ap.add_argument("--experiments", required=True, help="comma-separated experiment names")
    ap.add_argument("--ckpt", default="best")
    ap.add_argument("--out", default="results/rps_dump")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--limit", type=int, default=0, help="frames per set (0 = all)")
    ap.add_argument(
        "--grid",
        default="0,150,300",
        help="salience rate grid fmin,fmax,bins (the salv2 / r4_l4 value)",
    )
    ap.add_argument("--n-layers", type=int, default=4)
    a = ap.parse_args()

    fmin, fmax, bins = (float(x) for x in a.grid.split(","))
    reader = Readout(fmin, fmax, int(bins), a.n_layers, rate=(16000, 512))
    out = Path(a.out)

    # Every set's frames are held in memory once (a few hundred 8 s mono clips)
    # and every model is loaded once, so the (model x set) loop reads nothing
    # twice and a synthetic set is synthesized once.
    sets: list[tuple[str, list[Any]]] = []
    for name, path, overrides in parse_sets(a.sets):
        d = out / name
        d.mkdir(parents=True, exist_ok=True)
        ds = build_set(path, overrides)
        n = len(ds) if not a.limit else min(len(ds), a.limit)
        frames = [ds[i] for i in range(n)]
        sets.append((name, frames))
        if not (d / "_gt.npz").exists():
            gt, n_t = pad_stack([np.asarray(get_array(f, "rps"), dtype=np.float32) for f in frames])
            np.savez(d / "_gt.npz", rps=gt, n_t=n_t)
            (d / "_meta.json").write_text(json.dumps([meta_dict(f) for f in frames], default=str))
        print(f"set {name:14s} {n:4d} frames  <- {path} {overrides or ''}", flush=True)

    for exp in (e.strip() for e in a.experiments.split(",") if e.strip()):
        todo = [(name, frames) for name, frames in sets if not (out / name / f"{exp}.npz").exists()]
        if not todo:
            continue
        fm = zoo.load(exp, ckpt=a.ckpt, device=a.device)
        for name, frames in todo:
            preds, metrics = [], []
            for f in frames:
                p, m = reader(fm(f), f)
                preds.append(p)
                metrics.append(m)
            pred, n_t = pad_stack(preds)
            metric = np.asarray(metrics, dtype=np.float64)
            np.savez(out / name / f"{exp}.npz", pred=pred, n_t=n_t, metric=metric)
            print(
                f"  {exp:32s} {name:14s} mae={metric.mean():8.4f} "
                f"median={np.median(metric):8.4f} p90={np.percentile(metric, 90):8.4f}",
                flush=True,
            )
        del fm
        if a.device.startswith("cuda"):
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
