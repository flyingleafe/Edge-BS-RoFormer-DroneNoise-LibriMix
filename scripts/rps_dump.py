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

A set is the `valid` block of a Hydra data yaml, built exactly as training
builds it (`training.config.build_dataset`), with optional `key=value`
overrides after colons -- so the with-speech twin of a synthetic set is
`comb_speech=conf/data/salv2_comb_nomix.yaml:speech=true` and a smoke is
`comb=conf/data/salv2_comb_nomix.yaml:n=8`.

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
from typing import Any, cast

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import zoo  # noqa: E402
from data_processing.frames import meta_dict  # noqa: E402
from metrics import RPSMetric  # noqa: E402
from metrics._common import get_array  # noqa: E402
from metrics.salience_layers import LayerPeakRPSMetric, peak_readout  # noqa: E402
from models.multif0.utils import linear_freq_grid  # noqa: E402
from training.config import build_dataset  # noqa: E402


def parse_sets(spec: str) -> list[tuple[str, str, dict[str, Any]]]:
    """``name=path[:key=value...]`` items, comma separated -> (name, path, overrides)."""
    out = []
    for item in filter(None, spec.split(",")):
        name, _, rest = item.partition("=")
        path, *overrides = rest.split(":")
        kv: dict[str, Any] = {}
        for ov in overrides:
            k, _, v = ov.partition("=")
            kv[k] = yaml.safe_load(v)  # `true` -> True, `8` -> 8, text stays text
        out.append((name.strip(), path.strip(), kv))
    if not out:
        raise SystemExit("--sets is empty")
    return out


def build_set(path: str, overrides: dict[str, Any]) -> Any:
    cfg = OmegaConf.load(path)
    spec: dict[str, Any] = dict(cast(dict, OmegaConf.to_container(cfg.valid, resolve=True)))
    params: dict[str, Any] = dict(spec.get("params") or {})
    params.update(overrides)
    spec["params"] = params
    return build_dataset(spec)


def pad_stack(arrs: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """``[(R, T_i)]`` -> ``(N, R, max T)`` NaN-padded, plus ``(N,)`` lengths."""
    n_t = np.array([a.shape[-1] for a in arrs], dtype=np.int64)
    out = np.full((len(arrs), arrs[0].shape[0], int(n_t.max())), np.nan, dtype=np.float32)
    for i, a in enumerate(arrs):
        out[i, :, : a.shape[-1]] = a
    return out, n_t


class Reader:
    """Turn a model's output Frame into ``(R, T)`` rev/s plus the monitored metric."""

    def __init__(self, fmin: float, fmax: float, bins: int, n_layers: int, rate: tuple[int, int]):
        self.grid = np.asarray(linear_freq_grid(fmin, fmax, bins), dtype=np.float64)
        self.n_layers = n_layers
        self.reg_metric = RPSMetric("mae_frame", rate=rate)
        self.sal_metric = LayerPeakRPSMetric(
            out_fmin=fmin, out_fmax=fmax, out_bins=bins, n_layers=n_layers, rate=rate
        )

    def __call__(self, pred: Any, frame: Any) -> tuple[np.ndarray, float]:
        if "rps_pred" in pred:
            arr = np.asarray(get_array(pred, "rps_pred"), dtype=np.float32)
            return arr, float(self.reg_metric(pred, frame))
        if "salience" in pred:
            metric = float(self.sal_metric(pred, frame))
            logits = torch.as_tensor(get_array(pred, "salience")).unsqueeze(0)  # (1, R*G, T)
            _, fg, n_t = logits.shape
            if fg != self.n_layers * len(self.grid):
                raise ValueError(
                    f"model emits {fg} bins; expected "
                    f"{self.n_layers} layers x {len(self.grid)} bins"
                )
            layers = logits.reshape(1, self.n_layers, len(self.grid), n_t).double()
            speeds = peak_readout(F.logsigmoid(layers), self.grid)[0].numpy()
            return speeds.astype(np.float32), metric
        raise KeyError(f"no rps_pred or salience in {list(pred.keys())}")


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
    reader = Reader(fmin, fmax, int(bins), a.n_layers, rate=(16000, 512))
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
