#!/usr/bin/env python
"""Score a decoder on the static- or stochastic-comb benchmark, cell by cell.

One unit = one (regime, seed, method) clip. Methods:

  peel        the deployed `comb_salience.decode_peel_viterbi` (the baseline)
  slots       `comb_slots.SlotCombNet`, `--iters` joint sweeps after the peel init

`--family static` uses `data_processing.comb_bench` (analytic comb, sharp lines);
`--family stochastic` uses `comb_bench_stochastic` (Lorentzian lines realized as
a Gaussian process), which is a different task and is expected to score
differently — see docs/experiments/comb-slot-crf.md.
"""
from __future__ import annotations

import argparse
import itertools
import json
import numpy as np

from utils.gridrun import Unit, add_gridrun_args, gridrun_from_args


def pit_rmse(pred: np.ndarray, gt: np.ndarray) -> float:
    """Permutation-invariant RMSE over ``(R, T)`` rate tracks, in rev/s."""
    best = None
    for p in itertools.permutations(range(pred.shape[0])):
        e = float(np.sqrt(((pred[list(p)] - gt) ** 2).mean()))
        best = e if best is None else min(best, e)
    return best


def worker(unit: Unit) -> dict:
    import torch
    torch.set_num_threads(int(unit.params.get("threads", 2)))
    from models.comb_salience import CombSalienceNet, decode_peel_viterbi
    from models.comb_slots import SlotCombNet

    p = unit.params
    if p["family"] == "static":
        from data_processing.comb_bench import comb_clip
        a, rps, _ = comb_clip(seed=p["seed"], centre=p["centre"],
                              spread=p["spread"], excursion=p["excursion"])
    else:
        from data_processing.comb_bench_stochastic import stoch_comb_clip
        a, rps, _ = stoch_comb_clip(seed=p["seed"], centre=p["centre"],
                                    spread=p["spread"], excursion=p["excursion"],
                                    **p.get("stoch_kw", {}))
    au = torch.tensor(a, dtype=torch.float32)[None]

    ckpt = p.get("ckpt")
    if p["method"] == "peel":
        net = CombSalienceNet(head_mode=p.get("head", "classical"))
        if ckpt:
            net.head.load_state_dict(torch.load(ckpt, map_location="cpu"))
        with torch.no_grad():
            out = decode_peel_viterbi(net, au, octave_mode=p.get("octave", "scored"))
    else:
        net = SlotCombNet(head_mode=p.get("head", "classical"), n_iter=p["iters"],
                          union_mode=p.get("union", "noisyor"), use_checkpoint=False)
        if ckpt:
            net.head.load_state_dict(torch.load(ckpt, map_location="cpu"))
        net.eval()
        with torch.no_grad():
            out = net.decode(au, subgrid=p.get("subgrid", True))
    pred = out[0].numpy()
    gt = rps[:, : pred.shape[-1]]
    return {"regime": p["regime"], "seed": p["seed"], "method": p["tag"],
            "rmse": pit_rmse(pred, gt)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="static", choices=("static", "stochastic"))
    ap.add_argument("--methods", default="peel,slots0,slots2",
                    help="comma list; slotsN = SlotCombNet with N joint sweeps")
    ap.add_argument("--clips", type=int, default=8)
    ap.add_argument("--regimes", default="", help="comma list; default all")
    ap.add_argument("--head", default="classical")
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--line-mode", default="stochastic",
                    choices=("stochastic", "coherent"),
                    help="stochastic family only: Rayleigh realization or coherent tones")
    ap.add_argument("--gamma", default="", help="stochastic only: 'g0lo,g0hi,slo,shi' in Hz")
    ap.add_argument("--union", default="noisyor", choices=("sum", "max", "noisyor"))
    ap.add_argument("--out", default="results/comb_slots")
    add_gridrun_args(ap, jobs=6)
    args = ap.parse_args()

    from data_processing.comb_bench import REGIMES
    keep = set(args.regimes.split(",")) if args.regimes else None
    stoch_kw = {"line_mode": args.line_mode}
    if args.gamma:
        g = [float(x) for x in args.gamma.split(",")]
        stoch_kw |= {"gamma0_hz": (g[0], g[1]), "gamma_slope_hz": (g[2], g[3])}
    units = []
    for name, centre, spread, exc in REGIMES:
        if keep and name not in keep:
            continue
        for seed in range(args.clips):
            for m in args.methods.split(","):
                units.append(Unit(
                    uid=f"{args.family}__{name}__{m}__{seed}",
                    params={"family": args.family, "regime": name, "centre": centre,
                            "spread": spread, "excursion": exc, "seed": 1000 + seed,
                            "tag": m, "method": "peel" if m == "peel" else "slots",
                            "iters": 0 if m == "peel" else int(m[5:]),
                            "head": args.head, "ckpt": args.ckpt,
                            "threads": args.threads, "stoch_kw": stoch_kw,
                            "union": args.union}))
    res = gridrun_from_args(args, units, worker, args.out, mp_context="spawn")
    rows = [json.loads(p.read_text()) for p in sorted((res.out_dir / "raw").glob("*.json"))]
    tags = [m for m in args.methods.split(",")]
    names = [r[0] for r in REGIMES if not keep or r[0] in keep]
    print(f"\n{'regime':<14}" + "".join(f"{t:>12}" for t in tags))
    for n in names:
        cells = []
        for t in tags:
            v = [r["rmse"] for r in rows if r["regime"] == n and r["method"] == t]
            cells.append(f"{np.mean(v):12.3f}" if v else f"{'-':>12}")
        print(f"{n:<14}" + "".join(cells))
    print(f"{'GEOMEAN':<14}" + "".join(
        f"{np.exp(np.mean(np.log([np.mean([r['rmse'] for r in rows if r['regime']==n and r['method']==t]) for n in names]))):12.3f}"
        for t in tags))
    return res.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
