"""Train the comb-salience head on the controlled static-comb benchmark.

The head starts as the exact classical scan (all learned terms initialized to
zero effect), so any movement is attributable to training rather than to a
different starting point. The loss is BCE on the salience map, which is
permutation-free by construction: the target says "a rotor turns at this rate in
this frame" and never has to decide WHICH rotor, so there is no assignment to
average over and none of the mean-seeking pressure that produces the fan.

    PYTHONPATH=src python scripts/train_comb_salience.py --steps 400
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from itertools import permutations
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

if str(Path(__file__).resolve().parents[1] / "src") not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data_processing.comb_bench import REGIMES, comb_clip  # noqa: E402
from models.comb_salience import CombSalienceNet, decode_peel  # noqa: E402

SR = 16000


def salience_target(rps: np.ndarray, grid: np.ndarray, sigma: float) -> np.ndarray:
    """``(n_rot, T)`` rates -> ``(G, T)`` soft target: a bump at every rotor."""
    d = (grid[None, :, None] - rps[:, None, :]) / sigma
    return np.exp(-0.5 * d * d).max(axis=0)


def batch(rng, n, grid, sigma, spread_lo, spread_hi, centre_lo, centre_hi):
    ys, ts = [], []
    for _ in range(n):
        sd = int(rng.integers(1 << 30))
        y, r, _ = comb_clip(
            sd, centre=float(rng.uniform(centre_lo, centre_hi)),
            spread=float(rng.uniform(spread_lo, spread_hi)),
            excursion=float(rng.uniform(0.5, 4.0)),
        )
        ys.append(y)
        ts.append(salience_target(r, grid, sigma))
    return (torch.tensor(np.stack(ys), dtype=torch.float32),
            torch.tensor(np.stack(ts), dtype=torch.float32))


def pit(p, t):
    return min(float(np.sqrt(np.mean((p[list(q)] - t) ** 2)))
               for q in permutations(range(t.shape[0])))


def evaluate(model, grid_t, n=6, octave_mode="scored"):
    """Score every benchmark cell with the peel decoder.

    `decode_peel`, not a per-frame top-R pick: the peel is what supplies model
    order, and every threshold-based decoder was measured to trade coincident
    rotors against separated ones with no setting that serves both.
    """
    model.eval()
    out = {}
    with torch.no_grad():
        for name, ctr, spr, exc in REGIMES:
            e = []
            for s in range(n):
                y, T, _ = comb_clip(7000 + 137 * s, centre=ctr, spread=spr, excursion=exc)
                p = decode_peel(model, torch.tensor(y, dtype=torch.float32)[None], 4,
                                octave_mode=octave_mode)[0].numpy()
                e.append(pit(p[:, : T.shape[1]], T))
            out[name] = float(np.mean(e))
    model.train()
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--sigma", type=float, default=0.4, help="target bump width, rev/s")
    ap.add_argument("--n-grid", type=int, default=700)
    ap.add_argument("--k-max", type=int, default=32)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--out", default="results/comb_salience/run.json")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    model = CombSalienceNet(n_grid=args.n_grid, k_max=args.k_max, head_mode="learned")
    grid_t = model.grid.clone()
    grid = grid_t.numpy()
    model = model.to(args.device)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    rng = np.random.default_rng(12345)

    base = evaluate(model.cpu(), grid_t)
    model = model.to(args.device)
    print("corner case (untrained, learned head == classical):")
    print("   " + "  ".join(f"{k} {v:.3f}" for k, v in base.items()), flush=True)

    hist = []
    t0 = time.time()
    for step in range(1, args.steps + 1):
        y, tgt = batch(rng, args.batch, grid, args.sigma, 0.0, 22.0, 38.0, 92.0)
        sal = model(y.to(args.device))
        loss = F.binary_cross_entropy_with_logits(sal - 3.0, tgt.to(args.device))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % args.eval_every == 0 or step == args.steps:
            ev = evaluate(model.cpu(), grid_t)
            model = model.to(args.device)
            hist.append({"step": step, "loss": float(loss), **ev})
            print(f"step {step:5d}  loss {float(loss):.4f}  " +
                  "  ".join(f"{k} {v:.3f}" for k, v in ev.items())
                  + f"   [{time.time()-t0:.0f}s]", flush=True)
    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"baseline": base, "history": hist,
                             "args": vars(args)}, indent=1))
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
