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


def build_pool(rng, n_clips, grid, sigma, spread_lo, spread_hi, centre_lo, centre_hi):
    """Generate the training clips ONCE, up front.

    Clip synthesis dominates the step time — four rotors of a hundred harmonics
    over 128k samples, on CPU — so generating fresh clips every step starves
    the GPU and made a first run miss its wall before the first evaluation. The
    head has forty parameters; a fixed pool of a few hundred clips is far more
    data than it can overfit.
    """
    ys, ts = [], []
    for _ in range(n_clips):
        y, r, _ = comb_clip(
            int(rng.integers(1 << 30)),
            centre=float(rng.uniform(centre_lo, centre_hi)),
            spread=float(rng.uniform(spread_lo, spread_hi)),
            excursion=float(rng.uniform(0.5, 4.0)),
        )
        ys.append(y.astype(np.float32))
        ts.append(salience_target(r, grid, sigma).astype(np.float32))
    return torch.tensor(np.stack(ys)), torch.tensor(np.stack(ts))


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
    ap.add_argument("--head", default="learned_cond", choices=["learned", "learned_cond"])
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--pool", type=int, default=256, help="clips generated up front")
    ap.add_argument("--out", default="results/comb_salience/run.json")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    model = CombSalienceNet(n_grid=args.n_grid, k_max=args.k_max, head_mode=args.head)
    grid_t = model.grid.clone()
    grid = grid_t.numpy()
    model = model.to(args.device)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    rng = np.random.default_rng(12345)

    base = evaluate(model.cpu(), grid_t)
    model = model.to(args.device)
    print("corner case (untrained, learned head == classical):")
    print("   " + "  ".join(f"{k} {v:.3f}" for k, v in base.items()), flush=True)

    t_pool = time.time()
    pool_y, pool_t = build_pool(rng, args.pool, grid, args.sigma, 0.0, 22.0, 38.0, 92.0)
    print(f"pool: {args.pool} clips in {time.time()-t_pool:.0f}s", flush=True)

    # Select on the DECODE metric, not on the loss and not on the last step.
    # BCE over the salience map is dominated by whichever cells have the largest
    # map error, which is not the same as decoding rotors accurately: a run was
    # measured taking the 40 rev/s cell from 28.0 to 9.5 while pushing the
    # training-matched cell from 0.160 to 2.346. The loss went DOWN throughout.
    best_score, best_state, best_step = float("inf"), None, 0
    hist = []
    t0 = time.time()
    for step in range(1, args.steps + 1):
        sel = torch.from_numpy(rng.choice(args.pool, args.batch, replace=False))
        y, tgt = pool_y[sel], pool_t[sel]
        sal = model(y.to(args.device))
        loss = F.binary_cross_entropy_with_logits(sal - 3.0, tgt.to(args.device))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % args.eval_every == 0 or step == args.steps:
            ev = evaluate(model.cpu(), grid_t)
            model = model.to(args.device)
            # Geometric mean across cells: no single cell can dominate, and a
            # collapse anywhere is punished more than an arithmetic mean allows.
            gscore = float(np.exp(np.mean(np.log(np.maximum(list(ev.values()), 1e-6)))))
            if gscore < best_score:
                best_score, best_step = gscore, step
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            hist.append({"step": step, "loss": float(loss), "gscore": gscore, **ev})
            print(f"step {step:5d}  loss {float(loss):.4f}  " +
                  "  ".join(f"{k} {v:.3f}" for k, v in ev.items())
                  + f"   [{time.time()-t0:.0f}s]", flush=True)
    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"baseline": base, "history": hist, "best_step": best_step,
                             "best_score": best_score, "args": vars(args)}, indent=1))
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"restored best head from step {best_step} (gscore {best_score:.4f})", flush=True)
    # Save the head so a trained run can be re-scored with the Viterbi decoder,
    # which the in-loop evaluation does not use (it decodes per frame, so its
    # numbers are the weaker decoder's and understate the final result).
    torch.save({k: v for k, v in model.state_dict().items() if v.requires_grad or "head" in k},
               p.with_suffix(".pt"))
    print(f"saved head to {p.with_suffix('.pt')}", flush=True)
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
