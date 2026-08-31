#!/usr/bin/env python
"""Train the comb-slot head through its own decoder, with the CRF loss.

WHY THE LOSS IS THE CRF AND NOT CROSS-ENTROPY. The previous campaign trained on
binary cross-entropy over the salience map and selected checkpoints with an
argmax decoder while deploying a Viterbi one. The loss fell monotonically to its
lowest value while decode error on the training-matched cell rose seventyfold,
and the trained head's advantage did not survive the decoder switch. Here the
training objective IS the decoder: `log Z - score(gold path)` over the same chain
`comb_crf.viterbi` maximizes, so lowering the loss cannot mean anything other
than making the deployed path more likely.

SELECTION IS ON THE DEPLOYED DECODER. Validation runs `SlotCombNet.decode` — the
full pipeline including the octave moves — and the checkpoint is chosen by the
GEOMETRIC mean of per-cell PIT-RMSE, so a collapse in any one regime is punished
rather than averaged away.
"""
from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np
import torch


def make_pool(family: str, n: int, seed0: int, kw: dict) -> list:
    """Pre-generate clips. Synthesis is numpy and starves a GPU if done in-loop."""
    if family == "static":
        from data_processing.comb_bench import comb_clip as gen
    else:
        from data_processing.comb_bench_stochastic import stoch_comb_clip as gen
    from data_processing.comb_bench import REGIMES
    out = []
    for i in range(n):
        rg = REGIMES[i % len(REGIMES)]
        a, rps, _ = gen(seed=seed0 + i, centre=rg[1], spread=rg[2], excursion=rg[3], **kw)
        out.append((a.astype(np.float32), rps.astype(np.float32)))
    return out


def pit_rmse(pred: np.ndarray, gt: np.ndarray) -> float:
    return min(float(np.sqrt(((pred[list(p)] - gt) ** 2).mean()))
               for p in itertools.permutations(range(pred.shape[0])))


@torch.no_grad()
def validate(net, pool_by_regime, device) -> dict:
    net.eval()
    out = {}
    for name, clips in pool_by_regime.items():
        errs = []
        for a, rps in clips:
            au = torch.as_tensor(a, device=device)[None]
            pred = net.decode(au)[0].cpu().numpy()
            errs.append(pit_rmse(pred, rps[:, : pred.shape[-1]]))
        out[name] = float(np.mean(errs))
    net.train()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="static", choices=("static", "stochastic"))
    ap.add_argument("--head", default="learned_cond", choices=("learned", "learned_cond"))
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--crop", type=int, default=64, help="frames per training crop")
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--iters", type=int, default=1, help="joint sweeps in the model")
    ap.add_argument("--pool", type=int, default=224)
    ap.add_argument("--val-clips", type=int, default=4)
    ap.add_argument("--val-every", type=int, default=100)
    ap.add_argument("--line-mode", default="stochastic")
    ap.add_argument("--out", default="results/train_comb_slots")
    args = ap.parse_args()

    from data_processing.comb_bench import REGIMES
    from models.comb_slots import SlotCombNet

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    gen_kw = {} if args.family == "static" else {"line_mode": args.line_mode}

    print(f"device={dev} family={args.family} head={args.head}", flush=True)
    train = make_pool(args.family, args.pool, 0, gen_kw)
    val = {}
    for name, centre, spread, exc in REGIMES:
        if args.family == "static":
            from data_processing.comb_bench import comb_clip as gen
        else:
            from data_processing.comb_bench_stochastic import stoch_comb_clip as gen
        val[name] = [gen(seed=1000 + s, centre=centre, spread=spread,
                         excursion=exc, **gen_kw)[:2] for s in range(args.val_clips)]
        val[name] = [(a.astype(np.float32), r.astype(np.float32)) for a, r in val[name]]
    print(f"pool: {len(train)} train clips, {sum(map(len, val.values()))} val clips", flush=True)

    net = SlotCombNet(head_mode=args.head, n_iter=args.iters, use_checkpoint=True).to(dev)
    params = [p for p in net.parameters() if p.requires_grad]
    print(f"trainable parameters: {sum(p.numel() for p in params)}", flush=True)
    opt = torch.optim.Adam(params, lr=args.lr)

    base = validate(net, val, dev)
    hist = [{"step": 0, "loss": None, **base}]
    best = float(np.exp(np.mean(np.log(list(base.values())))))
    print(f"step 0 (untrained corner) geomean={best:.4f} " +
          " ".join(f"{k}={v:.3f}" for k, v in base.items()), flush=True)
    torch.save(net.head.state_dict(), outdir / "best_head.pt")

    hop = net.hop_length
    rng = np.random.default_rng(0)
    t0 = time.time()
    for step in range(1, args.steps + 1):
        idx = rng.integers(0, len(train), args.batch)
        aud, tgt = [], []
        for i in idx:
            a, rps = train[i]
            f0 = int(rng.integers(0, max(1, rps.shape[1] - args.crop - 1)))
            aud.append(a[f0 * hop: (f0 + args.crop) * hop])
            tgt.append(rps[:, f0: f0 + args.crop + 1])
        n = min(x.shape[0] for x in aud)
        au = torch.as_tensor(np.stack([x[:n] for x in aud]), device=dev)
        m = min(x.shape[1] for x in tgt)
        gt = torch.as_tensor(np.stack([x[:, :m] for x in tgt]), device=dev)
        loss = net.loss(au, gt)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 5.0)
        opt.step()
        if step % 25 == 0:
            print(f"step {step:5d}  loss {loss.item():10.2f}  "
                  f"{(time.time()-t0)/step:.2f}s/step", flush=True)
        if step % args.val_every == 0:
            v = validate(net, val, dev)
            gm = float(np.exp(np.mean(np.log(list(v.values())))))
            hist.append({"step": step, "loss": float(loss.item()), **v})
            star = ""
            if gm < best:
                best = gm
                torch.save(net.head.state_dict(), outdir / "best_head.pt")
                star = "  *"
            print(f"step {step:5d} geomean={gm:.4f}{star} " +
                  " ".join(f"{k}={x:.3f}" for k, x in v.items()), flush=True)
            (outdir / "history.json").write_text(json.dumps(hist, indent=2))
    (outdir / "history.json").write_text(json.dumps(hist, indent=2))
    print(f"best geomean {best:.4f} -> {outdir/'best_head.pt'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
