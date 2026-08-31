#!/usr/bin/env python
"""Run the comb-slot model on the real beat-VK validation recordings.

The bridge between a model built and measured entirely on synthetic combs and
the frozen protocol the campaign scores on. It emits the `npz:` prediction
layout `scripts/rps_eval.py` already accepts, so the metric, the windows and the
pools are the campaign's own and nothing here re-derives them:

    python scripts/comb_slots_real.py --out results/slots_real
    python scripts/rps_eval.py --protocol beatvk --pred npz:results/slots_real

The recordings are 8-channel at 44.1 kHz; the model assumes one channel at
16 kHz, because its rate grid, `f_max` and bin geometry were calibrated there.
Resampling is therefore part of the bridge, not a preprocessing choice hidden
somewhere else.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "src"))
sys.path.insert(0, str(_HERE))


def to_mono16k(audio: np.ndarray, sr_in: float, channel: str) -> np.ndarray:
    """``(C, T)`` at ``sr_in`` -> ``(T,)`` at 16 kHz."""
    import scipy.signal as sps

    a = np.asarray(audio, dtype=np.float64)
    if a.ndim == 1:
        a = a[None]
    if channel == "mean":
        x = a.mean(axis=0)
    elif channel == "best":
        # The loudest channel: on a quadrotor the mics differ by many dB and the
        # quietest ones carry mostly wash noise.
        x = a[int(np.argmax((a ** 2).mean(axis=1)))]
    else:
        x = a[int(channel)]
    g = np.gcd(int(round(sr_in)), 16000)
    return sps.resample_poly(x, 16000 // g, int(round(sr_in)) // g)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/slots_real")
    ap.add_argument("--channel", default="0", help="'0'..'7', 'mean' or 'best'")
    ap.add_argument("--iters", type=int, default=0)
    ap.add_argument("--read-width", type=int, default=0)
    ap.add_argument("--r-lo", type=float, default=30.0)
    ap.add_argument("--r-hi", type=float, default=100.0)
    ap.add_argument("--n-grid", type=int, default=700)
    ap.add_argument("--k-max", type=int, default=32)
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--head", default="classical")
    ap.add_argument("--recordings", default="", help="comma list; default all")
    ap.add_argument("--regimes", default="", help="comma list of window regimes to run")
    ap.add_argument("--threads", type=int, default=6)
    args = ap.parse_args()

    import torch
    torch.set_num_threads(args.threads)
    import beatvk_eval as bve
    from models.comb_slots import SlotCombNet

    wanted = set(args.recordings.split(",")) if args.recordings else None
    regimes = set(args.regimes.split(",")) if args.regimes else None
    recs = bve.load_recordings(None, wanted, keep_audio=True)

    net = SlotCombNet(head_mode=args.head, n_iter=args.iters, r_lo=args.r_lo,
                      r_hi=args.r_hi, n_grid=args.n_grid, k_max=args.k_max,
                      read_width=args.read_width, use_checkpoint=False).eval()
    if args.ckpt:
        net.head.load_state_dict(torch.load(args.ckpt, map_location="cpu"))

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    hop_s = net.hop_length / net.sr
    for rec in recs:
        rid = rec["recording_id"]
        sr_in = float(rec["audio"].tindex.rate)
        raw = np.asarray(rec["audio"].data)
        fts, rpss = [], []
        for w in rec["windows"]:
            if regimes and w["regime"] not in regimes:
                continue
            lo, hi = float(w["start_s"]), float(w["end_s"])
            seg = raw[:, int(round(lo * sr_in)): int(round(hi * sr_in))]
            x = to_mono16k(seg, sr_in, args.channel)
            x = x / (np.sqrt((x ** 2).mean()) + 1e-12)
            with torch.no_grad():
                pred = net.decode(torch.tensor(x, dtype=torch.float32)[None])[0].numpy()
            n = pred.shape[-1]
            fts.append(lo + np.arange(n) * hop_s)
            rpss.append(pred)
            print(f"  {rid} w{w['index']} ({w['regime']}) -> "
                  f"{np.round(pred.mean(-1), 1)}", flush=True)
        if not fts:
            continue
        ft = np.concatenate(fts)
        rps = np.concatenate(rpss, axis=1)
        order = np.argsort(ft)
        np.savez(outdir / f"{rid}.npz", ft=ft[order], rps=rps[:, order])
        print(f"[saved] {rid}: {len(ft)} frames", flush=True)
    print(f"\nscore with:\n  python scripts/rps_eval.py --protocol beatvk --pred npz:{outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
