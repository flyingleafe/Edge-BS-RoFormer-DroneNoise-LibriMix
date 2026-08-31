#!/usr/bin/env python
"""Fast knob sweep on the real beat-VK windows, scored the protocol's way.

`comb_slots_real.py` + `rps_eval.py` is the authoritative path and stays the
one that produces reported numbers. This is the inner loop: it caches the
decoded 16 kHz windows once, then scores any number of configurations against
the raw telemetry with the protocol's own metric (one Hungarian assignment per
window, mean absolute error in rev/s), so a knob can be swept in seconds
instead of minutes.
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "src"))
sys.path.insert(0, str(_HERE))


def window_mae(pred: np.ndarray, gt: np.ndarray) -> float:
    """PIT mean absolute error, one assignment for the whole window."""
    n = min(pred.shape[-1], gt.shape[-1])
    p, g = pred[:, :n], gt[:, :n]
    return min(float(np.abs(p[list(q)] - g).mean())
               for q in itertools.permutations(range(p.shape[0])))


def load_windows(regimes: set[str] | None, recordings: set[str] | None):
    """Cached (rid, index, regime, 16 kHz mono audio, GT on the hop grid)."""
    import beatvk_eval as bve
    out = []
    for r in bve.load_recordings(None, recordings, keep_audio=True):
        sr = float(r["audio"].tindex.rate)
        raw = np.asarray(r["audio"].data)
        for w in r["windows"]:
            if regimes and w["regime"] not in regimes:
                continue
            lo, hi = float(w["start_s"]), float(w["end_s"])
            m = (r["ts"] >= lo) & (r["ts"] < hi)
            if not m.any():
                continue
            import scipy.signal as sps
            g = np.gcd(int(round(sr)), 16000)
            x = sps.resample_poly(raw[:, int(lo * sr): int(hi * sr)].astype(np.float64),
                                  16000 // g, int(round(sr)) // g, axis=-1)
            out.append({"rid": r["recording_id"], "idx": w["index"],
                        "regime": w["regime"], "audio": x,
                        "gt_ts": r["ts"][m] - lo, "gt": r["vals"][:, m]})
        del r["audio"]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", required=True,
                    help="'name=v1,v2,...'; name in k_max,read_width,channel,r_hi,n_fft,iters")
    ap.add_argument("--regimes", default="cruise")
    ap.add_argument("--recordings", default="")
    ap.add_argument("--drop-ramps", action="store_true",
                    help="drop the takeoff windows (GT touches 0 rev/s)")
    ap.add_argument("--threads", type=int, default=6)
    args = ap.parse_args()

    import torch
    torch.set_num_threads(args.threads)
    from models.comb_slots import SlotCombNet

    regimes = set(args.regimes.split(",")) if args.regimes else None
    recordings = set(args.recordings.split(",")) if args.recordings else None
    wins = load_windows(regimes, recordings)
    if args.drop_ramps:
        wins = [w for w in wins if w["gt"].min() > 5.0]
    print(f"{len(wins)} windows", flush=True)

    name, _, vals = args.sweep.partition("=")
    cast = {"channel": str, "notch_width": float}.get(name, lambda s: int(float(s)))
    print(f"{name:>12}  " + "  ".join(f"{w['rid'][:9]}w{w['idx']}" for w in wins) + "     MEAN")
    for v in vals.split(","):
        kw = {name: cast(v)}
        chan = kw.pop("channel", "0")
        net = SlotCombNet(n_iter=kw.pop("iters", 0), k_max=kw.pop("k_max", 200),
                          read_width=kw.pop("read_width", 0), r_hi=kw.pop("r_hi", 100.0),
                          n_fft=kw.pop("n_fft", 4096),
                          k_refine=kw.pop("k_refine", 0),
                          notch_width=kw.pop("notch_width", 1.5), use_checkpoint=False).eval()
        hop_s = net.hop_length / net.sr
        errs = []
        for w in wins:
            a = w["audio"]
            if chan.startswith("pow"):
                # AVERAGE THE POWER SPECTRA, NOT THE WAVEFORMS. Summing waveforms
                # from mics metres apart comb-filters the very lines we are
                # reading. Averaging |STFT|^2 across C channels is incoherent
                # averaging: the mean is unchanged and the per-bin variance falls
                # by C, which is the exact quantity the score margin is being
                # destroyed by on real audio (0.03 nats against 1.65 synthetic).
                nch = a.shape[0] if chan == "pow" else int(chan[3:])
                x = a[:nch]
            elif chan == "mean":
                x = a.mean(axis=0)
            elif chan == "best":
                x = a[int(np.argmax((a ** 2).mean(axis=1)))]
            else:
                x = a[int(chan)]
            x = x / (np.sqrt((x ** 2).mean()) + 1e-12)
            with torch.no_grad():
                xt = torch.tensor(np.atleast_2d(x), dtype=torch.float32)
                p = net.decode(xt if xt.shape[0] > 1 else xt[None])[0].numpy()
            t = np.arange(p.shape[-1]) * hop_s
            gt = np.stack([np.interp(t, w["gt_ts"], g) for g in w["gt"]])
            errs.append(window_mae(p, gt))
        print(f"{v:>12}  " + "  ".join(f"{e:10.3f}" for e in errs) + f"  {np.mean(errs):7.3f}",
              flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
