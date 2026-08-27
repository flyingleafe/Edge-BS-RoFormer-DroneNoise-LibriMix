"""Does rotor SEPARATION explain the stochastic plateau?

A model-free harmonic-sum probe reads a rotor speed out of the stochastic
family to 0.16 to 1.04 rev/s, so the comb is present and lockable and the
plateau at 3.70 PIT-MAE is not explained by missing structure. The next
candidate is the part the probe does not attempt: telling FOUR interleaved
combs apart and assigning each to a rotor. Two rotors Delta rev/s apart put
their k-th harmonics k*Delta Hz apart, so a frame whose rotors sit close
together is the hard case, and one whose rotors have spread out is the easy one.

This buckets a checkpoint's per-frame error by the frame's own rotor spread,
on cruise columns only so the speed regime cannot confound it. A steep rise
towards the narrow-spread buckets supports separation as the bottleneck; a flat
profile rules it out.

    python scripts/spread_eval.py --exp stoch_s1id_scv2 --ckpt last --n 120
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

if str(Path(__file__).resolve().parents[1] / "src") not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from synth_regime_eval import DEFAULT_POLICY, build_stream  # noqa: E402
from valid_regime_eval import pit_abs_error  # noqa: E402

#: Spread buckets in rev/s. The real split's flight frames average 13.7 and the
#: synthetic model at aggressiveness 1.0 gives 9.4, so the interesting range is
#: the bottom of this list.
EDGES = [0.0, 2.0, 5.0, 10.0, 20.0, np.inf]
CRUISE_MIN = 45.0


def score(experiment, policy, n, base_seed, ckpt, duration_s, augment):
    import zoo

    model = zoo.load(experiment, ckpt=ckpt, device="cpu")
    stream = build_stream(policy, base_seed, duration_s, augment)
    buckets: list[list[np.ndarray]] = [[] for _ in range(len(EDGES) - 1)]
    seen = 0
    for frame in stream:
        target = np.asarray(frame["rps"].data, dtype=np.float64)
        pred = np.asarray(model(frame)["rps_pred"].data, dtype=np.float64)
        w = min(pred.shape[1], target.shape[1])
        t, p = target[:, :w], pred[:, :w]
        err = pit_abs_error(p, t)                       # (R, W)
        spread = t.max(axis=0) - t.min(axis=0)          # (W,)
        cruise = t.mean(axis=0) >= CRUISE_MIN
        for b in range(len(EDGES) - 1):
            m = cruise & (spread >= EDGES[b]) & (spread < EDGES[b + 1])
            if m.any():
                buckets[b].append(err[:, m].ravel())
        seen += 1
        if seen >= n:
            break

    rows = []
    for b in range(len(EDGES) - 1):
        v = np.concatenate(buckets[b]) if buckets[b] else np.array([np.nan])
        rows.append({
            "spread_lo": EDGES[b],
            "spread_hi": None if np.isinf(EDGES[b + 1]) else EDGES[b + 1],
            "mae": float(np.mean(v)),
            "n": int(v.size) if buckets[b] else 0,
        })
    return {"experiment": experiment, "ckpt": ckpt, "policy": policy,
            "n_samples": seen, "buckets": rows}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", nargs="+", required=True)
    ap.add_argument("--ckpt", default="last")
    ap.add_argument("--policy", default=None)
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--base-seed", type=int, default=770001)
    ap.add_argument("--duration", type=float, default=8.0)
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out = []
    for exp in args.exp:
        policy = args.policy or DEFAULT_POLICY.get(exp)
        if policy is None:
            print(f"no policy known for {exp}; pass --policy")
            continue
        r = score(exp, policy, args.n, args.base_seed, args.ckpt,
                  args.duration, not args.no_augment)
        out.append(r)
        print(f"\n{exp} @ {args.ckpt}   ({r['n_samples']} clips, cruise columns only)")
        print(f"  {'rotor spread (rev/s)':<24s} {'PIT MAE':>9s} {'frames':>9s}")
        for b in r["buckets"]:
            hi = "inf" if b["spread_hi"] is None else f"{b['spread_hi']:g}"
            print(f"  {b['spread_lo']:g} to {hi:<19s} {b['mae']:9.2f} {b['n']:9d}")
    if args.out and out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=1))
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
