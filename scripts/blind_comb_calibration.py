"""Calibrating a FULLY BLIND chain on the deterministic static comb.

The oracle probe established what phase-increment refinement can do when it is
handed a good starting point: about 0.03 rev/s in `k_scaled` band mode, against
the best neural model's 2.155. It also established what it needs in order to get
there — refinement is CAPTURE limited. In `k_scaled` mode a rate error `dr`
displaces harmonic k by `k*dr` Hz and the band is `k*band_b0` Hz, so the trust
region is a constant `band_b0` rev/s at every order. Oracle inits inside it
refine; an init at 0.402 against a 0.35 region barely moved.

That gives a LADDER CONDITION for any blind chain: every stage's capture must
exceed the previous stage's error. This measures each rung on the static comb,
where the truth is exact and the phase is deterministic, so nothing but the
algorithm is on trial:

  seed      blind_seed's constant per-rotor bases against the true mean speed.
  const     the same bases as a flat trajectory against the true time-varying
            speed — the error refinement actually receives, which is larger
            than `seed` by however much the rotor moves inside the window.
  refined   pi_kalman from that seed, sweeping band_b0 to find the trust region
            wide enough to swallow the blind error.

If a wide-then-annealed schedule carries the blind seed down to the oracle's
0.03, the comb is blindly solvable to arbitrary precision and the schedule IS
the calibration.

    python scripts/blind_comb_calibration.py --n 6
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

if str(Path(__file__).resolve().parents[1] / "src") not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data_processing.rotor_spectral_model import StaticCombNoisePool  # noqa: E402
from tracking.phase_increment_tracker import pi_kalman_refine  # noqa: E402
from tracking.vk_blind_seeding import blind_seed  # noqa: E402

SR, DUR, HOP = 16000, 8.0, 512


def clips(n: int, seed0: int = 1000):
    pool = StaticCombNoisePool(sample_rate=SR, duration_s=DUR, n_harmonics=100,
                               n_mics=8, n_rotors=4, rps_kind="full_flight",
                               flight_reuse=1)
    out, s = [], seed0
    while len(out) < n and s < seed0 + 300:
        tf = pool.sample_timeframe(np.random.default_rng(s), DUR)
        s += 1
        rps = np.asarray(tf["rps"].data, dtype=np.float64)
        if float(np.mean(rps)) < 45.0:
            continue
        audio = np.asarray(tf["audio"].data, dtype=np.float64)
        n_fr = rps.shape[1] // HOP
        ft = (np.arange(n_fr) * HOP + HOP / 2) / SR
        truth = np.stack([np.interp(ft, np.arange(rps.shape[1]) / SR, rps[r])
                          for r in range(rps.shape[0])])
        out.append((audio, truth, ft))
    return out


def match(pred: np.ndarray, truth: np.ndarray) -> float:
    """Mean abs error under the best rotor permutation (small R: brute force)."""
    import itertools
    return min(float(np.mean(np.abs(pred[list(p)] - truth)))
               for p in itertools.permutations(range(truth.shape[0])))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--iters", type=int, default=12)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    data = clips(args.n)
    print(f"{len(data)} cruise clips, 8 s, deterministic static comb\n")

    seeds, consts, wanders = [], [], []
    for audio, truth, ft in data:
        res = blind_seed(audio, float(SR), truth.shape[0])
        bases = np.asarray(res.bases, dtype=np.float64)
        r0 = np.tile(bases[:, None], (1, truth.shape[1]))
        seeds.append(match(bases[:, None], truth.mean(axis=1)[:, None]))
        consts.append(match(r0, truth))
        wanders.append(float(np.mean(truth.max(axis=1) - truth.min(axis=1))))
    print(f"blind seed vs true MEAN speed      {np.mean(seeds):7.3f} rev/s")
    print(f"blind seed vs true TRACK (flat)    {np.mean(consts):7.3f} rev/s")
    print(f"  in-window rotor excursion        {np.mean(wanders):7.3f} rev/s")
    print(f"\n  -> refinement must capture at least {np.mean(consts):.3f} rev/s\n")

    print(f"{'band_b0':>9s} {'blind->refined':>15s} {'ratio':>7s}")
    rows = [{"stage": "seed_mean", "mae": float(np.mean(seeds))},
            {"stage": "seed_track", "mae": float(np.mean(consts))},
            {"stage": "excursion", "mae": float(np.mean(wanders))}]
    for b0 in (0.35, 1.0, 2.0, 4.0, 8.0):
        errs = []
        for audio, truth, ft in data:
            res = blind_seed(audio, float(SR), truth.shape[0])
            r0 = np.tile(np.asarray(res.bases, dtype=np.float64)[:, None],
                         (1, truth.shape[1]))
            try:
                est, _ = pi_kalman_refine(audio, r0, ft, sr=SR, n_iter=args.iters,
                                          band_mode="k_scaled", band_b0=b0)
            except Exception as exc:
                print(f"   b0={b0}: {type(exc).__name__}: {exc}")
                continue
            errs.append(match(np.asarray(est, dtype=np.float64), truth))
        if not errs:
            continue
        m = float(np.mean(errs))
        rows.append({"stage": "refined", "band_b0": b0, "mae": m})
        print(f"{b0:>9.2f} {m:>15.4f} {m / np.mean(consts):>7.2f}")

    if args.out and rows:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(rows, indent=1))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
