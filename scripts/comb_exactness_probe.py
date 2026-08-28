"""Is the DETERMINISTIC comb solvable exactly by phase-increment refinement?

The static comb is built as sum_k a_k sin(k*Phi(t) + phi_k) with
Phi = 2*pi*cumsum(rps)/sr — exactly harmonic, one shared phase per rotor, the
per-order offsets constant in time. Phase-increment refinement is the matched
tool, because a constant offset cancels in the increment.

So the information is there. The question is what the estimator actually
recovers, and the diagnostic is the init sweep:

  init sigma 0    if the refined error is NOT zero here, the estimator is ADDING
                  error to an already-perfect answer, which indicts its priors
                  rather than the data.
  init sigma up   if the refined error is flat in the init, the estimator has a
                  genuine precision floor; if it tracks the init, it is capture
                  limited and the floor is a search property.

sigma_process is swept alongside, because it is the prior that says how much the
rotor WANDERS between frames. This comb does not wander at all, so a prior tuned
on real recordings is the leading suspect for a floor that has no business
existing here.

    python scripts/comb_exactness_probe.py --n 6
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

SR, DUR, HOP = 16000, 8.0, 512


def clips(n: int, seed0: int = 1000):
    pool = StaticCombNoisePool(sample_rate=SR, duration_s=DUR, n_harmonics=100,
                               n_mics=8, n_rotors=4, rps_kind="full_flight",
                               flight_reuse=1)
    out, s = [], seed0
    while len(out) < n and s < seed0 + 200:
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--iters", type=int, default=12)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    data = clips(args.n)
    print(f"{len(data)} cruise clips, 8 s, static comb (deterministic phase)\n")
    print(f"{'init sigma':>10s} {'sigma_process':>14s} {'init MAE':>9s} {'refined':>9s} {'ratio':>7s}")
    rows = []
    # band_mode "fixed" gives harmonic k a capture range of band_hz / k, so the
    # k=40 stage captures only 0.15 rev/s at the 6 Hz default — SMALLER than the
    # error the 3-iteration default converges to, which makes that stage a trap:
    # the top harmonics sit outside their bands and feed noise back in. Extra
    # iterations repeat the last k_cap, and "k_scaled" replaces the fixed Hz band
    # with a constant shaft-rate trust region (band_b0 rev/s at every order).
    for mode, sp in (("fixed", 2.0), ("k_scaled", 2.0), ("k_scaled", 0.25)):
        print(f"\n--- band_mode={mode}  sigma_process={sp}  n_iter={args.iters} ---")
        for sig in (0.0, 0.02, 0.1, 0.5):
            ini, ref = [], []
            for i, (audio, truth, ft) in enumerate(data):
                rng = np.random.default_rng(100 + i)
                r0 = truth + (rng.normal(0.0, sig, truth.shape) if sig > 0 else 0.0)
                try:
                    est, _ = pi_kalman_refine(audio, r0, ft, sr=SR,
                                              n_iter=args.iters, sigma_process=sp,
                                              band_mode=mode)
                except Exception as exc:
                    print(f"   clip {i} failed: {type(exc).__name__}: {exc}")
                    continue
                ini.append(float(np.mean(np.abs(r0 - truth))))
                ref.append(float(np.mean(np.abs(np.asarray(est) - truth))))
            if not ref:
                continue
            i_m, r_m = float(np.mean(ini)), float(np.mean(ref))
            rows.append({"band_mode": mode, "sigma_process": sp, "init_sigma": sig,
                         "init_mae": i_m, "refined_mae": r_m, "n": len(ref)})
            print(f"{sig:>10.2f} {sp:>14.2f} {i_m:>9.4f} {r_m:>9.4f} "
                  f"{(r_m / i_m if i_m > 0 else float('nan')):>7.2f}")
    if args.out and rows:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(rows, indent=1))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
