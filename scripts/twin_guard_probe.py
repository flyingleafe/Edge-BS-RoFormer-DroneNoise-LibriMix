"""Can a wider twin guard recover what cross-rotor collisions cost pi_kalman?

Measured on the deterministic static comb with an EXACT initialization: four
rotors inject 0.0160 rev/s of error, one rotor injects 0.0006 — a factor of 27,
so nearly all of what the estimator adds is collision bias leaking past the twin
gate. The gate drops harmonic k of rotor i at frames where any harmonic of any
other rotor comes within `band + guard_hz` of it.

Before building a joint 4-rotor filter, this asks the one-parameter question:
does simply widening `guard_hz` close the gap? Widening discards more
measurements, so there is a trade — too wide and the gate starves the smoother.
`n_meas` is reported alongside so the trade is visible rather than inferred.

    python scripts/twin_guard_probe.py --n 5
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--iters", type=int, default=6)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    data = clips(args.n)
    print(f"{len(data)} cruise clips, four-rotor deterministic comb, EXACT init")
    print("reference: one rotor injects 0.0006; four rotors at guard 1.0 inject 0.0160\n")
    print(f"{'guard_hz':>9s} {'pair_mode':>10s} {'injected':>10s} {'measurements kept':>18s}")
    rows = []
    for pair_mode in ("gate", "joint"):
        for guard in (1.0, 3.0, 8.0, 20.0):
            errs, kept = [], []
            for audio, truth, ft in data:
                try:
                    est, diag = pi_kalman_refine(audio, truth.copy(), ft, sr=SR,
                                                 n_iter=args.iters, band_mode="k_scaled",
                                                 guard_hz=guard, pair_mode=pair_mode)
                except Exception as exc:
                    print(f"   guard={guard} {pair_mode}: {type(exc).__name__}: {exc}")
                    continue
                errs.append(float(np.mean(np.abs(np.asarray(est) - truth))))
                n_m = [v.get("n_meas") for it in diag.get("iters", [])
                       for v in (it.get("rotors") or [])
                       if isinstance(v, dict) and v.get("n_meas") is not None]
                if n_m:
                    kept.append(float(np.mean(n_m)))
            if not errs:
                continue
            m = float(np.mean(errs))
            km = float(np.mean(kept)) if kept else float("nan")
            rows.append({"guard_hz": guard, "pair_mode": pair_mode,
                         "injected": m, "n_meas": km})
            print(f"{guard:>9.1f} {pair_mode:>10s} {m:>10.4f} {km:>18.0f}")
    if args.out and rows:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(rows, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
