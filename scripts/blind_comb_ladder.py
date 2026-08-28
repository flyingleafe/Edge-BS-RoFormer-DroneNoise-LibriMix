"""The FULL blind chain on the deterministic static comb, measured rung by rung.

A previous probe established that the shortcut does not exist: `blind_seed`
emits a CONSTANT base per rotor, the rotor moves 13.8 rev/s inside an 8 s
window, and phase refinement from that flat seed is inert at every trust region
from 0.35 to 8.0 rev/s (ratio 1.00). Something must supply the time dimension
before refinement can engage, which is what the vit2dsp ladder does.

So this measures the real chain, with the truth withheld from every stage — the
`rps` entry is stripped from the frame, so vit2dsp seeds itself blind:

  vit2dsp     blind seed -> Viterbi pair-mean -> spatial joint 2-rotor Viterbi
              -> midband -> refine. The validated ladder (DREGON pooled 0.688).
  pi_kalman   phase-increment refinement, applied repeatedly. `k_scaled` gives a
              constant band_b0 rev/s trust region; `band_anneal="posterior"`
              carries the region down across applications, which is the
              calibration knob: each application must capture the previous
              one's error.

The ladder condition is capture(n) > error(n-1). Reporting the error after every
rung shows which rung binds, and whether the chain reaches the oracle floor of
about 0.03 rev/s that pi_kalman achieves when handed a good start.

    python scripts/blind_comb_ladder.py --n 4
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

if str(Path(__file__).resolve().parents[1] / "src") not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import tdseries as td  # noqa: E402
from data_processing.rotor_spectral_model import StaticCombNoisePool  # noqa: E402
from tracking.top import pi_kalman_stage, vit2dsp_stage  # noqa: E402

SR, DUR, HOP = 16000, 8.0, 512


def match(pred: np.ndarray, truth: np.ndarray) -> float:
    return min(float(np.mean(np.abs(pred[list(p)] - truth)))
               for p in itertools.permutations(range(truth.shape[0])))


def score(frame: td.Frame, truth: np.ndarray, ft: np.ndarray) -> float:
    r = np.asarray(frame["rps"].data, dtype=np.float64)
    t = np.asarray(frame["rps"].timestamps if hasattr(frame["rps"], "timestamps") else ft,
                   dtype=np.float64)
    if r.shape[1] != truth.shape[1]:
        r = np.stack([np.interp(ft, t[: r.shape[1]], r[i]) for i in range(r.shape[0])])
    return match(r, truth)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--applications", type=int, default=4)
    ap.add_argument("--band-b0", type=float, default=2.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    pool = StaticCombNoisePool(sample_rate=SR, duration_s=DUR, n_harmonics=100,
                               n_mics=8, n_rotors=4, rps_kind="full_flight",
                               flight_reuse=1)
    got, s = 0, 1000
    per_rung: dict[str, list[float]] = {}
    while got < args.n and s < 1000 + 300:
        tf = pool.sample_timeframe(np.random.default_rng(s), DUR); s += 1
        rps = np.asarray(tf["rps"].data, dtype=np.float64)
        if float(np.mean(rps)) < 45.0:
            continue
        n_fr = rps.shape[1] // HOP
        ft = (np.arange(n_fr) * HOP + HOP / 2) / SR
        truth = np.stack([np.interp(ft, np.arange(rps.shape[1]) / SR, rps[r])
                          for r in range(rps.shape[0])])
        # Strip the truth: every stage below must work blind.
        blind = td.Frame({k: v for k, v in tf.entries.items() if k != "rps"})
        try:
            f = vit2dsp_stage()(blind)
            e = score(f, truth, ft)
            per_rung.setdefault("vit2dsp", []).append(e)
            for i in range(args.applications):
                f = pi_kalman_stage(band_mode="k_scaled",
                                    band_b0=args.band_b0 if i == 0 else None,
                                    band_anneal="posterior" if i else "none",
                                    n_iter=12)(f)
                per_rung.setdefault(f"pi_kalman x{i + 1}", []).append(
                    score(f, truth, ft))
        except Exception as exc:
            print(f"clip seed {s - 1}: {type(exc).__name__}: {exc}")
            continue
        got += 1

    print(f"\n{got} cruise clips, deterministic static comb, truth withheld\n")
    print(f"{'rung':<16s} {'PIT MAE (rev/s)':>16s} {'n':>4s}")
    rows = []
    for k, v in per_rung.items():
        print(f"{k:<16s} {np.mean(v):>16.4f} {len(v):>4d}")
        rows.append({"rung": k, "mae": float(np.mean(v)), "n": len(v)})
    print("\noracle-init pi_kalman reaches 0.028 rev/s; the neural floor is 2.155")
    if args.out and rows:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(rows, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
