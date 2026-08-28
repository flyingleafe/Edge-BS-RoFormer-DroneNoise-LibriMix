"""Does joint four-rotor fusion remove the collision cost of pi_kalman?

The sequential refiner injects 0.0160 rev/s into an EXACT initialization of a
four-rotor static comb, and 0.0006 rev/s when only one rotor is present. The
whole 27x gap is cross-rotor collision. `joint_pi_kalman_refine` keeps the
collided measurements and splits their observation row between the rotors that
share the band, instead of discarding them.

Arms, all on an exact initialization so the number IS the injected error:

  seq          the sequential refiner, four rotors      (target to beat: 0.0160)
  joint/power  the joint filter, power-weighted rows
  joint/hard   the joint filter, whole row to the strongest line — the control
               that says how much the SOFT split is worth
  seq/1rotor   the collision-free floor                 (target to reach: 0.0006)

    python scripts/joint_kalman_probe.py --n 5 --iters 6
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

if str(Path(__file__).resolve().parents[1] / "src") not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tracking.joint_phase_kalman import joint_pi_kalman_refine  # noqa: E402
from tracking.phase_increment_tracker import pi_kalman_refine  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pi_prior_probe import clips, single_rotor_clips  # noqa: E402


def synth_clips(n: int, n_rotors: int, seed0: int = 4000, spread: float = 10.0):
    """R combs of the SAME trajectory family, one channel, one noise level.

    The one-rotor control of `pi_prior_probe` came from a different synthesis
    path than the four-rotor pool clips (smooth trajectory, one mic, its own
    noise), so its 27x advantage confounds rotor count with everything else.
    Here only the rotor count changes, so whatever the ladder shows IS the
    collision cost.
    """
    from data_processing.rotor_spectral_model import (  # noqa: PLC0415
        ProfileRanges, _comb_waveform, sample_profile,
    )

    SR, DUR, HOP = 16000, 8.0, 512
    out = []
    for c in range(n):
        rng = np.random.default_rng(seed0 + c)
        n_t = int(SR * DUR)
        t = np.arange(n_t) / SR
        audio = np.zeros(n_t)
        truths = []
        offs = np.linspace(-0.5, 0.5, n_rotors) * spread if n_rotors > 1 else np.zeros(1)
        for i in range(n_rotors):
            ph = rng.uniform(0, 2 * np.pi, 2)
            rps = (
                75.0 + offs[i]
                + 6.0 * np.sin(2 * np.pi * 0.11 * t + ph[0])
                + 2.0 * np.sin(2 * np.pi * 0.37 * t + ph[1])
            )
            prof = sample_profile(rng, ProfileRanges(), n_harmonics=100,
                                  ref_rps=float(np.median(rps)), sample_rate=SR)
            comb = _comb_waveform(rps, np.asarray(prof.a_k, dtype=np.float64), SR, rng)
            audio += comb * (rps / 80.0) ** 2.5
            truths.append(rps)
        audio = audio[None] + 0.01 * rng.standard_normal((1, n_t))
        n_fr = n_t // HOP
        ft = (np.arange(n_fr) * HOP + HOP / 2) / SR
        truth = np.stack([np.interp(ft, t, tr) for tr in truths])
        out.append((audio, truth, ft))
    return out


def run(fn, data, **kw):
    errs, t0 = [], time.time()
    diag = None
    for audio, truth, ft in data:
        est, dg = fn(audio, truth.copy(), ft, sr=16000, **kw)
        diag = dg
        errs.append(float(np.mean(np.abs(np.asarray(est) - truth))))
    return float(np.mean(errs)), errs, time.time() - t0, diag


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--iters", type=int, default=6)
    ap.add_argument("--out", default=None)
    ap.add_argument("--arms", default="seq,power,hard,seq1")
    ap.add_argument("--ladder", action="store_true",
                    help="rotor-count ladder on matched synthesis instead")
    args = ap.parse_args()
    want = set(args.arms.split(","))

    if args.ladder:
        print("matched synthesis, only the rotor count changes, exact init\n")
        print(f"{'arm':>14s} {'injected err':>13s} {'per-clip spread':>18s} {'s':>7s}")
        rows = []
        for n_rot in (1, 2, 3, 4):
            data = synth_clips(args.n, n_rot)
            m, errs, secs, _ = run(pi_kalman_refine, data,
                                   n_iter=args.iters, band_mode="k_scaled")
            rows.append({"arm": f"seq/R={n_rot}", "n_rotors": n_rot, "injected": m,
                         "errs": errs, "secs": secs})
            print(f"{'seq/R=' + str(n_rot):>14s} {m:>13.4f} "
                  f"{min(errs):>8.4f}-{max(errs):<9.4f} {secs:>7.1f}")
            if n_rot == 4:
                for wm in ("power", "hard", "drop"):
                    m2, e2, s2, _ = run(joint_pi_kalman_refine, data, n_iter=args.iters,
                                        band_mode="k_scaled", weight_mode=wm)
                    rows.append({"arm": f"joint/{wm} R=4", "n_rotors": 4, "injected": m2,
                                 "errs": e2, "secs": s2})
                    print(f"{'joint/' + wm + ' R=4':>14s} {m2:>13.4f} "
                          f"{min(e2):>8.4f}-{max(e2):<9.4f} {s2:>7.1f}")
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(json.dumps(rows, indent=1))
        return 0

    d4 = clips(args.n, 4)
    d1 = single_rotor_clips(args.n)
    print(f"{len(d4)} four-rotor clips, {len(d1)} one-rotor clips, exact init\n")

    rows = []
    plan = [
        ("seq", "seq", d4, pi_kalman_refine, dict(n_iter=args.iters, band_mode="k_scaled")),
        ("power", "joint/power", d4, joint_pi_kalman_refine,
         dict(n_iter=args.iters, band_mode="k_scaled", weight_mode="power")),
        ("hard", "joint/hard", d4, joint_pi_kalman_refine,
         dict(n_iter=args.iters, band_mode="k_scaled", weight_mode="hard")),
        ("seq1", "seq/1rotor", d1, pi_kalman_refine,
         dict(n_iter=args.iters, band_mode="k_scaled")),
    ]
    print(f"{'arm':>12s} {'injected err':>13s} {'per-clip spread':>18s} {'s':>7s}")
    for key, label, data, fn, kw in plan:
        if key not in want or not data:
            continue
        m, errs, secs, diag = run(fn, data, **kw)
        rows.append({"arm": label, "injected": m, "errs": errs, "secs": secs})
        if key == "power" and diag is not None:
            last = diag["iters"][-1]
            rows[-1]["collided_frac"] = last.get("collided_frac")
            rows[-1]["never_clean"] = last.get("n_lines_never_clean")
            rows[-1]["n_lines"] = last.get("n_lines")
        print(f"{label:>12s} {m:>13.4f} {min(errs):>8.4f}-{max(errs):<9.4f} {secs:>7.1f}")

    if rows and any(r["arm"] == "joint/power" for r in rows):
        p = next(r for r in rows if r["arm"] == "joint/power")
        print(
            f"\ncollided band-frames {p.get('collided_frac')}, "
            f"{p.get('never_clean')}/{p.get('n_lines')} lines never clean"
        )
    if args.out and rows:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(rows, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
