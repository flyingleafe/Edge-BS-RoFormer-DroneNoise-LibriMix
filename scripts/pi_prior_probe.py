"""Why does pi_kalman move AWAY from an exact initialization?

pi_kalman does not refine a trajectory — it estimates a correction dr(t) and
adds it (module docstring step 4, `r_hat += dr`). Handed the truth, the correct
correction is zero, but the smoother rebuilds dr from noisy measurements and
adds whatever it finds. `p0 = sigma_prior**2` (line 1255) is the only term that
expresses trust in the input, and its default 2.0 rev/s is nearly diffuse.

Two competing explanations for the 0.028 rev/s it injects into a perfect init:

  prior   the diffuse prior discards the init and the output is the estimator's
          noise floor. Then shrinking sigma_prior should shrink the injection
          roughly in proportion.
  bias    the increments carry a real bias — cross-rotor collisions (the
          two-phasor winding-number bias the docstring warns about) or
          harmonics crossing Nyquist mid-window — which the smoother tracks
          faithfully. Then the injection is flat in sigma_prior and no prior
          setting fixes it.

n_rotors is swept alongside: at 1 rotor there are no cross-rotor collisions at
all, so if the injection collapses there, collisions are the bias source.

    python scripts/pi_prior_probe.py
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


def single_rotor_clips(n: int, seed0: int = 1000):
    """One-rotor combs built directly, bypassing the 4-rotor frame geometry.

    The pool cannot make these: `rotor_pos` is fixed at four positions and
    td.Frame rejects the dim mismatch. Going straight to the waveform gives the
    collision-free control the hypothesis needs.
    """
    from data_processing.rotor_spectral_model import ProfileRanges, sample_profile, _comb_waveform
    out = []
    for i in range(n):
        rng = np.random.default_rng(seed0 + i)
        n_t = int(SR * DUR)
        # A cruise trajectory with the same excursion scale as the pool's.
        t = np.arange(n_t) / SR
        rps = 75.0 + 6.0 * np.sin(2 * np.pi * 0.11 * t) + 2.0 * np.sin(2 * np.pi * 0.37 * t)
        prof = sample_profile(rng, ProfileRanges(), n_harmonics=100,
                              ref_rps=float(np.median(rps)), sample_rate=SR)
        comb = _comb_waveform(rps, np.asarray(prof.a_k, dtype=np.float64), SR, rng)
        amp = (rps / 80.0) ** 2.5
        audio = (comb * amp)[None] + 0.01 * rng.standard_normal((1, n_t))
        n_fr = n_t // HOP
        ft = (np.arange(n_fr) * HOP + HOP / 2) / SR
        truth = np.interp(ft, t, rps)[None]
        out.append((audio, truth, ft))
    return out


def clips(n: int, n_rotors: int, seed0: int = 1000):
    pool = StaticCombNoisePool(sample_rate=SR, duration_s=DUR, n_harmonics=100,
                               n_mics=8, n_rotors=n_rotors, rps_kind="full_flight",
                               flight_reuse=1)
    out, s = [], seed0
    while len(out) < n and s < seed0 + 300:
        tf = pool.sample_timeframe(np.random.default_rng(s), DUR)
        s += 1
        rps = np.asarray(tf["rps"].data, dtype=np.float64)
        if rps.ndim == 1:
            rps = rps[None]
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

    rows = []
    for n_rot in (4, 1):
        data = clips(args.n, n_rot) if n_rot > 1 else single_rotor_clips(args.n)
        if not data:
            print(f"n_rotors={n_rot}: no cruise clips"); continue
        print(f"\n=== n_rotors = {n_rot} "
              f"({'cross-rotor collisions possible' if n_rot > 1 else 'no collisions by construction'})"
              f", {len(data)} clips, EXACT init ===")
        print(f"{'sigma_prior':>12s} {'injected error':>15s}")
        for sp in (2.0, 0.5, 0.1, 0.02):
            errs = []
            for audio, truth, ft in data:
                try:
                    est, _ = pi_kalman_refine(audio, truth.copy(), ft, sr=SR,
                                              n_iter=args.iters, band_mode="k_scaled",
                                              sigma_prior=sp)
                except Exception as exc:
                    print(f"   {type(exc).__name__}: {exc}")
                    continue
                errs.append(float(np.mean(np.abs(np.asarray(est) - truth))))
            if not errs:
                continue
            m = float(np.mean(errs))
            rows.append({"n_rotors": n_rot, "sigma_prior": sp, "injected": m})
            print(f"{sp:>12.2f} {m:>15.4f}")
    if args.out and rows:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(rows, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
