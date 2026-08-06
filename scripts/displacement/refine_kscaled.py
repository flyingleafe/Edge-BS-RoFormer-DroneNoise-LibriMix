#!/usr/bin/env python3
"""Realign DREGON telemetry onto the acoustic comb: pi_kalman initialized FROM telemetry.

One arm is ``TAG:BAND_MODE:B0:PAIR_MODE``. ``band_mode=k_scaled`` gives harmonic
k the band ``k * B0`` Hz, that is the SAME capture range in rev/s at every
harmonic, so the comb moves coherently or not at all. ``band_mode=fixed`` uses
the ``--band-hz`` schedule instead and ``B0`` is ignored; that reproduces the two
superseded arms, both of which failed:

  narrow  --arm narrow:fixed:0:gate --band-hz 6,3,1.5 --k-caps 24,60,100 \\
          --k-max 100 --f-max 7000                      (INERT: moved -0.03 %)
  wide    --arm wide:fixed:0:gate --band-hz 24,8,3 --k-caps 8,24,60 \\
          --k-max 100 --f-max 7000 --off-comb-hz 40     (COLLAPSED onto twins)

The identity test is rotor ORDER plus the inter-rotor GAPS, before and after.
The nearest-neighbour test the earlier version of this script printed was WRONG:
every rotor moves down together, so under a common-mode shift r0's new rate sits
nearest r2's old rate and the test reports a collapse that did not happen.

The mean shift and the delta RMS are TWO different corrections and are reported
separately. The RMS is dominated by the de-staircasing of the tachometer's
0.269 rev/s / 49.7 Hz quantisation, not by the systematic scale error.

GitHub issue 17 cites this file path.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]  # this checkout (code)
sys.path.insert(0, str(ROOT / "src"))

import hk_core as H  # noqa: E402
import numpy as np  # noqa: E402

from tracking.phase_increment_tracker import pi_kalman_refine  # noqa: E402

# THE identity test of the campaign (order + inter-rotor gaps) lives in the
# library now, beside the fitter that has to pass it.
from tracking.telemetry_refit import order_and_gaps  # noqa: E402

DEFAULT_ARMS = (
    "kscaled_b3:k_scaled:3:gate",
    "kscaled_b3_joint:k_scaled:3:joint",
    "kscaled_b1:k_scaled:1:gate",
)


def _floats(text: str) -> tuple[float, ...]:
    return tuple(float(x) for x in text.split(",") if x.strip())


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--arm", action="append", default=None, help="TAG:BAND_MODE:B0:PAIR_MODE")
    ap.add_argument("--recording", default="free-flight_nosource_room1")
    ap.add_argument("--t0", type=float, default=22.56481)
    ap.add_argument("--dur", type=float, default=16.0)
    ap.add_argument("--fs-ft", type=float, default=62.5, help="trajectory frame rate (Hz)")
    ap.add_argument("--n-iter", type=int, default=3)
    ap.add_argument("--k-max", type=int, default=80)
    ap.add_argument("--k-caps", default="80,80,80", help="per-iteration harmonic caps")
    ap.add_argument("--band-hz", default="6,3,1.5", help="band_mode=fixed schedule (Hz)")
    ap.add_argument("--f-max", type=float, default=7500.0)
    ap.add_argument("--off-comb-hz", type=float, default=11.0)
    # Not the script directory: the committed kscaled_telemetry_init.json beside
    # this file is the FROZEN result of the recorded run, and a rerun must not
    # overwrite it.
    ap.add_argument("--out", default="results/displacement/refine_kscaled", help="output directory")
    ap.add_argument("--save-npz", action="store_true", help="write ft / r_init / r_ref per arm")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    audio, sr, g, _rates = H.load_raw(args.recording, args.t0, args.dur)
    ft = np.arange(0, args.dur, 1.0 / args.fs_ft)
    t_aud = np.arange(audio.shape[1]) / sr
    r_init = np.stack([np.interp(ft, t_aud, g[r]) for r in range(g.shape[0])])
    order0, gaps0 = order_and_gaps(r_init)
    print(f"init rates {np.round(r_init.mean(1), 3)}  order {order0}  gaps {gaps0}", flush=True)

    caps = tuple(int(c) for c in _floats(args.k_caps))
    bands = _floats(args.band_hz)
    if not bands:
        ap.error("--band-hz needs at least one value")
    band_arg: float | tuple[float, ...] = bands[0] if len(bands) == 1 else bands
    out: dict[str, Any] = {}
    for spec in args.arm or DEFAULT_ARMS:
        parts = spec.split(":")
        if len(parts) != 4:
            ap.error(f"bad --arm {spec!r}: expected TAG:BAND_MODE:B0:PAIR_MODE")
        tag, band_mode, b0, pair_mode = parts[0], parts[1], float(parts[2]), parts[3]
        r_ref, _diag = pi_kalman_refine(
            audio,
            r_init,
            ft,
            sr=sr,
            n_iter=args.n_iter,
            k_max=args.k_max,
            f_max=args.f_max,
            k_caps=caps,
            fs_env=args.fs_ft,
            band_hz=band_arg,
            off_comb_hz=args.off_comb_hz,
            band_mode=band_mode,
            band_b0=b0,
            pair_mode=pair_mode,
        )
        d = r_ref - r_init
        mean, rms = d.mean(1), d.std(1)
        pct = 100.0 * mean / r_init.mean(1)
        order1, gaps1 = order_and_gaps(r_ref)
        out[tag] = {
            "band_mode": band_mode,
            "band_b0": b0,
            "pair_mode": pair_mode,
            "delta_mean": [float(x) for x in mean],
            "pct_of_rate": [float(x) for x in pct],
            "delta_rms": [float(x) for x in rms],
            "final": [float(x) for x in r_ref.mean(1)],
            "order_init": order0,
            "order_ref": order1,
            "gaps_init": gaps0,
            "gaps_ref": gaps1,
            "order_kept": order1 == order0,
        }
        print(f"\n== {tag} ==")
        print("  delta mean rev/s:", np.round(mean, 4))
        print("  as % of rate    :", np.round(pct, 3))
        print("  delta rms       :", np.round(rms, 4))
        print("  final rates     :", np.round(r_ref.mean(1), 3))
        print(f"  order {order0} -> {order1}  ({'KEPT' if order1 == order0 else 'CHANGED'})")
        print(f"  gaps  {gaps0} -> {gaps1}")
        if args.save_npz:
            np.savez(out_dir / f"refined_{tag}.npz", ft=ft, r_init=r_init, r_ref=r_ref)
    (out_dir / "kscaled_telemetry_init.json").write_text(json.dumps(out, indent=1))
    print(f"\nwrote {out_dir / 'kscaled_telemetry_init.json'}")


if __name__ == "__main__":
    main()
