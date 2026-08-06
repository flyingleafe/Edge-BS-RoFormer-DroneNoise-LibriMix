#!/usr/bin/env python3
"""Comb displacement with its mandatory null controls (GitHub issue 17, section B).

One measurement — ``tracking.comb_displacement.measure_variant`` — run per frozen
window, per rotor, under three carriers:

  on    ``k * g_rot(t)``            the measurement itself
  off   ``(k + 0.5) * g_rot(t)``    off-comb null: no rotor line can exist there
  mis   ``k * g_partner(t)``        correspondence-breaking null: real spectra,
                                    telemetry from a different window

The nulls are not optional. A peak-pick inside a search window of half-width W
returns about W/2 on pure noise, which killed the campaign's high-k claim
(measured 0.0856 against a null of 0.0857). Report every number beside its
nulls.

The rotor-permutation null of issue 17 section B is NOT here, and the reason is
structural. This measurement is defined by its carrier, so a permuted carrier
``k * g_other(t)`` is the measurement OF the other rotor, under a different
name: the collision gate must skip the rotor whose line is the carrier, else the
carrier collides with its own line at every frame and the unit returns NaN. A
permutation null needs a quantity that is attached to a rotor INDEPENDENTLY of
the carrier — a fitted trajectory scored against telemetry, for example. It
belongs to the refinement driver (``refine_kscaled.py``), not here.

The windows come from ``tracking.protocols.load_prep_window``, the ONE reader of
the frozen ``beatvk-valid-raw`` prep cache. Code comes from this checkout, data
from the data root — the two are different roots.

Examples:
  python scripts/displacement/nullcontrol.py --jobs 8
  python scripts/displacement/nullcontrol.py --windows FLY124__w03 --variants on,off
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]  # this checkout (code)
sys.path.insert(0, str(ROOT / "src"))

from tracking.protocols import load_prep_window  # noqa: E402
from utils.gridrun import Unit, add_gridrun_args, gridrun_from_args  # noqa: E402

OUT_DEFAULT = "results/displacement/nullcontrol"
N_ROTORS = 4
VARIANTS = ("on", "off", "mis")
FIELDS = ("low_k_series_mae", "high_k_series_mae", "low_k_series_mean", "high_k_series_mean")

# mismatch pairing: audio of key -> telemetry of value (same class, other flight)
PARTNER = {
    "free-flight_nosource_room1__w00": "free-flight_speech-low_room1__w00",
    "free-flight_nosource_room1__w01": "free-flight_speech-low_room1__w01",
    "free-flight_nosource_room1__w02": "free-flight_speech-low_room1__w02",
    "free-flight_speech-low_room1__w00": "free-flight_whitenoise-low_room1__w00",
    "free-flight_speech-low_room1__w01": "free-flight_whitenoise-low_room1__w01",
    "free-flight_speech-low_room1__w02": "free-flight_whitenoise-low_room1__w02",
    "free-flight_whitenoise-low_room1__w00": "free-flight_nosource_room1__w00",
    "free-flight_whitenoise-low_room1__w01": "free-flight_nosource_room1__w01",
    "free-flight_whitenoise-low_room1__w02": "free-flight_nosource_room1__w02",
    "FLY124__w00": "FLY124__w01",
    "FLY124__w01": "FLY124__w00",
    "FLY124__w02": "FLY124__w04",
    "FLY124__w03": "FLY124__w05",
    "FLY124__w04": "FLY124__w02",
    "FLY124__w05": "FLY124__w03",
}

PROTOCOL = {
    "dataset": "beatvk-valid-raw@54849c13ed3a",
    "variants": {
        "on": "carrier k * g_rot(t) — the measurement",
        "off": "carrier (k + 0.5) * g_rot(t) — off-comb null, no rotor line",
        "mis": "carrier k * g_partner(t) — mismatched-telemetry null",
    },
    "band_hz_k": "min(3 k, 0.45 * mean rate) Hz",
    "search_half_width": "min(1.5 k, 8) Hz, further capped at 0.9 * band",
    "collision_gate": "a real rotor line within 1.6 * search_hz(k) of the carrier "
    "harmonic gates that (harmonic, frame) out; the carrier's own rotor is skipped",
    "entry_schema": "per_k[k] = [peak_offset, median_snr_lin, std, n_eff, "
    "prominence_db, prominence_offset, pulse_pair_offset, pp_coherence, "
    "frac_uncollided, search_half_width_rev_s]",
}


def worker(unit: Unit) -> dict[str, Any]:
    """One (window, rotor) unit: every requested variant of that measurement."""
    import numpy as np

    from tracking.comb_displacement import DisplacementConfig, measure_variant

    p = unit.params
    key, rot = str(p["key"]), int(p["rotor"])
    ks = list(range(1, int(p["k_max"]) + 1))
    cfg = DisplacementConfig()
    win = load_prep_window(key)
    r_true = win["r"]
    mis = load_prep_window(PARTNER[key])["r"][rot] if "mis" in p["variants"] else r_true[rot]
    # The gate's rotor argument names the rotor whose line IS the carrier, and
    # that line must not count as an interferer.
    carriers = {"on": r_true[rot], "off": r_true[rot], "mis": mis}
    out = {}
    for n in p["variants"]:
        args = (win["audio"], win["ft"], carriers[n], r_true, rot, ks)
        out[n] = measure_variant(*args, half=(n == "off"), cfg=cfg)
    return {
        "key": key,
        "regime": win["regime"],
        "rotor": rot,
        "rotor_mean_rev_s": round(float(np.mean(r_true[rot])), 3),
        "variants": out,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Protocol block plus the pooled DREGON / FLY124 offsets per variant."""
    import numpy as np

    pooled: dict[str, Any] = {}
    for name, is_fly in (("dregon", False), ("fly124", True)):
        sel = [r for r in rows if r["key"].startswith("FLY124") == is_fly]
        per_var: dict[str, Any] = {}
        for var in VARIANTS:
            got = [r["variants"][var] for r in sel if var in r["variants"]]
            if not got:
                continue
            per_var[var] = {f: round(float(np.mean([g[f] for g in got])), 4) for f in FIELDS}
            per_var[var]["n_units"] = len(got)
        pooled[name] = per_var
    return {"protocol": PROTOCOL, "pooled": pooled}


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--windows", default="", help="comma-separated keys (default: all 15)")
    ap.add_argument("--variants", default=",".join(VARIANTS), help="comma-separated variants")
    ap.add_argument("--k-max", type=int, default=40, help="highest harmonic index")
    ap.add_argument("--out", default=OUT_DEFAULT, help="output directory")
    add_gridrun_args(ap, jobs=8)
    args = ap.parse_args()

    keys = [k.strip() for k in args.windows.split(",") if k.strip()] or list(PARTNER)
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    for bad in [k for k in keys if k not in PARTNER]:
        ap.error(f"unknown window {bad!r}; known: {', '.join(PARTNER)}")
    for bad in [v for v in variants if v not in VARIANTS]:
        ap.error(f"unknown variant {bad!r}; known: {', '.join(VARIANTS)}")

    units = [
        Unit(f"{k}__r{r}", {"key": k, "rotor": r, "k_max": args.k_max, "variants": variants})
        for k in keys
        for r in range(N_ROTORS)
    ]
    print(f"[nullcontrol] {len(units)} units x {len(variants)} variants", flush=True)
    res = gridrun_from_args(args, units, worker, args.out, summarize=summarize)
    raise SystemExit(res.exit_code)


if __name__ == "__main__":
    main()
