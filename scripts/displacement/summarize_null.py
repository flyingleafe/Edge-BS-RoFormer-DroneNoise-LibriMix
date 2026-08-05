#!/usr/bin/env python3
"""Print every number the null-control verdict needs."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT))
import measure_displacement as M  # noqa: E402

D = json.loads((OUT / "nullcontrol.json").read_text())
W = D["windows"]
DRE = [k for k in W if k.split("__")[0] in M.DREGON_RECS]
FLY = [k for k in W if k.startswith("FLY124") and W[k]["regime"] == "cruise"]
CL = {"DREGON cruise": DRE, "FLY124 cruise": FLY}
I_OFF, I_SNR, I_PROM, I_PP, I_COH, I_SEARCH = 0, 1, 4, 6, 7, 9


def units(keys, variant, field):
    v = [W[k]["rotors"][str(r)][variant][field] for k in keys for r in range(4)]
    return np.array([x for x in v if x is not None and np.isfinite(x)])


def pk(keys, variant, idx, ks, absolute=False):
    v = []
    for k in keys:
        for r in range(4):
            for kk in ks:
                e = W[k]["rotors"][str(r)][variant]["per_k"][str(kk)][idx]
                if e is not None and np.isfinite(e):
                    v.append(abs(e) if absolute else e)
    return np.array(v)


for name, keys in CL.items():
    print(f"\n########## {name}  ({len(keys)} windows x 4 rotors)")
    print("  -- combined-series statistic (the published number) --")
    for f in ("low_k_series_mae", "high_k_series_mae", "low_k_series_mean", "high_k_series_mean"):
        row = {v: round(float(units(keys, v, f).mean()), 4) for v in ("on", "off", "mis")}
        n = len(units(keys, "on", f))
        print(
            f"   {f:<20} on {row['on']:+.4f}   OFF-NULL {row['off']:+.4f}   "
            f"MIS-NULL {row['mis']:+.4f}   (n={n})"
        )
    for bn, ks in (("low", M.LOW_K), ("high", M.HIGH_K)):
        hw = pk(keys, "on", I_SEARCH, ks).mean()
        print(
            f"   analytic {bn}-k: mean search half-width {hw:.3f} rev/s -> "
            f"E|uniform| {hw / 2:.4f} rev/s"
        )
    print("  -- per-k |offset| pooled over the band --")
    for bn, ks in (("low k=2..13", M.LOW_K), ("high k=16..40", M.HIGH_K)):
        on = pk(keys, "on", I_OFF, ks, True)
        off = pk(keys, "off", I_OFF, ks, True)
        mis = pk(keys, "mis", I_OFF, ks, True)
        hw = pk(keys, "on", I_SEARCH, ks)
        fill = np.array(
            [
                abs(e[I_OFF]) / e[I_SEARCH]
                for k in keys
                for r in range(4)
                for kk in ks
                for e in [W[k]["rotors"][str(r)]["on"]["per_k"][str(kk)]]
                if e[I_OFF] is not None
            ]
        )
        print(
            f"   {bn:<14} |delta| on {on.mean():.4f}  off {off.mean():.4f}  "
            f"mis {mis.mean():.4f}  analytic {hw.mean() / 2:.4f}  "
            f"| window fill on {fill.mean():.3f} (0.5 = uniform)"
        )
    print("  -- signed per-k mean offset --")
    for bn, ks in (("low", M.LOW_K), ("high", M.HIGH_K)):
        print(
            f"   {bn:<5} on {pk(keys, 'on', I_OFF, ks).mean():+.4f}  "
            f"off {pk(keys, 'off', I_OFF, ks).mean():+.4f}  "
            f"mis {pk(keys, 'mis', I_OFF, ks).mean():+.4f}"
        )
    print("  -- coherent pulse-pair (window-INDEPENDENT) --")
    for bn, ks in (("low", M.LOW_K), ("high", M.HIGH_K)):
        on = pk(keys, "on", I_PP, ks)
        off = pk(keys, "off", I_PP, ks)
        print(
            f"   {bn:<5} mean {on.mean():+.4f} / MAE {np.abs(on).mean():.4f}   "
            f"NULL mean {off.mean():+.4f} / MAE {np.abs(off).mean():.4f}   "
            f"coh on {pk(keys, 'on', I_COH, ks).mean():.3f} off "
            f"{pk(keys, 'off', I_COH, ks).mean():.3f}"
        )
    print("  -- prominence (dB over in-band floor) and demod SNR --")
    for bn, ks in (("low", M.LOW_K), ("high", M.HIGH_K)):
        on = pk(keys, "on", I_PROM, ks)
        off = pk(keys, "off", I_PROM, ks)
        print(
            f"   {bn:<5} prominence median on {np.median(on):.2f} dB  null {np.median(off):.2f}"
            f"  excess {np.median(on) - np.median(off):+.2f}  "
            f"frac>=6dB on {np.mean(on >= 6):.3f} null {np.mean(off >= 6):.3f}  "
            f"| median SNR on {10 * np.log10(np.median(pk(keys, 'on', I_SNR, ks))):.2f} dB "
            f"null {10 * np.log10(np.median(pk(keys, 'off', I_SNR, ks))):.2f} dB"
        )
    print("  -- restricted to units whose ridge clears the 6 dB bar --")
    for bn, ks in (("low k=2..13", M.LOW_K), ("high k=16..40", M.HIGH_K)):
        rows = [
            (
                W[k]["rotors"][str(r)]["on"]["per_k"][str(kk)],
                W[k]["rotors"][str(r)]["off"]["per_k"][str(kk)],
            )
            for k in keys
            for r in range(4)
            for kk in ks
        ]
        tot = len([1 for a, _ in rows if a[I_PROM] is not None])
        nulln = len([1 for _, b in rows if b[I_PROM] is not None and b[I_PROM] >= 6.0])
        good = [
            a for a, _ in rows if a[I_PROM] is not None and a[I_PROM] >= 6.0 and a[0] is not None
        ]
        if good:
            d = np.array([a[0] for a in good])
            pp = np.array([a[I_PP] for a in good if a[I_PP] is not None])
            print(
                f"   {bn:<14} {len(good)}/{tot} units over bar (null passes {nulln}/{tot}) "
                f"| mean delta {d.mean():+.4f}  MAE {np.abs(d).mean():.4f}  "
                f"pulse-pair mean {pp.mean():+.4f} MAE {np.abs(pp).mean():.4f}"
            )
        else:
            print(f"   {bn:<14} 0/{tot} units over the 6 dB bar (null passes {nulln}/{tot})")
    print(
        "  -- per-k table (k, prom_on, prom_null, snr_on_dB, |d|_on, |d|_null, "
        "pp_on, pp_null, halfwidth) --"
    )
    for kk in list(range(2, 14)) + list(range(16, 41, 2)):
        row = [kk]
        row += [round(float(np.median(pk(keys, v, I_PROM, [kk]))), 2) for v in ("on", "off")]
        row += [round(float(10 * np.log10(np.median(pk(keys, "on", I_SNR, [kk])))), 2)]
        row += [round(float(pk(keys, v, I_OFF, [kk], True).mean()), 3) for v in ("on", "off")]
        row += [round(float(pk(keys, v, I_PP, [kk]).mean()), 3) for v in ("on", "off")]
        row += [round(float(pk(keys, "on", I_SEARCH, [kk]).mean()), 3)]
        print("   ", row)
