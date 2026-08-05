#!/usr/bin/env python3
"""Two-pass RE-CENTRED harmonic sweep — the fix for the min(1.5k, 8) Hz window.

The search half-width is ``min(1.5k, 8)`` Hz, i.e. ``min(1.5, 8/k)`` rev/s: it
shrinks as 1/k. A comb displaced by a CONSTANT -0.42 rev/s therefore leaves the
window at k = 8/0.42 ~ 19, so "no line above k = 14 on DREGON" may be the window
falling off the line rather than the line being absent.

Pass 1  estimate a per-(window, rotor) scale s from the bar-clearing k = 2..13
        units, where the window is 0.6-1.5 rev/s wide and the lines are strong.
Pass 2  re-run the FULL sweep with the carrier at ``k * s * g_r(t)`` — the same
        band, the same half-width, the same collision gate, only re-centred —
        and re-run the half-integer null RE-CENTRED THE SAME WAY, so re-centring
        cannot manufacture a signal.

Reported per k band: median prominence on-comb and on the null, the count over
the 6 dB bar, and the residual offset after re-centring.
"""

from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import numpy as np  # noqa: E402

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT))
import measure_displacement as M  # noqa: E402
import nullcontrol as NC  # noqa: E402

SR = M.SR
N_ROTORS = 4
K_HI = 100
F_LIMIT = 7600.0  # audio Nyquist at 16 kHz, with margin
BAR_DB = 6.0
PASS1_K = list(range(2, 14))
BANDS = {
    "k2-13": (2, 13),
    "k14-25": (14, 25),
    "k26-40": (26, 40),
    "k41-60": (41, 60),
    "k61-100": (61, 100),
}

KEYS = list(NC.PARTNER)


def sweep(audio, ft, r_ft_true, rot, ks, scale, half):
    """(per-k rows) at carrier k*scale*g_rot (or (k+0.5)*scale*g_rot)."""
    row = r_ft_true[rot] * scale
    z_on, band_hz_k = NC.bank(audio, row, ft, ks, half)
    clean = ~NC.carrier_collision_mask(r_ft_true, row, rot, ks, half)
    out = []
    for a, k in enumerate(ks):
        tf, delta, snr, spec_db, rev_axis = M.ridge_from_envelope(
            z_on[:, a], float(band_hz_k[a]), k
        )
        keep_f = np.interp(tf, ft, clean[a].astype(float)) > 0.999
        w = np.where(keep_f, np.maximum(snr - 1.0, 0.0), 0.0)
        m, _, n_eff = M.weighted_stats(delta, w)
        prom, _ = NC.profile_prominence(spec_db, rev_axis, keep_f, k, float(band_hz_k[a]))
        out.append(
            {
                "k": k,
                "delta": None if not np.isfinite(m) else round(float(m), 5),
                "prom_db": None if not np.isfinite(prom) else round(float(prom), 3),
                "n_eff": round(float(n_eff), 2),
                "keep": round(float(np.mean(keep_f)), 3),
                "search_revs": round(NC.eff_search_revs(k, float(band_hz_k[a])), 4),
            }
        )
    return out


def run(key: str) -> dict[str, Any]:
    w = NC.load_window(key)
    audio, ft, r_ft = w["audio"], w["ft"], w["r_ft"]
    res: dict[str, Any] = {"key": key, "regime": w["regime"], "rotors": {}}
    for rot in range(N_ROTORS):
        r_bar = float(np.mean(r_ft[rot]))
        if r_bar < 20:
            continue
        # ── pass 1: scale from the bar-clearing low-k units ──────────────────
        p1 = sweep(audio, ft, r_ft, rot, PASS1_K, 1.0, False)
        good = [
            r
            for r in p1
            if r["prom_db"] is not None and r["prom_db"] >= BAR_DB and r["delta"] is not None
        ]
        if good:
            scale = 1.0 + float(np.mean([r["delta"] for r in good])) / r_bar
            src = f"{len(good)} bar-clearing k<=13 units"
        else:
            scale = 1.0
            src = "no bar-clearing low-k unit; scale forced to 1"
        k_max = min(K_HI, int(F_LIMIT / r_bar))
        ks = list(range(1, k_max + 1))
        res["rotors"][str(rot)] = {
            "rate": round(r_bar, 3),
            "scale": round(scale, 6),
            "scale_pct": round((scale - 1) * 100, 4),
            "scale_source": src,
            "k_max": k_max,
            "pass1": p1,
            "recentred_on": sweep(audio, ft, r_ft, rot, ks, scale, False),
            "recentred_null": sweep(audio, ft, r_ft, rot, ks, scale, True),
            "uncentred_on": sweep(audio, ft, r_ft, rot, ks, 1.0, False),
        }
    return res


def band_stats(rows, lo, hi, field="prom_db"):
    v = [r[field] for r in rows if lo <= r["k"] <= hi and r[field] is not None]
    return (
        (float(np.median(v)), int(sum(x >= BAR_DB for x in v)), len(v))
        if v
        else (float("nan"), 0, 0)
    )


def main() -> None:
    print(f"[recentre] {len(KEYS)} windows", flush=True)
    with ProcessPoolExecutor(max_workers=8) as pool:
        rows = list(pool.map(run, KEYS))
    (OUT / "recentre_raw.json").write_text(json.dumps(rows, indent=1))

    groups = {
        "dregon_cruise": lambda r: r["key"].startswith("free-flight") and r["regime"] == "cruise",
        "fly124_cruise": lambda r: r["key"].startswith("FLY124") and r["regime"] == "cruise",
    }
    summary: dict[str, Any] = {}
    for gname, pred in groups.items():
        sel = [r for r in rows if pred(r)]
        g: dict[str, Any] = {"n_windows": len(sel), "bands": {}}
        scales = [
            d["scale_pct"]
            for r in sel
            for d in r["rotors"].values()
            if "bar-clearing" in d["scale_source"]
        ]
        g["scale_pct_mean"] = round(float(np.mean(scales)), 4) if scales else None
        g["scale_pct_n"] = len(scales)
        for bname, (lo, hi) in BANDS.items():
            on_p, on_bar, on_n = [], 0, 0
            nl_p, nl_bar, nl_n = [], 0, 0
            un_p, un_bar, un_n = [], 0, 0
            res_d = []
            for r in sel:
                for d in r["rotors"].values():
                    for tag, acc_p, key in (
                        ("recentred_on", on_p, "on"),
                        ("recentred_null", nl_p, "null"),
                        ("uncentred_on", un_p, "unc"),
                    ):
                        v = [x for x in d[tag] if lo <= x["k"] <= hi and x["prom_db"] is not None]
                        acc_p.extend(x["prom_db"] for x in v)
                        if key == "on":
                            on_bar += sum(x["prom_db"] >= BAR_DB for x in v)
                            on_n += len(v)
                            res_d.extend(
                                x["delta"] / d["rate"] * 100
                                for x in v
                                if x["prom_db"] >= BAR_DB and x["delta"] is not None
                            )
                        elif key == "null":
                            nl_bar += sum(x["prom_db"] >= BAR_DB for x in v)
                            nl_n += len(v)
                        else:
                            un_bar += sum(x["prom_db"] >= BAR_DB for x in v)
                            un_n += len(v)
            g["bands"][bname] = {
                "recentred_on": {
                    "median_prom_db": round(float(np.median(on_p)), 3) if on_p else None,
                    "over_bar": on_bar,
                    "n": on_n,
                },
                "recentred_null": {
                    "median_prom_db": round(float(np.median(nl_p)), 3) if nl_p else None,
                    "over_bar": nl_bar,
                    "n": nl_n,
                },
                "uncentred_on": {
                    "median_prom_db": round(float(np.median(un_p)), 3) if un_p else None,
                    "over_bar": un_bar,
                    "n": un_n,
                },
                "residual_pct_of_rate_barclearing": round(float(np.mean(res_d)), 4)
                if res_d
                else None,
            }
        summary[gname] = g
        print(f"\n== {gname}  ({g['n_windows']} windows, scale {g['scale_pct_mean']} %)")
        print(
            f"{'band':10s} {'on med':>8s} {'null med':>9s} {'uncentr':>8s} "
            f"{'on>=6dB':>9s} {'null>=6':>8s} {'unc>=6':>7s} {'n':>5s}  resid%"
        )
        for bname, b in g["bands"].items():
            print(
                f"{bname:10s} {str(b['recentred_on']['median_prom_db']):>8s} "
                f"{str(b['recentred_null']['median_prom_db']):>9s} "
                f"{str(b['uncentred_on']['median_prom_db']):>8s} "
                f"{b['recentred_on']['over_bar']:>9d} {b['recentred_null']['over_bar']:>8d} "
                f"{b['uncentred_on']['over_bar']:>7d} {b['recentred_on']['n']:>5d}  "
                f"{b['residual_pct_of_rate_barclearing']}"
            )
    (OUT / "recentre.json").write_text(json.dumps(summary, indent=1))
    print(f"\n[recentre] wrote {OUT / 'recentre.json'}")


if __name__ == "__main__":
    main()
