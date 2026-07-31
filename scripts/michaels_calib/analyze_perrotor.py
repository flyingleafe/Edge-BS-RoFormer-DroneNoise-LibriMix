#!/usr/bin/env python3
"""Verdict tables for ``run_perrotor.py``: constancy, additive-vs-mult, offset-vs-lag.

Usage::

    python scripts/michaels_calib/analyze_perrotor.py [<raw dir>]

Four sections, all restricted to cruise windows and all flagging the twin pair
(rotors 1/3, LFront/RBack — near-equal speeds, which a previous study proved the
audio cannot resolve, so their per-rotor estimates are unreliable by default):

A  per-rotor offset per window -> per-rotor mean/sd vs the BETWEEN-rotor spread,
   plus additive / multiplicative / free-line / per-rotor-constant model rms.
B  the same offset re-minimised on frame subsets split by sign and magnitude of
   d(rps)/dt.  A calibration offset is FLAT in that split; a per-rotor lag tau
   makes the apparent offset ``-tau * rdot``, so it flips sign between the
   accelerating and decelerating halves.  The implied tau is printed.
C  the per-rotor lag scan: mean/sd of the optimal tau and the residual gain it
   buys over tau = 0.
D  the joint (tau_r, b_r) grid: residual at the origin, with b alone, with tau
   alone, and with both -> which parameter carries the explanatory power.
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np

RAW = Path(sys.argv[1] if len(sys.argv) > 1 else "results/michaels_perrotor/raw")
ROT = ["RFront", "LFront", "LBack", "RBack"]
TWIN = {1, 3}  # the near-equal-speed pair the audio cannot resolve


def parab(xs, ys):
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    i = int(np.argmin(ys))
    if i == 0 or i == len(xs) - 1:
        return float(xs[i]), True
    y0, y1, y2 = ys[i - 1], ys[i], ys[i + 1]
    den = y0 - 2 * y1 + y2
    if den <= 0:
        return float(xs[i]), True
    h = float(xs[i] - xs[i - 1])
    return float(xs[i] - 0.5 * h * (y2 - y0) / den), False


def load(stage):
    rows = []
    for p in sorted(glob.glob(str(RAW / f"*__{stage}_r*.json"))):
        rows.append(json.loads(Path(p).read_text()))
    return rows


def best_on(d, mask):
    """Best offset b restricted to the frames selected by `mask`."""
    num = np.asarray(d["num2"])[:, mask]
    ys = num.sum(axis=1)
    return parab(d["b_grid"], ys)


print("=" * 78)
print("A. PER-ROTOR OFFSET, ALL CRUISE WINDOWS (raw telemetry baseline)")
print("=" * 78)
off = load("off")
by = {}
for d in off:
    by.setdefault((d["rid"], d["rotor"]), []).append(d)
summary = {}
for rid in ("FLY124", "FLY125"):
    print(f"\n--- {rid}")
    print(
        f"{'rotor':<8}{'n':>3}{'mean_rps':>10}{'mean_b':>9}{'std_b':>8}"
        f"{'min_b':>8}{'max_b':>8}{'b/rps %':>9}  {'per-window b'}"
    )
    rows = []
    for r in range(4):
        ds = sorted(by.get((rid, r), []), key=lambda x: x["widx"])
        if not ds:
            continue
        bs = np.array([x["best_b"] for x in ds])
        mr = np.mean([x["mean_rps"] for x in ds])
        tw = " TWIN" if r in TWIN else ""
        rows.append((r, len(bs), mr, bs.mean(), bs.std(ddof=1) if len(bs) > 1 else np.nan))
        print(
            f"{ROT[r]:<8}{len(bs):>3}{mr:>10.2f}{bs.mean():>9.3f}"
            f"{(bs.std(ddof=1) if len(bs) > 1 else float('nan')):>8.3f}"
            f"{bs.min():>8.3f}{bs.max():>8.3f}{100 * bs.mean() / mr:>9.3f}  "
            + " ".join(f"{v:+.2f}" for v in bs)
            + tw
        )
    solo = [x for x in rows if x[0] not in TWIN]
    if len(solo) >= 2:
        spread = max(x[3] for x in solo) - min(x[3] for x in solo)
        scat = np.nanmean([x[4] for x in solo])
        print(
            f"  well-observed rotors: between-rotor spread {spread:.3f}, mean within-rotor sd {scat:.3f}"
        )
        print(f"  ratio spread/sd = {spread / scat:.2f}")
    # model comparison on per-window points (well-observed rotors only)
    pts = [(x["mean_rps"], x["best_b"], x["rotor"]) for x in off if x["rid"] == rid]
    solo_pts = [(a, b) for a, b, r in pts if r not in TWIN]
    if solo_pts:
        X = np.array([p[0] for p in solo_pts])
        Y = np.array([p[1] for p in solo_pts])
        add = Y.mean()
        rms_add = float(np.sqrt(np.mean((Y - add) ** 2)))
        g = float(np.dot(X, Y) / np.dot(X, X))
        rms_mul = float(np.sqrt(np.mean((Y - g * X) ** 2)))
        A = np.column_stack([X, np.ones(len(X))])
        c, *_ = np.linalg.lstsq(A, Y, rcond=None)
        rms_free = float(np.sqrt(np.mean((Y - A @ c) ** 2)))
        # per-rotor constant model
        pr = np.array([np.mean([b for a, b, rr in pts if rr == r]) for r in range(4)])
        rms_pr = float(np.sqrt(np.mean([(b - pr[rr]) ** 2 for a, b, rr in pts if rr not in TWIN])))
        print(
            f"  models on {len(solo_pts)} non-twin points: additive b={add:.3f} rms={rms_add:.3f} | "
            f"multiplicative g={g * 100:.3f}% rms={rms_mul:.3f} | free line slope={c[0]:.5f} "
            f"int={c[1]:.3f} rms={rms_free:.3f} | PER-ROTOR const rms={rms_pr:.3f}"
        )
    summary[rid] = rows

print()
print("=" * 78)
print("B. ACCELERATION SPLIT  (lag => apparent offset flips sign with sign(dr/dt))")
print("=" * 78)
for rid in ("FLY124", "FLY125"):
    print(f"\n--- {rid}")
    print(
        f"{'rotor':<8}{'b_all':>8}{'b_up':>8}{'b_dn':>8}{'b_up-b_dn':>11}"
        f"{'b_lo|a|':>9}{'b_hi|a|':>9}{'implied_tau_ms':>15}"
    )
    for r in range(4):
        ds = sorted(by.get((rid, r), []), key=lambda x: x["widx"])
        if not ds:
            continue
        acc = {"all": [], "up": [], "dn": [], "lo": [], "hi": [], "tau": []}
        for d in ds:
            g = np.asarray(d["drdt"])
            n = len(g)
            acc["all"].append(best_on(d, np.ones(n, bool))[0])
            up, dn = g > 0, g < 0
            bu = best_on(d, up)[0]
            bd = best_on(d, dn)[0]
            acc["up"].append(bu)
            acc["dn"].append(bd)
            q = np.quantile(np.abs(g), [1 / 3, 2 / 3])
            acc["lo"].append(best_on(d, np.abs(g) <= q[0])[0])
            acc["hi"].append(best_on(d, np.abs(g) >= q[1])[0])
            # b(t) = -tau * dr/dt  =>  tau = -(b_up - b_dn) / (mean g|up - mean g|dn)
            dg = g[up].mean() - g[dn].mean()
            acc["tau"].append(-(bu - bd) / dg if dg else np.nan)
        m = {k: float(np.mean(v)) for k, v in acc.items()}
        tw = "  TWIN" if r in TWIN else ""
        print(
            f"{ROT[r]:<8}{m['all']:>8.3f}{m['up']:>8.3f}{m['dn']:>8.3f}"
            f"{m['up'] - m['dn']:>11.3f}{m['lo']:>9.3f}{m['hi']:>9.3f}"
            f"{1000 * m['tau']:>15.1f}" + tw
        )

print()
print("=" * 78)
print("C. PER-ROTOR LAG SCAN")
print("=" * 78)
lag = load("lag")
byl = {}
for d in lag:
    byl.setdefault((d["rid"], d["rotor"]), []).append(d)
for rid in ("FLY124", "FLY125"):
    print(f"\n--- {rid}")
    print(
        f"{'rotor':<8}{'n':>3}{'mean_tau_ms':>13}{'sd':>8}{'gain_vs_tau0':>14}  per-window tau (ms)"
    )
    for r in range(4):
        ds = sorted(byl.get((rid, r), []), key=lambda x: x["widx"])
        if not ds:
            continue
        ta = np.array([x["best_tau_ms"] for x in ds])
        gain = np.mean([x["recon_at_zero"] - x["best_recon"] for x in ds])
        tw = "  TWIN" if r in TWIN else ""
        print(
            f"{ROT[r]:<8}{len(ta):>3}{ta.mean():>13.1f}"
            f"{(ta.std(ddof=1) if len(ta) > 1 else float('nan')):>8.1f}{gain:>14.5f}  "
            + " ".join(f"{v:+.0f}" for v in ta)
            + tw
        )

print()
print("=" * 78)
print("D. JOINT (tau_r, b_r): which axis carries the power")
print("=" * 78)
jt = load("joint")
byj = {}
for d in jt:
    byj.setdefault((d["rid"], d["rotor"]), []).append(d)
for rid in ("FLY124", "FLY125"):
    print(f"\n--- {rid}")
    print(
        f"{'rotor':<8}{'n':>3}{'origin':>9}{'b only':>9}{'tau only':>10}{'both':>9}"
        f"{'argmin_b':>10}{'argmin_tau_ms':>15}"
    )
    for r in range(4):
        ds = sorted(byj.get((rid, r), []), key=lambda x: x["widx"])
        if not ds:
            continue

        def col(k, ds=ds):
            return float(np.mean([x[k] for x in ds]))

        b_only = float(np.mean([min(x["recon"][x["tau_grid"].index(0.0)]) for x in ds]))
        t_only = float(
            np.mean([min(row[x["b_grid"].index(0.0)] for row in x["recon"]) for x in ds])
        )
        tw = "  TWIN" if r in TWIN else ""
        print(
            f"{ROT[r]:<8}{len(ds):>3}{col('recon_at_origin'):>9.4f}{b_only:>9.4f}"
            f"{t_only:>10.4f}{col('min_recon'):>9.4f}{col('argmin_b'):>10.2f}"
            f"{col('argmin_tau_ms'):>15.1f}" + tw
        )
