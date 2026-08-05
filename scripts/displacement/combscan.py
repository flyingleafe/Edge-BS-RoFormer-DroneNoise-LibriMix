#!/usr/bin/env python3
"""Comb-scale scan in ORDER space — no peak-search window anywhere.

The question: does the acoustic rotor comb sit on the telemetry, or beside it?
This driver answers it in the ORDER domain. It resamples each window uniformly
in the telemetry rotor phase and scores a whole comb at once, S(s) = the mean
excess in dB at the orders s*k of the band. A displaced comb moves the peak of
S off s = 1. The score is a mean over many harmonics, so the height of its peak
over its own background IS the significance, and no per-harmonic search window
exists anywhere to bias the answer.

The null is the same scan on a HALF-INTEGER comb, at the orders s*(k + 0.5),
where no rotor line can exist. Both scans are identical searches, thus max(on)
against max(null) is a fair contest.

``--seg-s`` selects the SHORT-segment scan, which is the reading to trust above
k ~ 40: the high-k comb decoheres in less than a second, so a 16 s spectrum
averages it away. The algorithm is `tracking.order_domain`; this file only
loads the data, fans the units out and writes JSON.

Usage::

    python scripts/displacement/combscan.py --jobs 8
    python scripts/displacement/combscan.py --seg-s 0.25 --bands k76-110:76-110
    python scripts/displacement/combscan.py --peaks --rotors 0 \\
        --window nosource_w01:free-flight_nosource_room1:22.565:measured
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # this checkout (code)
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))

from utils.gridrun import Unit, add_gridrun_args, gridrun_from_args  # noqa: E402

#: (label, recording, t0, telemetry entry) — the frozen DREGON cruise windows.
DREGON_WINDOWS = [
    ("nosource_w01", "free-flight_nosource_room1", 22.565, "measured"),
    ("speech-low_w01", "free-flight_speech-low_room1", 26.0, "measured"),
    ("whitenoise-low_w01", "free-flight_whitenoise-low_room1", 28.0, "measured"),
    ("nosource_room2_w01", "free-flight_nosource_room2", 24.0, "command"),
    ("hovering_room2_w00", "hovering_nosource_room2", 8.0, "command"),
    ("updown_room2_w00", "updown_nosource_room2", 7.0, "command"),
    ("rectangle_room2_w00", "rectangle_nosource_room2", 9.0, "command"),
    ("spinning_room2_w00", "spinning_nosource_room2", 10.0, "command"),
]
BANDS = {"k2-13": (2, 13), "k14-40": (14, 40), "k41-75": (41, 75), "k76-110": (76, 110)}
WHOLE_GRID = (0.975, 1.0251, 2e-5)  # numpy arange bounds, whole-window scan
SEG_GRID = (0.985, 1.01501, 5e-5)  # numpy arange bounds, short-segment scan
MIN_RATE = 30.0  # rev/s: a slower rotor idles, so it carries no cruise comb
F_LIMIT_FRAC = 0.45  # a harmonic above this fraction of sr is beyond Nyquist
PEAK_K_MAX = 120


def _worker(unit: Unit) -> dict:
    """One (window, rotor): the band scans, and the peak fan when asked."""
    import hk_core
    import numpy as np

    from tracking import order_domain as od

    p = unit.params
    channel = {"measured": hk_core.MEASURED, "command": hk_core.COMMAND}[p["entry"]]
    audio, sr, g, _ = hk_core.load_raw(p["rid"], p["t0"], p["dur"], channel=channel)
    rot, dur, seg_s = int(p["rotor"]), float(p["dur"]), float(p["seg_s"])
    rate = g[rot]
    # ``g`` is already on the audio grid of the slice, so the telemetry time
    # base is the slice's own sample grid and the slice starts at t = 0.
    t_tel = np.arange(audio.shape[1]) / sr
    s_grid = np.arange(*p["s_grid"])
    r_bar = float(rate.mean())
    rec = {"window": p["window"], "rotor": rot, "rate": round(r_bar, 3), "entry": p["entry"]}
    rec["bands"] = out = {}
    if r_bar < MIN_RATE:
        rec["skipped"] = f"mean rate below {MIN_RATE} rev/s"
        return rec
    spec = None
    if seg_s <= 0.0 or p["peaks"]:
        spec = od.order_spectrum(audio, sr, t_tel, rate, 0.0, dur)
    for name, (lo, hi) in p["bands"]:
        if seg_s > 0.0:
            args = (audio, sr, t_tel, rate, 0.0, dur, s_grid, lo, hi)
            kw = {"seg_s": seg_s, "f_limit_frac": F_LIMIT_FRAC}
            on, n_used = od.segment_comb_scan(*args, half=False, **kw)  # type: ignore[arg-type]
            off, _ = od.segment_comb_scan(*args, half=True, **kw)  # type: ignore[arg-type]
            n_harm = sum(k * r_bar < F_LIMIT_FRAC * sr for k in range(lo, hi + 1))
            extra = {"n_harmonics": int(n_harm), "n_segments": int(n_used)}
        elif spec is not None:  # seg_s <= 0 always builds the order spectrum
            if hi * r_bar > F_LIMIT_FRAC * sr:
                continue  # the whole band sits beyond the audio Nyquist
            on, count = od.comb_scan(spec.orders, spec.excess_db, s_grid, lo, hi)
            off, _ = od.comb_scan(spec.orders, spec.excess_db, s_grid, lo, hi, half=True)
            extra = {"n_harmonics": int(count.max())}
        out[name] = {
            "on": od.scan_summary(s_grid, on, null=off),
            "null_half_integer": od.scan_summary(s_grid, off),
            **extra,
        }
    if p["peaks"] and spec is not None:
        rec["peaks"] = [
            {"k": k, "order": round(o, 4), "prom_db": round(pr, 2), "ratio": round(ra, 5)}
            for k, o, pr, ra in od.peak_orders(spec.orders, spec.db, 1, PEAK_K_MAX)
        ]
    return rec


def _summarize(rows: list[dict]) -> dict:
    """Per window-rotor-band: the peak of the on-comb scan, and its margin."""
    return {
        f"{r['window']}__r{r['rotor']}__{name}": {
            "pct": b["on"]["pct"],
            "on_minus_null_db": b["on"]["peak_over_null_db"],
            "n_harmonics": b["n_harmonics"],
        }
        for r in rows
        for name, b in r.get("bands", {}).items()
    }


def _parse_window(spec: str) -> tuple[str, str, float, str]:
    """``LABEL:RID:T0[:CHANNEL]``, where the channel defaults to ``measured``."""
    label, rid, t0, *rest = spec.split(":")
    return label, rid, float(t0), rest[0] if rest else "measured"


def _parse_band(spec: str) -> tuple[str, tuple[int, int]]:
    """``NAME:LO-HI``, the harmonic range of one band."""
    name, span = spec.split(":")
    lo, hi = span.split("-")
    return name, (int(lo), int(hi))


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--window", action="append", metavar="LABEL:RID:T0[:CHANNEL]")
    ap.add_argument("--dur", type=float, default=16.0, help="window length, s")
    ap.add_argument("--rotors", default="0,1,2,3")
    ap.add_argument("--bands", metavar="NAME:LO-HI,...")
    ap.add_argument("--seg-s", type=float, default=0.0, help="0 = whole-window scan")
    ap.add_argument("--s-grid", metavar="LO,HI,STEP", help="numpy arange bounds of the scale grid")
    ap.add_argument("--peaks", action="store_true", help="add the per-harmonic peak fan")
    ap.add_argument("--out", default="results/displacement/combscan")
    add_gridrun_args(ap)
    args = ap.parse_args()

    windows = [_parse_window(w) for w in args.window] if args.window else DREGON_WINDOWS
    bands = [_parse_band(b) for b in args.bands.split(",")] if args.bands else list(BANDS.items())
    grid = WHOLE_GRID if args.seg_s <= 0.0 else SEG_GRID
    if args.s_grid:
        grid = tuple(float(v) for v in args.s_grid.split(","))
    base = {
        "dur": args.dur,
        "bands": bands,
        "seg_s": args.seg_s,
        "s_grid": grid,
        "peaks": bool(args.peaks),
    }
    units = [
        Unit(
            uid=f"{label}__r{rot}",
            params={"window": label, "rid": rid, "t0": t0, "entry": entry, "rotor": rot, **base},
        )
        for label, rid, t0, entry in windows
        for rot in (int(r) for r in args.rotors.split(","))
    ]
    result = gridrun_from_args(args, units, _worker, args.out, summarize=_summarize)
    raise SystemExit(result.exit_code)


if __name__ == "__main__":
    main()
