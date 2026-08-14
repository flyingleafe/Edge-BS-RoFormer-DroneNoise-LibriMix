#!/usr/bin/env python3
"""Cheap spectrogram triage of candidate recordings for blind annotation.

Answers the three questions that decide whether a recording is worth a full
blind-ladder run, WITHOUT running the ladder (no VK solve anywhere here):

* **Is a comb visible?** A coarse comb scan over a base-rate grid: the mean
  whitened log-magnitude sampled at ``k * base``. The peak of that curve
  against its own median is the comb prominence in dB — the same "line
  evidence above the local background" the blind seed's first pass reads,
  scored on one slice instead of the whole search.
* **What is the f0 range?** The argmax base of that scan on each slice, and
  the spread of it across slices.
* **Is it stationary?** The scan is run per slice; a hover holds its argmax,
  a maneuver or a takeoff does not.

It also reports the OCTAVE ratio ``v(b)/v(2b)`` that ``blind_fullrange``'s
coarse pass thresholds at 1.4 — so a recording that will be halved is visible
before a single window is annotated.

Reads the window cache a :mod:`scripts.blind_corpus` run already wrote
(``<run>/windows/*.npz``), so it costs one FFT per slice and no download.

Run:
  PYTHONPATH=src python scripts/blind_corpus_triage.py \
      --windows results/blind_corpus/avq_mono/windows --out results/blind_corpus/triage_avq.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent if (_HERE.parent / "src").is_dir() else Path.cwd().resolve()
sys.path.insert(0, str(_ROOT / "src"))

SR = 16000
K_SCAN = 24  # teeth summed per candidate base
F_MIN, F_MAX = 40.0, 4000.0  # band the teeth are read in


def comb_curve(
    white: np.ndarray, bin_hz: float, bases: np.ndarray, k_scan: int = K_SCAN
) -> np.ndarray:
    """Mean whitened value at ``k * base`` for every base — the scan curve.

    ``white`` is ``(F,)``: one frequency profile (a time average of the
    whitened log-magnitude). Teeth outside ``[F_MIN, F_MAX]`` do not count, and
    a base with fewer than four admissible teeth scores ``-inf`` rather than
    winning on a short sum.
    """
    n_f = len(white)
    f_max_bin = (n_f - 1) * bin_hz
    ks = np.arange(1, k_scan + 1)
    out = np.full(len(bases), -np.inf)
    for i, b in enumerate(bases):
        f = ks * b
        ok = (f >= F_MIN) & (f <= min(F_MAX, f_max_bin))
        if ok.sum() < 4:
            continue
        idx = f[ok] / bin_hz
        j = np.clip(np.floor(idx).astype(int), 0, n_f - 2)
        frac = idx - j
        out[i] = float(np.mean(white[j] * (1 - frac) + white[j + 1] * frac))
    return out


def triage_slice(audio: np.ndarray, n_slices: int = 4) -> dict[str, Any]:
    """Per-slice comb scan of one cached window."""
    from tracking.vk_blind_seeding import whitened_logmag

    white, bin_hz, _st = whitened_logmag(audio.astype(np.float32), float(SR))
    # Two grids, on purpose. The f0 argmax is searched only over the range
    # the ladder's own coarse pass searches (12-120 rev/s): a wider grid lets
    # a sparse high base win on fewer, luckier teeth, which is the scan bias
    # the real blind seed spends its octave logic on. The octave ratio needs
    # to reach TWICE the fastest rotor, so it reads the wide grid.
    bases = np.arange(15.0, 125.0, 0.25)
    bases_wide = np.arange(15.0, 260.0, 0.25)
    n_t = white.shape[1]
    edges = np.linspace(0, n_t, n_slices + 1).astype(int)

    peaks, proms = [], []
    for a, b in zip(edges[:-1], edges[1:], strict=False):
        if b - a < 4:
            continue
        curve = comb_curve(white[:, a:b].mean(axis=1), bin_hz, bases)
        live = np.isfinite(curve)
        if live.sum() < 8:
            continue
        pk = int(np.argmax(np.where(live, curve, -np.inf)))
        peaks.append(float(bases[pk]))
        proms.append(float(curve[pk] - np.median(curve[live])))

    if not peaks:
        return {"comb_prominence_db": None}

    # The octave ratio blind_fullrange thresholds at 1.4: strength at the
    # winning base against strength at twice it.
    prof = white.mean(axis=1)
    curve_all = comb_curve(prof, bin_hz, bases)
    live = np.isfinite(curve_all)
    b0 = float(bases[int(np.argmax(np.where(live, curve_all, -np.inf)))])
    wide = comb_curve(prof, bin_hz, bases_wide)
    lw = np.isfinite(wide)
    v_b = float(np.interp(b0, bases_wide[lw], wide[lw]))
    v_2b = float(np.interp(2 * b0, bases_wide[lw], wide[lw])) if 2 * b0 <= bases_wide[-1] else np.nan

    return {
        "comb_prominence_db": round(float(np.mean(proms)), 3),
        "f0_peak_rev_s": round(float(np.median(peaks)), 2),
        "f0_slice_spread_rev_s": round(float(np.max(peaks) - np.min(peaks)), 2),
        "f0_slice_std_rev_s": round(float(np.std(peaks)), 3),
        "octave_ratio_vb_v2b": (round(v_b / v_2b, 3) if np.isfinite(v_2b) and v_2b else None),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    ap.add_argument("--windows", required=True, help="a blind_corpus run's windows/ dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-recording", type=int, default=2, help="windows sampled per recording")
    ap.add_argument("--channel", type=int, default=0, help="mic used for the triage")
    args = ap.parse_args()

    wdir = Path(args.windows)
    by_rec: dict[str, list[Path]] = defaultdict(list)
    for p in sorted(wdir.glob("*.npz")):
        by_rec[p.stem.rsplit("__w", 1)[0]].append(p)

    rows = []
    for rid, paths in sorted(by_rec.items()):
        # Sample evenly through the recording so a maneuver is not missed.
        pick = [
            paths[i]
            for i in np.linspace(0, len(paths) - 1, min(args.per_recording, len(paths))).astype(int)
        ]
        per = []
        for p in pick:
            a = np.asarray(np.load(p)["audio"], dtype=np.float64)
            a = a[args.channel : args.channel + 1] if a.ndim == 2 else a[None, :]
            per.append({"window": p.stem, **triage_slice(a)})
        got = [d for d in per if d.get("comb_prominence_db") is not None]
        rows.append(
            {
                "recording_id": rid,
                "n_windows_total": len(paths),
                "n_sampled": len(pick),
                "comb_prominence_db": (
                    round(float(np.mean([d["comb_prominence_db"] for d in got])), 3)
                    if got
                    else None
                ),
                "f0_peak_rev_s": (
                    round(float(np.median([d["f0_peak_rev_s"] for d in got])), 2) if got else None
                ),
                "f0_within_window_spread": (
                    round(float(np.mean([d["f0_slice_spread_rev_s"] for d in got])), 2)
                    if got
                    else None
                ),
                "f0_across_window_spread": (
                    round(float(np.ptp([d["f0_peak_rev_s"] for d in got])), 2)
                    if len(got) > 1
                    else None
                ),
                "octave_ratio_vb_v2b": (
                    round(
                        float(
                            np.median(
                                [
                                    d["octave_ratio_vb_v2b"]
                                    for d in got
                                    if d.get("octave_ratio_vb_v2b")
                                ]
                            )
                        ),
                        3,
                    )
                    if any(d.get("octave_ratio_vb_v2b") for d in got)
                    else None
                ),
                "per_window": per,
            }
        )
        r = rows[-1]
        print(
            f"{rid:<28} prom={r['comb_prominence_db']} dB  f0={r['f0_peak_rev_s']} rev/s  "
            f"spread(in/across)={r['f0_within_window_spread']}/{r['f0_across_window_spread']}  "
            f"v(b)/v(2b)={r['octave_ratio_vb_v2b']}",
            flush=True,
        )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rows, indent=1))
    print(f"\nWritten to {args.out}")


if __name__ == "__main__":
    main()
