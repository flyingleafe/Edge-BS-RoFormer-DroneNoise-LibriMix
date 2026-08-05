#!/usr/bin/env python3
"""Regenerate only the delta_k(t) traces (task 3), reusing nullcontrol.json.

Separate from ``nullcontrol.py`` so the trace convention can be iterated on
without paying for the 3-variant measurement pass again.
"""

from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "2")

import numpy as np  # noqa: E402

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT))

import nullcontrol as NC  # noqa: E402

NULLJSON = json.loads((OUT / "nullcontrol.json").read_text())["windows"]


def one(key: str) -> str:
    w = NC.load_window(key)
    ks = list(range(1, NC.K_MAX + 1))
    rot = max(
        range(NC.N_ROTORS),
        key=lambda r: float(
            np.median(
                [
                    NULLJSON[key]["rotors"][str(r)]["on"]["per_k"][str(k)][4] or -99.0
                    for k in range(2, NC.K_MAX + 1)
                ]
            )
        ),
    )
    z_on, band_hz_k = NC.bank(w["audio"], w["r_ft"][rot], w["ft"], ks, half=False)
    clean = ~NC.carrier_collision_mask(w["r_ft"], w["r_ft"][rot], rot, ks, half=False)
    clean_off = ~NC.carrier_collision_mask(w["r_ft"], w["r_ft"][rot], rot, ks, half=True)
    tr = NC.trace_variant(z_on, band_hz_k, ks, clean, w["ft"])
    z_off, band_off = NC.bank(w["audio"], w["r_ft"][rot], w["ft"], ks, half=True)
    tr_off = NC.trace_variant(z_off, band_off, ks, clean_off, w["ft"])
    payload: dict[str, Any] = {
        "rotor": np.array(rot),
        "band_hz_k": band_hz_k,
        "ft": w["ft"],
        "r_ft": w["r_ft"],
        "clean_frac": clean.mean(axis=1),
    }
    payload.update({f"on__{a}": b for a, b in tr.items()})
    payload.update({f"off__{a}": b for a, b in tr_off.items()})
    np.savez_compressed(NC.TRACE_DIR / f"{key}.npz", **payload)
    n_ok = int(np.isfinite(tr["d_peak"]).sum())
    return f"{key}: rotor {rot}, {n_ok} finite trace points"


if __name__ == "__main__":
    NC.TRACE_DIR.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=5) as pool:
        for line in pool.map(one, list(NC.TRACE_WINDOWS)):
            print(line)
