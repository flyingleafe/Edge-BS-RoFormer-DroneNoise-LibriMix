#!/usr/bin/env python3
"""Build the DEMODULATED-ENVELOPE CACHE.

The envelope is the expensive, stable product; every figure choice (segment
length, harmonic set, axis range, colour limits, reference lines, selection
statistic) is downstream of it and costs milliseconds once it is on disk.
Caching the ENVELOPE rather than a spectrogram is deliberate: it keeps the
freedom to re-window at ANY segment length, which is the parameter that decides
whether a high-k line is visible at all.

For one recording window, one rotor and every harmonic k = 1..K_MAX, the 8-mic
audio is heterodyned by exp(-i k phi_r(t)) with

    phi_r(t) = 2 pi \\int g_r dt ,   g_r = DREGON `motors_measured` (rev/s),

then band-limited and decimated by DECIM to an envelope rate of
44100 / DECIM Hz.  Microphone channels are KEPT SEPARATE.

In the cached envelope the TELEMETRY rate is exactly DC, so an envelope
frequency f maps to an acoustic shaft-rate offset delta = f / k rev/s.

Layout::

    cache/manifest.json
    cache/<recording>__w<ww>__r<rotor>.npz
        z        complex64 (K, C, n_env)   demodulated envelopes, mic axis intact
        ks       int32     (K,)            harmonic orders, 1..K_MAX
        g        float64   (n_env,)        motors_measured of THIS rotor, env grid
        g_all    float64   (4, n_env)      all four rotors, env grid
        t_env    float64   (n_env,)        seconds from the window start
        scalars  fs_env, decim, sr_audio, t_start, dur, rotor, band_hz

Run: ``python hk_cache.py``  (about 2 minutes on 7 cores).
"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor

import hk_core as H
import numpy as np
from scipy.signal import resample_poly

CACHE = H.OUT / "cache"
K_MAX = 100
DECIM = 100  # 44100 -> 441 Hz envelope rate; keeps +-220.5 Hz
DUR = 16.0
RPS_CHANNEL = "motors_measured"
CODE_VERSION = "hk_cache.py 2026-08-05 v2 (+half-integer null)"
# The half-integer comb k+0.5 is the NULL: no rotor line can live there, so the
# same statistic computed on it is the matched false-alarm reference.  Cached
# for the primary recording only, to keep the cache under ~750 MB.
HALF_RECORDINGS = ("free-flight_nosource_room1",)

WINDOWS = [
    ("free-flight_nosource_room1", 1, 22.56481),
    ("free-flight_speech-low_room1", 1, 22.641476),
    ("free-flight_whitenoise-low_room1", 1, 23.537204),
]


def build(args):
    rid, widx, t0, rot, half = args
    audio, sr, g, rates = H.load_raw(rid, t0, DUR)
    phi = H.phase(g[rot], sr)
    n_env = int(np.ceil(audio.shape[1] / DECIM))
    ks = np.arange(1, K_MAX + 1, dtype=np.float64) + (0.5 if half else 0.0)
    z = np.empty((len(ks), audio.shape[0], n_env), np.complex64)
    for a, k in enumerate(ks):
        carrier = np.exp(-1j * float(k) * phi)
        for c in range(audio.shape[0]):
            d = resample_poly(audio[c] * carrier, 1, DECIM)
            z[a, c, : len(d)] = d.astype(np.complex64)
    t_env = np.arange(n_env) * DECIM / sr
    t_full = np.arange(audio.shape[1]) / sr
    g_all = np.stack([np.interp(t_env, t_full, g[r]) for r in range(4)])
    suf = "__half" if half else ""
    out = CACHE / f"{rid}__w{widx:02d}__r{rot}{suf}.npz"
    np.savez(
        out,
        z=z,
        ks=ks,
        g=g_all[rot],
        g_all=g_all,
        t_env=t_env,
        fs_env=np.float64(sr / DECIM),
        decim=np.int32(DECIM),
        sr_audio=np.int32(sr),
        t_start=np.float64(t0),
        dur=np.float64(DUR),
        rotor=np.int32(rot),
        band_hz=np.float64(sr / DECIM / 2.0),
        rates_mean=rates,
    )
    return {
        "file": out.name,
        "recording": rid,
        "window": widx,
        "rotor": rot,
        "half": half,
        "t_start": t0,
        "shape": list(z.shape),
        "rate_mean": round(float(rates[rot]), 4),
        "bytes": out.stat().st_size,
    }


def main() -> None:
    CACHE.mkdir(exist_ok=True)
    jobs = [(rid, w, t0, r, False) for rid, w, t0 in WINDOWS for r in range(4)]
    jobs += [
        (rid, w, t0, r, True) for rid, w, t0 in WINDOWS if rid in HALF_RECORDINGS for r in range(4)
    ]
    print(f"[cache] {len(jobs)} (window, rotor) units, k = 1..{K_MAX}", flush=True)
    entries = []
    with ProcessPoolExecutor(max_workers=6) as pool:
        for e in pool.map(build, jobs):
            entries.append(e)
            print(f"  {e['file']}  {e['shape']}  {e['bytes'] / 1e6:.1f} MB", flush=True)
    total = sum(e["bytes"] for e in entries)
    man = {
        "code_version": CODE_VERSION,
        "created": "2026-08-05",
        "what": "complex demodulated envelopes of DREGON rotor harmonics",
        "rps_channel": RPS_CHANNEL,
        "rps_note": "carrier phase = 2*pi*cumsum(motors_measured)/sr on the audio grid; "
        "telemetry rate is DC in the envelope, so envelope Hz / k = rev/s offset",
        "k_min": 1,
        "k_max": K_MAX,
        "half_integer_null": {
            "recordings": list(HALF_RECORDINGS),
            "files": "<recording>__w<ww>__r<rotor>__half.npz",
            "ks": f"k + 0.5 for k = 1..{K_MAX}; no rotor line can exist there, so "
            "the same statistic on these envelopes is the matched null",
        },
        "decim": DECIM,
        "fs_env_hz": 44100 / DECIM,
        "band_hz": 44100 / DECIM / 2.0,
        "band_note": "usable |offset| <= 220.5/k rev/s (>= 2.2 rev/s for k <= 100)",
        "dur_s": DUR,
        "dtype": "complex64",
        "axes": "z[k_index, mic, env_sample]; mic axis is NOT collapsed",
        "windows": [
            {"recording": r, "window": w, "t_start": t0, "regime": "cruise"} for r, w, t0 in WINDOWS
        ],
        "entries": entries,
        "total_bytes": total,
        "total_mb": round(total / 1e6, 1),
    }
    (CACHE / "manifest.json").write_text(json.dumps(man, indent=1))
    print(f"[cache] wrote {CACHE}/manifest.json - {total / 1e6:.0f} MB total")


if __name__ == "__main__":
    main()
