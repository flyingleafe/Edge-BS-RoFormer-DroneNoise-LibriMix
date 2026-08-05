#!/usr/bin/env python3
"""Null controls for the "displaced comb" measurement + prominence map + wiggle traces.

Three questions, one pass over the same 15 windows of ``measure_displacement.py``:

1. NULL CONTROL (task 1). Is the high-k offset a real measurement or a
   peak-search-window artifact? Three nulls, each run through the *identical*
   pipeline (same band, same search half-width in Hz, same collision gate, same
   weighted combination over k):
     - ``off``: heterodyne at ``(k + 0.5) * g_r(t)`` — the deliberately
       non-harmonic rate. Same trajectory, same band, no rotor line.
     - ``mis``: heterodyne at ``k * g_partner(t)`` where the telemetry comes
       from a different window/recording — real spectra, broken correspondence.
     - analytic: the offset a peak drawn uniformly in the search window would
       give (``W_revs / 4``).
   Plus a window-INDEPENDENT estimator (``pulse-pair`` / phase increment) run on
   both the on-comb and the off-comb envelopes.

2. PROMINENCE MAP (task 2). Per dataset x recording x rotor x k, the ridge
   height in dB over the in-band floor of the time-averaged envelope profile,
   for on-comb and off-comb — so "how far above the null does this harmonic
   actually sit".

3. WIGGLE TRACES (task 3). For the selected windows, ``delta_k(t)`` on a common
   2 s / 0.25 s time base for every k, both peak-pick and sliding pulse-pair,
   so the cross-k coherence of the DREGON k = 2 wiggle can be tested.
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

TRACE_DIR = OUT / "traces"
SR = M.SR
STRIDE = M.STRIDE
FS_ENV = M.FS_ENV
N_ROTORS = M.N_ROTORS
K_MAX = M.K_MAX
LOW_K = M.LOW_K
HIGH_K = M.HIGH_K

# wiggle / display STFT: one common time base for every k
DISP_SEG_S = 2.0
DISP_HOP_S = 0.25

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

TRACE_WINDOWS = (
    "free-flight_nosource_room1__w01",
    "free-flight_speech-low_room1__w01",
    "free-flight_whitenoise-low_room1__w01",
    "FLY124__w03",
    "FLY124__w04",
)


def load_window(key: str) -> dict[str, Any]:
    with np.load(M.PREP / f"{key}.npz") as z:
        return {
            "audio": np.asarray(z["audio"], np.float64),
            "ft": np.asarray(z["ft"], np.float64),
            "r_ft": np.asarray(z["r_meas"], np.float64),
            "regime": str(z["regime"]),
        }


def bank(
    audio: np.ndarray, r_row_ft: np.ndarray, ft: np.ndarray, ks: list[int], half: bool
) -> tuple[np.ndarray, np.ndarray]:
    """Demod bank around ``k * g(t)`` (``half=False``) or ``(k + 0.5) * g(t)``.

    The half-integer carrier reuses the tracker's own integer recursion by
    halving the phase and asking for ``2k + 1``:
    ``exp(-i (2k+1) phi/2) = exp(-i (k+1/2) phi)``. Band and decimation are
    identical to the on-comb call, so the two differ only in carrier rate.
    """
    from tracking.phase_increment_tracker import _demod_bank

    n_t = audio.shape[-1]
    t_aud = np.arange(n_t) / SR
    r_aud = np.interp(t_aud, ft, r_row_ft)
    phi = 2.0 * np.pi * np.cumsum(r_aud) / SR
    r_mean = float(np.mean(r_row_ft))
    band_hz_k = np.array(
        [min(M.B0_REVS * k, M.BAND_FRAC_OF_RATE * r_mean) for k in ks], dtype=np.float64
    )
    n_env = n_t // STRIDE
    y32 = np.asarray(audio, dtype=np.float32)
    phi_use = phi / 2.0 if half else phi
    ks_use = [2 * k + 1 for k in ks] if half else list(ks)
    z_on, _ = _demod_bank(
        y32,
        phi_use,
        t_aud,
        ks_use,
        0.0,
        STRIDE,
        n_env,
        float(np.max(band_hz_k)) / SR,
        band_cyc_k=band_hz_k / SR,
    )
    return z_on, band_hz_k


def eff_search_revs(k: int, band_hz: float) -> float:
    """Effective peak-search HALF width in rev/s (what the code really uses)."""
    return min(M.search_hz(k), 0.9 * band_hz) / k


def pulse_pair(z_k: np.ndarray, k: int, keep_env: np.ndarray) -> tuple[float, float]:
    """Coherent phase-increment (pulse-pair) offset in rev/s + its coherence.

    ``arg(sum_n sum_c z[c,n] conj(z[c,n-1])) / (2 pi k dt_env)``. The lag
    product's phase is the envelope frequency and carries no static per-mic
    phase, so the channel sum is coherent and legitimate. Unambiguous over
    ``|delta| < fs_env / (2k)`` rev/s — far outside the demod band, hence NOT
    confined by the peak-search window. Coherence = ``|sum| / sum|.|``.
    """
    lag = z_k[:, 1:] * np.conj(z_k[:, :-1])
    m = keep_env[1:] & keep_env[:-1]
    if m.sum() < 8:
        return float("nan"), 0.0
    v = lag[:, m]
    s = complex(v.sum())
    denom = float(np.abs(v).sum())
    dt = 1.0 / FS_ENV
    return float(np.angle(s) / (2.0 * np.pi * k * dt)), (abs(s) / denom if denom > 0 else 0.0)


def profile_prominence(
    spec_db: np.ndarray, rev_axis: np.ndarray, keep_f: np.ndarray, k: int, band_hz: float
) -> tuple[float, float]:
    """``(prominence_db, peak_offset_rev_s)`` of the time-averaged profile.

    Power spectra of the uncollided frames are averaged, normalized by the
    in-band median, smoothed to ~0.05 rev/s (so one noisy bin cannot claim the
    line) and peak-picked inside the search window.
    """
    spec = np.power(10.0, spec_db.T / 10.0)  # (T, F)
    src = spec[keep_f] if keep_f.sum() >= 3 else spec
    prof = src.mean(axis=0)
    prof_db = 10.0 * np.log10(prof / np.median(prof) + 1e-300)
    step = float(rev_axis[1] - rev_axis[0])
    n_sm = max(3, int(round(0.05 / step)) | 1)
    kern = np.hanning(n_sm)
    prof_sm = np.convolve(prof_db, kern / kern.sum(), mode="same")
    sw = np.abs(rev_axis) <= eff_search_revs(k, band_hz)
    if not sw.any():
        return float("nan"), float("nan")
    j = int(np.argmax(prof_sm[sw]))
    return float(prof_sm[sw][j]), float(rev_axis[sw][j])


def carrier_collision_mask(
    r_ft_true: np.ndarray,
    r_row_carrier: np.ndarray,
    rot: int,
    ks: list[int],
    half: bool,
    f_max: float = 6000.0,
    min_rate: float = 5.0,
) -> np.ndarray:
    """``(K, N)`` bool: another rotor's REAL harmonic inside harmonic k's search
    window, for an arbitrary carrier.

    The tracker's twin rule (``_twin_collision_mask``) assumes the carrier is
    ``k * r_i`` of a rotor that is itself in ``r_ft``. The nulls break that: the
    off-comb carrier is ``(k + 0.5) r_i`` and the mismatched carrier is
    ``k g_partner``, while the interferers are always the AUDIO's real rotor
    lines. Gating a null against fictional lines (or failing to gate it against
    the real ones) would make the null catch interference the measurement is
    protected from — so the rule is re-derived here against ``r_ft_true``, with
    rotor ``rot`` skipped exactly as the tracker skips its own rotor.
    """
    kf = np.asarray(ks, dtype=np.float64) + (0.5 if half else 0.0)
    sep = np.array([M.COLLISION_GUARD * M.search_hz(k) for k in ks], dtype=np.float64)[:, None]
    fi = kf[:, None] * r_row_carrier[None, :]
    coll = np.zeros(fi.shape, dtype=bool)
    for j in range(r_ft_true.shape[0]):
        if j == rot or float(np.mean(r_ft_true[j])) < min_rate:
            continue
        rj = np.maximum(r_ft_true[j], 1e-3)[None, :]
        base = fi / rj
        for kp in (np.floor(base), np.ceil(base)):
            fj = np.maximum(kp, 1.0) * rj
            coll |= (np.abs(fj - fi) < sep) & (fj <= f_max + sep)
    return coll


def measure_variant(
    audio: np.ndarray,
    ft: np.ndarray,
    r_ft_carrier: np.ndarray,
    r_ft_true: np.ndarray,
    rot: int,
    ks: list[int],
    half: bool,
) -> dict[str, Any]:
    """One (window, rotor, variant) pass: per-k offsets / SNR / prominence /
    pulse-pair, plus the low-k / high-k combined-series MAE."""
    z_on, band_hz_k = bank(audio, r_ft_carrier[rot], ft, ks, half)
    clean = ~carrier_collision_mask(r_ft_true, r_ft_carrier[rot], rot, ks, half)
    n_env = z_on.shape[-1]
    t_env = (np.arange(n_env) + 0.5) * STRIDE / SR
    d_grid = np.full((len(ks), ft.size), np.nan)
    w_grid = np.zeros((len(ks), ft.size))
    per_k: dict[str, list[float | None]] = {}
    for a, k in enumerate(ks):
        tf, delta, snr, spec_db, rev_axis = M.ridge_from_envelope(
            z_on[:, a], float(band_hz_k[a]), k
        )
        keep_f = np.interp(tf, ft, clean[a].astype(float)) > 0.999
        w = np.where(keep_f, np.maximum(snr - 1.0, 0.0), 0.0)
        m, sd, n_eff = M.weighted_stats(delta, w)
        prom, prom_off = profile_prominence(spec_db, rev_axis, keep_f, k, float(band_hz_k[a]))
        keep_env = np.interp(t_env, ft, clean[a].astype(float)) > 0.999
        pp, coh = pulse_pair(z_on[:, a], k, keep_env)
        snr_c = snr[keep_f] if keep_f.any() else snr
        per_k[str(k)] = [
            None if not np.isfinite(m) else round(m, 4),  # 0 peak-pick offset
            round(float(np.median(snr_c)), 4),  # 1 median per-frame SNR (lin)
            None if not np.isfinite(sd) else round(sd, 4),  # 2 std
            round(n_eff, 2),  # 3
            round(prom, 3) if np.isfinite(prom) else None,  # 4 profile prominence dB
            round(prom_off, 4) if np.isfinite(prom_off) else None,  # 5 profile peak offset
            None if not np.isfinite(pp) else round(pp, 4),  # 6 pulse-pair offset
            round(coh, 4),  # 7 pulse-pair coherence
            round(float(np.mean(keep_f)), 3),  # 8 uncollided frame fraction
            round(eff_search_revs(k, float(band_hz_k[a])), 4),  # 9 search half-width rev/s
        ]
        d_grid[a] = np.interp(ft, tf, delta)
        w_grid[a] = np.interp(ft, tf, w)

    def combine(kset: tuple[int, ...]) -> np.ndarray:
        idx = [k - 1 for k in kset]
        d, w = d_grid[idx], w_grid[idx]
        good = np.isfinite(d)
        w = np.where(good, w, 0.0)
        d = np.where(good, d, 0.0)
        tot = w.sum(axis=0)
        return np.where(tot > 0, (w * d).sum(axis=0) / np.maximum(tot, 1e-30), np.nan)

    lowk, highk = combine(LOW_K), combine(HIGH_K)
    return {
        "per_k": per_k,
        "low_k_series_mae": round(float(np.nanmean(np.abs(lowk))), 4),
        "high_k_series_mae": round(float(np.nanmean(np.abs(highk))), 4),
        "low_k_series_mean": round(float(np.nanmean(lowk)), 4),
        "high_k_series_mean": round(float(np.nanmean(highk)), 4),
    }


def trace_variant(
    z_on: np.ndarray, band_hz_k: np.ndarray, ks: list[int], clean: np.ndarray, ft: np.ndarray
) -> dict[str, np.ndarray]:
    """delta_k(t) on a common 2 s / 0.25 s base: peak-pick and sliding pulse-pair."""
    n_seg = int(DISP_SEG_S * FS_ENV)
    hop = int(DISP_HOP_S * FS_ENV)
    n_env = z_on.shape[-1]
    starts = np.arange(0, n_env - n_seg + 1, hop)
    t = (starts + n_seg / 2.0) / FS_ENV
    win = np.hanning(n_seg)
    freqs = np.fft.fftshift(np.fft.fftfreq(n_seg, d=1.0 / FS_ENV))
    dt = 1.0 / FS_ENV
    d_pk = np.full((len(ks), len(starts)), np.nan)
    d_pp = np.full((len(ks), len(starts)), np.nan)
    snr = np.zeros((len(ks), len(starts)))
    keep_all = np.zeros((len(ks), len(starts)), dtype=bool)
    t_env = (np.arange(n_env) + 0.5) * STRIDE / SR
    for a, k in enumerate(ks):
        band = float(band_hz_k[a])
        keep = np.abs(freqs) <= band
        rev = freqs[keep] / k
        sw = np.abs(rev) <= eff_search_revs(k, band)
        idx_off = int(np.argmax(sw))
        keep_env = np.interp(t_env, ft, clean[a].astype(float)) > 0.999
        z = z_on[:, a]
        lag = z[:, 1:] * np.conj(z[:, :-1])
        for b, s in enumerate(starts):
            seg = z[:, s : s + n_seg] * win
            p = (np.abs(np.fft.fftshift(np.fft.fft(seg, axis=-1), axes=-1)) ** 2).mean(0)[keep]
            j = int(np.argmax(p[sw])) + idx_off
            if 0 < j < len(p) - 1:
                y0, y1, y2 = np.log(p[j - 1 : j + 2] + 1e-300)
                den = y0 - 2 * y1 + y2
                frac = float(np.clip(0.5 * (y0 - y2) / den, -0.5, 0.5)) if abs(den) > 1e-12 else 0.0
            else:
                frac = 0.0
            d_pk[a, b] = rev[j] + frac * float(rev[1] - rev[0])
            snr[a, b] = float(p[j]) / max(float(np.median(p)), 1e-300)
            m = keep_env[s : s + n_seg]
            mm = m[1:] & m[:-1]
            if mm.sum() >= 8:
                v = lag[:, s : s + n_seg - 1][:, mm]
                d_pp[a, b] = float(np.angle(complex(v.sum())) / (2.0 * np.pi * k * dt))
        # The collision gate is REPORTED, not applied: on DREGON the k = 2
        # harmonic of a twin pair is collided at every frame, so applying the
        # gate would delete exactly the trace the wiggle question is about.
        keep_all[a] = np.interp(t, ft, clean[a].astype(float)) > 0.999
    return {
        "t": t,
        "d_peak": d_pk,
        "d_pp": d_pp,
        "snr": snr,
        "k": np.array(ks),
        "keep": keep_all,
    }


def run_window(key: str, do_trace: bool) -> dict[str, Any]:
    w = load_window(key)
    ks = list(range(1, K_MAX + 1))
    part_key = PARTNER[key]
    part = load_window(part_key)
    res: dict[str, Any] = {
        "key": key,
        "regime": w["regime"],
        "partner": part_key,
        "rotor_mean_rev_s": [round(float(np.mean(w["r_ft"][r])), 3) for r in range(N_ROTORS)],
        "partner_mean_rev_s": [round(float(np.mean(part["r_ft"][r])), 3) for r in range(N_ROTORS)],
        "rotors": {},
    }
    for rot in range(N_ROTORS):
        res["rotors"][str(rot)] = {
            "on": measure_variant(w["audio"], w["ft"], w["r_ft"], w["r_ft"], rot, ks, half=False),
            "off": measure_variant(w["audio"], w["ft"], w["r_ft"], w["r_ft"], rot, ks, half=True),
            "mis": measure_variant(
                w["audio"], w["ft"], part["r_ft"], w["r_ft"], rot, ks, half=False
            ),
        }
    if do_trace:
        # strongest rotor = highest median on-comb prominence over k = 2..40
        def med_prom(r: int) -> float:
            v = [
                res["rotors"][str(r)]["on"]["per_k"][str(k)][4]
                for k in range(2, K_MAX + 1)
                if res["rotors"][str(r)]["on"]["per_k"][str(k)][4] is not None
            ]
            return float(np.median(v)) if v else -99.0

        rot = max(range(N_ROTORS), key=med_prom)
        res["trace_rotor"] = rot
        z_on, band_hz_k = bank(w["audio"], w["r_ft"][rot], w["ft"], ks, half=False)
        clean = ~carrier_collision_mask(w["r_ft"], w["r_ft"][rot], rot, ks, half=False)
        clean_off = ~carrier_collision_mask(w["r_ft"], w["r_ft"][rot], rot, ks, half=True)
        tr = trace_variant(z_on, band_hz_k, ks, clean, w["ft"])
        z_off, band_off = bank(w["audio"], w["r_ft"][rot], w["ft"], ks, half=True)
        tr_off = trace_variant(z_off, band_off, ks, clean_off, w["ft"])
        payload: dict[str, Any] = {
            "rotor": np.array(rot),
            "band_hz_k": band_hz_k,
            "ft": w["ft"],
            "r_ft": w["r_ft"],
        }
        payload.update({f"on__{a}": b for a, b in tr.items()})
        payload.update({f"off__{a}": b for a, b in tr_off.items()})
        np.savez_compressed(TRACE_DIR / f"{key}.npz", **payload)
    return res


def _job(a: tuple[str, bool]) -> dict[str, Any]:
    return run_window(*a)


def main() -> None:
    TRACE_DIR.mkdir(parents=True, exist_ok=True)
    keys = list(PARTNER)
    jobs = [(k, k in TRACE_WINDOWS) for k in keys]
    print(f"[nullcontrol] {len(jobs)} windows x 3 variants", flush=True)
    with ProcessPoolExecutor(max_workers=8) as pool:
        rows = list(pool.map(_job, jobs))
    out = {
        "protocol": {
            "variants": {
                "on": "carrier k * g_r(t) — the measurement",
                "off": "carrier (k + 0.5) * g_r(t) — off-comb null, no rotor line",
                "mis": "carrier k * g_partner(t) — mismatched telemetry null",
            },
            "search_half_width": "min(1.5 k, 8) Hz, further capped at 0.9 * band",
            "band_hz_k": "min(3 k, 0.45 * mean rate) Hz",
            "entry_schema": "per_k[k] = [peak_offset, median_snr_lin, std, n_eff, "
            "prominence_db, prominence_offset, pulse_pair_offset, pp_coherence, "
            "frac_uncollided, search_half_width_rev_s]",
            "fs_env": FS_ENV,
            "low_k_set": list(LOW_K),
            "high_k_set": list(HIGH_K),
        },
        "windows": {r["key"]: r for r in rows},
    }
    (OUT / "nullcontrol.json").write_text(json.dumps(out, indent=1))
    print(f"[nullcontrol] wrote {OUT / 'nullcontrol.json'}")


if __name__ == "__main__":
    main()
