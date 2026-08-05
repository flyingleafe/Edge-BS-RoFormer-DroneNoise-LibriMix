#!/usr/bin/env python3
"""Per-harmonic displacement of the rotor comb from telemetry ("displaced lock").

For every cruise window of the frozen ``beatvk-valid-raw`` protocol and every
rotor r with telemetry trajectory g_r(t), the audio is heterodyned by
``exp(-i 2 pi k \\int g_r dt)`` for k = 1..K_MAX and brickwall-lowpassed to a
per-harmonic band. In the demodulated envelope the telemetry rate is DC, so a
ridge at envelope frequency f corresponds to an acoustic shaft-rate offset
``delta_k = f / k`` rev/s. A short-time spectrum of the envelope gives
``delta_k(t)`` plus a per-frame demod SNR weight.

Outputs ``displacement.json`` (per-window / per-rotor / per-k offsets and
weights, pooled profiles, and the three-way error decomposition) plus the
per-window envelope spectrograms needed by the figures.
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

PREP = ROOT / ("omnirun-outputs/bandadm-ladder-7fb2e4/results/beatvk_bandadm/vk_arms/prep_cache")
LADDER = ROOT / "omnirun-outputs/bandadm-ladder-7fb2e4/results/beatvk_bandadm"
OUT = Path(__file__).resolve().parent
SPEC_DIR = OUT / "specs"

SR = 16000
FRAME_S = 0.032
N_ROTORS = 4
K_MAX = 40
FS_ENV = 250.0  # envelope rate (Hz) after decimation
STRIDE = int(round(SR / FS_ENV))  # 64
B0_REVS = 3.0  # nominal per-harmonic band: +- B0_REVS rev/s
BAND_FRAC_OF_RATE = 0.45  # ... capped so the band never nears the next harmonic
SEARCH_REVS = 1.5  # peak search half-width (rev/s) ...
SEARCH_HZ_CAP = 8.0  # ... capped in Hz, so at high k the window stays inside the
# spacing of the interleaved 4-rotor comb (half-width = min(1.5 k, 8) Hz)
COLLISION_GUARD = 1.6  # gate an interferer within this multiple of the search window
LOW_K = tuple(range(2, 14))  # k = 2..13   — the "displaced" set
HIGH_K = tuple(range(16, 41))  # k = 16..40 — the "on-grid" set
JITTER_FLOOR = 0.2  # rev/s, the honest 0 dB label-jitter floor

DREGON_RECS = (
    "free-flight_nosource_room1",
    "free-flight_speech-low_room1",
    "free-flight_whitenoise-low_room1",
)
FLY124_REC = "FLY124"


# window class -> the pools of the frozen protocol
def window_class(rid: str, widx: int, regime: str) -> str:
    if rid == FLY124_REC:
        return "fly124_cruise" if regime == "cruise" else "fly124_warmup"
    return "dregon_ramp" if widx == 0 else "dregon_steady"


def seg_len_env(k: int) -> int:
    """Envelope-STFT segment length (samples) for harmonic k.

    Chosen so the rev/s resolution ``fs_env / (n_seg * k)`` stays ~constant
    (~0.06 rev/s) across k, while never exceeding the measured coherence
    time at high k (1 s floor, 8 s ceiling).
    """
    seg_s = float(np.clip(16.0 / max(k, 1), 1.0, 8.0))
    n = int(round(seg_s * FS_ENV))
    return n - (n % 2)


def envelope_bank(
    audio: np.ndarray, r_ft: np.ndarray, ft: np.ndarray, rotor: int, ks: list[int]
) -> tuple[np.ndarray, np.ndarray]:
    """``(z (C, K, n_env), band_hz_k (K,))`` — the tracker's demod bank at
    ``k * telemetry`` with a per-harmonic band."""
    from tracking.phase_increment_tracker import _demod_bank

    n_t = audio.shape[-1]
    t_aud = np.arange(n_t) / SR
    r_aud = np.interp(t_aud, ft, r_ft[rotor])
    phi = 2.0 * np.pi * np.cumsum(r_aud) / SR
    r_mean = float(np.mean(r_ft[rotor]))
    band_hz_k = np.array(
        [min(B0_REVS * k, BAND_FRAC_OF_RATE * r_mean) for k in ks], dtype=np.float64
    )
    n_env = n_t // STRIDE
    y32 = np.asarray(audio, dtype=np.float32)
    z_on, _ = _demod_bank(
        y32,
        phi,
        t_aud,
        ks,
        0.0,
        STRIDE,
        n_env,
        float(np.max(band_hz_k)) / SR,
        band_cyc_k=band_hz_k / SR,
    )
    return z_on, band_hz_k


def ridge_from_envelope(
    z_k: np.ndarray, band_hz: float, k: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Short-time ridge of one harmonic's envelope.

    ``z_k`` is ``(C, n_env)``. Returns ``(t_frames_s, delta_revs, snr_lin,
    spec_db (F, T), rev_axis (F,))``: per STFT frame the parabolically
    refined peak offset in rev/s and its power ratio over the in-band median,
    plus the channel-averaged spectrogram in rev/s units for the figures.
    """
    n_seg = seg_len_env(k)
    n_env = z_k.shape[-1]
    n_seg = min(n_seg, n_env)
    hop = max(n_seg // 2, 1)
    starts = list(range(0, n_env - n_seg + 1, hop))
    win = np.hanning(n_seg)
    freqs = np.fft.fftshift(np.fft.fftfreq(n_seg, d=1.0 / FS_ENV))
    keep = np.abs(freqs) <= band_hz
    rev_axis = freqs[keep] / k
    spec = np.empty((len(starts), int(keep.sum())))
    for a, s in enumerate(starts):
        seg = z_k[:, s : s + n_seg] * win
        p = np.abs(np.fft.fftshift(np.fft.fft(seg, axis=-1), axes=-1)) ** 2
        spec[a] = p.mean(axis=0)[keep]  # incoherent channel average
    t_frames = (np.array(starts) + n_seg / 2.0) / FS_ENV
    search = np.abs(rev_axis) <= min(search_hz(k), 0.9 * band_hz) / k
    idx_off = int(np.argmax(search))  # first True
    delta = np.full(len(starts), np.nan)
    snr = np.zeros(len(starts))
    for a in range(len(starts)):
        row = spec[a]
        sub = row[search]
        j = int(np.argmax(sub)) + idx_off
        # parabolic refinement on the log-power peak
        if 0 < j < len(row) - 1:
            y0, y1, y2 = np.log(row[j - 1 : j + 2] + 1e-300)
            den = y0 - 2 * y1 + y2
            frac = 0.5 * (y0 - y2) / den if abs(den) > 1e-12 else 0.0
            frac = float(np.clip(frac, -0.5, 0.5))
        else:
            frac = 0.0
        step = rev_axis[1] - rev_axis[0]
        delta[a] = rev_axis[j] + frac * step
        floor = float(np.median(row))
        snr[a] = float(row[j]) / max(floor, 1e-300)
    spec_db = 10.0 * np.log10(spec.T + 1e-300)
    return t_frames, delta, snr, spec_db, rev_axis


def weighted_stats(vals: np.ndarray, w: np.ndarray) -> tuple[float, float, float]:
    """``(weighted mean, weighted std, n_eff)``."""
    ok = np.isfinite(vals) & (w > 0)
    if not ok.any():
        return float("nan"), float("nan"), 0.0
    v, ww = vals[ok], w[ok]
    m = float(np.sum(ww * v) / np.sum(ww))
    var = float(np.sum(ww * (v - m) ** 2) / np.sum(ww))
    n_eff = float(np.sum(ww) ** 2 / np.sum(ww**2))
    return m, float(np.sqrt(var)), n_eff


def search_hz(k: int) -> float:
    """Peak-search half-width in Hz for harmonic k."""
    return min(SEARCH_REVS * k, SEARCH_HZ_CAP)


def collision_mask(r_ft: np.ndarray, rotor: int, ks: list[int]) -> np.ndarray:
    """``(K, N)`` bool: True where another rotor's harmonic enters the peak
    search window of harmonic k of ``rotor`` (the tracker's own twin rule at
    ``sep_hz = COLLISION_GUARD * search_hz(k)``)."""
    from tracking.phase_increment_tracker import _twin_collision_mask

    sep = np.array([COLLISION_GUARD * search_hz(k) for k in ks], dtype=np.float64)
    return _twin_collision_mask(r_ft, rotor, len(ks), sep, f_max=6000.0, min_rate=5.0)


def run_window(rid: str, widx: int, save_spec: bool) -> dict[str, Any]:
    with np.load(PREP / f"{rid}__w{widx:02d}.npz") as z:
        audio = np.asarray(z["audio"], np.float64)
        ft = np.asarray(z["ft"], np.float64)
        r_ft = np.asarray(z["r_meas"], np.float64)
        regime = str(z["regime"])
        start_s = float(z["start_s"])
    ks = list(range(1, K_MAX + 1))
    tg = ft  # the 0.032 s protocol grid inside this window
    res: dict[str, Any] = {
        "recording": rid,
        "window": widx,
        "regime": regime,
        "start_s": start_s,
        "klass": window_class(rid, widx, regime),
        "rotor_mean_rev_s": [round(float(np.mean(r_ft[r])), 3) for r in range(N_ROTORS)],
        "rotors": {},
    }
    per_rotor_lowk, per_rotor_highk = [], []
    for rot in range(N_ROTORS):
        z_on, band_hz_k = envelope_bank(audio, r_ft, ft, rot, ks)
        # A neighbouring rotor's harmonic inside the search window would capture
        # the peak. Gate those (harmonic, frame) pairs out with the tracker's own
        # twin rule, at the search half-width in Hz (sep_hz = SEARCH_REVS * k).
        clean = ~collision_mask(r_ft, rot, ks)
        rot_out: dict[str, Any] = {}
        d_grid = np.full((len(ks), tg.size), np.nan)
        w_grid = np.zeros((len(ks), tg.size))
        specs: dict[str, Any] = {}
        for a, k in enumerate(ks):
            tf, delta, snr, spec_db, rev_axis = ridge_from_envelope(
                z_on[:, a], float(band_hz_k[a]), k
            )
            w = np.maximum(snr - 1.0, 0.0)
            keep_f = np.interp(tf, ft, clean[a].astype(float)) > 0.999
            w = np.where(keep_f, w, 0.0)
            m, sd, n_eff = weighted_stats(delta, w)
            snr_clean = snr[keep_f] if keep_f.any() else snr
            rot_out[str(k)] = [
                round(m, 4) if np.isfinite(m) else None,
                round(float(np.median(snr_clean)), 3),
                round(sd, 4) if np.isfinite(sd) else None,
                round(n_eff, 2),
                round(float(band_hz_k[a]), 2),
                round(float(np.mean(keep_f)), 3),
            ]
            d_grid[a] = np.interp(tg, tf, delta)
            w_grid[a] = np.interp(tg, tf, w)
            if save_spec and k in (2, 5, 8, 13, 16, 22, 30):
                specs[str(k)] = {
                    "t": tf,
                    "rev": rev_axis,
                    "spec_db": spec_db.astype(np.float32),
                }
        res["rotors"][str(rot)] = rot_out

        def combine(
            kset: tuple[int, ...], d_grid: np.ndarray = d_grid, w_grid: np.ndarray = w_grid
        ) -> np.ndarray:
            idx = [k - 1 for k in kset]
            d, w = d_grid[idx], w_grid[idx]
            good = np.isfinite(d)
            w = np.where(good, w, 0.0)
            d = np.where(good, d, 0.0)
            tot = w.sum(axis=0)
            out = np.where(tot > 0, (w * d).sum(axis=0) / np.maximum(tot, 1e-30), np.nan)
            return out

        per_rotor_lowk.append(combine(LOW_K))
        per_rotor_highk.append(combine(HIGH_K))
        if save_spec:
            np.savez_compressed(
                SPEC_DIR / f"{rid}__w{widx:02d}__r{rot}.npz",
                **{f"{k}__{f}": v for k, d in specs.items() for f, v in d.items()},
            )
    lowk = np.vstack(per_rotor_lowk)
    highk = np.vstack(per_rotor_highk)
    res["comb_offset_series"] = {
        "low_k": {
            "per_rotor_mean": [round(float(np.nanmean(x)), 4) for x in lowk],
            "mae": round(float(np.nanmean(np.abs(lowk))), 4),
        },
        "high_k": {
            "per_rotor_mean": [round(float(np.nanmean(x)), 4) for x in highk],
            "mae": round(float(np.nanmean(np.abs(highk))), 4),
        },
    }
    return res


def _job(args: tuple[str, int, bool]) -> dict[str, Any]:
    return run_window(*args)


def main() -> None:
    SPEC_DIR.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((LADDER / "vk_arms" / "manifest.json").read_text())
    spec_windows = {
        ("free-flight_nosource_room1", 1),
        ("free-flight_speech-low_room1", 1),
        (FLY124_REC, 3),
    }
    jobs = [
        (rid, int(w["index"]), (rid, int(w["index"])) in spec_windows)
        for rid, rec in manifest["recordings"].items()
        for w in rec["windows"]
    ]
    print(f"[displacement] {len(jobs)} windows", flush=True)
    with ProcessPoolExecutor(max_workers=8) as pool:
        rows = list(pool.map(_job, jobs))
    rows.sort(key=lambda r: (r["recording"], r["window"]))

    # ── pooled per-k profiles per class ───────────────────────────────────
    pooled: dict[str, Any] = {}
    for klass in (
        "dregon_cruise",
        "dregon_steady",
        "dregon_ramp",
        "fly124_cruise",
        "fly124_warmup",
    ):
        if klass == "dregon_cruise":
            sel = [r for r in rows if r["recording"] in DREGON_RECS]
        else:
            sel = [r for r in rows if r["klass"] == klass]
        if not sel:
            continue
        prof: dict[str, Any] = {}
        for k in range(1, K_MAX + 1):
            vals, wts = [], []
            for r in sel:
                for rot in range(N_ROTORS):
                    e = r["rotors"][str(rot)][str(k)]
                    if e[0] is None:
                        continue
                    vals.append(e[0])
                    wts.append(max(e[1] - 1.0, 0.0))
            if not vals:
                continue
            v, w = np.array(vals), np.array(wts)
            m, sd, n_eff = weighted_stats(v, w)
            prof[str(k)] = {
                "mean_offset_rev_s": round(m, 4),
                "std_rev_s": round(sd, 4),
                "sem_rev_s": round(sd / max(np.sqrt(n_eff), 1e-9), 4),
                "n_units": len(vals),
                "mean_frame_retention": round(
                    float(
                        np.mean(
                            [
                                r["rotors"][str(rot)][str(k)][5]
                                for r in sel
                                for rot in range(N_ROTORS)
                                if r["rotors"][str(rot)][str(k)][0] is not None
                            ]
                        )
                    ),
                    3,
                ),
                "median_snr": round(float(np.median([x + 1 for x in wts])), 3),
            }

        def band_mean(kk: tuple[int, ...], prof: dict[str, Any] = prof) -> float:
            vals = [prof[str(k)]["mean_offset_rev_s"] for k in kk if str(k) in prof]
            return round(float(np.mean(vals)), 4)

        pooled[klass] = {
            "per_k": prof,
            "low_k_mean_offset_rev_s": band_mean(LOW_K),
            "high_k_mean_offset_rev_s": band_mean(HIGH_K),
            "n_windows": len(sel),
        }

    # ── three-way error decomposition (DREGON cruise) ─────────────────────
    report = json.loads((LADDER / "report.json").read_text())
    flag_rows = {
        f"{r['recording']}__w{r['window']:02d}": r["mae"] for r in report["per_window"]["peeled_x3"]
    }
    init_rows = {
        f"{r['recording']}__w{r['window']:02d}": r["mae"] for r in report["per_window"]["init"]
    }
    decomp: dict[str, Any] = {"per_window": {}, "pooled": {}}
    for r in rows:
        key = f"{r['recording']}__w{r['window']:02d}"
        decomp["per_window"][key] = {
            "klass": r["klass"],
            "low_k_comb_mae": r["comb_offset_series"]["low_k"]["mae"],
            "high_k_comb_mae": r["comb_offset_series"]["high_k"]["mae"],
            "flagship_track_mae": flag_rows.get(key),
            "blind_init_mae": init_rows.get(key),
        }
    for name, pred in (
        ("dregon_cruise", lambda r: r["recording"] in DREGON_RECS),
        ("dregon_steady", lambda r: r["klass"] == "dregon_steady"),
        ("fly124_cruise", lambda r: r["klass"] == "fly124_cruise"),
    ):
        sel = [
            decomp["per_window"][f"{r['recording']}__w{r['window']:02d}"] for r in rows if pred(r)
        ]
        agg = {}
        for field in ("low_k_comb_mae", "high_k_comb_mae", "flagship_track_mae", "blind_init_mae"):
            vs = [s[field] for s in sel if s[field] is not None]
            agg[field] = round(float(np.mean(vs)), 4) if vs else None
        agg["n_windows"] = len(sel)
        decomp["pooled"][name] = agg

    out = {
        "protocol": {
            "dataset": "beatvk-valid-raw@54849c13ed3a",
            "sr": SR,
            "k_max": K_MAX,
            "fs_env": FS_ENV,
            "band_hz_k": f"min({B0_REVS} * k, {BAND_FRAC_OF_RATE} * mean rate) Hz",
            "search_half_width": f"min({SEARCH_REVS} * k, {SEARCH_HZ_CAP}) Hz",
            "low_k_set": list(LOW_K),
            "high_k_set": list(HIGH_K),
            "jitter_floor_rev_s": JITTER_FLOOR,
            "entry_schema": "rotors[rotor][k] = [offset_rev_s, median_snr_lin, "
            "std_rev_s, n_eff_frames, band_hz, frac_frames_uncollided]",
            "collision_gate": "frames where another rotor's harmonic enters the "
            "+-SEARCH_REVS*k Hz search window are dropped (_twin_collision_mask)",
            "flagship_row": "protocol peeled_x3 (report.json per_window)",
        },
        "windows": {f"{r['recording']}__w{r['window']:02d}": r for r in rows},
        "pooled": pooled,
        "error_decomposition": decomp,
    }
    (OUT / "displacement.json").write_text(json.dumps(out, indent=1))
    print(f"[displacement] wrote {OUT / 'displacement.json'}")
    for klass, p in pooled.items():
        print(
            f"  {klass:<16} low-k {p['low_k_mean_offset_rev_s']:+.3f}  "
            f"high-k {p['high_k_mean_offset_rev_s']:+.3f} rev/s  ({p['n_windows']} win)"
        )
    print(json.dumps(decomp["pooled"], indent=1))


if __name__ == "__main__":
    main()
