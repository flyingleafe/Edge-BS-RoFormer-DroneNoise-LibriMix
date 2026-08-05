#!/usr/bin/env python3
"""Acoustic shaft rate on DREGON's STATIC BENCH recordings + an audio-clock check.

Part B — bench. ``DREGON_individual_motors_recordings/Motor{1-4}_{50..90}.wav``
(+ ``allMotors_70``) carry NO telemetry: the only reference is the nominal
number in the file name. The acoustic shaft rate is estimated by a comb fit
(harmonics 1..K_MAX of a candidate f0, log-magnitude score on a long FFT,
parabolic refinement), per 5 s block so the scatter is honest.

Part C — audio clock. ``silent-flight_whitenoise-low_room1`` is 78 s of the
2 min emitted white-noise file re-recorded by the SAME onboard 8-mic array with
the motors off. Cross-correlating successive blocks against the emitted file
gives the lag drift, i.e. (recording clock) / (playback clock) - 1. A 0.54 %
recording-clock error would show as 0.54 % lag drift (about 420 ms over 78 s).
This is what separates "telemetry over-reports" from "audio time base is
stretched" — the degeneracy WP14 flagged on Michael's rig.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
DREGON = ROOT / "data/DREGON"
OUT = Path(__file__).resolve().parent

K_MAX = 40
F_MAX = 6000.0
BLOCK_S = 5.0


def comb_score(mag_log: np.ndarray, freqs: np.ndarray, f0s: np.ndarray, k_max: int) -> np.ndarray:
    """MEAN log-magnitude at k*f0 over k = 1..k_max (f <= F_MAX).

    The mean (not the sum) is what kills the subharmonic trap: at f0/2 half the
    slots land on empty bins and drag the average down, whereas a sum over a
    denser low-frequency comb can win outright.
    """
    df = freqs[1] - freqs[0]
    tot = np.zeros(len(f0s))
    cnt = np.zeros(len(f0s))
    for k in range(1, k_max + 1):
        f = k * f0s
        ok = f <= F_MAX
        idx = np.clip(np.round(f / df).astype(int), 0, len(mag_log) - 1)
        tot += np.where(ok, mag_log[idx], 0.0)
        cnt += ok
    return tot / np.maximum(cnt, 1)


def estimate_f0(y: np.ndarray, sr: int, lo: float, hi: float) -> tuple[float, float]:
    """(f0, sharpness) by comb fit; sharpness = peak minus median score."""
    n = 1 << int(np.ceil(np.log2(len(y))))
    Y = np.abs(np.fft.rfft(y * np.hanning(len(y)), n))
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    mag_log = np.log(Y + 1e-12)
    grid = np.arange(lo, hi, 0.01)
    sc = comb_score(mag_log, freqs, grid, K_MAX)
    j = int(np.argmax(sc))
    fine = np.arange(max(lo, grid[j] - 0.05), grid[j] + 0.05, 0.0005)
    sf_ = comb_score(mag_log, freqs, fine, K_MAX)
    jj = int(np.argmax(sf_))
    return float(fine[jj]), float(sc[j] - np.median(sc))


def bench() -> dict:
    d = DREGON / "DREGON_individual_motors_recordings"
    rows = []
    for p in sorted(d.glob("*.wav")):
        m = re.match(r"(Motor\d|allMotors)_(\d+)\.wav", p.name)
        if not m:
            continue
        motor, nominal = m.group(1), float(m.group(2))
        x, sr = sf.read(p, always_2d=True)
        y = x.mean(axis=1)
        # steady part: drop the spin-up / spin-down thirds of a second
        w = int(0.5 * sr)
        rms = np.array([np.sqrt(np.mean(y[i : i + w] ** 2)) for i in range(0, len(y) - w, w)])
        live = rms > 0.5 * np.median(rms[rms > 0])
        i0 = int(np.argmax(live)) + 2
        i1 = len(live) - int(np.argmax(live[::-1])) - 2
        s0, s1 = i0 * w, i1 * w
        if s1 - s0 < 6 * sr:
            continue
        nb = int((s1 - s0) / (BLOCK_S * sr))
        f0s, sharp = [], []
        for b in range(nb):
            a = s0 + int(b * BLOCK_S * sr)
            # Search 0.6..1.4 x nominal. The window is 40 % wide against a
            # ~1 % effect, so it cannot manufacture the offset; it only stops
            # the comb fit locking onto f0/2 or 2 f0.
            f0, sh = estimate_f0(y[a : a + int(BLOCK_S * sr)], sr, 0.6 * nominal, 1.4 * nominal)
            f0s.append(f0)
            sharp.append(sh)
        f0s = np.array(f0s)
        # half/double sanity: score of f0/2 and 2*f0 on the whole steady span
        yy = y[s0 : s0 + int(min(20.0, (s1 - s0) / sr) * sr)]
        n = 1 << int(np.ceil(np.log2(len(yy))))
        Y = np.log(np.abs(np.fft.rfft(yy * np.hanning(len(yy)), n)) + 1e-12)
        fr = np.fft.rfftfreq(n, 1.0 / sr)
        med = float(np.median(Y))
        f0m = float(np.median(f0s))
        sub = {
            str(r): round(float(comb_score(Y, fr, np.array([f0m * r]), K_MAX)[0] / K_MAX - med), 3)
            for r in (0.5, 1.0, 2.0)
        }
        rows.append(
            {
                "file": p.name,
                "motor": motor,
                "nominal": nominal,
                "n_blocks": nb,
                "steady_s": round((s1 - s0) / sr, 1),
                "f0_median_rev_s": round(f0m, 4),
                "f0_std_rev_s": round(float(np.std(f0s)), 4),
                "f0_min": round(float(f0s.min()), 3),
                "f0_max": round(float(f0s.max()), 3),
                "sharpness": round(float(np.median(sharp)), 2),
                "comb_score_per_harmonic_at_half_one_double": sub,
                "acoustic_over_nominal": round(f0m / nominal, 5),
            }
        )
        print(
            f"  {p.name:20s} nominal {nominal:5.1f}  acoustic {f0m:8.4f} "
            f"+-{np.std(f0s):.4f}  ratio {f0m / nominal:.5f}",
            flush=True,
        )
    return {"rows": rows}


def audio_clock() -> dict:
    """Lag drift of the re-recorded emitted white noise vs the emitted file."""
    rec_p = DREGON / "DREGON_silent-flight_whitenoise-low_room1"
    wavs = sorted(rec_p.glob("*.wav"))
    ref_p = DREGON / "emitted_signals/2min_white_noise.wav"
    if not wavs or not ref_p.exists():
        return {"available": False}
    rec, sr = sf.read(wavs[0], always_2d=True)
    ref, sr2 = sf.read(str(ref_p), always_2d=True)
    ref = ref[:, 0]
    y = rec[:, 0]
    if sr2 != sr:
        # Resample the recording onto the emitted file's rate. A resample is a
        # pure change of time UNITS: it rescales lag and time together, so the
        # drift SLOPE (a dimensionless clock ratio) is untouched.
        import librosa

        y = librosa.resample(y, orig_sr=sr, target_sr=sr2, res_type="soxr_hq")
        sr = sr2
    seg = int(4.0 * sr)
    hop = int(2.0 * sr)

    def lag_of(block: np.ndarray, ref_slice: np.ndarray) -> tuple[float, float]:
        """GCC-PHAT lag of ``block`` inside ``ref_slice`` (samples, fractional).

        PHAT is restricted to 200-4000 Hz: the onboard array's response rolls
        off hard above 4 kHz (measured, see the band levels), so unweighted
        whitening there amplifies pure noise.
        """
        n = 1 << int(np.ceil(np.log2(len(ref_slice) + len(block))))
        A = np.fft.rfft(ref_slice, n)
        B = np.fft.rfft(block, n)
        f = np.fft.rfftfreq(n, 1.0 / sr)
        R = A * np.conj(B)
        R /= np.abs(R) + 1e-12
        R[(f < 200.0) | (f > 4000.0)] = 0.0
        c = np.fft.irfft(R, n)
        j = int(np.argmax(c))
        peak = c[j]
        jm, jp = c[(j - 1) % n], c[(j + 1) % n]
        den = jm - 2 * peak + jp
        frac = 0.5 * (jm - jp) / den if abs(den) > 1e-30 else 0.0
        lag = j + float(np.clip(frac, -0.5, 0.5))
        if lag > n / 2:
            lag -= n
        return float(lag), float(peak / (np.std(c) + 1e-30))

    # global search: the first block against the whole reference
    l0, _ = lag_of(y[:seg], ref)
    rows = []
    b = 0
    while b * hop + seg <= len(y):
        a = b * hop
        b += 1
        blk = y[a : a + seg]
        centre = int(round(l0)) + a
        lo = max(0, centre - 2 * sr)
        hi = min(len(ref), centre + 2 * sr + seg)
        if hi - lo < seg + sr:
            break
        lag, q = lag_of(blk, ref[lo:hi])
        rows.append((a / sr, (lag + lo - a) / sr, q))
    if len(rows) < 5:
        return {"available": False, "note": "alignment failed"}
    t = np.array([r[0] for r in rows])
    lag = np.array([r[1] for r in rows])
    q = np.array([r[2] for r in rows])
    good = q > np.median(q) * 0.5
    A = np.polyfit(t[good], lag[good], 1)
    resid = lag[good] - np.polyval(A, t[good])
    return {
        "available": True,
        "recording": wavs[0].name,
        "n_blocks": int(good.sum()),
        "span_s": round(float(t[good].max() - t[good].min()), 1),
        "lag_drift_slope": float(A[0]),
        "clock_ratio_rec_over_ref": round(float(1.0 - A[0]), 8),
        "clock_error_pct": round(float(-A[0] * 100), 5),
        "resid_rms_ms": round(float(np.std(resid) * 1e3), 3),
        "peak_quality_median": round(float(np.median(q)), 1),
        "lag_first_last_ms": [
            round(float(lag[good][0] * 1e3), 3),
            round(float(lag[good][-1] * 1e3), 3),
        ],
    }


def main() -> None:
    print("[bench] single-motor / all-motor static-bench acoustic rate")
    b = bench()
    print("[clock] audio-clock drift vs the emitted white-noise file")
    c = audio_clock()
    print(json.dumps(c, indent=1))
    (OUT / "bench_rate.json").write_text(json.dumps({"bench": b, "audio_clock": c}, indent=1))
    print(f"[bench] wrote {OUT / 'bench_rate.json'}")


if __name__ == "__main__":
    main()
