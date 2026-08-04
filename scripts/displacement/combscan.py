#!/usr/bin/env python3
"""Global comb-scale scan in ORDER space — no peak-search window anywhere.

Resample the audio uniformly in the telemetry rotor phase (order spectrum), then
score a whole comb at once:

    S(s) = mean over k in the band of  dB( order = s * k )

If the acoustic comb follows the telemetry exactly, S peaks at s = 1. If the
comb is displaced by a constant fraction, S peaks at that s. Because the score
is a mean over MANY harmonics, a spurious peak needs all of them to conspire —
so the peak height over the score's own background IS the significance, and no
per-harmonic search window exists to bias the answer.

Bands are scanned separately (low / mid / high k) so "does the comb survive to
k = 100" is answered band by band, and the fan of a scale error (which grows
linearly in order) cannot be confused with a fixed-frequency artefact.

The null is the same scan run on a HALF-INTEGER comb (orders s*(k + 0.5)),
where no rotor line can exist.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import scipy.io  # noqa: E402
import soundfile as sf  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
FIGS = OUT / "figs"
sys.path.insert(0, str(ROOT / "src"))

from utils.paths import get_data_path  # noqa: E402

DREGON = get_data_path("DREGON")

SPR = 1024  # samples per revolution after phase resampling
S_GRID = np.arange(0.975, 1.0251, 0.00002)
BANDS = {"k2-13": (2, 13), "k14-40": (14, 40), "k41-75": (41, 75), "k76-110": (76, 110)}

# (label, recording, rotor, t0, entry) — the frozen cruise windows
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
WIN_S = 16.0


def load(rid: str, entry: str):
    d = DREGON / f"DREGON_{rid}"
    mat = scipy.io.loadmat(str(d / f"DREGON_{rid}_motors.mat"))["motor"]
    ts = mat["timestamps"][0, 0].flatten().astype(np.float64)
    vals = mat[entry][0, 0].astype(np.float64).T
    t0 = float(
        scipy.io.loadmat(str(d / f"DREGON_{rid}_audiots.mat"))["audio_timestamps"].flatten()[0]
    )
    x, sr = sf.read(str(d / f"DREGON_{rid}.wav"), always_2d=True)
    return x.T.astype(np.float64), sr, ts - t0, vals


def order_spectrum(audio, sr, t_tel, rate, t0, t1):
    a0, a1 = int(t0 * sr), int(t1 * sr)
    t = np.arange(a0, a1) / sr
    r = np.interp(t, t_tel, rate)
    phi = np.cumsum(r) / sr
    phi -= phi[0]
    n_out = int(float(phi[-1]) * SPR)
    grid = np.arange(n_out) / SPR
    t_at = np.interp(grid, phi, t)
    win = np.hanning(n_out)
    acc = None
    for c in range(audio.shape[0]):
        y = np.interp(t_at, t, audio[c, a0:a1])
        Y = np.abs(np.fft.rfft((y - y.mean()) * win)) ** 2
        acc = Y if acc is None else acc + Y
    orders = np.fft.rfftfreq(n_out, d=1.0 / SPR)
    p = acc / audio.shape[0]
    # normalize by a slowly varying floor so the score is a true excess in dB
    db = 10.0 * np.log10(p + 1e-30)
    n_sm = max(51, int(2.0 / (orders[1] - orders[0])) | 1)  # ~2-order median
    from scipy.ndimage import median_filter

    floor = median_filter(db, size=n_sm, mode="nearest")
    return orders, db - floor, float(np.mean(r))


def scan(orders, excess_db, k_lo, k_hi, half=False):
    """S(s) over the scale grid for the comb k = k_lo..k_hi (or k + 0.5)."""
    do = orders[1] - orders[0]
    ks = np.arange(k_lo, k_hi + 1, dtype=np.float64) + (0.5 if half else 0.0)
    o_max = orders[-1]
    sc = np.zeros(len(S_GRID))
    cnt = np.zeros(len(S_GRID))
    for k in ks:
        o = S_GRID * k
        ok = o < o_max
        idx = np.clip(np.round(o / do).astype(int), 0, len(excess_db) - 1)
        sc += np.where(ok, excess_db[idx], 0.0)
        cnt += ok
    return sc / np.maximum(cnt, 1), cnt


def summarize(sc):
    j = int(np.argmax(sc))
    s_hat = float(S_GRID[j])
    bg = float(np.median(sc))
    spread = float(np.percentile(sc, 90) - np.percentile(sc, 10))
    return {
        "s_hat": round(s_hat, 6),
        "pct": round((s_hat - 1.0) * 100, 4),
        "peak_db": round(float(sc[j]), 3),
        "background_db": round(bg, 3),
        "peak_over_bg_db": round(float(sc[j]) - bg, 3),
        "z": round((float(sc[j]) - bg) / max(spread, 1e-9), 2),
    }


def main() -> None:
    FIGS.mkdir(exist_ok=True)
    res: dict = {}
    curves: dict = {}
    for label, rid, t0, entry in DREGON_WINDOWS:
        audio, sr, tt, vals = load(rid, entry)
        dur = audio.shape[-1] / sr
        if t0 + WIN_S > dur:
            continue
        for rot in range(4):
            if float(np.mean(np.interp([t0, t0 + WIN_S], tt, vals[rot]))) < 30:
                continue
            orders, ex, rbar = order_spectrum(audio, sr, tt, vals[rot], t0, t0 + WIN_S)
            key = f"{label}__r{rot}"
            res[key] = {"rate": round(rbar, 3), "entry": entry, "bands": {}}
            for bname, (lo, hi) in BANDS.items():
                if hi * rbar > 0.45 * sr:  # beyond audio Nyquist
                    continue
                s_on, cnt = scan(orders, ex, lo, hi, half=False)
                s_off, _ = scan(orders, ex, lo, hi, half=True)
                res[key]["bands"][bname] = {
                    "on": summarize(s_on),
                    "null_half_integer": summarize(s_off),
                    "n_harmonics": int(cnt.max()),
                }
                curves[f"{key}__{bname}"] = (s_on, s_off)
            b = res[key]["bands"]
            print(
                f"{key:28s} r={rbar:6.2f} "
                + "  ".join(
                    f"{n}: {b[n]['on']['pct']:+.3f}% ({b[n]['on']['peak_over_bg_db']:.2f} dB"
                    f" / null {b[n]['null_half_integer']['peak_over_bg_db']:.2f})"
                    for n in b
                ),
                flush=True,
            )
    (OUT / "combscan.json").write_text(json.dumps(res, indent=1))

    # figure: score curves for the 3 frozen DREGON cruise windows, rotor 0
    show = [
        k
        for k in curves
        if k.startswith(("nosource_w01__r0", "whitenoise-low_w01__r0", "hovering_room2_w00__r0"))
    ]
    bands = list(BANDS)
    fig, axes = plt.subplots(3, len(bands), figsize=(4.2 * len(bands), 9), squeeze=False)
    rows = ["nosource_w01__r0", "whitenoise-low_w01__r0", "hovering_room2_w00__r0"]
    for i, rk in enumerate(rows):
        for j, bn in enumerate(bands):
            ax = axes[i][j]
            key = f"{rk}__{bn}"
            if key not in curves:
                ax.set_visible(False)
                continue
            on, off = curves[key]
            ax.plot((S_GRID - 1) * 100, on, lw=0.9, color="#1f4e79", label="on-comb")
            ax.plot((S_GRID - 1) * 100, off, lw=0.9, color="0.6", label="half-integer null")
            ax.axvline(0, color="k", lw=0.7)
            ax.axvline(-0.54, color="crimson", ls="--", lw=0.9)
            if i == 0:
                ax.set_title(bn, fontsize=10)
            if j == 0:
                ax.set_ylabel(rk.replace("__r0", ""), fontsize=8)
            ax.set_xlabel("scale s - 1 [%]", fontsize=8)
            ax.tick_params(labelsize=7)
            if i == 0 and j == 0:
                ax.legend(fontsize=7)
    fig.suptitle(
        "Comb-scale scan in order space (no peak-search window). "
        "Red dashed = -0.54 % low-k estimate",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(FIGS / "F10_combscan.png", dpi=130)
    print(f"[combscan] wrote {FIGS / 'F10_combscan.png'} and combscan.json")
    _ = show


if __name__ == "__main__":
    main()
