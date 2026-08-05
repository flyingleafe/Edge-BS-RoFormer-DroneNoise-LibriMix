#!/usr/bin/env python3
"""High-k displacement with SHORT segments — the coherence-time fix.

Spectral autocorrelation of a DREGON cruise window shows a 172 Hz comb (the
blade-passage rate, 2 x the 86 rev/s shaft) alive at 5.5-6.5 kHz on 0.1 s
segments and GONE by 1 s. So the high-k line exists; it just has a coherence
time far shorter than the 1 s floor of ``measure_displacement.seg_len_env`` and
than the 16 s order spectrum. Both earlier "no high-k line" readings are
coherence-limited, not evidence of absence.

Here the window is split into SHORT phase-resampled segments (default 0.25 s ~
20 revolutions, order resolution 0.05), each segment's order spectrum is scanned
over a comb-scale grid, and the scores are averaged INCOHERENTLY over segments.
Resolution 0.05 orders against a 0.0054 x 75 = 0.4 order effect, so the high-k
displacement is measurable if the line is there.

Null: the same scan on the half-integer comb, where no rotor line can exist.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import scipy.io  # noqa: E402
import soundfile as sf  # noqa: E402

ROOT = Path("/home/flyingleafe/Research/PhD/projects/harmonic-noise-suppression")
OUT = Path(__file__).resolve().parent
FIGS = OUT / "figs"
DREGON = ROOT / "data/DREGON"

SPR = 512
SEG_S = 0.25
S_GRID = np.arange(0.985, 1.01501, 0.00005)
BANDS = {
    "k2-13": (2, 13),
    "k14-30": (14, 30),
    "k31-50": (31, 50),
    "k51-75": (51, 75),
    "k76-100": (76, 100),
}
WINDOWS = [
    ("nosource_w01", "free-flight_nosource_room1", 22.565),
    ("speech-low_w01", "free-flight_speech-low_room1", 26.0),
    ("whitenoise-low_w01", "free-flight_whitenoise-low_room1", 28.0),
]
WIN_S = 16.0


def load(rid: str):
    d = DREGON / f"DREGON_{rid}"
    mat = scipy.io.loadmat(str(d / f"DREGON_{rid}_motors.mat"))["motor"]
    ts = mat["timestamps"][0, 0].flatten().astype(np.float64)
    vals = mat["measured"][0, 0].astype(np.float64).T
    t0 = float(
        scipy.io.loadmat(str(d / f"DREGON_{rid}_audiots.mat"))["audio_timestamps"].flatten()[0]
    )
    x, sr = sf.read(str(d / f"DREGON_{rid}.wav"), always_2d=True)
    return x.T.astype(np.float64), sr, ts - t0, vals


def seg_scores(audio, sr, t_tel, rate, t0, t1, k_lo, k_hi, half):
    """Average over short segments of the comb score S(s)."""
    a0, a1 = int(t0 * sr), int(t1 * sr)
    t = np.arange(a0, a1) / sr
    r = np.interp(t, t_tel, rate)
    phi = np.cumsum(r) / sr
    phi -= phi[0]
    n_seg = int(SEG_S * sr)
    ks = np.arange(k_lo, k_hi + 1, dtype=np.float64) + (0.5 if half else 0.0)
    f_lim = 0.45 * sr
    acc = np.zeros(len(S_GRID))
    n_used = 0
    for s0 in range(0, (a1 - a0) - n_seg, n_seg):
        ph = phi[s0 : s0 + n_seg]
        n_rev = float(ph[-1] - ph[0])
        n_out = int(n_rev * SPR)
        if n_out < 64:
            continue
        grid = ph[0] + np.arange(n_out) / SPR
        t_at = np.interp(grid, phi, t)
        win = np.hanning(n_out)
        p = None
        for c in range(audio.shape[0]):
            y = np.interp(t_at, t, audio[c, a0:a1])
            Y = np.abs(np.fft.rfft((y - y.mean()) * win)) ** 2
            p = Y if p is None else p + Y
        orders = np.fft.rfftfreq(n_out, d=1.0 / SPR)
        db = 10.0 * np.log10(p / audio.shape[0] + 1e-30)
        from scipy.ndimage import median_filter

        n_sm = max(11, int(4.0 / (orders[1] - orders[0])) | 1)
        ex = db - median_filter(db, size=n_sm, mode="nearest")
        do = orders[1] - orders[0]
        rbar = float(np.mean(r[s0 : s0 + n_seg]))
        tot = np.zeros(len(S_GRID))
        cnt = np.zeros(len(S_GRID))
        for k in ks:
            o = S_GRID * k
            ok = (o < orders[-1]) & (k * rbar < f_lim)
            idx = np.clip(np.round(o / do).astype(int), 0, len(ex) - 1)
            tot += np.where(ok, ex[idx], 0.0)
            cnt += ok
        if cnt.max() < 1:
            continue
        acc += tot / np.maximum(cnt, 1)
        n_used += 1
    return (acc / max(n_used, 1)), n_used


def summarize(sc, null=None):
    """Peak location + height.

    ``peak_over_null`` is max(on) - max(null): a fair contest between two
    IDENTICAL searches. The earlier peak-minus-own-median statistic was wrong —
    the on-comb peak is broad (it spans most of the +-1.5 % grid), so its own
    median sits inside the peak and the excess collapses to nothing.
    """
    j = int(np.argmax(sc))
    out = {
        "pct": round((float(S_GRID[j]) - 1.0) * 100, 4),
        "peak_db": round(float(sc[j]), 3),
        "median_db": round(float(np.median(sc)), 3),
    }
    if null is not None:
        out["peak_over_null_db"] = round(float(sc[j]) - float(np.max(null)), 3)
        out["mean_over_null_db"] = round(float(np.mean(sc)) - float(np.mean(null)), 3)
    return out


def main() -> None:
    FIGS.mkdir(exist_ok=True)
    res: dict = {}
    curves: dict = {}
    for label, rid, t0 in WINDOWS:
        audio, sr, tt, vals = load(rid)
        for rot in range(4):
            rbar = float(np.mean(np.interp([t0, t0 + WIN_S], tt, vals[rot])))
            key = f"{label}__r{rot}"
            res[key] = {"rate": round(rbar, 2), "bands": {}}
            for bn, (lo, hi) in BANDS.items():
                on, n_used = seg_scores(audio, sr, tt, vals[rot], t0, t0 + WIN_S, lo, hi, False)
                off, _ = seg_scores(audio, sr, tt, vals[rot], t0, t0 + WIN_S, lo, hi, True)
                res[key]["bands"][bn] = {
                    "on": summarize(on, off),
                    "null": summarize(off),
                    "n_segments": n_used,
                }
                curves[f"{key}__{bn}"] = (on, off)
            b = res[key]["bands"]
            print(
                f"{key:24s} r={rbar:6.2f} "
                + "  ".join(
                    f"{n}: {b[n]['on']['pct']:+.3f}% "
                    f"(pk-null {b[n]['on']['peak_over_null_db']:+.2f}, "
                    f"mean {b[n]['on']['mean_over_null_db']:+.2f})"
                    for n in b
                ),
                flush=True,
            )
    (OUT / "shortscan.json").write_text(json.dumps(res, indent=1))
    np.savez_compressed(
        OUT / "shortscan_curves.npz",
        s_grid=S_GRID,
        **{f"{k}__on": v[0] for k, v in curves.items()},
        **{f"{k}__off": v[1] for k, v in curves.items()},
    )

    bands = list(BANDS)
    fig, axes = plt.subplots(3, len(bands), figsize=(3.6 * len(bands), 8.6), squeeze=False)
    for i, (label, _, _) in enumerate(WINDOWS):
        for j, bn in enumerate(bands):
            ax = axes[i][j]
            key = f"{label}__r0__{bn}"
            if key not in curves:
                ax.set_visible(False)
                continue
            on, off = curves[key]
            ax.plot((S_GRID - 1) * 100, on, lw=0.9, color="#1f4e79", label="on-comb")
            ax.plot((S_GRID - 1) * 100, off, lw=0.9, color="0.6", label="half-integer null")
            ax.axvline(0, color="k", lw=0.7)
            ax.axvline(-0.542, color="crimson", ls="--", lw=1.0)
            if i == 0:
                ax.set_title(bn, fontsize=10)
            if j == 0:
                ax.set_ylabel(label, fontsize=8)
            ax.set_xlabel("s - 1 [%]", fontsize=8)
            ax.tick_params(labelsize=7)
            if i == 0 and j == 0:
                ax.legend(fontsize=7)
    fig.suptitle(
        f"F11 - comb-scale scan on {SEG_S} s segments (rotor 0). "
        "Red dashed = -0.542 % measured at low k",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(FIGS / "F11_shortscan.png", dpi=130)
    print(f"[shortscan] wrote {FIGS / 'F11_shortscan.png'}")


if __name__ == "__main__":
    main()
