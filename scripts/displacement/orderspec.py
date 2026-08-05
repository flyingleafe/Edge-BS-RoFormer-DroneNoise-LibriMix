#!/usr/bin/env python3
"""Order spectrum: resample the audio uniformly in ROTOR PHASE, then FFT.

This is the search-window-free look at the question. If the acoustic comb sits
exactly on the telemetry, every harmonic lands on an INTEGER order. If the comb
is displaced by a factor s, harmonic k lands at order s*k — a fan that opens
linearly with k. Nothing here uses a peak-search window, a band, or a
collision gate, so it cannot inherit the min(1.5k, 8) Hz bug.

Uses NATIVE 44.1 kHz DREGON audio (order Nyquist ~275 at 80 rev/s), so it also
answers "how far up does the comb actually go".

Also does the cheap telemetry cross-check: per-rotor regression of
``motors_measured`` against ``motors_command`` on every DREGON recording that
has both.
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

ROOT = Path("/home/flyingleafe/Research/PhD/projects/harmonic-noise-suppression")
OUT = Path(__file__).resolve().parent
FIGS = OUT / "figs"
sys.path.insert(0, str(ROOT / "src"))

DREGON = ROOT / "data/DREGON"
SAMPLES_PER_REV = 1024  # order Nyquist = 512
ORDER_MAX = 120


def load_dregon(rid: str, entry: str = "measured"):
    d = DREGON / f"DREGON_{rid}"
    mat = scipy.io.loadmat(str(d / f"DREGON_{rid}_motors.mat"))["motor"]
    ts = mat["timestamps"][0, 0].flatten().astype(np.float64)
    vals = mat[entry][0, 0].astype(np.float64).T
    t0 = float(
        scipy.io.loadmat(str(d / f"DREGON_{rid}_audiots.mat"))["audio_timestamps"].flatten()[0]
    )
    x, sr = sf.read(str(d / f"DREGON_{rid}.wav"), always_2d=True)
    return x.T.astype(np.float64), sr, ts - t0, vals


def order_spectrum(
    audio: np.ndarray, sr: int, t_tel: np.ndarray, rate: np.ndarray, t0: float, t1: float
) -> tuple[np.ndarray, np.ndarray]:
    """(orders, power_db) of ``audio`` over [t0, t1] resampled in rotor phase."""
    a0, a1 = int(t0 * sr), int(t1 * sr)
    t = np.arange(a0, a1) / sr
    r = np.interp(t, t_tel, rate)
    phi = np.cumsum(r) / sr  # revolutions elapsed
    phi -= phi[0]
    n_rev = float(phi[-1])
    n_out = int(n_rev * SAMPLES_PER_REV)
    grid = np.arange(n_out) / SAMPLES_PER_REV
    t_at = np.interp(grid, phi, t)  # invert phi(t)
    win = np.hanning(n_out)
    acc = None
    for c in range(audio.shape[0]):
        y = np.interp(t_at, t, audio[c, a0:a1])
        Y = np.abs(np.fft.rfft((y - y.mean()) * win)) ** 2
        acc = Y if acc is None else acc + Y
    orders = np.fft.rfftfreq(n_out, d=1.0 / SAMPLES_PER_REV)
    p = acc / audio.shape[0]
    return orders, 10.0 * np.log10(p / np.median(p) + 1e-30)


def peak_orders(orders: np.ndarray, db: np.ndarray, k_lo: int, k_hi: int, tol: float = 0.35):
    """For each integer k in [k_lo, k_hi], the local peak order within +-tol."""
    rows = []
    for k in range(k_lo, k_hi + 1):
        m = np.abs(orders - k) <= tol
        if m.sum() < 5:
            continue
        j = int(np.argmax(db[m]))
        sub_o, sub_d = orders[m], db[m]
        # local floor = median over a +-0.5 order neighbourhood
        mf = np.abs(orders - k) <= 0.5
        floor = float(np.median(db[mf]))
        rows.append((k, float(sub_o[j]), float(sub_d[j] - floor), float(sub_o[j] / k)))
    return rows


def telemetry_crosscheck() -> dict:
    """Per-rotor regression measured = a * command + b on the cruise part."""
    out = {}
    for rid in sorted(p.name[7:] for p in DREGON.glob("DREGON_free-flight_*")):
        d = DREGON / f"DREGON_{rid}"
        f = d / f"DREGON_{rid}_motors.mat"
        if not f.exists():
            continue
        mat = scipy.io.loadmat(str(f))["motor"]
        if "measured" not in mat.dtype.names:
            continue
        me = mat["measured"][0, 0].astype(np.float64)
        cm = mat["command"][0, 0].astype(np.float64)
        ok = (me > 50).all(1) & (cm > 50).all(1)
        rec = {"n": int(ok.sum()), "rotors": []}
        for r in range(4):
            x, y = cm[ok, r], me[ok, r]
            a, b = np.polyfit(x, y, 1)
            pred = a * x + b
            r2 = 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)
            rec["rotors"].append(
                {
                    "slope": round(float(a), 5),
                    "intercept": round(float(b), 4),
                    "r2": round(float(r2), 5),
                    "mean_ratio": round(float((y / x).mean()), 6),
                    "mean_diff_rev_s": round(float((y - x).mean()), 4),
                }
            )
        out[rid] = rec
    return out


def main() -> None:
    FIGS.mkdir(exist_ok=True)
    res: dict = {"telemetry_measured_vs_command": telemetry_crosscheck()}
    print(json.dumps(res["telemetry_measured_vs_command"], indent=1)[:1200])

    panels = []
    # DREGON: the frozen cruise window of nosource_room1 (w01 = 22.565 s + 16 s)
    audio, sr, tt, vals = load_dregon("free-flight_nosource_room1", "measured")
    for rot in (0,):
        o, db = order_spectrum(audio, sr, tt, vals[rot], 22.565, 38.565)
        pk = peak_orders(o, db, 1, ORDER_MAX)
        panels.append(("DREGON free-flight_nosource_room1 w01 (measured, rotor 0)", o, db, pk))
        res["dregon_nosource_w01_rotor0"] = {
            "mean_rate": round(float(np.interp(30.5, tt, vals[rot])), 3),
            "peaks": [
                {"k": k, "order": round(op, 4), "prom_db": round(pr, 2), "ratio": round(ra, 5)}
                for k, op, pr, ra in pk
            ],
        }
    # a second DREGON recording, and the same with the COMMAND channel
    audio2, sr2, tt2, vals2 = load_dregon("free-flight_whitenoise-low_room1", "measured")
    o2, db2 = order_spectrum(audio2, sr2, tt2, vals2[0], 25.0, 41.0)
    panels.append(
        (
            "DREGON free-flight_whitenoise-low_room1 (measured, rotor 0)",
            o2,
            db2,
            peak_orders(o2, db2, 1, ORDER_MAX),
        )
    )
    res["dregon_whitenoise_rotor0"] = {
        "peaks": [
            {"k": k, "order": round(op, 4), "prom_db": round(pr, 2), "ratio": round(ra, 5)}
            for k, op, pr, ra in peak_orders(o2, db2, 1, ORDER_MAX)
        ]
    }

    fig, axes = plt.subplots(len(panels), 1, figsize=(15, 4.2 * len(panels)))
    axes = np.atleast_1d(axes)
    for ax, (title, o, db, pk) in zip(axes, panels):
        m = o <= ORDER_MAX
        ax.plot(o[m], db[m], lw=0.5, color="#1f4e79")
        for k in range(1, ORDER_MAX + 1):
            ax.axvline(k, color="0.75", lw=0.4, zorder=0)
        good = [(k, op, pr) for k, op, pr, _ in pk if pr >= 6.0]
        if good:
            ax.scatter(
                [g[1] for g in good],
                [db[np.argmin(np.abs(o - g[1]))] for g in good],
                s=14,
                color="crimson",
                zorder=5,
                label=f"{len(good)} peaks >= 6 dB",
            )
            ax.legend(loc="upper right", fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("order (harmonic of the telemetry shaft rate)")
        ax.set_ylabel("dB over in-band median")
        ax.set_xlim(0, ORDER_MAX)
    fig.tight_layout()
    fig.savefig(FIGS / "F8_order_spectrum.png", dpi=130)
    print(f"[orderspec] wrote {FIGS / 'F8_order_spectrum.png'}")

    # ratio-vs-k fan plot
    fig2, ax = plt.subplots(figsize=(9, 5))
    for name, key in (
        ("nosource w01", "dregon_nosource_w01_rotor0"),
        ("whitenoise-low", "dregon_whitenoise_rotor0"),
    ):
        pk = res[key]["peaks"]
        ks = [p["k"] for p in pk if p["prom_db"] >= 6]
        rs = [p["ratio"] for p in pk if p["prom_db"] >= 6]
        ax.scatter(ks, [100 * (r - 1) for r in rs], s=18, label=f"{name} (>=6 dB)")
    ax.axhline(0, color="k", lw=0.8)
    ax.axhline(-0.54, color="crimson", ls="--", lw=1.0, label="-0.54 % (low-k estimate)")
    ax.set_xlabel("harmonic k")
    ax.set_ylabel("peak order / k - 1  [%]")
    ax.set_ylim(-3, 3)
    ax.legend(fontsize=8)
    ax.set_title("Displacement of each harmonic, order-spectrum peaks (no search window)")
    fig2.tight_layout()
    fig2.savefig(FIGS / "F9_order_ratio.png", dpi=130)
    print(f"[orderspec] wrote {FIGS / 'F9_order_ratio.png'}")

    (OUT / "orderspec.json").write_text(json.dumps(res, indent=1))
    for key in ("dregon_nosource_w01_rotor0", "dregon_whitenoise_rotor0"):
        pk = [p for p in res[key]["peaks"] if p["prom_db"] >= 6]
        print(f"\n{key}: {len(pk)} peaks >= 6 dB up to k={ORDER_MAX}")
        print("  k / order / prom_db / ratio")
        for p in pk[:60]:
            print(f"   {p['k']:3d} {p['order']:8.3f} {p['prom_db']:6.2f} {p['ratio']:.5f}")


if __name__ == "__main__":
    main()
