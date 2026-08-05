#!/usr/bin/env python3
"""Regime ladder for the DREGON comb displacement: is the -0.54 % everywhere?

Extends the `nullcontrol.py` measurement (same demod bank, same per-harmonic
band, same corrected ``carrier_collision_mask`` gate, same >=6 dB prominence
bar) from the 3 frozen cruise recordings to EVERY DREGON recording that has
telemetry, built straight from the local raw tree so the room2 recordings and
the low-rate warm-up windows are included too.

Regimes (the ladder):
  hover        hovering_nosource_room2                        (command telemetry)
  translate    free-flight_* room1 + room2                    (measured / command)
  maneuver     updown / rectangle / spinning room2            (command telemetry)
and, cutting across them, warmup (mean rps < 45) vs cruise windows.

For every (window, rotor, k) unit that clears the bar the script records the
telemetry rate r, the acoustic offset delta and delta/r. Per regime it then
fits three models over the bar-clearing units:
  free    acoustic = a * r + b       (a scale error: a != 1, b ~ 0)
  prop    acoustic = a * r           (pure multiplicative)
  quad    delta    = -c * r^2        (a FIXED tick miscount in a reciprocal-
                                      period counter: delta/r grows with r)
with a block bootstrap over windows for the CIs.
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

ROOT = Path("/home/flyingleafe/Research/PhD/projects/harmonic-noise-suppression")
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT))
sys.path.insert(0, str(ROOT / "src"))

import measure_displacement as M  # noqa: E402
import nullcontrol as NC  # noqa: E402

PREP2 = OUT / "prep_ladder"
SR = M.SR
FRAME_S = M.FRAME_S
WINDOW_S = 16.0
N_ROTORS = 4
K_SET = list(range(1, 17))  # low-k only: nothing above k~14 clears the bar
BAR_DB = 6.0
LOW_K = tuple(range(2, 14))

DREGON_DIR = ROOT / "data/DREGON"

# recording -> (regime family, telemetry entry)
RECORDINGS: dict[str, tuple[str, str]] = {
    "free-flight_nosource_room1": ("translate", "measured"),
    "free-flight_speech-low_room1": ("translate", "measured"),
    "free-flight_speech-high_room1": ("translate", "measured"),
    "free-flight_whitenoise-low_room1": ("translate", "measured"),
    "free-flight_whitenoise-high_room1": ("translate", "measured"),
    "free-flight_nosource_room2": ("translate", "command"),
    "hovering_nosource_room2": ("hover", "command"),
    "updown_nosource_room2": ("maneuver", "command"),
    "rectangle_nosource_room2": ("maneuver", "command"),
    "spinning_nosource_room2": ("maneuver", "command"),
}
# the five room1 recordings are ALSO measured against the command channel, so
# the two telemetry channels can be compared on identical audio
DUAL = [r for r, (_, e) in RECORDINGS.items() if e == "measured"]


# ─── prep (raw tree -> 16 s windows at 16 kHz + telemetry on the frame grid) ──


def build_preps() -> list[dict[str, Any]]:
    """Materialize `prep_ladder/<rid>__<entry>__wNN.npz`; return the job list."""
    import librosa
    import scipy.io
    import soundfile as sf

    from data_processing.sources.dregon import clean_command_spikes

    PREP2.mkdir(parents=True, exist_ok=True)
    jobs: list[dict[str, Any]] = []
    for rid, (family, entry) in RECORDINGS.items():
        rec_dir = DREGON_DIR / f"DREGON_{rid}"
        mat = scipy.io.loadmat(str(rec_dir / f"DREGON_{rid}_motors.mat"))["motor"]
        ts = mat["timestamps"][0, 0].flatten().astype(np.float64)
        # Time zero is the FIRST AUDIO SAMPLE, exactly as `_beatvk_frame`
        # re-anchors it. Telemetry logging starts 4-5.5 s after the audio on
        # every DREGON recording, so anchoring on ts[0] misaligns the two by
        # that much and the measurement becomes meaningless.
        t_anchor = float(
            scipy.io.loadmat(str(rec_dir / f"DREGON_{rid}_audiots.mat"))[
                "audio_timestamps"
            ].flatten()[0]
        )
        entries = [entry] + (["command"] if rid in DUAL else [])
        audio16 = None
        for ent in entries:
            if ent not in mat.dtype.names:
                continue
            vals = mat[ent][0, 0].astype(np.float64).T  # (4, M)
            if ent == "command":
                vals = clean_command_spikes(vals)
            # live span: drop leading/trailing exact-constant logger runs
            same = np.all(vals[:, 1:] == vals[:, :-1], axis=0)
            lead = int(np.argmin(same)) if same.any() else 0
            trail = int(np.argmin(same[::-1])) if same.any() else 0
            t0 = float(ts[lead]) - t_anchor
            t1 = float(ts[len(ts) - 1 - trail]) - t_anchor
            tsr = ts - t_anchor
            if audio16 is None:
                x, sr = sf.read(str(rec_dir / f"DREGON_{rid}.wav"), always_2d=True)
                audio16 = librosa.resample(
                    x.T.astype(np.float32), orig_sr=sr, target_sr=SR, axis=-1, res_type="soxr_hq"
                )
            dur = audio16.shape[-1] / SR
            start = max(0.0, t0)
            widx = 0
            while start + WINDOW_S <= min(t1, dur) + 1e-9:
                a0, a1 = int(round(start * SR)), int(round((start + WINDOW_S) * SR))
                ft = np.arange(0.0, (a1 - a0) / SR - FRAME_S / 2, FRAME_S)
                r_ft = np.stack([np.interp(ft + start, tsr, vals[i]) for i in range(N_ROTORS)])
                mean_rps = float(np.mean(r_ft))
                regime = "ground" if mean_rps < 5 else ("warmup" if mean_rps < 45 else "cruise")
                key = f"{rid}__{ent}__w{widx:02d}"
                p = PREP2 / f"{key}.npz"
                if not p.exists():
                    np.savez_compressed(
                        p,
                        audio=audio16[:, a0:a1],
                        ft=ft,
                        r_meas=r_ft,
                        regime=np.str_(regime),
                        start_s=np.float64(start),
                    )
                jobs.append(
                    {
                        "key": key,
                        "recording": rid,
                        "entry": ent,
                        "family": family,
                        "widx": widx,
                        "regime": regime,
                        "mean_rps": round(mean_rps, 3),
                    }
                )
                widx += 1
                start += WINDOW_S
        print(
            f"[prep] {rid}: {sum(j['recording'] == rid for j in jobs)} window-variants", flush=True
        )
    return jobs


# ─── measurement (identical estimator to nullcontrol.py, on-comb only) ────────


def measure(job: dict[str, Any]) -> dict[str, Any]:
    with np.load(PREP2 / f"{job['key']}.npz") as z:
        audio = np.asarray(z["audio"], np.float64)
        ft = np.asarray(z["ft"], np.float64)
        r_ft = np.asarray(z["r_meas"], np.float64)
    units: list[dict[str, Any]] = []
    for rot in range(N_ROTORS):
        r_bar = float(np.mean(r_ft[rot]))
        if r_bar < 10.0:
            continue
        z_on, band_hz_k = NC.bank(audio, r_ft[rot], ft, K_SET, half=False)
        clean = ~NC.carrier_collision_mask(r_ft, r_ft[rot], rot, K_SET, half=False)
        for a, k in enumerate(K_SET):
            tf, delta, snr, spec_db, rev_axis = M.ridge_from_envelope(
                z_on[:, a], float(band_hz_k[a]), k
            )
            keep_f = np.interp(tf, ft, clean[a].astype(float)) > 0.999
            w = np.where(keep_f, np.maximum(snr - 1.0, 0.0), 0.0)
            m, sd, n_eff = M.weighted_stats(delta, w)
            prom, _ = NC.profile_prominence(spec_db, rev_axis, keep_f, k, float(band_hz_k[a]))
            if not np.isfinite(m) or not np.isfinite(prom):
                continue
            units.append(
                {
                    "rotor": rot,
                    "k": k,
                    "r_bar": round(r_bar, 4),
                    "delta": round(float(m), 5),
                    "prom_db": round(float(prom), 3),
                    "n_eff": round(float(n_eff), 2),
                    "keep": round(float(np.mean(keep_f)), 3),
                }
            )
    return {**job, "units": units}


# ─── fits ─────────────────────────────────────────────────────────────────────


def fits(rows: list[dict[str, Any]], n_boot: int = 2000, seed: int = 0) -> dict[str, Any]:
    """Slope / intercept / R^2 of acoustic vs telemetry rate + block bootstrap.

    ``rows`` are bar-clearing units carrying ``window`` (the bootstrap block),
    ``r`` (telemetry rate) and ``a`` (acoustic rate = r + delta).
    """
    if len(rows) < 6:
        return {"n_units": len(rows), "note": "too few bar-clearing units"}
    win = np.array([r["window"] for r in rows])
    r = np.array([r["r"] for r in rows], dtype=np.float64)
    ac = np.array([r["a"] for r in rows], dtype=np.float64)
    d = ac - r

    def one(rr: np.ndarray, aa: np.ndarray) -> tuple[float, float, float, float, float]:
        A = np.polyfit(rr, aa, 1)
        pred = np.polyval(A, rr)
        ss = 1.0 - np.sum((aa - pred) ** 2) / max(np.sum((aa - aa.mean()) ** 2), 1e-30)
        slope_prop = float(np.sum(rr * aa) / np.sum(rr * rr))  # through origin
        dd = aa - rr
        c_quad = float(-np.sum(rr**2 * dd) / np.sum(rr**4))  # delta = -c r^2
        return float(A[0]), float(A[1]), float(ss), slope_prop, c_quad

    a0, b0, r2, ap, cq = one(r, ac)
    blocks = np.unique(win)
    rng = np.random.default_rng(seed)
    boot: list[tuple[float, float, float, float]] = []
    for _ in range(n_boot):
        sel = rng.choice(blocks, len(blocks), replace=True)
        idx = np.concatenate([np.flatnonzero(win == b) for b in sel])
        if len(np.unique(np.round(r[idx], 1))) < 3:
            continue
        aa, bb, _, pp, cc = one(r[idx], ac[idx])
        boot.append((aa, bb, pp, cc))
    B = np.array(boot) if boot else np.zeros((1, 4))

    def ci(col: int) -> list[float]:
        return [
            round(float(np.percentile(B[:, col], 2.5)), 6),
            round(float(np.percentile(B[:, col], 97.5)), 6),
        ]

    # per-unit fractional offset, and the two model residual scatters
    frac = d / r
    return {
        "n_units": len(rows),
        "n_windows": int(len(blocks)),
        "rate_range": [round(float(r.min()), 2), round(float(r.max()), 2)],
        "mean_delta_rev_s": round(float(d.mean()), 4),
        "mean_frac_pct": round(float(frac.mean() * 100), 4),
        "sem_frac_pct": round(float(frac.std(ddof=1) / np.sqrt(len(frac)) * 100), 4),
        "slope_free": round(a0, 6),
        "slope_free_ci": ci(0),
        "intercept_free": round(b0, 4),
        "intercept_free_ci": ci(1),
        "r2": round(r2, 5),
        "slope_prop": round(ap, 6),
        "slope_prop_ci": ci(2),
        "scale_error_pct": round((ap - 1.0) * 100, 4),
        "scale_error_pct_ci": [round((c - 1.0) * 100, 4) for c in ci(2)],
        "quad_c": round(cq, 8),
        "quad_c_ci": ci(3),
        "rss_prop": round(float(np.sum((d - (ap - 1.0) * r) ** 2)), 4),
        "rss_quad": round(float(np.sum((d + cq * r**2) ** 2)), 4),
    }


def main() -> None:
    jobs = build_preps()
    print(f"[ladder] {len(jobs)} window-variants", flush=True)
    with ProcessPoolExecutor(max_workers=8) as pool:
        rows = list(pool.map(measure, jobs))
    (OUT / "ladder_raw.json").write_text(json.dumps(rows, indent=1))

    # ── pool the bar-clearing low-k units ────────────────────────────────────
    def collect(pred) -> list[dict[str, Any]]:
        out = []
        for w in rows:
            if not pred(w):
                continue
            for u in w["units"]:
                if u["k"] in LOW_K and u["prom_db"] >= BAR_DB:
                    out.append(
                        {
                            "window": w["key"],
                            "r": u["r_bar"],
                            "a": u["r_bar"] + u["delta"],
                            "k": u["k"],
                            "rotor": u["rotor"],
                        }
                    )
        return out

    groups: dict[str, Any] = {}
    prim = lambda w: w["entry"] == RECORDINGS[w["recording"]][1]  # noqa: E731
    specs = {
        "all_dregon_cruise": lambda w: prim(w) and w["regime"] == "cruise",
        "translate_cruise": lambda w: prim(w)
        and w["family"] == "translate"
        and w["regime"] == "cruise",
        "hover_cruise": lambda w: prim(w) and w["family"] == "hover" and w["regime"] == "cruise",
        "maneuver_cruise": lambda w: prim(w)
        and w["family"] == "maneuver"
        and w["regime"] == "cruise",
        "warmup_lowrate": lambda w: prim(w) and w["regime"] == "warmup",
        "room1_measured_cruise": lambda w: w["entry"] == "measured" and w["regime"] == "cruise",
        "room1_command_cruise": lambda w: (
            w["recording"] in DUAL and w["entry"] == "command" and w["regime"] == "cruise"
        ),
    }
    for name, pred in specs.items():
        sel = collect(pred)
        groups[name] = fits(sel)
        groups[name]["units"] = sel
        n = groups[name].get("n_units", 0)
        if "rate_range" in groups[name]:
            print(
                f"  {name:24s} n={n:4d} rate {groups[name]['rate_range']} "
                f"frac {groups[name]['mean_frac_pct']:+.3f} % "
                f"slope_prop {groups[name]['slope_prop']:.5f} "
                f"free {groups[name]['slope_free']:.5f} b={groups[name]['intercept_free']:+.2f}",
                flush=True,
            )
        else:
            print(f"  {name:24s} n=0", flush=True)

    # per-recording summary
    per_rec: dict[str, Any] = {}
    for rid in RECORDINGS:
        sel = collect(lambda w, rid=rid: prim(w) and w["recording"] == rid)
        f = fits(sel, n_boot=400)
        f.pop("units", None)
        per_rec[rid] = {k: v for k, v in f.items() if k != "units"}

    (OUT / "ladder.json").write_text(
        json.dumps(
            {
                "protocol": {
                    "window_s": WINDOW_S,
                    "k_set": K_SET,
                    "low_k": list(LOW_K),
                    "bar_db": BAR_DB,
                    "estimator": "nullcontrol.py on-comb peak-pick, corrected "
                    "carrier_collision_mask gate, SNR-weighted over uncollided frames",
                    "telemetry": {r: e for r, (_, e) in RECORDINGS.items()},
                },
                "windows": [{k: v for k, v in w.items() if k != "units"} for w in rows],
                "groups": groups,
                "per_recording": per_rec,
            },
            indent=1,
        )
    )
    print(f"[ladder] wrote {OUT / 'ladder.json'}")


if __name__ == "__main__":
    main()
