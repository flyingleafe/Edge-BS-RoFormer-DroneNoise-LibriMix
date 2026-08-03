#!/usr/bin/env python3
"""Calibrate the control-mode increment scales of a quadrotor's rotor speeds.

Why this exists
---------------
A proposed joint 4-rotor tracker (scratchpad design "joint Viterbi / beam
search", 2026-07-31) wanted to replace the coarse stage's single shared
trajectory ``c(t)`` with a genuinely joint search over the full speed vector
``w_t = (w1, w2, w3, w4)``, made tractable by a transition prior that is CHEAP
along the common mode and EXPENSIVE along the differential modes:

    dm = B^T dw / 4,        T(w_t | w_{t-1}) = sum_i psi_i(dm_i / sigma_i)

with ``B = rps_synthesis.MIXER`` (columns [common, roll, pitch, yaw],
``B^T B = 4 I``).  The design's own precondition was
``sigma_common / sigma_diff ~ 3-10``, measured from real telemetry, with an
explicit instruction to stop if it came out near 1.

It comes out near 1.  This script is that measurement, kept so the number can
be re-derived and challenged.  See ``docs/experiments/rps-refine-precision.md``
§ WP16.

What it measures
----------------
For every recording with rotor telemetry (DREGON ``motors_measured`` /
``motors_command`` at ~1 kHz nominal, Michael's calibrated FLY124/FLY125 at
~29.5 Hz), on the pipeline's own 32 ms frame grid:

  * per-mode robust scale (MAD x 1.4826) of the per-frame increments, split
    into cruise (every rotor >= 50 rev/s) and ramp (takeoff/landing) regimes;
  * the same at longer lags, because a WHITE label-noise floor contributes a
    lag-independent term to every mode equally and therefore drives the ratio
    towards 1 all by itself;
  * per-mode AMPLITUDE standard deviations over the cruise segment (the
    trajectory-level anisotropy, which is a different quantity);
  * ramp excursions per mode (how common-mode a takeoff really is);
  * the telemetry's own quantisation lattice, which turns out to dominate
    DREGON's 32 ms increments.

Run:
    PYTHONPATH=src .venv/bin/python scripts/mode_covariance_calib.py
    PYTHONPATH=src .venv/bin/python scripts/mode_covariance_calib.py --json out.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "4")

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
os.chdir(REPO)

import numpy as np  # noqa: E402
import scipy.io  # noqa: E402
from scipy.signal import filtfilt, firwin  # noqa: E402

from data_processing.rps_synthesis import MIXER  # noqa: E402
from data_processing.sources.michaels import (  # noqa: E402
    MICHAELS_FILES,
    load_raw_aligned,
)

#: The scorer's fixed evaluation grid (= 512 / 16000), the grid every RPS
#: trajectory in this project lives on.
FRAME_S = 0.032
MODES = ("common", "roll", "pitch", "yaw")
#: Increment lags to report, in frames (0.032 - 1.024 s).
LAGS = (1, 2, 4, 8, 16, 32)
#: Shaft band-limits to test.  A real rotor's inertia cannot follow a white
#: drive (WP4 item 5), so the band-limited increment scale is the physical one.
FCS = (12.0, 8.0, 5.0, 2.0)
#: A rotor is "in stable flight" above this (the `min_motor_rps 50` rule that
#: cleaned the V4-michaels valid set).
CRUISE_MIN_RPS = 50.0


def robust_scale(x: np.ndarray) -> float:
    """MAD x 1.4826 — the Gaussian-consistent robust scale."""
    return float(np.median(np.abs(x - np.median(x))) * 1.4826)


def to_modes(w: np.ndarray) -> np.ndarray:
    """``(4, N)`` rotor speeds -> ``(4, N)`` mode coefficients ``B^T w / 4``."""
    return (MIXER.T @ w) / 4.0


def frame_grid(ts: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate onto the uniform 32 ms grid — the pipeline's GT convention.

    Deliberately `np.interp` with no anti-alias filter: that is exactly what
    `Prepared.r_meas` is, so this is the increment statistic the tracker would
    actually be scored against.  The band-limited rows below give the physical
    counterpart.
    """
    ft = np.arange(float(ts[0]), float(ts[-1]), FRAME_S)
    return ft, np.stack([np.interp(ft, ts, w[i]) for i in range(len(w))])


def ratio(sig: list[float]) -> tuple[float, float]:
    """``(sigma_diff_rms, sigma_common / sigma_diff_rms)`` from 4 mode scales."""
    dr = float(np.sqrt(np.mean([sig[i] ** 2 for i in (1, 2, 3)])))
    return dr, sig[0] / max(dr, 1e-12)


def regime_masks(wf: np.ndarray) -> dict[str, np.ndarray]:
    cruise = wf.min(axis=0) >= CRUISE_MIN_RPS
    ramp = (wf.mean(axis=0) > 5.0) & ~cruise
    return {"cruise": cruise, "ramp": ramp}


def longest_true(mask: np.ndarray) -> slice:
    best = (0, 0)
    cur = start = 0
    for i, v in enumerate(mask):
        if v:
            start = i if cur == 0 else start
            cur += 1
            if cur > best[1] - best[0]:
                best = (start, i + 1)
        else:
            cur = 0
    return slice(*best)


def increment_stats(wf: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    """Per-mode increment scales on `mask` (both endpoints must be in regime)."""
    dw = np.diff(wf, axis=1)
    ok = mask[:-1] & mask[1:]
    if int(ok.sum()) < 50:
        return {}
    dm = to_modes(dw[:, ok])
    sig = [robust_scale(dm[i]) for i in range(4)]
    dr, rt = ratio(sig)
    return {
        "n": int(ok.sum()),
        "sigma": {MODES[i]: round(sig[i], 4) for i in range(4)},
        "sigma_diff_rms": round(dr, 4),
        "ratio": round(rt, 3),
    }


def lag_table(m: np.ndarray) -> list[dict[str, Any]]:
    out = []
    for lag in LAGS:
        d = m[:, lag:] - m[:, :-lag]
        sig = [robust_scale(d[i]) for i in range(4)]
        dr, rt = ratio(sig)
        out.append(
            {
                "lag": lag,
                "dt_s": round(lag * FRAME_S, 3),
                "sigma": [round(s, 4) for s in sig],
                "sigma_diff_rms": round(dr, 4),
                "ratio": round(rt, 3),
            }
        )
    return out


def bandlimited_table(w: np.ndarray) -> list[dict[str, Any]]:
    """Increment scales after a zero-phase lowpass at the frame rate.

    Also reports the **per-rotor** increment scale, which is the sigma an
    *isotropic* smoothness prior would use — the fallback that survives if the
    mode-space anisotropy turns out not to exist (it does not; see WP16).
    """
    fs = 1.0 / FRAME_S
    out = []
    for fc in [None, *FCS]:
        if fc is not None and fc >= fs / 2:
            continue
        if fc is None:
            wl = w
        else:
            taps = firwin(129, fc / (fs / 2), window="hamming")
            wl = filtfilt(taps, [1.0], w, axis=1)
        d = np.diff(to_modes(wl), axis=1)
        sig = [robust_scale(d[i]) for i in range(4)]
        dr, rt = ratio(sig)
        per_rotor = [robust_scale(x) for x in np.diff(wl, axis=1)]
        out.append(
            {
                "fc_hz": fc,
                "sigma": [round(s, 4) for s in sig],
                "ratio": round(rt, 3),
                "sigma_per_rotor_mean": round(float(np.mean(per_rotor)), 4),
            }
        )
    return out


def quantisation(w: np.ndarray) -> dict[str, Any]:
    """Telemetry lattice: distinct values and the local step in the 75-90 band.

    DREGON's ``motors_measured`` is a RECIPROCAL-PERIOD lattice (a period
    counter: ``1/v`` is uniformly spaced), so its absolute resolution degrades
    as speed^2 and reaches ~0.24 rev/s at 80 rev/s — of the same order as the
    32 ms increments themselves.
    """
    u = np.unique(w)
    band = u[(u > 75.0) & (u < 90.0)]
    step = float(np.median(np.diff(band))) if len(band) > 3 else float("nan")
    inv = np.diff(np.sort(1.0 / band)) if len(band) > 3 else np.array([np.nan])
    return {
        "n_unique": int(len(u)),
        "step_at_80rps": round(step, 4),
        "reciprocal_lattice": bool(np.nanstd(inv) < 0.05 * abs(np.nanmedian(inv))),
        "change_rate_hz": None,
    }


def ramp_stats(ft: np.ndarray, wf: np.ndarray) -> dict[str, Any]:
    """Takeoff excursion per mode + the heavy tail of the common increment."""
    mean = wf.mean(axis=0)
    cru = np.flatnonzero(wf.min(axis=0) >= CRUISE_MIN_RPS)
    if len(cru) == 0:
        return {}
    t_end = int(cru[0])
    lo = np.flatnonzero(mean[:t_end] < 10.0)
    t0 = int(lo[-1]) if len(lo) else 0
    seg = slice(max(0, t0 - 5), t_end + 5)
    m = to_modes(wf)
    d = np.diff(m[:, seg], axis=1)
    sig_c_cruise = robust_scale(np.diff(m[:, cru[0] :], axis=1)[0])
    return {
        "window_s": [
            round(float(ft[seg.start] - ft[0]), 2),
            round(float(ft[seg.stop - 1] - ft[0]), 2),
        ],
        "n_frames": seg.stop - seg.start,
        "excursion": {
            MODES[i]: round(float(m[i, seg].max() - m[i, seg].min()), 2) for i in range(4)
        },
        "excursion_ratio": round(
            float(
                (m[0, seg].max() - m[0, seg].min())
                / max(
                    np.sqrt(np.mean([(m[i, seg].max() - m[i, seg].min()) ** 2 for i in (1, 2, 3)])),
                    1e-9,
                )
            ),
            2,
        ),
        "d_common_pct": {
            "p50": round(float(np.percentile(np.abs(d[0]), 50)), 3),
            "p90": round(float(np.percentile(np.abs(d[0]), 90)), 3),
            "p99": round(float(np.percentile(np.abs(d[0]), 99)), 3),
            "max": round(float(np.abs(d[0]).max()), 3),
        },
        "cruise_sigma_common": round(sig_c_cruise, 4),
        "max_d_common_in_cruise_sigmas": round(
            float(np.abs(d[0]).max() / max(sig_c_cruise, 1e-9)), 1
        ),
    }


def ou_fit(m: np.ndarray) -> dict[str, Any]:
    """Fit a 4-D OU process to the mode trajectories, immune to white label noise.

    The per-frame increment scale is NOT a usable estimator here: DREGON's
    reciprocal-period lattice puts ~0.5 rev/s of white noise into every mode
    equally (see `quantisation`), which biases tau low and does it worst for
    the small-amplitude differential modes.  White noise contributes ONLY to
    the lag-0 autocovariance, so fit the decay on lags >= `LAG_MIN`:

        c(L) = V_true * exp(-L*dt/tau)        (L >= 1)
        c(0) = V_true + V_noise

    giving tau from the slope, the noise-free level variance V_true from the
    intercept, and V_noise for free.  Returns per-mode
    ``(tau, sigma_level, sigma_noise, a, s)`` where ``a = exp(-dt/tau)`` and
    ``s = sigma_level*sqrt(1-a^2)`` is the OU innovation scale of the
    discrete-time transition the tracker uses.
    """
    lag_min, lag_max = 2, 64
    out: dict[str, Any] = {}
    for i in range(4):
        x = m[i] - m[i].mean()
        n = len(x)
        cov = np.array([float(np.dot(x[: n - L], x[L:]) / (n - L)) for L in range(lag_max + 1)])
        pos = np.arange(lag_min, lag_max + 1)
        pos = pos[cov[lag_min:] > 0.05 * max(cov[lag_min], 1e-12)]
        if len(pos) < 4:
            out[MODES[i]] = {}
            continue
        A = np.stack([np.ones(len(pos)), -pos * FRAME_S], axis=1)
        coef, *_ = np.linalg.lstsq(A, np.log(cov[pos]), rcond=None)
        v_true = float(np.exp(coef[0]))
        tau = float(1.0 / max(coef[1], 1e-9))
        a = float(np.exp(-FRAME_S / tau))
        out[MODES[i]] = {
            "tau_s": round(tau, 3),
            "sigma_level": round(float(np.sqrt(v_true)), 4),
            "sigma_noise": round(float(np.sqrt(max(cov[0] - v_true, 0.0))), 4),
            "a": round(a, 6),
            "s_innov": round(float(np.sqrt(v_true * (1.0 - a * a))), 4),
        }
    if all(out.get(k) for k in MODES):
        # tau*V is the quantity the SUSTAINED-offset discrimination depends on
        tv = {k: out[k]["tau_s"] * out[k]["sigma_level"] ** 2 for k in MODES}
        tv_d = float(np.mean([tv[k] for k in MODES[1:]]))
        r_ou = tv["common"] / max(tv_d, 1e-12)
        out["discrimination"] = {
            "tauV_common": round(tv["common"], 4),
            "tauV_diff_mean": round(tv_d, 4),
            "R_ou": round(r_ou, 2),
            "sustained_cost_one_over_four": round(1 / 16 + 3 / 16 * r_ou, 2),
        }
    return out


def analyse(tag: str, ts: np.ndarray, w: np.ndarray) -> dict[str, Any]:
    ft, wf = frame_grid(ts, w)
    masks = regime_masks(wf)
    sl = longest_true(masks["cruise"])
    m_cruise = to_modes(wf[:, sl]) if sl.stop - sl.start > 200 else None
    res: dict[str, Any] = {
        "tag": tag,
        "native_rate_hz": round(1.0 / float(np.median(np.diff(ts))), 1),
        "n_frames": int(wf.shape[1]),
        "quantisation": quantisation(w),
        "regimes": {k: increment_stats(wf, v) for k, v in masks.items()},
        "ramp": ramp_stats(ft, wf),
    }
    res["quantisation"]["change_rate_hz"] = round(
        float((np.diff(w[0]) != 0).mean()) * res["native_rate_hz"], 1
    )
    if m_cruise is not None:
        amp = [float(np.std(m_cruise[i])) for i in range(4)]
        dr, rt = ratio(amp)
        res["cruise_segment_s"] = round((sl.stop - sl.start) * FRAME_S, 1)
        res["amplitude"] = {
            "std": {MODES[i]: round(amp[i], 3) for i in range(4)},
            "diff_rms": round(dr, 3),
            "ratio": round(rt, 2),
        }
        res["ou"] = ou_fit(m_cruise)
        res["lags"] = lag_table(m_cruise)
        res["bandlimited"] = bandlimited_table(wf[:, sl])
        res["perm_ratio_range"] = [
            round(
                min(
                    ratio([robust_scale(x) for x in np.diff(to_modes(wf[list(p), sl]), axis=1)])[1]
                    for p in itertools.permutations(range(4))
                ),
                3,
            ),
            round(
                max(
                    ratio([robust_scale(x) for x in np.diff(to_modes(wf[list(p), sl]), axis=1)])[1]
                    for p in itertools.permutations(range(4))
                ),
                3,
            ),
        ]
    return res


def load_all() -> list[tuple[str, np.ndarray, np.ndarray]]:
    out: list[tuple[str, np.ndarray, np.ndarray]] = []
    for d in sorted((REPO / "data" / "DREGON").glob("DREGON_free-flight*")):
        mats = list(d.rglob("*motor*.mat"))
        if not mats:
            continue
        mm = scipy.io.loadmat(str(mats[0]))["motor"]
        ts = mm["timestamps"][0, 0].flatten().astype(np.float64)
        for key in ("measured", "command"):
            if key not in mm.dtype.names:
                continue
            out.append(
                (
                    f"DREGON/{d.name.replace('DREGON_', '')}[{key}]",
                    ts,
                    mm[key][0, 0].astype(np.float64).T,
                )
            )
    # MICHAELS_FILES paths are relative to the `recording_with_motor_speed`
    # raw root (the sources-registry convention), not to <repo>/data.
    michaels_root = REPO / "data" / "recording_with_motor_speed"
    for wav_rel, csv_rel, off, dil in MICHAELS_FILES:
        _, ts, ms, _ = load_raw_aligned(
            michaels_root / wav_rel, michaels_root / csv_rel, off, dil, sr=16000
        )
        out.append((f"michaels/{Path(csv_rel).stem}", ts, ms))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Calibrate quadrotor control-mode increment scales from telemetry."
    )
    ap.add_argument("--json", default=None, help="write the full result dict here")
    ap.add_argument(
        "--measured-only",
        action="store_true",
        help="skip DREGON motors_command (keep only the measured rotor speeds)",
    )
    args = ap.parse_args()

    results = []
    for tag, ts, w in load_all():
        if args.measured_only and "[command]" in tag:
            continue
        results.append(analyse(tag, ts, w))

    print(f"\n{'=' * 100}\nPER-FRAME (32 ms) INCREMENT SCALES, per control mode\n{'=' * 100}")
    hdr = (
        f"{'recording':<44s} {'regime':<7s} {'n':>5s} "
        f"{'common':>7s} {'roll':>7s} {'pitch':>7s} {'yaw':>7s} {'diff':>7s} {'ratio':>6s}"
    )
    print(hdr)
    for r in results:
        for reg in ("cruise", "ramp"):
            s = r["regimes"].get(reg)
            if not s:
                continue
            g = s["sigma"]
            print(
                f"{r['tag']:<44s} {reg:<7s} {s['n']:5d} "
                f"{g['common']:7.4f} {g['roll']:7.4f} {g['pitch']:7.4f} {g['yaw']:7.4f} "
                f"{s['sigma_diff_rms']:7.4f} {s['ratio']:6.2f}"
            )

    print(
        f"\n{'=' * 100}\nTRAJECTORY-AMPLITUDE scales over the cruise segment (a DIFFERENT quantity)\n{'=' * 100}"
    )
    print(
        f"{'recording':<44s} {'span_s':>7s} {'common':>7s} {'roll':>7s} {'pitch':>7s} {'yaw':>7s} {'ratio':>6s}"
    )
    for r in results:
        a = r.get("amplitude")
        if not a:
            continue
        g = a["std"]
        print(
            f"{r['tag']:<44s} {r['cruise_segment_s']:7.1f} "
            f"{g['common']:7.3f} {g['roll']:7.3f} {g['pitch']:7.3f} {g['yaw']:7.3f} {a['ratio']:6.2f}"
        )

    print(
        f"\n{'=' * 100}\nINCREMENT RATIO vs LAG and vs SHAFT BAND-LIMIT (cruise segment)\n{'=' * 100}"
    )
    for r in results:
        if "lags" not in r:
            continue
        lg = "  ".join(f"{e['dt_s']}s:{e['ratio']:.2f}" for e in r["lags"])
        bl = "  ".join(
            f"{'raw' if e['fc_hz'] is None else 'fc' + format(e['fc_hz'], 'g')}:"
            f"{e['ratio']:.2f}/pr{e['sigma_per_rotor_mean']:.2f}"
            for e in r["bandlimited"]
        )
        print(f"{r['tag']:<44s} lag[{lg}]")
        print(f"{'':<44s} bl [{bl}]   perm-range {r['perm_ratio_range']}")

    print(f"\n{'=' * 100}\nRAMP: is a takeoff a common-mode-only excursion?\n{'=' * 100}")
    print(
        f"{'recording':<44s} {'common':>7s} {'roll':>7s} {'pitch':>7s} {'yaw':>7s} "
        f"{'exc.rat':>8s} {'p99|dc|':>8s} {'max/sig':>8s}"
    )
    for r in results:
        rp = r.get("ramp")
        if not rp:
            continue
        e = rp["excursion"]
        print(
            f"{r['tag']:<44s} {e['common']:7.2f} {e['roll']:7.2f} {e['pitch']:7.2f} {e['yaw']:7.2f} "
            f"{rp['excursion_ratio']:8.2f} {rp['d_common_pct']['p99']:8.3f} "
            f"{rp['max_d_common_in_cruise_sigmas']:8.1f}"
        )

    print(
        f"\n{'=' * 100}\nTELEMETRY LATTICE (why DREGON's 32 ms increments are isotropic)\n{'=' * 100}"
    )
    print(f"{'recording':<44s} {'n_uniq':>7s} {'step@80':>8s} {'recip?':>7s} {'update_hz':>10s}")
    for r in results:
        q = r["quantisation"]
        print(
            f"{r['tag']:<44s} {q['n_unique']:7d} {q['step_at_80rps']:8.4f} "
            f"{str(q['reciprocal_lattice']):>7s} {q['change_rate_hz']:10.1f}"
        )

    print(
        f"\n{'=' * 100}\nOU FIT (noise-immune, lags 2-64): tau / sigma_level / label-noise / "
        f"innovation s\n{'=' * 100}"
    )
    print(
        f"{'recording':<40s} {'mode':<7s} {'tau_s':>7s} {'sig_lvl':>8s} "
        f"{'sig_noise':>10s} {'a':>9s} {'s_innov':>8s}"
    )
    for r in results:
        ou = r.get("ou")
        if not ou:
            continue
        for k in MODES:
            e = ou.get(k)
            if not e:
                continue
            print(
                f"{r['tag'][:40]:<40s} {k:<7s} {e['tau_s']:7.3f} {e['sigma_level']:8.4f} "
                f"{e['sigma_noise']:10.4f} {e['a']:9.6f} {e['s_innov']:8.4f}"
            )
        d = ou.get("discrimination")
        if d:
            print(
                f"{'':<40s} {'-> R_ou = tauV_c/tauV_d =':<7s} {d['R_ou']:.2f}   "
                f"sustained cost(one rotor)/cost(all four) = "
                f"{d['sustained_cost_one_over_four']:.2f}"
            )

    print(f"\n{'=' * 100}\nBREAK-EVEN of the proposed mode-space transition cost\n{'=' * 100}")
    print("  quadratic psi, one sigma_d shared by roll/pitch/yaw (forced: rotor identity")
    print("  is arbitrary under PIT, and only the differential SUBSPACE is permutation-")
    print("  invariant).  For a move of size delta:")
    print("      all four rotors together : (delta / sigma_c)^2")
    print("      one rotor alone          : (delta^2/16) (1/sigma_c^2 + 3/sigma_d^2)")
    print("      cost(one) / cost(four)   =  1/16 + (3/16) (sigma_c/sigma_d)^2")
    print(f"\n  {'sigma_c/sigma_d':>16s} {'cost(one)/cost(four)':>22s}")
    for k in (1.0, 1.5, 2.0, 2.236, 3.0, 5.0, 10.0):
        note = "   <- BREAK-EVEN" if abs(k - 2.236) < 1e-3 else ""
        print(f"  {k:16.3f} {1 / 16 + 3 / 16 * k * k:22.3f}{note}")
    print(
        "\n  Below sqrt(5) = 2.236 the prior makes an UNCORRELATED single-rotor move\n"
        "  CHEAPER than the correlated move it was designed to prefer."
    )

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(results, indent=1))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
