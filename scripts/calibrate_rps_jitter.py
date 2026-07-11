"""Calibrate the RPS-jitter Ornstein-Uhlenbeck process from DREGON telemetry.

Real rotor speeds carry a fast, zero-mean *jitter* riding on top of the slow
commanded trajectory. Telemetry-conditioned harmonic generation cannot know
this jitter, so it renders clean tones; the true harmonic ``k`` is broadened by
``+/- k * sigma_jitter`` Hz. Injecting a matched OU perturbation into the
generator's conditioning RPS reproduces that broadening (see
``HarmonicNoiseGenNew`` ``rps_jitter_sigma`` / ``rps_jitter_tau``). This script
estimates the two OU parameters from the refined-RPS validation NPZs.

Method (deliberately simple and honest)
---------------------------------------
For each ``results/rps_refinement/validation/dregon_*.npz`` and each of the 4
rotors:

* ``jitter = measured - moving_average(measured, 0.25 s)`` -- the fast residual
  after removing the slow commanded trend.
* ``sigma`` = std of the jitter (rev/s).
* ``tau``   = correlation time, from an exponential fit of the jitter
  autocorrelation ``ACF(Delta) = exp(-Delta / tau)`` over lags up to ~0.5 s.

Pooled values concatenate the per-rotor jitter (sigma) and average the per-rotor
ACFs before the exponential fit (tau).

CAVEATS (printed and stored in the JSON):

* The *measured* telemetry has its own sensor noise, which inflates the residual
  std -- so the reported ``sigma`` is an **upper bound** on true aerodynamic
  jitter.
* The frame grid is ~32 ms (~31 Hz), so it **low-passes** any jitter faster than
  ~15 Hz. True high-frequency jitter is invisible here; ``tau`` is therefore a
  *lower*-resolution estimate bounded by the grid spacing.

Usage::

    .venv/bin/python scripts/calibrate_rps_jitter.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
VALID_DIR = REPO_ROOT / "results" / "rps_refinement" / "validation"
OUT_JSON = REPO_ROOT / "results" / "rps_refinement" / "jitter_calibration.json"

MA_WINDOW_S = 0.25  # moving-average window for the slow-trend removal
ACF_MAX_LAG_S = 0.5  # fit the exponential ACF over lags up to this
N_ROTORS = 4


def moving_average(x: np.ndarray, win: int) -> np.ndarray:
    """Centred moving average with edge replication (same length as ``x``)."""
    if win < 2:
        return x.copy()
    pad = win // 2
    xp = np.pad(x, pad, mode="edge")
    kernel = np.ones(win) / win
    ma = np.convolve(xp, kernel, mode="same")
    return ma[pad : pad + len(x)]


def autocorr(x: np.ndarray, max_lag: int) -> np.ndarray:
    """Biased, normalised autocorrelation ``acf[0..max_lag]`` (acf[0] == 1)."""
    x = x - x.mean()
    denom = np.sum(x * x)
    if denom <= 0:
        return np.zeros(max_lag + 1)
    acf = np.array([np.sum(x[: len(x) - k] * x[k:]) for k in range(max_lag + 1)])
    return acf / denom


def fit_tau(acf: np.ndarray, dt: float) -> float:
    """Fit ``ACF(Delta) = exp(-Delta / tau)`` -> tau (seconds).

    Linear least-squares of ``ln(acf)`` on lag-time, forcing the intercept to 0
    (acf[0] == 1). Only positive ACF values are usable (ln is undefined
    otherwise), so we fit the leading run of positive lags.
    """
    lags = np.arange(1, len(acf))
    vals = acf[1:]
    good = vals > 1e-3
    if not good.any():
        return float("nan")
    # take the leading contiguous run of positive ACF (decorrelation is monotone
    # enough over the first ~0.5 s that this avoids fitting noisy negative tails)
    first_bad = np.argmin(good) if not good.all() else len(good)
    lags = lags[:first_bad]
    vals = vals[:first_bad]
    if len(lags) < 1:
        return float("nan")
    t = lags * dt
    y = np.log(vals)
    # slope through origin: tau = -1 / (sum(t*y)/sum(t*t))
    slope = np.sum(t * y) / np.sum(t * t)
    if slope >= 0:
        return float("nan")
    return float(-1.0 / slope)


def main() -> int:
    files = sorted(VALID_DIR.glob("dregon_*.npz"))
    if not files:
        print(f"No dregon_*.npz found in {VALID_DIR}")
        return 1

    per_recording: dict[str, dict] = {}
    all_jitter: list[np.ndarray] = []
    all_acfs: list[np.ndarray] = []
    dt_global = 0.032  # frame-grid spacing; overwritten from each NPZ's frame_times

    print(f"Calibrating RPS jitter from {len(files)} DREGON recordings\n")
    header = f"{'recording':<40} {'rotor':>5} {'sigma[rev/s]':>13} {'tau[ms]':>9}"
    print(header)
    print("-" * len(header))

    for f in files:
        d = np.load(f)
        ft = d["frame_times"]
        measured = d["measured"]  # (4, N)
        dt = float(np.median(np.diff(ft)))
        dt_global = dt
        ma_win = max(2, int(round(MA_WINDOW_S / dt)))
        max_lag = max(1, int(round(ACF_MAX_LAG_S / dt)))

        rec_entry: dict[str, dict] = {}
        for r in range(min(N_ROTORS, measured.shape[0])):
            series = measured[r].astype(np.float64)
            trend = moving_average(series, ma_win)
            jitter = series - trend
            sigma = float(np.std(jitter))
            acf = autocorr(jitter, max_lag)
            tau = fit_tau(acf, dt)

            rec_entry[str(r)] = {"sigma": sigma, "tau": tau}
            all_jitter.append(jitter)
            all_acfs.append(acf)
            print(f"{f.stem:<40} {r:>5} {sigma:>13.4f} {tau * 1e3:>9.1f}")

        per_recording[f.stem] = rec_entry

    # Pooled: sigma from all jitter samples, tau from the mean ACF.
    pooled_jitter = np.concatenate(all_jitter)
    pooled_sigma = float(np.std(pooled_jitter))
    min_len = min(len(a) for a in all_acfs)
    mean_acf = np.mean([a[:min_len] for a in all_acfs], axis=0)
    pooled_tau = fit_tau(mean_acf, dt_global)

    sig_list = [v["sigma"] for rec in per_recording.values() for v in rec.values()]
    tau_list = [v["tau"] for rec in per_recording.values() for v in rec.values()]
    tau_list = [t for t in tau_list if np.isfinite(t)]

    print("-" * len(header))
    print(
        f"{'POOLED (all rotors/recordings)':<40} {'':>5} {pooled_sigma:>13.4f} {pooled_tau * 1e3:>9.1f}"
    )
    print(
        f"{'median-of-per-rotor':<40} {'':>5} {np.median(sig_list):>13.4f} {np.median(tau_list) * 1e3:>9.1f}"
    )

    print("\nCAVEATS:")
    print("  * sigma is an UPPER BOUND: the measured telemetry carries its own")
    print("    sensor noise, which adds to the aerodynamic jitter residual.")
    print(f"  * frame grid dt = {dt_global * 1e3:.1f} ms (~{1 / dt_global:.0f} Hz): jitter faster")
    print("    than ~half that rate is low-passed away, so tau is grid-limited.")

    result = {
        "meta": {
            "ma_window_s": MA_WINDOW_S,
            "acf_max_lag_s": ACF_MAX_LAG_S,
            "frame_dt_s": dt_global,
            "frame_rate_hz": 1.0 / dt_global,
            "n_recordings": len(files),
            "units": "sigma in rev/s, tau in seconds",
            "caveats": [
                "sigma is an upper bound (measured telemetry includes sensor noise)",
                "frame grid (~32 ms) low-passes jitter; tau is grid-limited",
            ],
        },
        "per_recording": per_recording,
        "pooled": {
            "sigma": pooled_sigma,
            "tau": pooled_tau,
            "sigma_median": float(np.median(sig_list)),
            "tau_median": float(np.median(tau_list)),
        },
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(result, indent=2))
    print(f"\nWrote {OUT_JSON.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
