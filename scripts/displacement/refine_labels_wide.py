"""Realign DREGON telemetry to the acoustic comb: pi_kalman initialised FROM telemetry."""

import json
import sys

sys.path.insert(0, "/home/flyingleafe/Research/PhD/projects/harmonic-noise-suppression/src")
import hk_core as H  # noqa: E402
import numpy as np  # noqa: E402

from tracking.phase_increment_tracker import pi_kalman_refine  # noqa: E402

JSON_OUT = "refined_labels_wide.json"

RID = "free-flight_nosource_room1"
T0 = 22.56481
DUR = 16.0
audio, sr, g, rates = H.load_raw(RID, T0, DUR)
FS_FT = 62.5
ft = np.arange(0, DUR, 1 / FS_FT)
tfull = np.arange(audio.shape[1]) / sr
r_init = np.stack([np.interp(ft, tfull, g[r]) for r in range(4)])
print("init", r_init.shape, "rates", np.round(r_init.mean(1), 3))

r_ref, diag = pi_kalman_refine(
    audio,
    r_init,
    ft,
    sr=sr,
    n_iter=3,
    band_hz=(24.0, 8.0, 3.0),
    k_max=100,
    f_max=7000.0,
    k_caps=(8, 24, 60),
    fs_env=FS_FT,
    off_comb_hz=40.0,
    pair_mode="gate",
)
d = r_ref - r_init
print("delta rev/s  mean", np.round(d.mean(1), 4), " rms", np.round(d.std(1), 4))
print("as % of rate", np.round(100 * d.mean(1) / r_init.mean(1), 3))
# lag estimate: cross-correlate detrended derivative of telemetry vs refined
lags = {}
for r in range(4):
    a = np.gradient(r_init[r])
    b = np.gradient(r_ref[r])
    a -= a.mean()
    b -= b.mean()
    c = np.correlate(b, a, "full")
    L = np.arange(-len(a) + 1, len(a))
    m = np.abs(L) <= int(0.5 * FS_FT)
    lags[r] = float(L[m][np.argmax(c[m])] / FS_FT)
print("best lag (s, +ve = refined lags telemetry):", lags)
np.savez(
    "refined_labels_wide.npz",
    ft=ft,
    r_init=r_init,
    r_ref=r_ref,
    rates=rates,
    lags=np.array([lags[r] for r in range(4)]),
)
with open(JSON_OUT, "w") as _fh:
    json.dump(
        {
            "delta_mean": [float(x) for x in d.mean(1)],
            "delta_rms": [float(x) for x in d.std(1)],
            "pct_of_rate": [float(x) for x in 100 * d.mean(1) / r_init.mean(1)],
            "lag_s": lags,
            "diag_keys": sorted(diag)[:25],
        },
        _fh,
        indent=1,
    )
print("saved")
