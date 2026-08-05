"""k-scaled band, k<=80, TELEMETRY init, DREGON. B0 = 3 (band_k = 3k Hz = 3 rev/s at every k)."""

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import hk_core as H
import numpy as np  # noqa: E402

from tracking.phase_increment_tracker import pi_kalman_refine

RID = "free-flight_nosource_room1"
T0 = 22.56481
DUR = 16.0
audio, sr, g, rates = H.load_raw(RID, T0, DUR)
FS = 62.5
ft = np.arange(0, DUR, 1 / FS)
tf = np.arange(audio.shape[1]) / sr
r_init = np.stack([np.interp(ft, tf, g[r]) for r in range(4)])
out: dict[str, Any] = {}
ARMS: list[tuple[str, float, str]] = [
    ("kscaled_b3", 3.0, "gate"),
    ("kscaled_b3_joint", 3.0, "joint"),
    ("kscaled_b1", 1.0, "gate"),
]
for tag, b0, pair_mode in ARMS:
    r_ref, diag = pi_kalman_refine(
        audio,
        r_init,
        ft,
        sr=sr,
        n_iter=3,
        k_max=80,
        f_max=7500.0,
        k_caps=(80, 80, 80),
        fs_env=FS,
        band_hz=6.0,
        off_comb_hz=11.0,
        band_mode="k_scaled",
        band_b0=b0,
        pair_mode=pair_mode,
    )
    d = r_ref - r_init
    mean = d.mean(1)
    pct = 100 * mean / r_init.mean(1)
    print(f"\n== {tag} ==")
    print("  delta mean rev/s:", np.round(mean, 4))
    print("  as % of rate    :", np.round(pct, 3))
    print("  delta rms       :", np.round(d.std(1), 4))
    print(
        "  final rates     :",
        np.round(r_ref.mean(1), 3),
        " (init",
        np.round(r_init.mean(1), 3),
        ")",
    )
    swap = [int(np.argmin(np.abs(r_init.mean(1) - v))) for v in r_ref.mean(1)]
    print(
        "  nearest init rotor per refined rotor:",
        swap,
        "-> IDENTITY",
        "OK" if swap == [0, 1, 2, 3] else "COLLAPSED",
    )
    out[tag] = {
        "delta_mean": [float(x) for x in mean],
        "pct": [float(x) for x in pct],
        "rms": [float(x) for x in d.std(1)],
        "final": [float(x) for x in r_ref.mean(1)],
        "identity_ok": swap == [0, 1, 2, 3],
    }
    np.savez(f"refined_{tag}.npz", ft=ft, r_init=r_init, r_ref=r_ref)
with open("kscaled_telemetry_init.json", "w") as fh:
    json.dump(out, fh, indent=1)
