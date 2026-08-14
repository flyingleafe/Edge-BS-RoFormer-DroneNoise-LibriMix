"""THE synthetic recording that obeys the v3 model, with every term known.

One builder, two callers: the acceptance test
(``tests/tracking/test_joint_decompose.py``, at the 20 s defaults) and the
refactoring guard (``tests/tracking/test_joint_regression.py``, at a small
size that fits a committed reference array). It lives beside them and not in
``src`` because it is a TEST input — nothing shipped builds a fixture.

The LABELS are a smooth trajectory (what telemetry gives). The TRUTH is those
labels plus a shaft-rate error, so ``theta`` is exactly the phase the
decomposition has to recover, and the per-track ``psi`` grows with ``k`` the way
the linewidth law says. The floor is smooth and colored, not white, which is the
second thing v2 assumes and does not have.
"""

from __future__ import annotations

from typing import Any

import numpy as np

SR = 16000
SECONDS = 20.0
N_ROT = 4
N_MIC = 3
K_MAX = 20
SHAFT_REV_S = 0.5
SHAFT_BW_HZ = 0.5
PSI_RMS_PER_K = 0.02
FLOOR_DB = -34.0


def lowpass_noise(
    rng: np.random.Generator, n: int, fs: float, bw_hz: float, rms: float
) -> np.ndarray:
    """Smooth zero-mean process with a -3 dB bandwidth of about ``bw_hz``."""
    f = np.fft.rfftfreq(n, d=1.0 / fs)
    h = 1.0 / (1.0 + (f / max(bw_hz, 1e-6)) ** 4)
    y = np.fft.irfft(np.fft.rfft(rng.standard_normal(n)) * h, n=n)
    y -= y.mean()
    s = float(y.std())
    return y * (rms / s) if s > 0 else y


def make_fixture(
    seed: int = 0,
    *,
    seconds: float = SECONDS,
    sr: int = SR,
    n_rot: int = N_ROT,
    n_mic: int = N_MIC,
    k_max: int = K_MAX,
) -> dict[str, Any]:
    """``{audio, r_hat, theta, floor_shape}`` — the fixture and its truth.

    The random draws are made in one fixed order, so a given ``seed`` and size
    give the same recording on every machine. The defaults are the acceptance
    test's 20 s recording; the guard asks for a short one.
    """
    rng = np.random.default_rng(seed)
    n_t = int(round(seconds * sr))
    t = np.arange(n_t) / sr
    base = np.array([70.0, 74.0, 78.0, 82.0])[:n_rot]
    r_hat = np.stack([b + 1.5 * np.sin(2 * np.pi * 0.07 * t + 0.7 * i) for i, b in enumerate(base)])

    dr_rig = lowpass_noise(rng, n_t, sr, SHAFT_BW_HZ, SHAFT_REV_S)
    theta = np.zeros((n_rot, n_t))
    for i in range(n_rot):
        dr = dr_rig + lowpass_noise(rng, n_t, sr, SHAFT_BW_HZ, 0.2 * SHAFT_REV_S)
        theta[i] = 2 * np.pi * np.cumsum(dr) / sr
        theta[i] -= theta[i].mean()
    phi_hat = 2 * np.pi * np.cumsum(r_hat, axis=-1) / sr

    ks = np.arange(1, k_max + 1)
    amp = 1.0 / ks**1.2
    mic_gain = 0.6 + 0.5 * rng.random((n_mic, n_rot))
    y = np.zeros((n_mic, n_t))
    for i in range(n_rot):
        for kk in ks:
            psi = lowpass_noise(rng, n_t, sr, min(0.6 * kk, 8.0), PSI_RMS_PER_K * kk)
            car = np.cos(kk * (phi_hat[i] + theta[i]) + psi + 2 * np.pi * rng.random())
            y += (mic_gain[:, i : i + 1] * amp[kk - 1]) * car[None, :]

    f = np.fft.rfftfreq(n_t, d=1.0 / sr)
    shape = (1.0 + (f / 120.0) ** 2) ** -0.6 * (1.0 + (f / 3000.0) ** 2) ** -0.5
    noise = np.zeros((n_mic, n_t))
    for c in range(n_mic):
        v = np.fft.irfft(np.fft.rfft(rng.standard_normal(n_t)) * shape, n=n_t)
        noise[c] = v / v.std()
    lvl = float(np.sqrt(np.mean(y**2))) * 10 ** (FLOOR_DB / 20.0)
    y = y + noise * lvl * (1.0 + 0.4 * np.arange(n_mic))[:, None]
    return {"audio": y, "r_hat": r_hat, "theta": theta, "floor_shape": shape}
