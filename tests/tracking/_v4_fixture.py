"""THE synthetic that obeys the v4 model, with every term known.

The v4 model says a window's power spectral density is a smooth floor plus a
comb of Lorentzians, so a fixture that tests the FIT must be built out of
exactly that — otherwise a measured error is a statement about the fixture's
line shape and not about the estimator. Both builders below therefore construct
their line field with :func:`tracking.joint_decompose._lorentzian_design`, the
same truncated basis the fit uses, at the same half width law
``gamma_k = max(0.6 k, one bin)``.

One number decides whether the numbers mean anything, and it is easy to get
wrong: the periodogram this package normalizes to
(``1 / (sr * sum(hann^2))``) estimates the TWO-SIDED power spectral density. So
white noise shaped in the frequency domain by ``H`` has ``S = |H|^2 / sr``, and
a fixture that wants a target ``S`` shapes with ``sqrt(S * sr)``. Using the
one-sided convention instead puts every level exactly 3.01 dB out.

The DENSITY axis is what the dense builder varies. With one comb at rate
``Delta`` the line spacing is ``Delta`` and the half width is ``0.6 k``, so
``k`` IS the density: ``gamma / Delta`` runs 0.06 to 0.48 over ``k`` 5 to 40 at
50 rev/s. The v3 masked floor fit stops having anything to read long before the
lines merge — its mask is ``+/- min(3 * 0.6 k, 0.45 r)`` per line, so it
blankets the band from about ``k`` 13 — which is the regime this fixture exists
to put a number on.
"""

from __future__ import annotations

import numpy as np

from tracking.joint_decompose import LINEWIDTH_HZ_PER_K, _lorentzian_design, comb_lines

SR = 8000
SECONDS = 12.0
N_FFT = 2048
RATE = 50.0
K_HI = 40
N_MIC = 2

#: The four bands the dense fixture is read in, and the density each covers.
#: ``k12-20`` upward is where the v3 mask blankets and the fit has to bridge.
BANDS = {
    "k05-10": (250.0, 500.0),
    "k12-20": (600.0, 1000.0),
    "k22-30": (1100.0, 1500.0),
    "k32-40": (1600.0, 2000.0),
}


def shape_to_psd(rng: np.random.Generator, n_t: int, psd: np.ndarray) -> np.ndarray:
    """White noise shaped so its TWO-SIDED power spectral density is ``psd``."""
    return np.fft.irfft(np.fft.rfft(rng.standard_normal(n_t)) * np.sqrt(psd * SR), n=n_t)


def true_psd(freq: np.ndarray, mic: int = 0, rate: float = RATE, k_hi: int = K_HI):
    """``(floor, line field, peak powers, lines, half widths)`` on ``freq``."""
    df = SR / N_FFT
    floor = 1e-4 * (1.0 + (freq / 400.0) ** 2) ** -0.7 * (0.7 + 0.3 * mic)
    lines, kk = comb_lines(np.array([rate]), k_hi)
    half = np.maximum(LINEWIDTH_HZ_PER_K * kk, df)
    a_mat, kept = _lorentzian_design(freq, np.arange(freq.size), lines, half)
    amp = 2e-3 / kk**0.8 * (0.7 + 0.3 * mic)
    return floor, a_mat @ amp[kept], amp, lines, half


def dense_fixture(seed: int = 0):
    """``(audio, rates, freq of the truth, per-microphone true floor)``."""
    rng = np.random.default_rng(seed)
    n_t = int(round(SECONDS * SR))
    f_t = np.fft.rfftfreq(n_t, d=1.0 / SR)
    audio, floors = [], []
    for c in range(N_MIC):
        floor, field, *_ = true_psd(f_t, c)
        floors.append(floor)
        audio.append(shape_to_psd(rng, n_t, floor + field))
    return np.stack(audio), np.full((1, n_t), RATE), f_t, floors


def floor_error_db(log_s, freq, f_t, s_true, lo, hi) -> tuple[float, float, float]:
    """``(rms, max, bias)`` of a fitted log floor against the truth, in decibels."""
    band = (np.asarray(freq) >= lo) & (np.asarray(freq) <= hi)
    err = (
        10.0 / np.log(10.0) * (np.asarray(log_s)[band] - np.log(np.interp(freq[band], f_t, s_true)))
    )
    return float(np.sqrt(np.mean(err**2))), float(np.max(np.abs(err))), float(np.mean(err))


def band_errors(log_s, freq, f_t, floors) -> dict[str, tuple[float, float, float]]:
    """The per-band ``(rms, max, bias)``, averaged over the microphones."""
    out = {}
    for name, (lo, hi) in BANDS.items():
        per = [floor_error_db(log_s[c], freq, f_t, floors[c], lo, hi) for c in range(len(floors))]
        out[name] = tuple(float(np.mean([p[i] for p in per])) for i in range(3))  # type: ignore[assignment]
    return out
