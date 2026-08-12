"""Free-field steering vectors for the rotor -> microphone array.

The conventions are taken **verbatim** from the repo's own propagation model
(:mod:`models.generative.positional_harmonic_gen`), so that a source rendered
with :func:`~models.generative.positional_harmonic_gen.propagate` and a source
modelled here are the same object:

* ``rel_pos[m, r] = mic_pos[m] - rotor_pos[r]`` (``tasks.noise_generation``)
* amplitude ``ref_distance / d`` with ``ref_distance = 1.0`` m
* delay ``tau = d / c`` with ``c = 343`` m/s, applied as ``exp(-j 2 pi f tau)``

So the steering vector of rotor ``r`` at frequency ``f`` is

    g[f, m, r] = (ref_distance / d_{m,r}) * exp(-j 2 pi f d_{m,r} / c)

and ``P_r(f)``, the fitted per-rotor source PSD, is the PSD the rotor would
produce **at 1 m** — the same normalisation the generator's ``noise_amps``
branch uses, which is what makes the fitted numbers usable as training targets
without a further conversion.
"""

from __future__ import annotations

import numpy as np

from models.generative.positional_harmonic_gen import SPEED_OF_SOUND

__all__ = ["SPEED_OF_SOUND", "distances", "steering", "max_mic_spacing", "aliasing_frequency"]


def distances(mic_pos: np.ndarray, rotor_pos: np.ndarray) -> np.ndarray:
    """``(M, R)`` rotor->mic distances in metres."""
    mic = np.asarray(mic_pos, dtype=np.float64)
    rot = np.asarray(rotor_pos, dtype=np.float64)
    return np.linalg.norm(mic[:, None, :] - rot[None, :, :], axis=-1)


def steering(
    mic_pos: np.ndarray,
    rotor_pos: np.ndarray,
    freqs_hz: np.ndarray,
    *,
    c: float = SPEED_OF_SOUND,
    ref_distance: float = 1.0,
    eps: float = 1e-6,
) -> np.ndarray:
    """``(F, M, R)`` complex steering matrix; see the module docstring."""
    d = np.maximum(distances(mic_pos, rotor_pos), eps)  # (M, R)
    f = np.asarray(freqs_hz, dtype=np.float64)[:, None, None]
    return (ref_distance / d)[None] * np.exp(-2j * np.pi * f * d[None] / c)


def max_mic_spacing(mic_pos: np.ndarray) -> float:
    """Largest inter-microphone distance (metres)."""
    m = np.asarray(mic_pos, dtype=np.float64)
    return float(np.linalg.norm(m[:, None, :] - m[None, :, :], axis=-1).max())


def aliasing_frequency(mic_pos: np.ndarray, c: float = SPEED_OF_SOUND) -> float:
    """Classical spatial-aliasing frequency ``c / (2 d_max)``.

    This is the plane-wave, unknown-direction bound. With **known** and finitely
    many source positions it is not a hard wall — the operative test is whether
    the four rotors' cross-spectral signatures stay distinguishable, which
    :mod:`.design` measures directly. The number is reported for orientation.
    """
    return float(c / (2.0 * max_mic_spacing(mic_pos)))
