"""Time-varying time-warp augmentation for the noise+RPS pair.

A rotating-noise recording with rotor-speed trajectory ``r(t)`` is resampled at
a slowly time-varying playback rate ``alpha(t)``. Physically this stretches
duration locally by ``1 / alpha`` and scales all frequencies by ``alpha``,
converting the recording into a *valid* recording of the same rig with a new
rotor trajectory

    r_tilde(t) = alpha(t) * r(tau(t)),     tau(t) = integral_0^t alpha(u) du,

where ``tau`` is the source-time warp map (monotone since ``alpha > 0``). The
warp is pure interpolation resampling (no phase vocoder). We apply it to the
noise *before* mixing with speech and transform the RPS label consistently, so
the trained RPS predictor sees a physically-plausible augmented pair.

Parametrization (kept deliberately "not too extreme"):

    alpha(t) = c + a * sin(2*pi*f*t + phi)

with ``c ~ U[1 - dev_const, 1 + dev_const]``, ``a ~ U[0, dev_sine]``,
``f ~ U[f_low, f_high]`` Hz and ``phi ~ U[0, 2*pi)``. With the defaults
(``dev_const=0.08``, ``dev_sine=0.04``) the worst-case deviation is
``|alpha - 1| <= 0.12`` and the average rate over any window lies in
``[1 - 0.12, 1 + 0.12]``, so reading a target duration ``T`` of warped audio
consumes at most ``(c + a) * T <= 1.12 * T`` of source time.

``tau`` has the closed form

    tau(t) = c*t + (a / (2*pi*f)) * (cos(phi) - cos(2*pi*f*t + phi)).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import tdseries as td

from data_processing.dregon import clean_command_spikes
from data_processing.frames import get_meta

DEFAULT_DEV_CONST = 0.08
DEFAULT_DEV_SINE = 0.04
DEFAULT_F_LOW = 0.1
DEFAULT_F_HIGH = 1.0

#: Uniform grid rate (Hz) for the warped ``rps`` label track. The downstream
#: :func:`interpolate_rps_to_stft_grid` re-interpolates this onto the STFT frame
#: grid, so the track only needs to densely cover ``[0, T]``.
DEFAULT_LABEL_RATE_HZ = 100.0

#: Small extra source margin (seconds) requested on top of the analytic
#: ``(1 + dev_const + dev_sine)`` worst case, guarding against off-by-one slice
#: rounding so ``tau(T)`` never indexes past the sampled source audio.
WARP_SOURCE_MARGIN_S = 0.1


@dataclass(frozen=True)
class WarpParams:
    """Sampled parameters of one ``alpha(t) = c + a*sin(2*pi*f*t + phi)`` warp."""

    c: float
    a: float
    f: float
    phi: float
    dev_const: float
    dev_sine: float

    def alpha(self, t: np.ndarray) -> np.ndarray:
        """Playback rate at target-time ``t`` (seconds)."""
        return self.c + self.a * np.sin(2.0 * np.pi * self.f * t + self.phi)

    def tau(self, t: np.ndarray) -> np.ndarray:
        """Source time ``tau(t) = integral_0^t alpha`` at target-time ``t``."""
        return self.c * t + (self.a / (2.0 * np.pi * self.f)) * (
            np.cos(self.phi) - np.cos(2.0 * np.pi * self.f * t + self.phi)
        )

    @property
    def source_duration_factor(self) -> float:
        """Worst-case ``tau(T) / T`` upper bound used to size the source request."""
        return 1.0 + self.dev_const + self.dev_sine


def sample_warp_params(spec: Mapping[str, Any], rng: np.random.Generator) -> WarpParams:
    """Draw warp parameters from ``spec`` using the worker's ``rng``.

    Draw order (``c``, ``a``, ``f``, ``phi``) is fixed so the RNG stream is
    reproducible. The caller is responsible for the single fire decision draw.
    """
    dev_const = float(spec.get("dev_const", DEFAULT_DEV_CONST))
    dev_sine = float(spec.get("dev_sine", DEFAULT_DEV_SINE))
    f_low = float(spec.get("f_low", DEFAULT_F_LOW))
    f_high = float(spec.get("f_high", DEFAULT_F_HIGH))
    c = float(rng.uniform(1.0 - dev_const, 1.0 + dev_const))
    a = float(rng.uniform(0.0, dev_sine))
    f = float(rng.uniform(f_low, f_high))
    phi = float(rng.uniform(0.0, 2.0 * np.pi))
    return WarpParams(c=c, a=a, f=f, phi=phi, dev_const=dev_const, dev_sine=dev_sine)


def source_duration_s(base_duration_s: float, params: WarpParams) -> float:
    """Source seconds to request so ``tau(base_duration_s)`` stays in-bounds."""
    return base_duration_s * params.source_duration_factor + WARP_SOURCE_MARGIN_S


def _resolve_rps_track(frame: td.Frame) -> tuple[str, bool]:
    """Return ``(rps_key, needs_cleaning)`` for the source noise Frame.

    Mirrors :func:`data_processing.online_mixing._resolve_motor_tracks` but is
    inlined here to avoid a circular import (that module imports this one).
    """
    if "motors_command" in frame or "motors_measured" in frame:
        rps_key = "motors_command" if "motors_command" in frame else "motors_measured"
        return rps_key, True
    if "rps" in frame:
        return "rps", False
    raise ValueError(
        f"{get_meta(frame, 'recording_id', '?')} has no rotor-speed track "
        "(expected 'motors_measured', 'motors_command', or 'rps')"
    )


def apply_time_warp(
    frame: td.Frame,
    params: WarpParams,
    *,
    target_len: int,
    sample_rate: int,
    label_rate_hz: float = DEFAULT_LABEL_RATE_HZ,
) -> td.Frame:
    """Warp a sampled noise Frame's audio + RPS pair by ``alpha(t)``.

    Returns a fresh ``td.Frame`` (timeline restarted at ``t_start=0``) with:

    - ``audio``: the source audio linearly interpolated (per channel) at the
      source positions ``tau(t_i)`` for the target uniform grid
      ``t_i = i / sample_rate``, exactly ``target_len`` samples long;
    - ``rps``: a generic *already-clean* rotor track on a uniform
      ``label_rate_hz`` grid holding ``alpha(t) * r(tau(t))``. Naming it ``rps``
      routes it through the no-cleaning branch of ``_resolve_motor_tracks``.

    The original motor tracks are dropped so the rotor track resolves
    unambiguously downstream. ``meta`` is carried through when present.
    """
    audio = cast(td.Series, frame["audio"])
    audio_sr = float(cast(td.GridIndex, audio.tindex).sr)
    if int(round(audio_sr)) != int(sample_rate):
        raise ValueError(f"warp source audio sr {audio_sr} != configured {sample_rate}")

    data = np.asarray(audio.data, dtype=np.float32)
    is_mono = data.ndim == 1
    if is_mono:
        data = data[None, :]
    n_src = data.shape[-1]

    # Target uniform grid (source-local seconds from the slice start).
    t_target = np.arange(target_len, dtype=np.float64) / float(sample_rate)
    src_pos = params.tau(t_target) * float(sample_rate)  # source positions in samples
    src_idx = np.arange(n_src, dtype=np.float64)
    if float(src_pos[-1]) > float(n_src - 1):
        raise ValueError(
            f"time warp reads source sample {src_pos[-1]:.1f} beyond available "
            f"{n_src} (increase the requested source duration)"
        )

    warped = np.empty((data.shape[0], target_len), dtype=np.float32)
    for ch in range(data.shape[0]):
        warped[ch] = np.interp(src_pos, src_idx, data[ch]).astype(np.float32)
    out_audio = warped[0] if is_mono else warped
    audio_series = td.uniform(out_audio, int(sample_rate), dims=audio.dims, t_start=0.0)

    # RPS label on a uniform grid covering [0, target duration].
    rps_key, needs_clean = _resolve_rps_track(frame)
    motor = cast(td.Series, frame[rps_key])
    target_duration = target_len / float(sample_rate)
    n_label = int(np.ceil(target_duration * label_rate_hz)) + 1
    t_label = np.arange(n_label, dtype=np.float64) / float(label_rate_hz)

    if motor.data is None or motor.dim_size("time") == 0:
        label = np.zeros((4, n_label), dtype=np.float32)
    else:
        motor_vals = np.asarray(motor.data, dtype=np.float64)
        if needs_clean:
            motor_vals = clean_command_spikes(motor_vals)
        # Source-local timestamps (the motor track shares the audio timeline).
        motor_times = cast(td.StampIndex, motor.tindex).abs_stamps - audio.t_start
        tau_label = params.tau(t_label)
        alpha_label = params.alpha(t_label)
        r = np.empty((motor_vals.shape[0], n_label), dtype=np.float64)
        for i in range(motor_vals.shape[0]):
            r[i] = np.interp(tau_label, motor_times, motor_vals[i])
        label = (alpha_label[None, :] * r).astype(np.float32)

    rps_series = td.uniform(label, int(round(label_rate_hz)), dims=("rotor", "time"), t_start=0.0)

    entries: dict[str, Any] = {"audio": audio_series, "rps": rps_series}
    if "meta" in frame:
        entries["meta"] = frame["meta"]
    return td.Frame(entries)
