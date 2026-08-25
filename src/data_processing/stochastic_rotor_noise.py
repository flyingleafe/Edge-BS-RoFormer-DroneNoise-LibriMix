"""Stochastic rotor-noise model — a colored floor plus Lorentzian harmonic
lines, with every amplitude drifting as a Gaussian process in time
(``kind: stochastic``).

Why this exists
---------------
The two analytic synthetic families in this project sit at the ends of a
spectrum and both are narrow. ``rotor_spectral_model.StaticCombNoisePool``
draws one static amplitude profile per clip and holds it fixed, deliberately,
so that the comb's *spacing* is the only cue an RPS predictor can use (E8).
That constraint is what makes the comb curriculum work, and it also makes every
clip of that family sound the same: no line breathes, no floor moves, and a
predictor sees one texture. The learned generator moves in time but ties its
amplitudes to the rotor speed, which hands the predictor an amplitude shortcut
that does not exist in real recordings (E7).

This module keeps the spacing-only property and gives up the staticness. Every
amplitude — each harmonic of each rotor, and the broadband floor — varies
slowly in time as a Gaussian process **drawn independently of the rotor-speed
trajectory**. A predictor therefore cannot read speed off any amplitude: the
amplitudes carry no information about it, by construction of the sampler. What
the family gains is variety, which is the point.

The model
---------
This is the generative direction of the v4 analysis model
(:mod:`tracking.joint_decompose`): a smooth colored floor with sparse
Lorentzian lines on top. The power spectral density at frequency ``f`` and time
``t`` is::

    S(f, t) = B(f, t) + sum_r sum_k P_rk(t) * L(f - k * rps_r(t); gamma_rk)

    L(d; gamma) = (1 / pi) * gamma / (d^2 + gamma^2)          (Cauchy density)

so ``P_rk(t)`` is the *power* of harmonic ``k`` of rotor ``r`` and
``gamma_rk`` is its half width at half maximum. A Lorentzian line is what a
tone with a random-walk phase actually produces, and the project has measured
the widening: a shaft that wanders by about 0.6 rev/s widens harmonic ``k`` to
about ``0.6 k`` Hz, which this model writes as
``gamma_rk = gamma0_r + slope_r * k``.

In decibels, with every term additive:

* ``10 log10 P_rk(t) = harm_mean_db + profile_db[r, k] + h_rk(t)``, where
  ``profile_db`` is the static per-rotor timbre (rolloff, blade-pass emphasis,
  per-harmonic irregularity) and ``h_rk`` is the Gaussian process.
* ``10 log10 B(f, t) = floor_mean_db + shape_db(f) + b(t)
  + tilt(t) * log2(f / f_ref)``, where ``shape_db`` is a smooth random curve in
  log frequency and ``b``, ``tilt`` are Gaussian processes.

The Gaussian processes use a squared-exponential kernel, so two knobs describe
each one: a standard deviation in decibels and a correlation time in seconds.
Those are the covariance parameters the notebook exposes as sliders.

Two structure knobs control how independent the pieces are. ``harm_coherence``
mixes a per-rotor common process into every harmonic's own process, so a rotor
can breathe as a whole instead of each line wandering alone.
``rotor_similarity`` mixes one per-clip drone timbre into every rotor's static
profile, so the four rotors of a clip can sound like four rotors of one
aircraft.

Synthesis
---------
The signal is a realization of a Gaussian process with the above spectrum, made
by filtering white noise with a time-varying filter in the short-time Fourier
domain (:func:`_ola_filter`): analyse, multiply each frame by ``sqrt(S)``,
synthesise by weighted overlap-add. Because white noise through the same
analysis and synthesis chain reconstructs exactly, the output's spectrum is
``S`` up to one global constant, and no scaling constants have to be tracked.

Two consequences of working at short-time-Fourier resolution:

* A line narrower than one frequency bin is represented at bin resolution.
  ``gamma`` is floored at ``0.6`` bins so that the discretized line keeps its
  power. At the default 2048-point analysis this floor is 4.7 Hz, which is the
  linewidth of harmonic 8 or so; lower harmonics come out slightly wider than
  the model asks for. The predictor's own front end has the same resolution.
* Everything the model produces is stochastic. There is no coherent tone and no
  absolute phase, and each microphone is an independent realization of its own
  spectrum. Across-microphone phase coherence is therefore not modeled, which
  matters for a beamformer and does not matter for a single-channel predictor.

Interfaces
----------
:func:`sample_params` draws a full parameter set, :func:`synthesize` renders one
clip from a parameter set and a rotor-speed trajectory, and
:class:`StochasticNoisePool` wraps both behind the
``sample_timeframe(rng, duration_s) -> td.Frame`` interface the other noise
pools use. Synthesis is numpy and scipy only, so it runs in the DataLoader
workers with no GPU and no producer process.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
from typing import Any

import numpy as np
import tdseries as td

from data_processing import rps_synthesis
from data_processing.frames import make_recording_frame

# ── Constants ───────────────────────────────────────────────────────────────

#: Analysis and synthesis geometry of the overlap-add filter. A 2048-point
#: window at 16 kHz is 128 ms, matching the front end every RPS model in this
#: project uses, and a quarter-window hop makes the Hann window satisfy the
#: constant-overlap-add condition exactly.
DEFAULT_N_FFT = 2048
DEFAULT_HOP_DIV = 4

#: How many half widths of a Lorentzian are rendered. The skirts are cheap to
#: cut and expensive to keep: the widest lines of a comb reach tens of hertz, so
#: their support sets the cost of the whole scatter. At 5 half widths the
#: density is down by a factor of 26 and 87% of the line's power is inside;
#: ``LORENTZ_TRUNC_NORM`` divides that fraction out, so the rendered line carries
#: its full power and only the far skirts are missing. The analysis side of the
#: project uses 8 (``tracking.joint_decompose.LORENTZ_SUPPORT_HWHM``), where the
#: skirts matter because a fit must explain them.
LORENTZ_SUPPORT_HWHM = 5.0

#: Smallest half width, in frequency bins. Below this a line falls inside one
#: bin and loses power to the discretization.
GAMMA_MIN_BINS = 0.6

#: The fraction of a Lorentzian's power inside its rendered support. Dividing
#: by it keeps the truncated line at the power the model asks for.
LORENTZ_TRUNC_NORM = float(2.0 / np.pi * np.arctan(LORENTZ_SUPPORT_HWHM))

#: Reference frequency of the floor's tilt term (Hz).
FLOOR_TILT_REF_HZ = 500.0

#: Lowest frequency of the floor's shape control grid (Hz). Below it the shape
#: is held flat: a 16 kHz recording carries nothing usable down there, and a
#: log-frequency grid would spend half its points on it.
FLOOR_SHAPE_F_MIN = 30.0

#: Number of control points of the floor's shape curve.
FLOOR_SHAPE_N_CTRL = 14


# ── Gaussian processes in time ──────────────────────────────────────────────


@lru_cache(maxsize=64)
def _se_cholesky(n: int, ratio: float) -> np.ndarray:
    """Cholesky factor of a unit-variance squared-exponential kernel.

    ``ratio`` is the sample spacing divided by the correlation time, so one
    factor serves every process with the same shape. The factor is cached
    because a clip draws hundreds of series from the same kernel.
    """
    d = np.arange(n, dtype=np.float64)
    k = np.exp(-0.5 * ((d[:, None] - d[None, :]) * ratio) ** 2)
    k[np.diag_indices(n)] += 1e-8
    return np.linalg.cholesky(k)


def sample_gp(
    rng: np.random.Generator,
    n_series: int,
    n: int,
    *,
    dt: float,
    tau: float,
    std: float,
) -> np.ndarray:
    """``(n_series, n)`` draws from a zero-mean squared-exponential process.

    ``dt`` is the sample spacing in seconds, ``tau`` the correlation time in
    seconds, ``std`` the standard deviation. A correlation time far below the
    sample spacing is white noise, which the kernel gives anyway; a zero
    standard deviation short-circuits to zeros.
    """
    if n <= 0 or n_series <= 0:
        return np.zeros((max(n_series, 0), max(n, 0)), dtype=np.float64)
    if std <= 0.0:
        return np.zeros((n_series, n), dtype=np.float64)
    ratio = float(dt) / max(float(tau), 1e-6)
    ratio = min(ratio, 12.0)  # beyond this the kernel is numerically the identity
    factor = _se_cholesky(int(n), round(ratio, 6))
    z = rng.standard_normal((n, n_series))
    return (float(std) * (factor @ z)).T


# ── Parameters ──────────────────────────────────────────────────────────────


@dataclass
class StochasticRanges:
    """Sampling ranges for :func:`sample_params` — the family, not one clip.

    The harmonic-profile and floor ranges start from the values
    :class:`data_processing.rotor_spectral_model.ProfileRanges` calibrated on
    real DREGON and Michael's recordings, and widen where the point is variety.
    The Gaussian-process ranges have no direct measurement behind them: the
    correlation times cover a wander that is faster than a clip and slower than
    a syllable, and the standard deviations cover a line that barely moves
    through one that comes and goes.
    """

    # Static harmonic timbre.
    rolloff_p: tuple[float, float] = (0.4, 1.9)
    harm_jitter_db: tuple[float, float] = (2.0, 8.0)
    blade_counts: tuple[int, ...] = (1, 2, 3)
    blade_emphasis_db: tuple[float, float] = (0.0, 10.0)
    #: How much of one clip's timbre is shared by its four rotors.
    rotor_similarity: tuple[float, float] = (0.3, 0.95)

    # Linewidth: gamma_k = gamma0 + slope * k, in Hz (half width at half max).
    gamma0_hz: tuple[float, float] = (0.5, 4.0)
    gamma_slope_hz: tuple[float, float] = (0.05, 0.8)

    # Broadband floor.
    floor_shape_std_db: tuple[float, float] = (2.0, 9.0)
    floor_shape_oct: tuple[float, float] = (0.7, 3.0)
    floor_tilt_db_oct: tuple[float, float] = (-9.0, -1.0)
    #: Where the floor sits under the typical line peak, in dB. Measured on real
    #: DREGON and Michael's single-rotor combs, the floor sits only 1.6 to
    #: 11.6 dB below the median in-band harmonic, which is why so many high
    #: harmonics wash out. This range brackets that.
    floor_rel_db: tuple[float, float] = (-22.0, -2.0)
    #: Smallest fraction of in-band lines that must stand above the floor. The
    #: floor is lowered until the draw satisfies it, so a clip always carries a
    #: trackable comb.
    min_lines_above_floor: float = 0.30

    # Time variation of the harmonic amplitudes.
    harm_gp_std_db: tuple[float, float] = (0.5, 6.0)
    harm_gp_tau_s: tuple[float, float] = (0.3, 6.0)
    harm_coherence: tuple[float, float] = (0.0, 1.0)

    # Time variation of the floor.
    floor_gp_std_db: tuple[float, float] = (0.5, 4.0)
    floor_gp_tau_s: tuple[float, float] = (0.5, 8.0)
    floor_tilt_gp_std: tuple[float, float] = (0.0, 1.5)
    floor_tilt_gp_tau_s: tuple[float, float] = (2.0, 15.0)

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> StochasticRanges:
        if not d:
            return cls()
        out = cls()
        for key, value in d.items():
            if hasattr(out, key):
                setattr(out, key, tuple(value) if isinstance(value, (list, tuple)) else value)
        return out


@dataclass
class StochasticParams:
    """A complete parameter set for one clip.

    Every field is editable, which is what the notebook's sliders write to.
    :func:`dataclasses.replace` gives a modified copy, so a slider move does not
    disturb the static random parts (the timbre and the floor's shape) and the
    same clip can be re-rendered with one number changed.
    """

    sample_rate: int
    n_rotors: int
    n_harmonics: int

    # Static per-rotor timbre, in dB, with the in-band median at 0.
    profile_db: np.ndarray  # (R, K)
    gamma0: np.ndarray  # (R,) Hz
    gamma_slope: np.ndarray  # (R,) Hz per harmonic

    # Static floor shape: a smooth zero-mean curve on a log-frequency grid.
    floor_ctrl_hz: np.ndarray  # (C,)
    floor_ctrl_db: np.ndarray  # (C,)
    floor_tilt_db_oct: float

    # Levels — the two amplitude-mean sliders.
    harm_mean_db: float = 0.0
    floor_mean_db: float = -8.0

    # Covariance sliders.
    harm_gp_std_db: float = 3.0
    harm_gp_tau_s: float = 1.5
    harm_coherence: float = 0.5
    floor_gp_std_db: float = 2.0
    floor_gp_tau_s: float = 3.0
    floor_tilt_gp_std: float = 0.5
    floor_tilt_gp_tau_s: float = 6.0

    # Rotor speed to level. Rotor aeroacoustic sound power grows about as the
    # fifth power of tip speed, so pressure amplitude grows as rps^2.5 and a
    # stopped rotor is silent. This is the one place where the audio depends on
    # the trajectory, and it is monotone, shared with the static-comb family,
    # and true of real recordings.
    amp_rps_exponent: float = 2.5
    amp_rps_ref: float = 80.0

    def with_(self, **changes: Any) -> StochasticParams:
        """A copy with fields replaced — the slider path."""
        return replace(self, **changes)


def _profile_db(
    rng: np.random.Generator,
    ranges: StochasticRanges,
    *,
    n_harmonics: int,
) -> np.ndarray:
    """One static per-harmonic timbre in dB: rolloff, blade emphasis, texture."""
    k = np.arange(1, n_harmonics + 1, dtype=np.float64)
    db = -10.0 * rng.uniform(*ranges.rolloff_p) * np.log10(k)
    blade = int(rng.choice(np.asarray(ranges.blade_counts)))
    emphasis = float(rng.uniform(*ranges.blade_emphasis_db))
    if blade > 1 and emphasis > 0.0:
        db[(np.arange(1, n_harmonics + 1) % blade) == 0] += emphasis
    db += rng.normal(0.0, float(rng.uniform(*ranges.harm_jitter_db)), size=n_harmonics)
    return db


def line_peak_db(params: StochasticParams, ref_rps: float = 80.0) -> tuple[np.ndarray, np.ndarray]:
    """``(peak level in dB, center frequency in Hz)`` of every in-band line.

    A line of power ``P`` and half width ``gamma`` has a peak spectral density
    of ``P / (pi * gamma)``, so a wide line stands lower than a narrow one of
    the same power. The comparison with the floor has to be made on the peak,
    which is the quantity a spectrogram shows and a tracker follows.
    """
    nyquist = params.sample_rate / 2.0
    k = np.arange(1, params.n_harmonics + 1, dtype=np.float64)
    peaks: list[np.ndarray] = []
    centers: list[np.ndarray] = []
    for r in range(params.n_rotors):
        gamma = np.maximum(params.gamma0[r] + params.gamma_slope[r] * k, 1e-3)
        freq = k * ref_rps
        live = freq < nyquist
        peaks.append(params.profile_db[r][live] - 10.0 * np.log10(np.pi * gamma[live]))
        centers.append(freq[live])
    return np.concatenate(peaks), np.concatenate(centers)


def calibrate_floor(
    params: StochasticParams,
    floor_rel_db: float,
    *,
    ref_rps: float = 80.0,
    min_lines_above_floor: float = 0.30,
) -> float:
    """The floor level that puts the floor ``floor_rel_db`` under the lines.

    The level is set so that the floor at the typical line's frequency sits
    ``floor_rel_db`` below the median line peak, and is then lowered in 1 dB
    steps until at least ``min_lines_above_floor`` of the in-band lines stand
    above it. Without the guard a draw can bury its whole comb, and a clip with
    no visible comb teaches nothing.
    """
    peaks, centers = line_peak_db(params, ref_rps)
    if peaks.size == 0:
        return floor_rel_db
    shape = floor_shape_db(replace(params, floor_mean_db=0.0), centers)
    level = float(np.median(peaks) + floor_rel_db - np.median(shape))
    for _ in range(60):
        if float(np.mean(peaks > level + shape)) >= min_lines_above_floor:
            break
        level -= 1.0
    return level


def sample_params(
    rng: np.random.Generator,
    ranges: StochasticRanges | None = None,
    *,
    n_rotors: int = 4,
    n_harmonics: int = 80,
    sample_rate: int = 16000,
) -> StochasticParams:
    """Draw one clip's worth of parameters from the family.

    The four rotors share ``rotor_similarity`` of one drone timbre and keep the
    rest of their own, so a clip can be four rotors of one aircraft or four
    unrelated sources.
    """
    ranges = ranges or StochasticRanges()

    drone = _profile_db(rng, ranges, n_harmonics=n_harmonics)
    similarity = float(rng.uniform(*ranges.rotor_similarity))
    profile = np.empty((n_rotors, n_harmonics), dtype=np.float64)
    for r in range(n_rotors):
        own = _profile_db(rng, ranges, n_harmonics=n_harmonics)
        profile[r] = similarity * drone + (1.0 - similarity) * own
        profile[r] -= np.median(profile[r])

    ctrl_hz = np.geomspace(FLOOR_SHAPE_F_MIN, sample_rate / 2.0, FLOOR_SHAPE_N_CTRL)
    oct_grid = np.log2(ctrl_hz / ctrl_hz[0])
    shape_std = float(rng.uniform(*ranges.floor_shape_std_db))
    shape_len = float(rng.uniform(*ranges.floor_shape_oct))
    ctrl_db = sample_gp(
        rng,
        1,
        FLOOR_SHAPE_N_CTRL,
        dt=float(oct_grid[1] - oct_grid[0]),
        tau=shape_len,
        std=shape_std,
    )[0]
    ctrl_db -= ctrl_db.mean()

    gamma0 = rng.uniform(*ranges.gamma0_hz, size=n_rotors)
    gamma_slope = rng.uniform(*ranges.gamma_slope_hz, size=n_rotors)
    tilt = float(rng.uniform(*ranges.floor_tilt_db_oct))
    draft = StochasticParams(
        sample_rate=int(sample_rate),
        n_rotors=int(n_rotors),
        n_harmonics=int(n_harmonics),
        profile_db=profile,
        gamma0=gamma0,
        gamma_slope=gamma_slope,
        floor_ctrl_hz=ctrl_hz,
        floor_ctrl_db=ctrl_db,
        floor_tilt_db_oct=tilt,
        harm_mean_db=0.0,
        floor_mean_db=0.0,
    )
    floor_mean_db = calibrate_floor(
        draft,
        float(rng.uniform(*ranges.floor_rel_db)),
        min_lines_above_floor=ranges.min_lines_above_floor,
    )

    return replace(
        draft,
        floor_mean_db=floor_mean_db,
        harm_gp_std_db=float(rng.uniform(*ranges.harm_gp_std_db)),
        harm_gp_tau_s=float(rng.uniform(*ranges.harm_gp_tau_s)),
        harm_coherence=float(rng.uniform(*ranges.harm_coherence)),
        floor_gp_std_db=float(rng.uniform(*ranges.floor_gp_std_db)),
        floor_gp_tau_s=float(rng.uniform(*ranges.floor_gp_tau_s)),
        floor_tilt_gp_std=float(rng.uniform(*ranges.floor_tilt_gp_std)),
        floor_tilt_gp_tau_s=float(rng.uniform(*ranges.floor_tilt_gp_tau_s)),
    )


# ── The spectrum ────────────────────────────────────────────────────────────


def floor_shape_db(params: StochasticParams, freqs: np.ndarray) -> np.ndarray:
    """The static part of the floor in dB, on any frequency grid.

    The control points are interpolated in log frequency, which is what keeps
    the curve smooth: a squared-exponential draw on a log grid has no structure
    a cubic interpolant cannot follow.
    """
    from scipy.interpolate import PchipInterpolator

    f = np.maximum(np.asarray(freqs, dtype=np.float64), FLOOR_SHAPE_F_MIN)
    octaves = np.log2(f / params.floor_ctrl_hz[0])
    ctrl_oct = np.log2(params.floor_ctrl_hz / params.floor_ctrl_hz[0])
    shape = PchipInterpolator(ctrl_oct, params.floor_ctrl_db, extrapolate=True)(octaves)
    tilt = params.floor_tilt_db_oct * np.log2(f / FLOOR_TILT_REF_HZ)
    return np.asarray(shape + tilt, dtype=np.float64)


def build_psd(
    params: StochasticParams,
    rps_frames: np.ndarray,
    freqs: np.ndarray,
    *,
    dt: float,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """Build the time-varying spectrum on a frame grid.

    Args:
        params: the parameter set.
        rps_frames: ``(R, N)`` rotor speeds in rev/s at the frame times.
        freqs: ``(F,)`` frequency grid in Hz, uniformly spaced.
        dt: frame spacing in seconds — the Gaussian processes' sample spacing.
        rng: the source of the Gaussian-process draws.

    Returns:
        A dict with ``floor`` ``(N, F)``, ``lines`` ``(R, N, F)`` (each rotor's
        line contribution, already scaled by its speed-dependent level),
        ``harm_gp`` ``(R, K, N)`` and ``floor_gp`` ``(N,)`` — the pieces the
        notebook plots, and what :func:`synthesize` mixes per microphone.
    """
    rps_frames = np.atleast_2d(np.asarray(rps_frames, dtype=np.float64))
    n_rotors, n_frames = rps_frames.shape
    freqs = np.asarray(freqs, dtype=np.float64)
    n_freqs = freqs.size
    df = float(freqs[1] - freqs[0])
    nyquist = float(freqs[-1])
    n_harm = params.n_harmonics

    # Speed-dependent level, one factor per rotor and frame. Zero speed is
    # silence, which is what lets a full-flight trajectory carry real silence.
    amp = (np.maximum(rps_frames, 0.0) / max(params.amp_rps_ref, 1e-6)) ** params.amp_rps_exponent

    # Floor: static shape, a slow level process, and a slow tilt process.
    level_gp = sample_gp(
        rng, 1, n_frames, dt=dt, tau=params.floor_gp_tau_s, std=params.floor_gp_std_db
    )[0]
    tilt_gp = sample_gp(
        rng, 1, n_frames, dt=dt, tau=params.floor_tilt_gp_tau_s, std=params.floor_tilt_gp_std
    )[0]
    octaves = np.log2(np.maximum(freqs, FLOOR_SHAPE_F_MIN) / FLOOR_TILT_REF_HZ)
    floor_db = (
        params.floor_mean_db
        + floor_shape_db(params, freqs)[None, :]
        + level_gp[:, None]
        + tilt_gp[:, None] * octaves[None, :]
    )
    floor = 10.0 ** (floor_db / 10.0) * amp.mean(axis=0)[:, None]

    # Harmonic amplitudes: a per-rotor common process mixed with one process
    # per line. The coherence is a variance split, so the total variance of
    # every line's process is harm_gp_std_db^2 whatever the mixing.
    rho = float(np.clip(params.harm_coherence, 0.0, 1.0))
    common = sample_gp(
        rng, n_rotors, n_frames, dt=dt, tau=params.harm_gp_tau_s, std=params.harm_gp_std_db
    )
    private = sample_gp(
        rng,
        n_rotors * n_harm,
        n_frames,
        dt=dt,
        tau=params.harm_gp_tau_s,
        std=params.harm_gp_std_db,
    ).reshape(n_rotors, n_harm, n_frames)
    harm_gp = np.sqrt(rho) * common[:, None, :] + np.sqrt(1.0 - rho) * private

    lines = np.zeros((n_rotors, n_frames, n_freqs), dtype=np.float64)
    gamma_min = GAMMA_MIN_BINS * df
    frame_idx = np.arange(n_frames)
    k_all = np.arange(1, n_harm + 1, dtype=np.float64)
    for r in range(n_rotors):
        power = 10.0 ** ((params.harm_mean_db + params.profile_db[r][:, None] + harm_gp[r]) / 10.0)
        power = power * amp[r][None, :]  # (K, N)
        gammas = np.maximum(params.gamma0[r] + params.gamma_slope[r] * k_all, gamma_min)
        centers_all = k_all[:, None] * rps_frames[r][None, :]  # (K, N)
        live_all = (centers_all > df) & (centers_all < nyquist - gammas[:, None])

        # Harmonics are rendered in buckets of equal support width. A line's
        # support grows with its own width, so a per-harmonic loop would spend
        # all of its time in Python overhead on tiny arrays; rounding the width
        # up to the next power of two puts every harmonic in one of about eight
        # buckets, each rendered in one vectorized block. The extra bins a
        # rounded-up width covers hold a small Lorentzian value, so the result
        # is the same spectrum, not an approximation of it.
        #
        # The frequency axis is padded by the widest support on each side, so a
        # line near an edge writes into the pad instead of needing a bounds
        # mask, and an out-of-band harmonic is silenced through its power
        # instead of through its indices. Both keep the inner block free of
        # boolean indexing, which is where a scatter of this size spends its
        # time.
        half_w_all = np.ceil(LORENTZ_SUPPORT_HWHM * gammas / df).astype(np.int64)
        bucket_w = np.maximum(1, 1 << np.ceil(np.log2(np.maximum(half_w_all, 1))).astype(np.int64))
        n_pad = int(bucket_w.max())
        n_wide = n_freqs + 2 * n_pad
        acc = np.zeros(n_frames * n_wide, dtype=np.float64)
        live_power = np.where(live_all, power, 0.0)
        for width in np.unique(bucket_w):
            sel = np.flatnonzero((bucket_w == width) & live_all.any(axis=1))
            if sel.size == 0:
                continue
            offsets = np.arange(-int(width), int(width) + 1)
            centers = centers_all[sel]  # (S, N)
            # A harmonic above Nyquist is silenced through its power, and its
            # center is then meaningless — but it still carries an index, and
            # k * rps reaches far past the grid at the top of the speed range.
            # Clamping the center keeps every index inside the padded axis;
            # the values written there are zero either way.
            base = np.clip(np.rint(centers / df), 0.0, n_freqs - 1).astype(np.int64) + n_pad
            bins = base[:, :, None] + offsets
            delta = (bins - n_pad) * df - centers[:, :, None]
            gamma = gammas[sel][:, None, None]
            dens = gamma / (np.pi * LORENTZ_TRUNC_NORM * (delta * delta + gamma * gamma))
            contrib = live_power[sel][:, :, None] * dens
            flat = frame_idx[None, :, None] * n_wide + bins
            acc += np.bincount(
                flat.reshape(-1), weights=contrib.reshape(-1), minlength=n_frames * n_wide
            )
        lines[r] = acc.reshape(n_frames, n_wide)[:, n_pad : n_pad + n_freqs]

    return {
        "floor": floor,
        "lines": lines,
        "harm_gp": harm_gp,
        "floor_gp": level_gp,
        "floor_tilt_gp": tilt_gp,
        "amp": amp,
    }


# ── Overlap-add synthesis ───────────────────────────────────────────────────


def _ola_filter(x: np.ndarray, gain: np.ndarray, n_fft: int, hop: int) -> np.ndarray:
    """Filter ``x`` ``(M, T)`` with per-frame spectral gains ``(M, N, F)``.

    Hann analysis and synthesis windows at a quarter-window hop reconstruct
    exactly when the gain is 1, so the output's spectrum is the input's times
    ``gain^2`` with no scaling constant to track. Every channel goes through one
    pair of transforms together, because a batched transform over all channels
    and frames costs far less than one call per channel.
    """
    x = np.atleast_2d(x)
    gain = gain if gain.ndim == 3 else gain[None]
    window = np.hanning(n_fft + 1)[:n_fft]
    n_mics, n_frames = gain.shape[0], gain.shape[1]
    need = (n_frames - 1) * hop + n_fft
    if x.shape[1] < need:
        x = np.pad(x, ((0, 0), (0, need - x.shape[1])))
    starts = np.arange(n_frames) * hop
    frames = np.stack([x[:, s : s + n_fft] for s in starts], axis=1) * window
    spec = np.fft.rfft(frames, axis=-1) * gain
    out_frames = np.fft.irfft(spec, n=n_fft, axis=-1) * window

    out = np.zeros((n_mics, need), dtype=np.float64)
    norm = np.zeros(need, dtype=np.float64)
    w2 = window * window
    for i, s in enumerate(starts):
        out[:, s : s + n_fft] += out_frames[:, i]
        norm[s : s + n_fft] += w2
    return out / np.maximum(norm, 1e-8)


#: Sample rate of the phase-noise random walk before it is interpolated to the
#: audio grid. A line's phase noise has the bandwidth of its own linewidth, tens
#: of hertz at the top of a comb, so a kilohertz path is far finer than needed
#: and costs a thirty-second of the random numbers.
PHASE_NOISE_FS = 1000.0


def coherent_lines(
    params: StochasticParams,
    rps: np.ndarray,
    psd: dict[str, np.ndarray],
    gains: np.ndarray,
    *,
    rng: np.random.Generator,
) -> np.ndarray:
    """``(M, T)`` harmonic lines rendered as TONES, not as narrowband noise.

    Why this exists. Filtering white noise through a Lorentzian gives a signal
    with exactly the right power spectrum and the wrong statistics: every frame
    draws its magnitude from a Rayleigh distribution, so each line flickers by
    about 5.2 dB frame to frame and dips 20 dB below its own mean several times
    a second. A rotor harmonic does not do that. It is a tone whose amplitude is
    steady and whose PHASE wanders, and the wandering phase is what gives it a
    Lorentzian line. The two descriptions share a spectrum and differ in
    everything a comb detector reads.

    So each harmonic is built as ``A_k(t) cos(k phi_r(t) + b_rk(t))`` with
    ``phi_r`` the shaft phase from the trajectory, ``A_k`` the same amplitude the
    spectrum model asks for, and ``b_rk`` a Wiener process. A Wiener phase of
    diffusion ``D`` rad^2/s gives a Lorentzian of half width ``D / (4 pi)`` Hz,
    so ``D = 4 pi gamma`` reproduces the linewidth the model asked for.

    The returned signal is unscaled; :func:`synthesize` sets its level from the
    ratio of the mean line spectrum to the mean floor spectrum.
    """
    rps = np.atleast_2d(np.asarray(rps, dtype=np.float64))
    n_rotors, n_samples = rps.shape
    sr = float(params.sample_rate)
    nyquist = sr / 2.0
    n_mics = gains.shape[0]
    n_harm = params.n_harmonics

    t_audio = np.arange(n_samples) / sr
    t_frames = np.linspace(0.0, n_samples / sr, psd["harm_gp"].shape[2])
    # Shaft phase: the running integral of the instantaneous rate.
    phase = 2.0 * np.pi * np.cumsum(rps, axis=1) / sr

    n_slow = max(int(np.ceil(n_samples / sr * PHASE_NOISE_FS)) + 2, 2)
    t_slow = np.arange(n_slow) / PHASE_NOISE_FS
    dt_slow = 1.0 / PHASE_NOISE_FS

    per_rotor = np.zeros((n_rotors, n_samples), dtype=np.float64)
    for r in range(n_rotors):
        power_frames = 10.0 ** (
            (params.harm_mean_db + params.profile_db[r][:, None] + psd["harm_gp"][r]) / 10.0
        ) * psd["amp"][r][None, :]
        for i in range(n_harm):
            k = i + 1
            centers = k * rps[r]
            live = centers < nyquist
            if not live.any():
                continue
            gamma = max(params.gamma0[r] + params.gamma_slope[r] * k, 1e-3)
            steps = rng.normal(0.0, np.sqrt(4.0 * np.pi * gamma * dt_slow), size=n_slow)
            walk = np.cumsum(steps)
            b = np.interp(t_audio, t_slow, walk) + rng.uniform(0.0, 2.0 * np.pi)
            amplitude = np.sqrt(2.0 * np.maximum(np.interp(t_audio, t_frames, power_frames[i]), 0.0))
            per_rotor[r] += np.where(live, amplitude, 0.0) * np.cos(k * phase[r] + b)

    out = np.empty((n_mics, n_samples), dtype=np.float64)
    for m in range(n_mics):
        out[m] = np.tensordot(np.sqrt(gains[m]), per_rotor, axes=(0, 0))
    return out


def synthesize(
    params: StochasticParams,
    rps: np.ndarray,
    *,
    rng: np.random.Generator,
    n_mics: int = 1,
    mic_gain_db: tuple[float, float] = (0.0, 0.0),
    n_fft: int = DEFAULT_N_FFT,
    hop: int | None = None,
    normalize_rms: float | None = 0.1,
    level_mode: str = "window",
    line_mode: str = "stochastic",
) -> tuple[np.ndarray, dict[str, Any]]:
    """Render one clip.

    Args:
        params: the parameter set.
        rps: ``(R, T)`` rotor speeds in rev/s at the audio rate.
        rng: the source of every random draw.
        n_mics: how many channels. Each is an independent realization of its
            own spectrum; the spectra differ by the per-microphone line gains.
        mic_gain_db: range of the per-(microphone, rotor) line gain draw.
        n_fft, hop: the synthesis resolution. ``hop`` defaults to a quarter
            window, which the Hann window needs for exact overlap-add.
        normalize_rms: scale the finished clip to this root-mean-square level,
            or ``None`` to keep the model's own arbitrary scale.
        level_mode: what ``normalize_rms`` refers to. ``"window"`` normalizes
            the window itself, so every window leaves at the same level
            whatever its rotors are doing. ``"flight"`` treats the number as
            the level the window would have AT THE REFERENCE SPEED and scales
            by the window's own speed-dependent amplitude instead, so a
            stopped-rotor window comes out quiet and a cruise window loud —
            which is what a real recording does, because one recorder gain
            covers a whole flight.
        line_mode: how the harmonic lines are realized. ``"stochastic"``
            filters white noise through the whole spectrum, which gives the
            right power spectrum and Rayleigh magnitudes — every line flickers
            by about 5.2 dB per frame. ``"coherent"`` renders the lines as tones
            with a wandering phase (:func:`coherent_lines`) over a filtered-noise
            floor, which is the same spectrum with the statistics a rotor
            actually produces.

    Returns:
        ``(audio (n_mics, T) float32, diagnostics)``. The diagnostics carry the
        model spectrum and the Gaussian-process draws, which is what the
        notebook plots next to the realized spectrogram.
    """
    rps = np.atleast_2d(np.asarray(rps, dtype=np.float64))
    n_rotors, n_samples = rps.shape
    sr = params.sample_rate
    hop = int(hop or n_fft // DEFAULT_HOP_DIV)

    # One window of padding at each end keeps the overlap-add taper away from
    # the clip, and the frame grid is placed on the padded signal.
    pad = n_fft
    n_padded = n_samples + 2 * pad
    n_frames = 1 + int(np.ceil(max(n_padded - n_fft, 0) / hop))
    frame_times = (np.arange(n_frames) * hop + n_fft / 2.0 - pad) / sr
    clip_t = np.arange(n_samples) / sr
    rps_frames = np.stack(
        [
            np.interp(frame_times, clip_t, rps[r], left=rps[r][0], right=rps[r][-1])
            for r in range(n_rotors)
        ]
    )

    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    psd = build_psd(params, rps_frames, freqs, dt=hop / sr, rng=rng)

    lo, hi = mic_gain_db
    gains = 10.0 ** (rng.uniform(lo, hi, size=(n_mics, n_rotors)) / 10.0)
    # (M, N, F): the floor is common, and each microphone weighs the rotors'
    # lines by its own gains.
    if line_mode == "coherent":
        floor_spec = np.repeat(psd["floor"][None], n_mics, axis=0)
        white = rng.standard_normal((n_mics, n_padded))
        floor_audio = _ola_filter(white, np.sqrt(np.maximum(floor_spec, 0.0)), n_fft, hop)[
            :, pad : pad + n_samples
        ]
        line_audio = coherent_lines(params, rps, psd, gains, rng=rng)
        # Put the lines at the level the spectrum asks for. A filtered-noise
        # signal's variance is the mean of its spectrum over bins, so the ratio
        # of the two mean spectra is the ratio the two parts must end up with —
        # and taking it from the floor's realized variance keeps every constant
        # of the transform out of the arithmetic.
        floor_mean = float(np.mean(psd["floor"])) or 1.0
        audio = np.empty((n_mics, n_samples), dtype=np.float32)
        for m in range(n_mics):
            want = float(np.mean(np.tensordot(gains[m], psd["lines"], axes=(0, 0)))) / floor_mean
            have = float(np.var(line_audio[m])) / max(float(np.var(floor_audio[m])), 1e-30)
            scale = np.sqrt(want / have) if have > 0 else 0.0
            audio[m] = (floor_audio[m] + scale * line_audio[m]).astype(np.float32)
    else:
        spectrum = psd["floor"][None] + np.tensordot(gains, psd["lines"], axes=(1, 0))
        white = rng.standard_normal((n_mics, n_padded))
        y = _ola_filter(white, np.sqrt(np.maximum(spectrum, 0.0)), n_fft, hop)
        audio = y[:, pad : pad + n_samples].astype(np.float32)

    if normalize_rms is not None:
        rms = float(np.sqrt(np.mean(np.square(audio)))) or 1.0
        gain = float(normalize_rms) / rms
        if level_mode == "flight":
            # Divide out only the part of the level that does NOT come from the
            # rotor speed, then put the speed's own contribution back. The
            # window's mean amplitude factor is (rps / ref) ** exponent, so a
            # window at half the reference speed leaves about 5.7 times quieter
            # instead of at the same level as cruise.
            gain *= float(np.mean(psd["amp"]))
        audio = (audio * gain).astype(np.float32)

    diag: dict[str, Any] = {
        "freqs": freqs,
        "frame_times": frame_times,
        "rps_frames": rps_frames,
        "mic_gains": gains,
        **psd,
    }
    return audio, diag


def model_psd_db(diag: dict[str, Any], mic: int = 0) -> np.ndarray:
    """``(N, F)`` model spectrum in dB for one microphone, from the diagnostics."""
    spectrum = diag["floor"] + np.tensordot(diag["mic_gains"][mic], diag["lines"], axes=(0, 0))
    return 10.0 * np.log10(np.maximum(spectrum, 1e-30))


# ── Pool ────────────────────────────────────────────────────────────────────


@dataclass
class _FlightCache:
    rps: np.ndarray
    t_low: np.ndarray
    uses: int = 0
    #: The flight's own hover speed (its 90th percentile). The amplitude law is
    #: written against THIS, not against a fixed 80 rev/s, so the level says
    #: "this aircraft is at such a fraction of its own hover" instead of "the
    #: speed is such a number". Without it a wide speed range would make the
    #: level a giveaway for the absolute speed — a shortcut, and a false one,
    #: since a small fast drone is not quieter than a big slow one.
    hover: float = 80.0


class StochasticNoisePool:
    """Stochastic rotor-noise source (``kind: stochastic``).

    Every window draws a fresh parameter set, so no two windows of a stream
    share a timbre, a floor color, or a wander rate. The
    ``sample_timeframe(rng, duration_s) -> td.Frame`` interface is the one the
    other noise pools use, and synthesis is cheap enough for the DataLoader
    workers.
    """

    def __init__(
        self,
        *,
        sample_rate: int = 16000,
        duration_s: float = 1.0,
        n_harmonics: int = 80,
        n_harm_max: int = 200,
        n_mics: int = 8,
        n_rotors: int = 4,
        rps_kind: str = "synthetic_intermittent",
        aggressiveness: float | tuple[float, float] = 1.0,
        flight_fs: float = 200.0,
        flight_reuse: int = 32,
        flight_phases: dict[str, Any] | None = None,
        drone_profile_range: tuple[float, float] = (0.0, 1.0),
        mic_gain_db: tuple[float, float] = (-12.0, 0.0),
        amp_rps_exponent: float = 2.5,
        amp_rps_ref: float = 80.0,
        rps_scale_range: tuple[float, float] = (1.0, 1.0),
        normalize_rms: float | tuple[float, float] = 0.1,
        level_mode: str = "window",
        line_mode: str = "stochastic",
        n_fft: int = DEFAULT_N_FFT,
        ranges: StochasticRanges | dict[str, Any] | None = None,
        seed: int = 0,
    ):
        self.sample_rate = int(sample_rate)
        self.chunk_s = float(duration_s)
        self.n_harmonics = int(n_harmonics)
        self.n_harm_max = int(n_harm_max)
        self.n_mics = int(n_mics)
        self.n_rotors = int(n_rotors)
        self.rps_kind = str(rps_kind)
        # How hard the aircraft is flown. A pair is a range drawn per window.
        # The knob scales the maneuver modes, so it sets how far the four rotors
        # separate — and that is what decides how the four combs interleave. On
        # the real validation split the per-rotor spread of a flight frame
        # averages 13.7 rev/s; at aggressiveness 1.0 this model gives 9.4, and
        # 2.0 to 3.0 brackets the real figure.
        self.aggressiveness: float | tuple[float, float] = (
            (float(aggressiveness[0]), float(aggressiveness[1]))
            if isinstance(aggressiveness, (list, tuple))
            else float(aggressiveness)
        )
        self.flight_fs = float(flight_fs)
        self.flight_reuse = int(flight_reuse)
        # Overrides for the flight-phase durations and the warm-up idle level
        # (``rps_synthesis.FlightPhaseRanges``). The default idle band is 0.38
        # to 0.52 of hover, so a default stream never shows a rotor between 10
        # and 30 rev/s — and that is most of what a real ramp passes through.
        self.flight_phases = dict(flight_phases) if flight_phases else None
        self.drone_profile_range = (float(drone_profile_range[0]), float(drone_profile_range[1]))
        self.mic_gain_db = (float(mic_gain_db[0]), float(mic_gain_db[1]))
        self.amp_rps_exponent = float(amp_rps_exponent)
        self.amp_rps_ref = float(amp_rps_ref)
        # Per-window multiplier on the whole trajectory. A synthetic family
        # renders its audio FROM the labels, so scaling the trajectory moves
        # every comb line and leaves the floor's shape where it is — which is
        # what a real speed change does, and what the frequency-scaling
        # augmentation cannot do (resampling moves the floor too). Its purpose
        # is to destroy the speed prior: over a wide enough range no cruise
        # level is more likely than another, and comb spacing becomes the only
        # thing a model can read the speed from.
        self.rps_scale_range = (float(rps_scale_range[0]), float(rps_scale_range[1]))
        # Output level, as a root-mean-square target. A pair is a log-uniform
        # range drawn per window. The other synthetic pools all normalize to a
        # fixed 0.1 while a real recording sits near 0.04, so a synthetic-only
        # model trains 8 dB away from the data it is asked to read; a range
        # covers the real level and takes level away as a cue at the same time.
        self.normalize_rms = (
            (float(normalize_rms[0]), float(normalize_rms[1]))
            if isinstance(normalize_rms, (list, tuple))
            else float(normalize_rms)
        )
        # "window" normalizes every window to the same level, so level says
        # nothing about whether the rotors are turning. "flight" keeps the
        # speed's own contribution, which is how a real recording behaves and
        # what lets a model tell a stopped rotor from a running one by level as
        # well as by structure.
        self.level_mode = str(level_mode)
        # "stochastic" filters white noise through the whole spectrum, so every
        # line's magnitude is Rayleigh and flickers about 5.2 dB per frame.
        # "coherent" renders the lines as tones with a wandering phase over a
        # filtered-noise floor — the same spectrum, and the statistics a rotor
        # actually produces.
        self.line_mode = str(line_mode)
        self.n_fft = int(n_fft)
        self.ranges = (
            ranges if isinstance(ranges, StochasticRanges) else StochasticRanges.from_dict(ranges)
        )
        self._base_seed = int(seed)
        self._flight: _FlightCache | None = None
        # Interface parity with the other pools: the analytic model has no
        # geometry, and the frame carries placeholders.
        self.mic_pos = np.zeros((self.n_mics, 3), dtype=np.float64)
        self.rotor_pos = np.zeros((self.n_rotors, 3), dtype=np.float64)

    @classmethod
    def from_config(cls, cfg: Any, *, duration_s: float, sample_rate: int) -> StochasticNoisePool:
        def g(key: str, default: Any = None) -> Any:
            if isinstance(cfg, dict):
                return cfg.get(key, default)
            return getattr(cfg, key, default)

        rps = g("rps", {}) or {}
        ranges = g("ranges")
        if ranges is not None and not isinstance(ranges, dict):
            from data_processing.generated_noise import _to_plain

            ranges = _to_plain(ranges)

        def pair(key: str, default: tuple[float, float]) -> tuple[float, float]:
            v = g(key, default)
            return (float(v[0]), float(v[1]))

        return cls(
            sample_rate=sample_rate,
            duration_s=duration_s,
            n_harmonics=int(g("n_harmonics", 80)),
            n_harm_max=int(g("n_harm_max", 200)),
            n_mics=int(g("n_mics", 8)),
            n_rotors=int(g("n_rotors", 4)),
            rps_kind=str(rps.get("kind", "synthetic_intermittent")),
            aggressiveness=(
                (float(rps["aggressiveness_range"][0]), float(rps["aggressiveness_range"][1]))
                if rps.get("aggressiveness_range") is not None
                else float(rps.get("aggressiveness", 1.0))
            ),
            flight_fs=float(rps.get("flight_fs", 200.0)),
            flight_reuse=int(rps.get("flight_reuse", 32)),
            flight_phases=(
                {k: tuple(v) for k, v in dict(rps["phases"]).items()}
                if rps.get("phases") is not None
                else None
            ),
            drone_profile_range=pair("drone_profile_range", (0.0, 1.0)),
            mic_gain_db=pair("mic_gain_db", (-12.0, 0.0)),
            amp_rps_exponent=float(g("amp_rps_exponent", 2.5)),
            amp_rps_ref=float(g("amp_rps_ref", 80.0)),
            rps_scale_range=pair("rps_scale_range", (1.0, 1.0)),
            normalize_rms=(
                pair("normalize_rms_range", (0.0, 0.0))
                if g("normalize_rms_range") is not None
                else float(g("normalize_rms", 0.1))
            ),
            level_mode=str(g("level_mode", "window")),
            line_mode=str(g("line_mode", "stochastic")),
            n_fft=int(g("n_fft", DEFAULT_N_FFT)),
            ranges=ranges,
            seed=int(g("seed", 0)),
        )

    def close(self) -> None:  # interface parity with GeneratedNoisePool
        return None

    def sample_rps(self, rng: np.random.Generator, duration_s: float) -> np.ndarray:
        """``(R, T)`` rotor speeds at the audio rate for one window.

        ``synthetic_intermittent`` draws a cruise window directly;
        ``full_flight`` windows a cached low-rate flight, so successive windows
        visit the ground, warm-up, takeoff, cruise and landing phases in
        proportion to their durations. Either way the window is multiplied by
        one draw from ``rps_scale_range``, which is exact: a stopped rotor stays
        stopped, and every other speed moves with its own comb.
        """
        n_samples = int(round(duration_s * self.sample_rate))
        # Log-uniform: the scale is a RATIO, and a decade of speed sampled
        # linearly would put four fifths of its draws in the top half.
        lo_s, hi_s = self.rps_scale_range
        scale = (
            float(np.exp(rng.uniform(np.log(lo_s), np.log(hi_s))))
            if lo_s > 0.0 and hi_s > lo_s
            else float(lo_s)
        )
        self._hover = self.amp_rps_ref * scale
        aggressiveness = (
            float(rng.uniform(*self.aggressiveness))
            if isinstance(self.aggressiveness, tuple)
            else self.aggressiveness
        )
        if self.rps_kind != "full_flight":
            blend = float(rng.uniform(*self.drone_profile_range))
            return (
                scale
                * rps_synthesis.generate_intermittent_batch(
                    1,
                    duration_s,
                    self.sample_rate,
                    drone_profile=blend,
                    aggressiveness=self.aggressiveness,
                    rng=rng,
                )[0]
            )

        if self._flight is None or self._flight.uses >= self.flight_reuse:
            blend = float(rng.uniform(*self.drone_profile_range))
            phases = (
                rps_synthesis.FlightPhaseRanges(**self.flight_phases)
                if self.flight_phases
                else None
            )
            flight = rps_synthesis.generate_full_flight(
                None,
                self.flight_fs,
                drone_profile=blend,
                aggressiveness=aggressiveness,
                phases=phases,
                rng=rng,
            )
            self._flight = _FlightCache(
                rps=flight,
                t_low=np.arange(flight.shape[1]) / self.flight_fs,
                hover=float(np.percentile(flight, 90.0)),
            )
        self._flight.uses += 1
        flight, t_low = self._flight.rps, self._flight.t_low
        max_start = max(0.0, float(t_low[-1]) - duration_s)
        start_s = float(rng.uniform(0.0, max_start)) if max_start > 0 else 0.0
        t_win = start_s + np.arange(n_samples) / self.sample_rate
        self._hover = max(scale * float(self._flight.hover), 1.0)
        window = np.stack([np.interp(t_win, t_low, flight[r]) for r in range(flight.shape[0])])
        return scale * window

    def render(
        self, rng: np.random.Generator, duration_s: float
    ) -> tuple[np.ndarray, np.ndarray, StochasticParams, dict[str, Any]]:
        """``(audio (M, T), rps (R, T), params, diagnostics)`` for one window."""
        rps = self.sample_rps(rng, duration_s)
        # Size the comb to the clip's own speed, so a slow aircraft still has a
        # comb that reaches the top of the band and a fast one does not pay for
        # harmonics that fall past Nyquist. Without this a 20 rev/s clip at 80
        # harmonics would carry a comb that stops at 1.6 kHz.
        hover = float(getattr(self, "_hover", self.amp_rps_ref))
        n_harm = int(np.clip(np.ceil(self.sample_rate / 2.0 / max(hover, 1.0)), 40, self.n_harm_max))
        params = sample_params(
            rng,
            self.ranges,
            n_rotors=rps.shape[0],
            n_harmonics=n_harm,
            sample_rate=self.sample_rate,
        )
        # Linewidth scales with the aircraft too. The half width of harmonic k
        # is the shaft's own speed jitter times k, and a shaft that turns at
        # 200 rev/s does not jitter by the same ABSOLUTE amount as one at 20 —
        # it jitters by the same fraction. Scaling gamma with the hover keeps
        # the RELATIVE linewidth constant across drone sizes, which is what
        # makes a small fast aircraft a different aircraft and not just a
        # sharper version of a big one.
        size = float(hover) / max(self.amp_rps_ref, 1e-6)
        params = params.with_(
            amp_rps_exponent=self.amp_rps_exponent,
            amp_rps_ref=float(hover),
            gamma0=params.gamma0 * size,
            gamma_slope=params.gamma_slope * size,
        )
        if isinstance(self.normalize_rms, tuple):
            lo, hi = self.normalize_rms
            level = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
        else:
            level = self.normalize_rms
        audio, diag = synthesize(
            params,
            rps,
            rng=rng,
            n_mics=self.n_mics,
            mic_gain_db=self.mic_gain_db,
            n_fft=self.n_fft,
            normalize_rms=level,
            level_mode=self.level_mode,
            line_mode=self.line_mode,
        )
        return audio, rps.astype(np.float32), params, diag

    def sample_timeframe(self, rng: np.random.Generator, duration_s: float) -> td.Frame:
        audio, rps, _, _ = self.render(rng, duration_s)
        audio_us = td.uniform(
            np.ascontiguousarray(audio), self.sample_rate, dims=("mic", "time"), t_start=0.0
        )
        t = np.arange(audio.shape[-1], dtype=np.float64) / self.sample_rate
        rps_es = td.events(t, np.ascontiguousarray(rps), dims=("rotor", "time"), t_start=0.0)
        return make_recording_frame(
            {"audio": audio_us, "rps": rps_es},
            meta={"recording_id": "stochastic"},
            mic_pos=self.mic_pos,
            rotor_pos=self.rotor_pos,
        )


__all__ = [
    "StochasticNoisePool",
    "StochasticParams",
    "StochasticRanges",
    "build_psd",
    "calibrate_floor",
    "floor_shape_db",
    "line_peak_db",
    "model_psd_db",
    "sample_gp",
    "sample_params",
    "synthesize",
]
