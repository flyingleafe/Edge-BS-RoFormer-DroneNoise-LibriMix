"""Synthetic RPS-trajectory generation via OU processes in quadrotor control-mode space.

A quadrotor controls four degrees of freedom — *collective thrust*, *roll*,
*pitch*, *yaw* — through four motors related by a fixed linear mixer ``B``.  Rather
than model four correlated rotor-speed channels directly, we model the four
**control modes** as *independent* mean-reverting (Ornstein–Uhlenbeck) processes
and recover rotor speeds via ``w = B @ m``.  The mixer's structure then produces
the strong inter-rotor correlation seen in real flights "for free".

Each mode ``m_k(t)`` is a scalar OU process

    dm = (1/tau) (mu - m) dt + sigma_drive dW ,

parametrised here by its **stationary mean** ``mu``, **stationary std**
``sigma`` (= ``sigma_drive * sqrt(tau / 2)``) and **correlation time** ``tau``.
These three numbers per mode are the entire model.  Calibrated defaults
(:data:`DEFAULT_CONFIG`) come from the DREGON ``in_flight_noise`` recordings
(929 Hz motor telemetry; see ``notebooks``/the project report); Michael's 29 Hz
telemetry is too coarse to estimate the sub-second maneuver time constants.

Control modes recovered from real flights show:
  * a ~80 RPS **common-mode** hover level with a few-RPS slow wander;
  * small, persistent **trim biases** on the maneuver modes (notably yaw ≈ +2.5,
    the CW/CCW rotor-drag imbalance), about which they fluctuate;
  * a clean ordering of *aggressiveness* by maneuver-mode std (hovering <
    translation < spinning), which the :func:`generate` ``aggressiveness`` knob
    reproduces by scaling the dynamic stds.

The public surface is :class:`RPSSynthConfig`, :func:`fit_config`,
:func:`generate` and :func:`generate_batch`.  Output is a ``(4, M)`` array of
rotor speeds in revolutions/second, the same convention as ``rps.npy`` and the
``rps`` track elsewhere in the project.

Intermittent ("pilot + airframe") model
---------------------------------------
A plain OU process wanders *continuously*, but a human-piloted drone is mostly
**steady**, holding attitude for seconds at a time and only occasionally
commanding a brief maneuver.  Measured on the real recordings, the differential
control modes are active only ~4–16 % of the time, with maneuver onsets every
5–14 s.  The :func:`generate_intermittent` model reproduces this with a two-layer
"pilot + airframe" structure per mode:

  1. an **intermittent setpoint** — a telegraph signal that holds at the trim
     value and, at Poisson-distributed onsets, deflects by a random amount for a
     short random duration before returning (the pilot's stick command);
  2. a **first-order motor/airframe lag** (time constant ``motor_tau``) that
     low-passes the setpoint, rounding its step edges as rotor inertia would;
  3. a small **cruise jitter** OU term so holds are not perfectly flat.

Two knobs control it.  ``aggressiveness`` scales the maneuver rate and amplitude;
``drone_profile`` in ``[0, 1]`` blends a :data:`DREGON_PROFILE` (small, fast,
~80 RPS hover) and a :data:`MICHAELS_PROFILE` (DJI Matrice 100 — larger, slower
motor response, ~72 RPS hover), with ``0.5`` an in-between airframe.  Public
surface: :class:`ManeuverModeParams`, :class:`DroneProfile`,
:func:`blend_profiles`, :func:`generate_intermittent`,
:func:`generate_intermittent_batch`.

Synthetic comb window
---------------------
The tracking evaluations need a window whose rotor speeds are known *exactly*,
not merely measured.  :func:`synth_comb_window` renders one: an OU trajectory
(above), optionally band-limited by a shaft-inertia low-pass, driving a
locked-phase harmonic comb with 2-blade blade-pass structure, plus white noise
at a given comb-to-noise ratio.  It returns a :class:`SynthCombWindow` — the
audio, the ground-truth trajectory on the audio grid, and the draw's
provenance.  The random draws happen in a fixed order, so one seed reproduces
one window bit-exactly.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np

# The mixer constants and mode projections moved to ``tracking.rotors`` (the
# tracking stack needs them and must not import data_processing); re-exported
# here so existing ``rps_synthesis`` consumers keep working.
from tracking.rotors import MIXER as MIXER
from tracking.rotors import MODE_NAMES as MODE_NAMES
from tracking.rotors import NUM_ROTORS as NUM_ROTORS
from tracking.rotors import modes_from_rps as modes_from_rps
from tracking.rotors import rps_from_modes as rps_from_modes


@dataclass(frozen=True)
class OUModeParams:
    """Stationary parameters of one OU control mode.

    Attributes:
        mean: stationary mean (RPS units, in mode space).
        std: stationary standard deviation.
        tau: correlation (relaxation) time in seconds.
    """

    mean: float
    std: float
    tau: float


@dataclass(frozen=True)
class RPSSynthConfig:
    """Full synthesizer configuration — one :class:`OUModeParams` per control mode."""

    common: OUModeParams
    roll: OUModeParams
    pitch: OUModeParams
    yaw: OUModeParams
    rps_min: float = 30.0  # physical floor (below = takeoff/landing, not in-flight)
    rps_max: float = 120.0  # physical ceiling

    @property
    def modes(self) -> tuple[OUModeParams, ...]:
        return (self.common, self.roll, self.pitch, self.yaw)


# Calibrated from DREGON in_flight_noise (median over the 6 recordings; common
# std/tau from the gentler free-flight subset so aggressiveness=1 is a typical,
# not extreme, flight).  See project report for the per-recording breakdown.
DEFAULT_CONFIG = RPSSynthConfig(
    common=OUModeParams(mean=80.0, std=4.0, tau=0.70),
    roll=OUModeParams(mean=0.0, std=0.70, tau=0.60),
    pitch=OUModeParams(mean=0.0, std=0.85, tau=0.75),
    yaw=OUModeParams(mean=2.5, std=1.40, tau=1.00),
    rps_min=30.0,
    rps_max=120.0,
)


def _estimate_mode_params(m: np.ndarray, dt: float) -> OUModeParams:
    """Estimate ``(mean, std, tau)`` of a 1-D OU series sampled at step ``dt``.

    ``tau`` comes from the lag-1 autocorrelation ``rho1 = exp(-dt/tau)``.
    """
    mu = float(np.mean(m))
    x = m - mu
    var = float(np.var(x))
    if x.size > 2 and var > 0.0:
        rho1 = float(np.mean(x[:-1] * x[1:]) / var)
        rho1 = min(max(rho1, 1e-4), 1.0 - 1e-4)
        tau = float(dt / -np.log(rho1))
    else:
        tau = dt
    return OUModeParams(mean=mu, std=float(np.sqrt(var)), tau=tau)


def fit_config(
    traces: list[np.ndarray],
    dts: list[float],
    *,
    rps_min: float = 30.0,
    rps_max: float = 120.0,
    inflight_min_rps: float = 30.0,
) -> RPSSynthConfig:
    """Fit a :class:`RPSSynthConfig` from real ``(4, M)`` rotor-speed traces.

    Each trace is projected onto the control modes; per-mode ``(mean, std, tau)``
    are estimated per recording and aggregated by the **median** across
    recordings (robust to the differing flight types).  Samples whose
    rotor-mean RPS is below ``inflight_min_rps`` are dropped before fitting so
    takeoff/landing ramps do not contaminate the in-flight statistics.

    Args:
        traces: list of ``(4, M)`` rotor-speed arrays (rev/s).
        dts: matching list of sample periods (seconds) for each trace.
        rps_min, rps_max: physical clamp range stored on the returned config.
        inflight_min_rps: rotor-mean threshold for the in-flight mask.

    Returns:
        Calibrated :class:`RPSSynthConfig`.
    """
    if len(traces) != len(dts):
        raise ValueError("traces and dts must have the same length")
    per_mode: list[list[OUModeParams]] = [[] for _ in MODE_NAMES]
    for w, dt in zip(traces, dts, strict=True):
        w = np.asarray(w, dtype=np.float64)
        mask = w.mean(axis=0) > inflight_min_rps
        if mask.sum() < 10:
            continue
        m = modes_from_rps(w[:, mask])
        for k in range(NUM_ROTORS):
            per_mode[k].append(_estimate_mode_params(m[k], dt))
    if any(len(p) == 0 for p in per_mode):
        raise ValueError("no traces had enough in-flight samples to fit")

    def _median(params: list[OUModeParams]) -> OUModeParams:
        return OUModeParams(
            mean=float(np.median([p.mean for p in params])),
            std=float(np.median([p.std for p in params])),
            tau=float(np.median([p.tau for p in params])),
        )

    return RPSSynthConfig(
        common=_median(per_mode[0]),
        roll=_median(per_mode[1]),
        pitch=_median(per_mode[2]),
        yaw=_median(per_mode[3]),
        rps_min=rps_min,
        rps_max=rps_max,
    )


def _ou_path(
    params: OUModeParams,
    n: int,
    dt: float,
    rng: np.random.Generator,
    std_scale: float,
) -> np.ndarray:
    """Exact discrete-time OU sample path of length ``n`` at step ``dt``.

    Uses the exact transition ``x[i+1] = mu + phi (x[i] - mu) + eps`` with
    ``phi = exp(-dt/tau)`` and ``eps ~ N(0, sigma^2 (1 - phi^2))``; this is exact
    for any ``dt`` (no Euler discretisation error).  ``std_scale`` scales the
    *dynamic* stationary std (the aggressiveness knob); the mean is unchanged.
    """
    sigma = params.std * std_scale
    if params.tau <= 0.0:
        # Degenerate: white noise about the mean.
        return params.mean + sigma * rng.standard_normal(n)
    phi = float(np.exp(-dt / params.tau))
    step_std = sigma * np.sqrt(max(1.0 - phi * phi, 0.0))
    x = np.empty(n, dtype=np.float64)
    x[0] = params.mean + sigma * rng.standard_normal()
    noise = step_std * rng.standard_normal(n)
    for i in range(1, n):
        x[i] = params.mean + phi * (x[i - 1] - params.mean) + noise[i]
    return x


def generate(
    duration: float,
    fs: float,
    *,
    config: RPSSynthConfig = DEFAULT_CONFIG,
    aggressiveness: float = 1.0,
    mode_scales: dict[str, float] | None = None,
    rng: np.random.Generator | int | None = None,
) -> np.ndarray:
    """Generate one synthetic ``(4, M)`` rotor-speed trajectory.

    Args:
        duration: trajectory length in seconds.
        fs: sample rate of the trajectory (Hz).  ``M = round(duration * fs)``.
        config: OU parameters per control mode (default: DREGON-calibrated).
        aggressiveness: global multiplier on every mode's dynamic std.  ``1.0``
            is a typical free flight; ``< 1`` is gentle/near-hover, ``> 1`` is
            aggressive maneuvering.  Means (hover level, trim biases) are fixed.
        mode_scales: optional per-mode std multipliers (keys from
            :data:`MODE_NAMES`), applied *on top of* ``aggressiveness`` — e.g.
            ``{"yaw": 3.0}`` for a spin-dominated flight.
        rng: ``np.random.Generator``, integer seed, or ``None``.

    Returns:
        ``(4, M)`` array of rotor speeds (rev/s), clamped to
        ``[config.rps_min, config.rps_max]``, rotor order matching
        :data:`MIXER` rows.
    """
    if duration <= 0.0 or fs <= 0.0:
        raise ValueError("duration and fs must be positive")
    if aggressiveness < 0.0:
        raise ValueError("aggressiveness must be non-negative")
    generator = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
    n = int(round(duration * fs))
    dt = 1.0 / fs
    scales = {name: 1.0 for name in MODE_NAMES}
    if mode_scales:
        unknown = set(mode_scales) - set(MODE_NAMES)
        if unknown:
            raise ValueError(f"unknown mode_scales keys: {sorted(unknown)}")
        scales.update(mode_scales)

    m = np.stack(
        [
            _ou_path(params, n, dt, generator, aggressiveness * scales[name])
            for name, params in zip(MODE_NAMES, config.modes, strict=True)
        ]
    )
    w = rps_from_modes(m)
    return np.clip(w, config.rps_min, config.rps_max)


def generate_batch(
    n_trajectories: int,
    duration: float,
    fs: float,
    **kwargs,
) -> np.ndarray:
    """Generate ``n_trajectories`` trajectories, returning a ``(N, 4, M)`` array.

    A single ``rng`` (passed via ``**kwargs``) is threaded through all
    trajectories so the batch is reproducible from one seed.  Remaining keyword
    arguments are forwarded to :func:`generate`.
    """
    if n_trajectories <= 0:
        raise ValueError("n_trajectories must be positive")
    rng = kwargs.pop("rng", None)
    generator = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
    return np.stack(
        [generate(duration, fs, rng=generator, **kwargs) for _ in range(n_trajectories)]
    )


def scaled_config(config: RPSSynthConfig, factor: float) -> RPSSynthConfig:
    """Return a copy of ``config`` with all mode stds multiplied by ``factor``.

    Convenience for baking an aggressiveness level into a config (e.g. to store
    a "gentle" or "aggressive" preset) instead of passing it at call time.
    """
    return replace(
        config,
        common=replace(config.common, std=config.common.std * factor),
        roll=replace(config.roll, std=config.roll.std * factor),
        pitch=replace(config.pitch, std=config.pitch.std * factor),
        yaw=replace(config.yaw, std=config.yaw.std * factor),
    )


# =============================================================================
# Intermittent ("pilot + airframe") model
# =============================================================================


@dataclass(frozen=True)
class ManeuverModeParams:
    """Intermittent-model parameters for one control mode.

    Attributes:
        trim: hold setpoint (mode units, rev/s) — the value the pilot holds.
        cruise_std: std of the small jitter added while holding (never perfectly
            still).
        maneuver_std: std of a maneuver's peak deflection amplitude (rev/s).
        rate_hz: Poisson rate of maneuver onsets (events per second).
        mean_maneuver_s: mean duration of a single maneuver excursion (seconds).
            The active fraction of time is approximately ``rate_hz *
            mean_maneuver_s``.
    """

    trim: float
    cruise_std: float
    maneuver_std: float
    rate_hz: float
    mean_maneuver_s: float


@dataclass(frozen=True)
class DroneProfile:
    """A full intermittent-model profile — per-mode params plus airframe dynamics.

    ``motor_tau`` is the first-order motor/airframe lag that rounds the setpoint
    step edges; it is the principal knob distinguishing a small, snappy airframe
    (DREGON) from a large, sluggish one (Michael's DJI Matrice 100).
    """

    common: ManeuverModeParams
    roll: ManeuverModeParams
    pitch: ManeuverModeParams
    yaw: ManeuverModeParams
    motor_tau: float = 0.2  # s: first-order motor/airframe response lag
    cruise_tau: float = 0.4  # s: correlation time of the cruise jitter
    rps_min: float = 30.0
    rps_max: float = 120.0

    @property
    def modes(self) -> tuple[ManeuverModeParams, ...]:
        return (self.common, self.roll, self.pitch, self.yaw)


# Calibrated from the real recordings: maneuver structure (rate ~0.12/s, active
# ~8 %, so ~0.7 s excursions) from the intermittency analysis; trim biases and
# amplitudes from the control-mode projection (maneuver_std chosen so the overall
# per-mode std, diluted by the ~8 % active fraction, matches the measured OU std).
# DREGON: small quad, ~80 RPS hover, fast (short motor_tau).
DREGON_PROFILE = DroneProfile(
    common=ManeuverModeParams(
        trim=80.0, cruise_std=0.6, maneuver_std=12.0, rate_hz=0.14, mean_maneuver_s=0.7
    ),
    roll=ManeuverModeParams(
        trim=1.3, cruise_std=0.2, maneuver_std=3.0, rate_hz=0.12, mean_maneuver_s=0.7
    ),
    pitch=ManeuverModeParams(
        trim=-0.4, cruise_std=0.25, maneuver_std=3.5, rate_hz=0.12, mean_maneuver_s=0.7
    ),
    yaw=ManeuverModeParams(
        trim=2.5, cruise_std=0.3, maneuver_std=5.0, rate_hz=0.10, mean_maneuver_s=0.9
    ),
    motor_tau=0.15,
    cruise_tau=0.3,
)

# Michael's DJI Matrice 100: larger airframe, ~72 RPS hover, slower response
# (longer motor_tau), slightly less frequent but larger maneuvers.
MICHAELS_PROFILE = DroneProfile(
    common=ManeuverModeParams(
        trim=72.0, cruise_std=0.9, maneuver_std=14.0, rate_hz=0.10, mean_maneuver_s=1.0
    ),
    roll=ManeuverModeParams(
        trim=1.8, cruise_std=0.3, maneuver_std=4.0, rate_hz=0.10, mean_maneuver_s=1.0
    ),
    pitch=ManeuverModeParams(
        trim=1.0, cruise_std=0.3, maneuver_std=4.0, rate_hz=0.10, mean_maneuver_s=1.0
    ),
    yaw=ManeuverModeParams(
        trim=4.5, cruise_std=0.4, maneuver_std=4.5, rate_hz=0.10, mean_maneuver_s=1.2
    ),
    motor_tau=0.35,
    cruise_tau=0.5,
)


def _blend_mode(a: ManeuverModeParams, b: ManeuverModeParams, t: float) -> ManeuverModeParams:
    lerp = lambda x, y: (1.0 - t) * x + t * y  # noqa: E731
    return ManeuverModeParams(
        trim=lerp(a.trim, b.trim),
        cruise_std=lerp(a.cruise_std, b.cruise_std),
        maneuver_std=lerp(a.maneuver_std, b.maneuver_std),
        rate_hz=lerp(a.rate_hz, b.rate_hz),
        mean_maneuver_s=lerp(a.mean_maneuver_s, b.mean_maneuver_s),
    )


def blend_profiles(
    a: DroneProfile = DREGON_PROFILE,
    b: DroneProfile = MICHAELS_PROFILE,
    t: float = 0.5,
) -> DroneProfile:
    """Linearly interpolate between two drone profiles.

    ``t = 0`` returns ``a`` (DREGON-like), ``t = 1`` returns ``b``
    (Michael's-like), ``0.5`` an in-between airframe.  Every numeric field —
    trims, amplitudes, rates, durations, ``motor_tau`` and ``cruise_tau`` — is
    blended, so the resulting profile is itself a valid drone.
    """
    if not 0.0 <= t <= 1.0:
        raise ValueError("blend factor t must be in [0, 1]")
    lerp = lambda x, y: (1.0 - t) * x + t * y  # noqa: E731
    return DroneProfile(
        common=_blend_mode(a.common, b.common, t),
        roll=_blend_mode(a.roll, b.roll, t),
        pitch=_blend_mode(a.pitch, b.pitch, t),
        yaw=_blend_mode(a.yaw, b.yaw, t),
        motor_tau=lerp(a.motor_tau, b.motor_tau),
        cruise_tau=lerp(a.cruise_tau, b.cruise_tau),
        rps_min=lerp(a.rps_min, b.rps_min),
        rps_max=lerp(a.rps_max, b.rps_max),
    )


def _first_order_lowpass(x: np.ndarray, tau: float, dt: float) -> np.ndarray:
    """Causal first-order (exponential) low-pass filter along the last axis.

    ``tau`` is the response time constant; ``tau <= 0`` is a no-op (instant
    response).  Implements ``y[i] = y[i-1] + alpha (x[i] - y[i-1])`` with the
    exact ``alpha = 1 - exp(-dt/tau)``.
    """
    if tau <= 0.0:
        return x.copy()
    alpha = 1.0 - np.exp(-dt / tau)
    y = np.empty_like(x)
    y[0] = x[0]
    for i in range(1, x.shape[0]):
        y[i] = y[i - 1] + alpha * (x[i] - y[i - 1])
    return y


def _telegraph_setpoint(
    params: ManeuverModeParams,
    n: int,
    dt: float,
    rng: np.random.Generator,
    rate_scale: float,
    amp_scale: float,
) -> np.ndarray:
    """Piecewise-constant setpoint: holds at trim, with random maneuver pulses.

    Maneuver onsets are Poisson(``rate_hz * rate_scale``); each adds a rectangular
    deflection of amplitude ``N(0, maneuver_std * amp_scale)`` lasting
    ``Exp(mean_maneuver_s)``.  Overlapping pulses sum (compound maneuvers).
    """
    setpoint = np.full(n, params.trim, dtype=np.float64)
    duration_s = n * dt
    expected = params.rate_hz * rate_scale * duration_s
    n_events = int(rng.poisson(expected))
    for _ in range(n_events):
        onset = int(rng.integers(0, n))
        dur = max(1, int(rng.exponential(params.mean_maneuver_s) / dt))
        amp = float(rng.normal(0.0, params.maneuver_std * amp_scale))
        setpoint[onset : onset + dur] += amp
    return setpoint


def generate_intermittent(
    duration: float,
    fs: float,
    *,
    profile: DroneProfile | None = None,
    drone_profile: float | None = None,
    aggressiveness: float = 1.0,
    rng: np.random.Generator | int | None = None,
) -> np.ndarray:
    """Generate one realistic, *intermittent* ``(4, M)`` rotor-speed trajectory.

    Each control mode is a held trim value perturbed by occasional Poisson
    maneuver pulses, low-passed by the airframe's ``motor_tau`` lag, plus a small
    cruise jitter — producing the "steady, then a brief maneuver" texture of real
    flights rather than continuous OU wander.

    Args:
        duration: trajectory length in seconds.
        fs: sample rate (Hz); ``M = round(duration * fs)``.
        profile: explicit :class:`DroneProfile`.  Mutually exclusive with
            ``drone_profile``.
        drone_profile: convenience blend in ``[0, 1]`` between
            :data:`DREGON_PROFILE` (0) and :data:`MICHAELS_PROFILE` (1).  Used
            when ``profile`` is ``None`` (default blend ``0.0`` = DREGON).
        aggressiveness: scales both maneuver rate and amplitude; ``1.0`` is a
            typical flight, ``<1`` calmer, ``>1`` busier/larger maneuvers.
        rng: ``np.random.Generator``, int seed, or ``None``.

    Returns:
        ``(4, M)`` rotor speeds (rev/s), clamped to the profile's range, rotor
        order matching :data:`MIXER`.
    """
    if duration <= 0.0 or fs <= 0.0:
        raise ValueError("duration and fs must be positive")
    if aggressiveness < 0.0:
        raise ValueError("aggressiveness must be non-negative")
    if profile is not None and drone_profile is not None:
        raise ValueError("pass either profile or drone_profile, not both")
    if profile is None:
        profile = DREGON_PROFILE if drone_profile is None else blend_profiles(t=drone_profile)

    generator = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
    n = int(round(duration * fs))
    dt = 1.0 / fs

    modes = []
    for params in profile.modes:
        setpoint = _telegraph_setpoint(
            params, n, dt, generator, rate_scale=aggressiveness, amp_scale=aggressiveness
        )
        lagged = _first_order_lowpass(setpoint, profile.motor_tau, dt)
        jitter = _ou_path(
            OUModeParams(mean=0.0, std=params.cruise_std, tau=profile.cruise_tau),
            n,
            dt,
            generator,
            std_scale=1.0,
        )
        modes.append(lagged + jitter)
    w = rps_from_modes(np.stack(modes))
    return np.clip(w, profile.rps_min, profile.rps_max)


def generate_intermittent_batch(
    n_trajectories: int,
    duration: float,
    fs: float,
    **kwargs,
) -> np.ndarray:
    """Generate ``n_trajectories`` intermittent trajectories as a ``(N, 4, M)`` array.

    A single ``rng`` (via ``**kwargs``) is threaded through all trajectories for
    reproducibility; remaining keyword args forward to :func:`generate_intermittent`.
    """
    if n_trajectories <= 0:
        raise ValueError("n_trajectories must be positive")
    rng = kwargs.pop("rng", None)
    generator = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
    return np.stack(
        [
            generate_intermittent(duration, fs, rng=generator, **kwargs)
            for _ in range(n_trajectories)
        ]
    )


# =============================================================================
# Full-flight model (ground -> warm-up -> takeoff -> cruise -> landing -> ground)
# =============================================================================


@dataclass(frozen=True)
class FlightPhaseRanges:
    """Sampling ranges (seconds) for the phases of one full flight.

    The common-mode rotor speed traces ``0 (ground) -> idle (warm-up) -> hover
    (cruise) -> idle -> 0 (ground)``; ``idle_frac`` is the warm-up/idle rotor
    speed as a fraction of the profile's hover level (real DJI warm-up sits at
    ~0.45x hover, e.g. ~36 of ~80 rev/s). Differential maneuvering is confined to
    the cruise phase. Defaults are loosely calibrated to the DREGON/Michael's
    recordings (warm-up 5-30 s, takeoff/landing ramps 2-4 s).
    """

    pre_ground_s: tuple[float, float] = (0.5, 3.0)
    spinup_s: tuple[float, float] = (0.6, 1.8)
    warmup_s: tuple[float, float] = (3.0, 25.0)
    takeoff_s: tuple[float, float] = (1.5, 4.0)
    cruise_s: tuple[float, float] = (20.0, 120.0)  # used only when duration is None
    landing_s: tuple[float, float] = (1.5, 4.5)
    spindown_s: tuple[float, float] = (0.6, 2.0)
    post_ground_s: tuple[float, float] = (0.5, 3.0)
    idle_frac: tuple[float, float] = (0.38, 0.52)


def generate_full_flight(
    duration: float | None,
    fs: float,
    *,
    profile: DroneProfile | None = None,
    drone_profile: float | None = None,
    aggressiveness: float = 1.0,
    phases: FlightPhaseRanges | None = None,
    mode_scales: dict[str, float] | None = None,
    rng: np.random.Generator | int | None = None,
) -> np.ndarray:
    """Generate one *full-flight* ``(4, M)`` rotor-speed trajectory spanning the
    whole envelope: ground (rotors off, **zero RPS**) -> warm-up idle -> takeoff
    ramp -> cruise (the intermittent hover model) -> landing ramp -> ground.

    Unlike :func:`generate_intermittent` (which holds hover the whole time), this
    reaches the low-/zero-RPS regimes a real recording contains, so a model
    trained on it sees warm-up, takeoff, landing and silence — not just cruise.

    Args:
        duration: total length in seconds. If ``None``, a realistic total is
            sampled from ``phases`` (cruise drawn from ``cruise_s``); if given,
            the cruise phase absorbs whatever remains after the other phases
            (raising if the fixed phases already exceed ``duration``).
        fs: sample rate (Hz); ``M = round(total * fs)``.
        profile / drone_profile: airframe, as in :func:`generate_intermittent`.
        aggressiveness: scales cruise maneuvering only.
        phases: :class:`FlightPhaseRanges` overriding the phase-duration ranges.
        rng: ``np.random.Generator``, int seed, or ``None``.

    Returns:
        ``(4, M)`` rotor speeds (rev/s), clamped to ``[0, profile.rps_max]``;
        endpoints are ~0 (rotors off).
    """
    if fs <= 0.0:
        raise ValueError("fs must be positive")
    if aggressiveness < 0.0:
        raise ValueError("aggressiveness must be non-negative")
    if profile is not None and drone_profile is not None:
        raise ValueError("pass either profile or drone_profile, not both")
    if profile is None:
        profile = DREGON_PROFILE if drone_profile is None else blend_profiles(t=drone_profile)
    phases = phases or FlightPhaseRanges()
    generator = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)

    def _u(rng_range: tuple[float, float]) -> float:
        return float(generator.uniform(*rng_range))

    # Phase durations (seconds). Cruise either fills the requested duration or is
    # sampled when duration is None.
    pre_g = _u(phases.pre_ground_s)
    spinup = _u(phases.spinup_s)
    warmup = _u(phases.warmup_s)
    takeoff = _u(phases.takeoff_s)
    landing = _u(phases.landing_s)
    spindown = _u(phases.spindown_s)
    post_g = _u(phases.post_ground_s)
    fixed = pre_g + spinup + warmup + takeoff + landing + spindown + post_g
    if duration is None:
        cruise = _u(phases.cruise_s)
        total = fixed + cruise
    else:
        total = float(duration)
        cruise = total - fixed
        if cruise <= 0.0:
            raise ValueError(
                f"duration {total:.1f}s too short for the fixed phases ({fixed:.1f}s); "
                "shorten phase ranges or pass a longer duration"
            )

    n = int(round(total * fs))
    dt = 1.0 / fs
    t = np.arange(n, dtype=np.float64) * dt

    hover = profile.common.trim
    idle = _u(phases.idle_frac) * hover

    # Common-mode envelope: piecewise-linear through the phase breakpoints.
    bt = np.cumsum([0.0, pre_g, spinup, warmup, takeoff, cruise, landing, spindown, post_g])
    bl = np.array([0.0, 0.0, idle, idle, hover, hover, idle, 0.0, 0.0])
    envelope = np.interp(t, bt, bl)

    # Cruise gate (maneuvers/jitter only while hovering); soft edges come from the
    # motor low-pass below. Rotors-spinning gate keeps the ground exactly silent.
    cruise_start, cruise_end = bt[4], bt[5]
    cruise_gate = ((t >= cruise_start) & (t < cruise_end)).astype(np.float64)
    spin_gate = ((t >= bt[1]) & (t < bt[7])).astype(np.float64)  # spinup_start..spindown_end

    # Common mode = envelope + gated zero-mean maneuver pulses + gated cruise jitter.
    common_pulses = (
        _telegraph_setpoint(
            profile.common, n, dt, generator, rate_scale=aggressiveness, amp_scale=aggressiveness
        )
        - profile.common.trim
    )
    common_jitter = _ou_path(
        OUModeParams(mean=0.0, std=profile.common.cruise_std, tau=profile.cruise_tau),
        n,
        dt,
        generator,
        std_scale=1.0,
    )
    modes = [envelope + cruise_gate * common_pulses + spin_gate * common_jitter]

    # Differential modes (roll/pitch/yaw): trim + maneuvers + jitter, confined to
    # cruise (near-zero attitude control on the ground / during warm-up).
    #
    # `mode_scales` scales each differential mode independently, which is what
    # sets HOW THE FOUR ROTORS SEPARATE rather than how far they wander. Yaw
    # drives the two diagonal pairs apart while leaving each pair together; roll
    # and pitch separate the rotors WITHIN a pair. Real cruise holds a yaw trim
    # with near-zero roll and pitch, so it shows a wide spread AND a
    # near-degenerate pair at the same time: on the frozen split's cruise frames
    # two rotors sit within 1 rev/s in 71.6% of DREGON frames and 42.9% of
    # Michael's, against 17 to 25% in every synthetic stream measured. Scaling
    # roll and pitch down and yaw up reproduces both statistics together.
    scale_map = {name: 1.0 for name in MODE_NAMES}
    if mode_scales:
        unknown = set(mode_scales) - set(MODE_NAMES)
        if unknown:
            raise ValueError(f"unknown mode_scales keys: {sorted(unknown)}")
        scale_map.update(mode_scales)
    for name, params in zip(MODE_NAMES[1:], (profile.roll, profile.pitch, profile.yaw), strict=True):
        sc = float(scale_map[name])
        # One consistent meaning for `sc`: this mode is `sc` times as strong.
        # Applied to the finished setpoint (so the mode's TRIM scales too — the
        # yaw trim is what holds the diagonal pairs apart) and to the jitter
        # std. NOT also to amp_scale, which would square it.
        setpoint = _telegraph_setpoint(
            params, n, dt, generator, rate_scale=aggressiveness, amp_scale=aggressiveness
        )
        jitter = _ou_path(
            OUModeParams(mean=0.0, std=params.cruise_std * sc, tau=profile.cruise_tau),
            n,
            dt,
            generator,
            std_scale=1.0,
        )
        modes.append(cruise_gate * (setpoint * sc + jitter))

    # Motor/airframe lag rounds the ramp corners and gate edges (realistic
    # spin-up/down and fade-in of control authority).
    lagged = np.stack([_first_order_lowpass(m, profile.motor_tau, dt) for m in modes])
    w = rps_from_modes(lagged)
    return np.clip(w, 0.0, profile.rps_max)


# =============================================================================
# Synthetic comb window (an OU trajectory rendered as audio)
# =============================================================================


@dataclass(frozen=True)
class SynthCombWindow:
    """One synthetic rotor-noise window: audio plus the trajectory that made it.

    Attributes:
        audio: ``(n_mic, N)`` float64 waveform.  Every channel carries the
            **same** signal (no propagation model — this is a single-point
            observation duplicated, which is what the tracking evaluations
            expect of a synthetic window).
        r_true: ``(4, N)`` rotor speeds (rev/s) on the *audio* sample grid —
            the ground truth of the window, band-limited exactly as the comb
            phase that was synthesized from it.
        t: ``(N,)`` seconds, the audio time grid.
        mode_means: the four control-mode means ``(common, roll, pitch, yaw)``
            actually used (drawn or supplied).
        rotor_means: ``(4,)`` per-rotor mean rev/s implied by ``mode_means``,
            in rotor order (``MIXER`` rows), **unsorted**.
        meta: provenance of the draw — seed, knobs, OU parameters, comb RMS.
    """

    audio: np.ndarray
    r_true: np.ndarray
    t: np.ndarray
    mode_means: tuple[float, float, float, float]
    rotor_means: np.ndarray
    meta: dict[str, Any]


#: Mode stds/taus of the synthetic comb window — the DREGON free-flight
#: calibration, with the common-mode std softened to 1.5 rev/s so a 16 s window
#: stays inside the cruise band instead of wandering out of it.
SYNTH_COMB_STDS: tuple[float, float, float, float] = (1.5, 0.70, 0.85, 1.40)
SYNTH_COMB_TAUS: tuple[float, float, float, float] = (0.70, 0.60, 0.75, 1.00)

#: Sample rate of the OU trajectory before it is interpolated onto the audio grid.
SYNTH_COMB_FS_TRAJ: float = 250.0


def _draw_mode_means(
    rng: np.random.Generator, seed: int
) -> tuple[tuple[float, float, float, float], np.ndarray]:
    """Rejection-sample the four control-mode means for a synthetic window.

    Draws ``common ~ U[76, 94]``, ``roll ~ U[-3, 3]``, ``pitch ~ U[-6, 6]``,
    ``yaw ~ U[-4, 4]`` until the implied rotor means all lie in
    ``[70, 100]`` rev/s with at least 2 rev/s pairwise separation — the regime
    where the four combs are individually resolvable but still overlap.
    """
    for _ in range(200):
        m_common = rng.uniform(76.0, 94.0)
        m_roll = rng.uniform(-3.0, 3.0)
        m_pitch = rng.uniform(-6.0, 6.0)
        m_yaw = rng.uniform(-4.0, 4.0)
        rotor_means = MIXER @ np.array([m_common, m_roll, m_pitch, m_yaw])
        seps = np.abs(rotor_means[:, None] - rotor_means[None, :])[np.triu_indices(4, 1)]
        if rotor_means.min() >= 70.0 and rotor_means.max() <= 100.0 and seps.min() >= 2.0:
            return (m_common, m_roll, m_pitch, m_yaw), rotor_means
    raise RuntimeError(f"synth seed {seed}: no valid rotor-mean draw in 200 tries")


def synth_comb_window(
    seed: int,
    *,
    aggressiveness: float = 1.0,
    mode_means: tuple[float, float, float, float] | None = None,
    fc_hz: float | None = None,
    snr_db: float = 0.0,
    dur: float = 16.0,
    sr: int = 16000,
    k_max: int = 30,
    n_mic: int = 2,
) -> SynthCombWindow:
    """Render one synthetic rotor-noise window with an exactly known trajectory.

    The signal model, in order:

      1. **Trajectory.** Four OU control modes (:func:`generate`) at
         :data:`SYNTH_COMB_FS_TRAJ`, with :data:`SYNTH_COMB_STDS` /
         :data:`SYNTH_COMB_TAUS` and the given (or drawn) means.
      2. **Shaft inertia** (optional).  ``fc_hz`` zero-phase low-passes the
         commanded shaft speed *before* audio synthesis, and the ground truth
         is defined from that same band-limited trajectory.  Without it the OU
         drive is white to the trajectory rate, and point-sampling the truth
         onto a 31.25 Hz frame grid aliases all of it into the comparison band.
      3. **Comb.** Each rotor contributes ``k = 1..k_max`` harmonics of its
         instantaneous shaft phase, at locked (uniform-random) initial phases.
         Amplitudes are ``1/k`` with a **2-blade blade-pass** emphasis — even
         harmonics ``1.6/k``, odd ``0.5/k`` — so blade-pass order 2 dominates,
         the regime the octave checks in ``tracking`` are calibrated for.
      4. **Noise.** White Gaussian at ``comb_rms * 10 ** (-snr_db / 20)``, so
         ``snr_db`` is the comb-to-noise ratio in dB.

    The random draws happen in a fixed order — mode means (only when drawn),
    OU trajectory, harmonic phases, noise — so ``fc_hz`` and ``snr_db`` change
    the signal without disturbing the stream, and a given ``seed`` reproduces
    the window bit-exactly.

    Args:
        seed: seed of the window's ``np.random.default_rng``.
        aggressiveness: OU dynamic-std multiplier, as in :func:`generate`.
        mode_means: fixed ``(common, roll, pitch, yaw)`` means; ``None`` draws
            them with :func:`_draw_mode_means`.
        fc_hz: shaft-inertia low-pass cutoff (Hz), or ``None`` for no filter.
        snr_db: comb-to-noise ratio in dB.
        dur: window length in seconds.
        sr: audio sample rate (Hz).
        k_max: highest harmonic order per rotor.
        n_mic: number of (identical) audio channels to emit.

    Returns:
        A :class:`SynthCombWindow`.
    """
    rng = np.random.default_rng(seed)
    if mode_means is not None:
        means = (
            float(mode_means[0]),
            float(mode_means[1]),
            float(mode_means[2]),
            float(mode_means[3]),
        )
        rotor_means = MIXER @ np.array(list(means))
    else:
        means, rotor_means = _draw_mode_means(rng, seed)
    m_common, m_roll, m_pitch, m_yaw = means

    n_t = int(dur * sr)
    t = np.arange(n_t) / sr
    s_common, s_roll, s_pitch, s_yaw = SYNTH_COMB_STDS
    tau_common, tau_roll, tau_pitch, tau_yaw = SYNTH_COMB_TAUS
    cfg = RPSSynthConfig(
        common=OUModeParams(mean=m_common, std=s_common, tau=tau_common),
        roll=OUModeParams(mean=m_roll, std=s_roll, tau=tau_roll),
        pitch=OUModeParams(mean=m_pitch, std=s_pitch, tau=tau_pitch),
        yaw=OUModeParams(mean=m_yaw, std=s_yaw, tau=tau_yaw),
    )
    fs_traj = SYNTH_COMB_FS_TRAJ
    r_lo = generate(dur, fs_traj, config=cfg, aggressiveness=aggressiveness, rng=rng)
    if fc_hz is not None:  # rotor inertia: zero-phase lowpass on the shaft speed
        from scipy.signal import filtfilt, firwin

        taps = firwin(255, fc_hz / (fs_traj / 2), window="hamming")
        r_lo = filtfilt(taps, [1.0], r_lo, axis=1)
    t_lo = np.arange(r_lo.shape[1]) / fs_traj
    r_true = np.stack([np.interp(t, t_lo, r_lo[i]) for i in range(NUM_ROTORS)])

    psi = rng.uniform(0, 2 * np.pi, (NUM_ROTORS, k_max))  # locked initial phases
    comb = np.zeros(n_t)
    for i in range(NUM_ROTORS):
        phi = 2 * np.pi * np.cumsum(r_true[i]) / sr
        for k in range(1, k_max + 1):
            amp = (1.6 if k % 2 == 0 else 0.5) / k
            comb += amp * np.cos(k * phi + psi[i, k - 1])
    comb_rms = float(np.sqrt(np.mean(comb**2)))
    noise = rng.normal(0.0, comb_rms * 10 ** (-snr_db / 20.0), n_t)
    x = (comb + noise).astype(np.float64)
    audio = np.stack([x] * n_mic)

    meta = {
        "seed": seed,
        "aggressiveness": aggressiveness,
        "shaft_fc_hz": fc_hz,
        "snr_db": snr_db,
        "duration_s": dur,
        "sample_rate": sr,
        "k_max": k_max,
        "n_mic": n_mic,
        "comb_rms": comb_rms,
        "fs_traj": fs_traj,
        "ou_modes": {
            name: {"mean": float(mean), "std": float(std), "tau": float(tau)}
            for name, mean, std, tau in zip(
                MODE_NAMES, means, SYNTH_COMB_STDS, SYNTH_COMB_TAUS, strict=True
            )
        },
    }
    return SynthCombWindow(
        audio=audio,
        r_true=r_true,
        t=t,
        mode_means=means,
        rotor_means=rotor_means,
        meta=meta,
    )
