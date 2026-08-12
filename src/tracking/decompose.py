"""Windowed decomposition of a recording into per-harmonic tracks + a residual.

The coupled Vold-Kalman solve of :mod:`tracking.fitness_vk`, applied window by
window to a WHOLE recording and stitched into one envelope bank::

    x(t) = sum_{rotor, k} Re[ a_{rotor,k}(t) e^{j k phi_rotor(t)} ] + residual(t)

The sum is EXACT by construction, because the residual is DEFINED as what the
track sum does not explain. What is estimated is the split, and that split is a
maximum-likelihood one: the VK cost is a penalized least squares, which is the
MAP estimate of the envelopes under a Gaussian residual and a per-track
second-difference (bandwidth) prior. A track therefore carries the energy the
model can explain at its own carrier and inside its own band, and the residual
carries the rest.

This module holds the array core only. The data loading, the unit harness and
the file formats are the driver's (``scripts/vk_decompose.py``).

Three sections, and the first one is not about the decomposition at all:

1. **The windowed-application primitives** — :func:`frame_grid`,
   :func:`interp_rps`, :func:`window_bounds`, :func:`window_span`,
   :func:`fade_weights`, :func:`to_audio_grid`, :func:`shaft_phase`. Tiling a
   recording into overlapping windows and cross-fading the per-window results
   back together is the same operation whatever the windows produce, so the
   windowed telemetry refiner (``scripts/refine_dregon_rps.py``) reads through
   the same functions.
2. **The solve** — :func:`solve_config`, :func:`solve_window`,
   :func:`group_plan` (THE memory model), :func:`reconstruct`,
   :func:`stitch_bank` (the phase re-reference + the cross-fade),
   :func:`phase_reference_deviation`.
3. **The readings** — the energy ledger and the phase-model tables
   (:func:`energy_ledger`, :func:`phase_model_report` and the per-track
   statistics they are built from).

Two conventions a caller must know:

**Every window is re-referenced to ONE global shaft phase before the stitch.**
The solver's own ``phase`` starts at the window (``phase = 2 pi cumsum(r) / fs``),
so a window that starts at audio sample ``a0`` carries the constant offset
``Phi(a0 - 1)``; :func:`stitch_bank` multiplies each track by
``exp(-j k Phi(a0 - 1))``. Without that, two overlapping windows hold the same
physical track at two different phase origins and the cross-fade cancels them.

**Windows start on a multiple of the envelope stride** (:func:`window_span`), so
every window's envelope grid is a slice of one global envelope grid and no
resampling is needed.

Purity: numpy and scipy only, plus the sibling tracking modules.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from tracking.fitness_vk import FVKConfig, solve_envelopes, to_audio_grid
from tracking.vk_tracking import (
    Envelopes,
    _coupling_groups,
    _track_table,
    _tuma_rho,
    env_stride,
    vk_envelopes,
)

__all__ = [
    "DEFAULT_BANDS",
    "BandwidthSchedule",
    "band_name",
    "band_summary",
    "base_bandwidths",
    "drift_increments",
    "energy_ledger",
    "fade_weights",
    "frame_grid",
    "group_plan",
    "interp_rps",
    "line_separations",
    "per_track_stats",
    "phase_model_report",
    "phase_reference_deviation",
    "rank_one_share",
    "reconstruct",
    "reference_mic",
    "residual_tones",
    "schedule_bandwidths",
    "schedule_rho2_gain",
    "shaft_phase",
    "solve_config",
    "solve_window",
    "stitch_bank",
    "to_audio_grid",
    "track_bands",
    "welch_psd",
    "window_bounds",
    "window_span",
]

#: Harmonic bands every band-wise reading of this module reports against.
DEFAULT_BANDS: tuple[tuple[int, int], ...] = ((1, 9), (10, 24), (25, 49), (50, 80))

#: Welch segment of :func:`welch_psd`.
NPERSEG = 4096


def band_name(lo: int, hi: int) -> str:
    return f"k{lo}-{hi}"


# ---------------------------------------------------------------------------
# 1. the windowed-application primitives


def frame_grid(n_t: int, sr: float, hop_s: float) -> np.ndarray:
    """The uniform ``hop_s`` frame grid of a recording, in relative seconds.

    Same construction as :func:`tracking.protocols.slice_window`, so a window
    cut here and a protocol window agree frame for frame.
    """
    return np.arange(0.0, n_t / float(sr) - hop_s / 2, hop_s)


def interp_rps(vals: Any, stamps: Any, ft: Any) -> np.ndarray:
    """``(R, M)`` telemetry at ``stamps`` -> ``(R, N)`` on ``ft``, in float64.

    ``data_processing.noise_rps_dataset.upsample_rps_to_audio_rate`` in float64:
    the same duplicate-stamp drop and the same clip against extrapolation, but a
    carrier (or a refiner init) must not carry a float32 rounding staircase.
    """
    ts = np.asarray(stamps, dtype=np.float64)
    _, uniq = np.unique(ts, return_index=True)
    uniq = np.sort(uniq)
    ts = ts[uniq]
    v = np.asarray(vals, dtype=np.float64)[:, uniq]
    q = np.clip(np.asarray(ft, dtype=np.float64), ts[0], ts[-1])
    return np.stack([np.interp(q, ts, row) for row in v])


def shaft_phase(r_audio: Any, sr: float) -> np.ndarray:
    """``(R, T)`` fundamental shaft phase in radians.

    :func:`tracking.vk_envelopes` computes exactly this
    (``2 pi cumsum(r) / fs``) on the window it is given, so a window that starts
    at sample ``a0`` differs from this global phase by the constant
    ``Phi(a0 - 1)``. That constant is what :func:`stitch_bank` removes.
    """
    return (
        2.0
        * np.pi
        * np.cumsum(np.atleast_2d(np.asarray(r_audio, dtype=np.float64)), axis=-1)
        / float(sr)
    )


def window_bounds(
    n_frames: int, window_s: float, hop_s: float, hop_frame_s: float
) -> list[tuple[int, int]]:
    """Window frame ranges over a whole recording, the last one right-aligned.

    ``hop_frame_s`` is the recording's own frame hop (what one index of
    ``n_frames`` is worth); ``window_s`` / ``hop_s`` are the window length and
    the window step. Every frame is covered.
    """
    w = max(1, int(round(window_s / hop_frame_s)))
    step = max(1, int(round(hop_s / hop_frame_s)))
    if n_frames <= w:
        return [(0, n_frames)]
    starts = list(range(0, n_frames - w + 1, step))
    if starts[-1] + w < n_frames:
        starts.append(n_frames - w)
    return [(s, s + w) for s in starts]


def window_span(
    ft: Any, i0: int, i1: int, n_t: int, stride: int, sr: float, hop_frame_s: float
) -> tuple[int, int]:
    """Audio sample range of one window, snapped to the envelope stride.

    Both ends land on a multiple of ``stride``, so the window's envelope grid is
    a slice of the recording's global envelope grid. Without the snap the two
    grids are offset by a fraction of a knot (a 0.032 s frame is 3.2 knots at
    16 kHz and 100 Hz) and the stitch would have to resample.
    """
    ftv = np.asarray(ft, dtype=np.float64)
    a0 = (int(round(float(ftv[i0]) * sr)) // stride) * stride
    a1_raw = min(n_t, int(round((float(ftv[i1 - 1]) + hop_frame_s) * sr)))
    a1 = a0 + ((a1_raw - a0) // stride) * stride
    return a0, a1


def fade_weights(n_win: int, ramp: int) -> np.ndarray:
    """Linear cross-fade weights over one window's frames.

    The floor keeps the weight positive everywhere, so a frame that only one
    window covers (the two ends of a recording) still resolves to that window.
    """
    idx = np.arange(n_win, dtype=np.float64)
    if ramp <= 0:
        return np.ones(n_win)
    rise = np.minimum(idx, idx[::-1]) + 1.0
    return np.clip(rise / (ramp + 1.0), 1e-3, 1.0)


# ---------------------------------------------------------------------------
# 2. the solve


def solve_config(
    k_max: int,
    *,
    sr: float,
    mics: int,
    bw_rps: float = 1.0,
    f_max: float = 6000.0,
) -> FVKConfig:
    """THE measurement geometry — one construction, so every solve agrees.

    ``f_max`` is the campaign's 6 kHz ceiling, held below three quarters of the
    Nyquist frequency so a lower sample rate (the tests) keeps a modelled
    harmonic inside the band instead of on top of it.
    """
    return FVKConfig(
        sr=int(sr),
        k_max=int(k_max),
        f_max=min(float(f_max), 0.375 * float(sr)),
        max_channels=int(mics),
        bw_rps=float(bw_rps),
    )


@dataclass(frozen=True)
class BandwidthSchedule:
    """Per-track envelope bandwidth: ``k``-scaled, and capped by the separation.

    THE v2 knob. The v1 decomposition solved every track with ONE bandwidth,
    because the solver's per-group clamp floors a dense comb at
    ``VKConfig.bw_hz`` (1 Hz) whatever ``bw_rps`` asks for. A real rotor line is
    not 1 Hz wide at every harmonic: the shaft jitter displaces harmonic ``k``
    by ``k`` times the rate error, so the linewidth grows with ``k`` and a flat
    1 Hz band LEAKS the comb into the residual above about ``k`` 10 (measured on
    ``free-flight_nosource_room1``: 0.92 dB of order contrast left in the
    residual at k10-24 against 1.29 dB in the original).

    The schedule is the linewidth law with the two guards the leak measurement
    demanded:

        bw_m = clip(bw0_hz + slope_hz_per_k * k_m, base, upper_m)
        upper_m = max(base, min(cap_frac_of_sep * sep_m, bw_abs_max))

    ``sep_m`` is the track's own line separation (:func:`line_separations`).
    Both guards are against OVER-subtraction, which the same measurement showed
    is the failure mode on the other side: a band wider than the distance to the
    neighbouring line swallows the broadband floor around the line and notches
    the residual (−0.91 dB of contrast at k25-40 at ``cap_frac_of_sep`` 1.5).
    ``base`` is the flat bandwidth the solver would have used, so a schedule
    never NARROWS a track below v1.
    """

    #: Bandwidth (Hz) of the fundamental — the intercept of the linewidth law.
    bw0_hz: float = 1.0
    #: Hz of bandwidth added per harmonic index — the slope of the same law.
    slope_hz_per_k: float = 0.0
    #: Fraction of a track's line separation the band may not exceed.
    cap_frac_of_sep: float = 1.0
    #: Absolute ceiling (Hz), against a track that is far from every other line.
    bw_abs_max: float = 10.0

    def __post_init__(self) -> None:
        if self.bw0_hz <= 0 or self.bw_abs_max <= 0:
            raise ValueError(f"bw0_hz and bw_abs_max must be positive, got {self}")
        if self.slope_hz_per_k < 0 or self.cap_frac_of_sep <= 0:
            raise ValueError(f"slope must be >= 0 and cap_frac_of_sep > 0, got {self}")

    @classmethod
    def parse(cls, text: str) -> BandwidthSchedule | None:
        """``"bw0,slope,capfrac,absmax"`` -> a schedule; empty text -> ``None``.

        The CLI spelling, so a driver carries the schedule as one JSON-safe
        string through its unit parameters and its provenance.
        """
        parts = [p for p in str(text).replace(" ", "").split(",") if p]
        if not parts:
            return None
        if len(parts) != 4:
            raise ValueError(
                f"bandwidth schedule needs 4 comma-separated numbers "
                f"(bw0,slope,capfrac,absmax), got {text!r}"
            )
        return cls(*(float(v) for v in parts))

    def as_dict(self) -> dict[str, float]:
        return {
            "bw0_hz": float(self.bw0_hz),
            "slope_hz_per_k": float(self.slope_hz_per_k),
            "cap_frac_of_sep": float(self.cap_frac_of_sep),
            "bw_abs_max": float(self.bw_abs_max),
        }

    def text(self) -> str:
        return (
            f"{self.bw0_hz:g},{self.slope_hz_per_k:g},{self.cap_frac_of_sep:g},{self.bw_abs_max:g}"
        )


def line_separations(r_audio: Any, rotor: Any, k: Any) -> np.ndarray:
    """``(M,)`` distance in Hz from each track's line to the NEAREST other line.

    Read at the window's MEAN rates, which is the same reading the schedule's
    cap is meant to express: how much room a track has before its passband
    starts to explain a neighbour's line. A single track has no neighbour and
    gets ``inf``.
    """
    r = np.atleast_2d(np.asarray(r_audio, dtype=np.float64))
    rot = np.asarray(rotor, dtype=int)
    ks = np.asarray(k, dtype=np.float64)
    f = ks * r.mean(axis=-1)[rot]
    if f.size < 2:
        return np.full(f.shape, np.inf)
    d = np.abs(f[:, None] - f[None, :])
    np.fill_diagonal(d, np.inf)
    return d.min(axis=-1)


def schedule_bandwidths(
    k: Any, sep_hz: Any, sched: BandwidthSchedule, base_bw_hz: Any
) -> np.ndarray:
    """``(M,)`` target −3 dB bandwidth per track — the law in the class docstring.

    ``base_bw_hz`` is a scalar or one value per track (:func:`base_bandwidths`);
    it is the FLOOR, so a schedule can only widen a band, never narrow it.
    """
    ks = np.asarray(k, dtype=np.float64)
    sep = np.asarray(sep_hz, dtype=np.float64)
    base = np.broadcast_to(np.asarray(base_bw_hz, dtype=np.float64), ks.shape)
    upper = np.maximum(base, np.minimum(sched.cap_frac_of_sep * sep, sched.bw_abs_max))
    want = sched.bw0_hz + sched.slope_hz_per_k * ks
    return np.clip(want, base, upper)


def schedule_rho2_gain(
    k: Any,
    sep_hz: Any,
    sched: BandwidthSchedule,
    base_bw_hz: Any,
    fs_env: float,
    p: int = 2,
) -> np.ndarray:
    """``(M,)`` gain on ``rho^2`` that turns the base band into the scheduled one.

    The solver takes a per-track multiplicative gain on the squared selectivity,
    not a bandwidth, so the schedule is expressed through the Tuma relation:
    ``gain_m = (rho(bw_m) / rho(base_m)) ^ 2``. The gain is applied AFTER the
    solver's own per-group clamp and ``base_m`` is that clamp's own answer
    (:func:`base_bandwidths`), so the ACHIEVED band is the scheduled one on any
    comb, dense or sparse. It is still not guessed:
    :attr:`tracking.Envelopes.bw_track` records what the solve really used.
    """
    bw = schedule_bandwidths(k, sep_hz, sched, base_bw_hz)
    base = np.broadcast_to(np.asarray(base_bw_hz, dtype=np.float64), bw.shape)
    return np.array(
        [
            (_tuma_rho(float(b), float(fs_env), p) / _tuma_rho(float(b0), float(fs_env), p)) ** 2
            for b, b0 in zip(bw, base, strict=True)
        ]
    )


def solve_window(
    audio: Any,
    r_audio: Any,
    cfg: FVKConfig,
    *,
    k_hi: int,
    mics: int | None = None,
    rho_scale: float = 1.0,
    bw_schedule: BandwidthSchedule | None = None,
) -> Envelopes:
    """One coupled VK solve of one window — the whole numerical content.

    The validity mask is disabled and the harmonic set is capped at ``k_hi``
    (which a windowed driver takes from the RECORDING's reference trajectory,
    not from the window), so every window of a recording holds the identical
    ``(rotor, harmonic)`` track set and the windows can be stitched track by
    track. ``mics`` defaults to ``cfg.max_channels``.

    ``bw_schedule`` is the v2 linewidth-matched bandwidth
    (:class:`BandwidthSchedule`). ``None`` — the default — takes the v1 path
    call for call, so a v1 unit is reproduced bit for bit.
    """
    n_mic = int(cfg.max_channels if mics is None else mics)
    y = np.ascontiguousarray(np.asarray(audio, dtype=np.float64)[:n_mic])
    r = np.atleast_2d(np.asarray(r_audio, dtype=np.float64))
    if bw_schedule is None:
        return solve_envelopes(y, r, cfg, k_hi=int(k_hi), rho_scale=float(rho_scale))
    vk = cfg.vk_config(int(k_hi))
    rotor, k = _track_table(int(r.shape[0]), vk.k_min, int(k_hi))
    _, fs_env = env_stride(vk)
    gain = (
        schedule_rho2_gain(
            k,
            line_separations(r, rotor, k),
            bw_schedule,
            base_bandwidths(r, int(k_hi), cfg),
            fs_env,
            vk.p,
        )
        * float(rho_scale) ** 2
    )
    return vk_envelopes(y, r, vk, rho2_gain=gain)


def _geometry(r_audio: Any, k_hi: int, vk: Any) -> dict[str, Any]:
    """The solver's own line geometry of one window, without solving it.

    ``(rotor, k)`` track table, the ``(M, N_env)`` line frequencies, the
    validity mask and the coupling partition — :func:`tracking.vk_envelopes`
    computes exactly these before it builds a system, and both
    :func:`group_plan` (memory) and :func:`base_bandwidths` (the v1 band) are
    readings of them. ONE copy of the rule, so the two readings cannot drift
    apart from each other.
    """
    stride, fs_env = env_stride(vk)
    r = np.atleast_2d(np.asarray(r_audio, dtype=np.float64))
    rotor, k = _track_table(int(r.shape[0]), vk.k_min, int(k_hi))
    r_dec = r[:, ::stride]
    f = k[:, None].astype(np.float64) * r_dec[rotor]
    valid = (f >= vk.f_min) & (f <= min(vk.f_max, 0.45 * vk.fs)) & (r_dec[rotor] >= vk.min_rps)
    couple = fs_env / 2.0 if vk.couple_hz is None else float(vk.couple_hz)
    return {
        "rotor": rotor,
        "k": k,
        "f": f,
        "valid": valid,
        "groups": _coupling_groups(f, valid, couple),
        "stride": stride,
        "fs_env": fs_env,
    }


def base_bandwidths(r_audio: Any, k_hi: int, cfg: FVKConfig) -> np.ndarray:
    """``(M,)`` bandwidth the solver uses per track with NO schedule — the v1 band.

    The reference the schedule's gain is taken against, and the one piece of
    :func:`tracking.vk_envelopes` this module has to know: a group's band is
    clamped to ``max(VKConfig.bw_hz, sep_bw_factor * minimum pair separation)``,
    so a DENSE comb floors every track at ``bw_hz`` (1 Hz) however large the
    ``bw_rps`` request was, while a sparse one keeps the ``k``-scaled request.
    Both cases matter, which is why this is computed and not assumed.

    It is a second implementation of the solver's rule, so it is pinned against
    the solver itself (``tests/tracking/test_decompose.py``) rather than trusted.
    """
    from tracking.vk_tracking import _tuma_bw_min

    vk = cfg.vk_config(int(k_hi))
    geo = _geometry(r_audio, int(k_hi), vk)
    f, valid, k = geo["f"], geo["valid"], geo["k"]
    fs_env = float(geo["fs_env"])
    bw = np.full(len(k), float(vk.bw_hz))
    b_lo, b_hi = _tuma_bw_min(fs_env, vk.p), 0.9 * fs_env
    for group in geo["groups"]:
        sep_cap = np.inf
        for a in range(len(group)):
            for b in range(a + 1, len(group)):
                m, n = group[a], group[b]
                both = valid[m] & valid[n]
                if both.any():
                    sep = float(np.min(np.abs(f[m, both] - f[n, both])))
                    sep_cap = min(sep_cap, max(vk.bw_hz, vk.sep_bw_factor * sep))
        if vk.bw_rps is not None:
            b_m = np.clip(k[group].astype(np.float64) * vk.bw_rps, b_lo, b_hi)
            bw[group] = np.minimum(b_m, max(sep_cap, b_lo))
        else:
            bw[group] = min(float(vk.bw_hz), sep_cap)
    return bw


def group_plan(r_audio: Any, k_hi: int, cfg: FVKConfig) -> dict[str, Any]:
    """Coupling-group partition of one window, and the memory it will cost.

    THE memory model of a windowed decomposition, and it must be read before a
    job is sized. :func:`tracking.vk_tracking._coupling_groups` is a union-find
    over the pairs whose lines come within ``couple_hz`` (50 Hz by default), so
    coupling is TRANSITIVE: at ``k_hi`` 62 the four DREGON rotors put 248 lines
    into 0 to 5.7 kHz, a mean spacing of 23 Hz, and the chain merges 244 of them
    into ONE group.

    One group of ``g`` tracks is solved as a Hermitian banded system of ``g``
    times ``n_env`` unknowns with ``2 g`` superdiagonals, and the factorization
    holds a second copy, so

        bytes = 2 (2 g + 1) g n_env 16 ,   approximately 1e-4 k_hi^2 window_s GB

    with ``g`` about ``4 k_hi`` and ``n_env`` about ``100 window_s``. Channels
    are right-hand sides and cost nothing here. The full-recording DREGON
    configuration (``k_hi`` 62, 16 s windows) therefore needs 6.3 GB per WORKER.
    """
    geo = _geometry(r_audio, int(k_hi), cfg.vk_config(int(k_hi)))
    groups = geo["groups"]
    g = max((len(x) for x in groups), default=0)
    n_env = int(geo["f"].shape[-1])
    return {
        "n_tracks": int(len(geo["k"])),
        "n_groups": len(groups),
        "max_group": int(g),
        "n_env": n_env,
        "banded_gb": round(2.0 * (2 * g + 1) * g * n_env * 16 / 1e9, 3),
    }


def _upsample_knots(vals: Any, ramp: Any, n_out: int) -> np.ndarray:
    """``(C, J)`` knots -> ``(C, n_out)`` linear upsample on the uniform grid."""
    v = np.asarray(vals)
    if v.shape[-1] > 1:
        diffs = np.diff(v, axis=-1)
        up = (v[:, :-1, None] + diffs[:, :, None] * ramp).reshape(v.shape[0], -1)
    else:
        up = np.zeros((v.shape[0], 0), dtype=v.dtype)
    if up.shape[-1] < n_out:  # hold the last knot beyond the grid
        tail = np.repeat(v[:, -1:], n_out - up.shape[-1], axis=-1)
        up = np.concatenate([up, tail], axis=-1)
    return up[:, :n_out]


def reconstruct(
    x: Any,
    k: Any,
    rotor: Any,
    phase: Any,
    stride: int,
    *,
    knots_per_chunk: int = 1024,
) -> tuple[np.ndarray, np.ndarray]:
    """``(C, M, N_env)`` envelopes -> the ``(C, T)`` track sum and per-track energy.

    Track ``m`` is ``Re[a_m(t) e^{j k_m phi(t)}]`` with ``a_m`` linearly
    interpolated from the envelope grid onto the audio grid, real and imaginary
    parts separately, and held constant beyond the last knot — the rule
    :func:`tracking.vk_reconstruct` uses. It is the STITCHED sibling of that
    function: the phase is passed in (already the GLOBAL phase the envelopes
    were re-referenced to, see :func:`stitch_bank`) instead of read off one
    window's :class:`Envelopes`, and the per-track energies come back with it.

    The work is chunked in time and the tracks are accumulated one at a time:
    the whole ``(C, M, T)`` bank is hundreds of gigabytes at a realistic
    ``k_hi``, and only the sum and the per-track energies are wanted.
    """
    xa = np.asarray(x)
    ph = np.atleast_2d(np.asarray(phase, dtype=np.float64))
    n_ch, n_tracks, n_env = xa.shape
    n_t = int(ph.shape[-1])
    recon = np.zeros((n_ch, n_t), dtype=np.float32)
    energy = np.zeros(n_tracks, dtype=np.float64)
    if n_tracks == 0 or n_env == 0 or n_t == 0:
        return recon, energy

    ramp = np.arange(stride, dtype=np.float32) / np.float32(stride)
    ks = np.asarray(k).astype(np.float64)
    rot = np.asarray(rotor).astype(int)
    for j0 in range(0, n_env, knots_per_chunk):
        j1 = min(n_env, j0 + knots_per_chunk)
        s0, s1 = j0 * stride, min(n_t, j1 * stride)
        if s1 <= s0:
            break
        # One extra knot when there is one: the last knot of the chunk needs the
        # difference to its successor, not a hold.
        jend = min(j1 + 1, n_env)
        n_out = s1 - s0
        for m in range(n_tracks):
            vals = xa[:, m, j0:jend]
            up_r = _upsample_knots(np.real(vals).astype(np.float32), ramp, n_out)
            up_i = _upsample_knots(np.imag(vals).astype(np.float32), ramp, n_out)
            arg = ks[m] * ph[rot[m], s0:s1]
            comp = up_r * np.cos(arg).astype(np.float32) - up_i * np.sin(arg).astype(np.float32)
            recon[:, s0:s1] += comp
            energy[m] += float(np.square(comp, dtype=np.float64).sum())
    return recon, energy


def stitch_bank(
    windows: list[dict[str, Any]],
    phi: Any,
    stride: int,
    ramp: int,
) -> dict[str, Any]:
    """Cross-fade per-window envelope banks onto one global envelope grid.

    ``windows`` is a list of ``{"a0": audio start sample, "x": (C, M, N)
    complex bank, "valid": (M, N) bool, "rotor", "k"}`` — already loaded, so
    this function reads no file. They must all carry the same
    ``(rotor, harmonic)`` track set (see :func:`solve_window`); the caller
    checks that before it gets here.

    Every ``a0`` must be a multiple of ``stride``, which is what
    :func:`window_span` guarantees. That is what makes a window's envelope grid
    an exact slice of the global one: the solver returns ``n_env * stride``
    samples' worth of knots for a window whose length is a multiple of the
    stride, so ``a0 + N * stride`` is the window's end and no resample is
    needed.

    Every window is first re-referenced to the global shaft phase ``phi``: the
    solver's ``phase`` starts at the window, so track ``m`` of the window that
    starts at sample ``a0`` is multiplied by ``exp(-j k_m Phi_rotor(a0 - 1))``.
    Then real and imaginary parts are cross-faded with linear ramps over the
    overlap.

    Returns ``{"x", "valid", "covered", "a_min", "a_max", "env_i0", "n_env"}``.
    """
    if not windows:
        raise ValueError("stitch_bank: no window to stitch")
    used = sorted(windows, key=lambda w: int(w["a0"]))
    a_min = min(int(w["a0"]) for w in used)
    a_max = max(int(w["a0"]) + int(np.asarray(w["x"]).shape[-1]) * stride for w in used)
    e0, e1 = a_min // stride, a_max // stride
    n_env = e1 - e0

    rotor = np.asarray(used[0]["rotor"], dtype=np.int64)
    k = np.asarray(used[0]["k"], dtype=np.int64)
    n_ch, n_tracks = int(np.asarray(used[0]["x"]).shape[0]), int(len(k))

    num = np.zeros((n_ch, n_tracks, n_env), dtype=np.complex64)
    den = np.zeros(n_env, dtype=np.float64)
    valid = np.zeros((n_tracks, n_env), dtype=bool)
    for w in used:
        x = np.asarray(w["x"], dtype=np.complex64).copy()
        a0 = int(w["a0"])
        j0 = a0 // stride - e0
        n_w = int(x.shape[-1])
        # The window's own phase origin, removed. Phi(-1) is 0 by definition.
        shift = np.zeros(int(np.max(rotor)) + 1) if a0 == 0 else np.asarray(phi)[:, a0 - 1]
        x *= np.exp(-1j * k[None, :, None] * shift[rotor][None, :, None]).astype(np.complex64)
        fade = fade_weights(n_w, min(ramp, n_w // 2))
        num[:, :, j0 : j0 + n_w] += x * fade[None, None, :].astype(np.complex64)
        den[j0 : j0 + n_w] += fade
        valid[:, j0 : j0 + n_w] |= np.asarray(w["valid"], dtype=bool)

    covered = den > 0.0
    num /= np.maximum(den, 1e-12).astype(np.float32)[None, None, :]
    valid &= covered[None, :]
    return {
        "x": num,
        "valid": valid,
        "rotor": rotor,
        "k": k,
        "covered": covered,
        "a_min": a_min,
        "a_max": a_max,
        "env_i0": e0,
        "n_env": n_env,
    }


def phase_reference_deviation(
    r_audio: Any, phi: Any, a0: int, sr: float, seconds: float = 4.0
) -> float:
    """Maximum radians by which a window's phase and the global phase disagree.

    The re-reference of :func:`stitch_bank` assumes
    ``phase_window(t) = Phi(t) - Phi(a0 - 1)``. That is an identity only while
    the window's carrier is the same array the global phase was built from, so
    it is MEASURED on one window instead of assumed.
    """
    p = np.asarray(phi)
    a1 = min(int(p.shape[-1]), a0 + int(round(seconds * sr)))
    local = shaft_phase(np.asarray(r_audio)[:, a0:a1], sr)
    shift = 0.0 if a0 == 0 else p[:, a0 - 1 : a0]
    return float(np.abs(local - (p[:, a0:a1] - shift)).max())


# ---------------------------------------------------------------------------
# 3. the readings


def reference_mic(audio: Any, ref_mic: int) -> int:
    """Which microphone the per-track statistics are read on.

    ``ref_mic < 0`` selects the channel with the most energy on the span. The
    DREGON array is not uniform — on one cruise window channels 1 and 4 hold
    378 and 347 units of energy against 26 to 84 for the other six — so a fixed
    channel 0 would report the phase and amplitude tables at 6 times less
    signal-to-noise ratio than the array can give.
    """
    y = np.asarray(audio, dtype=np.float64)
    if ref_mic >= 0:
        return min(int(ref_mic), int(y.shape[0]) - 1)
    return int(np.argmax((y**2).sum(axis=-1)))


def track_bands(
    k: Any, bands: tuple[tuple[int, int], ...] = DEFAULT_BANDS
) -> dict[str, np.ndarray]:
    """``{band name: boolean track mask}`` over the ``(M,)`` harmonic indices."""
    ks = np.asarray(k).astype(int)
    return {band_name(lo, hi): (ks >= lo) & (ks <= hi) for lo, hi in bands}


def band_summary(
    values: Any, k: Any, bands: tuple[tuple[int, int], ...] = DEFAULT_BANDS
) -> dict[str, float | None]:
    """Band means of a per-track quantity — ``None`` for a band with no track."""
    v = np.asarray(values, dtype=np.float64)
    return {
        name: (round(float(v[sel].mean()), 8) if sel.any() else None)
        for name, sel in track_bands(k, bands).items()
    }


def drift_increments(phase_err: Any, fs_env: float) -> np.ndarray:
    """``(M, N-1)`` time derivative of the per-track phase error, in rad/s.

    The statistic every phase-model test is built from. An increment larger than
    ``pi`` means the unwrap itself is ambiguous, so a report carries the maximum
    increment beside the standard deviations.
    """
    p = np.asarray(phase_err, dtype=np.float64)
    return np.diff(p, axis=-1) * float(fs_env)


def rank_one_share(increments: Any) -> dict[str, Any]:
    """Top-eigenvalue share and mean pairwise correlation of drift increments.

    ``increments`` is ``(K, N)`` — one row per harmonic of ONE rotor. A shaft
    jitter model says every harmonic sees ``k`` times the same phase, so the
    correlation matrix is rank one and the share is 1. Independent per-harmonic
    drift (the pi-kalman model) gives a share near ``1 / K``.
    """
    d = np.asarray(increments, dtype=np.float64)
    n_k, n_t = d.shape
    if n_k < 2 or n_t < n_k + 2:
        return {"lambda1_share": None, "mean_corr": None, "n_k": int(n_k), "n_frames": int(n_t)}
    sd = d.std(axis=-1)
    keep = sd > 0
    if int(keep.sum()) < 2:
        return {"lambda1_share": None, "mean_corr": None, "n_k": int(n_k), "n_frames": int(n_t)}
    corr = np.corrcoef(d[keep])
    lam = np.sort(np.linalg.eigvalsh(corr))[::-1]
    off = corr[~np.eye(corr.shape[0], dtype=bool)]
    return {
        "lambda1_share": round(float(lam[0] / max(lam.sum(), 1e-30)), 6),
        "mean_corr": round(float(off.mean()), 6),
        "n_k": int(keep.sum()),
        "n_frames": int(n_t),
    }


def per_track_stats(
    amp: Any, phase_err: Any, mask: Any, fs_env: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(mean amplitude, amplitude CV, drift std)`` per track, on masked frames.

    ``amp`` / ``phase_err`` are ``(M, N)`` — ONE microphone. The mask selects the
    frames that are covered and not idle; drift uses the frame PAIRS inside it.
    """
    a = np.asarray(amp, dtype=np.float64)
    m = np.asarray(mask, dtype=bool)
    mean = a[:, m].mean(axis=-1) if m.any() else np.zeros(a.shape[0])
    std = a[:, m].std(axis=-1) if m.any() else np.zeros(a.shape[0])
    cv = std / np.maximum(mean, 1e-30)
    inc = drift_increments(phase_err, fs_env)
    pair = m[:-1] & m[1:]
    drift = inc[:, pair].std(axis=-1) if pair.any() else np.zeros(a.shape[0])
    return mean, cv, drift


def energy_ledger(
    audio: Any,
    recon: Any,
    track_energy: Any,
    k: Any,
    bands: tuple[tuple[int, int], ...] = DEFAULT_BANDS,
) -> dict[str, Any]:
    """Total / track / residual / cross-term energy, and the per-band shares.

    The tracks are not orthogonal — neighbouring harmonics of two rotors overlap
    inside their bands — so the three parts do not add up to the total by
    themselves. The cross term is what is left, and its size is the honest
    statement of how much of the decomposition is interference between tracks.
    """
    y = np.asarray(audio, dtype=np.float64)
    rec = np.asarray(recon, dtype=np.float64)
    total = float((y**2).sum())
    resid = float(((y - rec) ** 2).sum())
    tracks = float(np.asarray(track_energy, dtype=np.float64).sum())
    e_k = np.asarray(track_energy, dtype=np.float64)
    shares = {
        name: round(float(e_k[sel].sum() / max(tracks, 1e-30)), 6)
        for name, sel in track_bands(k, bands).items()
    }
    return {
        "total": total,
        "per_channel_total": [round(float(v), 6) for v in (y**2).sum(axis=-1)],
        "per_channel_residual": [round(float(v), 6) for v in ((y - rec) ** 2).sum(axis=-1)],
        "tracks": tracks,
        "residual": resid,
        "cross_term": total - tracks - resid,
        "track_fraction": round(tracks / max(total, 1e-30), 6),
        "residual_fraction": round(resid / max(total, 1e-30), 6),
        "band_share_of_tracks": shares,
    }


def phase_model_report(
    amp0: Any,
    pherr0: Any,
    valid: Any,
    rotor: Any,
    k: Any,
    mask: Any,
    fs_env: float,
    bands: tuple[tuple[int, int], ...] = DEFAULT_BANDS,
) -> dict[str, Any]:
    """The per-rotor phase and amplitude tables, on the reference microphone.

    Two readings of the same increments: the drift standard deviation against
    ``k`` (does the drift grow with the harmonic, as a shaft model says) and the
    rank-one share (do the harmonics drift together, as a shaft model says).

    ``max_abs_step_rad`` is the guard on both: the phase error is unwrapped, so
    a step at or above ``pi`` radians makes the unwrap ambiguous and the drift
    of that harmonic is then a lower bound, not a measurement.
    """
    mask = np.asarray(mask, dtype=bool) & np.asarray(valid, dtype=bool).all(axis=0)
    mean, cv, drift = per_track_stats(amp0, pherr0, mask, fs_env)
    inc = drift_increments(pherr0, fs_env)
    pair = mask[:-1] & mask[1:]
    rot = np.asarray(rotor, dtype=int)
    ks = np.asarray(k, dtype=int)
    per_rotor: dict[str, Any] = {}
    for rr in sorted(set(rot.tolist())):
        sel = rot == rr
        order = np.argsort(ks[sel])
        idx = np.flatnonzero(sel)[order]
        per_rotor[str(rr)] = {
            "k": [int(v) for v in ks[idx]],
            "drift_std_rad_s": [round(float(v), 5) for v in drift[idx]],
            "amp_mean": [round(float(v), 8) for v in mean[idx]],
            "amp_cv": [round(float(v), 5) for v in cv[idx]],
            "rank_one": rank_one_share(inc[np.ix_(idx, np.flatnonzero(pair))]),
        }
    return {
        "n_frames": int(mask.sum()),
        "max_abs_step_rad": round(
            float(np.abs(inc[:, pair]).max() / fs_env) if pair.any() else 0.0, 5
        ),
        "max_abs_drift_rad_s": round(float(np.abs(inc[:, pair]).max()) if pair.any() else 0.0, 4),
        "drift_std_rad_s_by_band": band_summary(drift, k, bands),
        "amp_mean_by_band": band_summary(mean, k, bands),
        "amp_cv_by_band": band_summary(cv, k, bands),
        "per_rotor": per_rotor,
    }


#: What :func:`residual_tones` measures, carried into the report beside it.
RESIDUAL_TONES_NOTE = (
    "Non-comb TONAL components of the residual: quasi-stationary lines that are "
    "NOT rotor harmonics (their order against every rotor is non-integer and it "
    "drifts independently of the rotor speed between segments), so a smooth-PSD "
    "noise model cannot represent them. Measurement only — nothing removes them."
)


def residual_tones(
    residual: Any,
    sr: float,
    r_audio: Any,
    *,
    segment_s: float = 8.0,
    n_peaks: int = 10,
    f_max: float = 2000.0,
    nperseg: int = 8192,
    prominence_db: float = 6.0,
    t_start_s: float = 0.0,
) -> dict[str, Any]:
    """Top ``n_peaks`` tonal peaks of the residual, per ``segment_s`` segment.

    One power spectral density per segment (Welch at ``nperseg``, averaged over
    the microphones), peaks in decibels with a prominence of at least
    ``prominence_db``, the strongest ``n_peaks`` below ``f_max`` kept. Each peak
    carries its distance to the nearest ROTOR ORDER — ``order`` is the peak
    frequency over the segment's mean rate of the rotor that fits it best, and
    ``order_dist`` is how far that is from a whole number. A rotor harmonic the
    decomposition failed to take out reads ``order_dist`` near 0 at a whole
    ``order``; a foreign tone does not, and its ``order`` moves between segments
    while the rotor speed does not.

    ``r_audio`` is the ``(R, T)`` audio-rate rate array over the same span.
    """
    from scipy.signal import find_peaks, welch

    y = np.atleast_2d(np.asarray(residual, dtype=np.float64))
    r = np.atleast_2d(np.asarray(r_audio, dtype=np.float64))
    n_t = int(y.shape[-1])
    step = max(1, int(round(float(segment_s) * float(sr))))
    segments: list[dict[str, Any]] = []
    for s0 in range(0, n_t, step):
        s1 = min(n_t, s0 + step)
        if s1 - s0 < nperseg:  # a tail too short to resolve is not reported
            continue
        f, p = welch(y[:, s0:s1], fs=float(sr), nperseg=int(nperseg), axis=-1)
        psd = np.asarray(p, dtype=np.float64).mean(axis=0)
        db = 10.0 * np.log10(np.maximum(psd, 1e-300))
        keep = np.asarray(f, dtype=np.float64) <= float(f_max)
        idx, props = find_peaks(db[keep], prominence=float(prominence_db))
        prom = np.asarray(props["prominences"], dtype=np.float64)
        order = np.argsort(prom)[::-1][: int(n_peaks)]
        rates = r[:, s0 : min(s1, r.shape[-1])].mean(axis=-1)
        peaks: list[dict[str, Any]] = []
        for j in order:
            f_pk = float(np.asarray(f)[keep][idx[j]])
            ords = f_pk / np.maximum(rates, 1e-12)
            dist = np.abs(ords - np.round(ords))
            best = int(np.argmin(dist))
            peaks.append(
                {
                    "freq_hz": round(f_pk, 3),
                    "prominence_db": round(float(prom[j]), 2),
                    "rotor": best,
                    "order": round(float(ords[best]), 3),
                    "order_dist": round(float(dist[best]), 3),
                }
            )
        segments.append(
            {
                "t_start_s": round(s0 / float(sr) + float(t_start_s), 3),
                "t_end_s": round(s1 / float(sr) + float(t_start_s), 3),
                "mean_rev_s": [round(float(v), 3) for v in rates],
                "peaks": peaks,
            }
        )
    return {
        "note": RESIDUAL_TONES_NOTE,
        "params": {
            "segment_s": float(segment_s),
            "n_peaks": int(n_peaks),
            "f_max_hz": float(f_max),
            "nperseg": int(nperseg),
            "prominence_db": float(prominence_db),
        },
        "segments": segments,
    }


def welch_psd(audio: Any, sr: float, nperseg: int = NPERSEG) -> tuple[np.ndarray, np.ndarray]:
    """``(frequencies, power spectral density)`` of ``(C, T)`` audio."""
    from scipy.signal import welch

    f, p = welch(np.asarray(audio, dtype=np.float64), fs=float(sr), nperseg=nperseg, axis=-1)
    return np.asarray(f, dtype=np.float64), np.asarray(p, dtype=np.float64)
