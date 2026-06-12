"""Uniform (regular sample-rate) time series — audio, video frames, etc.

Storage model
-------------
We store the raw sample array, the sample rate, an absolute int64 anchor
(`t_start_ticks`), and a sub-sample **phase** — the offset of `samples[0]`'s
left edge from `t_start`, in sample units, ∈ [−1, 0].

No per-sample edge times are stored.  Every edge is derived from these four
quantities.  Because the phase is *relative* and bounded by one sample, a
float64 stores it to ≈ 1e‑21 s — effectively exact.  The Unix‑magnitude
precision problem never touches it; only the (exact int64) anchor carries the
magnitude.

For integer `sr` (all real audio rates), the cut‑time → sample‑index mapping
uses exact Python‑int arithmetic (`divmod(Δ·sr, TICKS_PER_SECOND)`) — no
float, no epsilon.  Non‑integer `sr` falls back to a float index with a fixed
`1e‑6`‑sample epsilon.

The public `.t_start`, `.t_end`, `.duration` return float seconds;
`*_ticks` accessors return exact int64 values.

    t_start_ticks + phase·1/sr        t_start_ticks + (phase + N)·1/sr
    │                                 │
    ▼                                 ▼
    ┌───┬───┬───┬───┬───┬───┬───┬───┐
    │ 0 │ 1 │ 2 │ 3 │ 4 │ 5 │ 6 │ 7 │   samples (each cell = 1/sr)
    └───┴───┴───┴───┴───┴───┴───┴───┘
        ▲                          ▲
        t_start                    t_end          (declared interval)

Invariants
~~~~~~~~~~
* `-1 ≤ phase ≤ 0`
* The declared `t_start` / `t_end` may sit inside the first / last sample
  cell (sub‑sample boundary slack).
* `slice(a,b) ⊕ slice(b,c) == slice(a,c)` holds exactly at the sample level
  for integer `sr`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from ._ticks import TICKS_PER_SECOND, _c_to_ticks, secs_to_ticks, ticks_array_to_secs, ticks_to_secs
from .base import DomainError, IncompatibleSeriesError, TimeSeries

# Index‑space epsilon used ONLY on the non‑integer‑sr fallback path.
INDEX_EPSILON: float = 1e-6


@dataclass(frozen=True, eq=False)
class UniformSeries(TimeSeries):
    """A regular-rate sequence of samples along axis 0.

    Parameters
    ----------
    samples : np.ndarray
        Shape ``(…, N)``.  Axis ‑1 is time.
    sr : float
        Sample rate in Hz.
    t_start_ticks : int
        Absolute domain-start anchor (int64 ticks).
    dur_ticks : int
        Declared domain duration (int64 ticks).  ``t_end_ticks == t_start_ticks + dur_ticks``.
        May differ from ``N/sr`` after sub‑sample slicing.
    phase : float
        Offset of ``samples[0]``'s left edge from ``t_start``, in **sample
        units**, ∈ [−1, 0].  The absolute edge time is ``t_first_edge``.
    """

    samples: np.ndarray = field(repr=False)
    sr: float
    t_start_ticks: int
    dur_ticks: int
    phase: float

    # ---- validation -----------------------------------------------------
    def __post_init__(self) -> None:
        if self.samples.ndim < 1:
            raise ValueError("samples must have at least one axis")
        if self.sr <= 0:
            raise ValueError("sr must be > 0")
        if self.phase < -1.0 or self.phase > 0.0:
            raise ValueError(f"phase ({self.phase}) must be in [-1, 0]")
        # The declared domain is bounded by the sample grid within one sample.
        N = self.samples.shape[-1]
        if N == 0:
            return
        # No stored dur field — derived from N/sr.  The grid anchor check:
        # t_first_edge = t_start + phase/sr must be ≤ t_start (phase ≤ 0)
        # and t_end = t_start + N/sr must be ≤ t_start + (phase+N)/sr + 1/sr
        # The latter is always true since N/sr ≤ (N+1)/sr and phase ≤ 0 => N/sr ≤ N/sr + 1/sr.
        # The check that the declared span doesn't exceed the grid by more than
        # one cell is:
        #   N/sr  ≤  (phase + N + 1)/sr   ⇔  -1 ≤ phase + 1  ⇔  phase ≥ 0 or phase ≥ -1
        # Already enforced above.

    # ---- constructors ---------------------------------------------------
    @classmethod
    def from_samples(
        cls,
        samples: np.ndarray,
        sr: float,
        *,
        t_start: float | int = 0.0,
    ) -> UniformSeries:
        """Build a series whose declared interval matches its sample grid exactly.

        ``t_start`` may be float seconds or int64 ticks.
        """
        samples = np.asarray(samples)
        N = samples.shape[-1]
        sr = float(sr)
        if isinstance(t_start, (int, np.integer)):
            t0 = int(t_start)
        else:
            t0 = secs_to_ticks(float(t_start))
        dur = round(N * TICKS_PER_SECOND / sr)
        return cls(
            samples=samples,
            sr=sr,
            t_start_ticks=t0,
            dur_ticks=dur,
            phase=0.0,
        )

    @classmethod
    def from_ticks(
        cls,
        samples: np.ndarray,
        sr: float,
        *,
        t_start: int = 0,
        dur: int = 0,
        phase: float = 0.0,
    ) -> UniformSeries:
        """Build from explicit int64 ticks and phase."""
        return cls(
            samples=np.asarray(samples),
            sr=float(sr),
            t_start_ticks=int(t_start),
            dur_ticks=int(dur),
            phase=float(phase),
        )

    # ---- domain properties (seconds) ------------------------------------
    @property
    def t_start(self) -> float:
        return ticks_to_secs(self.t_start_ticks)

    @property
    def t_end(self) -> float:
        return ticks_to_secs(self.t_start_ticks + self.dur_ticks)

    @property
    def t_end_ticks(self) -> int:
        return self.t_start_ticks + self.dur_ticks

    # ---- grid -----------------------------------------------------------
    @property
    def t_first_edge(self) -> float:
        return ticks_to_secs(self.t_start_ticks) + self.phase / self.sr

    @property
    def t_first_edge_ticks(self) -> int:
        return round(self.t_start_ticks + self.phase * TICKS_PER_SECOND / self.sr)

    @property
    def n_samples(self) -> int:
        return int(self.samples.shape[-1])

    @property
    def timestamps(self) -> np.ndarray:
        """Absolute sample times as float seconds."""
        return self.sample_times() - self.t_first_edge

    @property
    def timestamp_ticks(self) -> np.ndarray:
        """Absolute sample times as int64 ticks (nearest)."""
        return self.sample_times_ticks() - self.t_start_ticks

    @property
    def channel_shape(self) -> tuple[int, ...]:
        return tuple(self.samples.shape[:-1])

    @property
    def values(self) -> np.ndarray:
        return self.samples

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, i: Any) -> Any:
        return self.samples[..., i]

    # ---- sample‑index mapping (internal) --------------------------------
    def _cut_to_indices(self, ra: int, rb: int) -> tuple[int, int]:
        """Map relative tick cut-points ``[ra, rb)`` to sample indices ``(ka, kb)``.

        For integer ``sr`` the mapping is exact (``divmod``).  For
        non‑integer ``sr``, a float index with ``INDEX_EPSILON`` is used.
        ``ra, rb`` are relative to ``self.t_start_ticks``.
        """
        sr = self.sr
        sr_int = int(sr)
        if sr == float(sr_int):  # integer sr → exact path
            tps = TICKS_PER_SECOND
            pticks = round(-self.phase * tps)  # ≥ 0, since phase ≤ 0
            # ---- left (floor) ------------------------------------------
            # k = floor(ra·sr / tps - phase)
            qa, ra_rem = divmod(ra * sr_int, tps)
            ka = qa + (1 if ra_rem + pticks >= tps else 0)
            # ---- right (ceil) ------------------------------------------
            # k = ceil(rb·sr / tps - phase)
            qb, rb_rem = divmod(rb * sr_int, tps)
            if rb_rem == 0 and pticks == 0:
                kb = qb
            elif rb_rem + pticks <= tps:
                kb = qb + 1
            else:
                kb = qb + 2
        else:  # non‑integer sr fallback
            tps = TICKS_PER_SECOND
            k_ra = ra * sr / tps - self.phase
            k_rb = rb * sr / tps - self.phase
            ka = int(math.floor(k_ra + INDEX_EPSILON))
            kb = int(math.ceil(k_rb - INDEX_EPSILON))
        ka = max(0, ka)
        kb = min(self.n_samples, kb)
        if kb < ka:
            kb = ka
        return ka, kb

    # ---- slice ----------------------------------------------------------
    def slice(self, t_a: float | int, t_b: float | int) -> UniformSeries:
        ta_tick = _c_to_ticks(t_a) if not isinstance(t_a, int) else t_a
        tb_tick = _c_to_ticks(t_b) if not isinstance(t_b, int) else t_b
        t0 = self.t_start_ticks
        dur_ticks = self.t_end_ticks - t0
        if ta_tick < t0 or tb_tick > t0 + dur_ticks or ta_tick > tb_tick:
            raise DomainError(
                f"slice({ta_tick}, {tb_tick}) outside [{t0}, {t0 + dur_ticks}] (ticks)"
            )
        ta_tick = max(ta_tick, t0)
        tb_tick = min(tb_tick, t0 + dur_ticks)

        ra = ta_tick - t0
        rb = tb_tick - t0
        ka, kb = self._cut_to_indices(ra, rb)

        new_samples = self.samples[..., ka:kb]
        # Recover the new phase: offset of sample 0's left edge from the new
        # t_start, in sample units.  Derived from the floor of the left cut
        # so it is consistent with the stored ka.
        sr = self.sr
        new_phase = float(self.phase + ka - ra * sr / TICKS_PER_SECOND)
        # new_phase is already in [-1, 0] by construction; clamp to be safe.
        if new_phase > 0.0:
            new_phase -= 1.0
        elif new_phase < -1.0:
            new_phase += 1.0

        return UniformSeries(
            samples=new_samples,
            sr=sr,
            t_start_ticks=ta_tick,
            dur_ticks=rb - ra,
            phase=new_phase,
        )

    # ---- shift ----------------------------------------------------------
    def shift(self, t_delta: float | int) -> UniformSeries:
        dt = _c_to_ticks(t_delta) if not isinstance(t_delta, int) else t_delta
        if dt == 0:
            return self
        return replace(self, t_start_ticks=self.t_start_ticks + dt)

    # ---- concat ---------------------------------------------------------
    def concat(self, other: UniformSeries) -> UniformSeries:
        if not isinstance(other, UniformSeries):
            raise IncompatibleSeriesError(
                f"cannot concat UniformSeries with {type(other).__name__}"
            )
        if self.sr != other.sr:
            raise IncompatibleSeriesError(f"sample rates differ: {self.sr} vs {other.sr}")
        if self.samples.shape[:-1] != other.samples.shape[:-1]:
            raise IncompatibleSeriesError(
                f"channel shapes differ: {self.samples.shape[:-1]} vs {other.samples.shape[:-1]}"
            )

        sr = self.sr
        # Grid offset in sample‑index space.  ``other`` is glued so its
        # t_start lands at self's t_end; the offset from self's first edge
        # to other's first edge (aligned) is:
        #   self.dur_ticks + other.phase/sr - self.phase/sr
        # converted to sample units via (Δticks * sr / TPS).
        sdur = self.dur_ticks
        tps = TICKS_PER_SECOND
        sr_int = int(sr)
        if sr == float(sr_int):
            q, r = divmod(sdur * sr_int, tps)
            k_offset = q + r / tps + other.phase - self.phase
        else:
            k_offset = sdur * sr / tps + other.phase - self.phase

        k_int = round(k_offset)
        if abs(k_offset - k_int) > 0.1:
            raise IncompatibleSeriesError(
                f"incompatible sample grids: phase offset {k_offset} samples "
                f"(must be integer; nearest is {k_int})"
            )
        n_self = self.n_samples
        if k_int == n_self:
            new_samples = np.concatenate([self.samples, other.samples], axis=-1)
        elif k_int == n_self - 1:
            new_samples = np.concatenate([self.samples, other.samples[..., 1:]], axis=-1)
        else:
            raise IncompatibleSeriesError(
                f"incompatible sample grids: integer offset {k_int} "
                f"(expected {n_self} or {n_self - 1})"
            )

        new_dur_ticks = self.dur_ticks + other.dur_ticks
        return UniformSeries(
            samples=new_samples,
            sr=sr,
            t_start_ticks=self.t_start_ticks,
            dur_ticks=new_dur_ticks,
            phase=self.phase,
        )

    # ---- equality -------------------------------------------------------
    def equal(self, other: TimeSeries) -> bool:
        if not isinstance(other, UniformSeries):
            return False
        if not (
            self.t_start_ticks == other.t_start_ticks
            and self.dur_ticks == other.dur_ticks
            and self.sr == other.sr
        ):
            return False
        if not np.array_equal(self.samples, other.samples):
            return False
        # Compare phases loosely (float, but bounded and precision‑safe).
        return not abs(self.phase - other.phase) > 1e-12

    # ---- utilities ------------------------------------------------------
    def sample_times(self) -> np.ndarray:
        """Absolute left‑edge time of each sample in float seconds."""
        return self.t_first_edge + np.arange(self.n_samples) / self.sr

    def sample_times_ticks(self) -> np.ndarray:
        """Absolute left‑edge time of each sample in int64 ticks (nearest)."""
        edges_s = self.t_first_edge + np.arange(self.n_samples) / self.sr
        result = np.rint(edges_s * TICKS_PER_SECOND)
        return result.astype(np.int64)

    def time_to_index(self, t: float | int) -> int:
        """Index of the sample cell containing time ``t``."""
        t_tick = _c_to_ticks(t) if not isinstance(t, int) else t
        delta = t_tick - self.t_first_edge_ticks
        sr = self.sr
        sr_int = int(sr)
        if sr == float(sr_int):
            q, r = divmod(delta * sr_int, TICKS_PER_SECOND)
            k = q + r / TICKS_PER_SECOND
        else:
            k = delta * sr / TICKS_PER_SECOND
        return int(math.floor(k))

    # ---- interpolation / resampling ------------------------------------
    def interpolate(
        self,
        times,
        *,
        kind: str = "linear",
        fill: str = "clamp",
    ) -> np.ndarray:
        """Evaluate signal at absolute query times."""
        times = np.asarray(times)
        t_sec = ticks_array_to_secs(times) if times.dtype.kind == "i" else times.astype(np.float64)

        if self.n_samples == 0:
            if fill == "error":
                raise DomainError("interpolate on empty UniformSeries")
            fill_val = np.nan if fill == "nan" else self.samples.flat[0:0]
            shape = (*self.channel_shape, len(times))
            result = np.full(shape, fill_val, dtype=np.float64)
            if result.size == 0:
                return np.zeros((*self.channel_shape, len(times)), dtype=np.float64)
            return result

        # Sample grid: left-edges of each sample cell.
        grid_t = self.sample_times()  # shape (N,)

        if kind != "linear":
            raise ValueError(f"unsupported interpolation kind: {kind!r}")

        # Per-channel linear interpolation.
        vals = np.asarray(self.samples, dtype=np.float64)
        if vals.ndim == 1:
            result = np.interp(t_sec, grid_t, vals)
        else:
            # Multi-channel: reshape to (-1, N), interp per row.
            N = vals.shape[-1]
            rest = vals.shape[:-1]
            flat = vals.reshape(-1, N)
            n_ch = flat.shape[0]
            result_flat = np.empty((n_ch, len(t_sec)), dtype=np.float64)
            for c in range(n_ch):
                result_flat[c, :] = np.interp(t_sec, grid_t, flat[c, :])
            result = result_flat.reshape(*rest, len(t_sec))

        # -- extrapolation ------------------------------------------------
        if fill == "clamp":
            pass  # np.interp defaults to clamp
        elif fill == "nan":
            mask = (t_sec < grid_t[0]) | (t_sec > grid_t[-1])
            if result.ndim > 1:
                result[mask] = np.nan
            else:
                result[mask] = np.nan
        elif fill == "error":
            if t_sec[0] < grid_t[0] - 1e-12 or t_sec[-1] > grid_t[-1] + 1e-12:
                raise DomainError(
                    f"interpolate query times [{t_sec[0]:.6g}, {t_sec[-1]:.6g}] "
                    f"outside data span [{grid_t[0]:.6g}, {grid_t[-1]:.6g}]"
                )
        else:
            raise ValueError(f"unsupported fill: {fill!r}")

        return result

    def resample(
        self,
        new_sr: float,
        *,
        kind: str = "linear",
    ) -> UniformSeries:
        """Resample to a new sample rate over the same declared domain.

        The output grid uses ``phase=0`` (sample ``k`` at
        ``t_start + k/new_sr``), matching the legacy ``arange(F)*hop/sr``
        pattern.  Multi-channel values are interpolated per-channel.
        """
        if new_sr <= 0:
            raise ValueError(f"new_sr must be > 0, got {new_sr}")
        dur_s = self.duration
        N_new = max(1, round(dur_s * new_sr))
        # Adjust duration so N_new/sr fits inside the declared domain.
        new_dur_ticks = round(N_new * TICKS_PER_SECOND / new_sr)
        grid = self.t_start + np.arange(N_new) / new_sr
        vals = self.interpolate(grid, kind=kind, fill="clamp")
        return UniformSeries(
            samples=vals,
            sr=new_sr,
            t_start_ticks=self.t_start_ticks,
            dur_ticks=new_dur_ticks,
            phase=0.0,
        )
