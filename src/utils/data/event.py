"""Non-uniform / sparse event time series.

An `EventSeries` stores timestamps as int64 tick counts **relative to
`t_start_ticks`**.  `__getitem__` returns absolute float seconds;
`abs_timestamps` / `abs_timestamps_ticks` give the absolute arrays.

This makes `shift` O(1): only the scalar `t_start_ticks` anchor moves.
All time comparisons are exact (== on ints).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ._ticks import (
    TICKS_PER_SECOND,
    _c_to_ticks,
    secs_array_to_ticks,
    secs_to_ticks,
    ticks_array_to_secs,
    ticks_to_secs,
)
from .base import DomainError, IncompatibleSeriesError, TimeSeries


@dataclass(frozen=True, eq=False)
class EventSeries(TimeSeries):
    """Sorted point-event time series.

    Parameters
    ----------
    timestamp_ticks : np.ndarray, shape (M,), dtype int64
        Event times relative to `t_start_ticks` (int64 ticks).
    values : np.ndarray | None
        Shape `(M, ...)` payload, aligned to `timestamp_ticks`.
    t_start_ticks : int
        Absolute domain-start anchor (int64 ticks).
    dur_ticks : int
        Declared domain duration (int64 ticks).  `t_end_ticks == t_start_ticks + dur_ticks`.
    """

    timestamp_ticks: np.ndarray = field(repr=False)
    values: np.ndarray | None = field(repr=False, default=None)
    t_start_ticks: int = 0
    dur_ticks: int = 0

    # ---- validation -----------------------------------------------------
    def __post_init__(self) -> None:
        ts = np.asarray(self.timestamp_ticks, dtype=np.int64)
        object.__setattr__(self, "timestamp_ticks", ts)
        if ts.ndim != 1:
            raise ValueError("timestamps must be 1-D")
        if self.values is not None:
            vals = np.asarray(self.values)
            object.__setattr__(self, "values", vals)
            if vals.shape[0] != ts.shape[0]:
                raise ValueError(
                    f"values.shape[0]={vals.shape[0]} != len(timestamps)={ts.shape[0]}"
                )
        if self.dur_ticks < 0:
            raise ValueError(f"dur_ticks ({self.dur_ticks}) < 0")
        if ts.size:
            if ts[0] < 0:
                raise ValueError("events before t_start (relative < 0)")
            if ts[-1] >= self.dur_ticks:
                raise ValueError(
                    f"events outside domain (relative {ts[-1]} >= dur {self.dur_ticks})"
                )
            if not np.all(np.diff(ts) >= 0):
                raise ValueError("timestamps must be sorted ascending")

    # ---- constructors ---------------------------------------------------
    @classmethod
    def from_events(
        cls,
        timestamps: np.ndarray,
        values: np.ndarray | None = None,
        *,
        t_start: float | int | None = None,
        t_end: float | int | None = None,
    ) -> "EventSeries":
        """Build from absolute timestamps (float seconds or int ticks).

        ``t_start`` / ``t_end`` are inferred from the first/last event if
        not given.
        """
        ts = np.asarray(timestamps)
        if np.issubdtype(ts.dtype, np.floating):
            ts_ticks = secs_array_to_ticks(ts)
        else:
            ts_ticks = ts.astype(np.int64)

        if t_start is None:
            t_start_ticks = int(ts_ticks[0]) if ts_ticks.size else 0
        elif isinstance(t_start, (int, np.integer)):
            t_start_ticks = int(t_start)
        else:
            t_start_ticks = secs_to_ticks(float(t_start))

        rel_ts = ts_ticks - t_start_ticks

        if t_end is None:
            if rel_ts.size:
                dur_ticks = int(rel_ts[-1]) + 1  # strict half-open: > last event
            else:
                dur_ticks = 0
        elif isinstance(t_end, (int, np.integer)):
            dur_ticks = int(t_end) - t_start_ticks
        else:
            dur_ticks = secs_to_ticks(float(t_end)) - t_start_ticks

        return cls._from_relative(rel_ts, values, t_start_ticks, dur_ticks)

    @classmethod
    def from_ticks(
        cls,
        timestamps: np.ndarray,
        values: np.ndarray | None = None,
        *,
        t_start: int = 0,
        dur: int,
    ) -> "EventSeries":
        """Build from relative int64 ticks and explicit domain."""
        rel_ts = np.asarray(timestamps, dtype=np.int64)
        return cls._from_relative(rel_ts, values, int(t_start), int(dur))

    @classmethod
    def _from_relative(
        cls,
        timestamps: np.ndarray,
        values: np.ndarray | None,
        t_start_ticks: int,
        dur_ticks: int,
    ) -> "EventSeries":
        """Fast-path: timestamps already relative int64 ticks.  Bypasses
        __post_init__ conversion.
        """
        self = object.__new__(cls)
        object.__setattr__(self, "timestamp_ticks", np.asarray(timestamps, dtype=np.int64))
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "t_start_ticks", int(t_start_ticks))
        object.__setattr__(self, "dur_ticks", int(dur_ticks))
        return self

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

    @property
    def timestamp(self) -> np.ndarray:
        """Relative event times as float seconds."""
        return ticks_array_to_secs(self.timestamp_ticks)

    @property
    def timestamps(self) -> np.ndarray:
        """Backward-compatible alias for ``timestamp_ticks``."""
        return self.timestamp_ticks

    # ---- array accessors ------------------------------------------------
    @property
    def abs_timestamps(self) -> np.ndarray:
        """Absolute event times as float seconds."""
        return ticks_array_to_secs(self.timestamp_ticks + self.t_start_ticks)

    @property
    def abs_timestamps_ticks(self) -> np.ndarray:
        """Absolute event times as int64 ticks."""
        return self.timestamp_ticks + self.t_start_ticks

    # ---- shape ----------------------------------------------------------
    def __len__(self) -> int:
        return int(self.timestamp_ticks.shape[0])

    def __getitem__(self, i: Any):
        t = ticks_to_secs(int(self.timestamp_ticks[i]) + self.t_start_ticks)
        if self.values is None:
            return t
        return t, self.values[i]

    # ---- slice ----------------------------------------------------------
    def slice(self, t_a: float | int, t_b: float | int) -> "EventSeries":
        a_tick = _c_to_ticks(t_a) if not isinstance(t_a, int) else t_a
        b_tick = _c_to_ticks(t_b) if not isinstance(t_b, int) else t_b
        ta = self.t_start_ticks
        dur = self.dur_ticks
        if a_tick < ta or b_tick > ta + dur or a_tick > b_tick:
            raise DomainError(
                f"slice({a_tick}, {b_tick}) outside [{ta}, {ta + dur}] (ticks)"
            )
        a_tick = max(a_tick, ta)
        b_tick = min(b_tick, ta + dur)

        ra = a_tick - ta  # relative to self.t_start_ticks
        rb = b_tick - ta
        ts = self.timestamp_ticks  # already relative int64
        lo = int(np.searchsorted(ts, ra, side="left"))
        hi = int(np.searchsorted(ts, rb, side="left"))  # half-open at right
        new_ts = ts[lo:hi] - ra  # rebase to new t_start = a_tick
        new_vals = None if self.values is None else self.values[lo:hi]
        return EventSeries._from_relative(new_ts, new_vals, a_tick, rb - ra)

    # ---- shift ----------------------------------------------------------
    def shift(self, t_delta: float | int) -> "EventSeries":
        dt = _c_to_ticks(t_delta) if not isinstance(t_delta, int) else t_delta
        if dt == 0:
            return self
        return EventSeries._from_relative(
            self.timestamp_ticks, self.values,
            self.t_start_ticks + dt, self.dur_ticks,
        )

    # ---- concat ---------------------------------------------------------
    def concat(self, other: "EventSeries") -> "EventSeries":
        if not isinstance(other, EventSeries):
            raise IncompatibleSeriesError(
                f"cannot concat EventSeries with {type(other).__name__}"
            )
        if (self.values is None) != (other.values is None):
            raise IncompatibleSeriesError("one has values, the other does not")
        if self.values is not None and other.values is not None:
            if self.values.shape[1:] != other.values.shape[1:]:
                raise IncompatibleSeriesError(
                    f"value shapes differ: {self.values.shape[1:]} vs {other.values.shape[1:]}"
                )

        # Glue other so its t_start lands at self.t_end.  Both store relative
        # timestamps, so other's events relative to self.t_start_ticks are
        # just its relative timestamps + self's duration.
        self_dur = self.dur_ticks
        other_dur = other.dur_ticks
        new_ts = np.concatenate([self.timestamp_ticks, other.timestamp_ticks + int(self_dur)])
        new_t_start = self.t_start_ticks
        new_dur = self_dur + other_dur
        new_vals: np.ndarray | None
        if self.values is None:
            new_vals = None
        else:
            new_vals = np.concatenate([self.values, other.values])  # type: ignore[arg-type]
        return EventSeries._from_relative(new_ts, new_vals, new_t_start, new_dur)

    # ---- equality -------------------------------------------------------
    def equal(self, other: TimeSeries) -> bool:
        if not isinstance(other, EventSeries):
            return False
        if not (
            self.t_start_ticks == other.t_start_ticks
            and self.dur_ticks == other.dur_ticks
        ):
            return False
        if not np.array_equal(self.timestamp_ticks, other.timestamp_ticks):
            return False
        if (self.values is None) != (other.values is None):
            return False
        if self.values is None:
            return True
        return np.array_equal(self.values, other.values)

    # ---- interpolation / resampling ------------------------------------
    def interpolate(
        self, times, *, kind: str = "linear", fill: str = "clamp",
    ) -> np.ndarray:
        """Evaluate signal at absolute query times by interpolating between
        event values.

        Requires ``values`` not None.
        """
        if self.values is None:
            raise ValueError("cannot interpolate EventSeries with no values")

        times = np.asarray(times)
        if times.dtype.kind == 'i':
            t_sec = ticks_array_to_secs(times)
        else:
            t_sec = times.astype(np.float64)

        # Event times in float seconds.
        ev_t = self.abs_timestamps  # shape (M,)
        vals = np.asarray(self.values, dtype=np.float64)  # (M, ...)

        if len(ev_t) == 0:
            if fill == "error":
                raise DomainError("interpolate on empty EventSeries")
            fill_val = np.nan if fill == "nan" else 0.0
            shape = (len(times), *vals.shape[1:])
            return np.full(shape, fill_val, dtype=np.float64)

        if kind != "linear":
            raise ValueError(f"unsupported interpolation kind: {kind!r}")

        if vals.ndim == 1:
            result = np.interp(t_sec, ev_t, vals)
        else:
            M = vals.shape[0]
            rest = vals.shape[1:]
            flat = vals.reshape(M, -1)
            n_ch = flat.shape[1]
            result_flat = np.empty((len(t_sec), n_ch), dtype=np.float64)
            for c in range(n_ch):
                result_flat[:, c] = np.interp(t_sec, ev_t, flat[:, c])
            result = result_flat.reshape(len(t_sec), *rest)

        # -- extrapolation ------------------------------------------------
        if fill == "clamp":
            pass
        elif fill == "nan":
            mask = (t_sec < ev_t[0]) | (t_sec > ev_t[-1])
            if result.ndim > 1:
                result[mask] = np.nan
            else:
                result[mask] = np.nan
        elif fill == "error":
            if t_sec[0] < ev_t[0] - 1e-12 or t_sec[-1] > ev_t[-1] + 1e-12:
                raise DomainError(
                    f"interpolate query times [{t_sec[0]:.6g}, {t_sec[-1]:.6g}] "
                    f"outside event span [{ev_t[0]:.6g}, {ev_t[-1]:.6g}]"
                )
        else:
            raise ValueError(f"unsupported fill: {fill!r}")

        return result

    def interpolate_uniform(
        self,
        sr: float,
        *,
        t_start: float | int | None = None,
        t_end: float | int | None = None,
        kind: str = "linear",
    ) -> "UniformSeries":
        """Convert this event series to a uniformly-sampled ``UniformSeries``
        at sample rate ``sr``.

        The output grid uses ``phase=0`` (sample ``k`` at
        ``t_start + k/sr``).  Default domain is the event series's own
        ``[t_start, t_end)``.
        """
        from .uniform import UniformSeries

        if sr <= 0:
            raise ValueError(f"sr must be > 0, got {sr}")

        # Determine domain in ticks.
        if t_start is None:
            t0_ticks = self.t_start_ticks
        else:
            t0_ticks = _c_to_ticks(t_start) if not isinstance(t_start, int) else t_start
        if t_end is None:
            dur = self.dur_ticks
        else:
            te_ticks = _c_to_ticks(t_end) if not isinstance(t_end, int) else t_end
            dur = te_ticks - t0_ticks
        if dur <= 0:
            raise ValueError(f"domain duration must be > 0 ticks, got {dur}")

        dur_s = ticks_to_secs(dur)
        N = max(1, round(dur_s * sr))
        new_dur_ticks = round(N * TICKS_PER_SECOND / sr)
        grid_s = ticks_to_secs(t0_ticks) + np.arange(N) / sr

        vals = self.interpolate(grid_s, kind=kind, fill="clamp")
        return UniformSeries(
            samples=vals,
            sr=sr,
            t_start_ticks=t0_ticks,
            dur_ticks=new_dur_ticks,
            phase=0.0,
        )
