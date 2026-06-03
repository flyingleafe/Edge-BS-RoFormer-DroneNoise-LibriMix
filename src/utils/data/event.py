"""Non-uniform / sparse event time series.

An `EventSeries` stores timestamps **relative to `t_start`**.
`timestamps[0]` is the offset of the first event; `__getitem__` returns
absolute times by adding `t_start` back.

This makes `shift(t_delta)` cheap: only `t_start` changes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ._floats import DEFAULT_ATOL, DEFAULT_RTOL, t_atol_at, tclose
from .base import DomainError, IncompatibleSeriesError, TimeSeries


@dataclass(frozen=True, eq=False)
class EventSeries(TimeSeries):
    """Sorted point-event time series.

    Parameters
    ----------
    timestamps : np.ndarray, shape (M,)
        May be passed as **absolute** seconds (constructor normalises) or
        as **relative** offsets from `t_start`.  Internally stored relative.
    values : np.ndarray | None
        Shape `(M, ...)` or `None`. Aligned to `timestamps`.
    t_start, t_end : float
        Declared domain.
    """

    timestamps: np.ndarray = field(repr=False)
    values: np.ndarray | None = field(repr=False, default=None)
    t_start: float = 0.0
    t_end: float = 0.0

    def __post_init__(self) -> None:
        ts = np.asarray(self.timestamps, dtype=np.float64)
        object.__setattr__(self, "timestamps", ts)
        if ts.ndim != 1:
            raise ValueError("timestamps must be 1-D")
        if self.values is not None:
            vals = np.asarray(self.values)
            object.__setattr__(self, "values", vals)
            if vals.shape[0] != ts.shape[0]:
                raise ValueError(
                    f"values.shape[0]={vals.shape[0]} != len(timestamps)={ts.shape[0]}"
                )
        if self.t_end < self.t_start:
            raise ValueError(f"t_end ({self.t_end}) < t_start ({self.t_start})")
        if ts.size:
            duration = self.t_end - self.t_start
            lo_atol = t_atol_at(self.t_start)
            hi_atol = t_atol_at(self.t_end)
            # Heuristic: are timestamps absolute or already relative?
            # Relative offsets always lie in [0, duration) and, when t_start is
            # large, are clearly smaller than t_start.
            fits_in_duration = ts[-1] <= duration + hi_atol and ts[0] >= -lo_atol
            starts_near_t_start = tclose(ts[0], self.t_start, atol=lo_atol)
            t_start_is_zero = tclose(self.t_start, 0.0, atol=lo_atol)
            if fits_in_duration and not (starts_near_t_start and not t_start_is_zero):
                rel_ts = ts  # already relative
            else:
                rel_ts = ts - self.t_start  # convert absolute → relative
            if rel_ts[0] < -lo_atol:
                raise ValueError("events before t_start")
            if rel_ts[-1] >= duration + hi_atol:
                raise ValueError("events after t_end")
            if not np.all(np.diff(rel_ts) >= 0):
                raise ValueError("timestamps must be sorted ascending")
            object.__setattr__(self, "timestamps", rel_ts)

    @classmethod
    def from_events(
        cls,
        timestamps: np.ndarray,
        values: np.ndarray | None = None,
        t_start: float | None = None,
        t_end: float | None = None,
    ) -> "EventSeries":
        ts = np.asarray(timestamps, dtype=np.float64)
        if t_start is None:
            t_start = float(ts[0]) if ts.size else 0.0
        if t_end is None:
            # Half-open: choose an end strictly greater than the last event.
            t_end = float(ts[-1]) + 1e-9 if ts.size else float(t_start)
        # Pass relative timestamps so __post_init__ does not double-shift.
        rel_ts = ts - float(t_start)
        return cls(timestamps=rel_ts, values=values, t_start=float(t_start), t_end=float(t_end))

    @classmethod
    def _from_relative(
        cls,
        timestamps: np.ndarray,
        values: np.ndarray | None,
        t_start: float,
        t_end: float,
    ) -> "EventSeries":
        """Fast-path construction when timestamps are known to be relative.
        Bypasses the absolute/relative heuristic in __post_init__.
        """
        self = object.__new__(cls)
        object.__setattr__(self, "timestamps", np.asarray(timestamps, dtype=np.float64))
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "t_start", float(t_start))
        object.__setattr__(self, "t_end", float(t_end))
        return self

    # ------------------------------------------------------------------ helpers
    @property
    def abs_timestamps(self) -> np.ndarray:
        """Absolute event times (relative storage + t_start)."""
        return self.timestamps + self.t_start

    # ------------------------------------------------------------------ shape
    def __len__(self) -> int:
        return int(self.timestamps.shape[0])

    def __getitem__(self, i: Any):
        ts = self.timestamps[i] + self.t_start
        if self.values is None:
            return ts
        return ts, self.values[i]

    # ------------------------------------------------------------------ slice
    def slice(self, t_a: float, t_b: float) -> "EventSeries":
        lo_atol = t_atol_at(self.t_start)
        hi_atol = t_atol_at(self.t_end)
        ab_atol = t_atol_at(max(abs(t_a), abs(t_b)))
        if t_a < self.t_start - lo_atol or t_b > self.t_end + hi_atol or t_a > t_b + ab_atol:
            raise DomainError(
                f"slice({t_a}, {t_b}) outside [{self.t_start}, {self.t_end}]"
            )
        # When slicing *exactly* at a domain boundary, accept events that lie
        # within the boundary's atol slack (consistent with __post_init__).
        lo_search = self.t_start - lo_atol if t_a == self.t_start else t_a
        hi_search = self.t_end + hi_atol if t_b == self.t_end else t_b
        abs_ts = self.abs_timestamps
        lo = int(np.searchsorted(abs_ts, lo_search, side="left"))
        hi = int(np.searchsorted(abs_ts, hi_search, side="left"))  # half-open at right
        new_abs_ts = abs_ts[lo:hi]
        new_ts = new_abs_ts - t_a  # relative to new t_start
        new_vals = None if self.values is None else self.values[lo:hi]
        return EventSeries._from_relative(
            timestamps=new_ts,
            values=new_vals,
            t_start=float(t_a),
            t_end=float(t_b),
        )

    # ------------------------------------------------------------------ shift
    def shift(self, t_delta: float) -> "EventSeries":
        if t_delta == 0.0:
            return self
        from dataclasses import replace
        return replace(
            self,
            t_start=self.t_start + t_delta,
            t_end=self.t_end + t_delta,
        )

    # ------------------------------------------------------------------ concat
    def concat(
        self, other: "EventSeries",
        atol: float = DEFAULT_ATOL, rtol: float = DEFAULT_RTOL,
    ) -> "EventSeries":
        if not isinstance(other, EventSeries):
            raise IncompatibleSeriesError(f"cannot concat EventSeries with {type(other).__name__}")
        if (self.values is None) != (other.values is None):
            raise IncompatibleSeriesError("one has values, the other does not")
        if self.values is not None and self.values.shape[1:] != other.values.shape[1:]:  # type: ignore[union-attr]
            raise IncompatibleSeriesError(
                f"value shapes differ: {self.values.shape[1:]} vs {other.values.shape[1:]}"  # type: ignore[union-attr]
            )
        # Auto-shift other so its timeline aligns with self.t_end.
        delta = self.t_end - other.t_start
        other = other.shift(delta)
        # Build absolute timeline, then re-normalise relative to self.t_start.
        self_abs = self.abs_timestamps
        other_abs = other.abs_timestamps
        new_abs_ts = np.concatenate([self_abs, other_abs])
        new_t_start = self.t_start
        new_t_end = other.t_end
        new_ts = new_abs_ts - new_t_start
        new_vals: np.ndarray | None
        if self.values is None:
            new_vals = None
        else:
            new_vals = np.concatenate([self.values, other.values])  # type: ignore[arg-type]
        return EventSeries._from_relative(
            timestamps=new_ts,
            values=new_vals,
            t_start=new_t_start,
            t_end=new_t_end,
        )

    def equal(self, other: TimeSeries) -> bool:
        if not isinstance(other, EventSeries):
            return False
        if not (
            tclose(self.t_start, other.t_start, atol=t_atol_at(self.t_start))
            and tclose(self.t_end, other.t_end, atol=t_atol_at(self.t_end))
        ):
            return False
        if not np.allclose(
            self.abs_timestamps, other.abs_timestamps, atol=DEFAULT_ATOL, rtol=1e-12
        ):
            return False
        if (self.values is None) != (other.values is None):
            return False
        if self.values is None:
            return True
        return np.array_equal(self.values, other.values)
