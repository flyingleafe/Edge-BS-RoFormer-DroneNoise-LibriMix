"""Non-uniform / sparse event time series.

A `EventSeries` is a sorted list of point-timestamped events `(t_i, v_i)` lying
inside a declared half-open domain `[t_start, t_end)`. Values may be `None`
(timestamps only), a 1-D array, or any ndarray whose first axis is `M`.

Events at exactly `t_cut` go to the right half on slice (half-open convention).
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
        Sorted ascending. Each must satisfy `t_start <= ts < t_end`.
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
        ts = np.asarray(self.timestamps)
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
            if not np.all(np.diff(ts) >= 0):
                raise ValueError("timestamps must be sorted ascending")
            lo_atol = t_atol_at(self.t_start)
            hi_atol = t_atol_at(self.t_end)
            if ts[0] < self.t_start - lo_atol or ts[-1] >= self.t_end + hi_atol:
                raise ValueError(
                    f"events outside domain [{self.t_start}, {self.t_end}): "
                    f"min={ts[0]}, max={ts[-1]}"
                )

    @classmethod
    def from_events(
        cls,
        timestamps: np.ndarray,
        values: np.ndarray | None = None,
        t_start: float | None = None,
        t_end: float | None = None,
    ) -> "EventSeries":
        ts = np.asarray(timestamps)
        if t_start is None:
            t_start = float(ts[0]) if ts.size else 0.0
        if t_end is None:
            # Half-open: choose an end strictly greater than the last event.
            t_end = float(ts[-1]) + 1e-9 if ts.size else float(t_start)
        return cls(timestamps=ts, values=values, t_start=float(t_start), t_end=float(t_end))

    # ------------------------------------------------------------------ shape
    def __len__(self) -> int:
        return int(self.timestamps.shape[0])

    def __getitem__(self, i: Any):
        if self.values is None:
            return self.timestamps[i]
        return self.timestamps[i], self.values[i]

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
        # Exact float equality — not tclose — because anything else is an
        # interior cut that must remain strict half-open.
        lo_search = self.t_start - lo_atol if t_a == self.t_start else t_a
        hi_search = self.t_end + hi_atol if t_b == self.t_end else t_b
        ts = self.timestamps
        lo = int(np.searchsorted(ts, lo_search, side="left"))
        hi = int(np.searchsorted(ts, hi_search, side="left"))  # half-open at right
        new_ts = ts[lo:hi]
        new_vals = None if self.values is None else self.values[lo:hi]
        return EventSeries(
            timestamps=new_ts,
            values=new_vals,
            t_start=float(t_a),
            t_end=float(t_b),
        )

    # ------------------------------------------------------------------ concat
    def concat(
        self, other: "EventSeries",
        atol: float = DEFAULT_ATOL, rtol: float = DEFAULT_RTOL,
    ) -> "EventSeries":
        if not isinstance(other, EventSeries):
            raise IncompatibleSeriesError(f"cannot concat EventSeries with {type(other).__name__}")
        if not tclose(self.t_end, other.t_start, atol=atol, rtol=rtol):
            raise IncompatibleSeriesError(
                f"seam mismatch: self.t_end={self.t_end} other.t_start={other.t_start}"
            )
        if (self.values is None) != (other.values is None):
            raise IncompatibleSeriesError("one has values, the other does not")
        if self.values is not None and self.values.shape[1:] != other.values.shape[1:]:  # type: ignore[union-attr]
            raise IncompatibleSeriesError(
                f"value shapes differ: {self.values.shape[1:]} vs {other.values.shape[1:]}"  # type: ignore[union-attr]
            )
        new_ts = np.concatenate([self.timestamps, other.timestamps])
        new_vals: np.ndarray | None
        if self.values is None:
            new_vals = None
        else:
            new_vals = np.concatenate([self.values, other.values])  # type: ignore[arg-type]
        return EventSeries(
            timestamps=new_ts,
            values=new_vals,
            t_start=self.t_start,
            t_end=other.t_end,
        )

    def equal(self, other: TimeSeries) -> bool:
        if not isinstance(other, EventSeries):
            return False
        if not (
            tclose(self.t_start, other.t_start, atol=t_atol_at(self.t_start))
            and tclose(self.t_end, other.t_end, atol=t_atol_at(self.t_end))
        ):
            return False
        if not np.array_equal(self.timestamps, other.timestamps):
            return False
        if (self.values is None) != (other.values is None):
            return False
        if self.values is None:
            return True
        return np.array_equal(self.values, other.values)
