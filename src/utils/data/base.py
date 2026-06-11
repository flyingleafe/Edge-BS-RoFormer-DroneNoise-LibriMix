"""Abstract `TimeSeries` — the common algebra.

Every track in a `TimeFrame` is a `TimeSeries`: a value with a declared
half-open time domain `[t_start, t_end)` (in seconds), supporting

* `slice(t_a, t_b)` — restrict the declared domain (accepts float seconds or
  int64 ticks)
* `concat(other)`   — glue along time at a shared seam (auto-shifts `other`)
* `shift(t_delta)`  — change all time anchors by `t_delta` (O(1); accepts
  float seconds or int64 ticks)

The single invariant we promise (and test) is:

    self.slice(a, b).concat(self.slice(b, c)) == self.slice(a, c)

for any `t_start <= a <= b <= c <= t_end`. Equality is defined by each
concrete subclass via `equal()` (exact on ticks); the default `__eq__`
delegates there.

Time is stored as **int64 tick counts** at a fixed `TICKS_PER_SECOND`
(nanoseconds).  Public accessors (`.t_start`, `.t_end`, `.duration`) return
float seconds; `*_ticks` accessors return exact int64 values for exact
round-trips.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class TimeSeries(ABC):
    """A value with a declared half-open time interval `[t_start, t_end)`.

    Subclasses implement these as properties (float seconds / int ticks).
    """

    # -- domain (seconds; float) -----------------------------------------
    @property
    @abstractmethod
    def t_start(self) -> float: ...
    @property
    @abstractmethod
    def t_end(self) -> float: ...
    @property
    def duration(self) -> float:
        return float(self.t_end - self.t_start)

    # -- NOTE: `*_ticks` and `values` are NOT declared as abstract
    #    properties here — they are stored as dataclass fields / computed
    #    properties in subclasses, and a base-class @property would be
    #    misinterpreted as a field default by @dataclass, breaking field
    #    ordering.
    @abstractmethod
    def __len__(self) -> int:
        """Number of items (samples / events / segments)."""

    # -- core algebra (subclass) ------------------------------------------
    @abstractmethod
    def slice(self, t_a: float | int, t_b: float | int) -> TimeSeries:
        """Return the restriction to `[t_a, t_b)`.

        Requires `self.t_start <= t_a <= t_b <= self.t_end`.
        `slice(t_start, t_end)` must return an equal series.
        Arguments accept float seconds or int64 ticks.
        """

    @abstractmethod
    def concat(self, other: TimeSeries) -> TimeSeries:
        """Glue along time. `other` is automatically shifted so that its
        `t_start` aligns with `self.t_end`; no exact seam match is required.

        Subclasses must also reject incompatible parameters (sample rate,
        value dtype/shape, etc.).
        """

    @abstractmethod
    def shift(self, t_delta: float | int) -> TimeSeries:
        """Return a new series whose entire timeline is moved by `t_delta`.

        O(1).  Accepts float seconds or int64 ticks.
        """

    @abstractmethod
    def interpolate(
        self,
        times,
        *,
        kind: str = "linear",
        fill: str = "clamp",
    ) -> np.ndarray:
        """Evaluate the signal value(s) at absolute query times.

        Parameters
        ----------
        times : np.ndarray | list[float] | list[int]
            Query times — float seconds or int64 ticks.
        kind : str
            Interpolation kind.  ``"linear"`` only for now (matches legacy
            ``np.interp`` canon).
        fill : str
            Extrapolation policy for query times outside the data span:
            ``"clamp"`` (default) — hold endpoint values;
            ``"nan"`` — NaN outside;
            ``"error"`` — raise ``DomainError``.

        Returns
        -------
        values : np.ndarray
            Shape ``(len(times), *value_shape)``.
        """
        ...

    @abstractmethod
    def equal(self, other: TimeSeries) -> bool:
        """Structural equality — exact (no tolerance)."""

    # -- operators / dunders ----------------------------------------------
    def __add__(self, other: TimeSeries) -> TimeSeries:
        return self.concat(other)

    def __eq__(self, other: object) -> bool:  # type: ignore[override]
        if not isinstance(other, TimeSeries):
            return NotImplemented
        if type(self) is not type(other):
            return False
        return self.equal(other)

    def __hash__(self) -> int:  # frozen dataclasses provide hashes; subclasses override
        return id(self)


class IncompatibleSeriesError(ValueError):
    """Raised when two series cannot be concatenated or merged."""


class DomainError(ValueError):
    """Raised when a slice request is outside the declared domain."""
