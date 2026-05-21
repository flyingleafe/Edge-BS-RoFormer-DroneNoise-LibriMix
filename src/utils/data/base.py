"""Abstract `TimeSeries` — the common algebra.

Every track in a `TimeFrame` is a `TimeSeries`: a value with a declared
half-open time domain `[t_start, t_end)` (in seconds), supporting

* `slice(t_a, t_b)` — restrict the declared domain
* `concat(other)`   — glue along time at a shared seam

The single invariant we promise (and test) is:

    self.slice(a, b).concat(self.slice(b, c)) == self.slice(a, c)

for any `t_start <= a <= b <= c <= t_end`. Equality is defined by each
concrete subclass via `equal()`; the default `__eq__` delegates there.
"""
from __future__ import annotations

from abc import ABC, abstractmethod


class TimeSeries(ABC):
    """A value with a declared half-open time interval `[t_start, t_end)`."""

    # Required by every subclass (frozen dataclass field).
    t_start: float
    t_end: float

    # -- domain -----------------------------------------------------------
    @property
    def duration(self) -> float:
        return float(self.t_end - self.t_start)

    def contains_time(self, t: float) -> bool:
        return self.t_start <= t < self.t_end

    # -- core algebra (subclass) ------------------------------------------
    @abstractmethod
    def slice(self, t_a: float, t_b: float) -> "TimeSeries":
        """Return the restriction to `[t_a, t_b)`.

        Requires `self.t_start <= t_a <= t_b <= self.t_end`.
        `slice(t_start, t_end)` must return an equal series.
        """

    @abstractmethod
    def concat(self, other: "TimeSeries") -> "TimeSeries":
        """Glue along time. Requires `self.t_end ≈ other.t_start`.

        Subclasses must also reject incompatible parameters (sample rate,
        value dtype/shape, etc.).
        """

    @abstractmethod
    def equal(self, other: "TimeSeries") -> bool:
        """Structural equality, modulo float tolerance on time fields."""

    # -- operators / dunders ----------------------------------------------
    def __add__(self, other: "TimeSeries") -> "TimeSeries":
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
