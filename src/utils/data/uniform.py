"""Uniform (regular sample-rate) time series — audio, video frames, etc.

Storage model
-------------
We store the raw sample array plus the time of the *left edge of sample 0*
(`t_first_edge`) and a declared half-open interval `[t_start, t_end)`. The
declared endpoints may sit anywhere within the first and last sample cells:

    t_first_edge          t_first_edge + N/sr
    │                                    │
    ▼                                    ▼
    ┌───┬───┬───┬───┬───┬───┬───┬───┬───┐
    │ 0 │ 1 │ 2 │ 3 │ 4 │ 5 │ 6 │ 7 │ 8 │   samples (each is one cell wide)
    └───┴───┴───┴───┴───┴───┴───┴───┴───┘
        ▲                        ▲
        t_start                  t_end           (declared interval)

Invariants
~~~~~~~~~~
* `t_first_edge <= t_start <= t_first_edge + 1/sr`
* `t_first_edge + (N-1)/sr <= t_end <= t_first_edge + N/sr`
* `t_start <= t_end`

Slicing at a sub-sample `t_cut`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Let `k = (t_cut - t_first_edge) * sr`. The left half takes samples
`0 .. ceil(k)` (i.e. every sample whose cell overlaps `[t_start, t_cut)`);
the right half takes samples `floor(k) .. N` (every cell overlapping
`[t_cut, t_end)`). When `k` is not integer the sample at index `floor(k)`
appears in *both* halves — that is what makes concat lossless.

On concat we detect this overlap via the grid offset and drop the duplicate.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from ._floats import DEFAULT_ATOL, DEFAULT_RTOL, grid_atol, t_atol_at, tclose
from .base import DomainError, IncompatibleSeriesError, TimeSeries


@dataclass(frozen=True, eq=False)
class UniformSeries(TimeSeries):
    """A regular-rate sequence of samples along axis 0.

    Parameters
    ----------
    samples : np.ndarray
        Shape `(N, ...)`. Axis 0 is time. Trailing axes are channels / features.
    sr : float
        Sample rate in Hz.
    t_first_edge : float
        Time of the left edge of `samples[0]`, in seconds.
    t_start, t_end : float
        Declared half-open interval. See module docstring for invariants.
    """

    samples: np.ndarray = field(repr=False)
    sr: float
    t_first_edge: float
    t_start: float
    t_end: float

    # ------------------------------------------------------------------ ctors
    def __post_init__(self) -> None:
        if self.samples.ndim < 1:
            raise ValueError("samples must have at least one axis")
        if self.sr <= 0:
            raise ValueError("sr must be > 0")
        if self.t_end < self.t_start:
            raise ValueError(f"t_end ({self.t_end}) < t_start ({self.t_start})")
        N = self.samples.shape[0]
        atol = grid_atol(self.sr, self.t_first_edge)
        # The declared interval must lie inside the sample-grid span.
        right_edge = self.t_first_edge + N / self.sr
        if self.t_start < self.t_first_edge - atol:
            raise ValueError(
                f"t_start={self.t_start} precedes t_first_edge={self.t_first_edge}"
            )
        if self.t_end > right_edge + atol:
            raise ValueError(
                f"t_end={self.t_end} exceeds last-sample right edge={right_edge}"
            )
        # And the declared endpoints must each lie within at most one sample cell
        # of the corresponding edge — i.e. we never carry "extra" leading/trailing
        # samples beyond what slicing could have produced.
        if N == 0:
            return
        if self.t_start > self.t_first_edge + 1 / self.sr + atol:
            raise ValueError("t_start lies beyond the first sample cell")
        if self.t_end < right_edge - 1 / self.sr - atol:
            raise ValueError("t_end lies before the last sample cell")

    @classmethod
    def from_samples(
        cls, samples: np.ndarray, sr: float, t_start: float = 0.0
    ) -> "UniformSeries":
        """Build a series whose declared interval matches its sample grid exactly."""
        samples = np.asarray(samples)
        N = samples.shape[0]
        return cls(
            samples=samples,
            sr=float(sr),
            t_first_edge=float(t_start),
            t_start=float(t_start),
            t_end=float(t_start) + N / float(sr),
        )

    # ------------------------------------------------------------------ shape
    @property
    def n_samples(self) -> int:
        return int(self.samples.shape[0])

    @property
    def channel_shape(self) -> tuple[int, ...]:
        return tuple(self.samples.shape[1:])

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, i: Any) -> Any:
        return self.samples[i]

    # ------------------------------------------------------------------ slice
    def slice(self, t_a: float, t_b: float) -> "UniformSeries":
        atol = grid_atol(self.sr, self.t_first_edge)
        if t_a < self.t_start - atol or t_b > self.t_end + atol or t_a > t_b + atol:
            raise DomainError(
                f"slice({t_a}, {t_b}) outside [{self.t_start}, {self.t_end}]"
            )
        # Clamp into the declared range to absorb fp noise.
        t_a = max(t_a, self.t_start)
        t_b = min(t_b, self.t_end)

        sr = self.sr
        sample_atol = atol * sr  # tolerance expressed in sample units
        # Index of sample whose *left edge* is at or before t_a; first underlying
        # sample whose cell overlaps `[t_a, ...)`.
        ka = math.floor((t_a - self.t_first_edge) * sr + sample_atol)
        # Index one past the last sample whose cell overlaps `[..., t_b)`.
        # = smallest k such that (k/sr + t_first_edge) >= t_b, i.e. ceil((t_b - t_first_edge)*sr).
        kb = math.ceil((t_b - self.t_first_edge) * sr - sample_atol)
        ka = max(0, ka)
        kb = min(self.n_samples, kb)
        if kb < ka:
            kb = ka  # empty slice

        new_samples = self.samples[ka:kb]
        new_first_edge = self.t_first_edge + ka / sr
        return UniformSeries(
            samples=new_samples,
            sr=sr,
            t_first_edge=new_first_edge,
            t_start=float(t_a),
            t_end=float(t_b),
        )

    # ------------------------------------------------------------------ concat
    def concat(
        self, other: "UniformSeries",
        atol: float = DEFAULT_ATOL, rtol: float = DEFAULT_RTOL,
    ) -> "UniformSeries":
        if not isinstance(other, UniformSeries):
            raise IncompatibleSeriesError(f"cannot concat UniformSeries with {type(other).__name__}")
        if not tclose(self.sr, other.sr, atol=0.0, rtol=1e-12):
            raise IncompatibleSeriesError(f"sample rates differ: {self.sr} vs {other.sr}")
        if not tclose(self.t_end, other.t_start, atol=atol, rtol=rtol):
            raise IncompatibleSeriesError(
                f"seam mismatch: self.t_end={self.t_end} other.t_start={other.t_start}"
            )
        if self.samples.shape[1:] != other.samples.shape[1:]:
            raise IncompatibleSeriesError(
                f"channel shapes differ: {self.samples.shape[1:]} vs {other.samples.shape[1:]}"
            )

        sr = self.sr
        # Compute grid offset in *sample-index space* to avoid catastrophic
        # cancellation when t_first_edge is at Unix-timestamp magnitudes.
        # The subtraction `(other.t_first_edge - self.t_first_edge)` is bounded
        # by ulp(t_first_edge); multiplying by sr keeps the error well below 1.
        k_offset_float = (other.t_first_edge - self.t_first_edge) * sr
        k_int = round(k_offset_float)
        phase_err = abs(k_offset_float - k_int)
        if phase_err > 0.1:
            raise IncompatibleSeriesError(
                f"incompatible sample grids: phase offset {k_offset_float} samples "
                f"(must be integer; nearest is {k_int})"
            )
        n_self = self.n_samples
        if k_int == n_self:
            # Disjoint sample grids meet cleanly; just stack.
            new_samples = np.concatenate([self.samples, other.samples], axis=0)
        elif k_int == n_self - 1:
            # Grids overlap by exactly one sample (= a sub-sample slice happened).
            # The shared sample is `self.samples[-1]` == `other.samples[0]`. Drop one.
            new_samples = np.concatenate([self.samples, other.samples[1:]], axis=0)
        else:
            raise IncompatibleSeriesError(
                f"incompatible sample grids: integer offset {k_int} "
                f"(expected {n_self} or {n_self - 1})"
            )

        return UniformSeries(
            samples=new_samples,
            sr=sr,
            t_first_edge=self.t_first_edge,
            t_start=self.t_start,
            t_end=other.t_end,
        )

    # ------------------------------------------------------------------ misc
    def equal(self, other: TimeSeries) -> bool:
        if not isinstance(other, UniformSeries):
            return False
        atol = grid_atol(self.sr, self.t_first_edge)
        if not (
            tclose(self.sr, other.sr, atol=0.0, rtol=1e-12)
            and tclose(self.t_start, other.t_start, atol=t_atol_at(self.t_start))
            and tclose(self.t_end, other.t_end, atol=t_atol_at(self.t_end))
            and tclose(self.t_first_edge, other.t_first_edge, atol=atol)
        ):
            return False
        return np.array_equal(self.samples, other.samples)

    def sample_times(self) -> np.ndarray:
        """Left-edge time of each underlying sample."""
        return self.t_first_edge + np.arange(self.n_samples) / self.sr

    def time_to_index(self, t: float) -> int:
        """Index of the sample cell containing time `t` (`samples[i]` covers
        `[t_first_edge + i/sr, t_first_edge + (i+1)/sr)`).
        """
        return int(math.floor((t - self.t_first_edge) * self.sr))
