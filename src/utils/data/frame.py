"""`TimeFrame` — column-keyed container of aligned time series.

A `TimeFrame` holds a dict of named `TimeSeries` tracks all sharing the same
declared half-open interval `[t_start, t_end)`. It is dict-like across tracks
and time-sliceable as a whole.

Operations
~~~~~~~~~~
* Column-wise (DataFrame side):  `tf[key]`, `tf.drop(...)`, `tf.select(...)`,
  `tf.merge(other)`.
* Time-wise (array side):  `tf.slice(t_a, t_b)`, `tf.concat(other)`,
  `tf + other`.

Invariants
~~~~~~~~~~
* Every track satisfies `track.t_start == tf.t_start` and
  `track.t_end == tf.t_end` (within tolerance).
* `tf.slice(a, b).concat(tf.slice(b, c)) == tf.slice(a, c)`.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Iterable, Iterator

from ._floats import DEFAULT_ATOL, DEFAULT_RTOL, t_atol_at, tclose
from .base import DomainError, IncompatibleSeriesError, TimeSeries


@dataclass(frozen=True, eq=False)
class TimeFrame:
    """Dict-like container of aligned `TimeSeries`."""

    tracks: dict[str, TimeSeries] = field(default_factory=dict)
    t_start: float = 0.0
    t_end: float = 0.0

    def __post_init__(self) -> None:
        if self.t_end < self.t_start:
            raise ValueError(f"t_end ({self.t_end}) < t_start ({self.t_start})")
        lo_atol = t_atol_at(self.t_start)
        hi_atol = t_atol_at(self.t_end)
        for name, track in self.tracks.items():
            if not isinstance(track, TimeSeries):
                raise TypeError(f"track {name!r} is not a TimeSeries")
            if not (
                tclose(track.t_start, self.t_start, atol=lo_atol)
                and tclose(track.t_end, self.t_end, atol=hi_atol)
            ):
                raise ValueError(
                    f"track {name!r} domain [{track.t_start}, {track.t_end}) "
                    f"!= frame domain [{self.t_start}, {self.t_end})"
                )

    @classmethod
    def from_tracks(
        cls,
        tracks: dict[str, TimeSeries],
        t_start: float | None = None,
        t_end: float | None = None,
    ) -> "TimeFrame":
        """Build a frame, inferring the domain from the tracks if not given.

        When inferring, every track must agree on `(t_start, t_end)`.
        """
        if t_start is None or t_end is None:
            if not tracks:
                raise ValueError("cannot infer domain from empty tracks dict")
            ref = next(iter(tracks.values()))
            t_start = ref.t_start if t_start is None else t_start
            t_end = ref.t_end if t_end is None else t_end
        return cls(tracks=dict(tracks), t_start=float(t_start), t_end=float(t_end))

    # ------------------------------------------------------------------ dict-like
    def __getitem__(self, key: str) -> TimeSeries:
        return self.tracks[key]

    def __contains__(self, key: object) -> bool:
        return key in self.tracks

    def __iter__(self) -> Iterator[str]:
        return iter(self.tracks)

    def __len__(self) -> int:
        return len(self.tracks)

    def keys(self) -> Iterable[str]:
        return self.tracks.keys()

    def values(self) -> Iterable[TimeSeries]:
        return self.tracks.values()

    def items(self) -> Iterable[tuple[str, TimeSeries]]:
        return self.tracks.items()

    @property
    def duration(self) -> float:
        return float(self.t_end - self.t_start)

    # ------------------------------------------------------------------ column ops
    def select(self, keys: Iterable[str]) -> "TimeFrame":
        keys = list(keys)
        missing = [k for k in keys if k not in self.tracks]
        if missing:
            raise KeyError(f"missing tracks: {missing}")
        return replace(self, tracks={k: self.tracks[k] for k in keys})

    def drop(self, keys: Iterable[str]) -> "TimeFrame":
        keys = set(keys)
        return replace(self, tracks={k: v for k, v in self.tracks.items() if k not in keys})

    def with_track(self, name: str, track: TimeSeries) -> "TimeFrame":
        """Return a new frame with `name` mapped to `track` (must share domain)."""
        if not (
            tclose(track.t_start, self.t_start, atol=t_atol_at(self.t_start))
            and tclose(track.t_end, self.t_end, atol=t_atol_at(self.t_end))
        ):
            raise ValueError(
                f"track domain [{track.t_start}, {track.t_end}) != frame domain "
                f"[{self.t_start}, {self.t_end})"
            )
        return replace(self, tracks={**self.tracks, name: track})

    def merge(self, other: "TimeFrame", overwrite: bool = False) -> "TimeFrame":
        """Column-wise union of two frames sharing the same time domain."""
        if not (
            tclose(self.t_start, other.t_start, atol=t_atol_at(self.t_start))
            and tclose(self.t_end, other.t_end, atol=t_atol_at(self.t_end))
        ):
            raise IncompatibleSeriesError(
                f"domain mismatch: [{self.t_start}, {self.t_end}) "
                f"vs [{other.t_start}, {other.t_end})"
            )
        if not overwrite:
            collisions = set(self.tracks) & set(other.tracks)
            if collisions:
                raise ValueError(f"key collisions: {sorted(collisions)} (pass overwrite=True)")
        return replace(self, tracks={**self.tracks, **other.tracks})

    # ------------------------------------------------------------------ time ops
    def slice(self, t_a: float, t_b: float) -> "TimeFrame":
        lo_atol = t_atol_at(self.t_start)
        hi_atol = t_atol_at(self.t_end)
        ab_atol = t_atol_at(max(abs(t_a), abs(t_b)))
        if t_a < self.t_start - lo_atol or t_b > self.t_end + hi_atol or t_a > t_b + ab_atol:
            raise DomainError(
                f"slice({t_a}, {t_b}) outside [{self.t_start}, {self.t_end}]"
            )
        t_a = max(t_a, self.t_start)
        t_b = min(t_b, self.t_end)
        new_tracks = {name: tr.slice(t_a, t_b) for name, tr in self.tracks.items()}
        return TimeFrame(tracks=new_tracks, t_start=float(t_a), t_end=float(t_b))

    def concat(
        self, other: "TimeFrame",
        atol: float = DEFAULT_ATOL, rtol: float = DEFAULT_RTOL,
    ) -> "TimeFrame":
        if not tclose(self.t_end, other.t_start, atol=atol, rtol=rtol):
            raise IncompatibleSeriesError(
                f"seam mismatch: self.t_end={self.t_end} other.t_start={other.t_start}"
            )
        if set(self.tracks) != set(other.tracks):
            raise IncompatibleSeriesError(
                f"track sets differ: {sorted(set(self.tracks) ^ set(other.tracks))}"
            )
        new_tracks = {
            name: self.tracks[name].concat(other.tracks[name])
            for name in self.tracks
        }
        return TimeFrame(tracks=new_tracks, t_start=self.t_start, t_end=other.t_end)

    def __add__(self, other: "TimeFrame") -> "TimeFrame":
        return self.concat(other)

    # ------------------------------------------------------------------ misc
    def equal(self, other: "TimeFrame") -> bool:
        if not (
            tclose(self.t_start, other.t_start, atol=t_atol_at(self.t_start))
            and tclose(self.t_end, other.t_end, atol=t_atol_at(self.t_end))
        ):
            return False
        if set(self.tracks) != set(other.tracks):
            return False
        for k in self.tracks:
            if not self.tracks[k].equal(other.tracks[k]):
                return False
        return True

    def __eq__(self, other: object) -> bool:  # type: ignore[override]
        if not isinstance(other, TimeFrame):
            return NotImplemented
        return self.equal(other)

    def __hash__(self) -> int:
        return id(self)
