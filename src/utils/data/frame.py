"""`TimeFrame` — column-keyed container of time series.

A `TimeFrame` holds a dict of named `TimeSeries` tracks. Tracks may have
independent time domains; the frame's own declared domain is the **hull**
(min `t_start`, max `t_end`) of its contents, or an explicitly provided
superset.

Operations
~~~~~~~~~~
* Column-wise (DataFrame side):  `tf[key]`, `tf.drop(...)`, `tf.select(...)`,
  `tf.merge(other)`.
* Time-wise (array side):  `tf.slice(t_a, t_b)`, `tf.concat(other)`,
  `tf + other`, `tf.shift(t_delta)`.

Invariants
~~~~~~~~~~
* `tf.slice(a, b).concat(tf.slice(b, c)) == tf.slice(a, c)`.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Iterable, Iterator

from ._floats import DEFAULT_ATOL, DEFAULT_RTOL, t_atol_at, tclose
from .base import DomainError, IncompatibleSeriesError, TimeSeries


@dataclass(frozen=True, eq=False)
class TimeFrame:
    """Dict-like container of `TimeSeries` tracks with heterogeneous domains."""

    tracks: dict[str, TimeSeries] = field(default_factory=dict)
    t_start: float = 0.0
    t_end: float = 0.0

    def __post_init__(self) -> None:
        if self.t_end < self.t_start:
            raise ValueError(f"t_end ({self.t_end}) < t_start ({self.t_start})")
        for name, track in self.tracks.items():
            if not isinstance(track, TimeSeries):
                raise TypeError(f"track {name!r} is not a TimeSeries")
        # Frame domain must cover the hull of all tracks.
        if self.tracks:
            hull_start = min(tr.t_start for tr in self.tracks.values())
            hull_end = max(tr.t_end for tr in self.tracks.values())
            if self.t_start > hull_start + t_atol_at(hull_start):
                raise ValueError(
                    f"frame t_start ({self.t_start}) is after hull start ({hull_start})"
                )
            if self.t_end < hull_end - t_atol_at(hull_end):
                raise ValueError(
                    f"frame t_end ({self.t_end}) is before hull end ({hull_end})"
                )

    @classmethod
    def from_tracks(
        cls,
        tracks: dict[str, TimeSeries],
        t_start: float | None = None,
        t_end: float | None = None,
    ) -> "TimeFrame":
        """Build a frame, inferring the hull domain from the tracks if not given."""
        if t_start is None or t_end is None:
            if not tracks:
                raise ValueError("cannot infer domain from empty tracks dict")
            hull_start = min(tr.t_start for tr in tracks.values())
            hull_end = max(tr.t_end for tr in tracks.values())
            t_start = hull_start if t_start is None else t_start
            t_end = hull_end if t_end is None else t_end
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
        """Return a new frame with `name` mapped to `track`, expanding the hull domain."""
        new_t_start = min(self.t_start, track.t_start)
        new_t_end = max(self.t_end, track.t_end)
        return replace(
            self,
            tracks={**self.tracks, name: track},
            t_start=new_t_start,
            t_end=new_t_end,
        )

    def merge(self, other: "TimeFrame", overwrite: bool = False) -> "TimeFrame":
        """Column-wise union of two frames. Domains may differ; result hull is used."""
        if not overwrite:
            collisions = set(self.tracks) & set(other.tracks)
            if collisions:
                raise ValueError(f"key collisions: {sorted(collisions)} (pass overwrite=True)")
        new_tracks = {**self.tracks, **other.tracks}
        new_t_start = min(self.t_start, other.t_start)
        new_t_end = max(self.t_end, other.t_end)
        return TimeFrame(tracks=new_tracks, t_start=new_t_start, t_end=new_t_end)

    # ------------------------------------------------------------------ time ops
    def shift(self, t_delta: float) -> "TimeFrame":
        if t_delta == 0.0:
            return self
        new_tracks = {name: tr.shift(t_delta) for name, tr in self.tracks.items()}
        return replace(
            self,
            tracks=new_tracks,
            t_start=self.t_start + t_delta,
            t_end=self.t_end + t_delta,
        )

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
        new_tracks: dict[str, TimeSeries] = {}
        for name, tr in self.tracks.items():
            a_eff = max(t_a, tr.t_start)
            b_eff = min(t_b, tr.t_end)
            seam_atol = t_atol_at(max(abs(a_eff), abs(b_eff)))
            # Include tracks that overlap or just touch the slice interval.
            if a_eff <= b_eff + seam_atol:
                new_tracks[name] = tr.slice(a_eff, b_eff)
        return TimeFrame(tracks=new_tracks, t_start=float(t_a), t_end=float(t_b))

    def concat(
        self, other: "TimeFrame",
        atol: float = DEFAULT_ATOL, rtol: float = DEFAULT_RTOL,
    ) -> "TimeFrame":
        """Glue along time. Track sets may differ; missing tracks are carried over.
        `other` is shifted so that its domain aligns with `self.t_end`.
        """
        delta = self.t_end - other.t_start
        union_keys = set(self.tracks) | set(other.tracks)
        new_tracks: dict[str, TimeSeries] = {}
        for name in union_keys:
            if name in self.tracks and name in other.tracks:
                new_tracks[name] = self.tracks[name].concat(other.tracks[name].shift(delta))
            elif name in self.tracks:
                new_tracks[name] = self.tracks[name]
            else:
                new_tracks[name] = other.tracks[name].shift(delta)
        new_t_end = self.t_end + other.duration
        return TimeFrame(tracks=new_tracks, t_start=self.t_start, t_end=new_t_end)

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
