"""`TimeFrame` — column-keyed container of time series.

A `TimeFrame` holds a dict of named `TimeSeries` tracks whose `t_start` values
are stored **relative to the frame's `t_start`**.  The frame's own `t_start` is
the single absolute anchor (int64 ticks).

Construction accepts absolute tracks and re-bases them; public accessors
(`tf[key]`, `items()`, `values()`) hand back tracks re-anchored to absolute
time, so an extracted track is a self-contained absolute series.  Because every
series shifts in O(1) (relative storage), this re-basing is cheap.

Tags
~~~~
Each frame carries an optional ``tags`` mapping for **time-invariant** sample
metadata (``id``, ``input_snr``, ``recording_id``).  Tags are:
* preserved under ``slice``, ``shift``, ``select``, ``with_track``;
* checked for equality on shared keys on ``concat`` (raise
  ``IncompatibleSeriesError`` on conflict), union otherwise.

Operations
~~~~~~~~~~
* Column-wise:  `tf[key]`, `tf.drop(...)`, `tf.select(...)`, `tf.merge(other)`.
* Time-wise:    `tf.slice(t_a, t_b)`, `tf.concat(other)`, `tf + other`,
  `tf.shift(t_delta)`.

Invariants
~~~~~~~~~~
* `tf.slice(a, b).concat(tf.slice(b, c)) == tf.slice(a, c)`.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Hashable, Iterable, Iterator

import numpy as np

from ._ticks import _c_to_ticks, ticks_to_secs
from .base import DomainError, IncompatibleSeriesError, TimeSeries


@dataclass(frozen=True, eq=False)
class TimeFrame:
    """Dict-like container of `TimeSeries` tracks with heterogeneous domains.

    The ``tracks`` passed to the constructor are interpreted as **absolute**
    series and stored re-based relative to the frame's ``t_start_ticks``.
    Read them back (already re-anchored to absolute time) via ``tf[key]``,
    ``items()``, ``values()``.

    ``tags`` is a time-invariant mapping of per-sample metadata (``id``,
    ``input_snr``, ``recording_id``) — preserved by all time and column ops.
    """

    tracks: dict[str, TimeSeries] = field(default_factory=dict)
    t_start_ticks: int = 0
    dur_ticks: int = 0
    tags: Mapping[str, Hashable] = field(default_factory=dict)
    global_data: Mapping[str, Any] = field(default_factory=dict)

    # ---- construction ---------------------------------------------------
    def __post_init__(self) -> None:
        # Normalize tags to a plain dict (never None).
        if self.tags is None:
            object.__setattr__(self, "tags", {})
        elif not isinstance(self.tags, dict):
            object.__setattr__(self, "tags", dict(self.tags))
        # Normalize global_data to a plain dict (never None).
        if self.global_data is None:
            object.__setattr__(self, "global_data", {})
        elif not isinstance(self.global_data, dict):
            object.__setattr__(self, "global_data", dict(self.global_data))
        if self.dur_ticks < 0:
            raise ValueError(f"dur_ticks ({self.dur_ticks}) < 0")
        for name, track in self.tracks.items():
            if not isinstance(track, TimeSeries):
                raise TypeError(f"track {name!r} is not a TimeSeries")
        if self.tracks:
            # Validate hull against the incoming (absolute) tracks.
            hull_start = min(tr.t_start_ticks for tr in self.tracks.values())
            hull_end = max(tr.t_end_ticks for tr in self.tracks.values())
            if self.t_start_ticks > hull_start:
                raise ValueError(
                    f"frame t_start_ticks ({self.t_start_ticks}) after hull start ({hull_start})"
                )
            if self.t_start_ticks + self.dur_ticks < hull_end:
                raise ValueError(
                    f"frame t_end_ticks ({self.t_start_ticks + self.dur_ticks}) before hull end ({hull_end})"
                )
        # Re-base tracks to frame-relative.
        if self.t_start_ticks != 0:
            object.__setattr__(
                self,
                "tracks",
                {
                    name: tr.shift(-self.t_start_ticks)
                    for name, tr in self.tracks.items()
                },
            )

    @classmethod
    def _from_local(
        cls,
        local_tracks: dict[str, TimeSeries],
        t_start_ticks: int,
        dur_ticks: int,
        tags: Mapping[str, Hashable] | None = None,
        global_data: Mapping[str, Any] | None = None,
    ) -> "TimeFrame":
        """Build a frame from tracks already stored frame-local.  Bypasses
        re-basing and validation — internal use.
        """
        self = object.__new__(cls)
        object.__setattr__(self, "tracks", local_tracks)
        object.__setattr__(self, "t_start_ticks", int(t_start_ticks))
        object.__setattr__(self, "dur_ticks", int(dur_ticks))
        object.__setattr__(self, "tags", dict(tags or {}))
        object.__setattr__(self, "global_data", dict(global_data or {}))
        return self

    def _abs(self) -> dict[str, TimeSeries]:
        """The tracks re-anchored to absolute time (O(1) per track)."""
        if self.t_start_ticks == 0:
            return dict(self.tracks)
        return {name: tr.shift(self.t_start_ticks) for name, tr in self.tracks.items()}

    @classmethod
    def from_tracks(
        cls,
        tracks: dict[str, TimeSeries],
        *,
        t_start: float | int | None = None,
        t_end: float | int | None = None,
        tags: Mapping[str, Hashable] | None = None,
        global_data: Mapping[str, Any] | None = None,
    ) -> "TimeFrame":
        """Build a frame, inferring the hull domain from the tracks if not given."""
        if t_start is None or t_end is None:
            if not tracks:
                raise ValueError("cannot infer domain from empty tracks dict")
            hull_start = min(tr.t_start_ticks for tr in tracks.values())
            hull_end = max(tr.t_end_ticks for tr in tracks.values())
            t_start = hull_start if t_start is None else t_start
            t_end = hull_end if t_end is None else t_end
        t0 = _c_to_ticks(t_start) if not isinstance(t_start, int) else t_start
        te = _c_to_ticks(t_end) if not isinstance(t_end, int) else t_end
        return cls(tracks=dict(tracks), t_start_ticks=t0, dur_ticks=te - t0,
                   tags=tags if tags is not None else {},
                   global_data=global_data if global_data is not None else {})

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
    def duration_ticks(self) -> int:
        return self.dur_ticks

    @property
    def duration(self) -> float:
        return ticks_to_secs(self.dur_ticks)

    # ---- dict-like ------------------------------------------------------
    def __getitem__(self, key: str) -> TimeSeries:
        tr = self.tracks[key]
        return tr if self.t_start_ticks == 0 else tr.shift(self.t_start_ticks)

    def __contains__(self, key: object) -> bool:
        return key in self.tracks

    def __iter__(self) -> Iterator[str]:
        return iter(self.tracks)

    def __len__(self) -> int:
        return len(self.tracks)

    def keys(self) -> Iterable[str]:
        return self.tracks.keys()

    def values(self) -> Iterable[TimeSeries]:
        return self._abs().values()

    def items(self) -> Iterable[tuple[str, TimeSeries]]:
        return self._abs().items()

    # ---- column ops -----------------------------------------------------
    def select(self, keys: Iterable[str]) -> "TimeFrame":
        keys = list(keys)
        missing = [k for k in keys if k not in self.tracks]
        if missing:
            raise KeyError(f"missing tracks: {missing}")
        return TimeFrame._from_local(
            {k: self.tracks[k] for k in keys}, self.t_start_ticks, self.dur_ticks,
            tags=self.tags, global_data=self.global_data,
        )

    def drop(self, keys: Iterable[str]) -> "TimeFrame":
        keys = set(keys)
        return TimeFrame._from_local(
            {k: v for k, v in self.tracks.items() if k not in keys},
            self.t_start_ticks, self.dur_ticks, tags=self.tags,
            global_data=self.global_data,
        )

    def with_track(self, name: str, track: TimeSeries) -> "TimeFrame":
        """Return a new frame with ``name`` mapped to ``track``, expanding
        the hull domain as needed.
        """
        new_t_start = min(self.t_start_ticks, track.t_start_ticks)
        new_t_end = max(self.t_start_ticks + self.dur_ticks, track.t_end_ticks)
        return TimeFrame(
            tracks={**self._abs(), name: track},
            t_start_ticks=new_t_start,
            dur_ticks=new_t_end - new_t_start,
            tags=self.tags,
            global_data=self.global_data,
        )

    def merge(self, other: "TimeFrame", overwrite: bool = False) -> "TimeFrame":
        """Column-wise union of two frames.  Domains may differ; result hull
        is used.
        """
        if not overwrite:
            collisions = set(self.tracks) & set(other.tracks)
            if collisions:
                raise ValueError(
                    f"key collisions: {sorted(collisions)} (pass overwrite=True)"
                )
        new_tracks = {**self._abs(), **other._abs()}
        new_t_start = min(self.t_start_ticks, other.t_start_ticks)
        new_t_end = max(
            self.t_start_ticks + self.dur_ticks,
            other.t_start_ticks + other.dur_ticks,
        )
        # Merge tags (equality check on shared keys).
        new_tags = dict(self.tags)
        for k, v in other.tags.items():
            if k in new_tags:
                if new_tags[k] != v:
                    raise IncompatibleSeriesError(
                        f"tag {k!r} conflict on merge: {new_tags[k]!r} != {v!r}"
                    )
            else:
                new_tags[k] = v
        # Merge global_data (equality check on shared keys).
        new_global = dict(self.global_data)
        for k, v in other.global_data.items():
            if k in new_global:
                if not _global_leaf_equal(new_global[k], v):
                    raise IncompatibleSeriesError(
                        f"global_data {k!r} conflict on merge"
                    )
            else:
                new_global[k] = v
        return TimeFrame(
            tracks=new_tracks, t_start_ticks=new_t_start,
            dur_ticks=new_t_end - new_t_start, tags=new_tags,
            global_data=new_global,
        )

    # ---- time ops -------------------------------------------------------
    def shift(self, t_delta: float | int) -> "TimeFrame":
        dt = _c_to_ticks(t_delta) if not isinstance(t_delta, int) else t_delta
        if dt == 0:
            return self
        return TimeFrame._from_local(
            dict(self.tracks), self.t_start_ticks + dt, self.dur_ticks,
            tags=self.tags, global_data=self.global_data,
        )

    def slice(self, t_a: float | int, t_b: float | int) -> "TimeFrame":
        ta_tick = _c_to_ticks(t_a) if not isinstance(t_a, int) else t_a
        tb_tick = _c_to_ticks(t_b) if not isinstance(t_b, int) else t_b
        t0 = self.t_start_ticks
        t1 = t0 + self.dur_ticks
        if ta_tick < t0 or tb_tick > t1 or ta_tick > tb_tick:
            raise DomainError(
                f"slice({ta_tick}, {tb_tick}) outside [{t0}, {t1}] (ticks)"
            )
        ta_tick = max(ta_tick, t0)
        tb_tick = min(tb_tick, t1)
        abs_tracks = self._abs()
        new_tracks: dict[str, TimeSeries] = {}
        for name, tr in abs_tracks.items():
            a_eff = max(ta_tick, tr.t_start_ticks)
            b_eff = min(tb_tick, tr.t_end_ticks)
            if a_eff <= b_eff:
                new_tracks[name] = tr.slice(a_eff, b_eff)
        return TimeFrame(
            tracks=new_tracks, t_start_ticks=ta_tick,
            dur_ticks=tb_tick - ta_tick, tags=self.tags,
            global_data=self.global_data,
        )

    def concat(self, other: "TimeFrame") -> "TimeFrame":
        """Glue along time.  Track sets may differ; missing tracks are
        carried over.  ``other`` is aligned so its domain starts at
        ``self.t_end``.

        Tags must agree on shared keys; union otherwise.
        """
        delta = self.dur_ticks
        self_abs = self._abs()
        other_abs = other._abs()
        union_keys = set(self_abs) | set(other_abs)
        new_tracks: dict[str, TimeSeries] = {}
        for name in union_keys:
            if name in self_abs and name in other_abs:
                new_tracks[name] = self_abs[name].concat(
                    other_abs[name].shift(delta - other.t_start_ticks)
                )
            elif name in self_abs:
                new_tracks[name] = self_abs[name]
            else:
                new_tracks[name] = other_abs[name].shift(delta - other.t_start_ticks)
        # Merge tags: equality on shared keys, union on disjoint ones.
        new_tags = dict(self.tags)
        for k, v in other.tags.items():
            if k in new_tags:
                if new_tags[k] != v:
                    raise IncompatibleSeriesError(
                        f"tag {k!r} conflict on concat: {new_tags[k]!r} != {v!r}"
                    )
            else:
                new_tags[k] = v
        # Merge global_data: equality on shared keys, union on disjoint ones.
        new_global = dict(self.global_data)
        for k, v in other.global_data.items():
            if k in new_global:
                if not _global_leaf_equal(new_global[k], v):
                    raise IncompatibleSeriesError(
                        f"global_data {k!r} conflict on concat"
                    )
            else:
                new_global[k] = v
        return TimeFrame(
            tracks=new_tracks, t_start_ticks=self.t_start_ticks,
            dur_ticks=self.dur_ticks + other.dur_ticks,
            tags=new_tags,
            global_data=new_global,
        )

    def __add__(self, other: "TimeFrame") -> "TimeFrame":
        return self.concat(other)

    # ---- misc -----------------------------------------------------------
    def equal(self, other: "TimeFrame") -> bool:
        if not (
            self.t_start_ticks == other.t_start_ticks
            and self.dur_ticks == other.dur_ticks
            and set(self.tracks) == set(other.tracks)
        ):
            return False
        a = self._abs()
        b = other._abs()
        for k in a:
            if not a[k].equal(b[k]):
                return False
        if dict(self.tags) != dict(other.tags):
            return False
        if not _global_equal(self.global_data, other.global_data):
            return False
        return True

    def __eq__(self, other: object) -> bool:  # type: ignore[override]
        if not isinstance(other, TimeFrame):
            return NotImplemented
        return self.equal(other)

    def __hash__(self) -> int:
        return id(self)


# ---- global_data helpers ----------------------------------------------------


def _global_leaf_equal(a: Any, b: Any) -> bool:
    """Compare two global_data leaves, handling numpy arrays."""
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        return a.shape == b.shape and bool(np.all(a == b))
    return a == b


def _global_equal(a: dict[str, Any], b: dict[str, Any]) -> bool:
    """Compare two global_data dicts."""
    if a.keys() != b.keys():
        return False
    for k in a:
        if not _global_leaf_equal(a[k], b[k]):
            return False
    return True
