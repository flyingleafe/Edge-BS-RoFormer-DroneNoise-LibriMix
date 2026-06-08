"""Time-interval (segment) series.

A `SegmentSeries` stores segments `[t_begin, t_end)` with optional payloads.
Segments are sorted by `t_begin` and may overlap each other arbitrarily.

Storage model
-------------
Following the library-wide invariant, `starts` and `ends` are stored as int64
tick counts **relative to `t_start_ticks`**.  Absolute interval bounds are
available via `abs_starts` / `abs_ends` (float seconds) or `abs_starts_ticks` /
`abs_ends_ticks` (int64 ticks).  `__getitem__` returns absolute float seconds.

Slicing at a point `t_cut` *splits* any straddling segment into two halves
that share an identity tag (`ids`).  Concat at a matching seam re-merges
segments whose `ids` agree.

The identity tag is the algebraic marker that lets `slice(a,b) ⊕ slice(b,c)
== slice(a,c)` hold exactly.  Auto-generated ids are 62-bit random integers;
collision is negligible in practice but users may pass explicit ids.
"""
from __future__ import annotations

import secrets
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from ._ticks import TICKS_PER_SECOND, _c_to_ticks, secs_array_to_ticks, secs_to_ticks, ticks_array_to_secs, ticks_to_secs
from .base import DomainError, IncompatibleSeriesError, TimeSeries


def _new_ids(n: int) -> np.ndarray:
    """Return `n` random 62-bit positive int64 ids."""
    return np.array(
        [secrets.randbits(62) for _ in range(n)], dtype=np.int64
    )


@dataclass(frozen=True, eq=False)
class SegmentSeries(TimeSeries):
    """Sorted half-open interval series.

    Parameters
    ----------
    starts_ticks, ends_ticks : np.ndarray, shape (M,), dtype int64
        Interval bounds **relative to `t_start_ticks`** (int64 ticks).
        `starts_ticks[i] < ends_ticks[i]`, sorted by `starts_ticks`.
    values : np.ndarray | None
        Shape ``(…, M)`` payload (e.g. labels) — segment axis is ALWAYS LAST.  Optional.
    ids : np.ndarray, shape (M,), dtype int64
        Identity tags.  Segments with identical ids are treated as the same
        underlying segment when concatenating at a seam.
    t_start_ticks : int
        Absolute domain-start anchor (int64 ticks).
    dur_ticks : int
        Declared domain duration (int64 ticks).
    """

    starts_ticks: np.ndarray = field(repr=False)
    ends_ticks: np.ndarray = field(repr=False)
    values: np.ndarray | None = field(repr=False, default=None)
    ids: np.ndarray = field(repr=False, default=None)  # type: ignore[assignment]
    t_start_ticks: int = 0
    dur_ticks: int = 0

    def __post_init__(self) -> None:
        s = np.asarray(self.starts_ticks, dtype=np.int64)
        e = np.asarray(self.ends_ticks, dtype=np.int64)
        if s.shape != e.shape or s.ndim != 1:
            raise ValueError("starts and ends must be 1-D and same shape")
        if s.size and not np.all(e > s):
            raise ValueError("each segment must have end > start")
        if s.size and not np.all(np.diff(s) >= 0):
            order = np.argsort(s, kind="stable")
            s = s[order]
            e = e[order]
            if self.values is not None:
                object.__setattr__(self, "values", np.asarray(self.values)[..., order])
            if self.ids is not None:
                object.__setattr__(self, "ids", np.asarray(self.ids)[order])
        object.__setattr__(self, "starts_ticks", s)
        object.__setattr__(self, "ends_ticks", e)
        if self.ids is None:
            object.__setattr__(self, "ids", _new_ids(s.shape[0]))
        else:
            ids = np.asarray(self.ids, dtype=np.int64)
            if ids.shape != (s.shape[0],):
                raise ValueError("ids must be 1-D with length M")
            object.__setattr__(self, "ids", ids)
        if self.values is not None:
            v = np.asarray(self.values)
            if v.shape[-1] != s.shape[0]:
                raise ValueError("values.shape[-1] must equal M")
            object.__setattr__(self, "values", v)
        if self.dur_ticks < 0:
            raise ValueError(f"dur_ticks ({self.dur_ticks}) < 0")
        if s.size:
            if s[0] < 0:
                raise ValueError("segment starts before t_start (relative < 0)")
            if e.max() > self.dur_ticks:
                raise ValueError(
                    f"segment ends after t_end (relative {e.max()} > dur {self.dur_ticks})"
                )

    # ---- constructors ---------------------------------------------------
    @classmethod
    def from_segments(
        cls,
        starts: np.ndarray,
        ends: np.ndarray,
        values: np.ndarray | None = None,
        ids: np.ndarray | None = None,
        *,
        t_start: float | int | None = None,
        t_end: float | int | None = None,
    ) -> "SegmentSeries":
        """Build from absolute times (float seconds or int ticks)."""
        s = np.asarray(starts)
        e = np.asarray(ends)
        if np.issubdtype(s.dtype, np.floating):
            s_ticks = secs_array_to_ticks(s)
        else:
            s_ticks = s.astype(np.int64)
        if np.issubdtype(e.dtype, np.floating):
            e_ticks = secs_array_to_ticks(e)
        else:
            e_ticks = e.astype(np.int64)
        if t_start is None:
            t0 = int(s_ticks.min()) if s_ticks.size else 0
        elif isinstance(t_start, (int, np.integer)):
            t0 = int(t_start)
        else:
            t0 = secs_to_ticks(float(t_start))
        rel_s = s_ticks - t0
        rel_e = e_ticks - t0
        if t_end is None:
            dur = int(rel_e.max()) if rel_e.size else 0
        elif isinstance(t_end, (int, np.integer)):
            dur = int(t_end) - t0
        else:
            dur = secs_to_ticks(float(t_end)) - t0
        return cls(
            starts_ticks=rel_s, ends_ticks=rel_e, values=values, ids=ids,
            t_start_ticks=t0, dur_ticks=dur,
        )

    @classmethod
    def from_ticks(
        cls,
        starts: np.ndarray,
        ends: np.ndarray,
        values: np.ndarray | None = None,
        ids: np.ndarray | None = None,
        *,
        t_start: int = 0,
        dur: int,
    ) -> "SegmentSeries":
        """Build from relative int64 ticks."""
        rs = np.asarray(starts, dtype=np.int64)
        re = np.asarray(ends, dtype=np.int64)
        return cls(
            starts_ticks=rs, ends_ticks=re, values=values, ids=ids,
            t_start_ticks=int(t_start), dur_ticks=int(dur),
        )

    @property
    def starts(self) -> np.ndarray:
        """Relative start times as float seconds."""
        return ticks_array_to_secs(self.starts_ticks)

    @property
    def ends(self) -> np.ndarray:
        """Relative end times as float seconds."""
        return ticks_array_to_secs(self.ends_ticks)

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

    # ---- array accessors ------------------------------------------------
    @property
    def abs_starts(self) -> np.ndarray:
        return (self.starts_ticks + self.t_start_ticks).astype(np.float64) / TICKS_PER_SECOND

    @property
    def abs_ends(self) -> np.ndarray:
        return (self.ends_ticks + self.t_start_ticks).astype(np.float64) / TICKS_PER_SECOND

    @property
    def abs_starts_ticks(self) -> np.ndarray:
        return self.starts_ticks + self.t_start_ticks

    @property
    def abs_ends_ticks(self) -> np.ndarray:
        return self.ends_ticks + self.t_start_ticks

    # ---- shape ----------------------------------------------------------
    def __len__(self) -> int:
        return int(self.starts_ticks.shape[0])

    def __getitem__(self, i: Any):
        s = ticks_to_secs(int(self.starts_ticks[i]) + self.t_start_ticks)
        e = ticks_to_secs(int(self.ends_ticks[i]) + self.t_start_ticks)
        if self.values is None:
            return s, e, int(self.ids[i])
        return s, e, self.values[..., i], int(self.ids[i])

    # ---- slice ----------------------------------------------------------
    def slice(self, t_a: float | int, t_b: float | int) -> "SegmentSeries":
        a_tick = _c_to_ticks(t_a) if not isinstance(t_a, int) else t_a
        b_tick = _c_to_ticks(t_b) if not isinstance(t_b, int) else t_b
        t0 = self.t_start_ticks
        dur = self.dur_ticks
        if a_tick < t0 or b_tick > t0 + dur or a_tick > b_tick:
            raise DomainError(
                f"slice({a_tick}, {b_tick}) outside [{t0}, {t0 + dur}] (ticks)"
            )
        ra = a_tick - t0
        rb = b_tick - t0
        s, e = self.starts_ticks, self.ends_ticks  # relative int64
        keep = (e > ra) & (s < rb)
        ns = np.clip(s[keep], ra, None) - ra
        ne = np.clip(e[keep], None, rb) - ra
        nonzero = ne > ns
        ns = ns[nonzero]
        ne = ne[nonzero]
        nv = None if self.values is None else self.values[..., keep][..., nonzero]
        nids = self.ids[keep][nonzero]
        return SegmentSeries(
            starts_ticks=ns, ends_ticks=ne, values=nv, ids=nids,
            t_start_ticks=a_tick, dur_ticks=rb - ra,
        )

    # ---- shift ----------------------------------------------------------
    def shift(self, t_delta: float | int) -> "SegmentSeries":
        dt = _c_to_ticks(t_delta) if not isinstance(t_delta, int) else t_delta
        if dt == 0:
            return self
        return replace(self, t_start_ticks=self.t_start_ticks + dt)

    # ---- concat ---------------------------------------------------------
    def concat(self, other: "SegmentSeries") -> "SegmentSeries":
        if not isinstance(other, SegmentSeries):
            raise IncompatibleSeriesError(
                f"cannot concat SegmentSeries with {type(other).__name__}"
            )
        if (self.values is None) != (other.values is None):
            raise IncompatibleSeriesError("one has values, the other does not")

        # other is glued so its t_start lands at self.t_end.
        self_dur = self.dur_ticks
        other_dur = other.dur_ticks
        o_starts = other.starts_ticks + int(self_dur)
        o_ends = other.ends_ticks + int(self_dur)

        # Find pairs to merge: id present in self ending at seam AND in
        # other starting at seam.
        seam = self_dur  # relative to self.t_start_ticks
        left_at_seam = {
            int(self.ids[i]): i
            for i in np.where(self.ends_ticks == seam)[0]
        }
        right_at_seam = {
            int(other.ids[j]): j
            for j in np.where(o_starts == seam)[0]
        }
        merge_ids = set(left_at_seam) & set(right_at_seam)

        new_left_ends = self.ends_ticks.copy()
        for mid in merge_ids:
            i = left_at_seam[mid]
            j = right_at_seam[mid]
            new_left_ends[i] = o_ends[j]

        right_keep = np.ones(len(other), dtype=bool)
        for mid in merge_ids:
            right_keep[right_at_seam[mid]] = False

        new_starts = np.concatenate([self.starts_ticks, o_starts[right_keep]])
        new_ends = np.concatenate([new_left_ends, o_ends[right_keep]])
        new_ids = np.concatenate([self.ids, other.ids[right_keep]])
        if self.values is None:
            new_vals = None
        else:
            new_vals = np.concatenate([self.values, other.values[..., right_keep]], axis=-1)  # type: ignore[arg-type]

        return SegmentSeries(
            starts_ticks=new_starts, ends_ticks=new_ends, values=new_vals, ids=new_ids,
            t_start_ticks=self.t_start_ticks, dur_ticks=self_dur + other_dur,
        )

    # ---- interpolation -------------------------------------------------
    def interpolate(
        self, times, *, kind: str = "linear", fill: str = "clamp",
    ) -> "np.ndarray":  # type: ignore[name-defined]
        """Segment series have no point-wise values; raises TypeError."""
        raise TypeError(
            "SegmentSeries does not support interpolate; "
            "use .contains(times) or .overlap() for interval queries"
        )

    # ---- equality -------------------------------------------------------
    def equal(self, other: TimeSeries) -> bool:
        if not isinstance(other, SegmentSeries):
            return False
        if not (
            self.t_start_ticks == other.t_start_ticks
            and self.dur_ticks == other.dur_ticks
        ):
            return False
        if not np.array_equal(self.starts_ticks, other.starts_ticks):
            return False
        if not np.array_equal(self.ends_ticks, other.ends_ticks):
            return False
        if not np.array_equal(self.ids, other.ids):
            return False
        if (self.values is None) != (other.values is None):
            return False
        if self.values is None:
            return True
        return np.array_equal(self.values, other.values)
