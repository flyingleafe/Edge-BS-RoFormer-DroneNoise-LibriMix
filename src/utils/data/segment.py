"""Time-interval (segment) series.

A `SegmentSeries` stores segments `[t_begin, t_end)` with optional payloads.
Segments are sorted by `t_begin` and may overlap each other arbitrarily.

Slicing at a point `t_cut` *splits* any straddling segment into two halves
that share an identity tag (`ids`). Concat at a matching seam re-merges
segments whose `ids` agree.

The identity tag is the algebraic marker that lets `slice(a,b) ⊕ slice(b,c)
== slice(a,c)` hold exactly. Auto-generated ids are 62-bit random integers;
collision is negligible in practice but users may pass explicit ids.
"""
from __future__ import annotations

import secrets
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from ._floats import DEFAULT_ATOL, DEFAULT_RTOL, t_atol_at, tclose
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
    starts, ends : np.ndarray, shape (M,)
        `starts[i] < ends[i]` for all i. Sorted by `starts`.
    values : np.ndarray | None
        Shape `(M, ...)` payload (e.g. labels). Optional.
    ids : np.ndarray, shape (M,), dtype int64
        Identity tags. Segments with identical ids are treated as the same
        underlying segment when concatenating at a seam.
    t_start, t_end : float
        Declared domain. All segments must lie inside.
    """

    starts: np.ndarray = field(repr=False)
    ends: np.ndarray = field(repr=False)
    values: np.ndarray | None = field(repr=False, default=None)
    ids: np.ndarray = field(repr=False, default=None)  # type: ignore[assignment]
    t_start: float = 0.0
    t_end: float = 0.0

    def __post_init__(self) -> None:
        s = np.asarray(self.starts, dtype=np.float64)
        e = np.asarray(self.ends, dtype=np.float64)
        object.__setattr__(self, "starts", s)
        object.__setattr__(self, "ends", e)
        if s.shape != e.shape or s.ndim != 1:
            raise ValueError("starts and ends must be 1-D and same shape")
        if not np.all(e > s):
            raise ValueError("each segment must have end > start")
        if s.size and not np.all(np.diff(s) >= 0):
            # Sort if needed (silent fix-up is friendlier than raising).
            order = np.argsort(s, kind="stable")
            object.__setattr__(self, "starts", s[order])
            object.__setattr__(self, "ends", e[order])
            if self.values is not None:
                object.__setattr__(self, "values", np.asarray(self.values)[order])
            if self.ids is not None:
                object.__setattr__(self, "ids", np.asarray(self.ids)[order])
        if self.ids is None:
            object.__setattr__(self, "ids", _new_ids(s.shape[0]))
        else:
            ids = np.asarray(self.ids, dtype=np.int64)
            if ids.shape != (s.shape[0],):
                raise ValueError("ids must be 1-D with length M")
            object.__setattr__(self, "ids", ids)
        if self.values is not None:
            v = np.asarray(self.values)
            if v.shape[0] != s.shape[0]:
                raise ValueError("values.shape[0] must equal M")
            object.__setattr__(self, "values", v)
        if self.t_end < self.t_start:
            raise ValueError(f"t_end ({self.t_end}) < t_start ({self.t_start})")
        if s.size:
            lo_atol = t_atol_at(self.t_start)
            hi_atol = t_atol_at(self.t_end)
            if s[0] < self.t_start - lo_atol:
                raise ValueError(f"segment starts before t_start ({s[0]} < {self.t_start})")
            if e.max() > self.t_end + hi_atol:
                raise ValueError(f"segment ends after t_end ({e.max()} > {self.t_end})")

    @classmethod
    def from_segments(
        cls,
        starts: np.ndarray,
        ends: np.ndarray,
        values: np.ndarray | None = None,
        ids: np.ndarray | None = None,
        t_start: float | None = None,
        t_end: float | None = None,
    ) -> "SegmentSeries":
        s = np.asarray(starts, dtype=np.float64)
        e = np.asarray(ends, dtype=np.float64)
        if t_start is None:
            t_start = float(s.min()) if s.size else 0.0
        if t_end is None:
            t_end = float(e.max()) if e.size else float(t_start)
        return cls(
            starts=s, ends=e, values=values, ids=ids,
            t_start=float(t_start), t_end=float(t_end),
        )

    # ------------------------------------------------------------------ shape
    def __len__(self) -> int:
        return int(self.starts.shape[0])

    def __getitem__(self, i: Any):
        if self.values is None:
            return float(self.starts[i]), float(self.ends[i]), int(self.ids[i])
        return (
            float(self.starts[i]), float(self.ends[i]),
            self.values[i], int(self.ids[i]),
        )

    # ------------------------------------------------------------------ slice
    def slice(self, t_a: float, t_b: float) -> "SegmentSeries":
        lo_atol = t_atol_at(self.t_start)
        hi_atol = t_atol_at(self.t_end)
        ab_atol = t_atol_at(max(abs(t_a), abs(t_b)))
        if t_a < self.t_start - lo_atol or t_b > self.t_end + hi_atol or t_a > t_b + ab_atol:
            raise DomainError(
                f"slice({t_a}, {t_b}) outside [{self.t_start}, {self.t_end}]"
            )
        # Slicing exactly at a domain boundary uses atol slack (mirrors
        # __post_init__). Interior cuts remain strict.
        at_left = t_a == self.t_start
        at_right = t_b == self.t_end
        s, e = self.starts, self.ends
        # Keep segments whose intervals intersect [t_a, t_b).
        keep = (e > t_a) & (s < t_b)
        clip_lo = self.t_start - lo_atol if at_left else t_a
        clip_hi = self.t_end + hi_atol if at_right else t_b
        ns = np.clip(s[keep], clip_lo, None)
        ne = np.clip(e[keep], None, clip_hi)
        # Drop degenerate (zero-width) after clipping — but this shouldn't happen
        # given `end > start` and the intersection mask, except at exact-boundary
        # contact which is excluded by the strict inequalities above.
        nonzero = ne > ns
        ns = ns[nonzero]
        ne = ne[nonzero]
        nv = None if self.values is None else self.values[keep][nonzero]
        nids = self.ids[keep][nonzero]
        return SegmentSeries(
            starts=ns, ends=ne, values=nv, ids=nids,
            t_start=float(t_a), t_end=float(t_b),
        )

    # ------------------------------------------------------------------ shift
    def shift(self, t_delta: float) -> "SegmentSeries":
        if t_delta == 0.0:
            return self
        return replace(
            self,
            t_start=self.t_start + t_delta,
            t_end=self.t_end + t_delta,
            starts=self.starts + t_delta,
            ends=self.ends + t_delta,
        )

    # ------------------------------------------------------------------ concat
    def concat(
        self, other: "SegmentSeries",
        atol: float = DEFAULT_ATOL, rtol: float = DEFAULT_RTOL,
    ) -> "SegmentSeries":
        if not isinstance(other, SegmentSeries):
            raise IncompatibleSeriesError(f"cannot concat SegmentSeries with {type(other).__name__}")
        if (self.values is None) != (other.values is None):
            raise IncompatibleSeriesError("one has values, the other does not")

        # Auto-shift other so its timeline aligns with self.t_end.
        delta = self.t_end - other.t_start
        other = other.shift(delta)

        seam = self.t_end
        seam_atol = max(atol, t_atol_at(seam))

        # Find pairs to merge: id present in self ending at seam AND in other starting at seam.
        left_at_seam = {
            int(self.ids[i]): i
            for i in np.where(np.isclose(self.ends, seam, atol=seam_atol, rtol=rtol))[0]
        }
        right_at_seam = {
            int(other.ids[j]): j
            for j in np.where(np.isclose(other.starts, seam, atol=seam_atol, rtol=rtol))[0]
        }
        merge_ids = set(left_at_seam) & set(right_at_seam)

        # Build the merged arrays.
        # Step 1: extend left rows whose id is in merge_ids by the matching right end.
        new_left_ends = self.ends.copy()
        for mid in merge_ids:
            i = left_at_seam[mid]
            j = right_at_seam[mid]
            new_left_ends[i] = other.ends[j]

        # Step 2: drop matched rows from the right.
        right_keep = np.ones(len(other), dtype=bool)
        for mid in merge_ids:
            right_keep[right_at_seam[mid]] = False

        new_starts = np.concatenate([self.starts, other.starts[right_keep]])
        new_ends = np.concatenate([new_left_ends, other.ends[right_keep]])
        new_ids = np.concatenate([self.ids, other.ids[right_keep]])
        if self.values is None:
            new_vals = None
        else:
            new_vals = np.concatenate([self.values, other.values[right_keep]])  # type: ignore[arg-type]

        return SegmentSeries(
            starts=new_starts, ends=new_ends, values=new_vals, ids=new_ids,
            t_start=self.t_start, t_end=other.t_end,
        )

    def equal(self, other: TimeSeries) -> bool:
        if not isinstance(other, SegmentSeries):
            return False
        if not (
            tclose(self.t_start, other.t_start, atol=t_atol_at(self.t_start))
            and tclose(self.t_end, other.t_end, atol=t_atol_at(self.t_end))
        ):
            return False
        # Use rtol-only comparison to absorb ulp at Unix-magnitude anchors.
        if not (
            np.allclose(self.starts, other.starts, atol=DEFAULT_ATOL, rtol=1e-12)
            and np.allclose(self.ends, other.ends, atol=DEFAULT_ATOL, rtol=1e-12)
        ):
            return False
        if not np.array_equal(self.ids, other.ids):
            return False
        if (self.values is None) != (other.values is None):
            return False
        if self.values is None:
            return True
        return np.array_equal(self.values, other.values)
