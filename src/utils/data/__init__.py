"""Aligned time-series containers — exact int64 tick storage.

All time is stored as int64 nanosecond tick counts (``TICKS_PER_SECOND = 1e9``).
Public accessors (``.t_start``, ``.t_end``, ``.duration``, ``__getitem__``)
return float seconds for ergonomics; ``*_ticks`` accessors return exact int64
values for exact round-trips.  See ``API.md`` and ``AGENTS.md``.

Public API:
    TICKS_PER_SECOND  - resolution constant
    TimeSeries        - abstract base
    UniformSeries     - regular sample-rate signal (audio, video frames, ...)
    EventSeries       - sorted point-event series (RPS, IMU, ...)
    SegmentSeries     - half-open interval series (VAD, labels, ...)
    TimeFrame         - dict-keyed container of aligned tracks

Errors:
    DomainError, IncompatibleSeriesError
"""
from ._ticks import TICKS_PER_SECOND
from .base import DomainError, IncompatibleSeriesError, TimeSeries
from .event import EventSeries
from .frame import TimeFrame
from .segment import SegmentSeries
from .uniform import UniformSeries

__all__ = [
    "TICKS_PER_SECOND",
    "TimeSeries",
    "UniformSeries",
    "EventSeries",
    "SegmentSeries",
    "TimeFrame",
    "DomainError",
    "IncompatibleSeriesError",
]
