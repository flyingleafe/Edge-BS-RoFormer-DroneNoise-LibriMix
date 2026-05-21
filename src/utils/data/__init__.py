"""Aligned time-series containers — see `AGENTS.md` for the design rationale.

Public API:
    TimeSeries        - abstract base (don't instantiate)
    UniformSeries     - regular sample-rate signal (audio, video frames, ...)
    EventSeries       - sorted point-event series (RPS samples, IMU, ...)
    SegmentSeries     - half-open interval series (VAD, labels, ...)
    TimeFrame         - dict-keyed container of aligned tracks

Errors:
    DomainError, IncompatibleSeriesError
"""
from .base import DomainError, IncompatibleSeriesError, TimeSeries
from .event import EventSeries
from .frame import TimeFrame
from .segment import SegmentSeries
from .uniform import UniformSeries

__all__ = [
    "TimeSeries",
    "UniformSeries",
    "EventSeries",
    "SegmentSeries",
    "TimeFrame",
    "DomainError",
    "IncompatibleSeriesError",
]
