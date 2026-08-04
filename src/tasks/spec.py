"""Back-compat shim — the canonical home is ``framespec`` (shim removed in a later phase)."""

from __future__ import annotations

from framespec import *  # noqa: F403
from framespec import (
    SCALAR,
    EntrySpec,
    FrameSpec,
    ScalarSpec,
    SeriesSpec,
    TimeKind,
    check_subsumes,
    merge_specs,
    spec_of,
    without_batch,
)

__all__ = [
    "SCALAR",
    "EntrySpec",
    "FrameSpec",
    "ScalarSpec",
    "SeriesSpec",
    "TimeKind",
    "check_subsumes",
    "merge_specs",
    "spec_of",
    "without_batch",
]
