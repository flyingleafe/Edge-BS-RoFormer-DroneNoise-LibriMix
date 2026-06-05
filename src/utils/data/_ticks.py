"""Fixed-point time conversion helpers.

All time inside the library is stored as int64 tick counts at a fixed
TICKS_PER_SECOND (nanoseconds).  This module provides the conversion
boundary between the project's float-seconds ecosystem and the library's
exact int64 storage.

Exports
-------
TICKS_PER_SECOND     – resolution constant (1e9 = nanoseconds)
secs_to_ticks        – scalar float → scalar int
ticks_to_secs        – scalar int → scalar float
secs_array_to_ticks  – float array → int64 array
ticks_array_to_secs  – int64 array → float array
_c_to_ticks           – coerce float | int → int (for slice / shift / ctor args)
"""
from __future__ import annotations

import numpy as np

TICKS_PER_SECOND: int = 1_000_000_000


# ---- scalar ------------------------------------------------------------

def secs_to_ticks(seconds: float) -> int:
    """Convert scalar float-seconds to the nearest int64 tick."""
    return round(seconds * TICKS_PER_SECOND)


def ticks_to_secs(ticks: int) -> float:
    """Convert scalar int64 ticks to float seconds."""
    return ticks / TICKS_PER_SECOND


# ---- arrays ------------------------------------------------------------

def secs_array_to_ticks(seconds: np.ndarray) -> np.ndarray:
    """Convert an array of float seconds to int64 ticks (nearest)."""
    result = np.rint(np.asarray(seconds, dtype=np.float64) * TICKS_PER_SECOND)
    return result.astype(np.int64)


def ticks_array_to_secs(ticks: np.ndarray) -> np.ndarray:
    """Convert an array of int64 ticks to float seconds."""
    return np.asarray(ticks, dtype=np.float64) / TICKS_PER_SECOND


# ---- coerce (accept either) --------------------------------------------

def _c_to_ticks(value: float | int) -> int:
    """Coerce a ``float`` (seconds) or ``int`` (ticks) value to int ticks.

    This is the single conversion point for all slice / shift / constructor
    arguments, ensuring float seconds are quantised once and never re‑rounded.
    """
    if isinstance(value, int):
        return value
    return secs_to_ticks(float(value))
