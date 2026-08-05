"""Array coercion shared by every layer above.

One function, because three layers had their own copy of it
(``plots.audio.to_numpy``, ``metrics._common.to_numpy``,
``training.val_logging._to_numpy``).
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["to_numpy"]


def to_numpy(x: Any) -> np.ndarray:
    """``np.asarray`` for anything a Frame entry can hold.

    ``np.asarray`` already understands a CPU torch tensor through the
    array API, but it raises on one that requires grad and on a CUDA one —
    so detach and move to host first when the object offers it.
    """
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    return np.asarray(x)
