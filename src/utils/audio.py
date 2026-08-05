"""Audio-``Series`` helpers — mono collapse, sample rate, channel pick.

Nothing here is plot-specific (they started as private helpers of
``training.val_logging``, then lived in ``plots.audio``): `plots`, `metrics`
and the training-side wandb adapter all need the same conversions, so they
sit in the bottom layer where every one of them can reach them.

``to_numpy`` lives in :mod:`utils.arrays` — it is not audio at all.
"""

from __future__ import annotations

import numpy as np
import tdseries as td

from utils.arrays import to_numpy

__all__ = ["to_mono", "sample_rate_of", "first_channel"]


def to_mono(series: td.Series) -> np.ndarray:
    """Mono waveform from a ``"time"``-having ``Series``: passed through if
    already 1-D, else averaged over every non-time axis (mirrors the old
    ``np.mean(mix, axis=0)`` channel-collapse for multichannel audio)."""
    arr = to_numpy(series.data).astype(np.float32)
    if arr.ndim == 1:
        return arr
    return arr.mean(axis=tuple(range(arr.ndim - 1))).astype(np.float32)


def sample_rate_of(series: td.Series) -> int:
    """Integer sample rate of a uniformly sampled (``GridIndex``) ``Series``."""
    tindex = series.tindex
    if not isinstance(tindex, td.GridIndex):
        raise TypeError(f"expected a GridIndex time axis, got {type(tindex).__name__}")
    return int(round(float(tindex.sr)))


def first_channel(series: td.Series) -> td.Series:
    """A mono ``Series``: channel 0 of a multichannel one, else unchanged.

    Used before :func:`plots.timeframe.renderers.make_spectrogram_series`
    when one representative channel is sufficient (the spectrogram renderer
    draws 2-D data only).
    """
    extra = [d for d in series.dims if d is not None and d != "time"]
    if not extra:
        return series
    return series.slice[extra[0], 0]
