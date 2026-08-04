"""Small audio-``Series`` helpers shared across the plots package.

These used to be private helpers of ``training.val_logging``
(``_audio_to_mono`` / ``_sample_rate``). They moved here so the plots
package and the training-side wandb adapter use the same conversions.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import tdseries as td

__all__ = ["to_numpy", "to_mono", "sample_rate_of", "first_channel"]


def to_numpy(x: Any) -> np.ndarray:
    """``np.asarray`` that also accepts torch tensors (detach + cpu first)."""
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    return np.asarray(x)


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
