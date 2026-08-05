"""Speech-enhancement comparison figures + the mixture/target/output triple.

The sample-shape logic (which frame entries make an SE triple, and how they
collapse to mono audio) lived in ``training.val_logging``; it moved here so
the wandb adapter and ``plots.dwym`` share one recipe. ``val_logging`` keeps
only the wandb wrapping.

Canonical entries (see :mod:`data_processing.canonical`): the target/input frame carries
``mixture`` + ``target``, the prediction frame carries ``enhanced``.
"""

from __future__ import annotations

from typing import Any

import matplotlib.figure
import tdseries as td

from plots.timeframe import PlotTrack, plot_timeframe
from plots.timeframe.renderers import make_spectrogram_series
from utils.audio import first_channel, sample_rate_of, to_mono

__all__ = ["extract_se_triple", "se_comparison_tracks", "plot_se_comparison"]

#: Row order of the comparison figure; only the entries present are drawn.
SE_ENTRIES = ("mixture", "target", "enhanced")


def extract_se_triple(pred: td.Frame, target: td.Frame) -> dict[str, tuple[Any, int]] | None:
    """Mono ``{"mixture"/"target"/"output": (waveform, sr)}`` from a Frame pair.

    Returns ``None`` when the pair is not speech-enhancement-shaped
    (``mixture``/``target`` on the target frame, ``enhanced`` on the
    prediction frame). This is the exact eligibility rule the training-time
    sample logger applies.
    """
    if "mixture" not in target or "target" not in target or "enhanced" not in pred:
        return None
    mixture_series = target["mixture"]
    sr = sample_rate_of(mixture_series)
    return {
        "mixture": (to_mono(mixture_series), sr),
        "target": (to_mono(target["target"]), sr),
        "output": (to_mono(pred["enhanced"]), sr),
    }


def se_comparison_tracks(frame: td.Frame, *, fmax: float | None = None) -> list[PlotTrack]:
    """Spectrogram ``PlotTrack`` rows for the SE entries present in ``frame``.

    One row per present entry of :data:`SE_ENTRIES`, in that order, titled
    by the entry name. Multichannel entries render channel 0.
    """
    tracks: list[PlotTrack] = []
    for name in SE_ENTRIES:
        if name not in frame:
            continue
        series = frame[name]
        if not isinstance(series, td.Series):
            continue
        spec = make_spectrogram_series(first_channel(series), fmax=fmax)
        tracks.append(
            PlotTrack(
                series=spec.series, renderer=spec.renderer, hints={**spec.hints, "title": name}
            )
        )
    if not tracks:
        raise ValueError(f"Frame has none of the SE entries {SE_ENTRIES}")
    return tracks


def plot_se_comparison(
    frame: td.Frame,
    *,
    fmax: float | None = None,
    figsize: tuple[float, float] = (14, 9),
    **style: Any,
) -> matplotlib.figure.Figure:
    """Spectrogram rows (mixture / target / enhanced) on a shared time axis."""
    return plot_timeframe(
        frame, tracks=list(se_comparison_tracks(frame, fmax=fmax)), figsize=figsize, **style
    )
