"""Noise-generation real-vs-generated comparison figures + pair extraction.

The pair-shape logic (which frame entries make a real/generated pair)
lived in ``training.val_logging``; it moved here so the wandb adapter and
``plots.dwym`` share one recipe. ``val_logging`` keeps only the wandb
wrapping (captions, epoch/drone tags).

Canonical entries (see :mod:`data_processing.canonical`): the target frame carries the
REAL recording under ``audio``; the prediction frame carries the GENERATED
noise, also under ``audio``. A single merged frame uses ``audio`` +
``generated`` instead.
"""

from __future__ import annotations

from typing import Any

import matplotlib.figure
import tdseries as td

from plots.timeframe import PlotTrack, plot_timeframe
from plots.timeframe.renderers import make_spectrogram_series
from utils.audio import first_channel, sample_rate_of, to_mono

__all__ = [
    "extract_noise_gen_pair",
    "noise_gen_comparison_tracks",
    "plot_noise_gen_comparison",
]


def extract_noise_gen_pair(pred: td.Frame, target: td.Frame) -> dict[str, tuple[Any, int]] | None:
    """Mono ``{"real"/"generated": (waveform, sr)}`` from a Frame pair.

    Returns ``None`` when the pair is not noise-generation-shaped (both
    frames carry an ``audio`` entry). This is the exact eligibility rule
    the training-time sample logger applies.
    """
    if "audio" not in target or "audio" not in pred:
        return None
    real_series = target["audio"]
    sr = sample_rate_of(real_series)
    return {
        "real": (to_mono(real_series), sr),
        "generated": (to_mono(pred["audio"]), sr),
    }


def noise_gen_comparison_tracks(
    labeled_audio: dict[str, td.Series], *, fmax: float | None = None
) -> list[PlotTrack]:
    """One spectrogram ``PlotTrack`` row per labeled audio ``Series``.

    ``labeled_audio`` is an ordered ``{label: audio Series}`` mapping (for
    example ``{"real": ..., "generated": ...}``). Multichannel entries
    render channel 0.
    """
    if not labeled_audio:
        raise ValueError("labeled_audio is empty")
    tracks: list[PlotTrack] = []
    for label, series in labeled_audio.items():
        spec = make_spectrogram_series(first_channel(series), fmax=fmax)
        tracks.append(
            PlotTrack(
                series=spec.series, renderer=spec.renderer, hints={**spec.hints, "title": label}
            )
        )
    return tracks


def plot_noise_gen_comparison(
    labeled_audio: dict[str, td.Series],
    *,
    frame: td.Frame | None = None,
    fmax: float | None = None,
    figsize: tuple[float, float] = (14, 6),
    **style: Any,
) -> matplotlib.figure.Figure:
    """Aligned spectrogram grid: one row per labeled audio track.

    ``frame`` only supplies the shared time bounds; when omitted, a
    throwaway frame is built from ``labeled_audio``.
    """
    if frame is None:
        frame = td.Frame(dict(labeled_audio))
    return plot_timeframe(
        frame,
        tracks=list(noise_gen_comparison_tracks(labeled_audio, fmax=fmax)),
        figsize=figsize,
        **style,
    )
