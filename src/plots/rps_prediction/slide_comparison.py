"""Horizontal slide-friendly comparison: spectrogram + RPS side-by-side."""

from __future__ import annotations

from typing import cast

import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import tdseries as td

from plots.rps_prediction.sample_comparison import (
    ROTOR_COLORS,
    _load_sample,
    _plot_spectrogram,
)
from tasks.rps_prediction import HOP, N_FFT


def plot_slide_comparison(
    *,
    sample: td.Frame | None = None,
    sample_path: str | None = None,
    channels: list[int] | None = None,
    preds: dict[str, np.ndarray] | dict[int, dict[str, np.ndarray]] | None = None,
    figsize: tuple[float, float] = (18, 5),
    title: str | None = None,
    **style,
) -> matplotlib.figure.Figure:
    """Generate a horizontal figure: for each channel, spectrogram (left) + RPS (right).

    Parameters
    ----------
    sample : td.Frame | None
        Pre-loaded sample.
    sample_path : str | None
        Path to sample directory.
    channels : list[int] | None
        Which channels to show (default [0, 1]).
    preds : dict[str, np.ndarray] | dict[int, dict[str, np.ndarray]] | None
        Model predictions to overlay.
        - Flat dict: same predictions shown for every channel.
        - Per-channel dict: {ch: {model_name: pred_array}} — predictions per channel.
    figsize : tuple
        Overall figure size.  Recommended: (18, 5) for 2 channels, (24, 8) for 4.
    title : str | None
        Figure suptitle.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if sample is None and sample_path is None:
        raise ValueError("One of sample or sample_path is required")
    if sample is None:
        assert sample_path is not None, "sample_path is required when sample is None"
        sample = _load_sample(sample_path)
    if preds is None:
        preds = {}
    if channels is None:
        channels = [0, 1]

    # Normalise preds to per-channel dict
    preds_by_ch: dict[int, dict[str, np.ndarray]]
    if preds and isinstance(next(iter(preds.values())), np.ndarray):
        preds_by_ch = {ch: cast(dict[str, np.ndarray], preds) for ch in channels}
    else:
        preds_by_ch = cast(dict[int, dict[str, np.ndarray]], preds)

    audio_series = cast(td.Series, sample["audio"])
    rps_series = cast(td.Series, sample["rps"])
    audio_tindex = audio_series.tindex
    if not isinstance(audio_tindex, td.GridIndex):
        raise TypeError(f"'audio' entry must be a uniform (GridIndex) Series, got {audio_series}")
    audio = np.asarray(audio_series.data, dtype=np.float32)
    sr = audio_tindex.sr
    dur = audio.shape[-1] / sr

    sample_audio = audio[0] if audio.ndim > 1 else audio
    n_frames = len(sample_audio) // HOP + 1
    frame_times = np.arange(n_frames) * HOP / sr + rps_series.t_start + N_FFT / sr / 2
    gt = np.asarray(rps_series.interpolate(frame_times))

    n_ch = len(channels)
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(n_ch, 2, hspace=0.35, wspace=0.15, height_ratios=[1.0] * n_ch)

    for idx, ch in enumerate(channels):
        ch_audio = audio[ch] if audio.ndim > 1 else audio

        # --- Left: spectrogram ---
        ax_spec = fig.add_subplot(gs[idx, 0])
        _plot_spectrogram(ax_spec, ch_audio, sr, audio_series.t_start, dur)
        ax_spec.set_title(f"ch{ch} — Spectrogram", fontsize=10)
        ax_spec.set_xlabel("")
        if idx < n_ch - 1:
            ax_spec.set_xlabel("")

        # --- Right: RPS overlay ---
        ax_rps = fig.add_subplot(gs[idx, 1], sharex=ax_spec if idx == 0 else None)

        # GT dotted
        for r, color in enumerate(ROTOR_COLORS):
            ax_rps.plot(
                frame_times,
                gt[r],
                color=color,
                linewidth=2,
                linestyle=":",
                alpha=0.7,
                label=f"GT R{r + 1}" if idx == 0 else "",
            )

        # Predictions (per-channel)
        ch_preds = preds_by_ch.get(ch, {})
        for model_name, pred in ch_preds.items():
            T_pred = pred.shape[-1]
            pred_times = np.linspace(0.0, dur, T_pred) + audio_series.t_start
            for r, color in enumerate(ROTOR_COLORS):
                ax_rps.plot(
                    pred_times,
                    pred[r],
                    color=color,
                    linewidth=1.5,
                    alpha=0.9,
                    label=f"{model_name} R{r + 1}" if idx == 0 else "",
                )

        ax_rps.set_ylabel("RPS", fontsize=9)
        ax_rps.set_title(f"ch{ch} — Predictions", fontsize=10)
        ax_rps.grid(True, alpha=0.3)
        if idx == 0:
            ax_rps.legend(loc="upper right", ncol=2, fontsize=6)
        if idx == n_ch - 1:
            ax_rps.set_xlabel("Time (s)", fontsize=9)

    if title:
        fig.suptitle(title, fontsize=12, y=1.01)

    fig.tight_layout()
    return fig
