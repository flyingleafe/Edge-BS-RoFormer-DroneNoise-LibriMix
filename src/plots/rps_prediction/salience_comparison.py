"""Salience-map comparison: spectrogram + per-model salience heatmaps + RPS-over-GT.

This builds on the generic ``timeframe`` machinery. For each salience-map RPS
model (``outputs_salience=True``, e.g. ``LateDeepSalience`` / ``BasicPitchSalience``)
it produces:

* the input log-magnitude spectrogram (one selected channel),
* one salience-heatmap row per model, with the ground-truth RPS (dotted) and the
  model's tracked RPS prediction (solid) overlaid — see the ``"salience"``
  renderer in :mod:`plots.timeframe.renderers`,
* a final RPS-trajectory row overlaying GT and every model's prediction.

The inference helpers (:func:`model_salience_series`, :func:`model_rps_prediction`)
are independent of plotting and reusable on their own.
"""

from __future__ import annotations

from typing import Any, cast

import matplotlib.figure
import numpy as np
import torch

from models.multif0.utils import cqt_freq_grid
from models.salience_rps import SalienceRPSPredictor
from plots.timeframe import plot_timeframe
from plots.timeframe.renderers import (
    ROTOR_COLORS,
    make_salience_series,
    make_spectrogram_series,
)
from tasks.rps_prediction import HOP, N_FFT, align_rps_to_gt
from utils.data import TimeFrame, UniformSeries

__all__ = [
    "select_channel",
    "model_salience_series",
    "model_rps_prediction",
    "build_salience_frame",
    "plot_salience_comparison",
]


def select_channel(audio: UniformSeries, channel: int = 0) -> UniformSeries:
    """Return a mono ``UniformSeries`` for one channel of a (possibly) multichannel one."""
    samples = np.asarray(audio.samples, dtype=np.float32)
    if samples.ndim == 1:
        if channel != 0:
            raise ValueError(f"Mono audio only supports channel 0, got {channel}")
        mono = samples
    else:
        n_ch = samples.shape[0]
        if not (0 <= channel < n_ch):
            raise ValueError(f"Channel {channel} out of range (0..{n_ch - 1})")
        mono = samples[channel]
    return UniformSeries.from_samples(mono, sr=audio.sr, t_start=audio.t_start_ticks)


@torch.no_grad()
def model_salience_series(
    model: SalienceRPSPredictor,
    audio: UniformSeries,
    *,
    device: str | torch.device = "cpu",
    title: str | None = None,
    with_prediction: bool = True,
    track_threshold: float = 0.3,
) -> UniformSeries:
    """Run a salience model on a mono audio track → salience ``UniformSeries``.

    The returned series carries the bin centre frequencies (``plot.freqs``) and,
    if ``with_prediction``, the tracked RPS overlay (``plot.rps_pred``), so it is
    ready for the ``"salience"`` renderer.
    """
    model.eval()
    wav = torch.as_tensor(np.asarray(audio.samples, dtype=np.float32), device=device)
    if wav.ndim != 1:
        raise ValueError(f"model_salience_series expects mono audio, got shape {tuple(wav.shape)}")
    logits = model(wav.unsqueeze(0))  # (1, n_bins, T)
    salience = torch.sigmoid(logits)[0]  # (n_bins, T)
    # Frequency axis of the salience the model *emits*: the decoupled output grid
    # (super-resolution head) when present, else the CQT input grid.
    if getattr(model, "output_freqs", None) is not None:
        freqs = model.output_freqs()
    else:
        freqs = cqt_freq_grid(**model.grid_params())
    frame_sr = float(model.spec_sr) / float(model.spec_hop)
    rps_pred = None
    if with_prediction:
        rps_pred = model.predict_rps(wav.unsqueeze(0), threshold=track_threshold)[0]  # (R, T_stft)
    return make_salience_series(
        salience,
        freqs=freqs,
        frame_sr=frame_sr,
        t_start=audio.t_start,
        rps_pred=rps_pred,
        title=title,
    )


@torch.no_grad()
def model_rps_prediction(
    model: SalienceRPSPredictor,
    audio: UniformSeries,
    *,
    device: str | torch.device = "cpu",
    track_threshold: float = 0.3,
) -> np.ndarray:
    """Return the model's tracked RPS prediction ``(n_rotors, T_stft)`` as numpy."""
    model.eval()
    wav = torch.as_tensor(np.asarray(audio.samples, dtype=np.float32), device=device)
    pred = model.predict_rps(wav.unsqueeze(0), threshold=track_threshold)[0]
    return pred.detach().cpu().numpy()


def build_salience_frame(
    sample: TimeFrame,
    models: dict[str, SalienceRPSPredictor],
    *,
    channel: int = 0,
    device: str | torch.device = "cpu",
    fmax: float | None = 4000.0,
    track_threshold: float = 0.3,
) -> TimeFrame:
    """Assemble a ``TimeFrame`` with spectrogram + GT RPS + per-model salience tracks.

    The input ``sample`` must have an ``"audio"`` track and an ``"rps"`` (GT)
    ``EventSeries``. Track names: ``"spectrogram"``, ``"rps"`` (GT), and
    ``"salience_<model>"`` for each model.
    """
    audio_us = cast(UniformSeries, sample["audio"])
    mono = select_channel(audio_us, channel)

    tracks: dict[str, Any] = {"audio": mono}
    if "rps" in sample:
        tracks["rps"] = sample["rps"]
    tracks["spectrogram"] = make_spectrogram_series(mono, fmax=fmax)

    frame = TimeFrame.from_tracks(tracks, tags=dict(sample.tags))
    for name, model in models.items():
        sal = model_salience_series(
            model, mono, device=device, title=f"{name} salience", track_threshold=track_threshold
        )
        frame = frame.with_track(f"salience_{name}", sal)
    return frame


def plot_salience_comparison(
    sample: TimeFrame,
    models: dict[str, SalienceRPSPredictor],
    *,
    channel: int = 0,
    device: str | torch.device = "cpu",
    fmax: float | None = 4000.0,
    track_threshold: float = 0.3,
    show_rps_row: bool = True,
    figsize: tuple[float, float] | None = None,
    **style: Any,
) -> matplotlib.figure.Figure:
    """Spectrogram + per-model salience heatmaps + (optional) RPS-over-GT row.

    Parameters
    ----------
    sample
        ``TimeFrame`` with ``"audio"`` and ``"rps"`` (GT) tracks (e.g. from
        :func:`plots.rps_prediction.sample_comparison._load_sample`).
    models
        ``{display_name: loaded_model}`` mapping. Each model must expose the
        salience interface (``forward`` → logits, ``predict_rps``, ``grid_params``,
        ``spec_sr``, ``spec_hop``).
    channel
        Which audio channel to visualise (DREGON-LM-V4 is multichannel).
    show_rps_row
        Append a final row overlaying GT and every model's predicted RPS.
    """
    frame = build_salience_frame(
        sample, models, channel=channel, device=device, fmax=fmax, track_threshold=track_threshold
    )

    tracks = ["spectrogram"] + [f"salience_{name}" for name in models]
    if show_rps_row and "rps" in frame:
        tracks.append("rps")
    # Salience rows are taller than the spectrogram / line rows.
    height_ratios = []
    for t in tracks:
        height_ratios.append(2.0 if t.startswith("salience_") else 1.0)

    if figsize is None:
        figsize = (15, 3.0 * sum(height_ratios) / max(len(tracks), 1) + 2.0 * len(tracks))

    fig = plot_timeframe(
        frame,
        tracks=tracks,
        figsize=figsize,
        height_ratios=height_ratios,
        **style,
    )

    if show_rps_row and "rps" in frame:
        # Overlay each model's predicted RPS on the final (GT) rps row. The rps
        # track is always the last one added and (unlike salience rows) adds no
        # colorbar axis, so it is reliably ``fig.axes[-1]``.
        ax = fig.axes[-1]
        mono = select_channel(cast(UniformSeries, sample["audio"]), channel)
        dur = mono.duration
        gt_track = sample["rps"] if "rps" in sample else None  # noqa: SIM401 (TimeFrame has no .get)
        for model in models.values():
            pred = model_rps_prediction(model, mono, device=device, track_threshold=track_threshold)
            pred_times = np.linspace(mono.t_start, mono.t_start + dur, pred.shape[-1])
            # Align rotor order to GT (PIT match) so overlay colours are consistent.
            if gt_track is not None and getattr(gt_track, "values", None) is not None:
                pred = align_rps_to_gt(pred, np.asarray(gt_track.interpolate(pred_times)))
            for r in range(min(pred.shape[0], len(ROTOR_COLORS))):
                ax.plot(
                    pred_times,
                    pred[r],
                    color=ROTOR_COLORS[r],
                    linewidth=1.4,
                    linestyle="--",
                    alpha=0.9,
                )
        ax.set_title("RPS — GT (solid) vs predictions (dashed)")

    return fig


# Frame rate of the salience grid is model-specific; HOP / N_FFT are re-exported
# for callers that build their own STFT-grid overlays.
__all__ += ["HOP", "N_FFT"]
