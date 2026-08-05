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

from fractions import Fraction
from typing import Any, cast

import matplotlib.figure
import numpy as np
import tdseries as td
import torch

from losses.pit import align_rps_to_gt
from models.multif0.utils import cqt_freq_grid
from models.salience_rps import SalienceRPSPredictor
from plots.timeframe import PlotTrack, plot_timeframe
from plots.timeframe.renderers import (
    ROTOR_COLORS,
    make_salience_series,
    make_spectrogram_series,
)

__all__ = [
    "select_channel",
    "model_salience_series",
    "model_rps_prediction",
    "build_salience_tracks",
    "plot_salience_comparison",
]


def select_channel(audio: td.Series, channel: int = 0) -> td.Series:
    """Return a mono ``Series`` for one channel of a (possibly) multichannel one."""
    extra = [d for d in audio.dims if d != "time"]
    if not extra:
        if channel != 0:
            raise ValueError(f"Mono audio only supports channel 0, got {channel}")
        return audio
    if len(extra) > 1 or extra[0] not in ("mic", "channel"):
        raise ValueError(f"Unsupported audio dims {audio.dims!r} for channel selection")
    dim = extra[0]
    n_ch = audio.dim_size(dim)
    if not (0 <= channel < n_ch):
        raise ValueError(f"Channel {channel} out of range (0..{n_ch - 1})")
    return audio.slice[dim, channel]


@torch.no_grad()
def model_salience_series(
    model: SalienceRPSPredictor,
    audio: td.Series,
    *,
    device: str | torch.device = "cpu",
    title: str | None = None,
    with_prediction: bool = True,
    track_threshold: float = 0.3,
) -> PlotTrack:
    """Run a salience model on a mono audio track → salience ``PlotTrack``.

    The returned track carries the bin centre frequencies (``hints["freqs"]``)
    and, if ``with_prediction``, the tracked RPS overlay (``hints["rps_pred"]``),
    so it is ready for the ``"salience"`` renderer.
    """
    model.eval()
    wav_data = np.asarray(audio.data, dtype=np.float32)
    if wav_data.ndim != 1:
        raise ValueError(f"model_salience_series expects mono audio, got shape {wav_data.shape}")
    wav = torch.as_tensor(wav_data, device=device)
    logits = model(wav.unsqueeze(0))  # (1, n_bins, T)
    salience = torch.sigmoid(logits)[0]  # (n_bins, T)
    # Frequency axis of the salience the model *emits*: the decoupled output grid
    # (super-resolution head) when present, else the CQT input grid.
    if getattr(model, "output_freqs", None) is not None:
        freqs = model.output_freqs()
    else:
        freqs = cqt_freq_grid(**model.grid_params())
    # Exact frame rate — never a rounded float (spec_sr / spec_hop is exactly
    # rational by construction).
    frame_sr = Fraction(int(model.spec_sr), int(model.spec_hop))
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
    audio: td.Series,
    *,
    device: str | torch.device = "cpu",
    track_threshold: float = 0.3,
) -> np.ndarray:
    """Return the model's tracked RPS prediction ``(n_rotors, T_stft)`` as numpy."""
    model.eval()
    wav_data = np.asarray(audio.data, dtype=np.float32)
    wav = torch.as_tensor(wav_data, device=device)
    pred = model.predict_rps(wav.unsqueeze(0), threshold=track_threshold)[0]
    return pred.detach().cpu().numpy()


def build_salience_tracks(
    sample: td.Frame,
    models: dict[str, SalienceRPSPredictor],
    *,
    channel: int = 0,
    device: str | torch.device = "cpu",
    fmax: float | None = 4000.0,
    track_threshold: float = 0.3,
) -> dict[str, PlotTrack | td.Series]:
    """Build the ordered ``{name: track}`` mapping consumed by :func:`plot_salience_comparison`.

    Returns the spectrogram track, one salience track per model, and the GT
    ``"rps"`` track (if present). This is a plain mapping — not a ``Frame`` —
    because ``PlotTrack``s carry plot-only renderer/hints metadata that does
    not belong in the data model (see ``docs/refactor-unified-framework.md``).
    """
    audio_series = cast(td.Series, sample["audio"])
    mono = select_channel(audio_series, channel)

    tracks: dict[str, PlotTrack | td.Series] = {"audio": mono}
    if "rps" in sample:
        tracks["rps"] = cast(td.Series, sample["rps"])
    tracks["spectrogram"] = make_spectrogram_series(mono, fmax=fmax)
    for name, model in models.items():
        tracks[f"salience_{name}"] = model_salience_series(
            model, mono, device=device, title=f"{name} salience", track_threshold=track_threshold
        )
    return tracks


def plot_salience_comparison(
    sample: td.Frame,
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
        ``Frame`` with ``"audio"`` and ``"rps"`` (GT) entries (e.g. from
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
    tracks = build_salience_tracks(
        sample, models, channel=channel, device=device, fmax=fmax, track_threshold=track_threshold
    )

    plot_tracks: list[Any] = [tracks["spectrogram"]] + [
        tracks[f"salience_{name}"] for name in models
    ]
    # Salience rows are taller than the spectrogram / line rows.
    height_ratios = [1.0] + [2.0] * len(models)
    if show_rps_row and "rps" in tracks:
        plot_tracks.append(tracks["rps"])
        height_ratios.append(1.0)

    if figsize is None:
        figsize = (15, 3.0 * sum(height_ratios) / max(len(plot_tracks), 1) + 2.0 * len(plot_tracks))

    fig = plot_timeframe(
        sample,
        tracks=plot_tracks,
        figsize=figsize,
        height_ratios=height_ratios,
        **style,
    )

    if show_rps_row and "rps" in tracks:
        # Overlay each model's predicted RPS on the final (GT) rps row. The rps
        # track is always the last one added and (unlike salience rows) adds no
        # colorbar axis, so it is reliably ``fig.axes[-1]``.
        ax = fig.axes[-1]
        mono = select_channel(cast(td.Series, sample["audio"]), channel)
        dur = mono.duration
        gt_track = tracks.get("rps")
        for model in models.values():
            pred = model_rps_prediction(model, mono, device=device, track_threshold=track_threshold)
            pred_times = np.linspace(mono.t_start, mono.t_start + dur, pred.shape[-1])
            # Align rotor order to GT (PIT match) so overlay colours are consistent.
            if isinstance(gt_track, td.Series) and gt_track.data is not None:
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
