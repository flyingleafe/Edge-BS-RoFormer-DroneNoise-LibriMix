"""Default renderers and converters for ``plot_timeframe``."""

from __future__ import annotations

from typing import Any

import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np
import torch

from tasks.rps_prediction import HOP, N_FFT
from utils.data import EventSeries, SegmentSeries, UniformSeries

from .registry import RenderedTrack, TrackContext, register_renderer

ROTOR_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

# ---------------------------------------------------------------------------
# Channel selection helper
# ---------------------------------------------------------------------------


def _select_channels(series: UniformSeries, channel: int | list[int] | str | None) -> list[int]:
    """Return validated channel indices for a ``UniformSeries``."""
    if channel is None:
        return [0] if series.channel_shape == () else list(range(series.channel_shape[0]))
    if isinstance(channel, str):
        if channel != "all":
            raise ValueError(f"Unsupported value 'channel={channel}'")
        return [0] if series.channel_shape == () else list(range(series.channel_shape[0]))
    if isinstance(channel, int):
        channel = [channel]
    channel = list(channel)
    if series.channel_shape == ():
        for ch in channel:
            if ch != 0:
                raise ValueError(f"Mono audio only supports channel 0, got {ch}")
    else:
        n_ch = series.channel_shape[0]
        for ch in channel:
            if not (0 <= ch < n_ch):
                raise ValueError(f"Channel {ch} out of range (0..{n_ch - 1})")
    return channel


# ---------------------------------------------------------------------------
# Spectrogram converters
# ---------------------------------------------------------------------------


def make_spectrogram_series(
    audio: UniformSeries,
    *,
    n_fft: int = N_FFT,
    hop_length: int = HOP,
    fmax: float | None = 4000,
    log: bool = True,
) -> UniformSeries:
    """Return a time-frequency ``UniformSeries`` for an audio track.

    The returned series can be rendered with the ``"audio_spectrogram"``
    renderer.  Its samples have shape ``(n_freq_bins, n_time_frames)`` and
    its sample rate is the STFT frame rate ``audio.sr / hop_length``.
    """
    samples = np.asarray(audio.samples, dtype=np.float32)
    if samples.ndim == 1:
        waveform = torch.from_numpy(samples).float()
    elif samples.ndim == 2:
        # Flatten channels for a single spectrogram; the renderer handles one
        # channel at a time.
        waveform = torch.from_numpy(samples[0]).float()
    else:
        raise ValueError(f"Unsupported audio shape {samples.shape} for spectrogram")

    window = torch.hann_window(n_fft)
    X = torch.stft(
        waveform,
        n_fft=n_fft,
        hop_length=hop_length,
        window=window,
        return_complex=True,
    )
    S = torch.abs(X).numpy()
    if fmax is not None:
        max_bin = min(int(fmax / (audio.sr / 2) * S.shape[0]), S.shape[0])
        S = S[:max_bin, :]
    if log:
        S = 20 * np.log10(S + 1e-8)

    frame_rate = audio.sr / hop_length
    t0 = audio.t_start + n_fft / (2 * audio.sr)
    return UniformSeries.from_samples(
        S,
        sr=frame_rate,
        t_start=t0,
        tags={"plot.renderer": "audio_spectrogram"},
    )


def make_log_spectrogram_series(
    audio: UniformSeries,
    *,
    n_fft: int = N_FFT,
    hop_length: int = HOP,
    fmax: float | None = 4000,
) -> UniformSeries:
    """Convenience alias for ``make_spectrogram_series(..., log=True)``."""
    return make_spectrogram_series(
        audio,
        n_fft=n_fft,
        hop_length=hop_length,
        fmax=fmax,
        log=True,
    )


# ---------------------------------------------------------------------------
# Salience-map converter
# ---------------------------------------------------------------------------


def make_salience_series(
    salience: Any,
    *,
    freqs: Any,
    frame_sr: float,
    t_start: float = 0.0,
    rps_pred: Any | None = None,
    title: str | None = None,
) -> UniformSeries:
    """Wrap a model salience map as a ``UniformSeries`` for the ``"salience"`` renderer.

    Parameters
    ----------
    salience
        ``(n_bins, T)`` per-frequency-bin salience in ``[0, 1]`` (already
        sigmoid-activated). Tensors are accepted and converted to numpy.
    freqs
        ``(n_bins,)`` centre frequency (Hz) of each salience bin — used as the
        (log-scaled) y-axis. Get it from
        ``models.multif0.utils.cqt_freq_grid(**model.grid_params())``.
    frame_sr
        Salience frame rate in Hz (``model.spec_sr / model.spec_hop``).
    t_start
        Absolute start time (s) of the first salience frame.
    rps_pred
        Optional ``(n_rotors, T_pred)`` tracked-RPS trajectory to overlay as
        solid lines on the heatmap. Its own time axis is stretched to span the
        salience duration, matching how the tracker resamples to the STFT grid.
    title
        Optional subplot title (defaults to the track name).
    """
    samples = np.asarray(salience.detach().cpu() if hasattr(salience, "detach") else salience)
    samples = samples.astype(np.float32)
    if samples.ndim != 2:
        raise ValueError(f"salience must be 2-D (n_bins, T), got {samples.shape}")
    tags: dict[str, Any] = {
        "plot.renderer": "salience",
        "plot.freqs": np.asarray(freqs, dtype=np.float64),
    }
    if rps_pred is not None:
        rp = rps_pred.detach().cpu() if hasattr(rps_pred, "detach") else rps_pred
        tags["plot.rps_pred"] = np.asarray(rp, dtype=np.float64)
    if title is not None:
        tags["plot.title"] = title
    return UniformSeries.from_samples(samples, sr=frame_sr, t_start=t_start, tags=tags)


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------


def _render_spectrogram(ax: matplotlib.axes.Axes, series: UniformSeries) -> None:
    """Render a magnitude spectrogram produced by ``make_spectrogram_series``."""
    S = series.samples
    if S.ndim != 2:
        raise ValueError(f"Spectrogram series must be 2-D (freq, time), got {S.shape}")
    times = series.sample_times()
    freqs = np.linspace(0, series.sr / 2, S.shape[0])
    ax.pcolormesh(times, freqs, S, shading="auto", cmap="magma")
    ax.set_ylabel("Freq (Hz)")


def _render_audio_waveform(
    ax: matplotlib.axes.Axes,
    series: UniformSeries,
    channel: int,
    context: TrackContext,
) -> None:
    """Render one channel of an audio waveform as a line plot."""
    samples = series.samples if series.channel_shape == () else series.samples[channel]
    times = series.sample_times()
    ax.plot(times, samples, color=context.style.get("color", "#1f77b4"), linewidth=0.5)
    ax.set_ylabel("Amplitude")
    ax.set_xlim(context.t_start, context.t_end)


def render_audio(series: Any, context: TrackContext) -> RenderedTrack:
    """Render ``UniformSeries`` as waveform (one row per selected channel)."""
    if not isinstance(series, UniformSeries):
        raise TypeError(f"'audio' renderer expects UniformSeries, got {type(series).__name__}")
    channel = context.style.get("_channel", 0)
    _render_audio_waveform(context.ax, series, channel, context)
    return RenderedTrack(ax=context.ax, legend_handles=[])


def render_audio_spectrogram(series: Any, context: TrackContext) -> RenderedTrack:
    """Render a pre-computed spectrogram ``UniformSeries``."""
    if not isinstance(series, UniformSeries):
        raise TypeError(
            f"'audio_spectrogram' renderer expects UniformSeries, got {type(series).__name__}"
        )
    _render_spectrogram(context.ax, series)
    return RenderedTrack(ax=context.ax, legend_handles=[])


def render_uniform_fallback(series: Any, context: TrackContext) -> RenderedTrack:
    """Generic ``UniformSeries`` renderer: line plot of samples."""
    if not isinstance(series, UniformSeries):
        raise TypeError(
            f"'UniformSeries' renderer expects UniformSeries, got {type(series).__name__}"
        )
    times = series.sample_times()
    samples = series.samples
    ax = context.ax
    handles = []
    if samples.ndim == 1:
        (line,) = ax.plot(times, samples)
        handles.append(line)
    else:
        n_ch = samples.shape[0] if samples.ndim > 1 else 1
        for ch in range(n_ch):
            (line,) = ax.plot(times, samples[ch], label=f"ch{ch}")
            handles.append(line)
        ax.legend(handles=handles, loc="upper right")
    ax.set_xlim(context.t_start, context.t_end)
    return RenderedTrack(ax=ax, legend_handles=handles)


def render_event_fallback(series: Any, context: TrackContext) -> RenderedTrack:
    """Generic ``EventSeries`` renderer: line plot or rug plot."""
    if not isinstance(series, EventSeries):
        raise TypeError(f"'EventSeries' renderer expects EventSeries, got {type(series).__name__}")
    ax = context.ax
    handles = []
    if series.values is None:
        # Rug plot.
        for t in series.abs_timestamps:
            ax.axvline(t, color="gray", alpha=0.5)
    else:
        times = series.abs_timestamps
        values = np.asarray(series.values)
        if values.ndim == 1:
            (line,) = ax.plot(times, values)
            handles.append(line)
        else:
            n_rows = values.shape[0]
            colors = plt.cm.tab10(np.linspace(0, 1, max(n_rows, 10)))
            for r in range(n_rows):
                (line,) = ax.plot(times, values[r], color=colors[r], label=f"row{r}")
                handles.append(line)
            ax.legend(handles=handles, loc="upper right")
    ax.set_xlim(context.t_start, context.t_end)
    return RenderedTrack(ax=ax, legend_handles=handles)


def render_segment_fallback(series: Any, context: TrackContext) -> RenderedTrack:
    """Generic ``SegmentSeries`` renderer: colored spans."""
    if not isinstance(series, SegmentSeries):
        raise TypeError(
            f"'SegmentSeries' renderer expects SegmentSeries, got {type(series).__name__}"
        )
    ax = context.ax
    starts = series.abs_starts
    ends = series.abs_ends
    values = series.values
    if values is not None and values.ndim > 0 and values.shape[0] == 1:
        scalar = np.asarray(values).ravel()
        norm = plt.Normalize(scalar.min(), scalar.max())
        cmap = plt.cm.viridis
        for i, (s, e) in enumerate(zip(starts, ends)):
            ax.axvspan(s, e, color=cmap(norm(scalar[i])), alpha=0.4)
    else:
        for s, e in zip(starts, ends):
            ax.axvspan(s, e, color="gray", alpha=0.3)
    ax.set_xlim(context.t_start, context.t_end)
    return RenderedTrack(ax=ax, legend_handles=[])


def render_rps(series: Any, context: TrackContext) -> RenderedTrack:
    """Render ``EventSeries`` RPS with rotor colors."""
    if not isinstance(series, EventSeries):
        raise TypeError(f"'rps' renderer expects EventSeries, got {type(series).__name__}")
    ax = context.ax
    frame = context.style.get("_frame")
    if frame is not None and "audio" in frame:
        audio_us = frame["audio"]
        if isinstance(audio_us, UniformSeries):
            sample_audio = np.asarray(audio_us.samples)
            if sample_audio.ndim > 1:
                sample_audio = sample_audio[0]
            sr = audio_us.sr
            n_frames = len(sample_audio) // HOP + 1
            frame_times = np.arange(n_frames) * HOP / sr + audio_us.t_start + N_FFT / sr / 2
            gt = series.interpolate(frame_times)
        else:
            frame_times = series.abs_timestamps
            gt = series.values
    else:
        frame_times = series.abs_timestamps
        gt = series.values

    if gt is None:
        raise ValueError("'rps' renderer requires EventSeries with values")

    handles = []
    n_rotors = min(gt.shape[0], len(ROTOR_COLORS))
    for r in range(n_rotors):
        (line,) = ax.plot(
            frame_times,
            gt[r],
            color=ROTOR_COLORS[r],
            linewidth=2,
            label=f"Rotor {r + 1}",
        )
        handles.append(line)
    ax.set_ylabel("RPS")
    ax.set_xlim(context.t_start, context.t_end)
    ax.legend(handles=handles, loc="upper right", ncol=n_rotors, fontsize=8)
    ax.grid(True, alpha=0.3)
    return RenderedTrack(ax=ax, legend_handles=handles)


def render_salience(series: Any, context: TrackContext) -> RenderedTrack:
    """Render a model salience map (heatmap) with RPS overlays.

    Expects a ``UniformSeries`` built by :func:`make_salience_series`: samples
    ``(n_bins, T)`` in ``[0, 1]`` with a ``"plot.freqs"`` tag for the y-axis.
    The ground-truth RPS (from the frame's ``"rps"`` track, if present) is
    overlaid as dotted lines, and the model's tracked RPS prediction (from the
    ``"plot.rps_pred"`` tag, if present) as solid lines — so one can read how
    the salience peaks are tracked into a per-rotor trajectory.
    """
    if not isinstance(series, UniformSeries):
        raise TypeError(f"'salience' renderer expects UniformSeries, got {type(series).__name__}")
    S = series.samples
    if S.ndim != 2:
        raise ValueError(f"Salience series must be 2-D (n_bins, time), got {S.shape}")
    freqs = series.tags.get("plot.freqs")
    if freqs is None:
        raise ValueError("salience series needs a 'plot.freqs' tag (bin centre frequencies)")
    freqs = np.asarray(freqs, dtype=np.float64)

    ax = context.ax
    times = series.sample_times()
    vmax = context.style.get("salience_vmax", 1.0)
    if vmax == "auto":
        vmax = float(np.percentile(S, 99.5)) or 1.0
    mesh = ax.pcolormesh(times, freqs, S, shading="auto", cmap="magma", vmin=0.0, vmax=vmax)
    ax.set_yscale("log")
    ax.set_ylim(freqs[0], freqs[-1])
    ax.set_ylabel("Freq (Hz)")
    if context.style.get("salience_colorbar", True):
        ax.figure.colorbar(mesh, ax=ax, pad=0.01, fraction=0.025, label="salience")

    handles = []
    # Ground-truth RPS (dotted) — from the frame's rps track, on the salience grid.
    frame = context.style.get("_frame")
    if frame is not None and "rps" in frame:
        rps_track = frame["rps"]
        if isinstance(rps_track, EventSeries) and rps_track.values is not None:
            gt = np.asarray(rps_track.interpolate(times))
            for r in range(min(gt.shape[0], len(ROTOR_COLORS))):
                (line,) = ax.plot(
                    times, gt[r], color=ROTOR_COLORS[r], linewidth=1.4, linestyle=":",
                    alpha=0.9, label=f"GT R{r + 1}",
                )
                handles.append(line)

    # Predicted (tracked) RPS (solid) — stretched to the salience time span.
    rps_pred = series.tags.get("plot.rps_pred")
    if rps_pred is not None:
        rps_pred = np.asarray(rps_pred)
        pred_times = np.linspace(times[0], times[-1], rps_pred.shape[-1])
        for r in range(min(rps_pred.shape[0], len(ROTOR_COLORS))):
            (line,) = ax.plot(
                pred_times, rps_pred[r], color=ROTOR_COLORS[r], linewidth=1.6,
                alpha=0.95, label=f"pred R{r + 1}",
            )
            handles.append(line)

    if handles:
        ax.legend(handles=handles, loc="upper right", ncol=2, fontsize=6)
    ax.set_xlim(context.t_start, context.t_end)
    return RenderedTrack(ax=ax, legend_handles=handles)


# ---------------------------------------------------------------------------
# Register defaults
# ---------------------------------------------------------------------------

register_renderer("audio", render_audio)
register_renderer("audio_spectrogram", render_audio_spectrogram)
register_renderer("UniformSeries", render_uniform_fallback)
register_renderer("EventSeries", render_event_fallback)
register_renderer("SegmentSeries", render_segment_fallback)
register_renderer("rps", render_rps)
register_renderer("salience", render_salience)
