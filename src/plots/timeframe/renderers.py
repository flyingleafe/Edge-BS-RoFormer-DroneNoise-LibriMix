"""Default renderers and converters for ``plot_timeframe``."""

from __future__ import annotations

from typing import Any

import matplotlib
import matplotlib.axes
import matplotlib.colors
import numpy as np
import tdseries as td
import torch

from tasks.rps_prediction import HOP, N_FFT, align_rps_to_gt

from .registry import PlotTrack, RenderedTrack, TrackContext, register_renderer

ROTOR_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

# ---------------------------------------------------------------------------
# Channel selection helper
# ---------------------------------------------------------------------------


def _audio_channel_dim(series: td.Series) -> str | None:
    """Return the audio channel dim name (``"mic"``/``"channel"``), or ``None`` if mono."""
    extra = [d for d in series.dims if d != "time"]
    if not extra:
        return None
    if len(extra) > 1 or extra[0] not in ("mic", "channel"):
        raise ValueError(
            "'audio' renderer expects at most one extra dim named 'mic' or 'channel', "
            f"got dims={series.dims}"
        )
    return extra[0]


def _grid_index(series: td.Series) -> td.GridIndex:
    """Return ``series.tindex`` narrowed to ``GridIndex`` (raises otherwise)."""
    tindex = series.tindex
    if not isinstance(tindex, td.GridIndex):
        raise TypeError(f"expected a GridIndex time axis, got {type(tindex).__name__}")
    return tindex


def _stamp_index(series: td.Series) -> td.StampIndex:
    """Return ``series.tindex`` narrowed to ``StampIndex`` (raises otherwise)."""
    tindex = series.tindex
    if not isinstance(tindex, td.StampIndex):
        raise TypeError(f"expected a StampIndex time axis, got {type(tindex).__name__}")
    return tindex


def _span_index(series: td.Series) -> td.SpanIndex:
    """Return ``series.tindex`` narrowed to ``SpanIndex`` (raises otherwise)."""
    tindex = series.tindex
    if not isinstance(tindex, td.SpanIndex):
        raise TypeError(f"expected a SpanIndex time axis, got {type(tindex).__name__}")
    return tindex


def _select_channels(series: td.Series, channel: int | list[int] | str | None) -> list[int]:
    """Return validated channel indices for an audio-like ``Series``."""
    dim = _audio_channel_dim(series)
    n_ch = series.dim_size(dim) if dim is not None else 1
    if channel is None:
        return [0] if dim is None else list(range(n_ch))
    if isinstance(channel, str):
        if channel != "all":
            raise ValueError(f"Unsupported value 'channel={channel}'")
        return [0] if dim is None else list(range(n_ch))
    if isinstance(channel, int):
        channel = [channel]
    channel = list(channel)
    if dim is None:
        for ch in channel:
            if ch != 0:
                raise ValueError(f"Mono audio only supports channel 0, got {ch}")
    else:
        for ch in channel:
            if not (0 <= ch < n_ch):
                raise ValueError(f"Channel {ch} out of range (0..{n_ch - 1})")
    return channel


# ---------------------------------------------------------------------------
# Spectrogram converters
# ---------------------------------------------------------------------------


def make_spectrogram_series(
    audio: td.Series,
    *,
    n_fft: int = N_FFT,
    hop_length: int = HOP,
    fmax: float | None = None,
    log: bool = True,
) -> PlotTrack:
    """Return a time-frequency ``PlotTrack`` for an audio track.

    The returned track renders with the ``"audio_spectrogram"`` renderer.
    Its series has shape ``(n_freq_bins, n_time_frames)`` (mono input) or
    ``(n_channels, n_freq_bins, n_time_frames)``, and its sample rate is the
    exact STFT frame rate ``audio.tindex.rate / hop_length``.
    """
    samples = np.asarray(audio.data, dtype=np.float32)
    if samples.ndim not in (1, 2):
        raise ValueError(f"Unsupported audio shape {samples.shape} for spectrogram")
    extra_dims = tuple(d for d in audio.dims if d != "time")
    waveform = torch.from_numpy(samples).float()

    window = torch.hann_window(n_fft)
    X = torch.stft(
        waveform,
        n_fft=n_fft,
        hop_length=hop_length,
        window=window,
        return_complex=True,
    )
    S = torch.abs(X).numpy()

    audio_rate = _grid_index(audio).rate
    nyquist = float(audio_rate) / 2.0
    freq_max_hz = min(fmax, nyquist) if fmax is not None else nyquist
    if fmax is not None:
        max_bin = min(int(fmax / nyquist * S.shape[-2]), S.shape[-2])
        S = S[..., :max_bin, :]
    if log:
        S = 20 * np.log10(S + 1e-8)

    frame_rate = audio_rate / hop_length
    t0 = float(audio.t_start) + n_fft / (2 * float(audio_rate))
    series = td.uniform(S, frame_rate, dims=(*extra_dims, "freq", "time"), t_start=t0)
    return PlotTrack(
        series=series, renderer="audio_spectrogram", hints={"freq_max_hz": freq_max_hz}
    )


def make_log_spectrogram_series(
    audio: td.Series,
    *,
    n_fft: int = N_FFT,
    hop_length: int = HOP,
    fmax: float | None = 4000,
) -> PlotTrack:
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
    frame_sr: td.SampleRate,
    t_start: float = 0.0,
    rps_pred: Any | None = None,
    title: str | None = None,
) -> PlotTrack:
    """Wrap a model salience map as a ``PlotTrack`` for the ``"salience"`` renderer.

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
        Salience frame rate. Pass an exact rate (int, ``Fraction``, or
        ``(num, den)`` tuple) — e.g. ``Fraction(model.spec_sr, model.spec_hop)``
        — never a rounded float.
    t_start
        Absolute start time (s) of the first salience frame.
    rps_pred
        Optional ``(n_rotors, T_pred)`` tracked-RPS trajectory to overlay as
        solid lines on the heatmap. Its own time axis is stretched to span the
        salience duration, matching how the tracker resamples to the STFT grid.
    title
        Optional subplot title (defaults to the track name; pass ``None`` to
        use the default, or set the hint after construction to suppress it).
    """
    samples = np.asarray(salience.detach().cpu() if hasattr(salience, "detach") else salience)
    samples = samples.astype(np.float32)
    if samples.ndim != 2:
        raise ValueError(f"salience must be 2-D (n_bins, T), got {samples.shape}")
    series = td.uniform(samples, frame_sr, dims=("freq", "time"), t_start=t_start)

    hints: dict[str, Any] = {"freqs": np.asarray(freqs, dtype=np.float64)}
    if rps_pred is not None:
        rp = rps_pred.detach().cpu() if hasattr(rps_pred, "detach") else rps_pred
        hints["rps_pred"] = np.asarray(rp, dtype=np.float64)
    if title is not None:
        hints["title"] = title
    return PlotTrack(series=series, renderer="salience", hints=hints)


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------


def render_audio(series: td.Series, context: TrackContext) -> RenderedTrack:
    """Render one channel of a (possibly multi-channel) audio ``Series`` as a line plot."""
    channel = context.style.get("_channel", 0)
    dim = _audio_channel_dim(series)
    sub = series if dim is None else series.slice[dim, channel]
    times = _grid_index(sub).sample_times()
    samples = sub.data
    ax = context.ax
    ax.plot(times, samples, color=context.style.get("color", "#1f77b4"), linewidth=0.5)
    ax.set_ylabel("Amplitude")
    ax.set_xlim(context.t_start, context.t_end)
    return RenderedTrack(ax=ax, legend_handles=[])


def render_audio_spectrogram(series: td.Series, context: TrackContext) -> RenderedTrack:
    """Render a pre-computed spectrogram ``Series`` (from :func:`make_spectrogram_series`)."""
    S = series.data
    if S is None or np.ndim(S) != 2:
        shape = None if S is None else np.shape(S)
        raise ValueError(f"'audio_spectrogram' renderer expects 2-D (freq, time) data, got {shape}")
    times = _grid_index(series).sample_times()
    hints = context.style.get("_hints", {})
    freq_max_hz = hints.get("freq_max_hz", float(_grid_index(series).rate) / 2.0)
    freqs = np.linspace(0, float(freq_max_hz), S.shape[-2])
    ax = context.ax
    ax.pcolormesh(times, freqs, S, shading="auto", cmap="magma")
    ax.set_ylabel("Freq (Hz)")
    return RenderedTrack(ax=ax, legend_handles=[])


def render_waveform(series: td.Series, context: TrackContext) -> RenderedTrack:
    """Generic ``GridIndex`` renderer: line plot, one overlaid line per non-time row."""
    times = _grid_index(series).sample_times()
    samples = series.data
    ax = context.ax
    handles = []
    if samples.ndim == 1:
        (line,) = ax.plot(times, samples)
        handles.append(line)
    else:
        n_rows = samples.shape[0]
        for row in range(n_rows):
            (line,) = ax.plot(times, samples[row], label=f"ch{row}")
            handles.append(line)
        ax.legend(handles=handles, loc="upper right")
    ax.set_xlim(context.t_start, context.t_end)
    return RenderedTrack(ax=ax, legend_handles=handles)


def _rps_grid_from_audio(
    series: td.Series, frame: td.Frame
) -> tuple[np.ndarray, np.ndarray] | None:
    """If ``frame`` has an ``"audio"`` GridIndex entry, interpolate ``series`` onto its STFT grid."""
    if "audio" not in frame:
        return None
    audio_series = frame["audio"]
    if not (isinstance(audio_series, td.Series) and isinstance(audio_series.tindex, td.GridIndex)):
        return None
    sr = audio_series.tindex.sr
    n_frames = audio_series.dim_size("time") // HOP + 1
    frame_times = np.arange(n_frames) * HOP / sr + audio_series.t_start + N_FFT / sr / 2
    gt = np.asarray(series.interpolate(frame_times))
    return frame_times, gt


def render_rps(series: td.Series, context: TrackContext) -> RenderedTrack:
    """Render ``StampIndex`` RPS with rotor colors, or a rug plot if it has no values."""
    ax = context.ax
    if series.data is None:
        for t in _stamp_index(series).abs_stamps:
            ax.axvline(float(t), color="gray", alpha=0.5)
        ax.set_xlim(context.t_start, context.t_end)
        return RenderedTrack(ax=ax, legend_handles=[])

    frame = context.style.get("_frame")
    grid = _rps_grid_from_audio(series, frame) if isinstance(frame, td.Frame) else None
    if grid is not None:
        frame_times, gt = grid
    else:
        frame_times = np.asarray(_stamp_index(series).abs_stamps)
        gt = np.asarray(series.data)

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


def render_spans(series: td.Series, context: TrackContext) -> RenderedTrack:
    """Render ``SpanIndex`` as colored (scalar value) or grey (no value) axvspans."""
    ax = context.ax
    span_index = _span_index(series)
    starts = span_index.abs_starts
    ends = span_index.abs_ends
    values = series.data
    if values is not None and np.ndim(values) == 1:
        scalar = np.asarray(values, dtype=np.float64).ravel()
        norm = matplotlib.colors.Normalize(float(scalar.min()), float(scalar.max()))
        cmap = matplotlib.colormaps["viridis"]
        for i, (s, e) in enumerate(zip(starts, ends)):
            ax.axvspan(float(s), float(e), color=cmap(norm(scalar[i])), alpha=0.4)
    else:
        for s, e in zip(starts, ends):
            ax.axvspan(float(s), float(e), color="gray", alpha=0.3)
    ax.set_xlim(context.t_start, context.t_end)
    return RenderedTrack(ax=ax, legend_handles=[])


def render_salience(series: td.Series, context: TrackContext) -> RenderedTrack:
    """Render a model salience map (heatmap) with RPS overlays.

    Expects a ``Series`` built by :func:`make_salience_series`: samples
    ``(n_bins, T)`` in ``[0, 1]`` with a ``"freqs"`` hint for the y-axis.
    The ground-truth RPS (from the frame's ``"rps"`` track, if present) is
    overlaid as dotted lines, and the model's tracked RPS prediction (from the
    ``"rps_pred"`` hint, if present) as solid lines — so one can read how
    the salience peaks are tracked into a per-rotor trajectory.
    """
    S = series.data
    if S is None or np.ndim(S) != 2:
        shape = None if S is None else np.shape(S)
        raise ValueError(f"Salience series must be 2-D (n_bins, time), got {shape}")
    hints = context.style.get("_hints", {})
    freqs = hints.get("freqs")
    if freqs is None:
        raise ValueError("salience track needs a 'freqs' hint (bin centre frequencies)")
    freqs = np.asarray(freqs, dtype=np.float64)

    ax = context.ax
    times = _grid_index(series).sample_times()
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
    gt = None
    frame = context.style.get("_frame")
    if frame is not None and "rps" in frame:
        rps_track = frame["rps"]
        if isinstance(rps_track, td.Series) and rps_track.data is not None:
            gt = np.asarray(rps_track.interpolate(times))
            for r in range(min(gt.shape[0], len(ROTOR_COLORS))):
                (line,) = ax.plot(
                    times,
                    gt[r],
                    color=ROTOR_COLORS[r],
                    linewidth=1.4,
                    linestyle=":",
                    alpha=0.9,
                    label=f"GT R{r + 1}",
                )
                handles.append(line)

    # Predicted (tracked) RPS (solid) — stretched to the salience time span.
    rps_pred = hints.get("rps_pred")
    if rps_pred is not None:
        rps_pred = np.asarray(rps_pred)
        # PIT-trained predictors emit rotors in arbitrary order; align to GT so
        # colours mean a fixed rotor identity (matches the evaluation matching).
        if gt is not None:
            rps_pred = align_rps_to_gt(rps_pred, gt)
        pred_times = np.linspace(times[0], times[-1], rps_pred.shape[-1])
        for r in range(min(rps_pred.shape[0], len(ROTOR_COLORS))):
            (line,) = ax.plot(
                pred_times,
                rps_pred[r],
                color=ROTOR_COLORS[r],
                linewidth=1.6,
                alpha=0.95,
                label=f"pred R{r + 1}",
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
register_renderer("waveform", render_waveform)
register_renderer("audio_spectrogram", render_audio_spectrogram)
register_renderer("rps", render_rps)
register_renderer("spans", render_spans)
register_renderer("salience", render_salience)
