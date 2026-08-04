"""Per-sample comparison: spectrogram + GT + per-model prediction rows."""

from __future__ import annotations

from fractions import Fraction
from typing import Any, cast

import matplotlib.axes
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import tdseries as td
import torchaudio

from data_processing.frames import with_meta
from plots.timeframe import PlotTrack, plot_timeframe
from plots.timeframe.registry import TrackContext, get_renderer
from plots.timeframe.renderers import ROTOR_COLORS, make_spectrogram_series
from tasks.rps_prediction import HOP, N_FFT, align_rps_to_gt


def plot_sample_comparison(
    *,
    sample: td.Frame | None = None,
    sample_path: str | None = None,
    channel: int | list[int] | str | None = None,
    preds: dict[str, np.ndarray] | None = None,
    ax: Any = None,
    figsize: tuple[float, float] = (16, 12),
    show_separate_gt: bool | None = None,
    two_columns: bool = False,
    **style,
) -> matplotlib.figure.Figure:
    """Generate a multi-row figure: spectrograms + GT + per-model predictions.

    This is now a compatibility wrapper around :func:`plot_timeframe`.
    The legacy ``two_columns=True`` layout is preserved for visual parity.
    """
    if sample is None and sample_path is None:
        raise ValueError("One of sample or sample_path is required")

    if sample is None:
        assert sample_path is not None, "sample_path is required when sample is None"
        sample = _load_sample(sample_path)

    if preds is None:
        preds = {}

    if show_separate_gt is None:
        show_separate_gt = len(preds) == 0

    if two_columns:
        return _plot_two_columns(sample, preds, channel, figsize, **style)

    audio_series = cast(td.Series, sample["audio"])

    # PIT-trained predictors emit rotors in arbitrary order, so align each
    # prediction to the GT rotor order first.
    preds = _align_preds_to_gt(sample, preds)

    # Build the ordered track list. Predictions are not stored back into the
    # frame — they are plot-only, wrapped as PlotTracks and passed straight
    # to `plot_timeframe`.
    tracks: list[Any] = ["audio"]
    if show_separate_gt and "rps" in sample:
        tracks.append("rps")
    for name, pred in preds.items():
        tracks.append(_prediction_to_track(pred, audio_series, title=name))

    return plot_timeframe(
        sample,
        tracks=tracks,
        channel=channel,
        figsize=figsize,
        **style,
    )


def _align_preds_to_gt(sample: td.Frame, preds: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Reorder each prediction's rotor rows to the GT order (PIT match).

    Returns ``preds`` unchanged if the sample has no GT ``rps`` track.
    """
    if "rps" not in sample:
        return preds
    rps = sample["rps"]
    if not isinstance(rps, td.Series) or rps.data is None:
        return preds
    gt = np.asarray(rps.data, dtype=np.float64)  # resampled to pred grid inside
    return {name: align_rps_to_gt(np.asarray(pred), gt) for name, pred in preds.items()}


def _prediction_to_track(
    pred: np.ndarray, audio_series: td.Series, *, title: str | None = None
) -> PlotTrack:
    """Wrap a prediction array as a ``PlotTrack`` stretched across the audio span.

    The exact rate is derived from ``(n_frames * TICKS_PER_SECOND) /
    duration_ticks`` — an exact ``Fraction`` — rather than a rounded float, per
    the tdseries exact-rate rule.
    """
    if pred.ndim not in (1, 2):
        raise ValueError(f"Unsupported prediction shape {pred.shape}")
    T_pred = pred.shape[-1]
    dur_ticks = audio_series.duration_ticks
    rate = Fraction(T_pred * td.TICKS_PER_SECOND, dur_ticks) if dur_ticks > 0 else Fraction(T_pred)
    dims = ("rotor", "time") if pred.ndim == 2 else ("time",)
    series = td.uniform(pred, rate, dims=dims, t_start=audio_series.t_start)
    return PlotTrack(series=series, hints={"title": title})


def _plot_two_columns(
    sample: td.Frame,
    preds: dict[str, np.ndarray],
    channel: int | list[int] | str | None,
    figsize: tuple[float, float],
    **style,
) -> matplotlib.figure.Figure:
    """Legacy two-column layout used by ``plot_sample_comparison``."""
    audio_series = cast(td.Series, sample["audio"])
    rps_series = cast(td.Series, sample["rps"])
    audio_tindex = audio_series.tindex
    if not isinstance(audio_tindex, td.GridIndex):
        raise TypeError(f"'audio' entry must be a uniform (GridIndex) Series, got {audio_series}")
    audio = np.asarray(audio_series.data, dtype=np.float32)
    sr = audio_tindex.sr

    if channel is None:
        channel = [0]
    elif isinstance(channel, str):
        if channel != "all":
            raise ValueError(f"Unsupported value 'channel={channel}'")
        channel = [0] if audio.ndim == 1 else list(range(audio.shape[0]))
    elif isinstance(channel, int):
        channel = [channel]
    else:
        channel = list(channel)

    for ch in channel:
        if audio.ndim == 1:
            if ch != 0:
                raise ValueError(f"Mono audio only supports channel 0, got {ch}")
        else:
            if not (0 <= ch < audio.shape[0]):
                raise ValueError(f"Channel {ch} out of range (0..{audio.shape[0] - 1})")

    audio_by_ch: dict[int, np.ndarray] = {}
    for ch in channel:
        audio_by_ch[ch] = audio if audio.ndim == 1 else audio[ch, :]

    sample_audio = next(iter(audio_by_ch.values()))
    dur = len(sample_audio) / sr

    n_frames = len(sample_audio) // HOP + 1
    frame_times = np.arange(n_frames) * HOP / sr + rps_series.t_start + N_FFT / sr / 2
    gt = np.asarray(rps_series.interpolate(frame_times))

    _preds_by_ch: dict[int, dict[str, np.ndarray]] = {}
    _ch_keys = set()
    for name, pred in preds.items():
        if name.startswith("ch") and name[2:].isdigit():
            ch_idx = int(name[2:])
            _preds_by_ch.setdefault(ch_idx, {})[name] = pred
            _ch_keys.add(ch_idx)
    _use_per_ch = bool(_ch_keys) and all(ch in _ch_keys for ch in channel)

    fig = plt.figure(figsize=figsize)
    n_rows = len(channel)
    gs = fig.add_gridspec(n_rows, 2, height_ratios=[1.0] * n_rows, hspace=0.3, wspace=0.15)

    for idx, ch in enumerate(channel):
        ax_spec = fig.add_subplot(gs[idx, 0])
        _plot_spectrogram(ax_spec, audio_by_ch[ch], sr, audio_series.t_start, dur)
        ax_spec.set_ylabel(f"ch{ch}")
        if idx == 0:
            ax_spec.set_title("Input Spectrogram")

    for idx, ch in enumerate(channel):
        ax_rps = fig.add_subplot(gs[idx, 1])

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

        ch_preds = _preds_by_ch.get(ch, {}) if _use_per_ch else preds
        for model_name, pred in ch_preds.items():
            T_pred = pred.shape[-1]
            pred_times = np.linspace(0.0, dur, T_pred) + audio_series.t_start
            # Align rotor order to GT (PIT match) before plotting / MAE.
            pred = align_rps_to_gt(pred, np.asarray(rps_series.interpolate(pred_times)))
            for r, color in enumerate(ROTOR_COLORS):
                ax_rps.plot(
                    pred_times,
                    pred[r],
                    color=color,
                    linewidth=2,
                    alpha=0.9,
                    label=f"{model_name} R{r + 1}" if idx == 0 else "",
                )

            gt_interp = np.asarray(rps_series.interpolate(pred_times))
            mae = float(np.mean(np.abs(pred - gt_interp)))
            ax_rps.set_title(f"ch{ch} — MAE={mae:.2f}")

        ax_rps.set_ylabel("RPS")
        ax_rps.grid(True, alpha=0.3)
        if idx == 0:
            ax_rps.legend(loc="upper right", ncol=2, fontsize=6)
        if idx == n_rows - 1:
            ax_rps.set_xlabel("Time (s)")

    gs.tight_layout(fig)
    return fig


def _plot_spectrogram(
    ax: matplotlib.axes.Axes, audio: np.ndarray, sr: float, t_start: float, dur: float
) -> None:
    """Draw a log-magnitude spectrogram into ``ax``.

    Routed through the one shared spectrogram implementation
    (:func:`~plots.timeframe.renderers.make_spectrogram_series` + the
    ``"audio_spectrogram"`` renderer) — the manual ``torch.stft`` copy this
    function used to carry is gone.
    """
    series = td.uniform(np.asarray(audio, dtype=np.float32), sr, dims=("time",), t_start=t_start)
    track = make_spectrogram_series(series, n_fft=N_FFT, hop_length=HOP, log=True)
    ctx = TrackContext(
        ax=ax,
        name="spectrogram",
        t_start=t_start,
        t_end=t_start + dur,
        style={"_hints": track.hints},
    )
    get_renderer("audio_spectrogram")(track.series, ctx)
    ax.set_title("Input Spectrogram")


def _load_sample(path: str) -> td.Frame:
    from pathlib import Path

    d = Path(path)
    waveform, file_sr = torchaudio.load(str(d / "mixture.wav"))
    audio = waveform.squeeze(0).numpy().astype(np.float32)
    rps_raw = np.load(str(d / "rps.npy")).astype(np.float64)
    M = rps_raw.shape[1]
    dur_s = waveform.shape[-1] / file_sr
    motor_sr = M / dur_s if dur_s > 0 else 1000.0
    motor_times = np.arange(M) / motor_sr

    rps_series = td.events(motor_times, rps_raw, dims=("rotor", "time"), t_start=0.0, t_end=dur_s)
    audio_series = td.uniform(audio, file_sr, dims=("time",), t_start=0.0)
    frame = td.Frame({"audio": audio_series, "rps": rps_series})
    return with_meta(frame, id=d.name)
