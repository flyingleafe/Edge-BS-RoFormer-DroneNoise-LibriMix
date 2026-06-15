"""Per-sample comparison: spectrogram + GT + per-model prediction rows."""

from __future__ import annotations

from typing import Any, cast

import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchaudio

from tasks.rps_prediction import HOP, N_FFT, align_rps_to_gt
from utils.data import EventSeries, TimeFrame, UniformSeries
from utils.plots.timeframe import plot_timeframe

ROTOR_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]


def plot_sample_comparison(
    *,
    sample: TimeFrame | None = None,
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

    audio_us = cast(UniformSeries, sample["audio"])

    # Convert preds into overlay tracks. PIT-trained predictors emit rotors in
    # arbitrary order, so align each prediction to the GT rotor order first.
    preds = _align_preds_to_gt(sample, preds)
    for name, pred in preds.items():
        pred_us = _prediction_to_uniform(pred, audio_us)
        sample = sample.with_track(f"pred_{name}", pred_us)

    # Build ordered track list.
    tracks = ["audio"]
    if show_separate_gt and "rps" in sample:
        tracks.append("rps")
    tracks.extend(name for name in sample if name.startswith("pred_"))

    return plot_timeframe(
        sample,
        tracks=tracks,
        channel=channel,
        figsize=figsize,
        **style,
    )


def _align_preds_to_gt(sample: TimeFrame, preds: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Reorder each prediction's rotor rows to the GT order (PIT match).

    Returns ``preds`` unchanged if the sample has no GT ``rps`` track.
    """
    rps = sample["rps"] if "rps" in sample else None  # noqa: SIM401 (TimeFrame has no .get)
    if not isinstance(rps, EventSeries) or rps.values is None:
        return preds
    gt = np.asarray(rps.values, dtype=np.float64)  # resampled to pred grid inside
    return {name: align_rps_to_gt(np.asarray(pred), gt) for name, pred in preds.items()}


def _prediction_to_uniform(pred: np.ndarray, audio_us: UniformSeries) -> UniformSeries:
    """Convert a prediction array to a ``UniformSeries`` aligned with audio."""
    T_pred = pred.shape[-1]
    sr = T_pred / audio_us.duration if audio_us.duration > 0 else 1.0
    return UniformSeries.from_samples(
        pred,
        sr=sr,
        t_start=audio_us.t_start,
        tags={"plot.title": "prediction"},
    )


def _plot_two_columns(
    sample: TimeFrame,
    preds: dict[str, np.ndarray],
    channel: int | list[int] | str | None,
    figsize: tuple[float, float],
    **style,
) -> matplotlib.figure.Figure:
    """Legacy two-column layout used by ``plot_sample_comparison``."""
    audio_us = cast(UniformSeries, sample["audio"])
    rps_es = sample["rps"]
    audio = np.asarray(audio_us.samples, dtype=np.float32)
    sr = audio_us.sr

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
    frame_times = np.arange(n_frames) * HOP / sr + rps_es.t_start + N_FFT / sr / 2
    gt = rps_es.interpolate(frame_times)

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
        _plot_spectrogram(ax_spec, audio_by_ch[ch], sr, audio_us.t_start, dur)
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
            pred_times = np.linspace(0.0, dur, T_pred) + audio_us.t_start
            # Align rotor order to GT (PIT match) before plotting / MAE.
            pred = align_rps_to_gt(pred, np.asarray(rps_es.interpolate(pred_times)))
            for r, color in enumerate(ROTOR_COLORS):
                ax_rps.plot(
                    pred_times,
                    pred[r],
                    color=color,
                    linewidth=2,
                    alpha=0.9,
                    label=f"{model_name} R{r + 1}" if idx == 0 else "",
                )

            gt_interp = rps_es.interpolate(pred_times)
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


def _plot_spectrogram(ax, audio: np.ndarray, sr: float, t_start: float, dur: float) -> None:
    """Draw log-magnitude spectrogram up to 4 kHz."""
    window = torch.hann_window(N_FFT)
    X = torch.stft(
        torch.from_numpy(audio).float(),
        n_fft=N_FFT,
        hop_length=HOP,
        window=window,
        return_complex=True,
    )
    S = torch.abs(X).numpy()
    times = np.linspace(t_start, t_start + dur, S.shape[-1])
    freqs = np.linspace(0, sr / 2, S.shape[0])
    ax.pcolormesh(times, freqs, 20 * np.log10(S + 1e-8), shading="auto", cmap="magma")
    ax.set_ylabel("Freq (Hz)")
    ax.set_title("Input Spectrogram")


def _load_sample(path: str) -> TimeFrame:
    from pathlib import Path

    d = Path(path)
    waveform, file_sr = torchaudio.load(str(d / "mixture.wav"))
    audio = waveform.squeeze(0).numpy().astype(np.float32)
    rps_raw = np.load(str(d / "rps.npy")).astype(np.float64)
    M = rps_raw.shape[1]
    dur_s = waveform.shape[-1] / file_sr
    motor_sr = M / dur_s if dur_s > 0 else 1000.0
    motor_times = np.arange(M) / motor_sr

    rps_es = EventSeries.from_events(
        timestamps=motor_times,
        values=rps_raw,
        t_start=0.0,
        t_end=dur_s,
    )
    audio_us = UniformSeries.from_samples(audio, sr=float(file_sr), t_start=0.0)
    return TimeFrame.from_tracks(
        {"audio": audio_us, "rps": rps_es},
        tags={"id": d.name},
    )
