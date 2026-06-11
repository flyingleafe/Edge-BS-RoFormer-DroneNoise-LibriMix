"""Horizontal slide-friendly comparison: spectrogram + RPS side-by-side."""

from __future__ import annotations

import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchaudio

from tasks.rps_prediction import HOP, N_FFT
from utils.data import TimeFrame
from utils.plots.rps_prediction.sample_comparison import ROTOR_COLORS


def plot_slide_comparison(
    *,
    sample: TimeFrame | None = None,
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
    sample : TimeFrame | None
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
        sample = _load_sample(sample_path)
    if preds is None:
        preds = {}
    if channels is None:
        channels = [0, 1]

    # Normalise preds to per-channel dict
    if preds and isinstance(next(iter(preds.values())), np.ndarray):
        preds = {ch: preds for ch in channels}

    audio_us = sample["audio"]
    rps_es = sample["rps"]
    audio = np.asarray(audio_us.samples, dtype=np.float32)
    sr = audio_us.sr
    dur = audio.shape[-1] / sr

    # GT on frame grid
    sample_audio = audio[0] if audio.ndim > 1 else audio
    n_frames = len(sample_audio) // HOP + 1
    frame_times = np.arange(n_frames) * HOP / sr + rps_es.t_start + N_FFT / sr / 2
    gt = rps_es.interpolate(frame_times)

    n_ch = len(channels)
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(n_ch, 2, hspace=0.35, wspace=0.15, height_ratios=[1.0] * n_ch)

    for idx, ch in enumerate(channels):
        ch_audio = audio[ch] if audio.ndim > 1 else audio

        # --- Left: spectrogram ---
        ax_spec = fig.add_subplot(gs[idx, 0])
        _plot_spectrogram(ax_spec, ch_audio, sr, audio_us.t_start, dur)
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
        ch_preds = preds.get(ch, {})
        for model_name, pred in ch_preds.items():
            T_pred = pred.shape[-1]
            pred_times = np.linspace(0.0, dur, T_pred) + audio_us.t_start
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


def _plot_spectrogram(ax, audio: np.ndarray, sr: float, t_start: float, dur: float) -> None:
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
    im = ax.pcolormesh(times, freqs, 20 * np.log10(S + 1e-8), shading="auto", cmap="magma")
    ax.set_ylabel("Freq (Hz)", fontsize=8)
    ax.set_ylim(0, 4000)


def _load_sample(path: str) -> TimeFrame:
    from pathlib import Path

    from utils.data import EventSeries, UniformSeries

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
