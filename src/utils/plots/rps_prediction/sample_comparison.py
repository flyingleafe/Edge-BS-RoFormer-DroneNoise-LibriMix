# src/utils/plots/rps_prediction/sample_comparison.py
"""Per-sample comparison: spectrogram + GT + per-model prediction rows."""

from __future__ import annotations

from typing import Any, cast

import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchaudio

from tasks.rps_prediction import HOP, N_FFT
from utils.data import EventSeries, TimeFrame, UniformSeries

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
    **style,
) -> matplotlib.figure.Figure:
    """Generate a multi-row figure: spectrograms + GT + per-model predictions.

    Parameters
    ----------
    sample : TimeFrame | None
        Pre-loaded sample (from ``load_input_set``).
    sample_path : str | None
        Path to a sample directory (``sample_XXXXX/`` with ``mixture.wav``
        and ``rps.npy``).  Used if ``sample`` is None.
    channel : int, list[int], or "all"
        Channel(s) to display spectrograms for.  Default ``[0]``.
        ``"all"`` selects every available channel.  Mono audio only
        accepts ``0``.
    models : list of (name, predictor) | None
        Each predictor's ``predict(audio, sr)`` is called on the *first*
        selected channel and its output plotted on a separate row.
    ax : ignored (uses own figure).
    figsize : tuple

    Returns
    -------
    matplotlib.figure.Figure
    """
    if sample is None and sample_path is None:
        raise ValueError("One of sample or sample_path is required")

    if sample is None:
        assert sample_path is not None
        sample = _load_sample(sample_path)

    if preds is None:
        preds = {}

    if show_separate_gt is None:
        show_separate_gt = len(preds) == 0

    audio_us = cast(UniformSeries, sample["audio"])
    rps_es = cast(EventSeries, sample["rps"])
    audio = np.asarray(audio_us.samples, dtype=np.float32)
    sr = audio_us.sr

    # --- channel selection ---
    if channel is None:
        channel = [0]
    elif isinstance(channel, str):
        if channel != "all":
            raise ValueError(f"Unsupported value 'channel={channel}")
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

    # Extract per-channel waveforms
    audio_by_ch: dict[int, np.ndarray] = {}
    for ch in channel:
        audio_by_ch[ch] = audio if audio.ndim == 1 else audio[ch, :]

    sample_audio = next(iter(audio_by_ch.values()))
    dur = len(sample_audio) / sr

    # --- GT RPS on STFT frame grid ---
    n_frames = len(sample_audio) // HOP + 1
    frame_times = np.arange(n_frames) * HOP / sr + rps_es.t_start + N_FFT / sr / 2
    gt = rps_es.interpolate(frame_times)  # (4, n_frames)

    n_spec_rows = len(channel)
    n_model_rows = len(preds)
    n_gt_rows = int(show_separate_gt)
    n_rows = n_spec_rows + n_gt_rows + n_model_rows  # specs, GT, preds
    height_ratios = [1.2] * n_spec_rows + [1.0] * (n_gt_rows + n_model_rows)

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(n_rows, 1, height_ratios=height_ratios, hspace=0.3)

    # --- spectrogram rows ---
    ax_first = None
    for idx, ch in enumerate(channel):
        ax_spec = fig.add_subplot(gs[idx], sharex=ax_first)
        if ax_first is None:
            ax_first = ax_spec
        _plot_spectrogram(ax_spec, audio_by_ch[ch], sr, audio_us.t_start, dur)
        ax_spec.yaxis.set_label_position("right")
        ax_spec.set_ylabel(f"ch{ch}")

    # --- GT row ---
    if show_separate_gt:
        ax_gt = fig.add_subplot(gs[n_spec_rows], sharex=ax_first)
        sid = sample.tags.get("id", "")
        for r, color in enumerate(ROTOR_COLORS):
            ax_gt.plot(frame_times, gt[r], color=color, linewidth=2, label=f"Rotor {r + 1}")
        ax_gt.set_ylabel("RPS")
        ax_gt.set_title(f"Ground Truth — {sid}")
        ax_gt.legend(loc="upper right", ncol=4, fontsize=8)
        ax_gt.grid(True, alpha=0.3)

    # --- model prediction rows ---
    for idx, (model_name, pred) in enumerate(preds.items()):
        ax_pred = fig.add_subplot(gs[n_spec_rows + n_gt_rows + idx], sharex=ax_first)
        T_pred = pred.shape[-1]
        pred_times = np.linspace(0.0, dur, T_pred) + audio_us.t_start
        for r, color in enumerate(ROTOR_COLORS):
            ax_pred.plot(
                frame_times,
                gt[r],
                color=color,
                linewidth=2,
                linestyle=":",
                alpha=0.8,
            )
            ax_pred.plot(pred_times, pred[r], color=color, linewidth=2)

        gt_interp = rps_es.interpolate(pred_times)
        mae = float(np.mean(np.abs(pred - gt_interp)))
        ax_pred.set_title(f"{model_name} — MAE={mae:.2f}")
        ax_pred.set_ylabel("RPS")
        ax_pred.grid(True, alpha=0.3)
        if idx == n_model_rows - 1:
            ax_pred.set_xlabel("Time (s)")

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
    im = ax.pcolormesh(times, freqs, 20 * np.log10(S + 1e-8), shading="auto", cmap="magma")
    ax.set_ylabel("Freq (Hz)")
    ax.set_title("Input Spectrogram")
    # plt.colorbar(im, ax=ax, label="dB")


def _load_sample(path: str) -> TimeFrame:
    from pathlib import Path

    from utils.data import EventSeries, UniformSeries

    d = Path(path)
    waveform, file_sr = torchaudio.load(str(d / "mixture.wav"))
    audio = waveform.squeeze(0).numpy().astype(np.float32)
    rps_raw = np.load(str(d / "rps.npy")).astype(np.float64)
    M = rps_raw.shape[1]
    dur_s = len(audio) / file_sr
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
