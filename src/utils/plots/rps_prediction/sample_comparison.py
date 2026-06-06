# src/utils/plots/rps_prediction/sample_comparison.py
"""Per-sample comparison: spectrogram + GT + per-model prediction rows."""
from __future__ import annotations

from typing import Any, Iterable

import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchaudio

from utils.data import TimeFrame

from tasks.rps_prediction import RPSPredictor, SR_AUDIO, N_FFT, HOP

ROTOR_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]


def plot_sample_comparison(
    *,
    sample: TimeFrame | None = None,
    sample_path: str | None = None,
    models: list[tuple[str, RPSPredictor]] | None = None,
    ax: Any = None,
    figsize: tuple[float, float] = (16, 12),
    **style,
) -> matplotlib.figure.Figure:
    """Generate a multi-row figure: spectrogram + GT + per-model predictions.

    Parameters
    ----------
    sample : TimeFrame | None
        Pre-loaded sample (from ``load_input_set``).
    sample_path : str | None
        Path to a sample directory (``sample_XXXXX/`` with ``mixture.wav``
        and ``rps.npy``).  Used if ``sample`` is None.
    models : list of (name, predictor) | None
        Each predictor's ``predict(audio, sr)`` is called and its output
        plotted on a separate row.
    ax : ignored (uses own figure).
    figsize : tuple

    Returns
    -------
    matplotlib.figure.Figure
    """
    if sample is None and sample_path is None:
        raise ValueError("One of sample or sample_path is required")

    if sample is None:
        sample = _load_sample(sample_path)

    audio_us = sample["audio"]
    rps_es = sample["rps"]
    audio = np.asarray(audio_us.samples, dtype=np.float32)
    sr = audio_us.sr
    dur = len(audio) / sr

    # --- GT RPS on STFT frame grid ---
    n_frames = len(audio) // HOP + 1
    frame_times = np.arange(n_frames) * HOP / sr
    gt = rps_es.interpolate(frame_times).T  # (4, n_frames)

    # --- predictions ---
    preds: dict[str, np.ndarray] = {}
    if models:
        for name, predictor in models:
            p = predictor.predict(audio, sr=sr)
            T = min(p.shape[-1], n_frames)
            preds[name] = p[:, :T]

    n_model_rows = len(preds)
    n_rows = 2 + n_model_rows  # spectrogram, GT, then one per model
    height_ratios = [1.2] + [1.0] * (n_rows - 1)

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(n_rows, 1, height_ratios=height_ratios, hspace=0.3)

    # --- row 1: spectrogram ---
    ax_spec = fig.add_subplot(gs[0])
    _plot_spectrogram(ax_spec, audio, sr, dur)

    # --- row 2: GT ---
    ax_gt = fig.add_subplot(gs[1], sharex=ax_spec)
    sid = sample.tags.get("id", "")
    for r, color in enumerate(ROTOR_COLORS):
        ax_gt.plot(frame_times, gt[r], color=color, linewidth=2, label=f"Rotor {r+1}")
    ax_gt.set_ylabel("RPS")
    ax_gt.set_title(f"Ground Truth — {sid}")
    ax_gt.legend(loc="upper right", ncol=4, fontsize=8)
    ax_gt.grid(True, alpha=0.3)

    # --- rows 3+: model predictions ---
    for idx, (model_name, pred) in enumerate(preds.items()):
        ax_pred = fig.add_subplot(gs[2 + idx], sharex=ax_spec)
        T_pred = pred.shape[-1]
        t_pred = np.arange(T_pred) * HOP / sr

        # GT overlay (dotted)
        gt_crop = gt[:, :T_pred]
        for r, color in enumerate(ROTOR_COLORS):
            ax_pred.plot(t_pred, gt_crop[r], color=color, linewidth=1, linestyle=":", alpha=0.4)
            ax_pred.plot(t_pred, pred[r], color=color, linewidth=2)

        mae = float(np.mean(np.abs(pred - gt_crop)))
        ax_pred.set_title(f"{model_name} — MAE={mae:.2f}")
        ax_pred.set_ylabel("RPS")
        ax_pred.grid(True, alpha=0.3)
        if idx == n_model_rows - 1:
            ax_pred.set_xlabel("Time (s)")

    plt.tight_layout()
    return fig


def _plot_spectrogram(ax, audio: np.ndarray, sr: float, dur: float) -> None:
    """Draw log-magnitude spectrogram up to 4 kHz."""
    window = torch.hann_window(N_FFT)
    X = torch.stft(
        torch.from_numpy(audio).float(), n_fft=N_FFT, hop_length=HOP,
        window=window, return_complex=True,
    )
    S = torch.abs(X).numpy()
    times = np.linspace(0, dur, S.shape[1])
    freqs = np.linspace(0, sr / 2, S.shape[0])
    im = ax.pcolormesh(times, freqs, 20 * np.log10(S + 1e-8), shading="auto", cmap="magma")
    ax.set_ylabel("Freq (Hz)")
    ax.set_ylim(0, 4000)
    ax.set_title("Input Spectrogram")
    plt.colorbar(im, ax=ax, label="dB")


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
        timestamps=motor_times, values=rps_raw.T,
        t_start=0.0, t_end=dur_s,
    )
    audio_us = UniformSeries.from_samples(audio, sr=float(file_sr), t_start=0.0)
    return TimeFrame.from_tracks(
        {"audio": audio_us, "rps": rps_es},
        tags={"id": d.name},
    )
