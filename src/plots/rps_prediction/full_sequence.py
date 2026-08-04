# src/plots/rps_prediction/full_sequence.py
"""3-panel full-sequence plot: spectrogram + GT-vs-pred + per-frame MSE."""

from __future__ import annotations

from typing import Any

import matplotlib.figure
import matplotlib.lines
import matplotlib.pyplot as plt
import numpy as np
import torch

from plots.timeframe.renderers import ROTOR_COLORS
from tasks.rps_prediction import (
    HOP,
    N_FFT,
    SR_AUDIO,
    align_rps_to_gt,
)

__all__ = ["ROTOR_COLORS", "plot_full_sequence"]


def plot_full_sequence(
    *,
    audio: np.ndarray | None = None,
    rps_gt: np.ndarray | None = None,
    rps_pred: np.ndarray | None = None,
    sr: float = SR_AUDIO,
    title: str = "",
    ax: Any = None,
    figsize: tuple[float, float] = (7.1, 6.0),
    **style,
) -> matplotlib.figure.Figure:
    """Generate the 3-panel full-sequence figure.

    Panel 1: log-magnitude spectrogram (0–4 kHz)
    Panel 2: rotor-speed timeline (GT dotted, prediction solid)
    Panel 3: per-frame MSE (smoothed)

    Parameters
    ----------
    audio : np.ndarray
        1-D mono audio samples.
    rps_gt : np.ndarray
        (4, n_frames) ground-truth RPS at STFT frame rate.
    rps_pred : np.ndarray
        (4, n_frames) predicted RPS.
    sr : float
        Audio sample rate.
    title : str
        Optional title suffix for the spectrogram.
    """
    if audio is None or rps_gt is None or rps_pred is None:
        raise ValueError("audio, rps_gt, and rps_pred are all required")

    # PIT-trained predictors emit rotors in arbitrary order; align to GT so the
    # overlay colours and the per-frame MSE trace match the evaluation matching.
    rps_pred = align_rps_to_gt(rps_pred, rps_gt)

    audio = np.asarray(audio, dtype=np.float32)
    duration = len(audio) / sr
    n_frames = rps_gt.shape[1]
    t_stft = np.linspace(0, duration, n_frames)

    # Identify low-RPS regions (takeoff / landing) for highlighting.
    low_rps = np.all(rps_gt < 50, axis=0)
    transitions = np.diff(low_rps.astype(int))
    low_starts = np.where(transitions == 1)[0] + 1
    low_ends = np.where(transitions == -1)[0]
    if low_rps[0]:
        low_starts = np.r_[0, low_starts]
    if low_rps[-1]:
        low_ends = np.r_[low_ends, len(low_rps) - 1]

    fig, axes = plt.subplots(
        3,
        1,
        figsize=figsize,
        gridspec_kw={"height_ratios": [1.2, 1.0, 0.8], "hspace": 0.35},
    )

    # ── Panel 1: spectrogram ──
    ax = axes[0]
    window = torch.hann_window(N_FFT)
    X = torch.stft(
        torch.from_numpy(audio).float(),
        n_fft=N_FFT,
        hop_length=HOP,
        window=window,
        return_complex=True,
        normalized=True,
    )
    S = torch.abs(X).numpy()
    # torch.stft returns (freq, time) — exactly the row/column layout
    # imshow(origin="lower", extent=[0, dur, 0, f_max]) expects. The old
    # transpose here swapped the axes (time rendered vertically).
    log_mag = np.log1p(S)
    vmin = np.percentile(log_mag, 2)
    vmax = np.percentile(log_mag, 99)
    ax.imshow(
        log_mag,
        origin="lower",
        aspect="auto",
        extent=[0, duration, 0, sr / 2 / 1000],
        cmap="hot",
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_ylim(0, 4)
    ax.set_ylabel("freq [kHz]")
    ax.set_title(f"DREGON free-flight speech-high {title}".strip())
    ax.set_xticklabels([])
    ax.grid(False)
    for s, e in zip(low_starts, low_ends):
        ax.axvspan(t_stft[s], t_stft[e], color="gray", alpha=0.12, lw=0)

    # ── Panel 2: rotor speeds ──
    ax = axes[1]
    T = min(rps_gt.shape[1], rps_pred.shape[1])
    for r in range(4):
        ax.plot(t_stft[:T], rps_gt[r, :T], ":", color=ROTOR_COLORS[r], lw=0.5, alpha=0.55)
        ax.plot(t_stft[:T], rps_pred[r, :T], "-", color=ROTOR_COLORS[r], lw=0.5, alpha=0.75)
    for s, e in zip(low_starts, low_ends):
        ax.axvspan(t_stft[s], t_stft[e], color="gray", alpha=0.12, lw=0)
    ax.set_ylabel("rotor speed [rev/s]")
    ax.legend(
        handles=[
            matplotlib.lines.Line2D(
                [0], [0], color="black", lw=0.5, ls=":", alpha=0.55, label="GT"
            ),
            matplotlib.lines.Line2D(
                [0], [0], color="black", lw=0.5, ls="-", alpha=0.75, label="predicted"
            ),
        ],
        loc="upper right",
        fontsize=6,
    )
    ax.grid(True, alpha=0.2, lw=0.5)

    # ── Panel 3: per-frame MSE ──
    ax = axes[2]
    mse_trace = np.mean((rps_gt[:, :T] - rps_pred[:, :T]) ** 2, axis=0)
    # Smooth.
    frame_dur = HOP / sr
    w = max(1, int(1.0 / frame_dur))
    if w > 1:
        kernel = np.ones(w) / w
        mse_trace = np.convolve(mse_trace, kernel, mode="same")
    ax.fill_between(t_stft[:T], mse_trace, alpha=0.4, color="C0")
    ax.plot(t_stft[:T], mse_trace, color="C0", lw=0.8)
    for s, e in zip(low_starts, low_ends):
        ax.axvspan(t_stft[s], min(t_stft[e], t_stft[T - 1]), color="gray", alpha=0.12, lw=0)
    ax.set_ylabel("MSE")
    ax.set_xlabel("time [s]")
    ax.grid(True, alpha=0.2, lw=0.5)

    plt.tight_layout()
    return fig
