"""Generate the time-warp before/after figure (E5 of the narrative).

Takes one real DREGON clip and applies the project's actual time-warp
augmentation code (src/data_processing/time_warp.py) to it, then plots the
spectrogram + RPS trajectory before and after.
"""

import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from data_processing.dregon import load_dregon_timeframes
from data_processing.frames import get_meta
from data_processing.online_mixing import _extract_audio_array, interpolate_rps_to_stft_grid
from data_processing.time_warp import apply_time_warp, sample_warp_params

SR = 16000
DUR_S = 4.0
N = int(DUR_S * SR)


def stft_db(x, n_fft=1024, hop=256):
    X = (
        torch.stft(
            torch.from_numpy(np.ascontiguousarray(x)).float(),
            n_fft=n_fft,
            hop_length=hop,
            window=torch.hann_window(n_fft),
            return_complex=True,
        )
        .abs()
        .numpy()
    )
    return 20 * np.log10(X + 1e-6)


def main():
    frames = load_dregon_timeframes(
        PROJECT_ROOT / "data", splits=["in_flight_noise"], target_sr=SR, download=False
    )
    tf = next(
        (f for f in frames if "free-flight" in str(get_meta(f, "recording_id", ""))), frames[0]
    )
    t0 = tf["audio"].t_start
    avail = tf["audio"].t_end - t0

    # pick the highest-mean-RPS window so the RPS overlay is not near-zero
    best_s, best_m, s, stride_s = 0.0, -1.0, 0.0, 2.0
    while s + DUR_S <= avail:
        m = float(
            interpolate_rps_to_stft_grid(
                tf.time[t0 + s : t0 + s + DUR_S], n_frames=16, hop_length=int(DUR_S * SR / 16)
            ).mean()
        )
        if m > best_m:
            best_s, best_m = s, m
        s += stride_s

    margin_s = 1.0
    sl = tf.time[t0 + best_s : t0 + best_s + DUR_S + margin_s]

    rng = np.random.default_rng(3)
    params = sample_warp_params({}, rng)
    warped = apply_time_warp(sl, params, target_len=N, sample_rate=SR)

    orig_sl = tf.time[t0 + best_s : t0 + best_s + DUR_S]
    orig_audio = _extract_audio_array(orig_sl, target_len=N)[0]
    orig_rps = interpolate_rps_to_stft_grid(orig_sl, n_frames=N, hop_length=1)

    warp_audio: Any = np.asarray(warped["audio"].data, dtype=np.float32)
    if warp_audio.ndim > 1:
        warp_audio = warp_audio[0]
    warp_rps: Any = np.asarray(warped["rps"].data, dtype=np.float32)
    if warp_rps.ndim > 1:
        warp_rps = warp_rps[0]

    Sr = stft_db(orig_audio)
    Sw = stft_db(warp_audio)
    vmax = Sr.max()
    vmin = vmax - 80

    fig, axs = plt.subplots(2, 2, figsize=(9.5, 5.5), gridspec_kw=dict(height_ratios=[3, 1]))
    axs[0, 0].imshow(
        Sr,
        origin="lower",
        aspect="auto",
        cmap="magma",
        vmin=vmin,
        vmax=vmax,
        extent=[0, DUR_S, 0, SR / 2],
    )
    axs[0, 0].set_title("original")
    axs[0, 0].set_ylabel("Hz")
    axs[0, 1].imshow(
        Sw,
        origin="lower",
        aspect="auto",
        cmap="magma",
        vmin=vmin,
        vmax=vmax,
        extent=[0, DUR_S, 0, SR / 2],
    )
    axs[0, 1].set_title(f"time-warped (c={params.c:.2f}, a={params.a:.2f}, f={params.f:.2f} Hz)")

    t_orig = np.arange(N) / SR
    t_warp = np.arange(len(warp_rps)) / 100.0  # label_rate_hz default 100
    axs[1, 0].plot(t_orig, orig_rps[0], lw=0.9)
    axs[1, 0].set_xlabel("s")
    axs[1, 0].set_ylabel("rev/s")
    axs[1, 0].set_xlim(0, DUR_S)
    axs[1, 1].plot(t_warp, warp_rps, lw=0.9, color="tab:red")
    axs[1, 1].set_xlabel("s")
    axs[1, 1].set_xlim(0, DUR_S)

    fig.suptitle("Time-warp augmentation: one DREGON clip, alpha(t) = c + a*sin(2*pi*f*t+phi)")
    plt.tight_layout()
    outp = HERE / "assets/timewarp_before_after.png"
    fig.savefig(outp, dpi=130)
    print("saved", outp)


if __name__ == "__main__":
    main()
