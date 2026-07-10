"""Phase-0 experiment: Kalman harmonic tracker vs framed lstsq_VP baseline.

Two questions, one plot:
  H1 (sanity):     with *perfect* RPS, does the causal recursive tracker
                   remove rotor harmonics as well as the per-frame joint
                   least-squares projection?
  H2 (the bet):    when the RPS we feed both methods is slightly *wrong*
                   (slow drift, like real telemetry / pseudo-RPS), does the
                   tracker degrade gracefully while lstsq collapses?

Protocol: synthesize 4-rotor harmonic noise with known ground truth
(`oscillator_bank`), mix with speech(-proxy) at a target SNR, enhance by
harmonic subtraction using (a) the tracker, (b) lstsq_VP, both fed the SAME
corrupted RPS; report SI-SDR of the enhanced signal vs clean speech across
drift magnitudes.

Run:  python -m experiments.kalman_harmonic.phase0 [--speech path.wav]
Outputs: printed table + results/kalman_harmonic_phase0/{results.json,sweep.png}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from metrics.separation import si_sdr
from models.generative.dsp import harmonic_freq_series, oscillator_bank
from models.generative.harmonic_transform import (
    inverse_VP_transform,
    lstsq_VP_transform,
)

from .filter import TrackerConfig, kalman_harmonic_track

N_BLADES = 2  # DJI-Matrice-style two-bladed props; fundamental = N_BLADES * RPS


# --------------------------------------------------------------------------
# Synthetic scene
# --------------------------------------------------------------------------


def synth_rps(n_rotors: int, T: int, sr: int, g: torch.Generator) -> torch.Tensor:
    """Hover-ish RPS trajectories [R, T] in Hz: ~90 Hz, slow independent wander."""
    base = 85.0 + 10.0 * torch.rand(n_rotors, 1, generator=g)
    walk = torch.cumsum(torch.randn(n_rotors, T, generator=g) / sr, dim=-1) * 2.0
    return base + walk  # a few Hz of slow drift over seconds


def synth_rotor_noise(
    rps: torch.Tensor, n_harmonics: int, sr: int, g: torch.Generator
) -> torch.Tensor:
    """Harmonic rotor noise from ground-truth RPS with slowly-AM'd amplitudes."""
    R, T = rps.shape
    freqs = harmonic_freq_series(N_BLADES * rps, n_harmonics)  # [R, H, T]
    h = torch.arange(1, n_harmonics + 1).float()
    base_amp = (1.0 / h).sqrt()[None, :, None]  # ~1/√h rolloff
    am = 1.0 + 0.3 * torch.sin(
        2
        * torch.pi
        * (0.5 + torch.rand(R, n_harmonics, 1, generator=g))
        * torch.arange(T)[None, None, :]
        / sr
        + 2 * torch.pi * torch.rand(R, n_harmonics, 1, generator=g)
    )
    amps = base_amp * am
    noise = oscillator_bank(freqs, amps, return_sum=True, sr=sr)  # [R, T]
    return noise.sum(0)


def synth_speech_proxy(T: int, sr: int, g: torch.Generator) -> torch.Tensor:
    """Speech-shaped stand-in: band-limited noise with syllabic-rate bursts."""
    x = torch.randn(T, generator=g)
    X = torch.fft.rfft(x)
    f = torch.fft.rfftfreq(T, 1 / sr)
    shape = ((f / 500.0).clamp_min(0.2) ** -1.0) * (f > 100) * (f < 4000)
    x = torch.fft.irfft(X * shape, n=T)
    env = (
        1
        + torch.sin(
            2 * torch.pi * 3.0 * torch.arange(T) / sr + 2 * torch.pi * torch.rand(1, generator=g)
        )
    ) / 2
    return x * env**2


def corrupt_rps(rps: torch.Tensor, rel_sigma: float, sr: int, g: torch.Generator) -> torch.Tensor:
    """Slow multiplicative drift — the realistic, phase-accumulating error mode.

    OU-shaped relative error with correlation time ~0.5 s and stationary std
    `rel_sigma`. White per-sample jitter would be largely integrated away;
    slow drift is what de-coheres phasors.
    """
    if rel_sigma == 0.0:
        return rps
    R, T = rps.shape
    tau = 0.5 * sr
    a = float(np.exp(-1.0 / tau))
    e = torch.randn(R, T, generator=g) * rel_sigma * np.sqrt(1 - a * a)
    drift = torch.empty(R, T)
    acc = torch.zeros(R)
    for t in range(T):  # ok at phase-0 scale
        acc = a * acc + e[:, t]
        drift[:, t] = acc
    return rps * (1.0 + drift)


# --------------------------------------------------------------------------
# The two contenders
# --------------------------------------------------------------------------


def enhance_kalman(wav, rps_meas, n_harmonics, sr, cfg: TrackerConfig):
    out = kalman_harmonic_track(wav, N_BLADES * rps_meas, n_harmonics, sr=sr, cfg=cfg)
    return out["enhanced"]


def enhance_lstsq(wav, rps_meas, n_harmonics, sr, window_len=2048, hop_len=512):
    freqs = harmonic_freq_series(N_BLADES * rps_meas, n_harmonics)  # [R, H, T]
    V = lstsq_VP_transform(freqs, wav, window_len=window_len, hop_len=hop_len, sr=sr)
    noise_hat = inverse_VP_transform(freqs, V, window_len=window_len, hop_len=hop_len, sr=sr)
    L = min(noise_hat.shape[-1], wav.shape[-1])
    enhanced = wav.clone()
    enhanced[..., :L] = wav[..., :L] - noise_hat[..., :L]
    return enhanced


# --------------------------------------------------------------------------
# Experiment
# --------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--duration", type=float, default=4.0)
    ap.add_argument("--snr-db", type=float, default=-10.0)
    ap.add_argument("--n-harmonics", type=int, default=25)
    ap.add_argument("--n-rotors", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--speech", type=str, default=None, help="optional wav instead of proxy")
    ap.add_argument("--out", type=str, default="results/kalman_harmonic_phase0")
    ap.add_argument(
        "--sigmas", type=float, nargs="+", default=[0.0, 0.002, 0.005, 0.01, 0.02, 0.05]
    )
    args = ap.parse_args()

    g = torch.Generator().manual_seed(args.seed)
    sr, T = args.sr, int(args.sr * args.duration)

    rps_true = synth_rps(args.n_rotors, T, sr, g)
    noise = synth_rotor_noise(rps_true, args.n_harmonics, sr, g)
    if args.speech:
        import torchaudio

        speech, s_sr = torchaudio.load(args.speech)
        speech = torchaudio.functional.resample(speech[0], s_sr, sr)[:T]
        speech = torch.nn.functional.pad(speech, (0, T - speech.shape[-1]))
    else:
        speech = synth_speech_proxy(T, sr, g)

    # mix at target SNR (speech power relative to noise power)
    speech = speech / speech.std()
    noise = noise / noise.std()
    speech = speech * 10 ** (args.snr_db / 20)
    mix = speech + noise

    cfg = TrackerConfig()
    rows = []
    for sigma in args.sigmas:
        rps_meas = corrupt_rps(rps_true, sigma, sr, g)
        enh_kf = enhance_kalman(mix, rps_meas, args.n_harmonics, sr, cfg)
        enh_ls = enhance_lstsq(mix, rps_meas, args.n_harmonics, sr)
        row = {
            "sigma": sigma,
            "si_sdr_unprocessed": float(si_sdr(speech.numpy(), mix.numpy())),
            "si_sdr_kalman": float(si_sdr(speech.numpy(), enh_kf.numpy())),
            "si_sdr_lstsq": float(si_sdr(speech.numpy(), enh_ls.numpy())),
        }
        rows.append(row)
        print(
            f"sigma={sigma:6.3%}  unproc={row['si_sdr_unprocessed']:7.2f}  "
            f"kalman={row['si_sdr_kalman']:7.2f}  lstsq={row['si_sdr_lstsq']:7.2f}  [dB SI-SDR]"
        )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps({"args": vars(args), "rows": rows}, indent=2))

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        s = [r["sigma"] * 100 for r in rows]
        plt.figure(figsize=(6, 4))
        plt.plot(s, [r["si_sdr_kalman"] for r in rows], "o-", label="Kalman tracker")
        plt.plot(s, [r["si_sdr_lstsq"] for r in rows], "s-", label="lstsq_VP")
        plt.plot(s, [r["si_sdr_unprocessed"] for r in rows], "k--", label="unprocessed")
        plt.xlabel("RPS drift σ (% of true RPS)")
        plt.ylabel("SI-SDR (dB)")
        plt.title(f"Harmonic subtraction vs RPS error (SNR {args.snr_db} dB)")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(out / "sweep.png", dpi=150)
        print(f"wrote {out / 'sweep.png'}")
    except Exception as e:  # plotting is optional
        print(f"(no plot: {e})")


if __name__ == "__main__":
    main()
