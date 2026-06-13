"""
GPU HCQT using nnAudio's CQT2010v2 — the only GPU CQT we've tested that
correctly handles multi-rate processing and matches librosa peak frequencies.

Module structure mirrors hcqt.py's compute_hcqt_mag_phase interface.
"""

import math

import numpy as np
import torch
import torch.nn as nn
from nnAudio.features import CQT2010v2


class HCQT_nnAudio(nn.Module):
    """GPU Harmonic CQT built on nnAudio's CQT2010v2.

    For each harmonic h, creates a CQT2010v2 with fmin * h.
    Outputs magnitude (dB) and unwrapped phase differentials.

    Peak frequencies match librosa for stationary tones;
    phase differentials use nnAudio's phase convention (consistent
    within the module; original paper's libreosa phase convention
    will differ by up to π for raw phase but differentials converge).

    Usage:
        hcqt = HCQT_nnAudio()
        mag, dphase = hcqt(audio_tensor)   # (B, samples) at 22050 Hz
        # mag:    (B, n_harmonics, n_bins, n_frames)
        # dphase: (B, n_harmonics, n_bins, n_frames)
    """

    def __init__(
        self,
        sr: int = 22050,
        fmin: float = 32.7,
        n_octaves: int = 6,
        over_sample: int = 5,
        harmonics: list[int] | None = None,
        hop_length: int = 256,
        log_mag: bool = True,
    ):
        super().__init__()
        if harmonics is None:
            harmonics = [1, 2, 3, 4, 5]
        else:
            harmonics = list(harmonics)
        self.harmonics = harmonics
        self.sr = sr
        self.hop_length = hop_length
        self.log_mag = log_mag

        bins_per_octave = 12 * over_sample
        n_bins = n_octaves * bins_per_octave
        self.n_bins = n_bins

        # One CQT2010v2 per harmonic
        self.cqts = nn.ModuleList()
        for h in harmonics:
            cqt = CQT2010v2(
                sr=sr,
                hop_length=hop_length,
                fmin=fmin * h,
                n_bins=n_bins,
                bins_per_octave=bins_per_octave,
                output_format="Complex",
                verbose=False,
            )
            self.cqts.append(cqt)

    def forward(self, audio: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute HCQT magnitude and phase differentials.

        Args:
            audio: (B, samples) at self.sr Hz.

        Returns:
            mag:    (B, n_harmonics, n_bins, n_frames)  dB if log_mag=True
            dphase: (B, n_harmonics, n_bins, n_frames)  unwrapped phase diff
        """
        if audio.dim() == 3:
            audio = audio.squeeze(1)

        mags = []
        dphases = []

        for cqt in self.cqts:
            # CQT2010v2 Complex output: (B, n_bins, n_frames, 2) — [real, imag]
            C = cqt(audio)  # (B, F, T, 2)

            real, imag = C[..., 0], C[..., 1]
            mag = torch.sqrt(real**2 + imag**2 + 1e-10)  # (B, F, T)
            phase = torch.atan2(imag, real)  # (B, F, T)

            # Log-magnitude (match librosa: amplitude_to_db with ref=max)
            if self.log_mag:
                ref = mag.amax(dim=(1, 2), keepdim=True).clamp(min=1e-10)
                mag = 20.0 * torch.log10(mag / ref + 1e-10)
                mag = mag.clamp(min=-80.0)

            # Unwrapped phase differential along time
            dphase = _phase_diff_torch(phase)  # (B, F, T)

            mags.append(mag.unsqueeze(1))  # (B, 1, F, T)
            dphases.append(dphase.unsqueeze(1))

        mag = torch.cat(mags, dim=1)  # (B, H, F, T)
        dphase = torch.cat(dphases, dim=1)

        return mag, dphase


def _phase_diff_torch(phase: torch.Tensor) -> torch.Tensor:
    """Unwrapped phase differential along last axis (time).

    Replicates pumpp's phase_diff:
        dphase[:, 0] = phase[:, 0]
        dphase[:, 1:] = diff(unwrap(phase), axis=-1)

    Pure-torch, stays on-device. ``diff(unwrap(phase))`` is algebraically the
    wrapped per-step phase difference ``((dd + π) mod 2π) − π``; the old numpy
    path forced a GPU→CPU sync every forward (a major GPU training stall). The
    only subtlety is numpy's ``unwrap`` edge case: a wrapped diff of exactly
    ``−π`` flips to ``+π`` when the raw step ``dd`` is positive — replicated here
    so the output is bit-faithful to ``np.diff(np.unwrap(...))``.
    """
    dphase = torch.empty_like(phase)
    dphase[..., 0] = phase[..., 0]
    dd = phase[..., 1:] - phase[..., :-1]
    ddmod = torch.remainder(dd + math.pi, 2.0 * math.pi) - math.pi
    ddmod = torch.where(
        (ddmod == -math.pi) & (dd > 0), torch.full_like(ddmod, math.pi), ddmod
    )
    dphase[..., 1:] = ddmod
    return dphase


# ═══════════════════════════════════════════════════════════════════════════
# Equivalence test
# ═══════════════════════════════════════════════════════════════════════════


def test_nnaudio_equivalence(verbose: bool = True) -> dict:
    """Compare nnAudio HCQT against librosa HCQT."""
    import time

    sr, dur = 22050, 2.0
    t_arr = np.arange(int(sr * dur)) / sr
    # Stationary tone (where we expect perfect peak match)
    f0 = 80.0
    audio = (0.5 * np.sin(2 * np.pi * f0 * t_arr)).astype(np.float32)
    audio += 0.005 * np.random.randn(len(audio)).astype(np.float32)

    hcqt_kw = dict(
        sr=sr, fmin=32.7, n_octaves=6, over_sample=5, harmonics=[1, 2, 3, 4, 5], hop_length=256
    )
    hcqt_nn_kw = {**hcqt_kw, "log_mag": True}
    hcqt_ref_kw = {**hcqt_kw, "log": True}

    # nnAudio HCQT
    hcqt_nn = HCQT_nnAudio(**hcqt_nn_kw)
    hcqt_nn.eval()
    audio_t = torch.from_numpy(audio).unsqueeze(0)
    t0 = time.time()
    with torch.no_grad():
        mag_nn, dp_nn = hcqt_nn(audio_t)
    t_nn = time.time() - t0
    mag_nn = mag_nn.squeeze(0).cpu().numpy()
    dp_nn = dp_nn.squeeze(0).cpu().numpy()

    # Librosa reference
    from models.multif0.hcqt import compute_hcqt_mag_phase

    t1 = time.time()
    mag_ref, dp_ref = compute_hcqt_mag_phase(audio, **hcqt_ref_kw)
    t_ref = time.time() - t1

    # Align
    mt = min(mag_nn.shape[2], mag_ref.shape[2])
    mag_nn, dp_nn = mag_nn[:, :, :mt], dp_nn[:, :, :mt]
    mag_ref, dp_ref = mag_ref[:, :, :mt], dp_ref[:, :, :mt]

    # Magnitude difference
    mag_diff = np.abs(mag_nn - mag_ref)

    # Phase diff difference
    dp_diff = np.abs(dp_nn - dp_ref)
    dp_diff = np.minimum(dp_diff, 2 * np.pi - dp_diff)

    # Peak frequency accuracy
    import librosa

    freqs = librosa.cqt_frequencies(mag_nn.shape[1], fmin=32.7, bins_per_octave=12 * 5)
    peak_errs = []
    for h in range(mag_nn.shape[0]):
        for t in range(mt):
            pg = freqs[np.argmax(mag_nn[h, :, t])]
            pr = freqs[np.argmax(mag_ref[h, :, t])]
            peak_errs.append(abs(pg - pr))
    peak_errs = np.array(peak_errs)

    result = {
        "mag_max_db": float(mag_diff.max()),
        "mag_mean_db": float(mag_diff.mean()),
        "dp_max": float(dp_diff.max()),
        "dp_mean": float(dp_diff.mean()),
        "peak_max_hz": float(peak_errs.max()),
        "peak_mean_hz": float(peak_errs.mean()),
        "peak_exact_pct": float((peak_errs < 0.01).mean() * 100),
        "t_nn_s": t_nn,
        "t_ref_s": t_ref,
        "speedup": t_ref / t_nn if t_nn > 0 else float("inf"),
    }

    if verbose:
        print(f"  Signal: {dur}s, {f0} Hz tone, sr={sr}")
        print(
            f"  nnAudio: {t_nn:.3f}s  |  Librosa: {t_ref:.3f}s  |  speedup: {result['speedup']:.1f}×"
        )
        print(
            f"  Mag diff:     max={result['mag_max_db']:.1f} dB  mean={result['mag_mean_db']:.2f} dB"
        )
        print(f"  Dphase diff:  max={result['dp_max']:.4f} rad  mean={result['dp_mean']:.4f} rad")
        print(
            f"  Peak freq:    max err={result['peak_max_hz']:.3f} Hz  "
            f"mean={result['peak_mean_hz']:.6f} Hz  "
            f"exact={result['peak_exact_pct']:.1f}%"
        )
        pct = result["peak_exact_pct"]
        print(
            f"  {'✓ Peak accuracy: ' + str(pct) + '%' if pct > 99 else '✗ Peak accuracy: ' + str(pct) + '%'}"
        )

    return result


if __name__ == "__main__":
    print("=" * 70)
    print("nnAudio CQT2010v2 HCQT ↔ Librosa HCQT Equivalence Test")
    print("=" * 70)
    r = test_nnaudio_equivalence()
    print("=" * 70)
