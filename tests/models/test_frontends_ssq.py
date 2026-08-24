"""Synchrosqueezed STFT front-end (`stft_ssq`): grid parity with `stft_mag`,
energy conservation of the scatter, ridge sharpening on a fractional-bin tone,
and finiteness on silence / white noise.
"""

from __future__ import annotations

import math

import torch

from models.frontends import build_frontend
from models.frontends.ssq import STFTSSQMag

N_FFT, HOP, SR = 2048, 512, 16000


def _stft_power(audio: torch.Tensor, n_fft: int = N_FFT, hop: int = HOP) -> torch.Tensor:
    X = torch.stft(
        audio,
        n_fft=n_fft,
        hop_length=hop,
        window=torch.hann_window(n_fft),
        return_complex=True,
        normalized=True,
    )
    return X.abs() ** 2


def _tone(k0: int, frac: float, n: int = SR) -> torch.Tensor:
    f = (k0 + frac) * SR / N_FFT
    t = torch.arange(n, dtype=torch.float64) / SR
    return torch.sin(2 * math.pi * f * t).to(torch.float32).unsqueeze(0)


def test_ssq_reconcentrates_a_fractional_bin_tone():
    """A tone at k0+0.33 bins peaks at round(k0+0.33), sharper than |X|**2."""
    k0, frac = 100, 0.33
    tone = _tone(k0, frac)

    squeezed = torch.expm1(STFTSSQMag(n_fft=N_FFT, hop_length=HOP)(tone)[0, 0]) ** 2
    power = _stft_power(tone)[0]

    # steady-state frames only (skip the zero-padded first frame and the edges)
    ssq_mid = squeezed[:, 5:25]
    pow_mid = power[:, 5:25]

    assert int(ssq_mid.sum(dim=1).argmax()) == round(k0 + frac)

    ssq_share = float(ssq_mid.max(dim=0).values.div(ssq_mid.sum(dim=0)).mean())
    pow_share = float(pow_mid.max(dim=0).values.div(pow_mid.sum(dim=0)).mean())
    assert ssq_share > pow_share
    print(f"\npeak-energy share: plain {pow_share:.4f} -> squeezed {ssq_share:.4f}")


def test_ssq_shape_and_grid_match_stft_mag():
    fe = build_frontend("stft_ssq", n_fft=256, hop_length=64)
    fe_mag = build_frontend("stft_mag", n_fft=256, hop_length=64)
    assert isinstance(fe, STFTSSQMag)
    assert fe.out_channels == 1

    audio = torch.randn(2, 8000)
    out = fe(audio)
    assert out.shape == fe_mag(audio).shape
    assert out.shape == (2, 1, 256 // 2 + 1, 8000 // 64 + 1)
    assert fe.num_frames(8000) == out.shape[-1]


def test_ssq_conserves_energy():
    audio = torch.randn(2, 8000)
    fe = STFTSSQMag(n_fft=256, hop_length=64)
    squeezed = torch.expm1(fe(audio)[:, 0]) ** 2
    power = _stft_power(audio, n_fft=256, hop=64)
    assert torch.allclose(squeezed.sum(), power.sum(), rtol=1e-4)


def test_ssq_on_silence_is_zero():
    fe = STFTSSQMag(n_fft=256, hop_length=64)
    out = fe(torch.zeros(1, 8000))
    assert torch.isfinite(out).all()
    assert torch.equal(out, torch.zeros_like(out))


def test_ssq_on_white_noise_is_finite():
    torch.manual_seed(0)
    fe = STFTSSQMag(n_fft=256, hop_length=64)
    out = fe(torch.randn(3, 8000))
    assert torch.isfinite(out).all()
    assert (out >= 0).all()


def test_ssq_accepts_channel_dim_input():
    fe = STFTSSQMag(n_fft=256, hop_length=64)
    audio = torch.randn(2, 8000)
    assert torch.allclose(fe(audio.unsqueeze(1)), fe(audio))
