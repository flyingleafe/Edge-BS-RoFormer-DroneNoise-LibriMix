"""G2 front-end arms (VK-parity criterion 2.3): STFTMagIF numerics and the
two SimpleConvV2Transformer front-end variants (HCQT / IF) — forward shapes,
finiteness, registry construction, and a short optimizer-step training smoke.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from models.frontends import build_frontend
from models.frontends.stft import STFTMagIF
from models.registry import build_model
from models.rps_predictor import (
    ResidualConvBlock2d,
    SimpleConvV2Transformer,
    SimpleConvV2TransformerHCQT,
    SimpleConvV2TransformerIF,
)


def _first_conv_in_channels(model) -> int:
    block = model.encoder[0]
    assert isinstance(block, ResidualConvBlock2d)
    return block.conv.in_channels


def test_stft_mag_if_registered_shape_and_finite():
    fe = build_frontend("stft_mag_if", n_fft=256, hop_length=64)
    assert isinstance(fe, STFTMagIF)
    assert fe.out_channels == 2
    audio = torch.randn(2, 8000)
    out = fe(audio)
    assert out.shape == (2, 2, 256 // 2 + 1, 8000 // 64 + 1)
    assert torch.isfinite(out).all()
    assert fe.num_frames(8000) == 8000 // 64 + 1


def test_stft_mag_if_recovers_sub_bin_offset():
    """A tone offset +0.3 bins from bin center reads ~+0.3 in the IF channel."""
    n_fft, hop, sr = 2048, 512, 16000
    k0, frac = 100, 0.3
    f = (k0 + frac) * sr / n_fft
    t = torch.arange(sr, dtype=torch.float64) / sr
    tone = torch.sin(2 * math.pi * f * t).to(torch.float32).unsqueeze(0)

    fe = STFTMagIF(n_fft=n_fft, hop_length=hop)
    if_dev = fe(tone)[0, 1]  # (F, T)
    mid = if_dev[k0, 5:25]
    assert torch.allclose(mid, torch.full_like(mid, frac), atol=1e-3)
    # channel is bounded: wrapped |dev| <= pi -> n_fft/(2*hop) = 2 bins
    assert if_dev.abs().max() <= n_fft / (2 * hop) + 1e-4


def test_stft_mag_if_mag_channel_matches_stft_mag():
    fe_if = STFTMagIF(n_fft=512, hop_length=128)
    fe_mag = build_frontend("stft_mag", n_fft=512, hop_length=128)
    audio = torch.randn(1, 8000)
    assert torch.allclose(fe_if(audio)[:, 0], fe_mag(audio)[:, 0], atol=1e-6)


def test_baseline_transformer_unchanged_by_in_ch_refactor():
    """Default trunk still builds with a 1-channel first conv (checkpoint compat)."""
    model = SimpleConvV2Transformer(n_fft=256, hop_length=64)
    assert _first_conv_in_channels(model) == 1
    out = model(torch.randn(2, 8000))
    assert out.shape == (2, 4, 8000 // 64 + 1)


@pytest.mark.parametrize(
    "key,cls",
    [
        ("simple_conv_v2_transformer_hcqt", SimpleConvV2TransformerHCQT),
        ("simple_conv_v2_transformer_if", SimpleConvV2TransformerIF),
    ],
)
def test_registry_builds_g2_models(key, cls):
    model = build_model(key, n_fft=2048, hop_length=512, num_rotors=4)
    assert isinstance(model, cls)


def _assert_one_step_trains(model, audio, out_shape):
    model.train()
    target = torch.rand(out_shape) * 50.0 + 40.0
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    opt.zero_grad()
    loss = F.mse_loss(model(audio), target)
    assert torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    opt.step()
    loss2 = F.mse_loss(model(audio), target)
    assert torch.isfinite(loss2)


def test_if_transformer_forward_and_step():
    torch.manual_seed(0)
    model = SimpleConvV2TransformerIF(n_fft=256, hop_length=64)
    assert _first_conv_in_channels(model) == 2
    audio = torch.randn(2, 8000)
    out = model(audio)
    assert out.shape == (2, 4, 8000 // 64 + 1)
    assert torch.isfinite(out).all()
    _assert_one_step_trains(model, audio, out.shape)


def test_hcqt_transformer_forward_and_step():
    torch.manual_seed(0)
    model = SimpleConvV2TransformerHCQT()  # default 2048/512 grid, 16 kHz HCQT
    assert model.frontend.out_channels == 6  # harmonics [1,2,3], mag+dphase
    assert _first_conv_in_channels(model) == 6
    audio = torch.randn(2, 16000)
    out = model(audio)
    assert out.shape == (2, 4, 16000 // 512 + 1)  # STFT grid despite hop-256 HCQT
    assert torch.isfinite(out).all()
    _assert_one_step_trains(model, audio, out.shape)
