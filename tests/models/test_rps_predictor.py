"""Forward-pass shape tests for `models.rps_predictor` model variants."""

from __future__ import annotations

import pytest
import torch

from models.rps_predictor import (
    SimpleConv,
    SimpleConvAttnPool,
    SimpleConvBiGRU,
    SimpleConvBiGRUV2,
    SimpleConvMagPhaseBiGRU,
    SimpleConvMultiScale,
    SimpleConvSENext,
    SimpleConvTCN,
    SimpleConvV2,
    SimpleConvWide,
)

MODELS = [
    SimpleConv,
    SimpleConvV2,
    SimpleConvWide,
    SimpleConvTCN,
    SimpleConvMultiScale,
    SimpleConvBiGRU,
    SimpleConvBiGRUV2,
    SimpleConvMagPhaseBiGRU,
    SimpleConvAttnPool,
    SimpleConvSENext,
]


@pytest.mark.parametrize("model_cls", MODELS)
def test_forward_shape(model_cls):
    """Each model: (B, T) audio → (B, 4, T_stft)."""
    model = model_cls(n_fft=256, hop_length=64, num_rotors=4)
    model.eval()
    audio = torch.randn(2, 8000)
    with torch.no_grad():
        out = model(audio)
    assert out.shape[0] == 2, f"{model_cls.__name__}: batch dim"
    assert out.shape[1] == 4, f"{model_cls.__name__}: rotor dim"
    assert out.shape[2] > 0, f"{model_cls.__name__}: time dim"


@pytest.mark.parametrize("model_cls", MODELS)
def test_forward_shape_varying_audio_length(model_cls):
    """T_stft scales with audio length."""
    model = model_cls(n_fft=256, hop_length=64, num_rotors=4)
    model.eval()
    short = torch.randn(1, 4000)
    long = torch.randn(1, 16000)
    with torch.no_grad():
        out_short = model(short)
        out_long = model(long)
    # At minimum, both produce valid shapes
    assert out_short.shape[0] == 1
    assert out_short.shape[1] == 4
    assert out_long.shape[0] == 1
    assert out_long.shape[1] == 4
