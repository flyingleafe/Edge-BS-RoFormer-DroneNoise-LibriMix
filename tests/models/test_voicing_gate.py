"""Voicing-gated output projection + front-end/config plumbing.

Covers the three RPS architectures of the honest-base grid
(``simple_conv_v2``, ``simple_conv_v2_uni_gru128``,
``simple_conv_v2_transformer``): the ``voicing_gate`` keyword, the string
``frontend`` keyword, the front-end-adapted encoder input width, and the
checkpoint compatibility of the ungated default.

``stft_ssq`` is deliberately NOT exercised here — it is a separate front-end
key, tested with the front-end itself.
"""

from __future__ import annotations

from typing import cast

import pytest
import torch
import torch.nn as nn

from models.registry import build_model
from models.rps_predictor import (
    BiGRUHead,
    CausalGRUHead,
    GatedProjection,
    SimpleConvV2,
    TemporalTransformerHead,
)

ARCH_NAMES = [
    "simple_conv_v2",
    "simple_conv_v2_uni_gru128",
    "simple_conv_v2_transformer",
]
FRONTEND_KEYS = ["stft_mag", "stft_mag_if"]

N_FFT = 2048
HOP = 512
SR = 16000
NUM_ROTORS = 4


@pytest.mark.parametrize("name", ARCH_NAMES)
@pytest.mark.parametrize("frontend", FRONTEND_KEYS)
@pytest.mark.parametrize("voicing_gate", [False, True])
def test_build_and_forward(name: str, frontend: str, voicing_gate: bool):
    model = build_model(
        name,
        n_fft=N_FFT,
        hop_length=HOP,
        num_rotors=NUM_ROTORS,
        frontend=frontend,
        voicing_gate=voicing_gate,
    )
    model.eval()
    audio = torch.randn(2, SR)
    with torch.no_grad():
        out = model(audio)
    assert out.shape == (2, NUM_ROTORS, SR // HOP + 1)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize(
    ("head_cls", "kwargs"),
    [
        (BiGRUHead, {"hidden_ch": 64}),
        (CausalGRUHead, {"hidden_ch": 128}),
        (TemporalTransformerHead, {"hidden_ch": 64, "num_heads": 4}),
    ],
)
def test_gate_closes_output(head_cls, kwargs):
    """A very negative gate logit drives the output to (numerically) zero."""
    head = head_cls(128, num_rotors=NUM_ROTORS, num_layers=2, gated=True, **kwargs)
    assert isinstance(head.proj, GatedProjection)
    with torch.no_grad():
        head.proj.linear.bias[NUM_ROTORS:] = -20.0
    head.eval()
    with torch.no_grad():
        out = head(torch.randn(2, 128, 32))
    assert out.abs().max().item() < 1e-6


def test_gated_projection_shapes_and_identity():
    proj = GatedProjection(16, NUM_ROTORS)
    x = torch.randn(3, 5, 16)
    out = proj(x)
    assert out.shape == (3, 5, NUM_ROTORS)
    # speed * sigmoid(gate) — reproduce it from the raw Linear.
    raw = proj.linear(x)
    speed, gate = raw.chunk(2, dim=-1)
    assert torch.allclose(out, speed * torch.sigmoid(gate))


def test_ungated_state_dict_keys_unchanged():
    """The ungated default must stay checkpoint-compatible."""
    keys = set(SimpleConvV2().state_dict())
    assert "head.proj.weight" in keys
    assert "head.proj.bias" in keys
    assert not any(k.startswith("head.proj.linear.") for k in keys)


def test_gated_state_dict_uses_the_nested_linear():
    keys = set(SimpleConvV2(voicing_gate=True).state_dict())
    assert "head.proj.linear.weight" in keys
    assert "head.proj.weight" not in keys


def _first_conv(model: SimpleConvV2) -> nn.Conv2d:
    return cast(nn.Conv2d, cast(nn.Module, model.encoder[0]).conv)


def test_frontend_string_adapts_the_encoder_width():
    model = SimpleConvV2(frontend="stft_mag_if")
    assert model.frontend.out_channels == 2
    assert _first_conv(model).in_channels == 2
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(1, SR))
    assert out.shape == (1, NUM_ROTORS, SR // HOP + 1)


def test_default_frontend_stays_single_channel():
    model = SimpleConvV2()
    assert _first_conv(model).in_channels == 1
