"""Smoke tests for the TF-GridNet SE model + its speech_enhancement wiring.

Builds the model from ``conf/model/f1_tfgridnet.yaml`` through the exact
``training.config.instantiate_model`` dispatch a real Hydra run uses, then
exercises the ``SpeechEnhancementCodec`` call path (mono ``(B, T)`` → model
``(B, 1, T)`` → ``(B, T)``) end to end. Short signal so it runs on CPU fast.
"""

from __future__ import annotations

from pathlib import Path

import torch
from omegaconf import OmegaConf

from tasks.codecs import build_codec
from training.config import instantiate_model

CONF = Path(__file__).resolve().parents[2] / "conf" / "model" / "f1_tfgridnet.yaml"


def _load_model() -> torch.nn.Module:
    cfg = OmegaConf.load(CONF)
    return instantiate_model(cfg)


def test_forward_shape_and_finite() -> None:
    model = _load_model().eval()
    x = torch.randn(2, 1, 4096)
    with torch.no_grad():
        out = model(x)
    out = out.squeeze()  # tolerate interior singleton axes
    assert out.shape == (2, 4096), out.shape
    assert torch.isfinite(out).all()


def test_codec_roundtrip() -> None:
    model = _load_model().eval()
    codec = build_codec(
        "speech_enhancement",
        n_channels=None,
        use_rps=False,
        predict_rps=False,
        sr=[16000, 1],
    )
    # mono task spec mixture is (B, T); the codec unsqueezes it to (B, 1, T)
    inputs = {"mixture": torch.randn(2, 16000)}
    with torch.no_grad():
        out = codec.call_model(model, inputs)
    out = out.squeeze()
    assert out.shape == (2, 16000), out.shape
    assert torch.isfinite(out).all()


def test_param_count_midsize() -> None:
    model = _load_model()
    n_params = sum(p.numel() for p in model.parameters())
    assert 5_000_000 <= n_params <= 12_000_000, n_params
