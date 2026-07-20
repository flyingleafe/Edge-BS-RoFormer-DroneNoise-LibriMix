"""Smoke test for the MP-SENet SE baseline built from its conf/model yaml."""

from __future__ import annotations

from pathlib import Path

import torch
from omegaconf import OmegaConf

from training.config import instantiate_model

_CONF = Path(__file__).resolve().parents[2] / "conf" / "model" / "f1_mpsenet.yaml"


def test_mpsenet_builds_and_runs() -> None:
    cfg = OmegaConf.load(_CONF)
    model = instantiate_model(cfg)
    model.eval()

    b, t = 2, 8000
    x = torch.randn(b, 1, t)
    with torch.no_grad():
        out = model(x)

    # SpeechEnhancementCodec squeezes interior singletons to (B, T); the model
    # returns (B, T) directly. Length must equal the input length T.
    out = out.squeeze()
    assert out.shape == (b, t), out.shape
    assert torch.isfinite(out).all()


def test_mpsenet_param_count() -> None:
    cfg = OmegaConf.load(_CONF)
    model = instantiate_model(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    assert 1.5e6 < n_params < 3.0e6, n_params
