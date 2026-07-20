"""Smoke test for the SGMSE+ score-based diffusion SE baseline.

Builds the model from its conf/model yaml (via ``training.config.instantiate_model``)
and exercises both legs of the ``forward(mix, target=None)`` contract:
the training denoising-score-matching loss (target given -> finite scalar with
grad) and the eval predictor-corrector sampler (target None -> enhanced waveform).
The sampler is forced to N=2 steps via a config override to stay fast on CPU.
"""

from __future__ import annotations

from pathlib import Path

import torch
from omegaconf import OmegaConf

from training.config import instantiate_model

_CONF = Path(__file__).resolve().parents[2] / "conf" / "model" / "f1_sgmse.yaml"


def _load(n_sampler_steps: int = 2):
    cfg = OmegaConf.load(_CONF)
    cfg.params.config.sde.N = n_sampler_steps  # keep the PC sampler cheap on CPU
    return instantiate_model(cfg)


def test_sgmse_train_forward_returns_scalar_dsm_loss() -> None:
    torch.manual_seed(0)
    model = _load()
    model.train()

    mix = torch.randn(2, 1, 8192)
    target = torch.randn(2, 1, 8192)
    loss = model(mix=mix, target=target)

    assert loss.ndim == 0, loss.shape
    assert torch.isfinite(loss).all()
    assert loss.requires_grad
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, "no gradients flowed into model parameters"
    assert all(torch.isfinite(g).all() for g in grads)


def test_sgmse_eval_forward_returns_enhanced_waveform() -> None:
    torch.manual_seed(0)
    model = _load(n_sampler_steps=2)
    model.eval()

    mix = torch.randn(1, 1, 8192)
    with torch.no_grad():
        out = model(mix)

    assert torch.isfinite(out).all()
    assert out.squeeze().shape == (8192,), out.shape


def test_sgmse_param_count() -> None:
    model = _load()
    n_params = sum(p.numel() for p in model.parameters())
    # Full NCSN++ (nf=128, 7-level ch_mult, 2 res-blocks) ~ 65M params.
    assert 5.0e7 < n_params < 8.0e7, n_params
