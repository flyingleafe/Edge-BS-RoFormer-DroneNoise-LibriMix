"""Smoke tests for the Edge-BS-RoFormer RPS adaptation (models/edge_bs_rof/rps.py)."""

import torch

from models.registry import build_model


def test_forward_shape_and_grads():
    m = build_model("edge_bs_rof_rps", flash_attn=False)  # CPU: no flash SDPA backend
    audio = torch.randn(2, 32000)
    out = m(audio)
    assert out.shape == (2, 4, 32000 // 512 + 1)
    out.sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)


def test_channel_dim_accepted():
    m = build_model("edge_bs_rof_rps", flash_attn=False)
    out = m(torch.randn(1, 1, 16000))
    assert out.shape == (1, 4, 16000 // 512 + 1)


def test_mask_estimators_dropped():
    m = build_model("edge_bs_rof_rps", flash_attn=False)
    assert not hasattr(m.core, "mask_estimators")
    # keep the budget in the SimpleConv-family ballpark (sanity, not a gate)
    n = sum(p.numel() for p in m.parameters())
    assert 100_000 < n < 5_000_000
