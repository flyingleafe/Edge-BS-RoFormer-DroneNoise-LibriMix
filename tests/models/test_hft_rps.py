"""Contract tests for the hFT-Transformer rotor-rate port.

Four things a port must satisfy before a training job is worth a GPU slot:
the salience_rps shapes, the gather wiring (an untrained read of a synthetic
comb must peak at the comb), live gradients everywhere, and the hard-sparsity
claim itself — a rate token must attend to nothing outside its own harmonics.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from losses.salience import salience_bce_loss
from models.harmonic_ports.hft_rps import HFTRPS

SMALL = dict(hid_dim=32, n_layers=2, n_time_layers=1, n_heads=2, pf_dim=64, dropout=0.0)


def comb(rate: float, n: int = 16000, sr: int = 16000, k_max: int = 40, seed: int = 0):
    rng = np.random.default_rng(seed)
    t = np.arange(n) / sr
    x = 0.05 * rng.standard_normal(n)
    for k in range(1, k_max + 1):
        if k * rate >= 7500.0:
            break
        x += np.sin(2 * np.pi * k * rate * t + rng.uniform(0, 2 * np.pi)) / k
    return torch.from_numpy((x / np.abs(x).max()).astype(np.float32))


def test_shapes_and_grid():
    m = HFTRPS(**SMALL).eval()
    n = 16000
    with torch.no_grad():
        y = m(torch.randn(2, n), return_attention=True)
    assert y.shape == (2, 300, m.num_grid_frames(n))
    assert m.outputs_salience is True
    assert m.last_attention.shape == (2, y.shape[-1], 2, 300, 32)  # hFT's layout
    freqs = m.output_freqs()
    assert freqs.shape == (300,) and freqs[0] == 0.0 and freqs[-1] == 150.0
    assert torch.isfinite(y).all()


def test_untrained_gather_finds_the_comb():
    """The classical Whittle read through `evidence()` must peak at the comb.

    This is the corner case the family is built around: mean over harmonics of
    log1p(power/floor) IS the classical comb score, so if the gather is wired
    correctly an UNTRAINED model already localizes a synthetic comb to within
    one grid bin, with no learning at all.
    """
    m = HFTRPS(**SMALL).eval()
    grid = m.output_freqs()
    for rate in (37.0, 45.0, 60.0, 84.5, 120.0):
        with torch.no_grad():
            z = m.evidence(m.spectrum(comb(rate).unsqueeze(0)))
        band = m.band.to(z.dtype)
        score = (z.sum(1) / band.sum(0).clamp_min(1)[None, :, None])[0].mean(-1)
        assert abs(grid[int(score.argmax())] - rate) <= 0.6, rate


def test_every_parameter_gets_a_finite_nonzero_gradient():
    torch.manual_seed(0)
    m = HFTRPS(**SMALL)
    y = m(torch.randn(2, 16000))
    target = torch.zeros_like(y)
    target[:, 120, :] = 1.0
    salience_bce_loss(y, target).backward()
    for name, p in m.named_parameters():
        assert p.grad is not None, name
        assert torch.isfinite(p.grad).all(), name
        assert float(p.grad.abs().sum()) > 0.0, name


def test_attention_is_hard_sparse_outside_the_band():
    """A rate token must place exactly zero mass on out-of-band harmonics.

    That is the substitution the port exists to test: the gather is used as a
    structured sparsity PRIOR, not as a soft hint.
    """
    m = HFTRPS(**SMALL).eval()
    with torch.no_grad():
        m(comb(60.0).unsqueeze(0), return_attention=True)
    attn = m.last_attention  # (B, T, H, G, K)
    dead = ~m.band.t()  # (G, K): harmonic k of rate g is outside the band
    assert float(attn[:, :, :, dead].abs().max()) == 0.0
    live = m.any_band  # (G,): rates with at least one usable harmonic
    mass = attn[:, :, :, live].sum(-1)
    assert torch.allclose(mass, torch.ones_like(mass), atol=1e-5)
    assert float(attn[:, :, :, ~live].abs().max()) == 0.0


def test_loss_decreases_on_one_clip():
    torch.manual_seed(0)
    m = HFTRPS(**SMALL)
    x = comb(72.0).unsqueeze(0)
    target = None
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    losses = []
    for _ in range(20):
        opt.zero_grad(set_to_none=True)
        y = m(x)
        if target is None:
            target = torch.zeros_like(y)
            target[:, int(round(72.0 / (m.output_freqs()[1]))), :] = 1.0
        loss = salience_bce_loss(y, target, pos_weight=24.0)
        loss.backward()
        opt.step()
        losses.append(float(loss))
    assert losses[-1] < 0.5 * losses[0]


@pytest.mark.parametrize("n_samples", [16000, 4096, 40000])
def test_time_axis_is_preserved_across_block_boundaries(n_samples):
    """SAtime partitions the frame axis into hFT-sized blocks; T must survive."""
    m = HFTRPS(**SMALL, n_frame=8).eval()
    with torch.no_grad():
        y = m(torch.randn(1, n_samples))
    assert y.shape[-1] == m.num_grid_frames(n_samples)


# ── variant (ii): gather-as-bias ────────────────────────────────────────────

BIAS = dict(SMALL, attn_mode="bias", n_enc_layers=2)


def test_bias_variant_shapes_and_gradients():
    """Variant (ii) keeps hFT's frequency encoder and softens the mask.

    The attention is now over the POOLED frequency tokens, so its last axis is
    ``n_bin`` and not ``k_max`` — and nothing is masked, which is the whole
    difference from variant (i).
    """
    torch.manual_seed(0)
    m = HFTRPS(**BIAS)
    y = m(torch.randn(2, 16000), return_attention=True)
    assert y.shape == (2, 300, m.num_grid_frames(16000))
    assert m.last_attention.shape == (2, y.shape[-1], 2, 300, 256)
    mass = m.last_attention.sum(-1)
    assert torch.allclose(mass, torch.ones_like(mass), atol=1e-5)
    target = torch.zeros_like(y)
    target[:, 120, :] = 1.0
    salience_bce_loss(y, target).backward()
    for name, p in m.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
        assert float(p.grad.abs().sum()) > 0.0, name


def test_bias_initialization_prefers_the_rates_own_harmonics():
    """The learned bias must START at the gather's read positions.

    ``bias_scale * log1p(incidence)`` where incidence counts the harmonics of
    rate g falling in pooled token j. So every token a hard mask would keep
    carries a positive bias and every other token carries exactly zero.
    """
    m = HFTRPS(**BIAS).eval()
    bias = m.layers[0].bias[0].detach()  # (G, n_bin)
    df = 16000 / 4096
    grid = m.output_freqs()
    for g in (60, 120, 240):
        rate = grid[g]
        hot = torch.nonzero(bias[g] > 0).ravel().tolist()
        for tok in hot:
            lo, hi = tok * m.pool * df, (tok + 1) * m.pool * df
            assert any(lo <= k * rate < hi for k in range(1, m.k_max + 1)), (rate, tok)
        assert len(hot) > 4
