"""The three harmonic ports on the lossless per-rotor readout.

The properties locked here are the ones the campaign paid for: the layers reach
the loss and the decoder in the shape both expect, the encode/decode pair is an
isomorphism on trajectories the transition band admits, and the loss does not
care which layer holds which rotor.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from losses.salience_layers import LayerPITSalienceBCELoss, layer_pit_bce
from metrics.salience_layers import peak_readout
from models.harmonic_ports import HFTRPS, HarmoF0RPS, HPPNetRPS
from models.harmonic_ports.layer_readout import split_maps
from models.salience_crf import gaussian_layer_target

PORTS = [HarmoF0RPS, HPPNetRPS, HFTRPS]
# A small grid keeps the CRF's O(span * G * T) decode cheap in the test suite;
# the port's own 0-150/300 grid is exercised by the shape test.
SMALL = dict(r_lo=0.0, r_hi=50.0, n_grid=100)


def _tracks(seed=0, n=2, r=4, t=24, step=0.4):
    """Trajectories whose per-frame slew stays well inside the transition band."""
    rng = np.random.default_rng(seed)
    base = np.cumsum(rng.standard_normal((n, 1, t)) * step, axis=-1) + 25.0
    off = rng.uniform(-3.0, 3.0, size=(n, r, 1))
    return torch.tensor(np.clip(base + off, 0.0, 49.0))


@pytest.mark.parametrize("cls", PORTS)
def test_four_layers_ride_the_output_axis(cls):
    """The wire format stays 3-D; the layers are stacked along it."""
    m = cls(n_maps=4).eval()
    y = m(torch.randn(2, 16000))
    assert y.shape == (2, 4 * 300, 32)
    assert split_maps(y, 4).shape == (2, 4, 300, 32)
    assert torch.equal(split_maps(y, 4).reshape(2, 1200, 32), y)


@pytest.mark.parametrize("cls", PORTS)
def test_one_map_keeps_the_old_shape_and_the_old_decoder(cls):
    m = cls(n_maps=1).eval()
    assert m(torch.randn(1, 16000)).shape == (1, 300, 32)
    # n_maps == 1 must fall through to SalienceRPSPredictor's Hungarian path.
    assert m.predict_rps(torch.randn(1, 16000)).shape == (1, 4, 32)


@pytest.mark.parametrize("cls", PORTS)
def test_the_round_trip_is_exact(cls):
    """A PERFECTLY trained model's own target, through its own decode.

    Under `losses.LayerPITSalienceBCELoss` the optimum is
    ``sigmoid(z) == target``, so ``logit(target)`` is what a perfect model
    emits. ``LayerCRFReadout`` takes ``log sigmoid(z)`` of it, which is the log
    of a Gaussian, which a three-point parabolic fit inverts exactly.
    """
    m = cls(n_maps=4, **SMALL).double().eval()
    rps = _tracks()
    tgt = gaussian_layer_target(rps, m.out_freqs, sigma_bins=1.0)
    ideal = torch.logit(tgt.clamp(1e-15, 1 - 1e-15))
    rec = m.decode_salience(ideal.reshape(ideal.shape[0], -1, ideal.shape[-1]))
    assert float((rec - rps).abs().max()) < 1e-6


@pytest.mark.parametrize("cls", PORTS)
def test_a_stopped_rotor_decodes_to_zero_with_no_threshold(cls):
    m = cls(n_maps=4, **SMALL).double().eval()
    rps = torch.zeros(1, 4, 16, dtype=torch.float64)
    rps[0, 2] = 31.0
    tgt = gaussian_layer_target(rps, m.out_freqs, sigma_bins=1.0)
    ideal = torch.logit(tgt.clamp(1e-15, 1 - 1e-15))
    rec = m.decode_salience(ideal.reshape(1, -1, 16))
    assert float(rec[0, 0].abs().max()) < 1e-6
    assert abs(float(rec[0, 2].mean()) - 31.0) < 1e-6


def test_the_readout_and_the_loss_agree_on_which_target_is_optimal():
    """BCE is minimized at the target, and the peak readout inverts it exactly."""
    grid = np.linspace(0.0, 50.0, 100)
    rps = _tracks()
    tgt = gaussian_layer_target(rps, grid, sigma_bins=1.0)
    ideal = torch.logit(tgt.clamp(1e-15, 1 - 1e-15))
    assert float((peak_readout(F.logsigmoid(ideal), grid) - rps).abs().max()) < 1e-9
    at_opt = float(layer_pit_bce(ideal, tgt))
    for _ in range(5):
        worse = float(layer_pit_bce(ideal + torch.randn_like(ideal) * 0.5, tgt))
        assert worse > at_opt


def test_the_loss_does_not_care_which_layer_holds_which_rotor():
    logits = torch.randn(3, 4, 40, 12)
    target = torch.rand(3, 4, 40, 12)
    base = float(layer_pit_bce(logits, target))
    for perm in ([1, 0, 3, 2], [2, 3, 0, 1], [3, 2, 1, 0]):
        assert float(layer_pit_bce(logits[:, perm], target)) == pytest.approx(base, rel=1e-6)
        assert float(layer_pit_bce(logits, target[:, perm])) == pytest.approx(base, rel=1e-6)


@pytest.mark.parametrize("cls", PORTS)
def test_every_parameter_gets_a_gradient(cls):
    m = cls(n_maps=4, **SMALL).train()
    loss_fn = LayerPITSalienceBCELoss(out_fmin=0.0, out_fmax=50.0, out_bins=100)
    logits = m(torch.randn(2, 16000))
    layers, tgt = loss_fn.layers_and_target(logits, _tracks(t=32).float())
    layer_pit_bce(layers, tgt).backward()
    dead = [
        n
        for n, p in m.named_parameters()
        if p.requires_grad and (p.grad is None or not torch.any(p.grad != 0))
    ]
    assert not dead, f"no gradient reached: {dead}"
