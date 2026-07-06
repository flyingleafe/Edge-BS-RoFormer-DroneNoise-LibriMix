"""Tests for losses.regularizers (smoothness_penalty)."""

import numpy as np
import tdseries as td
import torch

from losses.regularizers import SmoothnessPenalty, smoothness_penalty

SR = 16000


def test_smoothness_penalty_zero_on_linear_ramp():
    # A linear ramp has zero 2nd difference everywhere.
    x = torch.linspace(0.0, 10.0, 50).reshape(1, 1, 50)
    penalty = smoothness_penalty(x, dims=(-1,))
    assert penalty.item() < 1e-8


def test_smoothness_penalty_nonzero_on_curved_signal():
    t = torch.linspace(0.0, 1.0, 50)
    x = (t**2).reshape(1, 1, 50)
    penalty = smoothness_penalty(x, dims=(-1,))
    assert penalty.item() > 0.0


def test_smoothness_penalty_short_axis_contributes_zero():
    # Axis with < 3 elements: guarded, contributes 0 instead of raising.
    x = torch.rand(1, 2)  # dim -1 has only 2 elements
    penalty = smoothness_penalty(x, dims=(-1,))
    assert penalty.item() < 1e-8


def test_smoothness_penalty_sums_over_multiple_dims():
    x = torch.linspace(0.0, 1.0, 5 * 5).reshape(5, 5)  # both dims linear -> 0
    penalty = smoothness_penalty(x, dims=(-2, -1))
    assert penalty.item() < 1e-8

    t = torch.linspace(0.0, 1.0, 5)
    curved = (t**2).reshape(5, 1).expand(5, 5).clone()
    penalty2 = smoothness_penalty(curved, dims=(-2, -1))
    assert penalty2.item() > 0.0


def test_smoothness_penalty_frame_adapter():
    ramp = np.linspace(0.0, 100.0, 40).astype(np.float32)
    rps = np.stack([ramp, ramp[::-1].copy()])  # (2 rotors, 40) both linear
    pred = td.Frame({"rps_pred": td.uniform(rps, 100, dims=("rotor", "time"))})
    target = td.Frame({})

    penalty_fn = SmoothnessPenalty(entry="rps_pred")
    value = penalty_fn(pred, target)
    assert isinstance(value, torch.Tensor)
    assert value.item() < 1e-8
    assert "rps_pred" in penalty_fn.requires_pred.entries
