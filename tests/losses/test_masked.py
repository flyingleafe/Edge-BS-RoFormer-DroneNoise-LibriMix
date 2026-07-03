"""Tests for losses.masked (quantile-masked MSE)."""

import numpy as np
import tdseries as td
import torch

from losses.masked import MaskedLoss, masked_loss

SR = 16000


def test_masked_loss_zero_on_identical_signal():
    y = torch.rand(1, 2, 3, 40)  # (stems, batch, channels, length)
    loss = masked_loss(y, y, q=0.9, coarse=False)
    assert loss.item() == 0.0


def test_masked_loss_suppresses_the_worst_pixel():
    torch.manual_seed(0)
    stems, batch, ch, length = 1, 1, 1, 100
    y_ = torch.zeros(stems, batch, ch, length)
    y = torch.zeros(stems, batch, ch, length)
    # One huge outlier error; the rest are tiny.
    y_[0, 0, 0, 0] = 1000.0
    full_mse = torch.nn.functional.mse_loss(y_, y).item()
    masked = masked_loss(y_, y, q=0.5, coarse=False).item()
    # The masked loss must exclude the outlier's huge contribution.
    assert masked < full_mse


def test_masked_loss_coarse_reduces_extra_axes():
    torch.manual_seed(1)
    y_ = torch.rand(2, 3, 4, 10)
    y = torch.rand(2, 3, 4, 10)
    loss = masked_loss(y_, y, q=0.75, coarse=True)
    assert loss.shape == ()
    assert loss.item() >= 0.0


def test_masked_loss_frame_adapter_mono():
    rng = np.random.default_rng(2)
    x = rng.standard_normal(4000).astype(np.float32)
    pred = td.Frame({"enhanced": td.uniform(x, SR, dims=("time",))})
    target = td.Frame({"target": td.uniform(x, SR, dims=("time",))})

    loss_fn = MaskedLoss(q=0.9)
    value = loss_fn(pred, target)
    assert isinstance(value, torch.Tensor)
    assert value.item() == 0.0


def test_masked_loss_frame_adapter_nonzero_on_different_signals():
    rng = np.random.default_rng(3)
    x = rng.standard_normal(4000).astype(np.float32)
    y = rng.standard_normal(4000).astype(np.float32)
    pred = td.Frame({"enhanced": td.uniform(x, SR, dims=("time",))})
    target = td.Frame({"target": td.uniform(y, SR, dims=("time",))})

    loss_fn = MaskedLoss(q=0.9)
    value = loss_fn(pred, target)
    assert value.item() > 0.0
