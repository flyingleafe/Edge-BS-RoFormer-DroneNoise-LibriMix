"""Tests for losses.spectral (MultiScaleSTFT, auraloss MRSTFT wrapper)."""

import numpy as np
import tdseries as td
import torch

from losses.spectral import (
    AuraMRSTFTLoss,
    MultiScaleSTFT,
    MultiScaleSTFTLoss,
    multistft_reshape,
)

SR = 16000


def _mono_frame(entry: str, samples: np.ndarray) -> td.Frame:
    return td.Frame({entry: td.uniform(samples.astype(np.float32), SR, dims=("time",))})


def _stereo_frame(entry: str, samples: np.ndarray) -> td.Frame:
    return td.Frame({entry: td.uniform(samples.astype(np.float32), SR, dims=("mic", "time"))})


def test_multiscale_stft_zero_on_identical_signal():
    rng = np.random.default_rng(0)
    x = torch.from_numpy(rng.standard_normal(SR).astype(np.float32))
    core = MultiScaleSTFT(n_ffts=[256, 128], loss_type="L1")
    loss = core(x, x)
    assert loss.shape == ()
    assert loss.item() == 0.0


def test_multiscale_stft_positive_on_different_signals():
    rng = np.random.default_rng(0)
    x = torch.from_numpy(rng.standard_normal(SR).astype(np.float32))
    y = torch.from_numpy(rng.standard_normal(SR).astype(np.float32))
    core = MultiScaleSTFT(n_ffts=[256, 128])
    loss = core(x, y)
    assert loss.item() > 0.0


def test_multistft_reshape_flattens_leading_axes():
    y4 = torch.zeros(2, 3, 1, 100)  # (stem, batch, channel, time)
    y3 = torch.zeros(2, 3, 100)
    y2 = torch.zeros(2, 100)
    r4, _ = multistft_reshape(y4, y4)
    assert r4.shape == (2, 3, 100)
    r3, _ = multistft_reshape(y3, y3)
    assert r3.shape == (2, 3, 100)
    r2, _ = multistft_reshape(y2, y2)
    assert r2.shape == (2, 1, 100)


def test_multiscale_stft_loss_frame_adapter_mono():
    rng = np.random.default_rng(1)
    x = rng.standard_normal(SR).astype(np.float32)
    pred = _mono_frame("enhanced", x)
    target = _mono_frame("target", x)
    loss_fn = MultiScaleSTFTLoss(n_channels=None, n_ffts=[256, 128])
    value = loss_fn(pred, target)
    assert isinstance(value, torch.Tensor)
    assert value.item() == 0.0
    assert "enhanced" in loss_fn.requires_pred.entries
    assert "target" in loss_fn.requires_target.entries


def test_multiscale_stft_loss_frame_adapter_multichannel():
    rng = np.random.default_rng(2)
    x = rng.standard_normal((4, SR)).astype(np.float32)
    y = rng.standard_normal((4, SR)).astype(np.float32)
    pred = _stereo_frame("enhanced", x)
    target = _stereo_frame("target", y)
    loss_fn = MultiScaleSTFTLoss(n_channels=4, n_ffts=[256, 128])
    value = loss_fn(pred, target)
    assert value.item() > 0.0


def test_aura_mrstft_loss_frame_adapter_mono():
    rng = np.random.default_rng(3)
    x = rng.standard_normal(SR).astype(np.float32)
    pred = _mono_frame("enhanced", x)
    target = _mono_frame("target", x)
    loss_fn = AuraMRSTFTLoss(
        n_channels=None,
        fft_sizes=[256],
        hop_sizes=[64],
        win_lengths=[256],
    )
    value = loss_fn(pred, target)
    assert isinstance(value, torch.Tensor)
    assert value.item() >= 0.0
