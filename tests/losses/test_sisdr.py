"""SI-SDR loss (F1 SE baselines) — matches the metric, differentiable, Frame adapter."""

from __future__ import annotations

import numpy as np
import tdseries as td
import torch

from losses import SISDRLoss, si_sdr_loss
from metrics.separation import si_sdr as si_sdr_metric


def test_si_sdr_loss_matches_metric():
    torch.manual_seed(0)
    ref = torch.randn(4, 16000)
    est = ref * 0.9 + 0.1 * torch.randn(4, 16000)
    loss = float(si_sdr_loss(est, ref))
    per = [float(si_sdr_metric(ref[i].numpy()[None, :], est[i].numpy()[None, :])) for i in range(4)]
    assert abs(loss - (-np.mean(per))) < 1e-2


def test_si_sdr_loss_differentiable():
    ref = torch.randn(2, 8000)
    est = (ref * 0.5).requires_grad_(True)
    si_sdr_loss(est, ref).backward()
    assert est.grad is not None and est.grad.abs().sum() > 0


def test_si_sdr_loss_better_estimate_lower_loss():
    torch.manual_seed(1)
    ref = torch.randn(3, 8000)
    good = ref + 0.01 * torch.randn(3, 8000)
    bad = ref + 1.0 * torch.randn(3, 8000)
    assert float(si_sdr_loss(good, ref)) < float(si_sdr_loss(bad, ref))


def test_sisdr_frame_adapter():
    loss = SISDRLoss(n_channels=None, sr=(16000, 1))
    ref = np.random.default_rng(0).standard_normal(16000).astype(np.float32)
    est = ref * 0.8
    pred = td.Frame({"enhanced": td.uniform(est[None, :], 16000, dims=("batch", "time"))})
    tgt = td.Frame({"target": td.uniform(ref[None, :], 16000, dims=("batch", "time"))})
    out = loss(pred, tgt)
    assert torch.is_tensor(out) and torch.isfinite(out)
