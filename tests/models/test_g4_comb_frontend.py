"""G4 comb matched-filter front-end (VK-parity criterion 2.3): CombIFFrontEnd
numerics — comb-score peaks on a synthetic 4-rotor comb, IF frequency-consensus
recovery of a sub-grid f0 offset — plus the SimpleConvV2TransformerComb model
smoke mirroring tests/models/test_g2_frontends.py.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from models.frontends import build_frontend
from models.frontends.comb import CombIFFrontEnd
from models.registry import build_model
from models.rps_predictor import ResidualConvBlock2d, SimpleConvV2TransformerComb

SR = 16000


def _comb_signal(f0s: list[float], amp_decay: float = 0.5, phase_step: float = 0.7):
    t = torch.arange(SR, dtype=torch.float64) / SR
    sig = torch.zeros_like(t)
    for f0 in f0s:
        for k in range(1, int(1200 / f0) + 1):
            sig += (1.0 / k**amp_decay) * torch.sin(2 * math.pi * k * f0 * t + phase_step * k)
    sig = (sig / sig.abs().max()).to(torch.float32)
    return sig + 0.01 * torch.randn_like(sig)


def test_comb_if_registered_shape_and_finite():
    fe = build_frontend("comb_if")
    assert isinstance(fe, CombIFFrontEnd)
    assert fe.out_channels == 3
    assert fe.n_rows == 361  # 30..120 rev/s, step 0.25
    audio = torch.randn(2, SR)
    out = fe(audio)
    assert out.shape == (2, 3, 361, SR // 512 + 1)
    assert torch.isfinite(out).all()
    # bounded auxiliary channels
    assert out[:, 1].abs().max() <= 2.0 + 1e-5  # consensus clamp (rev/s)
    assert out[:, 2].min() >= 0.0 and out[:, 2].max() <= 1.0  # occupancy fraction
    assert fe.num_frames(SR) == SR // 512 + 1


def test_comb_score_peaks_at_true_f0_rows():
    """G4-specific (i): a synthetic 4-rotor comb produces comb-score maxima at
    the correct candidate-f0 rows (greedy-NMS top-4, ±2 rows = ±0.5 rev/s)."""
    torch.manual_seed(0)
    fe = build_frontend("comb_if")
    assert isinstance(fe, CombIFFrontEnd)
    f0s = [45.0, 62.5, 80.0, 105.0]
    out = fe(_comb_signal(f0s).unsqueeze(0))
    scores = out[0, 0].mean(dim=-1).clone()  # (R,)

    f0_grid = fe.get_buffer("f0_grid")
    picks = []
    for _ in range(4):
        r = int(scores.argmax())
        picks.append(float(f0_grid[r]))
        scores[max(0, r - 8) : r + 9] = -1e9  # suppress ±2 rev/s
    for f0 in f0s:
        assert min(abs(p - f0) for p in picks) <= 0.5, (f0, sorted(picks))


def test_consensus_recovers_sub_grid_offset():
    """G4-specific (ii): a comb offset +0.2 rev/s from a grid row reads
    ~+0.2 rev/s in the frequency-consensus channel at that row."""
    torch.manual_seed(0)
    fe = build_frontend("comb_if")
    assert isinstance(fe, CombIFFrontEnd)
    out = fe(_comb_signal([60.2], phase_step=0.3).unsqueeze(0))
    row = int(round((60.0 - 30.0) / 0.25))  # the 60.0 rev/s grid row
    cons = out[0, 1, row, 5:25]  # interior frames
    assert (cons.mean() - 0.2).abs() <= 0.05, float(cons.mean())


def test_occupancy_discriminates_comb_rows():
    torch.manual_seed(0)
    fe = build_frontend("comb_if")
    assert isinstance(fe, CombIFFrontEnd)
    out = fe(_comb_signal([60.2], phase_step=0.3).unsqueeze(0))
    row_true = int(round((60.25 - 30.0) / 0.25))
    row_far = int(round((100.0 - 30.0) / 0.25))
    assert out[0, 2, row_true, 5:25].mean() > out[0, 2, row_far, 5:25].mean() + 0.3


def test_registry_builds_comb_model():
    model = build_model("simple_conv_v2_transformer_comb", n_fft=2048, hop_length=512, num_rotors=4)
    assert isinstance(model, SimpleConvV2TransformerComb)


def test_comb_transformer_forward_and_step():
    torch.manual_seed(0)
    model = SimpleConvV2TransformerComb()  # default 2048/512 grid
    block = model.encoder[0]
    assert isinstance(block, ResidualConvBlock2d)
    assert block.conv.in_channels == 3
    audio = torch.randn(2, SR)
    out = model(audio)
    assert out.shape == (2, 4, SR // 512 + 1)
    assert torch.isfinite(out).all()

    model.train()
    target = torch.rand(out.shape) * 50.0 + 40.0
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    opt.zero_grad()
    loss = F.mse_loss(model(audio), target)
    assert torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    opt.step()
    assert torch.isfinite(F.mse_loss(model(audio), target))
