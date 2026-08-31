"""The slot model must contain the deployed peel and keep its claims honest."""
from __future__ import annotations

import numpy as np
import torch

from data_processing.comb_bench import comb_clip
from models.comb_slots import SlotCombNet


def _net(**kw):
    return SlotCombNet(n_grid=180, k_max=16, use_checkpoint=False, **kw).eval()


def test_claims_stay_a_fraction_of_a_bin():
    """A claim is a share of a bin; a sum of harmonic bumps could exceed one."""
    net = _net()
    p = torch.zeros(1, net.gather.n_grid, 5)
    p[:, 40] = 1.0
    d = net.masks(p)
    assert float(d.min()) >= 0.0 and float(d.max()) <= 1.0 + 1e-6


def test_identical_rotors_leave_the_ranking_intact():
    """Share normalization, not mutual erasure: four slots on one rate survive.

    A plain mutual notch would have every slot delete every other slot's
    evidence, which is wrong precisely where a collapsed answer is correct.
    """
    net = _net()
    a, _, _ = comb_clip(seed=7, spread=0.0)
    au = torch.tensor(a, dtype=torch.float32)[None]
    with torch.no_grad():
        pw = net.spectrum(au)
        floor = torch.ones_like(pw) * 1e-6
        one = torch.zeros(1, net.gather.n_grid, pw.shape[-1])
        one[:, 90] = 1.0
        claims = torch.stack([net.masks(one)] * 4, dim=1)
        res = net._residual(pw, floor, claims, 0)
        # Four equal claimants: each keeps a quarter, and none is erased.
        line = net.masks(one)[0] > 0.9
        assert float((res[0][line] / pw[0][line]).mean()) > 0.2


def test_octave_move_needs_the_union_to_improve():
    """The gate is coverage, and coverage counts a bin once."""
    net = _net()
    a, _, _ = comb_clip(seed=7, spread=11.0)
    au = torch.tensor(a, dtype=torch.float32)[None]
    with torch.no_grad():
        pw = net.spectrum(au)
        floor = torch.ones_like(pw) * float(pw.median())
        one = torch.zeros(1, net.gather.n_grid, pw.shape[-1]); one[:, 90] = 1.0
        c1 = net.masks(one).unsqueeze(1)
        single = net.union_evidence(pw, floor, c1)
        double = net.union_evidence(pw, floor, torch.cat([c1, c1], dim=1))
    assert float(abs(double - single)) < 1e-3 * max(float(single), 1.0)


def test_decode_shapes_and_sorting():
    net = _net(n_iter=0)
    a, rps, _ = comb_clip(seed=3, spread=11.0)
    with torch.no_grad():
        out = net.decode(torch.tensor(a, dtype=torch.float32)[None], octave=False)
    assert out.shape[:2] == (1, 4)
    assert bool((out[:, 1:] >= out[:, :-1] - 1e-6).all())
