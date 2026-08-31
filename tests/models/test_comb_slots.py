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


def test_union_prefers_a_new_rotor_over_a_duplicate():
    """The property the octave gate needs, stated as the objective's job.

    Exact idempotence is not achievable with soft claims — the comb templates are
    Gaussians with tails, and the tails of a real drone spectrum sit on OTHER
    rotors' lines rather than on silence, so a duplicate always picks up a little.
    What must hold is the ranking: copying a slot must be worth far less than
    putting it on a rotor nobody covers. `max` gives exact idempotence and was
    measured WORSE end to end (typical-idle 11.39 against 6.84), because two
    adjacent rotors genuinely do share bins and only the louder got credit.
    """
    net = _net()
    a, rps, _ = comb_clip(seed=7, spread=11.0)
    au = torch.tensor(a, dtype=torch.float32)[None]
    with torch.no_grad():
        pw = net.spectrum(au)
        floor = torch.ones_like(pw) * float(pw.median())
        n_t = pw.shape[-1]
        grid = net.grid.numpy()

        def claim(rate):
            one = torch.zeros(1, net.gather.n_grid, n_t)
            one[:, int(np.abs(grid - rate).argmin())] = 1.0
            return net.masks(one).unsqueeze(1)

        r0, r1 = float(rps[0].mean()), float(rps[-1].mean())
        base = net.union_evidence(pw, floor, claim(r0))
        dup = net.union_evidence(pw, floor, torch.cat([claim(r0), claim(r0)], 1))
        new = net.union_evidence(pw, floor, torch.cat([claim(r0), claim(r1)], 1))
    gain_dup = float(dup - base)
    gain_new = float(new - base)
    assert gain_new > 2.0 * gain_dup > 0.0   # measured 3.5x


def test_decode_shapes_and_sorting():
    net = _net(n_iter=0)
    a, rps, _ = comb_clip(seed=3, spread=11.0)
    with torch.no_grad():
        out = net.decode(torch.tensor(a, dtype=torch.float32)[None], octave=False)
    assert out.shape[:2] == (1, 4)
    assert bool((out[:, 1:] >= out[:, :-1] - 1e-6).all())
