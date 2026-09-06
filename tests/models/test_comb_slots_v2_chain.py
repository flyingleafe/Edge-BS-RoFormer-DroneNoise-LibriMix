"""The v2 chain options must CONTAIN the chain they extend.

`docs/slot-comb-v2-design.md` sections 3.1 to 3.3 add an OFF state, a grid that
reaches below 30 rev/s, and a learned transition. The design rule for all three
is the campaign's: a new parameter group is a family that holds the current
model at initialization, so a run that helps can be told from a run that only
moved. These tests are that rule, stated once per group.

The grid is small (140 points, 8 harmonics) so the whole file runs in seconds.
Nothing here measures accuracy — the regime table does that.
"""

from __future__ import annotations

import numpy as np
import torch

from models import comb_crf
from models.comb_slots import SlotCombNet

torch.set_num_threads(4)

# One second at 16 kHz, batch 2: 32 frames, which is enough for a chain to
# have transitions and short enough to decode four slots in a moment.
N_SAMP, BATCH = 16000, 2
KW = dict(n_grid=140, k_max=8, use_checkpoint=False, n_iter=0)


def _net(**kw) -> SlotCombNet:
    return SlotCombNet(**{**KW, **kw}).eval()


def _audio(seed: int = 0, n: int = N_SAMP, b: int = BATCH) -> torch.Tensor:
    """``(B, 1, N)``: one microphone per item. A ``(B, N)`` tensor would be read
    as ONE item heard by B microphones, which is what `spectrum` documents."""
    g = torch.Generator().manual_seed(seed)
    return torch.randn(b, 1, n, generator=g)


def _rps(net: SlotCombNet, n_t: int, seed: int = 1) -> torch.Tensor:
    """Four constant in-grid rates per batch item.

    Constant on purpose: a gold path that never steps is inside every band this
    file builds, so the two spans compare their EMISSIONS and not their
    truncation rules.
    """
    g = torch.Generator().manual_seed(seed)
    lo, hi = float(net.grid[0]), float(net.grid[-1])
    r = lo + (hi - lo) * torch.rand(BATCH, 4, 1, generator=g)
    return r.expand(BATCH, 4, n_t).contiguous()


def _n_frames(net: SlotCombNet, n: int = N_SAMP) -> int:
    return n // net.hop_length + 1


# ── 3.1 The OFF state ────────────────────────────────────────────────────────


def test_off_state_starts_at_the_corner():
    """At ``theta0 = -1e4`` the OFF state is unreachable, so nothing moves.

    The loss must agree to 1e-6 and the decoded rates must be EQUAL, not close:
    `logaddexp(x, x - 1e4)` returns `x` bit-for-bit in float32, so the extra
    state costs the corner nothing at all.
    """
    au = _audio()
    base, off = _net(), _net(off_state=True)
    gt = _rps(base, _n_frames(base))
    with torch.no_grad():
        l0, l1 = float(base.loss(au, gt)), float(off.loss(au, gt))
        d0 = base.decode(au, subgrid=False, octave=False, relocate=True)
        d1 = off.decode(au, subgrid=False, octave=False, relocate=True)
    assert abs(l0 - l1) < 1e-6, (l0, l1)
    assert torch.equal(d0, d1)


def test_off_state_takes_a_silent_clip():
    """Lift ``theta0`` and pure noise reads 0 rev/s, at a finite gold-OFF loss.

    White noise has no comb, so every rate explains it equally and the OFF state
    is the right answer. At the initial ``theta0`` the gold-OFF path costs 1e4
    nats per frame, which is finite and enormous — that is the gradient that
    lifts ``theta0`` on data that holds stopped rotors.
    """
    au = _audio(seed=5)
    net = _net(off_state=True)
    n_t = _n_frames(net)
    zero = torch.zeros(BATCH, 4, n_t)
    with torch.no_grad():
        nll_shut = float(net.loss(au, zero))
        net.theta0.fill_(50.0)
        nll_open = float(net.loss(au, zero))
        out = net.decode(au, subgrid=False, octave=False, relocate=True)
    assert np.isfinite(nll_shut) and np.isfinite(nll_open)
    assert nll_open < nll_shut
    assert float(out.abs().max()) == 0.0


def test_off_transitions_score_by_hand():
    """`path_score` of a path that enters and leaves OFF is the manual sum."""
    torch.manual_seed(0)
    n_g, n_t = 5, 6
    s = torch.randn(1, n_g, n_t, dtype=torch.float64)
    u = torch.randn(1, n_t, dtype=torch.float64)
    span, pen = comb_crf.band_penalty(1.5, 3.0, dtype=torch.float64)
    c1, c2 = 0.7, 1.3
    off = comb_crf.Off(u, c1, c2)
    path = torch.tensor([[1, 2, n_g, n_g, 3, 3]])  # ON, ON, OFF, OFF, ON, ON
    want = (
        float(s[0, 1, 0])
        + float(s[0, 2, 1])
        + float(u[0, 2])
        + float(u[0, 3])
        + float(s[0, 3, 4])
        + float(s[0, 3, 5])
        - float(pen[span + 1])  # 1 -> 2
        - c1  # 2 -> OFF
        - 0.0  # OFF -> OFF
        - c2  # OFF -> 3
        - float(pen[span + 0])  # 3 -> 3
    )
    got = float(comb_crf.path_score(s, span, pen, path, off)[0])
    assert abs(got - want) < 1e-12, (got, want)


# ── 3.3 The learned transition ───────────────────────────────────────────────


def test_learned_transition_starts_at_the_hinge():
    """``pen`` reproduces the hinge, and the loss reproduces the corner's.

    The band is widened from a slew of 30 rev/s^2 while the VALUES stay the
    hinge built from 12, continued past its own truncation by the same quadratic
    law. The vector is compared in float64, because the widened hinge reaches
    hundreds of nats where one float32 step is already 1e-5.

    THE LOSS AGREES TO FOUR LAST-PLACE UNITS, NOT EXACTLY. The two bands hold the
    same cost at every offset a path can reach, and the offsets the wider band
    adds cost 721 nats or more, so they weigh ``exp(-721)`` — zero in both
    dtypes. What is left is the summation order: `log_partition` reduces 17 band
    entries instead of 7, and a float32 `logsumexp` of the same numbers in a
    different order lands one last-place unit away. In float64 the same
    comparison agrees to 3.6e-15, which is the proof that nothing else moved.
    """
    net = _net(learned_transition=True)
    base = _net()
    assert net.span > base.span
    half = comb_crf.hinge_half(net.step_free, 40.0, net.span)
    want = torch.cat([half.flip(0)[:-1], half])
    got = net._pen(torch.float64)
    assert float((got - want).abs().max()) < 1e-6

    au = _audio()
    gt = _rps(base, _n_frames(base))
    with torch.no_grad():
        l0, l1 = float(base.loss(au, gt)), float(net.loss(au, gt))
    assert abs(l0 - l1) <= 4.0 * float(np.spacing(np.float32(l0))), (l0, l1)


def test_learned_transition_gets_a_gradient():
    """One optimizer step, and ``d`` has moved on a finite gradient.

    Only the offsets a path can actually reach carry a gradient: the log
    partition weights an offset by ``exp(-pen)``, and the hinge is hundreds of
    nats past the free band. So the test asks for a non-zero maximum, not a
    non-zero entry.
    """
    net = _net(learned_transition=True)
    net.train()
    au = _audio(seed=2)
    gt = _rps(net, _n_frames(net), seed=3)
    opt = torch.optim.SGD(net.parameters(), lr=1e-3)
    before = net.trans.d.detach().clone()
    loss = net.loss(au, gt)
    loss.backward()
    grad = net.trans.d.grad
    assert grad is not None and bool(torch.isfinite(grad).all())
    assert float(grad.abs().max()) > 0.0
    opt.step()
    assert float((net.trans.d.detach() - before).abs().max()) > 0.0


# ── 3.2 The grid from 10 rev/s ───────────────────────────────────────────────


def test_grid_from_ten_takes_every_regime():
    """A rate below the grid is masked, a stopped rotor is OFF, the rest decode.

    ``mask_k_max`` is capped here only for speed: at ``r_lo=10`` its default is
    ``ceil(f_max / r_lo)`` = 750 harmonics, and the full grid's mask bank then
    takes 121 s to build on four CPU threads against 53 s at 30 rev/s.
    """
    net = _net(
        r_lo=10.0,
        n_grid=180,
        mask_k_max=80,
        off_state=True,
        mask_below_grid=True,
    )
    assert float(net.grid[0]) == 10.0
    au = _audio(seed=7)
    n_t = _n_frames(net)
    # Below the grid, in the grid, in the grid, and stopped.
    rates = torch.tensor([3.0, 12.0, 40.0, 0.0])[None, :, None].expand(BATCH, 4, n_t)
    with torch.no_grad():
        loss = float(net.loss(au, rates.contiguous()))
        out = net.decode(au, subgrid=True, octave=False, relocate=True)
    assert np.isfinite(loss), loss
    assert out.shape == (BATCH, 4, n_t)
    assert bool(torch.isfinite(out).all())
    assert bool((out[:, 1:] >= out[:, :-1] - 1e-6).all())
