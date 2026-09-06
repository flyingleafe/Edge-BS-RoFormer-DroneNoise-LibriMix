"""The pairwise rate prior (v2 § 3.6) must CONTAIN the model it is added to.

Three properties, and each one is a gate on the group:

* At initialization the amplitudes are zero, so the prior is identically zero
  and `SlotCombNet` with ``rate_prior=True`` scores bit-identically to the
  corner. A group that moves the corner cannot be ablated against it.
* With a hand-set amplitude the term does what section 3.6 says it does: on a
  FLAT unary the argmax moves to the earlier slot's rate plus the center of the
  bump. This is the only test that reads the sign and the position.
* The gradient reaches the amplitudes through the CRF loss and stays finite
  after one optimizer step. The campaign lost two arms to non-finite steps
  (`docs/experiments/candidate-tests-2026-09-04.md`), so a new parameter group
  is not accepted until one step is shown to be finite.
"""

from __future__ import annotations

import functools

import numpy as np
import torch

from models.comb_slots import SlotCombNet
from models.comb_slots_prior import RatePrior, add_prior, prior_term

# The C1 corner, at the smallest grid and harmonic count that still exercises
# the peel, the claim bank and the CRF loss. This file tests the PRIOR, not the
# decoder, so nothing here needs the deployed size.
KW = dict(
    n_grid=90,
    k_max=8,
    floor_hz=60.0,
    use_checkpoint=False,
    n_iter=0,
    multichannel=True,
    mask_k_max=64,
)


def _net(**kw) -> SlotCombNet:
    return SlotCombNet(**{**KW, **kw})


@functools.cache
def _audio(seed: int = 7, n: int = 16000) -> tuple[torch.Tensor, np.ndarray]:
    """One static comb heard by two microphones: ``(2, N)`` and its labels.

    Per-mic gains plus INDEPENDENT noise, because a slot reads the microphones
    and eight identical copies would test nothing.
    """
    from data_processing.comb_bench import comb_clip

    a, rps, _ = comb_clip(seed=seed, spread=11.0)
    mono = torch.tensor(np.asarray(a, dtype=np.float32)[:n])
    g = torch.Generator().manual_seed(seed)
    x = mono[None] * torch.tensor([[1.0], [0.8]]) + 0.02 * torch.randn(
        2, mono.shape[0], generator=g
    )
    return x.float(), np.asarray(rps, dtype=np.float32)


def test_prior_is_zero_at_initialization():
    """``psi`` is identically zero, whatever the difference is."""
    prior = RatePrior()
    d = torch.linspace(-120.0, 120.0, 97)
    assert torch.allclose(prior(d), torch.zeros_like(d))


def test_net_with_the_prior_reproduces_the_corner():
    """Same audio, same scores, to the last bit."""
    x, _ = _audio()
    off = _net(rate_prior=False).eval()
    on = _net(rate_prior=True).eval()
    on.load_state_dict(off.state_dict(), strict=False)
    with torch.no_grad():
        s_off, _ = off(x)
        s_on, _ = on(x)
    assert torch.equal(s_off, s_on)


def test_a_hand_set_bump_moves_the_argmax_of_a_flat_unary():
    """One bump of amplitude ``v`` at center ``c`` puts the peak at ``r_j + c``.

    The unary is FLAT, so every grid cell scores the same and the prior alone
    decides. The earlier slot sits at 50 rev/s and the bump is at +12, thus the
    argmax must land at 62 rev/s, inside one grid step.
    """
    grid = torch.arange(30.0, 100.0, 0.1)
    prior = RatePrior(n_centers=3, lo=-12.0, hi=12.0, width=2.0)
    with torch.no_grad():
        prior.v[-1] = 5.0  # the +12 rev/s bump, and only it
    prev = torch.full((1, 1, 4), 50.0)  # one earlier slot, four frames
    term = prior_term(prior, grid, prev)
    assert term.shape == (1, len(grid), 4)
    flat = torch.zeros_like(term)
    peak = grid[(flat + term).argmax(dim=1)]
    assert torch.allclose(peak, torch.full_like(peak, 62.0), atol=0.05)
    # ... and the term is the bump itself, not a scaled copy of it
    assert float(term.max()) == 5.0


def test_two_earlier_slots_add():
    """``psi`` over two earlier slots is the sum of the two single-slot terms."""
    grid = torch.arange(30.0, 100.0, 0.5)
    prior = RatePrior(n_centers=4, lo=-20.0, hi=20.0)
    with torch.no_grad():
        prior.v.copy_(torch.tensor([0.5, -1.0, 2.0, 0.25]))
    prev = torch.tensor([[[40.0, 41.0], [70.0, 71.0]]])  # (1, 2, 2)
    both = prior_term(prior, grid, prev)
    one = prior_term(prior, grid, prev[:, :1])
    two = prior_term(prior, grid, prev[:, 1:])
    assert torch.allclose(both, one + two, atol=1e-6)


def test_add_prior_is_a_no_op_without_a_prior_or_a_previous_slot():
    grid = torch.arange(30.0, 40.0, 0.1)
    s = torch.randn(2, len(grid), 5)
    assert add_prior(None, grid, [torch.full((2, 5), 50.0)], s) is s
    assert add_prior(RatePrior(), grid, [], s) is s


def test_one_training_step_stays_finite_and_reaches_the_amplitudes():
    """The CRF loss trains ``v`` through the peel, and the step is finite."""
    x, rps = _audio()
    net = _net(rate_prior=True)
    assert net.rate_prior is not None
    # The labels span the whole 8 s clip; the audio is the first 2 s of it, so
    # the label is cut to the crop's own 63 frames.
    n_t = x.shape[-1] // 512 + 1
    gt = torch.tensor(rps[:, :n_t], dtype=torch.float32)[None]
    # A nonzero start, because a gradient AT the zero corner can be zero by
    # symmetry and would say nothing about the parameter being reachable.
    with torch.no_grad():
        net.rate_prior.v.add_(0.1)
    opt = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=1e-3)
    loss = net.loss(x[None], gt)
    assert torch.isfinite(loss)
    loss.backward()
    grad = net.rate_prior.v.grad
    assert grad is not None and torch.isfinite(grad).all()
    assert float(grad.abs().sum()) > 0.0
    opt.step()
    assert all(torch.isfinite(p).all() for p in net.parameters())
