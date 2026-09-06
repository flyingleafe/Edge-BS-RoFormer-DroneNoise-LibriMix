"""Linear-chain CRF over a rate grid — the loss that matches the deployed decoder.

WHY THIS EXISTS. The comb-salience campaign trained with binary cross-entropy on
the salience map and selected checkpoints with an argmax decoder, then deployed a
Viterbi decoder. The trained head's advantage did not survive the switch
(`close` 0.459 under argmax, 0.669 under Viterbi), which is a
selection/deployment mismatch rather than a fact about the head.

The fix is to train through the decoder. The deployed decoder maximizes

    path score  =  sum_t S(g_t, t)  -  sum_t pen(g_t - g_{t-1})

over paths through the rate grid. Replacing the `max` of that recursion with
`logsumexp` gives `log Z` over all paths, and

    loss  =  log Z  -  score(gold path)

is the exact negative log-likelihood of the true rotor trajectory under the model
the decoder assumes. Selection and deployment then use one object by
construction. `viterbi` and `log_partition` here share `_band`, so the transition
structure cannot drift between them.

NOT CTC. CTC exists to marginalize an unknown alignment between a short label
sequence and many frames; its blank symbol and collapse rule solve that problem.
Here the labels are dense — there is a true rate in every frame — so the
alignment CTC integrates over does not exist, and its machinery would add a
degree of freedom the task does not have.

THE TRANSITION IS THE CLASSICAL ONE. `pen` is the hinge of
`tracking.comb_seed._viterbi_ridge`: free up to the airframe's physical slew,
quadratic past it, truncated to the same +-`span` band. `test_comb_crf.py` locks
`viterbi` to that function's output index-for-index.

TWO OPT-IN EXTENSIONS (slot-comb v2, `docs/slot-comb-v2-design.md`).

* `Off` adds ONE state to the chain, next to the banded ON states. It is the
  "no rotor turns" state. Every function below takes it as `off=None` by
  default, and at that default the code path is the one measured before.
* `BandPenalty` makes `pen` a learned vector instead of a fixed hinge. It is
  symmetric, it is zero at the origin, and it never decreases, so it stays a
  transition COST. Its initial value is the hinge, to float64 precision.

Both keep the design invariant of v2: the new family contains the old model at
initialization, and training can leave it only by lowering the CRF loss.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "Off",
    "BandPenalty",
    "band_penalty",
    "hinge_half",
    "viterbi",
    "log_partition",
    "path_score",
    "posterior_marginals",
    "crf_nll",
]

NEG = -1e30  # a finite stand-in for -inf: -inf * 0 = nan in the backward pass

#: `d_i` for a step of size zero. `softplus(-30)` is 9.4e-14, so `span` of them
#: add 3.6e-12 to the penalty. That is far below the 1e-6 the parity test asks
#: for, and it keeps the parameter finite and differentiable.
_DEAD = -30.0


class Off(NamedTuple):
    """The OFF state of one chain: its unary per frame, and the two switch costs.

    ``u`` is ``(B, T)``, the unary of the OFF state. The slot model builds it as
    ``theta0 - theta1 * contrast(t)``, an affine function of a statistic of the
    same scores the chain reads. ``c1`` charges ON -> OFF and ``c2`` charges
    OFF -> ON. OFF -> OFF is free. Both may be floats or 0-dim tensors.

    THE OFF INDEX IS ``G``, one past the last grid index. A path that `viterbi`
    returns, and a gold path that `path_score` reads, holds ``G`` in every OFF
    frame. A path therefore stays a ``(B, T)`` long tensor, and every ON index
    keeps the meaning it had.
    """

    u: torch.Tensor
    c1: torch.Tensor | float
    c2: torch.Tensor | float


def _neg(dtype: torch.dtype) -> float:
    """`NEG`, clipped to something the dtype can actually hold.

    Under AMP the scores arrive as float16, whose largest finite magnitude is
    65504, so padding with -1e30 raises "value cannot be converted to type
    at::Half without overflow". A quarter of the dtype's minimum is still far
    below any reachable path score (log-sigmoid emissions summed over a clip),
    so it blocks the out-of-grid transitions exactly as -1e30 does in float32.
    """
    return max(NEG, float(torch.finfo(dtype).min) / 4.0)


def hinge_half(step_free: float, stiff: float, span: int, dtype=torch.float64) -> torch.Tensor:
    """The hinge cost at offsets ``0..span``: a ``(span+1,)`` non-decreasing vector.

    This is the right half of `band_penalty`, continued past that function's own
    truncation by the same quadratic law. `BandPenalty` starts here, so a wider
    band holds the old hinge unchanged inside it.
    """
    step_free = max(float(step_free), 1e-9)
    j = torch.arange(int(span) + 1, dtype=dtype)
    excess = (j - step_free).clamp_min(0.0)
    return float(stiff) * (excess / step_free) ** 2


def band_penalty(step_free: float, stiff: float, device=None, dtype=torch.float32):
    """The hinge transition cost, as a ``(2*span+1,)`` vector of penalties.

    ``step_free`` is the number of grid steps a rotor may move between frames at
    no cost — the physical slew rate divided by the grid step and multiplied by
    the frame period. The band is truncated at four times that, exactly as the
    classical implementation truncates it.

    THE VALUES COME FROM `hinge_half`, so this function and `BandPenalty` are
    ONE formula. Built directly in float32 the two disagree by a last-place unit
    at the reachable offsets, which is enough to move a float32 CRF loss by its
    own last place, and the v2 parity test would then have nothing to measure.
    """
    step_free = max(float(step_free), 1e-9)
    span = max(1, int(round(4.0 * step_free)))
    half = hinge_half(step_free, stiff, span)
    pen = torch.cat([half.flip(0)[:-1], half]).to(dtype)
    return span, pen if device is None else pen.to(device)


def _inv_softplus(y: torch.Tensor) -> torch.Tensor:
    """``x`` with ``softplus(x) == y``, for ``y >= 0``. Zero maps to `_DEAD`.

    Written as ``y + log1p(-exp(-y))`` rather than ``log(exp(y) - 1)``. The
    successive differences of the hinge reach 180 nats at the edge of a wide
    band, and ``exp(180)`` is not a float32 number.
    """
    y = y.to(torch.float64)
    safe = y.clamp_min(1e-12)
    x = safe + torch.log1p(-torch.exp(-safe))
    return torch.where(y > 0.0, x, torch.full_like(x, _DEAD))


class BandPenalty(nn.Module):
    """A learned transition cost over the band, that starts at the hinge.

    WHY. The hinge is built once from a slew rate of 12 rev/s^2 and a stiffness
    of 40. A real DREGON ramp runs at 20 to 40 rev/s^2, which is 6 to 10 grid
    steps per frame against a free band of 3.8 steps, so the hinge charges about
    90 nats per frame to follow it against a salience difference of about 1 nat.
    The chain therefore lags and then jumps. The shape of that cost is not a
    measurement, so it is made a parameter.

    THE PARAMETERIZATION IS THE COST'S OWN SHAPE. ``pen(j) = sum_{i<=j}
    softplus(d_i)`` for ``j = 1..span``, and ``pen(0) = 0``. The vector is then
    mirrored, so the cost is symmetric in the offset, is zero at the origin, and
    never decreases. Those three properties are what make it a transition cost
    and not a free per-offset bias, and a free bias could pay a slot to jump.

    THE PARAMETER AND THE SUM ARE FLOAT64. The hinge reaches 3166 nats at the
    edge of a band built from 30 rev/s^2. A float32 `d` holds the largest step
    (183 nats) to 8e-6, and a float32 cumulative sum of 38 terms loses 2e-4 more,
    so neither reproduces the hinge to the 1e-6 the parity test asks for. The
    vector is 38 numbers, so the double costs nothing. `forward` casts the result
    to the dtype the caller asks for, and a model that is cast to float32 still
    works — it only loses those 8e-6 nats.
    """

    def __init__(self, span: int, half: torch.Tensor):
        super().__init__()
        self.span = int(span)
        half = torch.as_tensor(half, dtype=torch.float64)
        if half.numel() != self.span + 1:
            raise ValueError(f"half has {half.numel()} entries, span {self.span} needs {span + 1}")
        if float(half[0]) != 0.0:
            raise ValueError("the cost at offset zero must be zero")
        self.d = nn.Parameter(_inv_softplus(half[1:] - half[:-1]))

    def forward(self, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """The full ``(2*span+1,)`` band, in the requested dtype."""
        step = F.softplus(self.d.to(torch.float64))
        half = torch.cat([step.new_zeros(1), torch.cumsum(step, dim=0)])
        return torch.cat([half.flip(0)[:-1], half]).to(dtype)


def _shifted(prev: torch.Tensor, span: int) -> torch.Tensor:
    """``(B, G) -> (B, 2*span+1, G)`` with ``out[:, a, i] = prev[:, i + a - span]``.

    Out-of-range sources are `NEG`, so a path cannot enter or leave the grid.
    """
    padded = F.pad(prev, (span, span), value=_neg(prev.dtype))
    return padded.unfold(-1, prev.shape[-1], 1)


def _off_parts(off: Off, dtype: torch.dtype):
    return off.u.to(dtype), off.c1, off.c2


def viterbi(
    scores: torch.Tensor, span: int, pen: torch.Tensor, off: Off | None = None
) -> torch.Tensor:
    """Best path per batch item: ``(B, G, T)`` scores -> ``(B, T)`` grid indices.

    The same recursion `log_partition` runs with `max` in place of `logsumexp`,
    so the loss below is trained on exactly the object this returns. With `off`
    the path may hold the index ``G``, which is the OFF state.
    """
    b, n_g, n_t = scores.shape
    ar = torch.arange(n_g, device=scores.device)
    if off is None:
        best = scores[:, :, 0]
        back = scores.new_zeros((n_t, b, n_g), dtype=torch.long)
        for t in range(1, n_t):
            cand = _shifted(best, span) - pen[None, :, None]
            val, a = cand.max(dim=1)
            back[t] = (ar[None, :] + a - span).clamp(0, n_g - 1)
            best = val + scores[:, :, t]
        path = scores.new_zeros((b, n_t), dtype=torch.long)
        path[:, -1] = best.argmax(dim=1)
        for t in range(n_t - 1, 0, -1):
            path[:, t - 1] = back[t].gather(1, path[:, t : t + 1]).squeeze(1)
        return path

    u, c1, c2 = _off_parts(off, scores.dtype)
    best_on, best_off = scores[:, :, 0], u[:, 0]
    back = scores.new_zeros((n_t, b, n_g + 1), dtype=torch.long)
    for t in range(1, n_t):
        cand = _shifted(best_on, span) - pen[None, :, None]
        val, a = cand.max(dim=1)
        src = (ar[None, :] + a - span).clamp(0, n_g - 1)
        # A tie goes to the ON state. At the default `theta0` the OFF branch is
        # 1e4 nats down, so this branch never wins and the path is the one the
        # `off=None` recursion returns.
        from_off = (best_off - c2)[:, None]
        take_off = from_off > val
        back[t, :, :n_g] = torch.where(take_off, torch.full_like(src, n_g), src)
        new_on = torch.where(take_off, from_off, val) + scores[:, :, t]
        on_val, on_idx = best_on.max(dim=1)
        take_on = (on_val - c1) > best_off
        back[t, :, n_g] = torch.where(take_on, on_idx, torch.full_like(on_idx, n_g))
        best_on, best_off = new_on, torch.where(take_on, on_val - c1, best_off) + u[:, t]
    fin = torch.cat([best_on, best_off[:, None]], dim=1)
    path = scores.new_zeros((b, n_t), dtype=torch.long)
    path[:, -1] = fin.argmax(dim=1)
    for t in range(n_t - 1, 0, -1):
        path[:, t - 1] = back[t].gather(1, path[:, t : t + 1]).squeeze(1)
    return path


def log_partition(
    scores: torch.Tensor, span: int, pen: torch.Tensor, off: Off | None = None
) -> torch.Tensor:
    """``log Z`` over every path through the grid: ``(B, G, T) -> (B,)``.

    With `off` the sum also covers every path that visits the OFF state.
    """
    n_t = scores.shape[2]
    if off is None:
        a = scores[:, :, 0]
        for t in range(1, n_t):
            a = torch.logsumexp(_shifted(a, span) - pen[None, :, None], dim=1) + scores[:, :, t]
        return torch.logsumexp(a, dim=1)

    u, c1, c2 = _off_parts(off, scores.dtype)
    a_on, a_off = scores[:, :, 0], u[:, 0]
    for t in range(1, n_t):
        from_on = torch.logsumexp(_shifted(a_on, span) - pen[None, :, None], dim=1)
        new_on = torch.logaddexp(from_on, (a_off - c2)[:, None]) + scores[:, :, t]
        new_off = torch.logaddexp(a_off, torch.logsumexp(a_on, dim=1) - c1) + u[:, t]
        a_on, a_off = new_on, new_off
    return torch.logaddexp(torch.logsumexp(a_on, dim=1), a_off)


def path_score(
    scores: torch.Tensor,
    span: int,
    pen: torch.Tensor,
    path: torch.Tensor,
    off: Off | None = None,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Score of one given path: ``(B, G, T)``, ``(B, T)`` -> ``(B,)``.

    Steps outside the band are impossible under the model, so they are charged
    the band edge's penalty rather than silently ignored — a gold trajectory that
    slews harder than the band allows must cost something, or the loss would
    reward the model for a path it can never decode.

    ``mask`` is ``(B, T)``, and a False frame is a FREE STATE. Its emission
    contributes zero, and so does each of the two transitions that touch it. The
    gold trajectory is then only the observed part of the path, which is what
    "this frame is outside the loss" means for a rate the grid cannot hold. Note
    that `log_partition` still sums over those frames, so the masked frames push
    every in-grid score down and nothing pulls one back up. That is the intent
    for a rate below the grid, because no in-grid rate explains such a frame.
    """
    n_g = scores.shape[1]
    if off is None:
        emit = scores.gather(1, path.unsqueeze(1)).squeeze(1)
        d = (path[:, 1:] - path[:, :-1]).abs()
        trans = pen[span + d.clamp(max=span)] + (d > span).to(scores.dtype) * pen[0]
    else:
        u, c1, c2 = _off_parts(off, scores.dtype)
        is_off = path >= n_g
        idx = path.clamp(max=n_g - 1)
        emit = torch.where(is_off, u, scores.gather(1, idx.unsqueeze(1)).squeeze(1))
        prev_off, cur_off = is_off[:, :-1], is_off[:, 1:]
        d = (idx[:, 1:] - idx[:, :-1]).abs()
        on_on = pen[span + d.clamp(max=span)] + (d > span).to(scores.dtype) * pen[0]
        zero = torch.zeros_like(on_on)
        trans = torch.where(
            prev_off & cur_off,
            zero,
            torch.where(prev_off, zero + c2, torch.where(cur_off, zero + c1, on_on)),
        )
    if mask is not None:
        m = mask.to(scores.dtype)
        emit = emit * m
        trans = trans * (m[:, 1:] * m[:, :-1])
    return emit.sum(dim=1) - trans.sum(dim=1)


def posterior_marginals(
    scores: torch.Tensor, span: int, pen: torch.Tensor, off: Off | None = None
) -> torch.Tensor:
    """``p(g_t = g | scores)`` by forward-backward: ``(B, G, T) -> (B, G, T)``.

    This is the slot model's soft rate distribution. It is used rather than a
    per-frame softmax because it already carries the temporal model — a frame
    whose own evidence is ambiguous inherits its neighbours' certainty, which is
    the whole reason the decoder is a path and not a peak.

    With `off` the output is ``(B, G+1, T)``, and row ``G`` is the posterior of
    the OFF state.
    """
    b, n_g, n_t = scores.shape
    fwd = scores.new_empty((b, n_g, n_t))
    bwd = scores.new_zeros((b, n_g, n_t))
    if off is None:
        a = scores[:, :, 0]
        fwd[:, :, 0] = a
        for t in range(1, n_t):
            a = torch.logsumexp(_shifted(a, span) - pen[None, :, None], dim=1) + scores[:, :, t]
            fwd[:, :, t] = a
        z = scores.new_zeros((b, n_g))
        for t in range(n_t - 1, 0, -1):
            # The penalty is symmetric, so the same band serves both directions.
            z = torch.logsumexp(_shifted(z + scores[:, :, t], span) - pen[None, :, None], dim=1)
            bwd[:, :, t - 1] = z
        return torch.softmax(fwd + bwd, dim=1)

    u, c1, c2 = _off_parts(off, scores.dtype)
    f_off = scores.new_empty((b, n_t))
    b_off = scores.new_zeros((b, n_t))
    a_on, a_off = scores[:, :, 0], u[:, 0]
    fwd[:, :, 0], f_off[:, 0] = a_on, a_off
    for t in range(1, n_t):
        from_on = torch.logsumexp(_shifted(a_on, span) - pen[None, :, None], dim=1)
        new_on = torch.logaddexp(from_on, (a_off - c2)[:, None]) + scores[:, :, t]
        new_off = torch.logaddexp(a_off, torch.logsumexp(a_on, dim=1) - c1) + u[:, t]
        a_on, a_off = new_on, new_off
        fwd[:, :, t], f_off[:, t] = a_on, a_off
    z_on, z_off = scores.new_zeros((b, n_g)), scores.new_zeros((b,))
    for t in range(n_t - 1, 0, -1):
        m_on, m_off = z_on + scores[:, :, t], z_off + u[:, t]
        nz_on = torch.logaddexp(
            torch.logsumexp(_shifted(m_on, span) - pen[None, :, None], dim=1),
            (m_off - c1)[:, None],
        )
        nz_off = torch.logaddexp(torch.logsumexp(m_on, dim=1) - c2, m_off)
        z_on, z_off = nz_on, nz_off
        bwd[:, :, t - 1], b_off[:, t - 1] = z_on, z_off
    both = torch.cat([fwd + bwd, (f_off + b_off)[:, None, :]], dim=1)
    return torch.softmax(both, dim=1)


def crf_nll(
    scores: torch.Tensor,
    span: int,
    pen: torch.Tensor,
    path: torch.Tensor,
    off: Off | None = None,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """``log Z - score(gold)``, per batch item. Always non-negative up to float error.

    A `mask` makes the gold path free at the masked frames, so the difference is
    no longer a negative log-probability there and can fall below zero. Read
    `path_score` for what a masked frame costs and for why.
    """
    return log_partition(scores, span, pen, off) - path_score(scores, span, pen, path, off, mask)
