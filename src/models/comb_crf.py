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
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

__all__ = ["band_penalty", "viterbi", "log_partition", "path_score",
           "posterior_marginals", "crf_nll"]

NEG = -1e30  # a finite stand-in for -inf: -inf * 0 = nan in the backward pass


def band_penalty(step_free: float, stiff: float, device=None, dtype=torch.float32):
    """The hinge transition cost, as a ``(2*span+1,)`` vector of penalties.

    ``step_free`` is the number of grid steps a rotor may move between frames at
    no cost — the physical slew rate divided by the grid step and multiplied by
    the frame period. The band is truncated at four times that, exactly as the
    classical implementation truncates it.
    """
    step_free = max(float(step_free), 1e-9)
    span = max(1, int(round(4.0 * step_free)))
    offs = torch.arange(-span, span + 1, device=device, dtype=dtype)
    excess = (offs.abs() - step_free).clamp_min(0.0)
    return span, float(stiff) * (excess / step_free) ** 2


def _shifted(prev: torch.Tensor, span: int) -> torch.Tensor:
    """``(B, G) -> (B, 2*span+1, G)`` with ``out[:, a, i] = prev[:, i + a - span]``.

    Out-of-range sources are `NEG`, so a path cannot enter or leave the grid.
    """
    padded = F.pad(prev, (span, span), value=NEG)
    return padded.unfold(-1, prev.shape[-1], 1)


def viterbi(scores: torch.Tensor, span: int, pen: torch.Tensor) -> torch.Tensor:
    """Best path per batch item: ``(B, G, T)`` scores -> ``(B, T)`` grid indices.

    The same recursion `log_partition` runs with `max` in place of `logsumexp`,
    so the loss below is trained on exactly the object this returns.
    """
    b, n_g, n_t = scores.shape
    best = scores[:, :, 0]
    back = scores.new_zeros((n_t, b, n_g), dtype=torch.long)
    ar = torch.arange(n_g, device=scores.device)
    for t in range(1, n_t):
        cand = _shifted(best, span) - pen[None, :, None]
        val, a = cand.max(dim=1)
        back[t] = (ar[None, :] + a - span).clamp(0, n_g - 1)
        best = val + scores[:, :, t]
    path = scores.new_zeros((b, n_t), dtype=torch.long)
    path[:, -1] = best.argmax(dim=1)
    for t in range(n_t - 1, 0, -1):
        path[:, t - 1] = back[t].gather(1, path[:, t: t + 1]).squeeze(1)
    return path


def log_partition(scores: torch.Tensor, span: int, pen: torch.Tensor) -> torch.Tensor:
    """``log Z`` over every path through the grid: ``(B, G, T) -> (B,)``."""
    n_t = scores.shape[2]
    a = scores[:, :, 0]
    for t in range(1, n_t):
        a = torch.logsumexp(_shifted(a, span) - pen[None, :, None], dim=1) + scores[:, :, t]
    return torch.logsumexp(a, dim=1)


def path_score(scores: torch.Tensor, span: int, pen: torch.Tensor,
               path: torch.Tensor) -> torch.Tensor:
    """Score of one given path: ``(B, G, T)``, ``(B, T)`` -> ``(B,)``.

    Steps outside the band are impossible under the model, so they are charged
    the band edge's penalty rather than silently ignored — a gold trajectory that
    slews harder than the band allows must cost something, or the loss would
    reward the model for a path it can never decode.
    """
    emit = scores.gather(1, path.unsqueeze(1)).squeeze(1).sum(dim=1)
    d = (path[:, 1:] - path[:, :-1]).abs()
    trans = pen[span + d.clamp(max=span)]
    over = (d > span).to(scores.dtype) * pen[0]
    return emit - trans.sum(dim=1) - over.sum(dim=1)


def posterior_marginals(scores: torch.Tensor, span: int, pen: torch.Tensor) -> torch.Tensor:
    """``p(g_t = g | scores)`` by forward-backward: ``(B, G, T) -> (B, G, T)``.

    This is the slot model's soft rate distribution. It is used rather than a
    per-frame softmax because it already carries the temporal model — a frame
    whose own evidence is ambiguous inherits its neighbours' certainty, which is
    the whole reason the decoder is a path and not a peak.
    """
    b, n_g, n_t = scores.shape
    fwd = scores.new_empty((b, n_g, n_t))
    a = scores[:, :, 0]
    fwd[:, :, 0] = a
    for t in range(1, n_t):
        a = torch.logsumexp(_shifted(a, span) - pen[None, :, None], dim=1) + scores[:, :, t]
        fwd[:, :, t] = a
    bwd = scores.new_zeros((b, n_g, n_t))
    z = scores.new_zeros((b, n_g))
    for t in range(n_t - 1, 0, -1):
        # The penalty is symmetric, so the same band serves both directions.
        z = torch.logsumexp(_shifted(z + scores[:, :, t], span) - pen[None, :, None], dim=1)
        bwd[:, :, t - 1] = z
    return torch.softmax(fwd + bwd, dim=1)


def crf_nll(scores: torch.Tensor, span: int, pen: torch.Tensor,
            path: torch.Tensor) -> torch.Tensor:
    """``log Z - score(gold)``, per batch item. Always non-negative up to float error."""
    return log_partition(scores, span, pen) - path_score(scores, span, pen, path)
