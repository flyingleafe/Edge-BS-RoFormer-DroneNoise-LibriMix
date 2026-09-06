"""The pairwise rate prior across slots — section 3.6 of the v2 design.

WHAT IT IS FOR. The eight 80 rev/s FLY124 clips lose one rotor of four to a
decoy (24 to 42 % of the rotor-frames sit on a duplicate or on an unrelated
rate). The decoder repairs a part of this with the relocate move, which is
coverage ascent with a hand-written acceptance rule. The joint model itself has
no term that says "the four rotors of one airframe are near one another, and
they are not multiples of one another".

WHERE IT ENTERS. `SlotCombNet._forward_ev` peels the slots in sequence, so slot
``i`` is scored after the slots ``j < i`` are decoded. Their paths are constants
for slot ``i``, thus a term that reads them is still an ordinary unary and the
chain, the Viterbi decode and the CRF loss stay unchanged:

    s_i(g, t) += sum_{j<i} psi(r_g - r_j(t))

    psi(d) = sum_m v_m exp(-(d - c_m)^2 / (2 w_m^2))

This is explain-away on the RATE axis, next to the explain-away on the frequency
axis that the claim allocation already does.

THE FAMILY CONTAINS THE CURRENT MODEL. The amplitudes ``v_m`` start at zero, so
``psi`` is identically zero and the corner is bit-identical at initialization.
Training can leave the corner only by lowering the CRF loss.

THE RISK, AND THE TEST FOR IT. The prior can learn the rate spread of the
training rigs and then refuse a correct answer on an unseen airframe. The ramp
windows, where all four rotors move together, keep it honest, and the ablation
must include an unseen rig (`docs/slot-comb-v2-design.md` § 5).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

__all__ = ["RatePrior", "add_prior", "prior_term"]


class RatePrior(nn.Module):
    """``psi(d)``: a sum of Gaussian bumps over the rate difference ``d``, in rev/s.

    Args:
        n_centers: how many bumps. 16 over -70 to 70 rev/s puts one every
            9.3 rev/s, which resolves "the same rate", "a few rev/s apart" and
            "twice the rate" at the airframe speeds this project reads.
        lo, hi: the first and the last center, in rev/s. The range covers the
            whole grid difference: a 10 to 100 rev/s grid gives differences in
            -90 to 90, and a bump at +-70 still reaches the edges through its
            tail.
        width: the standard deviation of every bump, in rev/s. The default is
            the center spacing, so the bumps overlap and ``psi`` is smooth.

    The centers and the log-widths are parameters, so the shape of the prior is
    learned and not only its amplitudes. The amplitudes start at ZERO, which is
    what keeps the zero-parameter corner inside the family.
    """

    def __init__(
        self,
        n_centers: int = 16,
        lo: float = -70.0,
        hi: float = 70.0,
        width: float | None = None,
    ):
        super().__init__()
        if int(n_centers) < 2:
            raise ValueError(f"n_centers must be 2 or more, got {n_centers}")
        centers = torch.linspace(float(lo), float(hi), int(n_centers))
        w = float(width) if width is not None else float(centers[1] - centers[0])
        if w <= 0.0:
            raise ValueError(f"width must be positive, got {w}")
        self.n_centers = int(n_centers)
        self.centers = nn.Parameter(centers)
        self.log_width = nn.Parameter(torch.full((int(n_centers),), math.log(w)))
        self.v = nn.Parameter(torch.zeros(int(n_centers)))

    def extra_repr(self) -> str:
        return f"n_centers={self.n_centers}"

    def forward(self, d: torch.Tensor) -> torch.Tensor:
        """``(...)`` rate differences in rev/s -> ``(...)`` prior, in nats.

        The bump axis is added at the end and summed away, so the peak memory is
        ``n_centers`` times the input. At 16 centers on a ``(2, 900, 63)`` grid
        that is 7 MB, which is small next to the emission's own readings.
        """
        z = (d.unsqueeze(-1) - self.centers.to(d.dtype)) / self.log_width.exp().to(d.dtype)
        return (self.v.to(d.dtype) * torch.exp(-0.5 * z * z)).sum(dim=-1)


def prior_term(prior: RatePrior, grid: torch.Tensor, prev_paths_rps: torch.Tensor) -> torch.Tensor:
    """``sum_j psi(r_g - r_j(t))`` for one slot: ``(B, J, T)`` -> ``(B, G, T)``.

    Args:
        prior: the shared :class:`RatePrior`.
        grid: the rate grid ``(G,)`` in rev/s.
        prev_paths_rps: the decoded rates of the earlier slots, ``(B, J, T)`` in
            rev/s. They are constants of slot ``i``'s chain, so this function
            does not care how they were decoded.

    The slots are accumulated one at a time, so the ``n_centers`` axis exists
    for one slot at a time and not for all of them together.
    """
    if prev_paths_rps.dim() != 3:
        raise ValueError(f"prev_paths_rps must be (B, J, T), got {tuple(prev_paths_rps.shape)}")
    r = grid.to(prev_paths_rps.dtype)[None, :, None]  # (1, G, 1)
    out: torch.Tensor | None = None
    for j in range(int(prev_paths_rps.shape[1])):
        term = prior(r - prev_paths_rps[:, j][:, None, :])  # (B, G, T)
        out = term if out is None else out + term
    if out is None:  # no earlier slot: the prior has nothing to say
        n_b, n_t = int(prev_paths_rps.shape[0]), int(prev_paths_rps.shape[2])
        out = grid.new_zeros((n_b, int(grid.shape[0]), n_t))
    return out


def add_prior(
    prior: RatePrior | None,
    grid: torch.Tensor,
    prev: list[torch.Tensor],
    s: torch.Tensor,
) -> torch.Tensor:
    """Add the prior of the earlier slots to one slot's unary ``(B, G, T)``.

    The hook `SlotCombNet._forward_ev` calls. It returns ``s`` unchanged when
    the prior is off or when this is the first slot, so the peel of the corner
    is untouched.
    """
    if prior is None or not prev:
        return s
    return s + prior_term(prior, grid, torch.stack(prev, dim=1)).to(s.dtype)
