"""Global coherent fit of a static comb.

WHY A GLOBAL OBJECTIVE. The phase-increment refiner is local: it reads one
frame-to-frame increment at a time, so its precision is set by the demod band
and its reach by that same band. Measured on a four-rotor static comb, it
reaches 0.002 rev/s from a good initialization and cannot get there from the
blind seed (2.744 rev/s). Nothing inside that family closes the gap — joint
attribution, twin guards and per-frame adaptive bands were each measured and
each saturates.

THE STATIC COMB'S OWN STRUCTURE. A static comb is exactly

    y(t) = sum_i sum_k a_ik cos(k Phi_i(t) + phi_ik),   Phi_i = 2 pi int r_i

with `a_ik` and `phi_ik` CONSTANT in time. Given the trajectories, the
amplitudes and phases are linear, so they can be eliminated (variable
projection) and the whole problem becomes a search over trajectories alone. The
eliminated objective is the fraction of the clip's energy the comb explains,

    J(r) = sum_ik |<y, exp(-i k Phi_i)>|^2 / (N/2 ||y||^2)

which is exact when the lines are orthogonal over the clip, and near-exact
otherwise (distinct lines decorrelate as 1/(spacing * T)).

WHAT THIS BUYS. J scores a WHOLE trajectory at once, so it can reject a wrong
rotor assignment that every local measure accepts. What it costs is basin: a
coherent sum over K harmonics and T seconds is sharp to about 1 / (K T) in
rev/s, so it must be annealed in K and T rather than optimized directly.
"""

from __future__ import annotations

import numpy as np
import torch

__all__ = ["coherent_energy", "comb_objective", "BasisTraj"]


def _phase(r: torch.Tensor, sr: float) -> torch.Tensor:
    """Rotor phase from rate: ``(R, N)`` rev/s -> ``(R, N)`` radians."""
    return 2.0 * torch.pi * torch.cumsum(r, dim=-1) / sr


def coherent_energy(
    y: torch.Tensor,
    r: torch.Tensor,
    sr: float,
    k_max: int,
    f_max: float = 7500.0,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fraction of the clip's energy explained by the comb at rates ``r``.

    ``y``: ``(N,)`` real waveform. ``r``: ``(R, N)`` rates in rev/s. Returns a
    scalar in roughly ``[0, 1]``, larger is better.

    Harmonics above ``f_max`` are dropped per rotor with a smooth taper, so the
    objective stays differentiable as a line crosses the limit.
    """
    n = y.shape[-1]
    phi = _phase(r, sr)  # (R, N)
    ks = torch.arange(1, k_max + 1, dtype=y.dtype, device=y.device)
    ang = ks[None, :, None] * phi[:, None, :]  # (R, K, N)
    c = torch.cos(ang) @ y
    s = torch.sin(ang) @ y
    p = c * c + s * s  # (R, K)
    # Smooth high-frequency taper: a line is worth nothing far above f_max.
    f_line = ks[None, :] * r.mean(dim=-1)[:, None]
    taper = torch.sigmoid((f_max - f_line) / (0.02 * f_max))
    if weights is not None:
        taper = taper * weights
    return (p * taper).sum() / (0.5 * n * (y * y).sum())


class BasisTraj:
    """Rate trajectories as a mean plus a low-order cosine basis.

    ``r_i(t) = m_i + sum_{b=1}^{M} c_ib cos(pi b t / T)``. The half-cosine basis
    spans smooth trajectories without forcing periodicity, and ``M`` sets the
    only real regularizer: a rotor speed cannot wiggle faster than the basis
    allows, which is what makes the search low-dimensional.
    """

    def __init__(self, n_rot: int, n_basis: int, n: int, sr: float, dtype=torch.float64):
        t = torch.arange(n, dtype=dtype) / sr
        b = torch.arange(1, n_basis + 1, dtype=dtype)[:, None]
        self.basis = torch.cos(torch.pi * b * t[None, :] / t[-1])  # (M, N)
        self.n_rot = n_rot
        self.n_basis = n_basis

    def fit(self, r: np.ndarray, sr: float) -> torch.Tensor:
        """Least-squares projection of a sampled trajectory onto the basis."""
        rt = torch.as_tensor(np.asarray(r, dtype=np.float64))
        m = rt.mean(dim=-1, keepdim=True)
        d = torch.linalg.lstsq(self.basis.T, (rt - m).T).solution.T  # (R, M)
        return torch.cat([m, d], dim=1)  # (R, 1+M)

    def rates(self, params: torch.Tensor) -> torch.Tensor:
        return params[:, :1] + params[:, 1:] @ self.basis


def comb_objective(y: np.ndarray, sr: float, k_max: int, f_max: float = 7500.0):
    """Return ``f(r_tensor) -> scalar loss`` (negative explained energy)."""
    yt = torch.as_tensor(np.asarray(y, dtype=np.float64))

    def loss(r: torch.Tensor) -> torch.Tensor:
        return -coherent_energy(yt, r, sr, k_max, f_max)

    return loss
