"""Farassat 1A formulation of the Ffowcs-Williams Hawkings equation."""

from collections.abc import Callable
from typing import cast

import torch


class Farassat1ASolver:
    """Time-domain FWH acoustic propagation using Farassat Formulation 1A.

    Implements the loading-noise terms for compact surface sources:

        4π p'(x,t) = Σ  [ 1/c0 · Ḟ_r/(r(1-M_r)²)
                        + F_r/(r²(1-M_r)²)
                        + 1/c0 · F_r(r·Ṁ_r + c0(M_r - M²))/(r²(1-M_r)³) ]_ret

    where F is total force on a strip, subscript r denotes projection onto
    radiation direction, and _ret means evaluate at retarded time.
    """

    def __init__(self, c0: float = 343.0, rho0: float = 1.225):
        self.c0 = float(c0)
        self.rho0 = float(rho0)

    def solve_retarded_time(
        self,
        t: torch.Tensor,
        x_observer: torch.Tensor,
        y_func: Callable,
        v_func: Callable,
        tau_guess: torch.Tensor | None = None,
        max_iter: int = 8,
        tol: float = 1e-12,
    ) -> torch.Tensor:
        """Vectorized Newton-Raphson retarded time solver.

        Solves  τ + |x - y(τ)|/c0 = t  for each source-observer-time triplet.

        Args:
            t: Observer times, shape [N_t].
            x_observer: Observer position, shape [3] or [N_obs, 3].
            y_func: Callable(tau) -> positions, shape [N_sources, ..., N_t, 3].
            v_func: Callable(tau) -> velocities, shape [N_sources, ..., N_t, 3].
            tau_guess: Initial guess. If None, t - |x|/c0.
            max_iter: Newton iterations.
            tol: Convergence tolerance.

        Returns:
            Retarded times, shape [N_sources, N_t] or [N_sources, N_obs, N_t].
        """
        single_observer = x_observer.dim() == 1

        if tau_guess is None:
            if single_observer:
                R0 = torch.norm(x_observer)
                tau_guess = t - R0 / self.c0  # [N_t]
                with torch.no_grad():
                    y_test = y_func(tau_guess)
                    N_sources = y_test.shape[0]
                tau = cast(torch.Tensor, tau_guess).unsqueeze(0).expand(N_sources, -1)
            else:
                N_obs = x_observer.shape[0]
                R0 = torch.norm(x_observer, dim=-1)  # [N_obs]
                tau_guess = t - R0[:, None] / self.c0  # [N_obs, N_t]
                with torch.no_grad():
                    # y_func only needs a 1D tau to determine N_sources;
                    # passing [N_obs, N_t] would create ambiguous batch dims.
                    y_test = y_func(cast(torch.Tensor, tau_guess)[0])
                    N_sources = y_test.shape[0]
                tau = cast(torch.Tensor, tau_guess).unsqueeze(0).expand(N_sources, -1, -1)
        else:
            tau = cast(torch.Tensor, tau_guess).clone()
        for _ in range(max_iter):
            y = y_func(tau)  # [..., N_t, 3]
            v = v_func(tau)  # [..., N_t, 3]

            if single_observer:
                r_vec = x_observer.view(*([1] * (y.dim() - 1)), 3) - y
            else:
                # x_observer: [N_obs, 3] -> [1, N_obs, 1, ..., 3]
                # y is [N_sources, N_obs, N_t, 3] (or more batch dims)
                n_trailing = max(0, y.dim() - 3)
                r_vec = x_observer.view(1, x_observer.shape[0], *([1] * n_trailing), 3) - y
            r = torch.norm(r_vec, dim=-1)  # [..., N_t]
            r_hat = r_vec / (r[..., None] + 1e-20)  # [..., N_t, 3]

            M = v / self.c0
            M_r = torch.sum(M * r_hat, dim=-1)  # [..., N_t]

            g = tau + r / self.c0 - t.view(*([1] * (tau.dim() - 1)), t.shape[0])
            dg = 1.0 - M_r

            # Newton update
            tau_new = tau - g / (dg + 1e-20)

            # Clamp to avoid divergence
            tau_new = torch.clamp(tau_new, tau - 0.1, tau + 0.1)

            converged = torch.abs(g) < tol
            tau = torch.where(converged, tau, tau_new)

        return tau

    def compute_pressure(
        self,
        t: torch.Tensor,
        x_observer: torch.Tensor,
        y_func: Callable,
        v_func: Callable,
        F_func: Callable,
        Fdot_func: Callable,
        Mdot_func: Callable | None = None,
        include_term3: bool = True,
    ) -> torch.Tensor:
        """Compute acoustic pressure at observer from surface sources.

        Args:
            t: Observer times, shape [N_t].
            x_observer: Observer position, shape [3] or [N_obs, 3].
            y_func: Callable(tau) -> source positions [N_sources, ..., N_t, 3].
            v_func: Callable(tau) -> source velocities [N_sources, ..., N_t, 3].
            F_func: Callable(tau) -> source forces [N_sources, ..., N_t, 3].
            Fdot_func: Callable(tau) -> dF/dtau [N_sources, ..., N_t, 3].
            Mdot_func: Callable(tau) -> dM/dtau [N_sources, ..., N_t, 3]. Optional.
            include_term3: Include the third (acceleration) FWH term.

        Returns:
            Pressure signal p'(t), shape [N_t] or [N_obs, N_t].
        """
        single_observer = x_observer.dim() == 1

        # Solve retarded time
        tau_ret = self.solve_retarded_time(t, x_observer, y_func, v_func)

        # Evaluate all quantities at retarded time
        y = y_func(tau_ret)
        v = v_func(tau_ret)
        F = F_func(tau_ret)
        Fdot = Fdot_func(tau_ret)

        # Radiation geometry
        if single_observer:
            r_vec = x_observer[None, None, :] - y
        else:
            n_trailing = max(0, y.dim() - 3)
            r_vec = x_observer.view(1, x_observer.shape[0], *([1] * n_trailing), 3) - y
        r = torch.norm(r_vec, dim=-1)
        r_hat = r_vec / (r[..., None] + 1e-20)

        # Mach numbers
        M = v / self.c0
        M_r = torch.sum(M * r_hat, dim=-1)
        M_sq = torch.sum(M**2, dim=-1)

        # Doppler factor
        denom = 1.0 - M_r
        denom2 = denom**2
        denom3 = denom**3

        # Projections
        F_r = torch.sum(F * r_hat, dim=-1)
        Fdot_r = torch.sum(Fdot * r_hat, dim=-1)
        F_M = torch.sum(F * M, dim=-1)

        # Term 1: unsteady loading (dominant for tonal rotor noise)
        term1 = Fdot_r / (self.c0 * r * denom2 + 1e-20)

        # Term 2: steady loading (near-field decay)
        term2 = (F_r - F_M) / (r**2 * denom2 + 1e-20)

        # Term 3: acceleration / thickness correction
        if include_term3 and Mdot_func is not None:
            Mdot = Mdot_func(tau_ret)
            Mdot_r = torch.sum(Mdot * r_hat, dim=-1)
            term3 = F_r * (r * Mdot_r + self.c0 * (M_r - M_sq)) / (self.c0 * r**2 * denom3 + 1e-20)
        else:
            term3 = 0.0

        # Sum over all sources
        p_prime = torch.sum(term1 + term2 + term3, dim=0) / (4 * torch.pi)
        return p_prime
