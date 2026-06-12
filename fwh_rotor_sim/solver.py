"""High-level FWH rotor acoustic solver."""

from collections.abc import Callable
from typing import Any, cast

import torch

from .bemt import BEMTAerodynamics
from .fwh import Farassat1ASolver
from .geometry import Rotor


class FWHRotorSolver:
    """End-to-end rotor acoustic field simulator.

    Combines BEMT aerodynamics with Farassat 1A FWH acoustic propagation.
    """

    def __init__(
        self,
        rotor: Rotor,
        c0: float = 343.0,
        rho0: float = 1.225,
        airfoil: Any = None,
        source_dt: float | None = None,
    ):
        """
        Args:
            rotor: Rotor geometry.
            c0: Speed of sound (m/s).
            rho0: Air density (kg/m³).
            airfoil: Airfoil polar model. Default: thin airfoil.
            source_dt: Time step for source-time sampling when Omega is
                       a callable. If None, auto-determined.
        """
        self.rotor = rotor
        self.c0 = c0
        self.rho0 = rho0
        self.aero = BEMTAerodynamics(rho=rho0, airfoil=airfoil)
        self.fwh = Farassat1ASolver(c0=c0, rho0=rho0)
        self.source_dt = source_dt

    def _integrate_azimuth(
        self,
        tau: torch.Tensor,
        Omega: Callable[[torch.Tensor], torch.Tensor] | float | torch.Tensor,
    ) -> torch.Tensor:
        """Integrate Omega(tau) to get azimuth angle psi."""
        # If Omega is constant or tensor, analytic
        if isinstance(Omega, (int, float, torch.Tensor)):
            Omega_val = float(Omega) if not isinstance(Omega, torch.Tensor) else Omega.item()
            return Omega_val * tau

        # Omega is callable: need numerical integration
        # For variable speed, sample on a fine grid and integrate
        if self.source_dt is None:
            # Auto: use 1/100th of rotation period at mean Omega
            with torch.no_grad():
                O_mean = Omega(torch.tensor(0.0))
                if isinstance(O_mean, torch.Tensor):
                    O_mean = O_mean.item()
                dt = max(1e-4, 0.01 * 2 * torch.pi / abs(O_mean))
            self.source_dt = dt

        # Build source-time grid covering [tau_min, tau_max]
        tau_min = tau.min().item()
        tau_max = tau.max().item()
        n_src = max(1000, int((tau_max - tau_min) / self.source_dt) + 1)
        tau_grid = torch.linspace(tau_min, tau_max, n_src)

        # Evaluate Omega on grid
        with torch.no_grad():
            O_vals = torch.stack([Omega(tg) for tg in tau_grid])
            # Trapezoidal integration
            dpsi = torch.zeros_like(tau_grid)
            dpsi[1:] = 0.5 * (O_vals[:-1] + O_vals[1:]) * torch.diff(tau_grid)
            psi_grid = torch.cumsum(dpsi, dim=0)

        # Interpolate psi(tau)
        psi = torch.nn.functional.interpolate(
            psi_grid.view(1, 1, -1),
            size=tau.numel(),
            mode="linear",
            align_corners=True,
        ).view(-1)
        return psi

    def _compute_source_quantities(
        self,
        tau: torch.Tensor,
        Omega_val: float | torch.Tensor,
        blade_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute position, velocity, force, and force derivative for one blade.

        Args:
            tau: Source times. Shape [N_t], [N_sources, N_t],
                 or [N_sources, N_obs, N_t].
            Omega_val: Angular velocity (rad/s), scalar, [N_t], or broadcastable
                       to tau's shape.
            blade_idx: Blade index (0 to B-1).

        Returns:
            y, v, F, Fdot: shapes follow tau's batch dims, e.g.
                [N_r, N_t, 3], [N_r, N_sources, N_t, 3], etc.
        """
        blade = self.rotor.blade
        phi_b = self.rotor.phi_offset[blade_idx]

        # Evaluate if callable
        if callable(Omega_val):
            Omega_val = Omega_val(tau)
        # Ensure Omega is tensor with shape [N_t]
        if not isinstance(Omega_val, torch.Tensor):
            Omega_val = torch.tensor(Omega_val, dtype=tau.dtype)
        if Omega_val.dim() == 0:
            Omega_val = Omega_val.expand(tau.shape[-1])

        # Azimuth: psi = Omega * tau + phi_b  (for constant Omega)
        psi = Omega_val * tau + phi_b  # [N_t]

        # Positions and velocities
        y = blade.panel_positions(psi)  # [N_r, N_t, 3]
        v = blade.panel_velocities(psi, Omega_val)  # [N_r, N_t, 3]

        # Forces: BEMT in local frame, then rotate to ground frame
        # For compact chord, force on each strip is total BEMT force
        F_local, _, _ = self.aero.compute_forces(blade, Omega_val.mean())
        # F_local: [N_r, 3] (constant in body frame for hover)

        # Rotate to ground frame at each tau
        cos_p = torch.cos(psi)
        sin_p = torch.sin(psi)

        # F_local slices are [N_r, 1]; add trailing singletons to broadcast
        # against cos_p which may have extra batch dims (e.g. [N_r, N_obs, N_t])
        def _expand_local(Fc):
            while Fc.dim() < cos_p.dim():
                Fc = Fc.unsqueeze(-2)
            return Fc

        Flx = _expand_local(F_local[:, 0:1])
        Fly = _expand_local(F_local[:, 1:2])
        Flz = _expand_local(F_local[:, 2:3])

        # R_z(psi) @ F_local
        F_x = cos_p * Flx - sin_p * Fly
        F_y = sin_p * Flx + cos_p * Fly
        F_z = Flz.expand_as(F_x)
        F = torch.stack([F_x, F_y, F_z], dim=-1)  # [N_r, ..., N_t, 3]

        # Force derivative: d/dtau [R_z(psi) @ F_local]
        # For constant body-frame force: Fdot = Omega * dR/dpsi @ F_local
        # dR/dpsi @ F_local = [-sin*Fx0 - cos*Fy0, cos*Fx0 - sin*Fy0, 0]
        Fdot_x = -Omega_val * (sin_p * Flx + cos_p * Fly)
        Fdot_y = Omega_val * (cos_p * Flx - sin_p * Fly)
        Fdot_z = torch.zeros_like(Fdot_x)
        Fdot = torch.stack([Fdot_x, Fdot_y, Fdot_z], dim=-1)

        return y, v, F, Fdot

    def compute_pressure(
        self,
        x_observer: torch.Tensor,
        t: torch.Tensor,
        Omega: Callable[[torch.Tensor], torch.Tensor] | float | torch.Tensor,
    ) -> torch.Tensor:
        """Compute acoustic pressure time history at observer(s).

        Args:
            x_observer: Observer position [3] or positions [N_obs, 3] (m).
            t: Observer times [N_t] (s).
            Omega: Angular velocity. Can be:
                   - scalar (constant rad/s)
                   - torch.Tensor (constant)
                   - Callable(tau) -> Omega(tau) (time-varying)

        Returns:
            p'(t): Acoustic pressure [N_t] or [N_obs, N_t] (Pa).
        """
        if x_observer.dim() == 1:
            p_total = torch.zeros_like(t)
        else:
            p_total = torch.zeros(x_observer.shape[0], t.shape[0], dtype=t.dtype, device=t.device)

        for b in range(self.rotor.B):
            # Source functions for this blade
            # Omega is always float|Tensor when called through these closures
            _O = cast(float | torch.Tensor, Omega)

            def y_func(tau, _O=_O, b=b):
                return self._compute_source_quantities(tau, _O, b)[0]

            def v_func(tau, _O=_O, b=b):
                return self._compute_source_quantities(tau, _O, b)[1]

            def F_func(tau, _O=_O, b=b):
                return self._compute_source_quantities(tau, _O, b)[2]

            def Fdot_func(tau, _O=_O, b=b):
                return self._compute_source_quantities(tau, _O, b)[3]

            # Mach derivative (only needed for term 3)
            def Mdot_func(tau, _O=_O, b=b):
                # M = v/c0, Mdot = vdot/c0
                # vdot = Omega_dot × y + Omega × v
                # For constant Omega: vdot = Omega × v = -Omega^2 * y_perp
                _, v, _, _ = self._compute_source_quantities(tau, _O, b)
                if isinstance(Omega, (int, float, torch.Tensor)) and (
                    not isinstance(Omega, torch.Tensor) or Omega.dim() == 0
                ):
                    Omega_val = float(Omega)
                    # vdot_x = -Omega^2 * y_x, vdot_y = -Omega^2 * y_y
                    y = y_func(tau)
                    vdot = -(Omega_val**2) * y
                    vdot[..., 2] = 0.0
                else:
                    # Variable Omega: approximate numerically
                    dt = 1e-6
                    _, v_plus, _, _ = self._compute_source_quantities(tau + dt, _O, b)
                    _, v_minus, _, _ = self._compute_source_quantities(tau - dt, _O, b)
                    vdot = (v_plus - v_minus) / (2 * dt)
                return vdot / self.c0

            p_blade = self.fwh.compute_pressure(
                t,
                x_observer,
                y_func,
                v_func,
                F_func,
                Fdot_func,
                Mdot_func=Mdot_func,
                include_term3=True,
            )
            p_total += p_blade

        return p_total
