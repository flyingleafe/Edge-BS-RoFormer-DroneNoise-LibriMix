"""Blade Element Momentum Theory aerodynamics."""

import torch
from typing import Callable


class ThinAirfoilPolar:
    """Simple airfoil polar using thin-airfoil + parabolic drag."""

    def __init__(self, a0: float = 2 * torch.pi, alpha0_deg: float = 0.0,
                 cd0: float = 0.01, cd_induced_factor: float = 0.0):
        """
        Args:
            a0: Lift curve slope (dCl/dα). Default 2π for thin airfoil.
            alpha0_deg: Zero-lift angle (degrees).
            cd0: Minimum drag coefficient.
            cd_induced_factor: Induced drag factor (Cd_induced = factor * Cl^2).
        """
        self.a0 = a0
        self.alpha0 = torch.deg2rad(torch.tensor(alpha0_deg))
        self.cd0 = cd0
        self.k = cd_induced_factor

    def __call__(self, alpha: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return Cl, Cd for given angle of attack (rad)."""
        cl = self.a0 * (alpha - self.alpha0)
        cd = self.cd0 + self.k * cl ** 2
        return cl, cd


class BEMTAerodynamics:
    """Blade Element Momentum Theory for hover / axial flight.

    Computes sectional lift/drag forces on each blade strip.
    Uses uniform inflow (momentum theory) with optional iteration.
    """

    def __init__(
        self,
        rho: float = 1.225,
        airfoil: Callable = None,
        max_iter: int = 20,
        tol: float = 1e-6,
    ):
        self.rho = rho
        self.airfoil = airfoil or ThinAirfoilPolar()
        self.max_iter = max_iter
        self.tol = tol

    def compute_forces(
        self,
        blade,
        Omega: torch.Tensor,
        v_induced: torch.Tensor = None,
        advance_ratio: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute force on each blade strip.

        Args:
            blade: Blade geometry.
            Omega: Angular velocity (rad/s). Scalar or tensor.
            v_induced: Induced inflow velocity (m/s, positive downward).
                       If None, computed from momentum theory.
            advance_ratio: mu = V_inf / (Omega * R). For hover, 0.

        Returns:
            F_local: [n_radial, 3] force per strip in local blade frame.
            dT: [n_radial] thrust contribution per strip (N).
            dQ: [n_radial] torque contribution per strip (N·m).
        """
        r = blade.r
        c = blade.c
        theta = blade.theta_rad
        dr = blade.dr

        # Ensure Omega is a tensor
        if not isinstance(Omega, torch.Tensor):
            Omega = torch.tensor(Omega, dtype=r.dtype)

        # Local velocity components
        U_T = Omega * r  # tangential
        U_P = torch.zeros_like(r) if v_induced is None else v_induced
        if advance_ratio != 0:
            # Simple forward flight correction: add uniform inflow component
            U_P = U_P + Omega * blade.R * advance_ratio

        # Resultant velocity and inflow angle
        U = torch.sqrt(U_T ** 2 + U_P ** 2)
        phi = torch.atan2(U_P, U_T)  # angle between disk plane and U

        # Angle of attack
        alpha = theta - phi

        # Clamp alpha to avoid stall (thin airfoil valid ~ ±15°)
        alpha = torch.clamp(alpha, torch.deg2rad(torch.tensor(-20.0)),
                                     torch.deg2rad(torch.tensor(20.0)))

        # Airfoil coefficients
        cl, cd = self.airfool(alpha) if hasattr(self, 'airfool') else self.airfoil(alpha)

        # Sectional lift and drag per unit span (N/m)
        q = 0.5 * self.rho * U ** 2
        L_prime = q * c * cl  # lift per span
        D_prime = q * c * cd  # drag per span

        # Thrust and torque per strip (N and N·m)
        # dT = (L' cos(phi) + D' sin(phi)) * dr
        # dQ = (L' sin(phi) - D' cos(phi)) * r * dr  (opposing rotation)
        dT = (L_prime * torch.cos(phi) + D_prime * torch.sin(phi)) * dr
        dQ = (L_prime * torch.sin(phi) - D_prime * torch.cos(phi)) * r * dr

        # Force vector in local blade frame
        # x_local: radial outward
        # y_local: tangential in direction of rotation
        # z_local: upward normal
        #
        # U vector: [0, U_T, -U_P]  (mostly tangential, some downward)
        # Lift: perpendicular to U, toward +z for positive alpha
        #   lift_dir = [0, sin(phi), cos(phi)]
        # Drag: opposite to U
        #   drag_dir = [0, -cos(phi), sin(phi)]
        dF_y = L_prime * torch.sin(phi) - D_prime * torch.cos(phi)
        dF_z = L_prime * torch.cos(phi) + D_prime * torch.sin(phi)

        F_local = torch.stack([
            torch.zeros_like(dF_y),
            dF_y,
            dF_z,
        ], dim=-1) * dr[:, None]  # multiply by span to get total force on strip

        return F_local, dT, dQ

    def induced_velocity_momentum(
        self,
        blade,
        Omega: torch.Tensor,
        total_thrust: torch.Tensor,
    ) -> torch.Tensor:
        """Compute uniform induced velocity from actuator disk momentum theory.

        Args:
            blade: Blade geometry.
            Omega: Angular velocity (rad/s).
            total_thrust: Total thrust from all blades (N).

        Returns:
            Induced velocity (m/s, positive downward).
        """
        A = torch.pi * blade.R ** 2
        # T = 2 * rho * A * v_i^2  (momentum theory for hover)
        v_i = torch.sqrt(total_thrust / (2 * self.rho * A))
        return v_i

    def solve_hover(
        self,
        blade,
        Omega: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Iterative BEMT solution for hover.

        Returns:
            F_local, dT, dQ, v_induced
        """
        # Initial guess: zero induced velocity
        v_i = torch.tensor(0.0)
        B = blade.n_radial

        for _ in range(self.max_iter):
            F_local, dT, dQ = self.compute_forces(blade, Omega, v_i)
            T_total = dT.sum()
            v_i_new = self.induced_velocity_momentum(blade, Omega, T_total)
            if torch.abs(v_i_new - v_i) < self.tol:
                break
            v_i = v_i_new

        # Recompute with converged inflow
        F_local, dT, dQ = self.compute_forces(blade, Omega, v_i)
        return F_local, dT, dQ, v_i
