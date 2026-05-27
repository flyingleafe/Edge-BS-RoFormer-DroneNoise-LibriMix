"""Vortex Lattice Method (VLM) for rotor blade aerodynamics and acoustics.

Implements:
- Horseshoe vortices (bound vortex + two trailing legs)
- Biot-Savart induction (analytic, GPU-vectorized)
- Wake vortex particles (tip vortex rollup)
- Differentiable: chord(r), twist(r), RPM → Γ(r) → acoustic pressure

Key physics:
- Kutta-Joukowski: L' = ρ * U × Γ (per unit span)  → Γ = L' / (ρ * U)
- Force decomposition: F_norm (thrust/lift), F_tang (drag), F_radial (zero for hover)
- FWH: pressure from time-varying force on rotating bound vortex system
"""

import torch
from typing import Optional, Tuple
from .geometry import Blade


class VortexParticle:
    """A vortex particle with position and scalar circulation.

    Biot-Savart velocity: v = Γ/(4π) * cross(r_vec, dr) / (|r|² + ε²)^(3/2)
    Fully differentiable, GPU-vectorized.
    """

    def __init__(self, pos: torch.Tensor, Gamma: torch.Tensor, core_radius: float = 1e-4):
        self.pos = pos                      # [N, 3]
        self.Gamma = Gamma                   # [N]
        self.core_radius = core_radius      # [scalar]

    def velocity_at(self, x: torch.Tensor) -> torch.Tensor:
        """Biot-Savart velocity at points x. Vectorized, differentiable."""
        if x.dim() == 1:
            x = x.unsqueeze(0)
        dr = x.unsqueeze(1) - self.pos.unsqueeze(0)          # [M, N, 3]
        r_sq = torch.sum(dr ** 2, dim=-1)                     # [M, N]
        r_sq_core = r_sq + self.core_radius ** 2
        r_cube = r_sq_core * torch.sqrt(r_sq_core)
        v_mag = self.Gamma.unsqueeze(0) / (4 * torch.pi * r_cube + 1e-20)  # [M, N]
        return (v_mag.unsqueeze(2) * dr).sum(dim=1)          # [M, 3]

    def velocity_self(self) -> torch.Tensor:
        """Self-induction for wake dynamics. O(N²). Use sparingly."""
        N = self.pos.shape[0]
        dr = self.pos.unsqueeze(1) - self.pos.unsqueeze(0)
        mask = ~torch.eye(N, dtype=torch.bool, device=dr.device)
        dr = dr * mask.unsqueeze(2).float()
        r_sq = torch.sum(dr ** 2, dim=-1)
        r_sq_core = r_sq + self.core_radius ** 2
        r_cube = r_sq_core * torch.sqrt(r_sq_core)
        v_mag = self.Gamma.unsqueeze(1) / (4 * torch.pi * r_cube + 1e-20)
        return (v_mag.unsqueeze(2) * dr).sum(dim=1)

    def advect(self, v_freestream: torch.Tensor, dt: float) -> 'VortexParticle':
        """Euler advection step."""
        v_total = self.velocity_self() + v_freestream
        return VortexParticle(self.pos + v_total * dt, self.Gamma, self.core_radius)


class VortexLatticeSolver:
    """Vortex Lattice Method for rotor blade loading and acoustic prediction.

    Physics:
    - Bound vortex: horseshoe vortex at each radial station
    - Kutta condition: Γ(r) = L' / (ρ * U)  →  Γ = 0.5 * c * U * Cl
    - Force: F_norm = L' * dr  (normal/thrust),  F_tang = D' * dr  (drag)
    - Acoustic: loading noise via FWH from rotating bound vortex system

    Differentiability:
    - chord(r) → c(r) → L' → Γ → F_norm → pressure: full autograd chain
    - twist_deg(r) → θ(r) → α → Cl → Γ → pressure: full autograd chain
    """

    def __init__(
        self,
        blade: Blade,
        num_blades: int,
        c0: float = 343.0,
        rho0: float = 1.225,
    ):
        self.blade = blade
        self.B = num_blades
        self.c0 = c0
        self.rho0 = rho0
        self.n_r = blade.n_radial
        self.r = blade.r      # [n_r] radial stations
        self.dr = blade.dr    # [n_r] panel widths
        self.c = blade.c      # [n_r] chord at each station
        self.theta = blade.theta_rad  # [n_r] twist (geometric pitch)

    def compute_circulation(
        self,
        Omega: torch.Tensor,
        v_induced: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute bound circulation and force per radial strip.

        Kutta-Joukowski: L' = ρ * U × Γ (per unit span)
        => Γ(r) = L' / (ρ * U) = 0.5 * c * U * Cl
        => Γ_strip(r) = Γ(r) * dr

        Force in blade-local frame (x=radial, y=tangential, z=normal):
        - F_norm = L' * dr  (lift/thrust component)   [n_r] N
        - F_tang = D' * dr  (drag component)         [n_r] N
        - F_radial ≈ 0     (zero for hover)

        Args:
            Omega: Angular velocity (rad/s)
            v_induced: [n_r] optional induced inflow (m/s), positive = downwash

        Returns:
            Gamma:  [n_r] circulation per strip (m²/s)
            F_norm: [n_r] normal force per strip (N)  — dominant
            F_tang: [n_r] tangential force per strip (N) — small for hover
        """
        r = self.r
        c = self.c
        theta = self.theta
        dr = self.dr

        U_T = Omega * r
        U_P = torch.zeros_like(r) if v_induced is None else v_induced
        U = torch.sqrt(U_T ** 2 + U_P ** 2)

        phi = torch.atan2(U_P, U_T)
        alpha = theta - phi
        alpha = torch.clamp(alpha, -0.35, 0.35)

        cl = 2 * torch.pi * alpha
        cd = 0.01  # Cd0 ~ 0.01 (parasitic drag)

        # Lift and drag per unit span [N/m]
        L_prime = 0.5 * self.rho0 * U ** 2 * c * cl
        D_prime = 0.5 * self.rho0 * U ** 2 * c * cd

        # Circulation per unit span: Γ(r) = L' / (ρ * U) = 0.5 * c * U * cl
        Gamma = 0.5 * c * U * cl.abs() * dr   # [n_r] m²/s

        # Force per strip [n_r] N
        F_norm = L_prime.abs() * dr            # normal (lift/thrust) component
        F_tang = D_prime * dr                  # tangential (drag) component

        return torch.clamp(Gamma, min=1e-10), F_norm, F_tang

    def compute_pressure(
        self,
        x_obs: torch.Tensor,
        t: torch.Tensor,
        Omega: torch.Tensor,
        Gamma: torch.Tensor,
        F_norm: torch.Tensor,
        F_tang: torch.Tensor,
        wake: Optional[VortexParticle] = None,
    ) -> torch.Tensor:
        """Compute acoustic pressure via FWH loading terms.

        Force model in blade-local frame (x_radial, y_tang, z_normal):
            F_local = [0, F_tang, F_norm]

        Rotated to ground frame at azimuth ψ:
            F_ground = R_z(ψ) @ F_local
            Fx = -F_tang * sin(ψ)
            Fy =  F_tang * cos(ψ)
            Fz =  F_norm

        Farassat 1A: p' ∝ 1/c₀ * d/dt [F_r / (r(1-M_r)²)] + ...

        Args:
            x_obs:     [3] or [N_obs, 3] observer position
            t:         [N_t] observer times
            Omega:     Angular velocity (rad/s)
            Gamma:     [n_r] bound circulation (m²/s)
            F_norm:    [n_r] normal force per strip (N)
            F_tang:    [n_r] tangential force per strip (N)
            wake:      Optional wake VortexParticle
        Returns:
            p: [N_t] or [N_obs, N_t] acoustic pressure (Pa)
        """
        from .fwh import Farassat1ASolver
        fwh = Farassat1ASolver(c0=self.c0, rho0=self.rho0)
        n_r = self.n_r
        r = self.r

        p_total = torch.zeros_like(t)

        for b in range(self.B):
            phi_b = 2 * torch.pi * b / self.B

            for i in range(n_r):
                ri = r[i]
                Fni = F_norm[i]
                Fti = F_tang[i]

                def y_func(tau, ri=ri, phi_b=phi_b):
                    psi = Omega * tau + phi_b
                    # y = (r*cos ψ, r*sin ψ, 0) — same rotation axis as BEMT (z)
                    return torch.stack([
                        ri * torch.cos(psi),
                        ri * torch.sin(psi),
                        torch.zeros_like(psi),
                    ], dim=-1)

                def v_func(tau, ri=ri, phi_b=phi_b):
                    psi = Omega * tau + phi_b
                    return torch.stack([
                        -Omega * ri * torch.sin(psi),
                         Omega * ri * torch.cos(psi),
                         torch.zeros_like(psi),
                    ], dim=-1)

                def F_func(tau, Fni=Fni, Fti=Fti, phi_b=phi_b):
                    psi = Omega * tau + phi_b
                    # F_local = [0, -F_tang, F_norm] (drag opposes rotation)
                    # R_z(ψ) @ F_local = [F_tang*sin(ψ), -F_tang*cos(ψ), F_norm]
                    return torch.stack([
                        Fti * torch.sin(psi),
                       -Fti * torch.cos(psi),
                        Fni * torch.ones_like(psi),
                    ], dim=-1)

                def Fdot_func(tau, Fni=Fni, Fti=Fti, phi_b=phi_b):
                    psi = Omega * tau + phi_b
                    # dF/dpsi = [F_tang*cos(ψ), F_tang*sin(ψ), 0]
                    return torch.stack([
                        Fti * Omega * torch.cos(psi),
                        Fti * Omega * torch.sin(psi),
                        torch.zeros_like(psi),
                    ], dim=-1)

                def Mdot_func(tau):
                    return torch.zeros_like(y_func(tau))

                p_i = fwh.compute_pressure(
                    t, x_obs, y_func, v_func, F_func, Fdot_func,
                    Mdot_func=Mdot_func, include_term3=False,
                )
                p_total += p_i

        return p_total

    def build_wake_particles(
        self,
        Gamma: torch.Tensor,
        Omega: torch.Tensor,
        n_steps: int = 200,
        dt: float = 5e-4,
        max_per_ring: int = 16,
    ) -> VortexParticle:
        """Build tip-vortex wake as discrete vortex particles.

        Particles are shed from each blade at each time step and convected
        downstream. For hover, the wake descends at approximately the
        induced velocity.

        Args:
            Gamma: [n_r] bound circulation
            Omega: Angular velocity (rad/s)
            n_steps: Number of wake time steps
            dt: Time step per shed (s)
            max_per_ring: Particles per vortex ring
        Returns:
            VortexParticle containing all wake vortices
        """
        device = self.r.device
        n_r = self.n_r
        R = self.blade.R

        max_n = n_steps * n_r * self.B * max_per_ring
        positions = torch.zeros(max_n, 3, device=device)
        gammas = torch.zeros(max_n, device=device)

        v_downstream = 0.5 * Omega * R  # approximate downstream speed
        pidx = 0

        for step in range(n_steps):
            t_shed = step * dt
            z_shed = -v_downstream * t_shed

            for b in range(self.B):
                phi_b = 2 * torch.pi * b / self.B + Omega * t_shed

                for i in range(n_r):
                    ri = self.r[i]
                    Gamma_i = Gamma[i].abs()
                    n_pts = max(4, int(max_per_ring * ri / R))

                    theta_pts = torch.linspace(0, 2 * torch.pi, n_pts + 1, device=device)[:-1]
                    y_ring = ri * torch.cos(theta_pts)
                    z_ring = ri * torch.sin(theta_pts) + z_shed

                    for j in range(n_pts):
                        if pidx < max_n:
                            positions[pidx] = torch.stack([
                                torch.zeros_like(theta_pts[j]),
                                y_ring[j],
                                z_ring[j],
                            ])
                            gammas[pidx] = Gamma_i / n_pts
                            pidx += 1

        return VortexParticle(positions[:pidx].contiguous(),
                              gammas[:pidx].contiguous(),
                              core_radius=2e-4)

    def velocity_at(self, x: torch.Tensor, Gamma: torch.Tensor,
                    wake: Optional[VortexParticle] = None) -> torch.Tensor:
        """Compute induced velocity from bound vortices and wake."""
        if x.dim() == 1:
            x = x.unsqueeze(0)
        v_total = torch.zeros_like(x)

        n_r = self.n_r
        for b in range(self.B):
            phi_b = 2 * torch.pi * b / self.B
            for i in range(n_r):
                ri = self.r[i]
                n_seg = max(8, int(12 * ri / self.blade.R + 1))
                theta_ring = torch.linspace(0, 2 * torch.pi, n_seg, device=x.device)
                y_ring = ri * torch.cos(theta_ring)
                z_ring = ri * torch.sin(theta_ring)
                ring_pos = torch.stack([torch.zeros_like(theta_ring), y_ring, z_ring], dim=1)
                ring_gamma = Gamma[i].abs() / n_seg

                dr = x.unsqueeze(1) - ring_pos.unsqueeze(0)
                for j in range(n_seg):
                    v_start = ring_pos[j]
                    v_end = ring_pos[(j + 1) % n_seg]
                    dl = v_end - v_start
                    r1 = x - v_start
                    r2 = x - v_end
                    cross = torch.cross(r1, r2, dim=-1)
                    r1n = torch.norm(r1, dim=-1, keepdim=True) + 1e-10
                    r2n = torch.norm(r2, dim=-1, keepdim=True) + 1e-10
                    r1d = torch.sum(r1 * dl, dim=-1, keepdim=True)
                    r2d = torch.sum(r2 * dl, dim=-1, keepdim=True)
                    denom = torch.sum(cross ** 2, dim=-1, keepdim=True) + 1e-20
                    factor = (r1d / r1n - r2d / r2n) / denom
                    v_total += ring_gamma / (4 * torch.pi) * factor * cross

        if wake is not None:
            v_total += wake.velocity_at(x)
        return v_total


class VLMFWHIntegrator:
    """High-level wrapper: blade params → VLM → FWH → pressure.

    Chain: chord_scale, twist_scale → blade.c(r), blade.theta(r)
         → L'(r), D'(r) → Γ(r), F_norm(r), F_tang(r)
         → FWH pressure

    Example:
        integrator = VLMFWHIntegrator(blade, num_blades=2)
        p = integrator(x_obs, t, Omega)  # fully differentiable
    """

    def __init__(self, blade: Blade, num_blades: int,
                 c0: float = 343.0, rho0: float = 1.225):
        self.blade = blade
        self.num_blades = num_blades
        self.c0 = c0
        self.rho0 = rho0
        self.solver = VortexLatticeSolver(blade, num_blades, c0=c0, rho0=rho0)

    def __call__(
        self,
        x_obs: torch.Tensor,
        t: torch.Tensor,
        Omega: torch.Tensor,
        return_wake: bool = False,
    ) -> torch.Tensor:
        """Compute acoustic pressure. Fully differentiable to blade geometry."""
        Gamma, F_norm, F_tang = self.solver.compute_circulation(Omega)
        wake = None
        p = self.solver.compute_pressure(x_obs, t, Omega, Gamma, F_norm, F_tang, wake=wake)
        return p

    def gradient_to_geometry(self, x_obs: torch.Tensor, t: torch.Tensor,
                              Omega: torch.Tensor, target_spl: float = 85.0) -> dict:
        """Compute d(SPL)/d(blade geometry). For inverse design."""
        p = self(x_obs, t, Omega)
        prms = torch.sqrt(torch.mean(p ** 2))
        spl = 20 * torch.log10(prms / 20e-6)
        loss = (spl - target_spl) ** 2

        grads = {}
        for name, param in [('c', self.blade.c), ('theta', self.blade.theta_deg)]:
            if param.requires_grad and param.grad is not None:
                grads[name] = param.grad.clone()

        return {'spl': spl.item(), 'grads': grads}