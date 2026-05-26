"""Rotor geometry definitions and panel generation."""

import torch
import torch.nn as nn
from typing import Union, Callable


class Blade:
    """Parametric rotor blade geometry.

    Defines blade shape via radius, chord, and twist distributions.
    Panels are generated as radial strips (compact chord approximation).
    """

    def __init__(
        self,
        radius: float,
        chord: Union[float, Callable[[torch.Tensor], torch.Tensor]],
        twist_deg: Union[float, Callable[[torch.Tensor], torch.Tensor]],
        hub_radius: float = 0.0,
        n_radial: int = 50,
    ):
        self.R = float(radius)
        self.r_hub = float(hub_radius)
        self.n_radial = n_radial

        # Radial stations (equal spacing for now)
        self.r = torch.linspace(hub_radius, radius, n_radial)
        # Panel widths
        dr_full = (radius - hub_radius) / max(n_radial - 1, 1)
        self.dr = torch.full_like(self.r, dr_full)
        # Half-size end panels for trapezoidal rule consistency
        if n_radial > 1:
            self.dr[0] = dr_full / 2
            self.dr[-1] = dr_full / 2

        # Chord distribution
        if callable(chord):
            self.c = chord(self.r)
        else:
            self.c = torch.full_like(self.r, float(chord))

        # Twist distribution (degrees)
        if callable(twist_deg):
            self.theta_deg = twist_deg(self.r)
        else:
            self.theta_deg = torch.full_like(self.r, float(twist_deg))
        self.theta_rad = torch.deg2rad(self.theta_deg)

        # Panel centers in body frame:
        #   x: along blade span (radial, from hub to tip)
        #   y: chordwise (in blade section plane)
        #   z: normal to blade surface (thickness direction)
        # For compact chord, each strip acts as a point source at its center.
        self.y0 = torch.stack([
            self.r,
            torch.zeros_like(self.r),
            torch.zeros_like(self.r),
        ], dim=-1)  # [n_radial, 3]

        # Panel area (one side): chord × span
        self.dS = self.c * self.dr  # [n_radial]

    def panel_positions(self, psi: torch.Tensor) -> torch.Tensor:
        """Rotate body-frame panel centers to ground frame.

        Args:
            psi: Azimuth angle(s) in radians. Scalar, [N_t], [N_r, N_t],
                 [N_obs, N_t], [N_r, N_obs, N_t], etc.

        Returns:
            Ground-frame positions. Shape [n_radial, ..., 3] where ... are the
            leading dimensions of psi.
        """
        cos_p = torch.cos(psi)
        sin_p = torch.sin(psi)

        y0x = self.y0[:, 0]  # [N_r]
        y0y = self.y0[:, 1]
        y0z = self.y0[:, 2]

        if psi.dim() == 0:
            x_g = cos_p * y0x - sin_p * y0y
            y_g = sin_p * y0x + cos_p * y0y
            z_g = y0z
            return torch.stack([x_g, y_g, z_g], dim=-1)

        # Determine how many trailing singletons y0 needs so it broadcasts
        # correctly against psi's leading dimensions.
        # If psi's first dim equals n_radial, that dim is the panel/source
        # dimension (from retarded-time) and y0 should align with it.
        # Otherwise all dims of psi are batch dimensions to broadcast over.
        if psi.dim() == 1:
            n_trailing = 1
        elif psi.shape[0] == self.n_radial:
            n_trailing = psi.dim() - 1
        else:
            n_trailing = psi.dim()

        y0x_b = y0x.view(y0x.shape + (1,) * n_trailing)
        y0y_b = y0y.view(y0y.shape + (1,) * n_trailing)
        y0z_b = y0z.view(y0z.shape + (1,) * n_trailing)

        x_g = cos_p * y0x_b - sin_p * y0y_b
        y_g = sin_p * y0x_b + cos_p * y0y_b
        z_g = y0z_b.expand_as(x_g)
        return torch.stack([x_g, y_g, z_g], dim=-1)

    def panel_velocities(self, psi: torch.Tensor, Omega: torch.Tensor) -> torch.Tensor:
        """Compute panel velocities from rigid-body rotation.

        Args:
            psi: Azimuth angle(s). Same shape convention as panel_positions.
            Omega: Angular velocity (rad/s). Same shape as psi or broadcastable.

        Returns:
            Velocities in ground frame. Shape [..., n_radial, 3].
        """
        y_g = self.panel_positions(psi)
        if psi.dim() == 0:
            vx = -Omega * y_g[..., 1]
            vy = Omega * y_g[..., 0]
            vz = torch.zeros_like(vx)
            return torch.stack([vx, vy, vz], dim=-1)
        # Omega and y_g[..., 0/1] naturally broadcast: e.g. [N_t]*[N_r,N_t] or [N_r,N_t]*[N_r,N_t]
        vx = -Omega * y_g[..., 1]
        vy = Omega * y_g[..., 0]
        vz = torch.zeros_like(vx)
        return torch.stack([vx, vy, vz], dim=-1)

    def __repr__(self):
        return (f"Blade(R={self.R:.3f}m, n_r={self.n_radial}, "
                f"theta_tip={self.theta_deg[-1]:.1f}deg)")


class Rotor:
    """Multi-blade rotor assembly."""

    def __init__(
        self,
        blade: Blade,
        num_blades: int = 2,
        shaft_tilt_deg: float = 0.0,
    ):
        self.blade = blade
        self.B = num_blades
        self.shaft_tilt = torch.deg2rad(torch.tensor(shaft_tilt_deg))

        # Blade azimuthal offsets
        self.phi_offset = torch.linspace(0, 2 * torch.pi, num_blades + 1)[:-1]

    @property
    def disk_area(self) -> float:
        return torch.pi * self.blade.R ** 2

    def __repr__(self):
        return (f"Rotor(B={self.B}, blade={self.blade}, "
                f"tilt={torch.rad2deg(self.shaft_tilt):.1f}deg)")
