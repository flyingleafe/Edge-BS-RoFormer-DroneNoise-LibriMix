"""Plotting utilities for rotor geometry and acoustic results."""

import matplotlib.pyplot as plt
import numpy as np

from .geometry import Blade, Rotor


def plot_blade_geometry(
    blade: Blade,
    figsize: tuple = (14, 4.5),
    fill_color: str = "C0",
    alpha: float = 0.3,
) -> tuple:
    """Plot blade envelope in three projections / panels.

    Panels:
      1. Planform       — projection onto the rotor disk plane (x-y, top view).
      2. Twist profile  — side elevation showing chord tilt (x-z projection).
      3. Distributions  — chord c(r) and twist θ(r) vs radius.

    Body-frame conventions (see geometry.py):
      x : radial outward (span)
      y : tangential (direction of rotation)
      z : normal to disk plane (thrust direction)

    Args:
        blade: Blade geometry object.
        figsize: Figure size.
        fill_color: Color used for blade-fill and distribution lines.
        alpha: Alpha for filled blade regions.

    Returns:
        fig, axes: Matplotlib figure and axes array.
    """
    r = blade.r.detach().cpu().numpy()
    c = blade.c.detach().cpu().numpy()
    theta = blade.theta_rad.detach().cpu().numpy()

    # Half-chord projections
    c_half = c / 2.0
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    # Leading / trailing edges in body frame
    le_x = r - c_half * sin_t
    te_x = r + c_half * sin_t
    le_y = -c_half * cos_t
    te_y = +c_half * cos_t
    le_z = -c_half * sin_t
    te_z = +c_half * sin_t

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # ------------------------------------------------------------------
    # Panel 1: Planform (x-y projection, top view)
    # ------------------------------------------------------------------
    ax = axes[0]
    ax.fill(
        np.concatenate([le_x, te_x[::-1]]),
        np.concatenate([le_y, te_y[::-1]]),
        color=fill_color,
        alpha=alpha,
        label="Blade planform",
    )
    ax.plot(le_x, le_y, "k-", lw=0.8, label="Leading edge")
    ax.plot(te_x, te_y, "k-", lw=0.8, label="Trailing edge")
    ax.plot(r, np.zeros_like(r), "k--", lw=0.5, alpha=0.5, label="Quarter-chord")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Radial x (m)")
    ax.set_ylabel("Tangential y (m)")
    ax.set_title("Planform (disk-plane view)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=7)

    # ------------------------------------------------------------------
    # Panel 2: Twist profile (x-z projection, side elevation)
    # ------------------------------------------------------------------
    ax = axes[1]
    ax.fill(
        np.concatenate([le_x, te_x[::-1]]),
        np.concatenate([le_z, te_z[::-1]]),
        color=fill_color,
        alpha=alpha,
    )
    ax.plot(le_x, le_z, "k-", lw=0.8, label="Leading edge")
    ax.plot(te_x, te_z, "k-", lw=0.8, label="Trailing edge")
    ax.plot(r, np.zeros_like(r), "k--", lw=0.5, alpha=0.5, label="Zero-lift plane")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Radial x (m)")
    ax.set_ylabel("Normal z (m)")
    ax.set_title("Twist profile (side elevation)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=7)

    # ------------------------------------------------------------------
    # Panel 3: Parameter distributions
    # ------------------------------------------------------------------
    ax_c = axes[2]
    ax_t = ax_c.twinx()

    (l_c,) = ax_c.plot(r * 1000, c * 1000, color=fill_color, lw=2, label="Chord (mm)")
    (l_t,) = ax_t.plot(r * 1000, np.rad2deg(theta), "C1", lw=2, label="Twist (deg)")

    ax_c.set_xlabel("Radius (mm)")
    ax_c.set_ylabel("Chord (mm)", color=fill_color)
    ax_t.set_ylabel("Twist (deg)", color="C1")
    ax_c.set_title("Blade parameter distributions")
    ax_c.grid(True, alpha=0.3)
    ax_c.tick_params(axis="y", labelcolor=fill_color)
    ax_t.tick_params(axis="y", labelcolor="C1")

    # Combined legend
    lines = [l_c, l_t]
    labels = [line.get_label() for line in lines]
    ax_c.legend(lines, labels, loc="upper right")

    plt.tight_layout()
    return fig, axes


def plot_rotor_top_view(
    rotor: Rotor,
    figsize: tuple = (6, 6),
    fill_color: str = "C0",
    alpha: float = 0.3,
) -> tuple:
    """Plot a top-down view of the full multi-blade rotor assembly.

    Args:
        rotor: Rotor geometry object.
        figsize: Figure size.
        fill_color: Color for blade fills.
        alpha: Alpha for filled regions.

    Returns:
        fig, ax: Matplotlib figure and axes.
    """
    blade = rotor.blade
    r = blade.r.detach().cpu().numpy()
    c = blade.c.detach().cpu().numpy()
    theta = blade.theta_rad.detach().cpu().numpy()

    c_half = c / 2.0
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    # LE / TE in body frame (x-y plane)
    le_x_body = r - c_half * sin_t
    te_x_body = r + c_half * sin_t
    le_y_body = -c_half * cos_t
    te_y_body = +c_half * cos_t

    fig, ax = plt.subplots(figsize=figsize)

    for b in range(rotor.B):
        phi = rotor.phi_offset[b].item()
        cos_p, sin_p = np.cos(phi), np.sin(phi)

        # Rotate each point around z-axis by blade offset phi
        def _rot(x, y, cos_p=cos_p, sin_p=sin_p):
            return cos_p * x - sin_p * y, sin_p * x + cos_p * y

        le_x, le_y = _rot(le_x_body, le_y_body)
        te_x, te_y = _rot(te_x_body, te_y_body)

        ax.fill(
            np.concatenate([le_x, te_x[::-1]]),
            np.concatenate([le_y, te_y[::-1]]),
            color=fill_color,
            alpha=alpha,
        )
        ax.plot(le_x, le_y, "k-", lw=0.6)
        ax.plot(te_x, te_y, "k-", lw=0.6)

    # Hub circle
    hub_r = float(blade.r_hub)
    if hub_r > 0:
        hub = plt.Circle((0, 0), hub_r, color="gray", alpha=0.5)  # pyright: ignore[reportPrivateImportUsage]
        ax.add_patch(hub)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"Rotor top view — {rotor.B} blades")
    ax.grid(True, alpha=0.3)

    # Symmetric limits
    lim = float(blade.R) * 1.1
    ax.set_xlim((-lim, lim))
    ax.set_ylim((-lim, lim))

    plt.tight_layout()
    return fig, ax
