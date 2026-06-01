"""Plot real drone blade geometry from open databases.

This example uses the APC 10x7 Thin Electric propeller data from the
FLOWUnsteady / UIUC propeller database.

Data sources:
- FLOWUnsteady rotor database (github.com/byuflowlab/FLOWUnsteady)
- UIUC Propeller Database (m-selig.ae.illinois.edu/props/)
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fwh_rotor_sim.geometry import Blade, Rotor
from fwh_rotor_sim.plotting import plot_blade_geometry, plot_rotor_top_view


def load_apc_10x7():
    """Load APC 10x7 geometry from FLOWUnsteady CSV data.

    Returns (radius, hub_radius, num_blades, r_norm, c_norm, twist_deg).
    """
    # Chord distribution: c/R vs r/R
    chord_data = """r/R,c/R
0.0,0.134
0.086,0.137106
0.16,0.144606
0.2,0.154291
0.25,0.175
0.3,0.20500000000000002
0.35,0.20800000000000002
0.4,0.21200000000000002
0.45,0.20500000000000002
0.5,0.195
0.55,0.186
0.6,0.174
0.65,0.161
0.7,0.145
0.75,0.129
0.8,0.112
0.85,0.096
0.9,0.081
0.9245,0.071125
0.954,0.066125
1.0,0.0375
"""
    # Twist distribution: twist (deg) vs r/R
    twist_data = """r/R,twist (deg)
0.0,17.0
0.04715,22.0
0.088145,30.0
0.15,37.86
0.2,45.82
0.25,44.19
0.3,38.35
0.35,33.64
0.4,29.9
0.45,27.02
0.5,24.67
0.55,22.62
0.6,20.88
0.65,19.36
0.7,17.98
0.75,16.74
0.8,15.79
0.85,14.64
0.9,13.86
0.95,12.72
1.0,11.53
"""

    r_c, c_r = [], []
    for line in chord_data.strip().split("\n")[1:]:
        parts = line.split(",")
        r_c.append(float(parts[0]))
        c_r.append(float(parts[1]))

    r_t, twist = [], []
    for line in twist_data.strip().split("\n")[1:]:
        parts = line.split(",")
        r_t.append(float(parts[0]))
        twist.append(float(parts[1]))

    # APC 10x7: 10-inch diameter = 0.254 m, so R = 0.127 m
    R = 0.127  # m
    Rhub = 0.0095325  # m
    B = 2

    r_c = np.array(r_c)
    c_r = np.array(c_r)
    r_t = np.array(r_t)
    twist = np.array(twist)

    # Convert to absolute units
    r_abs_c = r_c * R
    c_abs = c_r * R
    r_abs_t = r_t * R

    return R, Rhub, B, r_abs_c, c_abs, r_abs_t, twist


def make_blade_from_data(r_chord, chord, r_twist, twist_deg, radius, hub_radius):
    """Create a Blade from tabulated data using interpolation.

    We create callable functions that interpolate the tabulated data,
    then pass them to Blade().
    """
    def chord_fn(r):
        # r is a torch tensor; interpolate using numpy then convert back
        import torch
        if isinstance(r, torch.Tensor):
            r_np = r.detach().cpu().numpy()
            c_np = np.interp(r_np, r_chord, chord)
            return torch.from_numpy(c_np).to(r.dtype)
        return np.interp(r, r_chord, chord)

    def twist_fn(r):
        import torch
        if isinstance(r, torch.Tensor):
            r_np = r.detach().cpu().numpy()
            t_np = np.interp(r_np, r_twist, twist_deg)
            return torch.from_numpy(t_np).to(r.dtype)
        return np.interp(r, r_twist, twist_deg)

    return Blade(
        radius=radius,
        chord=chord_fn,
        twist_deg=twist_fn,
        hub_radius=hub_radius,
        n_radial=60,
    )


def plot_comparison_with_uiuc():
    """Create side-by-side comparison with UIUC reference data."""
    R, Rhub, B, r_c, c, r_t, twist = load_apc_10x7()

    blade = make_blade_from_data(r_c, c, r_t, twist, R, Rhub)
    rotor = Rotor(blade=blade, num_blades=B)

    # Our visualization
    fig_geom, axes_geom = plot_blade_geometry(blade, figsize=(14, 4.5))

    # Add a dedicated comparison figure for distributions
    fig_comp, axes_comp = plt.subplots(1, 3, figsize=(15, 4.5))

    # Panel 1: Our planform
    ax = axes_comp[0]
    r = blade.r.detach().cpu().numpy()
    c_arr = blade.c.detach().cpu().numpy()
    theta = blade.theta_rad.detach().cpu().numpy()
    c_half = c_arr / 2.0
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    le_x = r - c_half * sin_t
    te_x = r + c_half * sin_t
    le_y = -c_half * cos_t
    te_y = +c_half * cos_t
    ax.fill(
        np.concatenate([le_x, te_x[::-1]]),
        np.concatenate([le_y, te_y[::-1]]),
        color="C0", alpha=0.3,
    )
    ax.plot(le_x, le_y, "k-", lw=0.8)
    ax.plot(te_x, te_y, "k-", lw=0.8)
    ax.plot(r, np.zeros_like(r), "k--", lw=0.5, alpha=0.5)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Radial x (m)")
    ax.set_ylabel("Tangential y (m)")
    ax.set_title("Our planform — APC 10x7")
    ax.grid(True, alpha=0.3)

    # Panel 2: Our distributions
    ax_c = axes_comp[1]
    ax_t = ax_c.twinx()
    (l_c,) = ax_c.plot(r * 1000, c_arr * 1000, color="C0", lw=2, label="Chord (mm)")
    (l_t,) = ax_t.plot(r * 1000, np.rad2deg(theta), "C1", lw=2, label="Twist (deg)")
    ax_c.set_xlabel("Radius (mm)")
    ax_c.set_ylabel("Chord (mm)", color="C0")
    ax_t.set_ylabel("Twist (deg)", color="C1")
    ax_c.set_title("Our distributions — APC 10x7")
    ax_c.grid(True, alpha=0.3)
    ax_c.tick_params(axis="y", labelcolor="C0")
    ax_t.tick_params(axis="y", labelcolor="C1")
    lines = [l_c, l_t]
    labels = [l.get_label() for l in lines]
    ax_c.legend(lines, labels, loc="upper right")

    # Panel 3: Tabulated raw data (like UIUC plot)
    ax_c2 = axes_comp[2]
    ax_t2 = ax_c2.twinx()
    (l_c2,) = ax_c2.plot(r_c / R, c / R, "ko-", lw=1, markersize=3, label="Chord c/R")
    (l_t2,) = ax_t2.plot(r_t / R, twist, "ks--", lw=1, markersize=3, label="Twist β (deg)")
    ax_c2.set_xlabel("r / R")
    ax_c2.set_ylabel("c / R", color="C0")
    ax_t2.set_ylabel("Twist β (deg)", color="C1")
    ax_c2.set_title("Raw tabulated data (UIUC style)")
    ax_c2.grid(True, alpha=0.3)
    ax_c2.tick_params(axis="y", labelcolor="C0")
    ax_t2.tick_params(axis="y", labelcolor="C1")
    lines2 = [l_c2, l_t2]
    labels2 = [l.get_label() for l in lines2]
    ax_c2.legend(lines2, labels2, loc="upper right")

    plt.tight_layout()

    # Rotor top view
    fig_top, _ = plot_rotor_top_view(rotor, figsize=(6, 6))

    return fig_geom, fig_comp, fig_top


if __name__ == "__main__":
    print("Loading APC 10x7 geometry from FLOWUnsteady / UIUC database...")
    fig_geom, fig_comp, fig_top = plot_comparison_with_uiuc()

    out_dir = Path("/tmp/apc_10x7_comparison")
    out_dir.mkdir(exist_ok=True)

    fig_geom.savefig(out_dir / "our_blade_geometry.png", dpi=150)
    fig_comp.savefig(out_dir / "comparison.png", dpi=150)
    fig_top.savefig(out_dir / "our_rotor_top_view.png", dpi=150)

    print(f"Saved comparison plots to {out_dir}")
    print("  - our_blade_geometry.png")
    print("  - comparison.png")
    print("  - our_rotor_top_view.png")

    plt.show()
