---
name: load-real-propeller-geometry
description: Load real drone/UAV propeller geometry (chord and twist distributions) from open databases (FLOWUnsteady / UIUC / APC .peo) and instantiate a Blade object in the FWH simulator.
---

# Load Real Propeller Geometry

Find and load real propeller geometry data from open databases into the FWH rotor simulator.

## Data sources

1. **FLOWUnsteady database** (github.com/byuflowlab/FLOWUnsteady/tree/master/database/rotors/)
   - CSV files: `{prop}_chorddist.csv`, `{prop}_pitchdist.csv`
   - Also contains: sweepdist, heightdist, airfoil_files
   - Props available: `apc10x7`, `apc11x8`, `apc12x6`, `dji9443`, etc.

2. **APC .peo files** (https://www.apcprop.com/technical-information/file-downloads/)
   - Download `PE0-FILES_WEB-*.zipx`, unzip to get `.peo` text files
   - Each file contains radial stations with chord, twist, airfoil

3. **UIUC Propeller Database** (m-selig.ae.illinois.edu/props/)
   - Wind tunnel data + digitized geometry for ~140 propellers
   - Some have geometry plots and tabulated data

## Workflow

### Option A: FLOWUnsteady CSV (easiest)

```python
import numpy as np
from fwh_rotor_sim.geometry import Blade, Rotor

# Download chord and pitch CSVs from GitHub raw URLs
# Example: APC 10x7
def load_apc_10x7():
    chord_data = """r/R,c/R
0.0,0.134
...
1.0,0.0375"""

    twist_data = """r/R,twist (deg)
0.0,17.0
...
1.0,11.53"""

    # Parse...
    R = 0.127  # m (10-inch diameter)
    Rhub = 0.0095325  # m
    B = 2

    def chord_fn(r):
        import torch
        if isinstance(r, torch.Tensor):
            r_np = r.detach().cpu().numpy()
            c_np = np.interp(r_np, r_tab, c_tab)
            return torch.from_numpy(c_np).to(r.dtype)
        return np.interp(r, r_tab, c_tab)

    def twist_fn(r):
        # same pattern
        ...

    return Blade(radius=R, chord=chord_fn, twist_deg=twist_fn,
                 hub_radius=Rhub, n_radial=60)
```

### Option B: APC .peo file

Format (text file):
```
APC 10x7E
RADIUS = 5.0000
...
  r/R    c/R    beta(deg)  airfoil
  0.15   0.180   35.0      CLARK-Y
  ...
```

Parse with simple line splitting, then interpolate as above.

### Visualization

```python
from fwh_rotor_sim.plotting import plot_blade_geometry, plot_rotor_top_view

blade = load_apc_10x7()
rotor = Rotor(blade=blade, num_blades=2)

fig, _ = plot_blade_geometry(blade)
fig, _ = plot_rotor_top_view(rotor)
```

## Key gotchas

- **Interpolation:** The tabulated data has ~20 radial stations. Use `np.interp` (or `scipy.interpolate.CubicSpline` for smoother curves) to evaluate at the `n_radial` simulation grid points.
- **Twist convention:** The FLOWUnsteady data uses "twist (deg)" — the angle between the blade section chord line and the rotor disk plane. This matches our `theta(r)` convention in `geometry.py`.
- **Hub radius:** The tabulated data often starts at r/R > 0 (e.g., 0.15). The actual hub extends to r = Rhub. Our `Blade` class handles this by starting panels at `r_hub`.
- **Differentiability:** Since we use interpolation inside callable `chord_fn`/`twist_fn`, the resulting `Blade.c` and `Blade.theta_rad` tensors are differentiable w.r.t. any upstream parameters.

## Common propeller specs

| Propeller | Diameter (in) | R (m) | Rhub (m) | Blades | Source |
|-----------|--------------|-------|----------|--------|--------|
| APC 10x7E | 10 | 0.127 | 0.00953 | 2 | FLOWUnsteady |
| APC 11x8E | 11 | 0.1397 | 0.00953 | 2 | FLOWUnsteady |
| DJI 9443 | 9.4 | 0.119 | ~0.008 | 2 | FLOWUnsteady |
| DJI 6030 | 6 | 0.0762 | ~0.005 | 2 | Approximate |

## References

- UIUC Propeller Database: https://m-selig.ae.illinois.edu/props/propDB.html
- APC geometry files: https://www.apcprop.com/technical-information/file-downloads/
- FLOWUnsteady database: https://github.com/byuflowlab/FLOWUnsteady/tree/master/database
