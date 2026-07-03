# fwh_rotor_sim/ — FWH Rotor Acoustic Simulator

PyTorch-based time-domain Ffowcs-Williams Hawkings (FWH) acoustic simulator for rotating sources. Combines BEMT aerodynamics with Farassat 1A formulation.

## Key modules

| File | Purpose |
|------|---------|
| `geometry.py` | `Blade`, `Rotor` classes. Parametric chord `c(r)` and twist `θ(r)` distributions. Compact chord (radial strip) approximation. |
| `bemt.py` | `BEMTAerodynamics`, `ThinAirfoilPolar`. Computes sectional lift/drag → strip forces. |
| `fwh.py` | `Farassat1ASolver`. Vectorized Newton-Raphson retarded-time solver + FWH pressure terms. |
| `solver.py` | `FWHRotorSolver`. End-to-end: `compute_pressure(x_observer, t, Omega)`. |
| `plotting.py` | `plot_blade_geometry(blade)` — planform, twist profile, distributions. `plot_rotor_top_view(rotor)` — multi-blade top-down view. |
| `test_validation.py` | 4 tests: dipole, hovering rotor, variable speed, multi-observer vectorization. |

## API

```python
from fwh_rotor_sim import Blade, Rotor, FWHRotorSolver

blade = Blade(
    radius=0.152,
    chord=lambda r: 0.015 + 0.010 * (r / r.max()),  # differentiable
    twist_deg=lambda r: 10.0 - 5.0 * (r / r.max()),  # differentiable
    hub_radius=0.02,
    n_radial=30,
)
rotor = Rotor(blade=blade, num_blades=2)
solver = FWHRotorSolver(rotor, c0=343.0, rho0=1.225)

# Single observer: x [3] → p [N_t]
# Multi-observer: x [N_obs, 3] → p [N_obs, N_t]
p = solver.compute_pressure(x_observer, t, Omega)
```

## Differentiability

**The entire pipeline is differentiable in PyTorch autograd.**

Verified: loss on pressure waveform → backward() → gradients flow to `chord(r)` and `twist_deg(r)` parameters.

Chain: `Loss(p(t))` → FWH terms → `F, Fdot` from BEMT → `cl, cd` from thin airfoil → `α = θ(r) − φ` → chord/twist parameters.

### Differentiable parameters
- `chord(r)` distribution parameters ✅
- `twist_deg(r)` distribution parameters ✅
- BEMT airfoil polar (`a0`, `cd0`, etc.) ✅

### NOT differentiable
- `n_radial`, `num_blades` — integers (tensor size)  
- `radius`, `hub_radius` — currently cast to `float()` in `__init__`  
- `torch.clamp(alpha, -20°, +20°)` — gradient is 0 outside clamp range (use soft sigmoid or remove if optimizing)

## Application ideas (future sessions)

1. **Inverse design**: optimize `c(r)` and `θ(r)` to match a target acoustic signature (recorded or desired). Loss = MSE between simulated and target pressure/spectrum. Already works end-to-end.

2. **Noise minimization**: gradient descent on twist distribution to reduce SPL at a specific observer direction or frequency band.

3. **Differentiable co-design**: train a neural network that outputs `c(r)` and `θ(r)` directly, supervised by simulated microphone recordings. The FWH simulator acts as a differentiable physics layer.

## Tests

Run: `python src/fwh_rotor_sim/test_validation.py` (the package is importable as `fwh_rotor_sim` — `src/` is on `sys.path` via the editable install, no `PYTHONPATH` needed)

All 4 tests pass. The multi-observer test verifies float32 (err ~1e-9, order-of-operations roundoff) and float64 (err ~1e-18, machine epsilon).
