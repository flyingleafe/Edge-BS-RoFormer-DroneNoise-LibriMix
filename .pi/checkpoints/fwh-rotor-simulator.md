# FWH Rotor Acoustic Simulator

**Goal:** Working PyTorch-based FWH rotor noise simulator: input geometry + RPM → acoustic pressure at observer point. Differentiable, GPU-ready.
**Status:** v2 — VLM-FWH implemented, validated against BEMT-FWH, FWH ambiguity bug fixed
**Last touched:** 2026-05-26

## Done (v1: 2025-05-25)
- Read 4 papers: FWH original, da Vinci aerial screw (Prakhar et al.), OpenCFD-FWH (Zhang et al.), far-field approximations (Zhou & Wang)
- Implemented core modules in `fwh_rotor_sim/`:
  - `geometry.py`: `Blade`, `Rotor` classes with panel discretization
  - `bemt.py`: `BEMTAerodynamics`, `ThinAirfoilPolar`
  - `fwh.py`: `Farassat1ASolver` with vectorized Newton-Raphson retarded-time solver
  - `solver.py`: `FWHRotorSolver` end-to-end API
  - `test_validation.py`: 4 tests (dipole, hover, variable speed)
- Stationary dipole test: **PASSED**
- Hovering rotor test: **PASSED** (BPF at 166.6 Hz, SPL ~42.9 dB)
- Variable speed test: **PASSED**

## Done (v2: 2026-05-26 — VLM Implementation + Physical Fidelity Upgrade)

### Vortex Lattice Method (VLM) — implemented ✅

**Architecture:**
- `vlm.py`: VortexLatticeSolver, VortexParticle, VLMFWHIntegrator
- VortexParticle: Biot-Savart velocity, GPU-vectorized, fully differentiable
- VortexLatticeSolver: bound circulation from Kutta-Joukowski, force decomposition (F_norm/F_tang), FWH loading

**Physics chain:**
```
chord_scale → blade.c(r) → L'(r) = 0.5*rho*U²*c*cl → Γ(r) = L'/rho
                                    → F_norm = L'*dr (normal force)
                                    → F_tang = D'*dr (drag force)
→ FWH pressure via Farassat 1A loading terms
```

**Key fixes:**
1. **FWH retarded-time ambiguity bug**: When y_func(tau) with tau [N_t] returns y [N_t, 3],
   the solver now correctly interprets this as a single source (N_sources=1), not
   N_sources = N_t sources. Fixed by adding disambiguation logic in `solve_retarded_time`.

2. **VLM coordinate system**: Fixed y = (r*cos, r*sin, 0) to match BEMT rotation axis (z).
   Previously used (0, r*cos, r*sin) which is a vertical rotor — gave wrong acoustic.

3. **VLM force direction**: F = [F_tang*sin, -F_tang*cos, F_norm] (drag opposes rotation).
   Corrected from previous wrong sign convention.

**Validation:**
- VLM vs BEMT: SPL difference = 2.0 dB (within tolerance) ✅
- All 4 existing tests still pass ✅
- Full autograd chain: chord_scale → blade.c → Γ → pressure → SPL ✅
- Gradient from pressure to chord_scale: 8.69 (meaningful, non-zero)

**Libraries tested:**
- **Lettuce**: LBM via D2Q9, GPU-native (102 MLUPS), BGK collision is differentiable.
  Issue: API rough edges (D2Q9 class vs instance, reporter takes list, `sim(100)` syntax).
- **PhiFlow**: Fluid simulation framework, torch backend works. API changed (Box needs dimension names).
  Issue: API versioning (v2 vs v3), different extrapolation module names.
- **VLM/VPM**: Biot-Savart on GPU — 94x speedup from batching, fully differentiable chain.
  Memory: 58KB for 5k vortex elements.

### Decision: VLM/VPM over LBM ✅

**VLM/VPM wins because:**
1. **Differentiable end-to-end**: PyTorch autograd handles Biot-Savart natively (matmul, cross, norm all differentiable).
   LBM collision (BGK) is differentiable, but LBM requires ~10-100 simulation steps
   per pressure evaluation → slow for inverse design gradients.
2. **GPU efficiency**: VLM with 40000 particles, 100 observers, 50 steps: 0.025s on GPU (404M vortex-ops/s).
   LBM at 64×64 needs thousands of steps → even with GPU, much slower.
3. **Memory**: VLM uses <100KB for 40k particles. LBM at 64³ needs 12.8M cells × 9 distribution × 4 bytes = ~460MB.
4. **Physical fidelity for this use case**: For tonal noise (blade passage harmonics), the
   bound vortex + loading model is the dominant source. VLM captures this directly.
   LBM is needed only for broadband / separated flow (dynamic stall, etc.).
5. **Gradient quality**: VLM gradients w.r.t. chord are well-defined (∂Γ/∂c via cl(α(c))).
   LBM gradients would require backprop through the full simulation.

**LBM (Lettuce) still useful for:**
- Broadband noise from turbulent inflow (needs separate modeling)
- Validation of VLM against high-fidelity simulation
- Cases with large separation (landing, aggressive maneuvers)

## Pending

### Short term (improve BEMT/VLM)
- Non-uniform inflow from momentum theory (currently uses uniform inflow)
- Real airfoil polars from XFOIL instead of thin-airfoil approximation
- Dynamic stall model (Leishman-Beddoes) for broadband noise

### Medium term (VLM enhancements)
- **Wake vortex particles**: roll up tip vortices, advect downstream, include in acoustic
- **BVI (Blade-Vortex Interaction)**: close wake modeling for low-altitude effects
- **GPU acceleration**: vectorize over strips (all strips in one FWH call instead of loop)

### Long term (full Navier-Stokes)
- **Lattice Boltzmann Method (Lettuce)**: for broadband + separated flow
  - Surface pressure extraction from LBM field → FWH input
  - Need to decide: extract on-body (noisy) vs off-body permeable surface
- **Neural surrogate (FNO/DeepONet)**: trained on {geometry → CFD pressure}
  - Requires dataset of {blade geometry → CFD surface pressure}
  - Fully differentiable but interpolates training distribution

## State
- Working VLM-FWH with full differentiability
- FWH ambiguity bug fixed (now correctly handles strip-by-strip calls)
- Both lettuce and phiflow installed and tested on GPU
- VLM dominates LBM for this use case (differentiation + speed + memory)

## Resume
```bash
cd /home/flyingleafe/Research/PhD/projects/harmonic-noise-suppression
PYTHONPATH=/home/flyingleafe/Research/PhD/projects/harmonic-noise-suppression:$PYTHONPATH \
  python fwh_rotor_sim/test_validation.py
```
All 4 tests pass. VLM-FWH gives SPL within 2 dB of BEMT-FWH. Differentiable chain verified.

## Files
- `fwh_rotor_sim/vlm.py`: VortexLatticeSolver, VortexParticle, VLMFWHIntegrator
- `fwh_rotor_sim/fwh.py`: Farassat1ASolver (with retarded-time ambiguity fix)
- `fwh_rotor_sim/geometry.py`: Blade, Rotor (unchanged)
- `fwh_rotor_sim/bemt.py`: BEMTAerodynamics (unchanged)
- `fwh_rotor_sim/solver.py`: FWHRotorSolver (unchanged)
- `fwh_rotor_sim/test_validation.py`: 4 tests (unchanged)