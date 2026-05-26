# FWH Rotor Acoustic Simulator

**Goal:** Working PyTorch-based FWH rotor noise simulator: input geometry + RPM → acoustic pressure at observer point. Differentiable, GPU-ready.
**Status:** in-progress
**Last touched:** 2025-05-25
**Resume on:** any

## Done
- Read 4 papers: FWH original, da Vinci aerial screw (Prakhar et al.), OpenCFD-FWH (Zhang et al.), far-field approximations (Zhou & Wang)
- Searched existing tools; wrote plan at `docs/fwh_rotor_acoustic_simulator_plan.md`
- Implemented core modules in `fwh_rotor_sim/`:
  - `geometry.py`: `Blade`, `Rotor` classes with panel discretization
  - `bemt.py`: `BEMTAerodynamics`, `ThinAirfoilPolar`
  - `fwh.py`: `Farassat1ASolver` with vectorized Newton-Raphson retarded-time solver
  - `solver.py`: `FWHRotorSolver` end-to-end API
  - `test_validation.py`: 3 tests (dipole, hover, variable speed)
- Fixed multiple broadcasting bugs across `geometry.py`, `solver.py`, `fwh.py` related to retarded-time solver shapes
- Stationary dipole test: **PASSED** (perfect match with analytic)
- Hovering rotor test: **PASSED** (BPF peak at 166.6 Hz vs expected 166.7 Hz, SPL ~42.9 dB)
- Variable speed test: **PASSED** (no NaN/Inf, amplitude modulation visible)

## Pending
1. ~~Vectorize observer points~~ DONE
2. Add plotting utilities and example scripts.
3. ~~Audio generation notebook from RPS timeseries~~ DONE (`notebooks/fwh_rotor_audio_generator.ipynb`)

## Prospective improvements to simulation fidelity

### Short term (improve BEMT)
- Non-uniform inflow from momentum theory (currently uses uniform inflow)
- Real airfoil polars from XFOIL instead of thin-airfoil approximation
- Simple dynamic stall model (Leishman-Beddoes)
- Ground effect correction

### Medium term (replace BEMT)
- **Vortex Lattice + Vortex Particle Method (VLM/VPM)** in PyTorch
  - Captures 3D tip vortices, blade-vortex interaction, wake distortion
  - Biot-Savart law and particle advection are analytic and differentiable
  - Sweet spot: higher fidelity than BEMT, much cheaper than CFD, fully PyTorch-native
  - Effort: ~2-4 weeks

### Long term (full Navier-Stokes)
- **Lattice Boltzmann Method (LBM) via Lettuce** — PyTorch-native, GPU-accelerated
  - Handles low-Mach flows well; complex geometry via immersed boundaries
  - Need to extract surface pressure from LBM field and feed into FWH
  - Effort: ~1-2 months
- **Neural surrogate** trained on CFD data (Fourier Neural Operator / DeepONet)
  - Replace BEMT with a learned operator
  - Requires dataset of {geometry → CFD surface pressure}
  - Fully differentiable but only interpolates training distribution
- **Adjoint CFD (SU2, OpenFOAM)** — industry standard but breaks PyTorch workflow
  - Requires external C++ solver; gradients via adjoint equations, not autograd
  - Highest fidelity but not compatible with end-to-end PyTorch optimization

## State
- Dirty working tree: new files in `fwh_rotor_sim/` not committed.
- No external dependencies beyond PyTorch, scipy, matplotlib.
- Run tests with: `cd $PROJECT && PYTHONPATH=$PROJECT:$PYTHONPATH python fwh_rotor_sim/test_validation.py`

## Decisions
- Use **compact chord approximation** (radial strips) — valid for drone rotors (chord ≪ wavelength).
- Use **Farassat 1A loading noise only** for Phase 1 (thickness negligible at low Mach).
- PyTorch backend for autodiff + GPU potential.
- BEMT aerodynamics (thin airfoil + uniform inflow) sufficient for now.

## Open questions
- None

## Real blade geometry found
- **Source:** FLOWUnsteady rotor database (github.com/byuflowlab/FLOWUnsteady) → APC .peo geometry files
- **Propeller:** APC 10x7 Thin Electric (2 blades, 10" diameter, 7" pitch)
- **Parameters:**
  - R = 0.127 m, Rhub = 0.00953 m, B = 2
  - Chord c/R peaks at ~0.21 around r/R ≈ 0.4, tapers to 0.0375 at tip
  - Twist β peaks at ~45.8° around r/R ≈ 0.2, decreases to ~11.5° at tip
- **Script:** `fwh_rotor_sim/examples/plot_real_blade.py` loads tabulated data, interpolates into `Blade` object, and plots planform / distributions / rotor top view
- **Validation:** Our interpolated distributions match UIUC reference plot (`/tmp/apce_10x7_geom.png`) from UIUC Propeller Database. Planform shape (wider mid-span, tapered tip) is visually consistent with real APC 10x7 photos.

## Resume
```bash
cd /home/flyingleafe/Research/PhD/projects/harmonic-noise-suppression
PYTHONPATH=/home/flyingleafe/Research/PhD/projects/harmonic-noise-suppression:$PYTHONPATH \
  python fwh_rotor_sim/test_validation.py
```
All 4 tests pass. Simulator is fully differentiable through blade envelope params (`c(r)`, `θ(r)`). See `fwh_rotor_sim/AGENTS.md` for API, differentiability details, and future application ideas.
