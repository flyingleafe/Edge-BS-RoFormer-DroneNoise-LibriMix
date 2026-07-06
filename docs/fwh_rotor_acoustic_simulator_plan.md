# Plan: FWH-Based Rotor Acoustic Field Simulator

## Implementation Status

Phase 1 (Foundation) is complete and validated; the simulator lives in `src/fwh_rotor_sim/`. See `src/fwh_rotor_sim/AGENTS.md` for the API and differentiability details.

**Implemented modules:**
- `geometry.py` — `Blade`, `Rotor` classes with panel discretization
- `bemt.py` — `BEMTAerodynamics`, `ThinAirfoilPolar` (thin-airfoil + uniform inflow, per the Phase 1 plan)
- `fwh.py` — `Farassat1ASolver`, with a vectorized Newton-Raphson retarded-time solver (batched over panels and observers)
- `solver.py` — `FWHRotorSolver`, the end-to-end geometry+RPM → pressure API

**Validation** (`src/fwh_rotor_sim/test_validation.py`, all passing):
- Stationary dipole — exact match against the analytic solution
- Hovering rotor — BPF peak at 166.6 Hz vs. expected 166.7 Hz, SPL ≈ 42.9 dB
- Variable-speed rotor — stable (no NaN/Inf), correct amplitude modulation
- Vectorized multi-observer evaluation

This satisfies the Section 7 validation strategy for the compact/hover cases; forward-flight and thickness-noise validation (also listed in Section 7) are not yet exercised.

**Real blade geometry** (extends Section 8's "reuse for mesh/geometry" decision): real chord/twist distributions for the APC 10×7 Thin Electric propeller (FLOWUnsteady/UIUC database) load into `Blade` via `src/fwh_rotor_sim/examples/plot_real_blade.py`; interpolated planform matches the UIUC reference plot. See the `load-real-propeller-geometry` skill for the recipe.

**Also done:** an audio-generation notebook driving the simulator from RPS timeseries (`notebooks/fwh_rotor_audio_generator.ipynb`). The simulator is confirmed differentiable through the blade envelope parameters `c(r)`, `θ(r)`.

**Still open:** everything under Phase 2/3 below beyond what's listed as done — non-uniform BEMT inflow, real (XFOIL) airfoil polars, dynamic stall, ground effect, VLM/VPM, permeable-surface/CFD coupling, broadband noise, and gradient-based optimization demos.

---

## 1. Knowledge Extracted from Papers

### 1.1 Original FWH Equation (Ffowcs Williams & Hawkings, 1969)
The FWH equation extends Lighthill's acoustic analogy to flows with moving/permeable boundaries:

```
□² c²(ρ−ρ₀) = ∂/∂t [Qₙ δ(f)] − ∂/∂xᵢ [Lᵢ δ(f)] + ∂²/∂xᵢ∂xⱼ [Tᵢⱼ H(f)]
```

Where:
- `□² = (1/c²)∂²/∂t² − ∇²` is the D'Alembert operator
- `Qₙ = [ρ₀vᵢ + ρ(uᵢ−vᵢ)]n̂ᵢ` — thickness (monopole) source on surface `f=0`
- `Lᵢ = [Pᵢⱼ + ρuᵢ(uⱼ−vⱼ)]n̂ⱼ` — loading (dipole) source on surface `f=0`
- `Tᵢⱼ = ρuᵢuⱼ + Pᵢⱼ − c²(ρ−ρ₀)δᵢⱼ` — Lighthill stress tensor (quadrupole) in volume

Key insight: For low Mach number rotors, the quadrupole term is negligible; thickness + loading on the blade surface dominate.

### 1.2 Farassat 1A Formulation (Standard for Rotor Noise)
The time-domain integral solution most widely used for rotorcraft:

```
4π p'(x,t) = 1/c ∫ [L̇ᵣ / (r(1−Mᵣ)²)]_ret dS
           + ∫ [Lᵣ − L_M / (r²(1−Mᵣ)²)]_ret dS
           + 1/c ∫ [Lᵣ(rṀᵣ + c(Mᵣ−M²)) / (r²(1−Mᵣ)³)]_ret dS
           + (thickness terms with Q)
```

Where:
- `r` = distance from source to observer
- `Mᵣ = Mᵢ r̂ᵢ` = Mach number in radiation direction
- `Lᵣ = Lᵢ r̂ᵢ`, `L_M = Lᵢ Mᵢ`
- Subscript `ret` means evaluate at retarded time: `τ_ret = t − r/c`
- The `(1−Mᵣ)` denominator is the Doppler factor

### 1.3 Permeable Surface Formulation
For wind-tunnel / CFD-coupled cases, a permeable (off-body) surface can be used. The surface encloses all noise sources and flow data (ρ, u, p) is recorded on it. This avoids need for body-fitted meshes in the acoustic propagation.

### 1.4 Wind-Tunnel / Moving-Frame Simplification (Garrick Triangle)
When both source and observer are stationary in a uniform flow (common in CFD), a coordinate transform to a frame where the source moves through quiescent medium simplifies the retarded-time computation:
- Effective acoustic distance `R` replaces `r`
- `R = (−M₁d₁ + R*)/β²` where `β² = 1−M₀²`, `R* = √((M₁d₁+M₂d₂)² + β²(d₁²+d₂²+d₃²))`
- Surface normal and Mach vector derivatives vanish (uniform motion)
- This is the formulation used in OpenCFD-FWH and libAcoustics

### 1.5 Rotor-Specific Simplifications (from da Vinci paper & VSP2WOPWOP)
For subsonic rotors at low Mach numbers:
- **Compact source approximation**: Blade chord ≪ acoustic wavelength
- The blade can be treated as a line of compact sources along the span
- Only the loading term (dipole) dominates; thickness can often be neglected for thin blades
- For a given blade element at radius `r`, the local lift and drag forces become the dipole strength
- The acoustic pressure at observer is the sum over all blade elements, each retarded appropriately

### 1.6 Far-Field Approximations (Zhou & Wang 2025)
For the Green's function derivatives in frequency-domain FWH, far-field approximations simplify high-order derivative computations. Useful if we later implement frequency-domain formulations.

---

## 2. Existing Libraries & Implementations

| Name | Language | Type | Notes |
|------|----------|------|-------|
| **PSU-WOPWOP** | Fortran | Standalone FWH solver | Gold standard for rotor noise; supports permeable/impermeable, Farassat 1A, broadband models. Free by email. |
| **libAcoustics** | C++ | OpenFOAM library | Curle + FWH (Farassat 1A + GT) + BEM. Tight CFD coupling. |
| **OpenCFD-FWH** | Fortran | Standalone post-processor | Permeable FWH with MPI+OpenMP. Open source. Garrick Triangle for wind tunnel. |
| **Antares FWH** | Python | CFD post-processor | Embedded in Antares (Cerfacs). Advanced time formulation. |
| **SU2PY_FWH** | Python | SU2 post-processor | Basic Python FWH for SU2 CFD. |
| **mcmehrtens/FW-H-Solver** | Python | Educational | Farassat 1A in pure Python. Small scale. |
| **VSP2WOPWOP** | Python | Pre-processor | OpenVSP → BEMT aerodynamics → PSU-WOPWOP input. Parameter sweeps. |
| **OpenCOPTER** | D + Python bindings | Aerodynamics | Multirotor aerodynamics with wake modeling; couples to PSU-WOPWOP. |
| **helinoise** | Python | Post-processor | PSU-WOPWOP wrapper for CAMRAD II results. |

**Key Gap**: No existing library provides a **self-contained, differentiable, PyTorch/JAX-native** rotor acoustic simulator that takes geometry + RPM as input and outputs acoustic field without requiring external CFD.

---

## 3. Proposed Architecture

### 3.1 High-Level Design

```
┌─────────────────────────────────────────────────────────────┐
│                    ROTOR ACOUSTIC SIMULATOR                  │
├─────────────────────────────────────────────────────────────┤
│  Layer 1: Geometry & Kinematics                              │
│    - Rotor geometry definition (blade params, hub)          │
│    - Surface mesh generation (blade panels)                  │
│    - Rigid-body motion: rotation + optional translation      │
├─────────────────────────────────────────────────────────────┤
│  Layer 2: Aerodynamics (Load Generation)                     │
│    Option A: Analytic / BEMT (fast, approximate)              │
│    Option B: Prescribed loads (user-defined forces)           │
│    Option C: CFD coupling (permeable surface, future)        │
├─────────────────────────────────────────────────────────────┤
│  Layer 3: FWH Acoustic Propagation                         │
│    - Farassat 1A time-domain formulation                     │
│    - Retarded-time computation (scalar root finding)         │
│    - Surface integration over blade panels                     │
│    - Observer time interpolation (spline)                    │
├─────────────────────────────────────────────────────────────┤
│  Layer 4: Output & Analysis                                  │
│    - Time-domain pressure signals                            │
│    - SPL, OASPL, spectra, directivity patterns               │
│    - Optional: differentiable w.r.t. geometry/RPM (PyTorch)  │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 Why PyTorch / JAX?

Differentiability is valuable for:
- **Inverse design**: optimize blade geometry for minimal noise at specific observer locations
- **Sensitivity analysis**: ∂(SPL)/∂(RPM), ∂(SPL)/∂(blade pitch), etc.
- **Integration with ML**: train surrogate models, embed in neural networks
- **GPU acceleration**: batched evaluation of many observers or many rotor configurations

**Recommendation**: Build on **PyTorch** for immediate compatibility with the existing project (which already uses PyTorch for neural networks). JAX is more elegant for pure scientific computing but adds a dependency.

### 3.3 Core Formulation to Implement

For the user's stated goal (rotor geometry + RPM → acoustic field), the **compact moving-surface formulation** is the right level of fidelity:

**Input**:
- Blade geometry: chord distribution `c(r)`, twist `θ(r)`, span `R`, number of blades `B`
- Operating condition: angular velocity `Ω`, advance ratio `μ` (optional, for forward flight)
- Observer position `x` in ground frame

**Intermediate** (Aerodynamics):
- Blade element loads: `dL(r,t)`, `dD(r,t)` using BEMT or analytic lift models
- Convert to body-frame force density `Lᵢ(y, τ)` on each surface panel

**Acoustic Propagation** (Farassat 1A, loading term only for thin blades):

```
p'(x,t) = Σ_panels  1/(4π) * [
    (1/c) * Ḻᵣ / (r(1−Mᵣ)²)
  + Lᵣ / (r²(1−Mᵣ)²)
  + (1/c) * Lᵣ(rṀᵣ + c(Mᵣ−M²)) / (r²(1−Mᵣ)³)
]_ret * dS
```

**Retarded time equation**:
```
t = τ + r(τ)/c,   where r(τ) = |x − y(τ)|
```
This must be solved for `τ` for each panel and each observer time `t`.

For a rigid rotor in hover: `y(τ) = R_z(Ωτ) y₀`, making the retarded-time equation scalar and solvable via Newton-Raphson or bisection.

---

## 4. Implementation Plan

### Phase 1: Foundation (Minimal Viable Simulator)
**Goal**: Compute tonal noise from a hovering rotor at a single observer point.

1. **Geometry module**
   - Define a rotor via: number of blades `B`, radius `R`, hub radius `r_hub`
   - Blade discretization: `N_r` radial stations × `N_c` chordwise panels
   - Panel center positions `y₀` and normals `n` in body frame

2. **Kinematics module**
   - Given `Ω(t)`, compute `y(τ) = R_z(Ωτ) y₀` and surface velocity `v = ∂y/∂τ`
   - Compute `M = v/c`, `Mᵣ = M·r̂`

3. **Aerodynamics module (BEMT)**
   - Inflow model: uniform inflow `λ` from momentum theory
   - Blade element: local angle of attack → lift coefficient `C_l(α)` → `dL = ½ρV² c C_l dr`
   - Include drag if desired
   - Map `dL` to panel force vectors `L = p·n̂ · dS` (pressure × normal × area)

4. **FWH Kernel (Farassat 1A)**
   - For a sequence of observer times `t`, solve retarded time `τ` for each panel
   - Interpolate source quantities (`L`, `Ḻ`) at retarded times
   - Accumulate the three loading terms
   - **Optimization**: exploit azimuthal periodicity to reduce computation

5. **Validation**
   - Monopole / dipole in uniform motion (analytic solutions)
   - Compare hover SPL trends with known scaling: `SPL ∝ Ω^6` for compact loading noise

### Phase 2: Efficiency & Features

1. **Vectorization / GPU**
   - Rewrite core loops as PyTorch tensor operations
   - Batch over panels and observer times simultaneously
   - Use `torch.vmap` or manual batching for multiple observers

2. **Multiple observers & directivity**
   - Spherical array of observers
   - Compute OASPL directivity pattern

3. **Forward flight**
   - Add hub translation: `y(τ) = R_z(Ωτ)y₀ + V_hub τ`
   - Non-uniform inflow (BEMT with advance ratio `μ`)
   - Doppler shift from forward motion

4. **Thickness noise**
   - Add monopole (thickness) terms from blade volume displacement
   - Important for high-tip-Mach rotors

### Phase 3: Advanced (Optional)

1. **Permeable surface FWH**
   - Interface with CFD data (OpenFOAM, etc.) for high-fidelity predictions
   - Read surface data (ρ, u, p) on a user-defined permeable surface
   - Implement Garrick Triangle for wind-tunnel cases

2. **Broadband noise**
   - Semi-empirical Brooks-Pope-Marcolini airfoil self-noise model
   - Turbulent inflow noise models

3. **Differentiable pipeline**
   - Ensure all operations are PyTorch-differentiable
   - Demonstrate gradient-based optimization of blade twist for noise reduction

---

## 5. Key Algorithms & Challenges

### 5.1 Retarded-Time Solver
For each panel `i` and observer time `t_j`, solve:
```
g(τ) = τ + |x − y_i(τ)|/c − t_j = 0
```
- Use Newton-Raphson with good initial guess (previous τ or geometric approximation)
- For hover: `g(τ)` is periodic and smooth; typically 3-5 iterations
- For forward flight: ensure uniqueness in the causal interval
- **Critical**: Must handle subsonic motion only (`|Mᵣ| < 1`); supersonic requires special treatment

### 5.2 Source-Time Differentiation
`Ḻ = dL/dτ` is needed. Options:
- Analytic derivative if `L(τ)` has analytic form (e.g., harmonic loads)
- Numerical differentiation via finite differences on the sampled source history
- PyTorch autodiff if `L(τ)` is computed through differentiable ops

### 5.3 Interpolation
Source quantities are known at discrete source times `τ_k`. At retarded time `τ_ret`, use:
- Cubic spline interpolation (standard in FWH codes)
- Linear interpolation (faster, less accurate for high frequencies)

### 5.4 Singularities
- `(1−Mᵣ)` in denominator → never exactly zero for subsonic rotors, but can be small near Mach-wave angles
- `r → 0` for observers near the rotor disk → near-field formulation needed; for far-field `r` is safe

---

## 6. Dependencies

| Library | Purpose |
|---------|---------|
| `torch` | Core arrays, autodiff, GPU |
| `numpy` | Fallback / pre-processing |
| `scipy` | Spline interpolation (`CubicSpline`), root-finding utilities |
| `matplotlib` / `plotly` | Visualization |
| `trimesh` or `pyvista` | Surface mesh I/O and generation (optional) |
| `pytest` | Testing |

---

## 7. Validation Strategy

| Test | Expected Result |
|------|-----------------|
| Stationary monopole | Spherical spreading, `p ∝ 1/r` |
| Stationary dipole | Figure-8 directivity, analytic pressure |
| Moving dipole (subsonic) | Doppler-shifted frequency, retarded-time exact |
| Hovering rotor, compact | SPL scaling `∝ Ω^6`, dipole directivity in plane |
| Forward flight | Asymmetric directivity, higher SPL in forward arc |

---

## 8. What to Build vs. Reuse

| Component | Decision | Rationale |
|-----------|----------|-----------|
| FWH integral kernel | **Build** | Core value; must be differentiable |
| BEMT aerodynamics | **Build** (simple) | Lightweight; existing BEMT codes are heavy |
| Advanced CFD coupling | **Defer / reuse** | libAcoustics, OpenCFD-FWH already exist |
| Mesh generation | **Reuse** | `pyvista`, `trimesh`, or simple parametric panels |
| SPL analysis | **Build** | Simple RMS/FFT; project-specific needs |

---

## 9. Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Retarded-time solver is slow / unstable | Vectorized Newton solver; precompute causal time windows |
| BEMT inaccurate for complex rotors | Document limitation; provide prescribed-loads pathway |
| PyTorch autodiff through interpolation | Use differentiable splines or analytic harmonic representation |
| High memory for many panels + observers | Chunk over observers; use `torch.cuda.amp` |
| Thickness noise neglected | Add in Phase 2; not dominant for low-Mach drones |

---

## 10. Next Steps

1. **Review this plan** — confirm scope and fidelity level
2. **Prototype retarded-time solver** — scalar case for one panel, one observer
3. **Implement analytic dipole validation** — moving dipole with known solution
4. **Build minimal rotor case** — hovering flat-plate blade, BEMT loads, single observer
5. **Profile and vectorize** — convert to batched PyTorch operations

---

*Document version: 2025-05-25*
*Based on: Ffowcs Williams & Hawkings (1969), Zhang et al. OpenCFD-FWH (2023), Prakhar et al. da Vinci screw (2025), Zhou & Wang far-field approximations (2025), plus survey of existing open-source tools.*
