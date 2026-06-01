# How to predict rotor noise from blade geometry

## 1. Why we care

A drone rotor makes noise. We want to know: **if I change the blade shape, how does the sound change?**

The sound comes from two physical sources:
- **Loading noise:** the blade pushes air, and that unsteady force radiates sound
- **Thickness noise:** the blade displaces air as it spins

At low Mach numbers (drone rotors, tip Mach ~ 0.1–0.3), thickness noise is negligible. So we only need to model loading noise.

The chain is:

```
blade geometry + RPM  →  aerodynamic forces  →  acoustic pressure at microphone
```

This lecture walks through that chain step by step.

---

## 2. Blade geometry

A rotor blade is defined by three functions of radius $r$:

| Quantity | Symbol | Meaning |
|----------|--------|---------|
| Chord | $c(r)$ | Width of the blade at radius $r$ |
| Twist | $\theta(r)$ | Angle of the blade section relative to the rotor disk |
| Airfoil | — | Cross-sectional shape (we approximate with thin-airfoil theory) |

We also need:
- $R$: tip radius
- $r_\text{hub}$: hub radius (where the blade starts)
- $B$: number of blades

For this work we use the **APC 10×7 Thin Electric** propeller — a real, commercially available propeller whose geometry is available from the UIUC Propeller Database.

---

## 3. Blade Element Momentum Theory (BEMT)

### 3.1 The core idea

Instead of solving the full 3D flow around the blade, BEMT slices the blade into **radial strips** and treats each strip as a **2D airfoil** in a local flow. This is valid because the chord is much smaller than the radius, so cross-sectional flow dominates spanwise flow.

### 3.2 Local velocity at a strip

A strip at radius $r$ moves tangentially at speed $\Omega r$ (from rotation) and sees the inflow velocity $v_\infty$ (from the aircraft moving forward, or from induced downwash in hover). The local velocity magnitude is:

$$V(r) = \sqrt{v_\infty^2 + (\Omega r)^2}$$

The local flow hits the strip at an **inflow angle**:

$$\phi = \arctan\frac{v_\infty}{\Omega r}$$

### 3.3 Angle of attack

The blade section is twisted by $\theta(r)$ relative to the rotor disk. The angle between the local flow and the chord line is:

$$\alpha = \theta(r) - \phi$$

This is the **angle of attack** — the quantity that determines how much lift and drag the airfoil produces.

### 3.4 Lift and drag per strip

Using thin-airfoil theory (or tabulated polars), the lift and drag coefficients are functions of $\alpha$. The forces on a strip of width $dr$ are:

$$L = \frac{1}{2} \rho V^2 \, c \, C_\ell(\alpha)$$

$$D = \frac{1}{2} \rho V^2 \, c \, C_d(\alpha)$$

These are in the **local airfoil frame**: drag is parallel to the local flow $V$, lift is perpendicular to it.

### 3.5 From airfoil frame to disk axes

The rotor disk has three natural axes:
- **Radial** ($x$): along the blade span
- **Tangential** ($y$): in the direction of rotation
- **Thrust** ($z$): normal to the disk plane

The lift and drag are defined in a frame rotated by $\phi$ relative to the disk. To get forces in the disk axes, we apply a rotation matrix:

$$\begin{bmatrix} F_\text{radial} \\ F_\text{tangential} \\ F_\text{thrust} \end{bmatrix} = \begin{bmatrix} \cos\phi & -\sin\phi \\ \sin\phi & \cos\phi \\ 0 & 0 \end{bmatrix} \begin{bmatrix} D \\ L \end{bmatrix}$$

Or, written out:
- $F_\text{radial} = D \cos\phi - L \sin\phi$
- $F_\text{tangential} = D \sin\phi + L \cos\phi$
- $F_\text{thrust} = 0$ (thin-airfoil approximation: no force normal to the disk plane)

**In hover**, $v_\infty \to 0$, so $\phi \to 0$ and $F_\text{tangential} \approx L$ — the lift essentially becomes the thrust.

### 3.6 Summary of BEMT output

For each radial strip $i$ at time $t$, BEMT gives us a **force vector**:

$$\mathbf{F}_i(t) = \big[ F_\text{radial}, \; F_\text{tangential}, \; F_\text{thrust} \big]^\top$$

This force changes with time because:
- $\Omega(t)$ changes (variable RPM)
- The blade azimuth $\psi(t)$ changes (rotation)
- The local flow changes (if the aircraft is maneuvering)

---

## 4. Where is the force applied?

The force $\mathbf{F}_i(t)$ is applied at the position of the strip on the rotating blade. If the blade azimuth is $\psi(t)$, the strip at radius $r_i$ is at:

$$\mathbf{y}_i(t) = \big[ r_i \cos\psi(t), \; r_i \sin\psi(t), \; 0 \big]^\top$$

For a rotor with $B$ blades, blade $b$ has an additional phase offset:

$$\mathbf{y}_{i,b}(t) = \big[ r_i \cos(\psi + \phi_b), \; r_i \sin(\psi + \phi_b), \; 0 \big]^\top$$

where $\phi_b = 2\pi b / B$ is the azimuthal offset of blade $b$.

---

## 5. The Ffowcs-Williams Hawkings (FW-H) equation

### 5.1 What is it?

The FW-H equation is the **compressible Navier-Stokes equations rewritten as a wave equation**. Instead of solving for the full flow field, we use a mathematical trick (Green's function) to express the acoustic pressure at an observer point in terms of **surface quantities only**.

The key insight: the blade is a solid surface. The air cannot penetrate it. This boundary condition is enough — we don't need to know what happens in the air *away* from the blade.

### 5.2 The retarded time

Sound travels at finite speed $c_0$. The pressure the observer hears at time $t$ was emitted by the source at an earlier time $\tau$, called the **retarded time**:

$$t - \tau = \frac{|\mathbf{x} - \mathbf{y}(\tau)|}{c_0}$$

This is a nonlinear equation because the source is moving ($\mathbf{y}$ depends on $\tau$). We solve it with Newton-Raphson iteration.

### 5.3 Farassat 1A (compact source)

When the chord is much smaller than the acoustic wavelength (true for drone rotors below ~1 kHz), each strip acts as a **compact point source**. The pressure from strip $i$, blade $b$ is:

$$p_{ib}(\mathbf{x}, t) = \frac{1}{4\pi} \left[ \frac{\hat{\mathbf{r}}_{ib} \cdot \frac{d\mathbf{F}_{ib}}{d\tau}}{r_{ib} \, (1 - M_{r,ib})^2} \right]_\mathrm{ret}$$

The total pressure is the sum over all strips and blades:

$$p(\mathbf{x}, t) = \sum_{i=1}^{N_r} \sum_{b=1}^{B} p_{ib}(\mathbf{x}, t)$$

### 5.4 What each term means

| Term | Meaning |
|------|---------|
| $\hat{\mathbf{r}}_{ib}$ | Unit vector from source to observer |
| $r_{ib}$ | Distance from source to observer |
| $M_{r,ib}$ | Mach number of the source toward the observer |
| $d\mathbf{F}_{ib}/d\tau$ | Rate of change of the force (unsteady loading) |
| $(\cdot)_\mathrm{ret}$ | Everything evaluated at retarded time $\tau$ |
| $(1 - M_r)^{-2}$ | Doppler amplification (source moving toward observer) |

The **unsteady loading term** $d\mathbf{F}/d\tau$ is the source of the tonal noise. Each blade pass creates a pulse. At constant speed, these pulses are periodic, producing a line spectrum at the **blade passing frequency** (BPF = $B \times$ RPS) and its harmonics.

---

## 6. Full pipeline summary

```
Step 1: Geometry
  c(r), theta(r), R, B  →  Blade object with N_r radial strips

Step 2: Kinematics
  Omega(t)  →  azimuth psi(t)  →  panel positions y_i(t)

Step 3: Aerodynamics (BEMT)
  y_i(t), Omega(t)  →  local velocity V  →  angle of attack alpha
  alpha, c(r)  →  lift L, drag D  →  disk forces F_radial, F_tan, F_thrust

Step 4: Acoustics (FW-H)
  F_i(t), y_i(t)  →  retarded time tau  →  observer pressure p(x,t)
```

---

## 7. What does the simulation produce?

### 7.1 Constant 90 RPS, rotor 0 → microphone 0

At constant speed, the force on each strip is periodic. The FW-H sum produces a **purely tonal signal**:
- Fundamental at 180 Hz (BPF = 2 blades × 90 RPS)
- Harmonics at 360, 540, 720, 900 Hz...

The waveform is a train of identical pulses. The spectrum is a line spectrum.

### 7.2 Real recording

The real recording (DREGON Motor1_90.wav) shows:
- The same tonal peaks at 180, 360, 540 Hz
- **Additional broadband noise** above ~2 kHz

This broadband noise comes from:
- Turbulent trailing-edge noise (not captured by BEMT)
- Tip vortex noise (3D effect, BEMT is 2D per strip)
- Motor/gearbox mechanical noise

### 7.3 The gap

| Feature | BEMT + FW-H | Reality |
|---|---|---|
| Tonal harmonics | ✅ Exact | ✅ Present |
| Broadband noise | ❌ Absent | ✅ Present |
| Amplitude accuracy | ⚠️ Approximate | ✅ |

The tonal part is qualitatively correct because the BPF is determined purely by kinematics (RPM × blades), not by the aerodynamic model. But the **broadband part requires a higher-fidelity flow solver**.

---

## 8. Why this is exciting: differentiability

The entire pipeline is implemented in **PyTorch**. That means:
- $c(r)$ and $\theta(r)$ are differentiable parameters
- The loss $\mathcal{L} = \| p_\text{simulated} - p_\text{recorded} \|^2$ has a gradient
- We can run **gradient descent** to find the blade shape that produces a target sound

This is the endgame: **optimize blade geometry from acoustic recordings**.

But first, we need the simulation to be accurate enough. The current BEMT model captures the tonal signature but misses broadband. The next step is to replace BEMT with a higher-fidelity aerodynamic model (Vortex Lattice / Vortex Particle Method, or Lattice Boltzmann) that can capture turbulent noise, while keeping the pipeline differentiable.

---

## 9. The first research question

Before we optimize blade shapes, we need to answer a simpler question:

> **Can we deduce the exact rotor phase from the blade passing frequency phase of real sound?**

In other words: given a microphone recording of a single rotor, can we extract the instantaneous azimuth angle $\psi(t)$ from the acoustic signal?

If yes, then the FWH solver becomes a **forward model** for rotor state estimation. If no, then the acoustic signal is not informative enough about rotor kinematics, and we need to rethink the approach.

That is the bet.
