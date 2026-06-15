#import "/writing/templates/typst/slides.typ": hns-slides

#let small-table = table.with(
  inset: 3pt,
  align: left,
)

#let small-image(src) = image(src, width: 100%, height: 60%)

#show: hns-slides.with(
  title: [FWH Rotor Noise Simulator],
  subtitle: [Differentiable acoustic prediction from blade geometry + RPM],
  author: [Dmitrii Mukhutdinov],
  date: [2026-05-25],
)

= Why

- Revisited aeroacoustics reading
- Built FWH solver in a weekend to see how close it gets
- First sanity check: does simulated sound match real recordings?

= BEMT: Blade Forces

#grid(
  columns: (7fr, 3fr),
  gutter: 8pt,
  [
    *Blade Element Momentum Theory* --- slice blade into radial strips, each a 2D airfoil in combined inflow + rotation.

    Local velocity magnitude:
    $ V(r) = sqrt(v_infinity^2 + (Omega r)^2) $

    Inflow angle + angle of attack:
    $ phi = arctan(v_infinity / (Omega r)), quad alpha = theta(r) - phi $

    Strip forces:
    $ L = 1/2 rho V^2 c C_ell(alpha) $
    $ D = 1/2 rho V^2 c C_d(alpha) $

    Project to disk axes → thrust / torque per strip.
  ],
  [
    #set text(size: 0.7em)
    #table(
      columns: (1fr, 2fr),
      inset: 3pt,
      align: left,
      table.header([*Symbol*], [*Meaning*]),
      [$v_infinity$], [Inflow velocity],
      [$Omega$], [Rotor angular speed],
      [$r$], [Radial station],
      [$theta(r)$], [Geometric twist],
      [$phi$], [Inflow angle],
      [$alpha$], [Angle of attack],
      [$c$], [Chord length],
      [$C_ell, C_d$], [Lift / drag coeffs],
      [$L, D$], [Strip lift / drag],
    )
  ],
)

= Ffowcs-Williams Hawkings

#grid(
  columns: (7fr, 3fr),
  gutter: 8pt,
  [
    *FW-H = Compressible Navier-Stokes rewritten as a wave equation.* Green's function moves all nonlinear terms to the RHS as *acoustic sources*. Observer needs only surface quantities --- no volume mesh.

    Retarded time:
    $ t - tau = (|bold(x) - bold(y)(tau)|) / c_0 $

    Farassat 1A:
    $ p(bold(x),t) = 1/(4pi) [ (hat(bold(r)) dot dif(bold(F)) / dif(tau)) / (r (1 - M_r)^2) ]_"ret" $

    Only *loading noise* --- thickness is $O(M^2)$, negligible for drones.
  ],
  [
    #set text(size: 0.7em)
    #table(
      columns: (1fr, 2fr),
      inset: 3pt,
      align: left,
      table.header([*Symbol*], [*Meaning*]),
      [$bold(x)$], [Observer position],
      [$bold(y)(tau)$], [Source position],
      [$c_0$], [Speed of sound],
      [$tau$], [Retarded time],
      [$hat(bold(r))$], [Unit vector $bold(x)-bold(y)$],
      [$bold(F)$], [Strip force (loading)],
      [$r$], [Source-observer distance],
      [$M_r$], [Mach number in $r$-direction],
      [$[dot]_"ret"$], [Evaluated at retarded time],
    )
  ],
)

= Step 1: Blade Motion → Local Forces

#grid(
  columns: (7fr, 3fr),
  gutter: 8pt,
  [
    *Given:* chord $c(r)$, twist $theta(r)$, RPM $Omega(t)$. *Goal:* lift and drag on each strip.

    Azimuth (integrated from RPM):
    $ psi(t) = integral_0^t Omega(t') dif(t') $

    Panel position on the rotating blade:
    $ bold(y)(r, psi) = [r cos psi, r sin psi, 0]^top $

    Local velocity, inflow angle, AoA:
    $ V(r) = sqrt(v_infinity^2 + (Omega r)^2), quad phi = arctan(v_infinity / (Omega r)), quad alpha = theta(r) - phi $

    Lift and drag in the local airfoil frame:
    $ L = 1/2 rho V^2 c C_ell(alpha), quad D = 1/2 rho V^2 c C_d(alpha) $
  ],
  [
    #set text(size: 0.7em)
    #table(
      columns: (1fr, 2fr),
      inset: 3pt,
      align: left,
      table.header([*Symbol*], [*Meaning*]),
      [$c(r)$], [Chord distribution],
      [$theta(r)$], [Twist distribution],
      [$R$], [Blade tip radius],
      [$r$], [Radial station],
      [$Omega(t)$], [Angular velocity],
      [$psi(t)$], [Azimuth angle],
      [$bold(y)$], [Panel position],
      [$V$], [Local velocity],
      [$phi$], [Inflow angle],
      [$alpha$], [Angle of attack],
      [$L, D$], [Lift / drag per strip],
    )
  ],
)

= Step 2: Local Forces → Disk Forces

#grid(
  columns: (7fr, 3fr),
  gutter: 8pt,
  [
    *Given:* lift $L$ and drag $D$ in the local airfoil frame. *Goal:* forces in the rotor disk axes.

    The airfoil frame (drag parallel to $V$, lift normal to $V$) is rotated into the disk axes (radial, tangential, thrust) by the inflow angle $phi$:

    $ mat(F_"radial"; F_"tangential"; F_"thrust") = mat(cos phi, -sin phi; sin phi, cos phi; 0, 0) mat(D; L) $

    Written out per component:
    $ F_"radial" = D cos phi - L sin phi $
    $ F_"tangential" = D sin phi + L cos phi $
    $ F_"thrust" = 0 $ (thin-airfoil: no normal force component)

    In hover $v_infinity -> 0$, so $phi -> 0$ and $F_"tangential" approx L$ (thrust ≈ lift). Compact-chord: each strip acts as a point source carrying $bold(F) = [F_r, F_t, 0]^top$ at $bold(y)(r, psi)$.
  ],
  [
    #set text(size: 0.7em)
    #table(
      columns: (1fr, 2fr),
      inset: 3pt,
      align: left,
      table.header([*Symbol*], [*Meaning*]),
      [$L$], [Lift (normal to flow)],
      [$D$], [Drag (parallel to flow)],
      [$phi$], [Inflow angle],
      [$F_"radial"$], [Force along blade span],
      [$F_"tangential"$], [Force in rotation direction],
      [$F_"thrust"$], [Force normal to disk],
      [$bold(F)$], [Strip force vector $[F_r, F_t, 0]^top$],
      [$bold(y)$], [Strip source position],
    )
  ],
)

= Step 3: Disk Forces → Observer Pressure

#grid(
  columns: (7fr, 3fr),
  gutter: 8pt,
  [
    *Given:* strip forces $bold(F)_(i b)$ at positions $bold(y)_(i b)$. *Goal:* pressure at microphone $bold(x)$.

    Retarded time per source (sound travel delay):
    $ t - tau = (|bold(x) - bold(y)_(i b)(tau)|) / c_0 $

    Each strip radiates as a compact point source (chord $<<$ wavelength). Sum over $N_r$ strips and $B$ blades:

    $ p(bold(x), t) = sum_(i=1)^(N_r) sum_(b=1)^B 1/(4pi) [ (hat(bold(r))_(i b) dot dif(bold(F))_(i b) / dif(tau)) / (r_(i b) (1 - M_(r,i b))^2) ]_"ret" $

    Derivative $dif(bold(F)) / dif(tau)$ captures the unsteady loading --- this is where the BPF harmonics come from. The entire chain is differentiable w.r.t. $c(r)$, $theta(r)$.
  ],
  [
    #set text(size: 0.7em)
    #table(
      columns: (1fr, 2fr),
      inset: 3pt,
      align: left,
      table.header([*Symbol*], [*Meaning*]),
      [$bold(x)$], [Microphone position],
      [$bold(y)_(i b)$], [Strip $i$, blade $b$ position],
      [$c_0$], [Speed of sound],
      [$tau$], [Retarded time],
      [$hat(bold(r))_(i b)$], [Unit vector source → observer],
      [$r_(i b)$], [Source-observer distance],
      [$M_(r,i b)$], [Mach number in observer direction],
      [$[dot]_"ret"$], [Evaluated at retarded time],
      [$p(bold(x),t)$], [Pressure at microphone],
      [$N_r$], [Number of radial strips],
      [$B$], [Number of blades],
    )
  ],
)

= Blade Geometry

APC 10x7 Thin Electric -- real tabulated data

#image("assets/blade_geometry.png", width: 100%)

= DREGON Setup

8 microphones in cubic array underneath 4 rotors

#image("assets/dregon_geometry.png", width: 60%, height: 60%)

= Simulated -- 90 RPS, Rotor 0 → Mic 0

Constant speed. Pure tonal harmonics.

#image("assets/simulated_90rps.png", width: 80%, height: 60%)

= Real Recording -- Motor1_90.wav, 6–10 s

Same BPF. Plus broadband noise.

#grid(
  columns: (1fr, 1fr),
  gutter: 8pt,
  image("assets/real_90rps.png", width: 100%, height: 55%),
  image("assets/real_spectrogram.png", width: 100%, height: 55%),
)

= Takeaway

#table(
  columns: (1fr, 2fr, 2fr),
  inset: 6pt,
  align: left,
  table.header(
    [], [*Simulated*], [*Real*],
  ),
  strong[Tonal], [yes BPF + harmonics], [yes BPF + harmonics],
  strong[Broadband], [no], [yes turbulent + motor noise],
)

Next: replace BEMT with VLM/VPM or LBM to capture broadband.

= First Goal

Can we deduce the *exact rotor phase* from the *blade passing frequency phase* of real sound?

That is the bet.
