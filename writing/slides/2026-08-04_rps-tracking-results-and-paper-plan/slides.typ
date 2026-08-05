#import "@preview/touying:0.7.4": *
#import "/writing/templates/typst/slides.typ": hns-slides

// Notes-visible build: `make notes` (typst --input notes=1). Default build is clean.
#let notes-mode = sys.inputs.at("notes", default: none)

#show: hns-slides.with(
  title: [RPS tracking: results, the ceiling, and the paper plan],
  subtitle: [What limits blind rotor-speed reading from audio, and what that dictates],
  author: [Dmitrii Mukhutdinov],
  date: [2026-08-04],
  show-notes: if notes-mode != none { bottom } else { none },
)

= Agenda

- Headline RPS-tracking results, both drones
- The classical (VK) tracking chain, step by step
- What actually limits accuracy --- measured, not guessed
- The combined paper: status and the next 4 weeks

#speaker-note[
  Two months went into one question: how well can rotor speed be read from
  audio alone, with no telemetry? This deck has four parts: the numbers, how
  the classical chain gets them, why a hard physical limit shows up in the
  data, and what that means for the paper due in about four weeks and for
  the next research question after it.
]

= Headline results --- fixed 15-window raw protocol

#text(size: 0.82em)[
  #table(
    columns: (1.5fr, auto, auto),
    align: (left, center, center),
    stroke: 0.5pt + luma(180),
    inset: 6pt,
    table.header([*tracker*], [*DREGON cruise*], [*FLY124 cruise*]),
    [blind VK, full-range init], [*1.807*], [2.515 #super[#sym.dagger.double]],
    [guarded peeled alternation (flagship)], [1.841], [2.274],
    [VK, telemetry-init (oracle)], [0.85], [0.78 #super[#sym.ast]],
    table.hline(start: 0, stroke: 0.4pt + luma(210)),
    [CKLA, phase-only, 4 s context], [2.55], [1.29],
    [KLA (plain, no rotation)], [3.13], [1.81],
    [transformer], [3.38], [3.47],
    [uni_gru128], [4.25], [2.26],
    table.hline(start: 0, stroke: 0.4pt + luma(210)),
    [hybrid (neural init + pi_kalman)], [flat #super[†]], [*0.64* #super[†]],
  )
]

#v(0.3em)
#text(size: 0.72em)[
  *VK* = Vold--Kalman comb tracker (classical, no learning). *CKLA/KLA* =
  our neural tracker family. *pi_kalman* = phase-increment refiner that
  closes the VK chain. *Guarded peeled alternation* = flagship: peel
  (subtract the other rotors' reconstructed combs) + pi_kalman, iterated to
  plateau, falling back to the init on windows where the peel-energy guard
  fails.
]
#v(0.15em)
#text(size: 0.72em)[
  MAE#footnote[PIT-MAE: mean absolute error after permutation-invariant
  (Hungarian) rotor matching to ground truth.] (rev/s), PIT-aligned to raw
  telemetry, 15 frozen 16 s windows.
  #super[#sym.ast] oracle FLY124 cell: re-score after the telemetry
  recalibration is still pending, so the number shown predates it.
  #super[#sym.dagger.double] corrects a stale pre-recalibration pin (was
  2.699). #super[†] smoke-window subset, not the full 15-window protocol;
  "flat" = no gain over the blind chain on that pool.
]

#speaker-note[
  One table, one protocol: 15 frozen 16 s windows, PIT-matched MAE against
  raw (unsmoothed) telemetry. Two things happened since the last deck.
  First, Michael's telemetry had a drifting clock and a 0.7% rev/s scale
  error --- both measured, fixed, and every number here is post-fix.
  Second, a seeding bug in the blind chain flipped one FLY124 window between
  two tied candidate rates; fixing it recovered 1.6 rev/s and the FLY124
  init cell is now 2.515, not the stale 2.699 pin from before the fix.
  Third and newest: the flagship method declared today adds a peeled
  alternation loop on top of the init (next five slides show it step by
  step) --- it wins the FLY124 column among fully-blind rows, 2.515 to
  2.274, but on DREGON the plain init stays best (1.807 vs 1.841): the
  extra pass cannot recover the displaced-comb bias there, so bold the
  init on that column. Read the rows: blind VK wins DREGON outright, CKLA
  wins FLY124 among neural rows, and the oracle row --- VK started from the
  true answer --- still sits at 0.8, not 0. That gap is the ceiling this
  deck is about. The neural rows are final post-recalibration numbers; only
  the starred oracle cell still awaits its re-score.
]

// ─── VK algorithm, step by step ────────────────────────────────────────────

#let vk-steps = (
  [Input & global stage],
  [Ramp handling],
  [Per-rotor decoupling],
  [Envelope solve + peel],
  [Peeled pi_kalman (loop)],
)

#let stepper(active) = {
  set text(size: 0.62em)
  align(center, stack(
    dir: ltr,
    spacing: 0.4em,
    ..vk-steps.enumerate().map(((i, s)) => {
      let on = i == active
      box(
        fill: if on { rgb("#1f77b4") } else { luma(230) },
        inset: (x: 0.6em, y: 0.5em),
        radius: 3pt,
      )[#text(fill: if on { white } else { luma(90) }, weight: if on { "bold" } else { "regular" })[#s]]
    }),
  ))
}

#let cbox(body, fill: luma(235), stroke-color: luma(160)) = box(
  fill: fill, stroke: 0.7pt + stroke-color, inset: 8pt, radius: 3pt, body,
)

= The VK chain, step 1 --- input & global stage

#stepper(0)
#v(0.5em)

#grid(
  columns: (0.85fr, 1.15fr),
  gutter: 1.2em,
  [
    Whitened multi-channel spectrogram #sym.arrow full-range comb-matched
    Viterbi (emission = comb score, smooth transition penalty) finds one
    coarse rate track $c(t)$ before anything is assigned to a rotor.
    #v(0.5em)
    #text(size: 0.78em)[
      $ J[c] = -sum_t "score"(c(t) mid(|) y(t)) + lambda sum_t (c(t) - c(t-1))^2 $
    ]
  ],
  align(center, image("assets/stepper_coarse_init.png", height: 62%)),
)

#speaker-note[
  This is the entry point: whiten the multichannel spectrogram, then run a
  full-rate-range Viterbi search whose emission score is how well a comb at
  rate c(t) matches the spectrum, with a transition penalty that keeps the
  path smooth. The output is one coarse global track, not yet per rotor ---
  that split comes two steps later.
]

= The VK chain, step 2 --- ramp handling

#stepper(1)
#v(0.5em)

#grid(
  columns: (0.95fr, 1.05fr),
  gutter: 1.2em,
  align(horizon)[
    #align(left)[
      #cbox(text(size: 0.72em)[wrong octave\ (blade-pass lines only)])
      #text(size: 1.1em)[#sym.arrow.r]
      #cbox(text(size: 0.72em)[line-ratio check])
      #text(size: 1.1em)[#sym.arrow.r]
      #cbox(fill: rgb("#dceaf7"), stroke-color: rgb("#1f77b4"))[#text(size: 0.72em)[energy-timed bridge across spool-up]]
    ]
    #v(0.6em)
    Warm-up spectra carry only even (blade-pass) lines, so the seeder used
    to octave-promote onto the wrong track. Fix: line-ratio check to
    disambiguate the octave, plus an energy-timed bridge across spool-up.
    #v(0.5em)
    #align(left, text(size: 0.85em, style: "italic")[
      Ramp/warmup windows went *15--36 #sym.arrow 2.9--4.0* MAE.
    ])
  ],
  align(center, image("assets/stepper_viterbi_c.png", height: 62%)),
)

#speaker-note[
  Take-off and landing are where blind VK used to fail outright. During
  spool-up the spectrum only shows blade-pass lines, and the old seeder read
  that as a lower rate one octave down. The fix checks the blade-pass
  line-ratio to pick the right octave, and bridges across the ramp using the
  timing of the energy rise rather than trying to track frequency alone
  through it. This single fix took the worst windows from 15-36 rev/s MAE
  down to 2.9-4.0.
]

= The VK chain, step 3 --- per-rotor decoupling

#stepper(2)
#v(0.5em)

#grid(
  columns: (0.95fr, 1.05fr),
  gutter: 1.2em,
  align(horizon)[
    #text(size: 0.68em)[
      #cbox(text(size: 0.85em)[global track $c(t)$])
      #text(size: 1.1em)[#sym.arrow.r.double]
      #cbox(text(size: 0.85em)[$r_1$]) #cbox(text(size: 0.85em)[$r_2$]) #cbox(text(size: 0.85em)[$r_3$])
      #cbox(fill: rgb("#fde3e3"), stroke-color: rgb("#d62728"))[
        #text(size: 0.85em)[$r_4$ #sym.arrow.b residual re-seed]
      ]
    ]
    #v(0.6em)
    - Split a corridor around the global track per candidate rotor rate.
    - Gated residual re-seed recovers rotors invisible in the global comb.
    - Twin pairs 0.5--0.65 rev/s apart: two residual candidates can tie to
      within 0.06% of score.
    #v(0.4em)
    #text(size: 0.82em, style: "italic")[
      A seeding coin-flip on one tied pair cost 1.57 rev/s on FLY124: fixed
      by two guards on the residual re-scan, not by better search.
    ]
  ],
  align(center, image("assets/stepper_vit2dsp.png", height: 62%)),
)

#speaker-note[
  Once the global track and the ramp are handled, each rotor needs its own
  seed. The corridor split proposes candidates around the global track; a
  residual re-scan looks for a rotor the global comb missed entirely (the
  4th rotor case). The hard failure mode: twin rotors 0.5 to 0.65 rev/s
  apart produce two residual candidates that score within 0.06% of each
  other, so the ranking is a coin flip. Two guards fixed it: a maximum span
  on the accepted candidate set (an accepted residual can't stretch past
  what a real quadrotor's rotor band allows), and mutual de-duplication of
  the accepted candidates against each other. Together they were worth 1.57
  rev/s on the one FLY124 window that flipped.
]

= The VK chain, step 4 --- envelope solve, then peel

#stepper(3)
#v(0.5em)

#grid(
  columns: (0.95fr, 1.05fr),
  gutter: 1.2em,
  align(horizon)[
    All rotors solved *jointly* (envelopes $a_m(t)$ at the current tracks,
    bandwidth about 1 Hz), then *peel*: reconstruct each rotor's comb
    waveform and subtract the OTHER rotors' combs from the audio.
    #v(0.4em)
    #text(size: 0.8em)[
      $ J[a] = sum_t |y(t) - sum_m "Re"[a_m (t) c_m (t)]|^2 + sum_m rho_m^2 norm(Delta^p a_m)^2 $
    ]
    #v(0.5em)
    #text(size: 0.82em, style: "italic")[
      Guard: if the subtraction fails to remove energy (ramp/warmup
      windows, mis-phased peel), fall back to the init instead of the
      peeled residual.
    ]
  ],
  align(center, image("assets/stepper_refine.png", height: 62%)),
)

#speaker-note[
  Two things happen in this step. First, the usual Vold-Kalman data +
  smoothness solve, jointly across all four rotors because overlapping
  combs share spectral bins. Second, new today: each rotor's own comb
  reconstruction is used to peel the OTHER rotors out of the audio, so the
  next step sees a rotor's signal with its neighbours' energy subtracted
  instead of mixed in. The chart shows one application on a real FLY124
  window: most of each rotor's own comb energy is removed from the others'
  contribution, and the guard checks that the subtraction actually reduced
  residual energy. On the ramp and warmup windows the tracks are not locked
  yet, so the peel is mis-phased and would inject energy back in --- the
  guard catches that and keeps the init instead.
]

= The VK chain, step 5 --- peeled pi_kalman, iterate to plateau

#stepper(4)
#v(0.5em)

#grid(
  columns: (0.95fr, 1.05fr),
  gutter: 1.2em,
  align(horizon)[
    pi_kalman runs on the *peeled* residual: $Delta psi_k = arg(z_k (t)
    macron(z)_k (t-1))$, variance from envelope SNR + a per-harmonic
    random-walk rate $q_k$; pair-joint mode for twins; RTS smoother.
    #v(0.5em)
    Loop back to step 4 with the refined tracks, re-solve, re-peel: 2--4
    applications reach a plateau instead of degrading.
    #v(0.4em)
    #text(size: 0.82em, style: "italic")[
      FLY124 w03: 1.17 #sym.arrow 0.77. Pooled twin-pair windows (w03+w04):
      0.94 #sym.arrow 0.51.
    ]
  ],
  align(center, image("assets/stepper_pi_kalman.png", height: 62%)),
)

#speaker-note[
  This closes the loop: pi_kalman reads a per-frame phase-increment
  measurement off the demodulated envelope, weights it by that frame's
  SNR, and models each harmonic's own rate drift as a random walk with its
  own variance q_k, now on audio that has had the other rotors peeled out.
  The revised finding: naive re-application of pi_kalman degrades (it is
  re-reading its own noise back), but peeled iteration converges, because
  each pass sees cleaner data instead of the same mixture. The chart is
  the real trace on FLY124 window 03: naive plateaus higher than it
  starts, peeled comes down and holds. Two to four applications is enough;
  the twin-pair windows gain the most because that is where the neighbour's
  comb was contaminating the fit the most. On DREGON the picture is flatter
  --- the displaced-comb bias there is not something either the envelope
  solve or the peel can fix, so the flagship's honest DREGON number is
  still its init.
]

= Output comparison --- DREGON cruise window

#align(center, image("assets/compare_dregon.png", height: 78%))

#align(center, text(size: 0.85em)[
  Dotted = raw telemetry, solid = tracker, PIT-aligned, one live 16 s window.
])

#speaker-note[
  Same window, two trackers. VK tracks the fast fluctuations tightly ---
  about 1.0 rev/s of scatter around a steady window, consistent with the
  0.688-vs-smoothed reference. CKLA's residual on this window is the
  anchor-collapse pattern: one rotor sits off by roughly the pair spacing,
  because free flight is where its precision gap is largest. DREGON is
  where the classical chain still wins outright.
]

= Output comparison --- FLY124 cruise window

#align(center, image("assets/compare_fly124.png", height: 70%))

#align(center, text(size: 0.85em)[
  Dotted = raw telemetry, solid = tracker, PIT-aligned, one live 16 s window
  --- this is blind VK's *best* FLY124 window.
])

#align(center, text(size: 0.85em, style: "italic")[
  Pooled over all 4 FLY124 cruise windows: CKLA 1.29 beats blind VK gated
  2.38 --- blind VK has two much worse windows this window does not show.
])

#speaker-note[
  This is one of blind VK's best FLY124 windows, and on it the two families
  are close (0.95 vs 1.06 rev/s) --- CKLA needs no telemetry seed to get
  there. Zoom out to the pooled table: over all four FLY124 cruise windows
  CKLA (1.29) clearly beats the blind chain (2.38), because blind VK has
  two much worse windows this one doesn't show, and the hybrid (neural init
  feeding pi_kalman, 0.64 on the smoke-window subset) is the best row on the
  whole table. Neither family dominates across both drones --- that split
  is the paper's empirical spine, and it is also why the paper reports both.
]

= Noise model v0 --- what the textbook assumes

#v(0.6fr)
#align(center, text(size: 1.25em)[
  $ n(t) = sum_i sum_k a_(i k) cos(k phi_i (t) + psi_(i k)),
    quad phi_i (t) = 2 pi integral r_i (tau) dif tau $
])

#v(1em)

#align(center, text(size: 1.0em)[
  Sum over rotors $i$ and harmonics $k$: each rotor's harmonic comb has
  random *initial* phases $psi_(i k)$, then deterministic evolution from its
  own shaft rate $r_i (t)$. This is what our generator, and every standard
  order-tracking method, assumes.
])
#v(0.6fr)

#speaker-note[
  Start from the textbook model: the recorded noise is a sum over four
  rotors and, for each rotor, a sum over harmonics k. Each term is a cosine
  at k times that rotor's shaft phase, plus a random phase picked once at
  the start of the recording. The shaft phase itself is just the integral of
  the shaft rate. All the standard order-tracking machinery, and our own
  synthetic noise generator, is built on this. The next few slides test it
  against measurement.
]

= Model v1 --- admit an uncertain shaft

#v(1fr)
#grid(
  columns: (1.1fr, 1fr),
  gutter: 1.4em,
  align(horizon)[
    Admit the shaft's instantaneous phase is not known exactly, but every
    harmonic stays *locked together*.
    #v(0.5em)
    #text(size: 1.05em)[
      $ epsilon_k (t) = k dot epsilon_"shaft" (t) $
    ]
    #v(0.3em)
    #text(size: 0.88em, style: "italic")[
      One hidden scalar process. Know the phase error at one harmonic, and
      you know it at every other harmonic.
    ]
  ],
  align(center + horizon)[
    #cbox(text(size: 0.85em)[$epsilon_"shaft"$])
    #v(0.4em)
    #text(size: 1.1em)[#sym.arrow.b #h(1em) #sym.arrow.b #h(1em) #sym.arrow.b]
    #v(0.3em)
    #cbox(text(size: 0.78em)[$epsilon_1$]) #cbox(text(size: 0.78em)[$epsilon_2 = 2 epsilon_"shaft"$]) #cbox(text(size: 0.78em)[$epsilon_k = k epsilon_"shaft"$])
  ],
)
#v(1fr)

#speaker-note[
  A softer model: the shaft phase itself is an uncertain, slowly-varying
  process, but the harmonics don't decouple from each other --- if harmonic
  2 drifts, harmonic 4 drifts exactly twice as much, because they're all
  reading the same one physical shaft. This coherence claim is directly
  measurable, and that's what the next slide does.
]

= The lock measurement --- the realism ladder

#grid(
  columns: (1fr, 1.15fr),
  gutter: 1.4em,
  align(horizon)[
    #text(size: 0.85em)[
      Demodulate harmonic $k$ by the telemetry shaft phase, then take the
      resultant length of the phase-increment phasors:
    ]
    #v(0.5em)
    #text(size: 0.82em)[
      $ z_k (t) = "bp"_k [n](t) dot e^(-i k phi_"shaft" (t)) $
      $ "lock"_k = abs(chevron.l e^(i chevron.l Delta z_k (t) chevron.r) chevron.r_t) $
    ]
    #v(0.5em)
    #text(size: 0.78em)[
      1 = shaft model explains the phase; 0 = residual phase is a random
      walk.
    ]
  ],
  align(center + horizon, image("assets/lock_ladder.png", width: 100%)),
)

#v(0.4em)
#align(center, text(size: 0.95em)[
  *Running four rotors at once destroys phase coherence*, before flight,
  before per-rotor aero noise, before translation ever enters.
])

#speaker-note[
  Lock is measured by demodulating each harmonic against the telemetry
  shaft phase and taking the resultant length of the phase increments ---
  1.0 means every increment points the same way, 0 means they cancel. A
  single motor on the bench locks at 0.7-0.9 for the low harmonics. The
  moment a second, third and fourth motor start running on the same static
  bench, lock collapses to 0.02-0.13 --- staggered setpoints included. Hover
  and free flight sit at the same floor. So the collapse arrives with
  multi-rotor operation itself, not with flight and not with per-rotor aero
  noise; model v1's single hidden shaft process is refuted by this
  measurement.
]

= "Use more microphones?" --- no

#v(1fr)
#align(center, image("assets/beamform.png", height: 70%))

#align(center, text(size: 0.95em)[
  The incoherence arrives *identically* at every microphone: arrays cannot
  restore coherence that was never emitted.
])
#v(1fr)

#speaker-note[
  Natural objection: maybe one mic is just noisy, and combining channels
  recovers the coherence. The self-steered combiner here is an oracle ---
  given the best per-mic phase alignment after the fact, a bound no
  physical beamformer can beat. Even it barely improves on a single channel
  and stays far below the lock needed for the model to hold. The loss is
  common-mode at the source, not a microphone-level noise problem.
]

= The displaced comb

#v(1fr)
#align(center, text(size: 1.05em)[
  Even where strong harmonics *do* exist, they sit off the mechanical grid
  in translating flight:
])

#v(0.8em)
#align(center, text(size: 1.0em)[
  #table(
    columns: 2,
    align: (left, left),
    stroke: 0.5pt + luma(180),
    inset: 9pt,
    [$k = 2$--$13$ (low harmonics)], [0.3--0.5 rev/s *below* the mechanical rate],
    [$k gt.eq 16$], [on-grid, but 3--4#sym.times weaker],
    [hover], [low-harmonic *displacement* 3--4#sym.times weaker (needs translation)],
  )
])

#v(0.8em)
#align(center, text(size: 1.05em, style: "italic")[
  At low harmonics, acoustic truth #sym.eq.not mechanical truth.
])
#v(1fr)

#speaker-note[
  This one survived four separate checks meant to kill it: clock skew
  between audio and telemetry (refuted, ratio 1.0000 plus-minus 0.0005 on
  steady clips), an estimator bug (refuted on synthetic data with known
  ground truth), a hover contrast (the effect is 3-4x weaker in hover,
  meaning it needs translation), and a check that the bias scales
  proportionally with harmonic number the way a Doppler or calibration
  error would (it doesn't -- it spares the high harmonics, which rules that
  class of explanation out). Leading hypothesis: forward-flight loading
  modulates the strong low harmonics and skews them. The mechanism is still
  open, but the measurement itself is solid, and it means any refiner that
  is scored against raw telemetry pays this bias as if it were error.
]

= Model v2 --- what the data forces

#v(1fr)
#align(center, text(size: 1.5em)[
  $ phi_k (t) = k dot phi_"shaft" (t) + b_k (t) $
])
#v(0.6em)
#align(center, text(size: 1.05em)[
  $b_k$ = its own per-harmonic random walk, budgets measured directly:
  $tau_k approx 0.4$--$1.7$ s at $k = 8$--$40$.
])

#v(1em)
#align(center, text(size: 1.05em, style: "italic")[
  pi_kalman is the matching estimator: it sits at the measured limit, the
  full chain moves the blind baseline under 2% (1.825 #sym.arrow 1.819 on
  DREGON).
])
#v(1fr)

#speaker-note[
  What the ladder and the displaced-comb finding force: each harmonic needs
  its own drift term b_k, not one shared shaft error. We measured the
  budget directly -- coherence time tau_k of 0.4 to 1.7 seconds at
  harmonics 8 to 40. pi_kalman's per-harmonic random-walk model IS this
  model, and building the matching estimator only moved the blind baseline
  under 2% on DREGON, because the estimator was already close to the
  physics-imposed floor there. The guarded peeled alternation (flagship)
  found a real 10% gain on FLY124 by re-solving and re-peeling, but it too
  is bounded by the same displaced-comb floor on DREGON. This measured
  model is also the design brief for CKLA -- its phase-first, per-channel
  structure is the neural mirror of the same finding.
]

= Significance

#v(0.5fr)
#align(center, block(width: 65%, align(left, text(size: 1.2em)[
  - Blind per-rotor tracking: no prior art
  - Limits are measured facts, not hunches
  - Required RPS precision for SE: now known
    #text(size: 0.7em)[($tau_k approx 0.4$--$1.7$ s, previous slide)]
])))
#v(0.5fr)

#speaker-note[
  Three claims, checked against this week's adversarial literature crawl.
  First: no prior method recovers four per-rotor rate trajectories blind
  from audio when the combs overlap within a filter bandwidth -- nearest
  prior art is a 2012 supervised helicopter neural network (labelled, one
  rotor), aggregate blade-pass-frequency estimators (no per-rotor split),
  and a 2026 tacholess method for a single shaft -- none handle four
  interleaved rotors blind. Second: everything on the last several slides
  was a measurement, not a guess -- the lock ladder, the beamform bound,
  the displaced comb. Third, and this sets up the next question: we now
  know how precise RPS needs to be for speech enhancement to use it, which
  is what the rest of this deck turns to.
]

= The paper (WIP)

#v(1fr)
#grid(
  columns: (1fr, 1fr),
  gutter: 1.2em,
  text(size: 0.85em)[
    *Methods*
    - VK chain (this deck)
    - Neural trackers (CKLA family)
    - Hybrid (neural init + pi_kalman)

    *Protocol*
    - Synthetic: S1 single rotor, S2 twins @ 0 dB, S3 measured phase noise
      @ #sym.minus\5 dB
    - Real: the frozen 15-window protocol
  ],
  text(size: 0.85em)[
    *Baselines*
    - IAVKF, VKF+SWT
    - Tacholess order tracking (iterated warp)
    - Multi-pitch salience (multif0, Basic Pitch)
    - Ablations

    *Status*
    #text(fill: rgb("#d62728"), weight: "bold")[Draft WIP --- baselines +
    figures are the remaining work.]
  ],
)
#v(1fr)

#speaker-note[
  The plan is the baseline list, run against every proposed method on the
  same frozen protocol. The current 8-page draft covers the VK side in
  full; the neural section extends it into this combined paper. Venue input
  is wanted -- that's on the asks slide.
]

= Next 4 weeks

#v(1fr)
#align(center, text(size: 0.92em)[
  #table(
    columns: 2,
    align: (left, left),
    stroke: 0.5pt + luma(180),
    inset: 7pt,
    table.header([*week*], [*paper*]),
    [1], [baselines + figures],
    [2], [neural section + full results],
    [3], [polish + internal review],
    [4], [buffer + #text(weight: "bold")[SUBMIT]],
  )
])

#v(0.4em)
#align(center, text(size: 0.9em, style: "italic")[
  Parallel: RPS-conditioned SE starts now --- oracle RPS into MP-SENet /
  Edge-BS-RoFormer via comb subtraction, harmonic-aligned bands, FiLM.
  First read end of August.
])
#v(1fr)

#speaker-note[
  Four weeks to submission, one lane at a time on the paper. In parallel,
  the SE foundation was fixed this week: a mixing bug that zeroed about 6%
  of samples was found and fixed, valid sets were rebuilt and verified
  clean, the TF-GridNet "failure" turned out to be compute starvation not a
  model problem, and all four baselines are retraining now on clean curves
  with a new NaN guard. RPS-conditioned SE is the core of the next paper,
  and this deck's ceiling numbers are exactly the input budget that
  conditioning needs to work with.
]

= Asks

#v(1fr)
#align(center, block(width: 70%, align(left, text(size: 1.15em)[
  - Venue for paper 1?
  - Reaction to the ceiling argument: it becomes the thesis backbone.
  - Agree: RPS-conditioned SE is the next-paper question?
])))
#v(1fr)

#speaker-note[
  Three questions, one sentence each: where should this go; does the
  ceiling argument -- measured limits driving model choice -- read as
  strongly to you as it does to me; and do you agree the next paper is
  conditioning speech enhancement on RPS, now that we know the precision
  budget it has to work within.
]

=

#align(center + horizon)[
  #text(size: 1.3em, weight: "bold")[Backup]
]

#speaker-note[
  Divider. Everything after this is backup material for questions: the
  four displaced-comb verification probes in more detail, and the
  per-window scoreboard behind the headline table.
]

= Backup --- the four displaced-comb probes

#v(1fr)
#align(center, text(size: 0.95em)[
  #table(
    columns: (auto, 1fr),
    align: (left, left),
    stroke: 0.5pt + luma(180),
    inset: 7pt,
    table.header([*probe*], [*result*]),
    [Per-$k$ structure], [low $k$ displaced, high $k$ on-grid --- not uniform],
    [Clock skew], [audio/telemetry rate ratio 1.0000 #sym.plus.minus 0.0005 --- refuted],
    [Estimator], [exact on synthetic data with a known comb --- exonerated],
    [Hover control], [3--4#sym.times weaker without translation],
  )
])

#v(0.6em)
#align(center, text(size: 0.9em, style: "italic")[
  Leading hypothesis: forward-flight aerodynamic loading modulates the
  strong low harmonics. Mechanism still open.
])
#v(1fr)

#speaker-note[
  Backup detail for the displaced-comb slide. A proportional mechanism
  (Doppler shift, a sample-rate calibration error) would displace every
  harmonic by the same fraction of its frequency --- it does not explain
  why k >= 16 sits exactly on-grid while k = 2-13 sits 0.3-0.5 rev/s low.
  [TODO verify: a plotted per-k offset figure for the paper --- this deck
  shows the qualitative table only.]
]

= Backup --- the full fixed-protocol scoreboard

#v(1fr)
#align(center, text(size: 0.78em)[
  #table(
    columns: (1.3fr, auto, auto, auto, auto),
    align: (left, center, center, center, center),
    stroke: 0.5pt + luma(180),
    inset: 7pt,
    table.header([*pool*], [*blind VK, baseline*], [*blind VK, gated*], [*cross-window seeded*], [*n windows*]),
    [DREGON cruise], [1.825], [*1.819*], [--], [9],
    [FLY124 cruise], [2.418], [*2.380*], [*1.864*], [4],
    [FLY124 warm-up], [3.644], [3.605], [--], [2],
    [all 15 windows], [2.226], [*2.207*], [--], [15],
  )
])

#v(0.6em)
#align(center, text(size: 0.85em)[
  Cross-window seeding needs three assumptions (same recording, cruise
  regime, stable rotor identity across windows) and only fires when the
  blind seeder finds all four rotors: off by default.
])
#v(1fr)

#speaker-note[
  The gate (which declines a per-rotor proposal unless it passes three
  truth-free checks) turns a 0.26/0.09 rev/s regression from an earlier
  seeding bug into a tie-or-small-win against the ungated baseline, on both
  pools. The one arm that beats the baseline by a real margin is
  cross-window-seeded refinement on FLY124 cruise (-23%), but it needs a
  successful 4-rotor blind seed on a neighbouring window of the same
  recording and is off by default for that reason.
]
