#import "@preview/touying:0.7.4": *
#import "/writing/templates/typst/slides.typ": hns-slides

// Notes-visible build: `make notes` (typst --input notes=1). Default build is clean.
#let notes-mode = sys.inputs.at("notes", default: none)

#show: hns-slides.with(
  title: [Annotation is the bottleneck],
  subtitle: [Measuring, then optimizing, the quality of rotor-speed labels],
  author: [Dmitrii Mukhutdinov],
  date: [2026-08-11],
  show-notes: if notes-mode != none { bottom } else { none },
)

#let cbox(body, fill: luma(235), stroke-color: luma(160)) = box(
  fill: fill,
  stroke: 0.7pt + stroke-color,
  inset: 8pt,
  radius: 3pt,
  body,
)

= Why blind annotation

#grid(
  columns: (1.05fr, 1fr),
  gutter: 1.2em,
  align(horizon)[
    Neural RPS predictors and noise generators are limited by *labeled real
    audio*: rotor-speed trajectories aligned to the recording.
    #v(0.4em)
    A precise *blind* annotator — even a slow one — turns every unlabeled
    drone recording into training data.
  ],
  align(horizon)[
    #text(size: 0.8em)[
      #table(
        columns: 3,
        align: (left, right, left),
        stroke: 0.5pt + luma(200),
        inset: 5pt,
        [*dataset*], [*size*], [*RPS labels*],
        [DREGON], [2.8 GiB], [telemetry (biased)],
        [Michael's FLY124/125], [6.6 GiB], [telemetry (recalibrated)],
        [AVQ ego-noise], [0.5 GiB], [VK pseudo-labels only],
        [DroneAudioSet], [88.4 GiB], [*none*],
        [AeroSonicDB], [4.1 GiB], [*none*],
        [HornBase + HUSTmotor], [0.4 GiB], [*none*],
      )
    ]
    #v(0.3em)
    #text(size: 0.78em, style: "italic")[
      #sym.approx 93 GiB of drone audio has no rotor-speed labels at all.
    ]
  ],
)

#speaker-note[
  The bottleneck is not model capacity, it is labeled real audio. Everything
  below 3 rows of this table is unusable for conditioning or supervision
  today. A blind annotator that is precise enough converts the entire tail.
]

= A constant label bias collapses the generator's high harmonics

#align(center, image("assets/gen_label_bias.png", width: 82%))

#align(center, text(size: 0.85em)[
  One rotor, synthetic comb, exact truth known. Only the conditioning label
  changes between arms.
])

#speaker-note[
  This is an A/B on synthetic data where the truth is known exactly, so only
  the label is corrupted. Train the generator on exact labels and the learned
  line power is flat to a fifth of a dB out to harmonic 80. Add the
  tachometer staircase alone and nothing much happens: minus 0.3 dB in the
  top band. Add the constant 0.54 percent scale error that DREGON's telemetry
  carries, and the top band falls by 8.6 dB. The reason is arithmetic: a rate
  error of 0.54 percent displaces harmonic 80 by a full analysis bin, so the
  objective's best answer is a smeared, attenuated line. The generator was
  never the problem. The label was.
]

= The prominent harmonic does not sit on the label

#align(center, image("assets/spec_h70_telemetry.png", height: 72%))

#text(size: 0.85em)[
  DREGON, harmonic 70 of rotor 0, band 5.7--6.4 kHz. The dashed line is
  70 #sym.times the telemetry rate. The line in the audio runs *next to* it.
  So: how precise do labels have to be, and how do we even measure that?
]

#speaker-note[
  This is the raw observation that started the whole question. At the 70th
  harmonic a fraction of a percent of rate error is tens of hertz, and the
  telemetry visibly misses the acoustic line. Whether this matters for
  training, and by how much, needs a measurement, not an eyeball.
]

= One rig's telemetry sits on the comb, the other's does not

#grid(
  columns: (1fr, 1.15fr),
  gutter: 1.2em,
  align(horizon)[
    #text(size: 0.85em)[
      *The lock meter.* Demodulate the audio at the carrier the labels imply
      (harmonic $k$ #sym.times rotor rate). If the labels are correct, a line
      stands at DC of the demodulated spectrum $E_k$:
    ]
    #v(0.3em)
    $
      "ridge" = 10 log_10 frac(angle.l abs(E_k (f))^2 angle.r_(abs(f) <= w), "med"_(w < abs(f) <= W) abs(E_k)^2 \/ ln 2)
    $
    #v(0.3em)
    #text(size: 0.78em)[
      Numerator: mean power in the line window ($w$ = 0.1 rev/s around the
      carrier). Denominator: the local noise floor — median power at nearby
      frequencies just outside the window; the $ln 2$ calibrates pure noise to
      exactly 0 dB. Mean over rotors #sym.times mics #sym.times harmonics
      #sym.times blocks. Hatched bars: the same reading on an off-comb carrier
      (the null).
    ]
    #v(0.4em)
    #text(size: 0.85em)[
      FLY124 clears its null by 3.5 dB. DREGON reads its null.
    ]
  ],
  align(center + horizon, image("assets/ridge_telemetry.png", width: 100%)),
)

#speaker-note[
  The third symptom, and the one that names the cause. This is a lock reading:
  how far a spectral line stands above the local noise floor after the audio is
  demodulated at the carrier the labels imply. It is calibrated so that pure
  noise reads zero. Michael's drone, after its telemetry was recalibrated
  earlier this summer, reads plus 2.72 dB and stands 3.5 dB clear of both its
  nulls. DREGON's raw telemetry reads minus 0.60 against its own off-comb null
  of minus 0.66, which is no separation at all. The comb is there in the audio;
  the labels are not on it. Everything trained on DREGON labels has been
  trained on a carrier that misses the lines.
]

= Roughly right is not enough

#v(1fr)
#align(center, cbox(text(size: 1.1em)[
  A constant 0.54 % label bias alone costs *−8.6 dB* on high harmonics
  (exact labels train flat). The telemetry itself is off by 0.35--0.85 %.
  #v(0.5em)
  *So we need labels refined beyond telemetry precision — and a measure that
  says when they are right.*
]))
#v(1fr)

#speaker-note[
  The generator experiment and the lock meter close the loop: label error at
  the level real telemetry actually has is enough to destroy the training
  signal on the harmonics that matter. Roughly right does not exist here.
]

= Refinement puts the label back on the line

#grid(
  columns: (1.15fr, 1fr),
  gutter: 1.0em,
  align(center + horizon, image("assets/spec_h70_refined.png", width: 100%)),
  align(center + horizon)[
    #image("assets/ridge_candidates.png", width: 100%)
    #text(size: 0.8em)[
      The fitted trajectory locks *+2.47 dB* over DREGON telemetry; on FLY124
      (recalibrated) it has almost nothing left to fix (+0.26 dB).
    ]
  ],
)

#speaker-note[
  Left: the same window and harmonic as before, telemetry dashed, the refined
  trajectory solid — the refined line sits on the acoustic line. Right: the
  lock meter agrees across all 15 frozen windows and both rigs.
]

= The measure, and what we do with it

#text(size: 0.9em)[
  Assume the candidate trajectories are true, solve the coupled Vold--Kalman
  envelopes under them, and score the *profiled residual*:
]
$
  F(r) = min_a norm(y - sum_m "Re"[a_m c_m (r)])^2 + sum_m rho_m^2 norm(Delta^2 a_m)^2, quad c_m (t) = e^(j 2 pi k_m integral r)
$
#text(size: 0.9em)[
  All rotors and harmonics compete for the same energy (twin-aware), and $F$
  is *differentiable in the trajectory* — the trajectory can literally be
  moved around by gradient descent. Three things to do with it:
  + *L-BFGS on $F$* from any starting point — slow, precise, the oracle.
  + *Ours: blind Viterbi seed #sym.arrow peel #sym.arrow.l.r pi-Kalman* — fast, how close to the oracle?
  + *Baselines*: IAVKF-style adaptive VK, ridge/DP tracking.
]

#speaker-note[
  One scalar answers "how well do these trajectories fit this audio". It is
  the VK solve we already trust, profiled over the envelopes, so its gradient
  with respect to the trajectory is exact and cheap. The rest of the deck is
  those three contestants.
]

= Our seed: comb-matched Viterbi over the whole rate range

#grid(
  columns: (0.85fr, 1.15fr),
  gutter: 1.2em,
  align(horizon)[
    Whitened multi-channel spectrogram, then a full-range search whose emission
    is a comb score and whose transition penalty keeps the path smooth.
    #v(0.5em)
    #text(size: 0.78em)[
      $ J[c] = -sum_t "score"(c(t) mid(|) y(t)) + lambda sum_t (c(t) - c(t-1))^2 $
    ]
    #v(0.5em)
    #text(size: 0.8em, style: "italic")[
      Summing $K$ harmonics buys about $10 log_10 K$ dB, which is what puts a
      $-20$ dB comb above the search's own noise-only path.
    ]
  ],
  align(center, image("assets/stepper_coarse_init.png", width: 100%)),
)

#speaker-note[
  The blind chain needs a seed inside the basin, and this is it. Whiten the
  multichannel spectrogram, then run a Viterbi search over the whole plausible
  rate range whose emission score is how well a comb at that rate matches the
  spectrum. The transition penalty keeps the path smooth. The reason this works
  at very low signal-to-noise is the harmonic sum: coherently adding K
  harmonics buys roughly ten log ten K decibels, about 16 dB at forty
  harmonics, and the dynamic-programming literature says the best noise-only
  path grows only with the square root of the log of the branching factor.
  Passive sonar tracks four simultaneous lines at minus twenty decibels on
  exactly this arithmetic. The output is one coarse global track, before
  anything is assigned to a rotor.
]

= Our algorithm, step 2: ramp handling

#grid(
  columns: (0.95fr, 1.05fr),
  gutter: 1.2em,
  align(horizon)[
    Warm-up spectra carry only blade-pass lines, so the seeder used to promote
    onto the wrong octave. Fix: a line-ratio check, plus an energy-timed bridge
    across spool-up.
    #v(0.5em)
    #align(left, text(size: 0.85em, style: "italic")[
      Ramp and warm-up windows went *15--36 #sym.arrow 2.9--4.0* MAE.
    ])
  ],
  align(center, image("assets/stepper_viterbi_c.png", width: 100%)),
)

#speaker-note[
  Take-off and landing are where the blind chain used to fail outright. During
  spool-up the spectrum shows only blade-pass lines, and the old seeder read
  that as a lower rate one octave down. The fix checks the line ratio to pick
  the octave and bridges the ramp using the timing of the energy rise. It took
  the worst windows from 15 to 36 rev per second of error down to 2.9 to 4.0.
]

= Our algorithm, step 3: per-rotor decoupling

#grid(
  columns: (0.95fr, 1.05fr),
  gutter: 1.2em,
  align(horizon)[
    - Split a corridor around the global track per candidate rotor rate.
    - A gated residual re-scan recovers rotors invisible in the global comb.
    - Twin pairs 0.5 to 0.65 rev/s apart: two candidates can tie to within
      0.06% of score.
    #v(0.4em)
    #text(size: 0.82em, style: "italic")[
      A seeding coin-flip on one tied pair cost 1.57 rev/s: fixed by two guards
      on the re-scan, not by a better search.
    ]
  ],
  align(center, image("assets/stepper_vit2dsp.png", width: 100%)),
)

#speaker-note[
  Once the global track and the ramp are handled, each rotor needs its own
  seed. The corridor split proposes candidates around the global track, and a
  residual re-scan looks for a rotor the global comb missed entirely. The hard
  failure mode is twin rotors half a rev per second apart, whose two candidates
  score within 0.06 percent of each other, so the ranking is a coin flip. Two
  guards fixed it: a maximum span on the accepted set, and mutual
  de-duplication of the accepted candidates.
]

= Our algorithm, step 4: envelope solve, then peel

#grid(
  columns: (0.95fr, 1.05fr),
  gutter: 1.2em,
  align(horizon)[
    All rotors solved jointly (envelopes at the current tracks, bandwidth about
    1 Hz), then each rotor's comb is reconstructed and the OTHER rotors' combs
    are subtracted.
    #v(0.4em)
    #text(size: 0.78em)[
      $ J[a] = sum_t |y(t) - sum_m "Re"[a_m (t) c_m (t)]|^2 + sum_m rho_m^2 norm(Delta^p a_m)^2 $
    ]
  ],
  align(center, image("assets/stepper_refine.png", width: 100%)),
)

#speaker-note[
  The joint solve and the peel. The solve is joint across all four rotors
  because overlapping combs share spectral bins. The peel then removes the
  other rotors from each rotor's audio. The guard checks that the subtraction
  actually reduced residual energy; on ramp windows the tracks are not locked
  yet, the peel is mis-phased, and the guard keeps the seed instead. This same
  cost, with the envelopes profiled out, is the objective the new plan
  optimizes over the trajectory.
]

= Peel, then re-read the phase, and iterate to plateau

#grid(
  columns: (0.95fr, 1.05fr),
  gutter: 1.2em,
  align(horizon)[
    Solve all rotors jointly, reconstruct each rotor's comb, subtract the OTHER
    rotors' combs, then re-read the phase increment on the peeled residual.
    #v(0.4em)
    #text(size: 0.78em)[
      $ Delta psi_k = arg(z_k (t) macron(z)_k (t-1)) $
    ]
    #v(0.4em)
    #text(size: 0.8em)[
      2 to 4 applications reach a plateau; naive re-application stalls higher.
    ]
    #v(0.3em)
    #text(size: 0.8em, style: "italic")[
      A guard falls back to the seed when the subtraction fails to remove
      energy.
    ]
  ],
  align(center, image("assets/stepper_pi_kalman.png", width: 100%)),
)

#speaker-note[
  The refinement half, and it is the thing the fitness campaign scored at plus
  1.88 dB on DREGON. Solve the Vold-Kalman envelopes jointly across all four
  rotors, because overlapping combs share spectral bins. Then use each rotor's
  own reconstruction to peel the other three out of the audio, so the phase
  refiner sees one rotor's signal with its neighbours subtracted rather than
  mixed in. Re-reading the phase increment naively degrades, because the
  estimator re-reads its own noise; peeled re-reading converges, because each
  pass sees cleaner data. The chart is the real trace on one of Michael's
  cruise windows. This is the ancestor of the optimizer on the previous
  slides: alternation with a hand-built update, where the new plan does
  gradient descent on the same residual.
]

= Step 4: only gradient descent on $F_"VK"$ improves a telemetry init

#v(0.4fr)
#align(center, image("assets/step4_arms.png", width: 100%))

#v(0.5em)
#align(center, text(size: 0.82em)[
  Synthetic, DREGON-style corrupted init, rms rate error against truth: init
  0.598 #sym.arrow.r peel#sym.harpoons.rtlb pi-Kalman 0.474, IAVKF-style 0.546,
  *L-BFGS on $F_"VK"$ 0.182* at 0 dB. At $-10$ dB only L-BFGS moves at all
  (0.188), at 8 to 10 times the cost of the pi-Kalman arm. On real DREGON audio
  the objective goes 0.609 #sym.arrow.r 0.581 / 0.605 / *0.467*.
])
#v(0.6fr)

#speaker-note[
  Step four asks the question that matters for pseudo-labels: given a corrupted
  label set, does anything actually recover the truth. Synthetic first, because
  it is the only setting where the answer is a measurement rather than an
  inference: take a known trajectory, corrupt it the way DREGON's telemetry
  corrupts one, and let each refiner work. Our own alternation chain takes 0.598
  down to 0.474, an adaptive-Vold-Kalman-style arm reaches 0.546, and gradient
  descent on the profiled residual reaches 0.182, which is a factor of three.
  At minus ten decibels the two hand-built arms stop moving altogether while
  L-BFGS holds its number, and that is the interesting result: the alternation
  is capture-range limited exactly as the basin law predicts. The price is eight
  to ten times the compute. On real DREGON audio there is no truth, so the two
  right-hand panels read the objective instead, and the ordering is the same.
]

= Step 5: blind annotation, precision against compute

#v(0.3fr)
#align(center, image("assets/step5_pareto.png", width: 100%))

#v(0.4em)
#align(center, text(size: 0.8em)[
  The full blind ladder is the most precise contestant everywhere a reference
  exists: DREGON 1.311 against 1.366 rms for the seed alone, FLY124 3.01
  against 3.32, at 8 to 165 s per window. Multi-start L-BFGS costs 12 to 25
  times more and is 2 to 6 times worse.
])
#v(0.3em)
#align(center, cbox(fill: rgb("#fdece8"), stroke-color: rgb("#d62728"))[
  #text(size: 0.8em)[
    On DREGON multi-start *wins the objective* (0.633 against 0.765) while
    sitting 8 rev/s away from telemetry: the alias degeneracy, on real data. The
    order penalty is the missing term.
  ]
])
#v(0.5fr)

#speaker-note[
  The deployable question, and it is a front rather than a point. The blind
  ladder, with no telemetry at any stage, is the most precise contestant on
  every material where a reference exists, and it runs in seconds to a couple of
  minutes per window. Each rung of the ladder buys a little: the seed alone,
  the seed with one Vold-Kalman refinement, then the full alternation. Multi-start
  L-BFGS is the expensive anchor, and it is the slide's warning. On DREGON it
  reaches the best objective of any arm and sits eight rev per second away from
  the telemetry, which is more than the twin-rotor spacing. That is the nested
  sub-multiple from part three, demonstrated on real audio rather than derived:
  a better objective value at a wrong trajectory. Until the order penalty is in
  the cost, an unconstrained search on this objective is free to find the alias,
  and only the seeded ladder is safe.
]

// ═══ PART 5: the payoff ═══════════════════════════════════════════════════

= Blind fit against ground truth: DREGON cruise

#align(center, image("assets/compare_dregon.png", height: 76%))

#align(center, text(size: 0.85em)[
  Dotted is raw telemetry, solid is the tracker, permutation-aligned.
])

#speaker-note[
  One live window, two trackers, scored against raw telemetry. The classical
  chain tracks the fast fluctuations tightly. The neural tracker's residual
  here is the anchor-collapse pattern, one rotor sitting off by roughly the
  pair spacing. Read this plot remembering the caveat on the scoreboard slide:
  the dotted reference on DREGON is the label set that has no acoustic lock.
]

= Blind fit against ground truth: FLY124 cruise

#align(center, image("assets/compare_fly124.png", height: 70%))

#align(center, text(size: 0.85em)[
  Dotted is recalibrated telemetry, which the fitness instrument confirms sits
  on the comb.
])

#speaker-note[
  The same comparison on Michael's drone, where the reference is trustworthy:
  its telemetry reads plus 2.72 dB of lock. On this window the two families are
  close, and the neural tracker needs no telemetry seed to get there. Pooled
  over all four cruise windows the neural tracker is ahead, because the blind
  chain has two much worse windows that this one does not show.
]
= The paper

#text(size: 0.9em)[
  #table(
    columns: 3,
    align: (left, left, left),
    stroke: 0.5pt + luma(200),
    inset: 6pt,
    [*contribution*], [*status*], [*evidence*],
    [1. A measure of trajectory--audio fit], [done], [lock meter + $F_"VK"$; landscape benchmark],
    [2. A precise blind-annotation algorithm],
    [ours is fastest and closest,
      *not yet precise enough* (#sym.approx 1.3 rev/s blind vs 0.2 target)],
    [Pareto: ours 8--165 s/window; L-BFGS 12--25#sym.times cost],

    [3. Pseudo-labels for all unannotated drone audio], [unlocked by 1+2], [93 GiB unlabeled],
  )
]

#speaker-note[
  The contribution order is the narrative order: the measure first, with the
  characteristics it must have; the algorithm second, judged by that measure —
  ours is not precise enough yet and the gap is quantified; the pseudo-label
  program third, which is why the whole thing matters.
]

= Backup --- the fixed 15-window tracking scoreboard

#v(0.4fr)
#text(size: 0.95em)[
  #table(
    columns: (1.5fr, auto, auto),
    align: (left, center, center),
    stroke: 0.5pt + luma(180),
    inset: 6pt,
    table.header([*tracker*], [*DREGON cruise*], [*FLY124 cruise*]),
    [blind VK, full-range init], [*1.807*], [2.515],
    [guarded peeled alternation], [1.841], [2.274],
    [VK, telemetry-init (oracle)], [0.85], [0.78 #super[#sym.ast]],
    [CKLA, phase-only, 4 s context], [2.55], [1.29],
    [KLA (plain, no rotation)], [3.13], [1.81],
    [transformer], [3.38], [3.47],
    [uni_gru128], [4.25], [2.26],
  )
]

#v(0.6em)
#text(size: 0.85em)[
  PIT-MAE (rev/s), permutation-matched to raw telemetry, 15 frozen 16 s
  windows. #super[#sym.ast] predates the FLY124 telemetry recalibration.
  Note the metric's own limit: on DREGON it scores against labels that the
  fitness instrument says are off the comb.
]
#v(0.4fr)

#speaker-note[
  The tracking scoreboard from the previous deck, kept for reference. Two
  readings. First, neither family dominates: the classical chain wins DREGON,
  the neural family wins Michael's drone. Second, and this is why the metric
  moved to the back of the deck, every cell scores a tracker against raw
  telemetry, and on DREGON the telemetry is the thing that has no acoustic
  lock. A tracker that found the true rotor speeds on DREGON would be
  penalised by this table. That is the measurement problem the fitness
  instrument was built to escape.
]

= Backup --- the measurement procedure, in full

#grid(
  columns: (1.25fr, 1fr),
  gutter: 1.0em,
  align(center + horizon, image("assets/ridge_instrument.png", width: 100%)),
  align(horizon)[
    #text(size: 0.85em)[
      Every ridge number on the next slides is this procedure:
      + Take one candidate set of rotor-rate trajectories.
      + For every cell (rotor #sym.times mic #sym.times harmonic $k$
        #sym.times time block): demodulate the audio at $k$
        #sym.times the candidate rate.
      + Read the lock meter of slide 4 on that cell: line power against the
        local floor, dB, 0 = no line there.
      + Average the cells. One number per candidate.
    ]
    #v(0.4em)
    #text(size: 0.82em)[
      #table(
        columns: 2,
        align: (left, left),
        stroke: 0.5pt + luma(200),
        inset: 5pt,
        [fixed], [band, harmonic set, blocks, mask — all pinned to the reference, identical for every candidate],
        [nulls], [off-comb (half-integer), mismatched window, permuted rotors],
        [hold-outs], [even/odd $k$, mic 0, half the blocks],
        [calibrated], [pure noise reads 0 dB],
      )
    ]
  ],
)

#speaker-note[
  How the reading works, and why it cannot be gamed. Demodulate one microphone,
  one harmonic, one time block at the carrier the candidate implies. If the
  candidate is right, the energy piles up in a narrow band around zero; if the
  carrier is off by more than the band width, which is 0.1 rev per second, the
  line leaves the band and the reading falls to the noise value. The floor
  comes from an annulus of the same spectrum, with the median-to-mean
  correction, so pure noise reads zero rather than plus 1.6 dB. Everything
  except the carrier is pinned to the window's reference trajectory, so a
  flexible candidate cannot buy itself an easier cell set. A test asserts the
  cell count is identical across every candidate and every control. And every
  reading carries three nulls plus four hold-out families, which is what makes
  it a verdict instrument rather than a score.
]

= Backup --- the fitted trajectory locks 2.47 dB better

#align(center, image("assets/ridge_candidates.png", width: 82%))

#align(center, text(size: 0.85em)[
  DREGON hold-out families for the fitted trajectory: 1.38, 2.35, 1.97, 1.78 dB,
  all clearing both nulls.
])

#speaker-note[
  Now score candidate annotations on the same cells. DREGON's raw telemetry
  sits at its null. One constant scale factor lifts it to plus 0.14, which just
  clears the null. A freely fitted trajectory, from the refinement chain
  described later, reaches plus 1.88, which is 2.47 dB above the labels. The
  negative control is the point of the slide: the identical procedure on
  Michael's already-correct labels buys 0.26 dB, nine times less. A fitter that
  was simply flexible would have bought a similar amount on both. It did not,
  so the DREGON gain is a real correction rather than overfitting. Coverage,
  honestly stated: the gate sees 5.7 percent of DREGON's comb line energy
  against 18.6 percent of Michael's, because four rotors at high harmonic
  numbers are genuinely unresolvable.
]

= Backup --- a single constant explains only a third of the error

#align(center, image("assets/scale_profile.png", width: 82%))

#align(center, text(size: 0.85em)[
  Hold-out families agree to 0.06 pp. The best constant buys 0.74 dB; the free
  trajectory buys 2.47 dB.
])

#speaker-note[
  This is the one-parameter version of the same question: slide a constant rate
  scale across the labels and read the lock. The profile has a single free
  parameter, so its extremum cannot be bought with flexibility, and every
  hold-out family is scored on cells the fit never chose. The extremum is minus
  0.683 percent with a confidence interval from minus 0.88 to minus 0.53, and
  the four hold-out families land within 0.06 of a percentage point of each
  other. The off-comb null has eight times less depth and no interior extremum.
  Michael's recalibrated labels return zero with a basin four times deeper,
  which is what a correct label set should look like. And the last number is
  the interesting one: the best constant recovers only about a third of what
  the free trajectory recovers, so the label error is not a scale error.
]

// ═══ PART 3: from detector to objective ═══════════════════════════════════

= Backup --- the landscape, measured

#align(center, image("assets/bench_gra.png", height: 68%))

#align(center, text(size: 0.82em)[
  $F_"VK"$ at the coarse rung is the only measure with a positive optimum margin
  over the structured alias set. It stays monotone out to 2.83 rev/s, and reads
  0.87 to 0.93 at $-20$ dB, where ridge, broadband and harmonic sum are at chance.
])
#v(0.3em)
#align(center, text(size: 0.82em, style: "italic")[
  And the nested sub-multiple ($f_0 \/ 2$, $2K$ harmonics) beats every measure in
  100% of units: the degeneracy of the previous slide, now measured. A fixed
  harmonic cap only hid it.
])

#speaker-note[
  Step one of the matrix, run before the optimizer was trusted with anything.
  Six candidate measures, scored on synthetic combs with known truth and on real
  cruise audio, at three signal-to-noise levels. Three readings matter. First,
  the optimum margin: only the coupled residual at the coarse harmonic rung puts
  the true trajectory below every member of the structured alias set, and every
  other measure is negative, which means some alias beats truth. Second, the
  basin: the coarse rung stays monotone out to 2.8 rev per second, roughly twenty
  times the fine rung, which is what makes the harmonic anneal a real ladder
  rather than a story. Third, the gradient-sign accuracy on this figure: at minus
  twenty decibels the coarse rung still ranks a five percent worse step correctly
  most of the time, while the ridge, the broadband share and the harmonic sum sit
  on the coin-flip line. The last line is the honest one. The nested sub-multiple
  beats the truth for every measure in every single unit, exactly as the algebra
  said it must. Earlier campaigns did not see it only because they capped the
  harmonic count. So the order penalty is not a refinement, it is load-bearing.
]

= Backup --- $F_"VK"$, the full definition

#v(0.7fr)
#align(center, text(size: 1.25em)[
  $ F(phi) = y^H y - y^H C(phi) [C(phi)^H C(phi) + S^T R^T R S]^(-1) C(phi)^H y $
])

#v(1.0em)
#grid(
  columns: (1fr, 1fr),
  gutter: 1.2em,
  text(size: 1.1em)[
    - Envelopes solved in closed form, then substituted back
    - Gradient in closed form or by autograd through the solve
    - Off-diagonal blocks make near-equal rotors compete for energy
  ],
  text(size: 1.1em)[
    - $rho$ maps to a bandwidth in Hz, so capture range is a knob with units
    - Annealing $K$ is graduated non-convexity with a known basin at each level
    - Without the smoothness term it is the maximum-likelihood projector
  ],
)
#v(0.7fr)

#speaker-note[
  The Vold-Kalman cost is quadratic in the envelopes once the trajectory is
  fixed, so the envelopes can be solved and substituted back, leaving a
  function of the trajectory alone. Four properties earn it the job. It is
  smooth in the trajectory with a closed-form gradient by the envelope theorem,
  and our solver is already torch-only, so autograd is nearly free. The
  off-diagonal coupling blocks are the only published mechanism that makes two
  nearly equal rotors compete for the same energy, which is exactly the twin
  problem. The regularization weight maps analytically to a bandwidth in hertz,
  so the capture range is set in physical units rather than tuned. And
  annealing the harmonic count is graduated non-convexity where, uniquely, the
  basin width is known in closed form at every level: start around five
  harmonics, where the basin is about 0.2 rev per second and swallows DREGON's
  label error, and ladder up to eighty.
]

= Backup --- the oracle moves DREGON, and leaves FLY124 alone

#align(center, image("assets/oracle_sanity.png", height: 66%))

#align(center, text(size: 0.82em)[
  Six DREGON steady windows move by a scale of #sym.minus\0.596%
  (per-window range #sym.minus\0.665 to #sym.minus\0.540), all inside the ridge
  interval [#sym.minus\0.877, #sym.minus\0.533]. FLY124 cruise moves
  #sym.minus\0.0007%, which is 0.16 rev/s rms.
])
#v(0.25em)
#align(center, text(size: 0.78em, style: "italic")[
  Outside cruise the descent breaks: on a ramping trajectory the fixed
  smoothness prior swamps the data term. Cruise only, for now.
])

#speaker-note[
  The sanity condition, and it passes. The optimizer starts from each rig's own
  telemetry, runs L-BFGS on the profiled residual under the harmonic anneal, and
  the plot reads the constant-scale component of the movement. DREGON's six
  steady windows all land near minus 0.6 percent, and every one of them sits
  inside the interval the ridge scale profile published before this optimizer
  existed. Michael's cruise windows move by seven ten-thousandths of a percent,
  which is 0.16 rev per second of jitter and no scale at all. Two instruments
  that share no code and no statistic now agree on the same label error, and the
  negative control stays flat. That is the difference between a correction and a
  flexible fitter. The caveat is on the slide and it is a real limit: the hollow
  markers are ramp and warm-up windows, where a fixed smoothness prior fights a
  trajectory that is genuinely moving, and the descent walks off. Cruise only
  until the prior is scheduled.
]

= Backup --- phase coherence collapses with more than one motor

#align(center, image("assets/lock_ladder.png", width: 82%))

#align(center, text(size: 0.85em)[
  Measured on the bench, before flight and before aerodynamic noise.
])

#speaker-note[
  A measurement that constrains how far coherent integration over harmonics can
  go. Demodulate a harmonic against the telemetry shaft phase and take the
  resultant length of the phase increments: one means every increment points
  the same way. A single motor on the bench reaches 0.7 to 0.9 at low
  harmonics. As soon as four motors run on the same static bench, that collapses
  to between 0.02 and 0.13, and hover and free flight sit at the same floor. So
  each harmonic needs its own drift term, and fully coherent summation over
  eighty harmonics is not available: the high harmonics must enter with
  measured weights.
]

= Backup --- more microphones do not restore coherence

#align(center, image("assets/beamform.png", width: 66%))

#align(center, text(size: 0.85em)[
  The self-steered combiner is an oracle bound, and it barely beats one
  microphone.
])

#speaker-note[
  The natural objection is that one microphone is simply noisy and combining
  channels recovers the coherence. The self-steered combiner here is an oracle:
  the best per-microphone phase alignment chosen after the fact, a bound no
  physical beamformer can beat. Even it barely improves on a single channel.
  The loss is common-mode at the source, so arrays cannot restore coherence
  that was never emitted.
]

= Backup --- the displaced comb

#v(0.6fr)
#align(center, text(size: 1.15em)[
  #table(
    columns: 2,
    align: (left, left),
    stroke: 0.5pt + luma(180),
    inset: 9pt,
    [$k = 2$--$13$], [0.3--0.5 rev/s *below* the mechanical rate],
    [$k gt.eq 16$], [on-grid, but 3--4#sym.times weaker],
    [hover], [displacement 3--4#sym.times weaker, so it needs translation],
    [clock skew], [audio/telemetry ratio 1.0000 #sym.plus.minus 0.0005, refuted],
    [estimator], [exact on synthetic data with a known comb, exonerated],
  )
])
#v(0.6fr)

#speaker-note[
  A second reason acoustic truth and mechanical truth differ, independent of
  the label error. In translating flight the strong low harmonics sit below the
  mechanical rate, while harmonics above sixteen stay on the grid. Four checks
  meant to kill it failed to: clock skew, an estimator bug, a hover contrast,
  and the proportional-mechanism test that a Doppler or calibration error would
  pass. The leading hypothesis is forward-flight loading modulating the strong
  low harmonics. It matters here because any annotation scored against raw
  telemetry pays this displacement as if it were error.
]

= Backup --- the six refinement steps, ranked by lock

#v(0.5fr)
#align(center, text(size: 1.0em)[
  #table(
    columns: (1fr, auto, auto),
    align: (left, center, center),
    stroke: 0.5pt + luma(180),
    inset: 7pt,
    table.header([*arm*], [*DREGON ridge*], [*FLY124 cruise*]),
    [full chain], [*+1.88*], [+2.98],
    [without the harmonic ladder], [+1.75], [+3.04],
    [without pair mode], [+1.69], [+2.96],
    [without the peel], [+1.64], [+3.07],
    [without pre-smoothing], [+1.43], [+3.20],
    [wide capture band], [#sym.minus\0.20], [+1.47],
    [raw telemetry], [#sym.minus\0.60], [+2.72],
  )
])
#v(0.5fr)

#speaker-note[
  The ablation, in decibels of lock rather than in rev per second of error
  against suspect labels. Every step is worth something on DREGON:
  pre-smoothing 0.45 dB, the peel 0.24, pair mode 0.18, the harmonic ladder
  0.13. On Michael's drone the same four differences are within a couple of
  tenths of each other and change sign, which is what "the labels are already
  right" should look like. The wide-capture arm is the one that matters for the
  optimizer design: widening the band does not merely inflate the estimate, it
  walks the carrier off the comb entirely.
]

= Backup --- step 2: the same verdict, read by $F_"VK"$

#v(0.3fr)
#align(center, text(size: 0.95em)[
  #table(
    columns: (1.3fr, auto, auto, auto),
    align: (left, center, center, center),
    stroke: 0.5pt + luma(180),
    inset: 7pt,
    table.header([*pool*], [*fitted*], [*scaled*], [*raw telemetry*]),
    [DREGON, 9 windows], [*0.6729*], [0.6724], [0.7474],
    [FLY124 cruise, 4 windows], [*0.3984*], [---], [0.4419],
    [FLY124 warm-up, 2 windows], [0.7478], [---], [0.7485],
  )
])

#v(0.5em)
#align(center, text(size: 0.8em)[
  Objective, lower is better, 40 harmonics, 1280 cells. $F_"VK"$ separates the
  two rigs 9.5#sym.times less sharply than the ridge, because the energy share
  saturates: the two instruments have different jobs. Warm-up is inert.
])
#v(0.4em)
#align(center, text(size: 0.78em, style: "italic")[
  Reconciliation: an earlier arms run measured a FLY124 drift of
  #sym.minus\0.35%. That run scored against the pre-recalibration alignment it
  rebuilt itself; on the frozen recalibrated cache the drift is about zero.
  Label versions matter.
])
#v(0.3fr)

#speaker-note[
  Step two of the matrix, scored by the new objective rather than by the ridge,
  so the two instruments can be compared on the same windows. The ordering is
  the same one the ridge campaign published: on DREGON the fitted trajectory and
  the constant-scale correction both beat raw telemetry by a wide margin, and
  the two of them are within five ten-thousandths of each other, which says
  again that most of the DREGON error at this harmonic count is a scale. On
  Michael's cruise the fitted trajectory improves the raw labels only slightly,
  and on warm-up nothing separates at all. The number worth carrying is the
  contrast in sharpness. This objective separates the two rigs about nine and a
  half times less strongly than the ridge does, because the energy share
  saturates once the lines are captured. That is not a fault: a verdict
  instrument should be sharp, an optimization objective should be smooth, and
  these are the two different jobs. The last line is a bookkeeping warning that
  cost real time. An earlier run of the same arms reported a FLY124 drift of
  minus a third of a percent, and it was scoring against an alignment that the
  run itself rebuilt, from before the telemetry recalibration. On the frozen
  recalibrated cache the drift is zero. Always name the label version.
]

= Backup --- the time-shift reading

#v(0.4fr)
#align(center, text(size: 1.0em)[
  #table(
    columns: (1fr, auto, auto, auto),
    align: (left, center, center, center),
    stroke: 0.5pt + luma(180),
    inset: 7pt,
    table.header([*group*], [*best shift*], [*95% CI*], [*basin depth*]),
    [DREGON, at the fitted scale], [#sym.minus\41.5 ms], [[#sym.minus\85, #sym.minus\31]], [1.36 dB],
    [DREGON, no scale correction], [+157.6 ms], [[+115, +250]], [0.50 dB],
    [FLY124 cruise], [+4.1 ms], [[+2.3, +6.0]], [3.23 dB],
  )
])

#v(0.5em)
#align(center, text(size: 0.82em)[
  A lag plus a scale is still an incomplete model: together they buy 0.53 dB
  where the free trajectory buys 1.88 dB.
])
#v(0.4fr)

#speaker-note[
  A second one-parameter probe, and a warning about one-axis sweeps. Without
  the scale correction the shift axis reads plus 158 milliseconds; with it,
  minus 42, and the sign flips, because on a window with a trend a lag looks
  like a scale. The negative control is exact: Michael's labels put their
  maximum at plus 4 milliseconds with a sharp basin, and the four windows agree
  to a few milliseconds. The negative sign on DREGON excludes the proposed
  period-counter mechanism, which can only be positive, and points at a stream
  alignment offset instead. Most importantly, a lag plus a scale still recovers
  far less than the free trajectory, which is the same conclusion the scale
  profile reached.
]
