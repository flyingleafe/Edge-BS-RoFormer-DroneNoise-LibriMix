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
  fill: fill, stroke: 0.7pt + stroke-color, inset: 8pt, radius: 3pt, body,
)

#let pending(caption) = {
  align(center, block(
    width: 90%,
    height: 64%,
    fill: luma(247),
    stroke: (paint: luma(150), thickness: 1pt, dash: "dashed"),
    radius: 4pt,
    align(center + horizon, text(size: 1.1em, fill: luma(90))[
      result pending: running tonight
    ]),
  ))
  v(0.5em)
  align(center, text(size: 0.85em, style: "italic")[#caption])
}

= What this deck asks

#v(0.8fr)
#block(width: 100%, text(size: 1.15em)[
- Why label quality, not model size, is the current ceiling
- How annotation quality is measured, with nulls and hold-outs
- Why a detector is not an objective, and what a usable measure needs
- How the measure gets optimized, and what that unlocks
])
#v(0.8fr)

#speaker-note[
  The story in one line: every neural model in this project is limited by the
  rotor-speed labels it is trained on, so the next contribution is a way to
  produce better labels from audio alone. Four parts. First the evidence that
  labels are the bottleneck. Second, an instrument that says how good an
  annotation is, with calibrated nulls. Third, why that instrument cannot be
  used as a training objective, and what the literature says a usable measure
  must look like. Fourth, the optimization ladder and the experiments that are
  running now. The deck closes on the payoff, which is pseudo-labels for every
  drone recording we have.
]

// ═══ PART 1: labels are the bottleneck ════════════════════════════════════

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

= Generator-trained predictors do not survive real audio

#align(center, image("assets/sim2real.png", height: 76%))

#align(center, text(size: 0.85em)[
  Same architecture, same real validation set. Generated-noise training scores
  worse than predicting the mean.
])

#speaker-note[
  The second symptom of the same disease. Train an RPS predictor on generated
  drone noise, where the labels are exact by construction, and it converges
  nicely on the generated stream, then reads 222 PIT-MSE on the real
  validation set with a negative R squared, against 7.3 for the same
  architecture trained on real recordings. The predictor learns the
  generator's amplitude-to-rate shortcut, which does not exist in real audio.
  So we cannot escape the label problem by synthesizing our way around it: the
  generator inherits its own conditioning error, and its output does not
  transfer.
]

= One rig's telemetry sits on the comb, the other's does not

#grid(
  columns: (1.15fr, 1fr),
  gutter: 1.2em,
  align(center + horizon, image("assets/ridge_telemetry.png", width: 100%)),
  align(horizon)[
    #text(size: 0.85em)[
      Same statistic, same settings, 15 frozen windows. Each bar's own
      off-comb null is the hatched bar.
    ]
    #v(0.5em)
    #text(size: 0.85em)[
      FLY124 clears its null by 3.5 dB. DREGON reads its null.
    ]
    #v(0.5em)
    #text(size: 0.78em, style: "italic")[
      The instrument itself is the next section.
    ]
  ],
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

= The three symptoms have one cure

#v(0.5fr)
#grid(
  columns: (1fr, auto, 1fr),
  gutter: 1.0em,
  align(horizon)[
    #cbox(text(size: 1.0em)[
      *Symptoms*
      - Generator: high harmonics collapse
      - Predictors: no sim-to-real transfer
      - DREGON: labels are off the comb
    ])
  ],
  align(center + horizon, text(size: 2.0em)[#sym.arrow.r]),
  align(horizon)[
    #cbox(fill: rgb("#dceaf7"), stroke-color: rgb("#1f77b4"))[
      #text(size: 1.0em)[
        *What is needed*
        - Precise blind annotation of real recordings
        - Slow is acceptable: it runs once per recording
        - Then every recording carries a pseudo-label
      ]
    ]
  ],
)
#v(0.5fr)

#speaker-note[
  Put the three together. Better telemetry is not available: DREGON's rig is
  fixed, and half the useful drone-noise data in the world has no telemetry at
  all. Synthetic data does not transfer. So the missing capability is
  annotation from the audio itself, at a precision better than the telemetry we
  do have. It is allowed to be slow, because it runs once per recording rather
  than once per training step. That single capability unlocks the generator,
  the predictors, and every unlabelled recording we can get.
]

// ═══ PART 2: measuring annotation quality ═════════════════════════════════

= The instrument: a fixed band against a local floor

#grid(
  columns: (1.25fr, 1fr),
  gutter: 1.0em,
  align(center + horizon, image("assets/ridge_instrument.png", width: 100%)),
  align(horizon)[
    #text(size: 0.78em)[
      $ "ridge" = 10 log_10 ("power in" abs(f) <= f_"dc" (k)) / ("floor density") $
    ]
    #v(0.4em)
    #text(size: 0.85em)[
      #table(
        columns: 2,
        align: (left, left),
        stroke: 0.5pt + luma(200),
        inset: 5pt,
        [fixed], [band, harmonic set, blocks, mask, all from the reference],
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

= The fitted trajectory locks 2.47 dB better than the labels

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

= A single constant explains only a third of the error

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

= The basin law: precision and basin width are one phenomenon

#align(center, image("assets/basin_law.png", width: 78%))

#align(center, text(size: 0.85em)[
  Three literatures derive the same law: harmonic least squares, Vold--Kalman
  passband, phase-locked-loop lock-in range.
])

#speaker-note[
  Why the detector cannot be used as an objective. Any measure that sums
  coherent evidence over K harmonics on a window of T seconds has a main lobe
  in rate of width one over K times T. Harmonic nonlinear least squares derives
  it as a grid requirement, the Vold-Kalman literature as the speed error its
  passband tolerates, and the phase-locked-loop literature as the lock-in
  range. At 80 harmonics and a one-second window that is 0.0125 Hz, so the 40
  to 100 Hz search interval holds about 4800 local maxima per comb. DREGON's
  measured label error is 16 to 40 basin widths away from truth. The same
  arithmetic explains why the phase refiner is inert outside its capture range,
  and it also says the price is unavoidable: the high harmonics carry the
  precision, and the narrow basin is that precision seen from the other side.
]

= Sub-multiples are nested, so no smoothing removes them

#align(center, image("assets/alias_lattice.png", width: 94%))

#align(center, text(size: 0.85em)[
  The fit term is *identical* at $f_0 \/ 2$. Escapes: an order penalty, or
  discrete multiplier moves $u in {1\/3, 1\/2, 2\/3, 3\/2, 2, 3}$ with the
  harmonic count corrected jointly.
])

#speaker-note[
  The second landscape fact, and it is worse than a local minimum. The comb at
  half the rate with twice the harmonics contains the true comb as a subset, so
  under any energy-sum measure its residual is the true residual plus whatever
  the empty slots collect. The degeneracy is exact, not approximate, and it is
  independent of how many harmonics are summed, so no annealing schedule and no
  temporal smoothing ever removes it. This derives the octave errors that the
  pitch literature reports empirically. Two escapes exist and both are needed:
  a model-order penalty that charges for predicted lines landing on no energy,
  fixed rather than evidence-adapted because the adaptive version weakens
  exactly at low signal-to-noise; and discrete multiplier moves that rescale
  the trajectory and correct the harmonic count at the same time, since a bare
  halving proposal is always rejected.
]

= What an optimizable measure has to do

#v(0.7fr)
#text(size: 0.95em)[
  #table(
    columns: (1.15fr, 1fr, 1.15fr, 1.15fr),
    align: (left, left, left, left),
    stroke: 0.5pt + luma(180),
    inset: 8pt,
    table.header(
      [*requirement*], [*ridge (detector)*], [*spectral loss*], [*profiled VK residual*]
    ),
    [optimum at truth], [yes, sharp], [octave-broken], [yes, with an order penalty],
    [usable gradient],
    [none beyond 0.1 rev/s],
    [mel ranking 0.511, a coin flip],
    [closed form, envelope theorem],
    [twin rotors], [scored per carrier], [no mechanism], [coupling blocks compete],
    [capture range],
    [fixed 0.10 rev/s window],
    [window length, no units],
    [bandwidth $rho$, in Hz],
    [continuation], [none], [window ladder], [$K$ ladder, basin known at every level],
  )
]

#v(0.9em)
#align(center, text(size: 0.92em)[
  Gradient descent on a default multi-scale spectral loss lands 2.3 octaves
  off (20% correct); the optimal-transport variant fixes it to 75%.
])
#v(0.7fr)

#speaker-note[
  Put the requirements next to the candidates. The ridge is a sharp detector
  and a dead objective: beyond a tenth of a rev per second it carries no slope
  at all. Spectral losses are refuted as the outer objective by direct
  measurement in the differentiable-DSP literature: gradient-sign ranking
  accuracy for a mel distance on a single clean sinusoid is 0.511, which is a
  coin flip, and descending a default multi-scale spectral loss lands 2.3
  octaves away on average. Configuration fixes part of it, and the
  optimal-transport variant fixes most of it, but neither gives twin-rotor
  competition or a capture range with units. The right-hand column is the
  measure the requirements point at, and it already exists in our own solver.
]

= The measure: the profiled coupled Vold--Kalman residual

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

= The gap: nobody optimizes it over the trajectory

#v(0.6fr)
#align(center, text(size: 1.15em)[
  #table(
    columns: (1fr, 1.1fr, 1.2fr),
    align: (left, left, left),
    stroke: 0.5pt + luma(180),
    inset: 10pt,
    table.header([*line of work*], [*what it optimizes*], [*what it never does*]),
    [Vold--Kalman order tracking], [envelopes, speed given by a tachometer], [treat the speed as free],
    [Adaptive VK, 2024], [bandwidth adaptation], [calls optimizing over the speed "not feasible"],
    [Sinusoidal modeling, 40 years], [discrete peak assignment], [a continuous step over a trajectory],
  )
])

#v(0.8em)
#align(center, text(size: 1.0em, style: "italic")[
  The continuous step over a whole trajectory is the contribution.
])
#v(0.6fr)

#speaker-note[
  The contribution claim, and it is a specific one. The order-tracking
  literature takes the shaft speed as given by a tachometer, because in
  mechanical engineering there always is one. The 2024 adaptive Vold-Kalman
  paper considers optimizing the cost over the instantaneous frequency and
  states outright that it is not feasible. Sinusoidal modeling, from the
  eighties onward, stops at assigning discrete spectral peaks to tracks and
  smoothing them. So the continuous optimization of a whole speed trajectory
  against the coupled residual is a named, verifiable gap, not a
  differently-flavoured variant of something published.
]

// ═══ PART 4: optimizing the measure ═══════════════════════════════════════

= Three rungs: the slow oracle, the baselines, and ours

#v(0.6fr)
#align(center, text(size: 1.1em)[
  #table(
    columns: (auto, 1.25fr, 1fr),
    align: (left, left, left),
    stroke: 0.5pt + luma(180),
    inset: 10pt,
    table.header([*rung*], [*method*], [*role*]),
    [oracle],
    [L-BFGS on the profiled residual under the $K$ anneal, started from telemetry],
    [how good can an annotation get],
    [baselines],
    [adaptive-VK refinement; tacholess and ridge order tracking],
    [what the literature already reaches],
    [ours],
    [blind comb Viterbi seed, then peel and phase-increment alternation],
    [no telemetry at any stage],
  )
])

#v(0.5em)
#align(center, text(size: 0.85em, style: "italic")[
  Sanity condition for the oracle: it must barely move Michael's labels, and
  move DREGON's by 0.35 to 0.85 percent.
])
#v(0.6fr)

#speaker-note[
  Three rungs, and they answer different questions. The top rung starts from
  telemetry and runs quasi-Newton optimization of the profiled residual under
  the harmonic anneal. It is slow and it needs a seed, so it is not a
  deployable tracker; it is the ceiling, the best annotation the audio can
  support. The middle rung is what the literature reaches today, and it is what
  a reviewer will ask about: adaptive Vold-Kalman refinement given a seed, and
  the tacholess order-tracking family that estimates speed from the signal.
  The bottom rung is ours and it is fully blind. The sanity condition matters
  more than any number on this slide: the oracle must leave Michael's
  recalibrated labels almost untouched while moving DREGON's by the amount the
  fitness campaign already measured independently. If it moves both, it is a
  flexible fitter and nothing it says can be trusted.
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

= The experiment matrix

#v(0.5fr)
#text(size: 1.0em)[
  #table(
    columns: (auto, 1.4fr, 1fr, auto),
    align: (left, left, left, center),
    stroke: 0.5pt + luma(180),
    inset: 8pt,
    table.header([*step*], [*question*], [*evidence*], [*status*]),
    [1], [Is the measure the right one], [landscape benchmark: optimum margin, basin profiles, gradient ranking], [running],
    [2], [How good are the labels we have], [ridge verdicts, nulls, hold-outs, scale profile], [done],
    [3], [Does the oracle behave], [L-BFGS from telemetry: DREGON moves, Michael's does not], [tonight],
    [4], [Can we improve telemetry], [synthetic recovery first, then both rigs], [tonight],
    [5], [How good is blind annotation], [precision against compute, multi-start L-BFGS as the anchor], [next],
  )
]

#v(0.5em)
#align(center, text(size: 0.82em)[
  Every row is scored by the same instrument, on the same 15 frozen windows,
  at fixed degrees of freedom.
])
#v(0.5fr)

#speaker-note[
  The plan as a matrix, so the pending rows are visible rather than implied.
  Step one is the landscape benchmark: measure each candidate measure's
  optimum margin over the structured alias set, its directional basin profiles
  and its gradient-sign accuracy, on synthetic combs where truth is known.
  Step two is the campaign already reported. Step three is the oracle sanity
  check that runs tonight. Step four asks whether the oracle actually improves
  a label set, starting on synthetic data where improvement is verifiable, then
  on both real rigs. Step five is the deployable question, and it is a
  trade-off rather than a single number: how much precision per unit of
  compute, with multi-start optimization as the expensive anchor at one end.
]

= Step 3: does the oracle move the right rig?

#pending[
  L-BFGS from telemetry under the $K$ anneal: rate change per rig, DREGON
  against Michael's, with the fitness reading before and after.
]

#speaker-note[
  Placeholder for tonight's run. The figure will be a two-panel comparison:
  how far the optimizer moves each rig's labels, and what the lock reading does
  in response. The prediction is on the record, from an instrument that never
  saw this optimizer: DREGON moves by something between a third and nine
  tenths of a percent, Michael's barely moves, and the lock reading rises on
  DREGON and stays flat on Michael's. If both rigs move, the optimizer is
  fitting noise and step four does not run.
]

= Step 4: does the optimizer improve a label set?

#pending[
  Synthetic first, where improvement is verifiable against known truth, then
  DREGON and Michael's, scored by the fitness instrument.
]

#speaker-note[
  Placeholder for the second run. Synthetic first: corrupt a known trajectory
  the way telemetry corrupts it, optimize, and measure the recovered error
  directly, because that is the only setting where the answer is not an
  inference. Then the two real rigs, scored by the same instrument as step
  two, so the numbers drop into the same table. The comparison of interest is
  against the alternation chain: the current 2.47 dB is what a hand-built
  alternation reaches, and the question is how much of the remaining gap
  gradient optimization closes.
]

= Step 5: blind annotation, precision against compute

#pending[
  Pareto front: annotation error against compute per window, from the fast
  blind chain to multi-start optimization.
]

#speaker-note[
  Placeholder for the deployable result. This is a front, not a point:
  the fast blind chain is seconds per window, multi-start optimization under
  the full anneal is minutes to hours, and the useful statement is how much
  precision each buys. The anchor at the expensive end is multi-start
  L-BFGS, which is also the ceiling estimate. The practical target is the
  point where blind annotation is better than the telemetry we already have,
  because past that point the labels stop being the bottleneck.
]

// ═══ PART 5: the payoff ═══════════════════════════════════════════════════

= The bootstrap this unlocks

#v(0.3fr)
#align(center)[
  #cbox(fill: rgb("#dceaf7"), stroke-color: rgb("#1f77b4"))[
    #text(size: 1.0em)[blind annotation of real recordings]
  ]
  #v(0.4em)
  #text(size: 1.4em)[#sym.arrow.b]
  #v(0.4em)
  #cbox[#text(size: 1.0em)[pseudo-labels: DREGON, Michael's, the audio-visual quadrotor set, future data]]
  #v(0.4em)
  #text(size: 1.4em)[#sym.arrow.b]
  #v(0.4em)
  #cbox[#text(size: 1.0em)[RPS predictors trained on real audio]]
  #h(0.6em)
  #cbox[#text(size: 1.0em)[generators with scale-correct conditioning]]
  #v(0.4em)
  #text(size: 1.4em)[#sym.arrow.b]
  #v(0.4em)
  #cbox(fill: rgb("#e8f5e9"), stroke-color: rgb("#2ca02c"))[
    #text(size: 1.0em)[rotor-speed-conditioned speech enhancement]
  ]
]
#v(0.3fr)

#speaker-note[
  Why this is worth a paper and not just a tooling fix. An annotator that works
  from audio alone turns every drone recording into a labelled recording,
  including the ones with no telemetry at all, which is most of the public
  data. Those pseudo-labels feed the two models that this deck opened with. The
  generator gets conditioning that is on the comb, so its high harmonics stop
  collapsing. The predictors get real audio with trustworthy labels instead of
  synthetic audio with exact labels that do not transfer. And both of those
  feed the actual goal, which is conditioning speech enhancement on rotor
  speed. The loop closes: better labels give better models, and better models
  give better seeds for the next annotation pass.
]

= Where this stands

#v(0.4fr)
#align(center, block(width: 82%, align(left, text(size: 1.35em)[
  - Annotation quality is now measurable, with nulls and hold-outs
  - The best label set we have is 2.47 dB off the comb
  - The measure to optimize is chosen, and the gap in the literature is named
  - Tonight: the oracle sanity run, then telemetry improvement
])))
#v(0.4fr)

#speaker-note[
  Four sentences. Annotation quality stopped being a matter of opinion in the
  comb explorer and became a measurement with a calibrated zero, three nulls
  and four hold-out families. Applied to the labels we actually train on, it
  says DREGON's are 2.47 dB off what the audio supports. The measure to
  optimize is the profiled coupled residual, chosen against a requirements
  table rather than by taste, and the fact that nobody optimizes it over the
  trajectory is the contribution claim. The two runs that go out tonight are
  the sanity check and the first improvement result, and both drop into the
  matrix without changing the story.
]

=

#align(center + horizon)[
  #text(size: 1.3em, weight: "bold")[Backup]
]

#speaker-note[
  Divider. Backup material: the tracking scoreboard, the middle steps of the
  blind chain, the phase-coherence measurements, the displaced comb, and the
  two campaign tables behind the fitness verdicts.
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

= Backup --- blind chain step 2: ramp handling

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

= Backup --- blind chain step 3: per-rotor decoupling

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

= Backup --- blind chain step 4: envelope solve, then peel

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

= Backup --- output comparison, DREGON cruise window

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

= Backup --- output comparison, FLY124 cruise window

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
