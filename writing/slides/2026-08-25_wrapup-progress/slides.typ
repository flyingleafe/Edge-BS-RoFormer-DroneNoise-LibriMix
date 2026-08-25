#import "@preview/touying:0.7.4": *
#import "/writing/templates/typst/slides.typ": hns-slides

#let notes-mode = sys.inputs.at("notes", default: none)

#show: hns-slides.with(
  title: [Rotor speeds from ego-noise: the wrap-up],
  subtitle: [One frozen protocol, every method, one fix],
  author: [Dmitrii Mukhutdinov],
  date: [2026-08-25],
  show-notes: if notes-mode != none { bottom } else { none },
)

#let cbox(body, fill: luma(235), stroke-color: luma(160)) = box(
  fill: fill,
  stroke: 0.7pt + stroke-color,
  inset: 8pt,
  radius: 3pt,
  body,
)

#let small(body) = text(size: 0.82em, body)

// ---------------------------------------------------------------------
= The question and the paper

#v(1fr)
#align(center)[
  #text(size: 1.3em)[*Can we get per-rotor speeds from onboard audio alone?*]
]
#v(1em)

#text(size: 1.05em)[
  - A rotor spinning at rate $f$ leaves harmonics at $f, 2f, 3f, ...$ in the recording.
  - Knowledge of rotor speed enables informed ego-noise removal and drone monitoring.
  - The task has no published direct method — we build the baseline suite ourselves.
]
#v(1em)
#cbox(text(size: 1.05em)[
  The paper's arc: baselines from two adjacent tasks (multi-pitch, tacholess
  order tracking) $arrow.r$ a 26-architecture search $arrow.r$ a training-regime
  diagnosis $arrow.r$ a fix $arrow.r$ a blind classical tracker as the
  precision ceiling $arrow.r$ synthetic data re-measured.
])
#v(1fr)

#speaker-note[
  Framing slide. The audience has seen VK tracking, generators, curricula
  before, but not the unified leaderboard this deck is built around.
]

// ---------------------------------------------------------------------
= Data reality: about an hour of labeled audio, two drones

#v(1fr)
#table(
  columns: (auto, 1fr, 1fr),
  stroke: 0.5pt + luma(200),
  inset: 7pt,
  align: (left, left, left),
  table.header([*Split*], [*DREGON*], [*MD2*]),
  [Train], [5 flight recordings (room 2)], [1 flight (FLY125)],
  [Validation (frozen)], [3 recordings (room 1)], [1 flight (FLY124)],
  [Test (reserved)], [2 recordings (room 1)], [2 flights (FLY103/108)],
)
#v(1em)
#text(size: 0.95em)[
  - The three-way split is now formal: every number in the paper comes from
    the same 37-clip frozen validation set.
  - Two MD2 test flights (FLY103/108) are newly calibrated: the catch was a
    shared *1.19% session clock dilation* in the telemetry.
  - The test split is reserved — no experiment in this paper reads it yet.
]
#v(1fr)

#speaker-note[
  Splits table from paper sec:splits. Cut the lag-fit mechanics — one line
  is enough, the full story is report material.
]

// ---------------------------------------------------------------------
= One protocol, five regimes

#v(1fr)
#cbox(text(size: 1.05em)[
  Every number in the paper comes from *37 clips x 8 channels*, per-frame
  *Hungarian PIT* matching, pooled into regimes: zero, low, flight; the
  tables here report zero, flight and the all-frame aggregate.
])
#v(1.2em)
#table(
  columns: (auto, 1.3fr, 1.3fr),
  stroke: 0.5pt + luma(200),
  inset: 6pt,
  align: (left, left, left),
  table.header([*regime*], [*training data*], [*what it isolates*]),
  [R1], [light augmentation, old validation split], [architecture search],
  [R2], [full flight envelope + freq. scaling + time warp + honest silence fix], [the real-only, honest base (HB)],
  [R3], [generator + comb curriculum, fine-tuned on R2], [generator-first coverage],
  [R4], [comb-only curriculum, fine-tuned on R2], [comb coverage alone],
  [R5], [one mixed pool: real / generated / comb, single stage], [single-stage mixing],
)
#v(1fr)

#speaker-note[
  This is the frame the rest of the deck hangs numbers on. R1 through R5 get
  referenced by name from here on.
]

// ---------------------------------------------------------------------
= The training-free floor: classical methods miss by 2-3 orders of magnitude

#v(1fr)
#table(
  columns: (auto, 1fr, 1fr, 1fr),
  stroke: 0.5pt + luma(200),
  inset: 6pt,
  align: (left, right, right, right),
  table.header([*method*], [*zero MAE*], [*flight MAE*], [*all MAE*]),
  [PYIN], [90.8], [34.1], [43.0],
  [cepstral], [95.5], [19.6], [35.7],
  [HPS], [64.1], [20.9], [27.3],
  [matched filter], [87.2], [30.4], [42.5],
  [NMF], [83.8], [*8.1*], [24.7],
  [OT multi-pitch (2026)], [68.8], [16.3], [24.5],
)
#v(1em)
#text(size: 0.9em)[
  MAE in rev/s. NMF is the best training-free cruise number we have (8.1) and
  beats the 2026 optimal-transport multi-pitch method (16.3) on the same
  frames. No classical method can say *"stopped"* — every method here clamps its
  search grid above 50 rev/s, so a stopped rotor scores near the clamp floor.
]
#v(1fr)

#speaker-note[
  Classical five + OT table, from unified-baseline-eval.md. NMF ranking
  matches the archived May report; that report is a footnote at most.
]

// ---------------------------------------------------------------------
= What the models actually read

#v(1fr)
#cbox(text(size: 1.15em)[
  A 2% frequency scaling of the input spectrum moves predictions by *0.03%*.
])
#v(1.2em)
#text(size: 0.95em)[
  - Real-only models mostly ignore the frequency axis: they read loudness,
    spectral shape, and the prior over speed values.
  - Traced to a level shortcut in the training data: every zero-labeled
    training chunk is globally quiet, from one room only.
  - Result: on a real zero-speed clip that still carries sound (a rumble
    clip), the real-only model does not know to say "stopped."
]
#v(0.8em)
#table(
  columns: (auto, 1fr, 1fr),
  stroke: 0.5pt + luma(200),
  inset: 6pt,
  align: (left, left, left),
  table.header([*split*], [*zero-time share*], [*unique silence*]),
  [Train], [6.25% of chunks], [26.4 s, one room],
  [Validation], [12.7% of frames], [—],
)
#v(1fr)

#speaker-note[
  This is the diagnosis slide. It sets up the fix (next slide) and the
  punchline (slide after).
]

// ---------------------------------------------------------------------
= The fix: an honest regime (R2) with voicing gates

#v(1fr)
#text(size: 1.0em)[
  Three changes to the training recipe, all landing in R2 — models trained
  this way are the honest base (HB).
]
#v(0.6em)
#grid(
  columns: (1fr, 1fr, 1fr),
  gutter: 1em,
  cbox(text(size: 0.9em)[
    *1. Zero-labeled silence with content* \
    Room tone, colored noise floors up to flight level, low-frequency
    rumble — at 17% of chunks, not just quiet air.
  ]),
  cbox(text(size: 0.9em)[
    *2. Reference-power floor* \
    Speech is mixed against a floor, not the noise chunk's own power, so
    silent chunks still carry speech at a normal level.
  ]),
  cbox(text(size: 0.9em)[
    *3. Sigmoid voicing gate* \
    The output head gets an explicit off state instead of relying on the
    linear head's conditional mean.
  ]),
)
#v(1fr)

#speaker-note[
  Design sketch from honest-base-frontends.md. Sets up the punchline grid.
]

// ---------------------------------------------------------------------
= PUNCHLINE: the honest regime closes most of the zero gap

#v(0.3em)
#align(center)[
  #text(size: 1.05em)[*Real-only 11.8 $arrow.r$ HB best 3.7 MAE on zero, at almost no cruise cost*]
]
#v(0.4em)
#grid(
  columns: (1.4fr, 0.8fr),
  gutter: 1em,
  align(horizon)[
    #table(
      columns: (auto, 1fr, 1fr, 1fr, 1fr),
      stroke: 0.5pt + luma(200),
      inset: 5pt,
      align: (left, right, right, right, right),
      table.header([*trunk*], [*mag*], [*IF*], [*ssq*], [*agg*]),
      [scv2], [3.68/2.57], [4.50/2.49], [6.04/2.52], [39.7 / *34.9* / 47.1],
      [transf.], [3.70/2.79], [3.55/3.08], [4.42/2.55], [*31.7* / 33.6 / 36.7],
      [GRU], [10.62/2.66], [7.20/2.52], [4.57/2.67], [60.7 / 50.5 / *39.8*],
    )
    #v(0.4em)
    #text(size: 0.75em)[
      Cells: zero/flight MAE (rev/s). agg = aggregate validation score for
      mag / IF / ssq; bold marks the best aggregate cell per architecture.
      Front-end winner is architecture-dependent — IF for scv2, magnitude
      for the transformer, synchrosqueezed for the causal GRU (repairs the
      zero deficit 10.62 $arrow.r$ 4.57). Best cell:
      *transformer + magnitude, 31.7 aggregate* vs real-only control *42.3*
      — better in every regime, no ramp regression.
    ]
  ],
  align(center)[
    #image("assets/qual_zero.png", height: 3.4in)
    #v(0.2em)
    #small[Rumble clip: HB 0.51 vs real-only 19.6 MAE]
  ],
)

#speaker-note[
  The 3x3 grid from unified-baseline-eval.md HB rows, plus the qual_zero
  figure from the paper. This is the deck's center of gravity.
]

// ---------------------------------------------------------------------
= The control that settled it

#v(1fr)
#align(center)[
  #text(size: 1.15em)[
    *Overnight: `hb_scv2_mag_nogate` reaches 22.1 aggregate MSE — the best
    neural cell of the whole campaign, real data only, no voicing gate.*
  ]
]
#v(0.6em)
#text(size: 0.95em)[
  Best-in-class in every regime: zero *3.36*, ramps *4.18*, cruise MSE
  *11.1*. The gate from the fix slide was not the source of the punchline's
  gain — the honest-regime data was.
]
#v(0.8em)
#table(
  columns: (auto, 1fr, 1fr, 1fr),
  stroke: 0.5pt + luma(200),
  inset: 6pt,
  align: (left, right, right, right),
  table.header([*trunk*], [*old real-only*], [*R2 ungated*], [*R2 gated (best)*]),
  [conv (scv2)], [52.5], [*22.1*], [39.7],
  [Transformer], [42.3], [41.8], [*33.6*],
  [causal GRU], [59.2], [61.9], [*39.8*],
)
#v(0.8em)
#text(size: 0.9em)[
  Aggregate validation MSE. Gate effect per trunk: conv $-17$ (hurts),
  transformer $+8$ (helps), causal GRU $+22$ (helps most).
]
#v(0.6em)
#cbox(text(size: 0.95em)[
  The honest-data windfall is trunk-dependent. The voicing gate helps heads
  that cannot infer silence from context on their own (attention, causal
  recurrence) — and hurts the convolutional trunk, which already reads
  silence from local context without one.
])
#v(1fr)

#speaker-note[
  Overnight ablation isolating gate vs data. Frames the punchline as a
  data effect, not an architecture trick — sets up "gate is a tool, not a
  default" for anyone re-running the sweep.
]

// ---------------------------------------------------------------------
= The blind tracker meets the same protocol

#v(1fr)
#table(
  columns: (auto, 1fr, 1fr, 1fr, 1fr),
  stroke: 0.5pt + luma(200),
  inset: 6pt,
  align: (left, right, right, right, left),
  table.header([*convention*], [*zero*], [*flight*], [*all*], [*note*]),
  [ungated], [79.36], [*2.27*], [17.01], [finds combs in silence],
  [gated, refusal $arrow.r$ 0], [*0.01*], [48.35], [39.72], [8/20 windows accepted],
)
#v(1em)
#text(size: 0.95em)[
  - The blind Vold-Kalman order tracker (the 08-18 deck's method), ungated,
    beats *every* neural cell on cruise MAE (2.27 vs 2.49-3.08) — it fails
    silence completely.
  - Gated by its per-window acceptance gates, it decides silence perfectly
    but loses over half its cruise windows: the gates were
    precision-calibrated, not recall-calibrated.
  - Cost: *9.87 CPU-s per audio-second* — about 10x realtime on CPU.
]
#v(1fr)

#speaker-note[
  Two-row table + compute number, unified-baseline-eval.md blind rows.
]

// ---------------------------------------------------------------------
= Synthetic data: coverage, not realism — resolved

#v(1fr)
#table(
  columns: (auto, 1fr, 1fr),
  stroke: 0.5pt + luma(200),
  inset: 6pt,
  align: (left, right, right),
  table.header([*trunk*], [*R2 control*], [*+ curriculum*]),
  [conv (scv2, best trunk)], [22.1], [22.6 (gen+comb, R3/R4) — no gain],
  [causal GRU (weakest trunk)], [61.9], [41.8 (gen+comb) / *37.6* (comb-only, R4)],
)
#v(0.8em)
#text(size: 0.95em)[
  - On top of the honest base, the generator+comb curriculum (R3/R4) adds
    *nothing* for the best trunk — R2 already supplies the coverage it used
    to buy.
  - It still helps the weakest trunk, and there the free analytic comb
    alone (R4) beats the trained generator (R3): 37.6 vs 41.8.
  - Mixed one-stage training (R5: real / generated / comb, single pool)
    *loses on all three trunks* against their R2 controls — 147.6 / 59.2 /
    85.8 vs 22.1 / 41.8 / 61.9. Staging is not optional.
]
#v(1fr)

#speaker-note[
  Reruns landed overnight. Old open question (how much of R3/R4's edge was
  silence coverage R2 now supplies for free) is answered: almost all of it,
  for the trunk that matters. Staging-necessity claim now unconditional
  across all three trunks, not just the best one.
]

// ---------------------------------------------------------------------
= Salience baselines: retrained on R2

#v(1fr)
#table(
  columns: (auto, 1fr, 1fr, 1fr),
  stroke: 0.5pt + luma(200),
  inset: 6pt,
  align: (left, right, right, right),
  table.header([*model*], [*zero*], [*ramps*], [*cruise*]),
  [Basic Pitch], [34.0], [13.3], [31.7],
  [multi-F0 CNN, standard grid], [52.8], [—], [*4.0*],
  [multi-F0 CNN, widened fine grid], [48.2], [*16.1*], [—],
)
#v(0.8em)
#text(size: 0.95em)[
  RMSE, rev/s. Basic Pitch stays broken across every regime — an
  architectural ceiling, not a data problem.
  The multi-F0 CNN is cruise-decent (4.0) but silence-blind (52.8) on the
  standard grid; widening the grid fixes ramps (21.0 $arrow.r$ 16.1) but
  not silence (48.2). Ground-truth round-trip floor: *2.25*.
]
#v(0.6em)
#cbox(text(size: 0.95em)[
  The salience family lights up on content-rich silence — the same failure
  mode the RPS trunks had before the honest fix. Here it is a model
  limitation, not (yet) a data one: the grid fix that helped ramps did not
  touch it.
])
#v(1fr)

#speaker-note[
  Salience retrain finished overnight. Standard-grid cruise number and
  widened-grid ramps number come from the same sweep; kept as two rows
  since the grids trade off differently per regime.
]

// ---------------------------------------------------------------------
= Seeing the errors

#text(size: 0.9em)[
  The rumble-silence panel is on the punchline slide. Here: the two other
  error modes, at full size.
]
#v(0.3em)
#grid(
  columns: (1fr, 1fr),
  gutter: 0.8em,
  align(center)[
    #image("assets/qual_transition.png", height: 4.0in)
    #small[Stop-start transition]
  ],
  align(center)[
    #image("assets/qual_cruise.png", height: 4.0in)
    #small[Steady cruise]
  ],
)

#speaker-note[
  Three generated panels from writing/papers/2026-08_wrapup/figures/.
]

// ---------------------------------------------------------------------
= In flight tonight

#v(1fr)
#text(size: 0.95em)[
  - R3/R4 transformer cells and the R4 convolutional cell are still on
    cloud lanes, filling out the attribution matrix at the other trunks.
  - hb\_ebsrof (band-split RoPE trunk) is *learning for the first time*:
    best 438 vs the flat July control ~1150 — the honest regime unblocked
    it, though it is still far behind the leaderboard.
  - The phase-only Kalman-attention row and the HG-CKLA stage-A refiner are
    both running (next slide covers HG-CKLA in full).
]
#v(1em)
#table(
  columns: (auto, 1fr),
  stroke: 0.5pt + luma(200),
  inset: 6pt,
  align: (left, left),
  table.header([*row*], [*status*]),
  [R3/R4 transformer + R4 conv cells], [running, cloud lanes],
  [hb\_ebsrof (band-split RoPE trunk)], [running — best 438, up from ~1150],
  [Phase-only Kalman-attention row], [running],
  [HG-CKLA stage-A refiner (G1)], [running],
)
#v(1fr)

#speaker-note[
  Fleet table trimmed to what's actually still open — synthetic reruns and
  salience retrains both resolved earlier in the deck now.
]

// ---------------------------------------------------------------------
= Next bet: HG-CKLA

#v(1fr)
#text(size: 1.0em)[
  *The gap:* CKLA was meant to be a learned pi_kalman pass, but pooling the
  spectral axis before the recurrence removes "at harmonic $k$ of rotor $r$"
  from what the layer can read. A sequence mixer over pooled features can
  filter; it cannot run an extended-Kalman measurement update.
]
#v(1em)
#cbox(text(size: 0.95em)[
  *HG-CKLA:* move the measurement inside the recurrence — state-conditioned
  harmonic gathers, innovation phasors, WP18 phase-noise weights (~$k^2$).
  The cell reads the spectrogram at positions its own state predicts, every
  step.
])
#v(1em)
#text(size: 0.9em)[
  Gates, in order: *G1* synthetic combs with known corruption; *G2* the
  frozen tracking protocol against the classical tracker; *G3* end-to-end
  vs the HB winner on the frozen split.
]
#v(1fr)

#speaker-note[
  One architecture sketch from docs/pikalman-ckla-design.md. Closing slide.
]
