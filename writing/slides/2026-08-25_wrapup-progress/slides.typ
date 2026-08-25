#import "@preview/touying:0.7.4": *
#import "/writing/templates/typst/slides.typ": hns-slides

#let notes-mode = sys.inputs.at("notes", default: none)

#show: hns-slides.with(
  title: [Rotor speeds from ego-noise: the wrap-up],
  subtitle: [Figures and tables on one frozen protocol],
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

#let small(body) = text(size: 0.78em, body)
#let fig(path, h) = align(center, image(path, height: h, fit: "contain"))

// ---------------------------------------------------------------------
= The question and the paper

#v(1fr)
#align(center)[
  #text(size: 1.3em)[*Can we get per-rotor speeds from onboard audio alone?*]
]
#v(1em)

#text(size: 1.02em)[
  - Per-rotor speeds from onboard audio: each rotor at rate $f$ leaves harmonics at $f, 2f, 3f, ...$
  - No published direct method exists — we build the baseline suite ourselves.
  - A 26-variant architecture search gives three winner architectures.
  - Augmentations and a full-envelope real training regime repair what models actually read.
  - A blind two-stage classical tracker sets the cruise precision ceiling, at high CPU cost.
  - Synthetic data supplies coverage, not realism — and loses its value once real data supplies the same coverage.
]
#v(1fr)

// ---------------------------------------------------------------------
= Data split

#v(1fr)
#table(
  columns: (auto, 1fr, 1fr, auto),
  stroke: 0.5pt + luma(200),
  inset: 6pt,
  align: (left, left, left, left),
  table.header([*Split*], [*DREGON (small quad)*], [*MD2 (DJI M100)*], [*Speech*]),
  [Training],
  [free-flight and hover recordings, refined per-rotor labels],
  [FLY124, FLY125 (recalibrated telemetry)],
  [LibriSpeech],
  [Validation (frozen)],
  [8 held-out slices, full envelope (ground, ramps, cruise)],
  [29 held-out slices from FLY124/FLY125],
  [—],
  [Test (reserved, untouched)],
  [free-flight speech-high room1, whitenoise-high room1],
  [FLY103, FLY108 (calibrated, frames published)],
  [—],
)
#v(0.6em)
#cbox[
  The validation set: 37 clips $times$ 8 channels $times$ 8 s. Per-frame
  Hungarian matching of 4 predicted to 4 true speeds; errors pooled per
  regime — *zero* (all rotors stopped), *ramps*, *flight* (all rotors
  $gt.eq$ 45 rev/s) — and overall (frame-weighted).
]
#v(1fr)

// ---------------------------------------------------------------------
= Training regimes

#small[
#table(
  columns: (auto, 1fr),
  stroke: 0.5pt + luma(200),
  inset: 6pt,
  align: (left, left),
  table.header([*Regime*], [*Full description*]),
  [R1 — architecture search],
  [Online mixing: random 1-s noise chunks from the real pool, random speech
   overlay at random SNR, new mixtures every epoch. Augmentations after a
   50k-sample warm-up: random gain, polarity flip, channel drop (each
   $p = 0.5$). No silence, no label-transforming augmentations.],
  [R2 — full-envelope real (final real-only)],
  [R1 mixing plus: *silence arm* — 16.7% of chunks are synthetic room
   tone / colored noise / low-frequency rumble with all-zero speed labels;
   *SNR reference floor* — quiet chunks are not amplified to the target SNR
   (reference RMS floor 0.02), so level stops predicting speed;
   label-transforming *frequency-scale* and *time-warp* augmentations
   (labels scaled/warped together with audio).],
  [R3 — generated + comb curriculum],
  [Stage 1: pre-train on synthetic noise from the neural generator plus the
   analytic comb, full-envelope speed trajectories with exact silence.
   Stage 2: fine-tune under R2. Same optimizer, loss, budget.],
  [R4 — comb-only curriculum],
  [R3 with the neural generator removed: stage 1 is the analytic comb
   alone, stage 2 is R2.],
  [R5 — mixed, one stage],
  [One pool, one stage: real 1/2, generated 1/4, comb 1/4, plus the R2
   silence arm; the same augmentation schedule. No curriculum.],
)
]

// ---------------------------------------------------------------------
= Label-transforming augmentations

#grid(
  columns: (1fr, 1fr),
  gutter: 10pt,
  [
    #fig("assets/timewarp_before_after.png", 5.6cm)
    #small[*Time-warp*: resample audio and speed labels on one warped clock
    ($alpha lt.eq 1.12$) — new speed trajectories from the same recording.]
  ],
  [
    #fig("assets/freqscale_illustration.png", 5.6cm)
    #small[*Frequency-scale*: scale the spectrum and the labels by the same
    $alpha$ — a genuine comb-spacing change the model must follow.]
  ],
)

// ---------------------------------------------------------------------
= The three regimes of the validation set

#grid(
  columns: (1fr, 1fr, 1fr),
  gutter: 6pt,
  [
    #align(center)[*zero — stopped rotors*]
    #fig("assets/qual_zero.png", 5.8cm)
  ],
  [
    #align(center)[*ramps — start/stop transition*]
    #fig("assets/qual_transition.png", 5.8cm)
  ],
  [
    #align(center)[*flight — cruise*]
    #fig("assets/qual_cruise.png", 5.8cm)
  ],
)
#small[Validation slices: spectrogram on top, per-rotor speed tracks below.
The same three clips return full-size later in the deck.]

// ---------------------------------------------------------------------
= Classical pitch-tracking baselines

#small[
#table(
  columns: (auto, 1fr, auto, auto, auto, auto),
  stroke: 0.5pt + luma(200),
  inset: 5.5pt,
  align: (left, left, right, right, right, right),
  table.header([*Method*], [*What it is*], [*zero*], [*ramps*], [*cruise*], [*all*]),
  [PYIN $times$4], [probabilistic YIN, greedy 4-peak pick], [90.8], [46.7], [34.1], [43.0],
  [Cepstral], [cepstral peak picking, greedy], [95.5], [67.5], [19.6], [35.7],
  [HPS], [harmonic product spectrum, greedy], [64.1], [27.6], [20.9], [27.3],
  [Matched filter], [comb-template bank correlation], [87.2], [66.8], [30.4], [42.5],
  [NMF], [non-negative factorization, harmonic dictionary], [83.8], [59.5], [*8.1*], [24.6],
  [IHC], [inverse harmonic clustering (Björkman & Elvander)], [68.8], [28.3], [16.3], [24.6],
)
]
#v(0.5em)
#cbox[
  PIT MAE, rev/s. NMF is the best training-free cruise tracker (8.1).
  Every method floors at its grid boundary on stopped rotors — silence
  is unrepresentable for a pitch grid that starts at 50 rev/s.
]

// ---------------------------------------------------------------------
= Salience baselines: what they are

#grid(
  columns: (1fr, 1fr),
  gutter: 10pt,
  [
    #fig("assets/multif0-illustration-1.png", 5.4cm)
    #small[*Multi-F0 CNN* (Cuesta et al.): HCQT input, per-bin salience
    map, trained with BCE on the comb positions; Hungarian tracking
    decodes 4 speed tracks from the map.]
  ],
  [
    #fig("assets/basic-pitch-illustration.png", 5.4cm)
    #small[*Basic Pitch* (Spotify): lightweight polyphonic pitch model,
    ported to PyTorch; same salience-map decoding. Both retrained under
    the same R2 stream and augmentations as our models.]
  ],
)

// ---------------------------------------------------------------------
= Salience baselines: outputs and results

#grid(
  columns: (1.15fr, 1fr),
  gutter: 10pt,
  [
    #fig("assets/sample_00026_multif0_salience_salience.png", 4.0cm)
    #fig("assets/sample_00026_multif0_salience_rps.png", 4.0cm)
    #small[Multi-F0 salience map (top) and decoded tracks (bottom) on a
    cruise clip.]
  ],
  [
    #small[
    #table(
      columns: (auto, auto, auto, auto, auto),
      stroke: 0.5pt + luma(200),
      inset: 5.5pt,
      align: (left, right, right, right, right),
      table.header([*Retrained model*], [*zero*], [*ramps*], [*cruise*], [*all*]),
      [Basic Pitch], [34.0], [13.3], [31.7], [29.5],
      [multi-F0, standard grid], [52.8], [21.0], [4.0], [12.5],
      [multi-F0, wide fine grid], [48.2], [16.1], [4.7], [11.7],
    )
    ]
    #v(0.5em)
    #cbox[
      Cruise-capable (4.0 rev/s) and silence-blind: the salience map
      lights up on room rumble, and no retraining regime fixes it — a
      model-class limitation, not a data one.
    ]
  ],
)

// ---------------------------------------------------------------------
= Our classical tracking: blind two-stage annotation

#grid(
  columns: (1fr, 1fr),
  gutter: 10pt,
  [
    #fig("assets/stepper_viterbi_c.png", 5.6cm)
    #small[*Stage 1 — Viterbi ridge-seeking*: comb-score lattice over
    candidate speeds; joint 8-channel dynamic programming picks the
    smoothest high-score path per rotor.]
  ],
  [
    #fig("assets/stepper_vit2dsp.png", 5.6cm)
    #small[*Stage 2 — phase-increment Kalman refinement*: demodulate each
    harmonic at the current estimate, read the phase increments, fuse
    them with SNR-dependent weights, smooth, repeat coarse-to-fine.]
  ],
)
#v(0.4em)
#cbox[
  *Zero gate*: a window is accepted only when the comb score clears its
  off-comb floor (margin and clearance gates); a refused window emits
  zero speeds. Gate-free: cruise MAE *2.27* rev/s (best of all methods).
  Gated: zero MAE *0.01*, at the price of refusing hard flight windows.
]

// ---------------------------------------------------------------------
= Architecture search: 26 variants, 52 runs

#grid(
  columns: (1.25fr, 1fr),
  gutter: 10pt,
  [
    #fig("assets/s1_online_leaderboard.png", 7.2cm)
    #small[The R1 sweep leaderboard (online-mixing arm): six variant groups
    over one conv trunk — temporal heads, input features, pooling, causal
    heads, dilated backbones.]
  ],
  [
    #let node(body, fill: luma(240)) = box(fill: fill, stroke: 0.8pt + luma(120),
      inset: 6pt, radius: 3pt, width: 100%, align(center, text(size: 0.75em, body)))
    #let arrow = align(center, text(size: 0.8em, sym.arrow.b))
    #node[STFT front-end (1 s chunk)\ magnitude / +IF / synchrosqueezed]
    #arrow
    #node[residual conv encoder\ squeeze-and-excitation blocks]
    #arrow
    #node[learned frequency-attention pool]
    #arrow
    #grid(columns: (1fr, 1fr, 1fr), gutter: 4pt,
      node(fill: rgb("#dce8f5"))[*BiGRU*\ SimpleConvV2],
      node(fill: rgb("#dce8f5"))[*Transformer*\ head],
      node(fill: rgb("#dce8f5"))[*causal GRU-128*\ streaming])
    #arrow
    #node[4 speeds per frame, PIT-MSE]
    #v(0.3em)
    #small[The shared trunk and the three winner heads kept for the paper.]
  ],
)

// ---------------------------------------------------------------------
= Spectral front-ends

#fig("assets/frontends_render.png", 6.4cm)
#v(0.4em)
#small[
One cruise validation clip through the three front-ends.
*Log-magnitude*: the default; comb teeth blurred by window leakage.
*IF deviation*: per-bin instantaneous-frequency offset in fractional bins —
phase information the magnitude discards; zero-mean on noise, structured on
tones. *Synchrosqueezed magnitude*: bin power reassigned to the frequency
where its phase actually advances — leakage collapses back into sharp
ridges, one channel. Best per architecture: IF for SimpleConvV2, magnitude
for the Transformer, synchrosqueezed for the causal GRU.
]

// ---------------------------------------------------------------------
= The leaderboard: every family on one protocol

#small[
#table(
  columns: (auto, auto, auto, auto, auto, auto),
  stroke: 0.5pt + luma(200),
  inset: 5.5pt,
  align: (left, right, right, right, right, left),
  table.header([*Method*], [*zero*], [*ramps*], [*cruise*], [*all*], [*cost /audio-s*]),
  [NMF (best training-free)], [83.8], [59.5], [8.1], [24.6], [0.1 CPU-s],
  [multi-F0 wide grid (best salience)], [48.2], [16.1], [4.7], [11.7], [2 CPU-s],
  [blind tracker, no gates], [79.4], [39.1], [*2.27*], [17.0], [9.9 CPU-s],
  [blind tracker, gates, refusal $arrow.r$ 0], [*0.01*], [29.8], [48.4], [39.7], [9.9 CPU-s],
  [SimpleConvV2 (BiGRU), R2 real-only], [3.4], [4.2], [2.35], [2.72], [0.25 CPU-s],
  [SimpleConvV2, R4 comb curriculum], [*2.9*], [*3.5*], [2.49], [*2.68*], [0.25 CPU-s],
  [Transformer head, R2], [5.5], [5.1], [2.65], [3.35], [0.25 CPU-s],
  [causal GRU-128, R2], [6.0], [8.2], [3.06], [4.12], [0.25 CPU-s],
)
]
#v(0.5em)
#cbox[
  PIT MAE, rev/s. The regression models win every column except cruise,
  where the gate-free blind tracker stays ahead (2.27 vs 2.35) at
  $tilde$40$times$ the cost. The best row is the comb-curriculum
  convolutional model; only the gated tracker and the full-envelope
  models decide silence correctly.
]

// ---------------------------------------------------------------------
= Outputs: stopped rotors

#grid(
  columns: (2fr, 1fr),
  gutter: 10pt,
  fig("assets/qual_zero.png", 11.8cm),
  small[Rumble-heavy silence clip. The R2 model holds zero. NMF floors at its grid boundary, the salience model lights up on the rumble, and the gate-free tracker locks onto false ridges (the gates decode this window to zero).],
)

// ---------------------------------------------------------------------
= Outputs: start/stop transition

#grid(
  columns: (2fr, 1fr),
  gutter: 10pt,
  fig("assets/qual_transition.png", 11.8cm),
  small[Ramp clip. The best model follows the ramp through the comb collapse. The salience model stays near its floor, NMF answers noise below its grid, and the blind tracker holds a false cruise lock until the true comb appears.],
)

// ---------------------------------------------------------------------
= Outputs: cruise

#grid(
  columns: (2fr, 1fr),
  gutter: 10pt,
  fig("assets/qual_cruise.png", 11.8cm),
  small[Cruise clip. All four track; the differences are precision: blind tracker 2.27, best neural 2.35, NMF 8.1 rev/s cruise MAE on the full split.],
)

// ---------------------------------------------------------------------
= Why the frequency-scale augmentation is mandatory

#fig("assets/freqscale_illustration.png", 6.0cm)
#v(0.4em)
#cbox[
  Models trained without it do not read frequency: a genuine 2%
  comb-spacing shift moves their prediction by 0.03--0.06% — they answer
  from loudness and the speed prior instead of the comb. The
  label-transforming frequency-scale augmentation destroys that shortcut
  and forces the model onto the harmonic structure.
]

// ---------------------------------------------------------------------
= Synthetic data, measured: generator vs comb

#grid(
  columns: (1fr, 1fr),
  gutter: 10pt,
  [
    #fig("assets/real_vs_gen_dregon.png", 5.4cm)
    #small[Neural generator output vs real DREGON noise: the timbre gap
    is audible and visible.]
  ],
  [
    #fig("assets/static_comb_vs_generator.png", 5.4cm)
    #small[The analytic comb: guaranteed harmonic structure at exact
    labels — no generator training, no timbre modeling.]
  ],
)
#v(0.4em)
#cbox[
  Pre-training on synthetic noise (R3), then fine-tuning on real data,
  was the best recipe of the July campaigns. The question the wrap-up
  answers: does it survive an equally rich *real* regime?
]

// ---------------------------------------------------------------------
= Synthetic data vs the full-envelope real regime

#small[
#table(
  columns: (auto, auto, auto, auto, auto),
  stroke: 0.5pt + luma(200),
  inset: 5.5pt,
  align: (left, right, right, right, right),
  table.header([*Run (SimpleConvV2 trunk)*], [*zero*], [*ramps*], [*cruise*], [*all*]),
  [R2 real-only], [3.4], [4.2], [*2.35*], [2.72],
  [R3 generated+comb curriculum], [3.9], [4.3], [2.49], [2.90],
  [R4 comb-only curriculum], [*2.9*], [*3.5*], [2.49], [*2.68*],
  [R5 mixed one-stage], [16.1], [9.1], [5.20], [7.11],
)
#v(0.5em)
#table(
  columns: (auto, auto, auto, auto),
  stroke: 0.5pt + luma(200),
  inset: 5.5pt,
  align: (left, right, right, right),
  table.header([*Aggregate (val MSE)*], [*SimpleConvV2*], [*Transformer*], [*causal GRU*]),
  [R2 real-only control], [22.1], [*41.8*], [61.9],
  [R3 curriculum on R2], [22.6], [40.5], [41.8],
  [R4 comb-only on R2], [*17.6*], [42.6], [*37.6*],
  [R5 mixed on R2], [147.6], [59.2], [85.8],
)
]
#v(0.4em)
#cbox[
  The comb-only curriculum is the campaign's best recipe: 17.6 for
  SimpleConvV2 and 37.6 for the causal GRU, against real-only controls
  of 22.1 and 61.9. The neural-generator curriculum never beats it, and
  mixed one-stage training loses on all three trunks.
]

// ---------------------------------------------------------------------
= The verdict on generated data

#v(1fr)
#cbox(fill: luma(240))[
  #text(size: 1.05em)[
  - Synthetic pre-training pays exactly for its *coverage*: guaranteed
    harmonic structure on exact labels across the full speed envelope.
    Realism contributes nothing measurable.
  - The coverage needs no learned model: the closed-form comb curriculum
    gives the campaign's best cells (SimpleConvV2 17.6, causal GRU 37.6)
    while the neural-generator curriculum never beats it anywhere.
  - Mixing synthetic data into the real pool acts as label noise and
    degrades every architecture; staging is mandatory wherever synthetic
    data is used at all.
  - The trained generator survives nowhere: its curriculum ties the
    real-only control on the strong trunks and loses to the comb on the
    weak one.
  ]
]
#v(1fr)
