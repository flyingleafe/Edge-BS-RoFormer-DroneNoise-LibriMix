#import "@preview/touying:0.7.4": *
#import "/writing/templates/typst/slides.typ": hns-slides

// Notes-visible build: `make notes` (typst --input notes=1). Default build is clean.
#let notes-mode = sys.inputs.at("notes", default: none)

#show: hns-slides.with(
  title: [Progress report: RPS prediction and drone noise suppression],
  subtitle: [Surprising revisit into baselines + some interesting results in RPS prediction],
  author: [Dmitrii Mukhutdinov],
  date: [2026-07-27],
  show-notes: if notes-mode != none { bottom } else { none },
)

#let good(body) = text(fill: rgb("#1f77b4"), weight: "bold", body)
#let bad(body) = text(fill: rgb("#d62728"), weight: "bold", body)

= Noise suppression baselines

#text(size: 0.9em)[
  #table(
    columns: (auto, 1fr, auto, 1.1fr, auto),
    align: (left, left, center, left, left),
    stroke: 0.5pt + luma(180),
    inset: 5.5pt,
    table.header([*model*], [*architecture*], [*year*], [*why it is here*], [*train loss*]),
    [*MP-SENet*],
    [parallel magnitude + phase decoders, conformer trunk (1.71 M)],
    [2023],
    [modern strong SE baseline],
    [SI-SDR + MR-STFT],

    [*TF-GridNet*#footnote[Did not fully train - training hits cluster time limits / there are some bugs with resuming. Results are lower-bounds]],
    [dense full-band + sub-band dual-path RNN (8.38 M)],
    [2023],
    [modern strong SE baseline],
    [SI-SDR + MR-STFT],

    [*Edge-BS-RoFormer*],
    [band-split transformer with rotary attention],
    [2025],
    [Paper 1's own model on this task],
    [SI-SDR + MR-STFT],

    [*DCUNet*],
    [complex-valued U-Net, complex ratio mask (2.81 M)],
    [2019],
    [winner of our 2023 12-model drone survey],
    [SI-SDR + MR-STFT],

    [*SGMSE+*#footnote[Only one iteration on training setup complete. Diffusion training is hard. Results are not ready.]],
    [score-based diffusion, NCSN++ backbone (65 M)],
    [2022],
    [generative contrast to the discriminative four],
    [denoising score matching],
  )
]



#v(0.4em)

// #text(size: 0.85em)[
//   Two extra DCUNet arms probe the loss: pure SI-SDR, and MR-STFT weighted
//   #sym.times\0.05. Both land within 1 dB of the composite.
// ]

#speaker-note[
  Five architectures, all trained by us on the same stream. DCUNet is the old
  one and the one this deck is about. SGMSE+ is the negative control: a 65 M
  diffusion model from scratch needs ~100x our compute budget.
]

= The data: online mixing, held-out noise

#grid(
  columns: (1.15fr, 1fr),
  gutter: 1.2em,
  text(size: 0.88em)[
    *Training stream* (mixed fresh every sample)
    #v(0.2em)
    #table(
      columns: 2,
      align: (left, left),
      stroke: 0.5pt + luma(180),
      inset: 5pt,
      [draw noise], [1 s from a weighted pool],
      [draw speech], [LibriSpeech train-clean-100, 221 of 246 speakers],
      [draw SNR], [$cal(U)(-30, 0)$ dB],
      [mix], [scale noise to that SNR, add],
      [augment], [$p=0.5$: random gain #sym.plus.minus\6 dB, or polarity flip],
    )
    #v(0.5em)
    *Two noise pools = the two passes*
    #v(0.2em)
    #table(
      columns: 2,
      align: (left, left),
      stroke: 0.5pt + luma(180),
      inset: 5pt,
      [*A --- drone only*],
      [DREGON + Michael's real drone flights, plus
        `drone_audio`, DroneAudioSet, SPCUP19-egonoise, new-drone-noises],

      [*B --- all harmonic*],
      [pool A, plus MIMII, MIMII-DG, AeroSonicDB
        (aircraft), motors (HUST + KAIST), horns --- uniform over the 6 categories],
    )
  ],
  text(size: 0.88em)[
    *Validation --- fixed, published, pinned*
    #v(0.2em)
    #table(
      columns: 2,
      align: (left, left),
      stroke: 0.5pt + luma(180),
      inset: 5pt,
      [`SE-valid-drone`], [350 mixtures, drone noise],
      [`SE-valid-harmonic`], [2100 mixtures, 6 categories],
      [holdout], [noise recordings/shards *and* 25 speakers never in training],
      [SNR grid], [−30, −25, −20, −15, −10, −5, 0 dB, #sym.gt.eq\50 clips each],
    )
    #v(0.5em)
    *Training parameters*
    #v(0.2em)
    #table(
      columns: 2,
      align: (left, left),
      stroke: 0.5pt + luma(180),
      inset: 5pt,
      [audio], [16 kHz mono, 1 s chunks],
      [batch], [16],
      [epoch], [20 000 samples #sym.approx 1250 steps],
      [optimiser], [AdamW, lr $10^(-3)$],
      [LR schedule], [#sym.times\0.5 on plateau, patience 15],
      [early stop], [patience 30, cap 150 epochs],
      [monitor], [SI-SDR on `SE-valid-drone`],
    )
  ],
)

#speaker-note[
  The point of this slide is the holdout line: no noise recording in validation
  appears in training, and no speaker either. Epoch length and the patience
  numbers were chosen to match the 2023 survey's schedule so DCUNet gets the
  training length it originally won under.
]

= Pass A (drone-only pool) --- results per SNR

#align(center, image("assets/f1_persnr_a.png", width: 100%))

#align(center, text(size: 0.82em)[
  `SE-valid-drone`, 345 clips. MP-SENet and TF-GridNet were compute-limited ---
  their curves are lower bounds. SGMSE+ Pass A was not evaluated.
])

#speaker-note[
  Left: output SI-SDR against the do-nothing line. Right: output eSTOI, also
  against do-nothing. Ordering is the same on both metrics: MP-SENet, then
  TF-GridNet and Edge-BS-RoFormer, then DCUNet. DCUNet is the only model below
  the input eSTOI at every SNR, and above −5 dB it is below the input on SI-SDR
  too.
]

= Pass B (all-harmonic pool) --- results per SNR

#align(center, image("assets/f1_persnr_b.png", width: 100%))

#align(center, text(size: 0.82em)[
  Same valid set, same colours. SGMSE+ from scratch sits far below the input at
  every SNR --- a compute-bounded negative control, not a usable baseline.
])

#speaker-note[
  Widening the training noise to all harmonic families does not change the
  ordering. It costs the three stronger models (Edge-BS-RoFormer most: eSTOI
  0.344 to 0.302 at −15 dB) and slightly helps DCUNet on SI-SDR. Under a fixed
  budget, breadth dilutes.
]

= Why is DCUNet so bad here?

#grid(
  columns: (1fr, 1fr),
  gutter: 1.4em,
  [
    *What prior work reports*
    #v(0.3em)
    #text(size: 0.88em)[
      - 2023 IEEE Access survey, at −15 dB: SI-SDR *+3.7 dB*, eSTOI *0.4* ---
        best of twelve models.
      - Paper 1 / DN-LM: DCUNet ranks *1st of four* on SI-SDR and STOI.
    ]
  ],
  [
    *What we measure on held-out noise*
    #v(0.3em)
    #text(size: 0.88em)[
      - `SE-valid-drone`: DCUNet *last of five*, SI-SDR *−10.88 dB*.
      - eSTOI #bad[0.193] against an unprocessed #good[0.233] --- it *removes*
        intelligibility.
    ]
  ],
)

#v(1em)

#align(center, text(size: 1.1em)[
  Either our pipeline is broken, or the benchmarks are not\
  measuring the same thing.
])

#speaker-note[
  This contradiction is what started the investigation. Two prior results in
  this project's own lineage say DCUNet is excellent; our held-out benchmark
  says it is the worst of five.
]

= The survey's number reproduces exactly

#text(size: 0.92em)[
  Same model, loss, crop, SNR range and schedule as the survey --- and, as the
  survey does, the *same five ego-noise recordings for training and validation*.
]

#v(0.4em)

#align(center, table(
  columns: 4,
  align: (left, center, center, center),
  stroke: 0.5pt + luma(180),
  table.header([*at −15 dB input*], [*do nothing*], [*ours*], [*published*]),
  [SI-SDR (dB)], [−15.05], [*+3.82*], [+3.7],
  [eSTOI], [0.126], [*0.408*], [0.4],
))

#v(0.3em)

#align(center)[So the pipeline is fine. The difference is in the data.]

#speaker-note[
  SI-SDR and eSTOI, the two metrics carrying the claim, land on the published
  values. PESQ does not replicate — short by about 0.36 even after correcting
  wideband versus narrowband scoring — and that residual is unexplained, so
  PESQ is not shown here.
]

= Same drone, one session held out

#align(center, image("assets/seen_unseen.png", width: 90%))

#align(center, text(size: 0.85em)[
  Train on AVQ session 1 only, score both halves. Everything else identical.
])

#speaker-note[
  At −15 dB: +3.60 dB on the session it trained on, −9.30 dB on the session it
  never heard. eSTOI 0.339 versus 0.168. Red barely leaves the do-nothing line.
]

= The control: session 2 is not the harder half

#align(center, image("assets/control.png", width: 92%))

#align(center, text(size: 0.85em)[
  Unprocessed the halves are identical: SI-SDR −14.93 vs −14.97, eSTOI 0.114 vs
  0.122, 250 clips each.
])

#speaker-note[
  Left panel is the model trained on all five recordings: it scores the two
  halves within 0.3 dB. Right panel is the held-out model: 12.9 dB apart.
  Without this control the whole claim is open to "session 2 is just harder".
]

= Breadth is the same effect, diluted

#align(center, image("assets/ladder.png", width: 100%))

#align(center, text(size: 0.85em)[
  AVQ share of training 100% #sym.arrow 14% #sym.arrow 2%.
  The unseen half (−9.30 dB) lands on the broad-pool arms (−9.64 dB).
])

#speaker-note[
  ΔeSTOI falls +0.276, +0.006, −0.002 while ΔSI-SDR survives. One effect, not
  two: what governs the score is exposure to the specific recording under test.
]

= An architecture that was never shown this drone

#grid(
  columns: (1.1fr, 1fr),
  gutter: 1em,
  align(center + horizon, image("assets/mpsenet.png", width: 96%)),
  align(horizon, text(size: 0.8em)[
    F1's drone pool contains *no AVQ audio at all* --- MP-SENet had never heard
    this drone.
    #v(0.5em)
    At −15 dB, ΔeSTOI: *MP-SENet +0.342* vs *DCUNet +0.046*.
    #v(0.5em)
    MP-SENet generalises to an *unheard* drone better than DCUNet generalises
    between two sessions of a drone it *trained on*.
  ]),
)

#speaker-note[
  This localises the limitation to the architecture rather than to the task —
  the problem is learnable. What it does not show is that MP-SENet would resist
  memorisation under narrow training; that was never tested, there is no
  narrowly-trained MP-SENet checkpoint.
]

= What the outputs look like --- −15 dB input

#align(center, image("assets/f1_spec_1.png", width: 88%))

#speaker-note[
  One `SE-valid-drone` clip, Pass A checkpoints, each panel peak-normalised and
  on the same dB scale. The rotor comb is the horizontal banding in the noisy
  panel. MP-SENet and TF-GridNet recover visible harmonic speech structure;
  DCUNet removes the comb but leaves little speech behind.
]

= What the outputs look like --- −5 dB input

#align(center, image("assets/f1_spec_2.png", width: 88%))

#speaker-note[
  Same four models on an easier clip. At −5 dB every model has speech to work
  with, and the difference is how much of it survives suppression.
]

= What is not established

- *Why* DCUNet specifically fails to generalise. Capacity (2.81 M params), the
  complex-mask formulation, no cross-band mixing --- all untested candidates.
- Whether Edge-BS-RoFormer or MP-SENet would memorise under equally narrow
  training. Not tested --- there is no narrowly-trained checkpoint for either.
- Whether the *released* DN-LM data matches its published description (neither
  data nor builder is shipped).
- *The published papers are not wrong.* They measure performance on seen noise;
  we measure it on unseen noise. Both are correct within their protocols.

#speaker-note[
  Being explicit about the limits matters more than the positive result here,
  because the headline is a claim about someone else's benchmark.
]

=

#align(center + horizon)[
  #text(size: 1.6em, weight: "bold")[Part 2]
  #v(0.4em)
  #text(size: 1.1em)[From SE baselines to the RPS-parity campaign]
  #v(1em)
  #text(size: 0.95em, fill: luma(90))[
    Predicting rotor speed from audio: augmentations, architecture, results
  ]
]

#speaker-note[
  Part 1 was a leakage audit of published SE numbers. Part 2 is a different
  problem in the same project: predicting rotor speed (RPS) from audio alone
  — needed because ground-truth telemetry is not available at inference
  time. Three parts: how we augment the training stream, a new
  architecture (CKLA) built around the idea that the target is a rotating
  phasor rather than a generic sequence, and where the numbers stand today.
  The results near the end of this part are marked WIP — training runs are
  in flight and the numbers will move.
]

#let wip(body) = align(top + right, box(
  fill: rgb("#d62728"),
  inset: (x: 0.5em, y: 0.3em),
  radius: 3pt,
)[#text(fill: white, size: 0.7em, weight: "bold")[#body]])

= Augmentations — the base set

For weeks, mixture-level augmentations ran post-mix, on the `(mixture,
target)` pair.

#v(0.3em)

#align(center, table(
  columns: 2,
  align: (left, left),
  stroke: 0.5pt + luma(180),
  inset: 6pt,
  table.header([*transform*], [*what it does*]),
  [`random_gain`], [scale the pair by a random ±6 dB gain],
  [`random_polarity`], [flip the sign of the whole pair],
  [`channel_drop`], [zero 1+ mic channels in both mixture and target],
  [`noise_time_warp`], [resample the noise (and its RPS label) by a slowly time-varying rate, ±12%],
))

#speaker-note[
  These four are the ones that have run from the start of training in every
  real-data run this campaign: applied post-mix, with the same draw hitting
  mixture and target so SI-SDR-style losses stay well-defined. Only
  `noise_time_warp` touches the RPS label (it changes the physical rotor
  trajectory); the other three are either invisible to a magnitude/IF
  front-end (`random_polarity`) or a bulk transform that leaves the comb
  spacing untouched (`random_gain`, `channel_drop`).
]

= Augmentations — the base set, visually

Clean vs. each base transform on the same clip — spectrogram (top) and the 4
motor-speed tracks (bottom). Same axes and the same fixed dB scale throughout,
so the two rows of every column line up exactly.

#align(center, image("assets/aug_base_grid.png", width: 100%))

#speaker-note[
  Same colour scale, same frequency range, same rev/s range across every
  panel, so any real change is visible without reading numbers off an axis.
  `random_gain` and `random_polarity` leave the comb pattern and the RPS
  curves untouched — expected, since they're bulk/sign transforms.
  `channel_drop` shows the *dropped* channel going silent; the RPS label is
  unaffected (dropping a mic doesn't change the physical rotor speed).
  `noise_time_warp` is the interesting one: the comb rides up and down
  slightly and the RPS curves visibly wobble in sync with it — the only
  base-set transform where audio and label move together.
]

= Augmentations — the expanded set

Six noise-level transforms, applied to the noise *before* mixing (two of them
need the RPS label directly).

#v(0.3em)

#align(center, table(
  columns: 2,
  align: (left, left),
  stroke: 0.5pt + luma(180),
  inset: 6pt,
  table.header([*transform*], [*what it does*]),
  [#text(fill: rgb("#d62728"), weight: "bold")[`freq_scale`]],
  [resample noise + RPS pair by α ~ U(0.75, 1.3) — a genuine comb rescale],

  [`spectral_recolor`], [smooth random per-channel EQ curve, ±8 dB],
  [`random_reverb`], [convolve with a synthetic room impulse response],
  [`tooth_dropout`], [zero 1–4 label-aware harmonic bins (rotor, harmonic) pairs],
  [`spec_mask`], [SpecAugment-style random frequency/time blocks zeroed],
  [`floor_inject`], [add colored-noise floor, −20 to 0 dB relative RMS],
))

#speaker-note[
  All six run on the noise-chunk `(audio, RPS)` pair, before speech mixing —
  that's why `freq_scale` and `tooth_dropout` can read/rewrite the label
  directly. `freq_scale` is the one transform that makes comb *spacing*
  load-bearing: after it, amplitude and timbre alone can no longer predict
  the label, only spacing can. The other five are content transforms
  (EQ, reverb, masking, floor) that leave the label untouched.
]

= Augmentations — the expanded set, visually (1/2)

#align(center, image("assets/aug_expanded_grid1.png", width: 96%))

#speaker-note[
  `freq_scale` (highlighted, α=1.2 here for visibility — the real draw is
  α ~ U(0.75, 1.3)) visibly rescales both the comb *and* the RPS curves by
  the same factor, and shortens/pads the clip consistently. That's the
  transform that forces a spacing-reading rather than an amplitude-reading
  strategy. `spectral_recolor` and `random_reverb` change the texture of the
  spectrogram but leave the comb spacing and the RPS labels untouched —
  same physical event, different acoustic "costume."
]

= Augmentations — the expanded set, visually (2/2)

#align(center, image("assets/aug_expanded_grid2.png", width: 96%))

#speaker-note[
  Same clean panel repeated on the left, same axes as the previous slide.
  `tooth_dropout` is visible as a thin dark horizontal band where one
  harmonic tooth got zeroed across the clip — forces the model to read
  redundant teeth rather than lock onto one. `spec_mask` and `floor_inject`
  are milder, broadband perturbations; none of the three touch the RPS
  label.
]

= Predictions under frequency shift

#text(size: 0.85em)[
  A model trained *without* `freq_scale`, on a clip unshifted / shifted 2% /
  shifted 10%. Dashed = ground truth, solid = prediction, same axes throughout.
]

#align(center, image("assets/freqshift_predictions.png", width: 62%))

#align(center, text(size: 0.7em)[
  Why 2%: it stays inside the physically plausible rotor-speed band, so the
  ideal response is unambiguous. 10% is out-of-band, shown only because the
  failure is then obvious to the eye.
])

#speaker-note[
  This is the g2_if transformer, the best model we have that never saw
  `freq_scale` during training. At 2% the ground truth (dashed) visibly
  separates from the unshifted case, but the prediction (solid) barely
  moves — it stays anchored near the mean of the training RPS distribution.
  At 10% the mismatch is unmissable: the comb has clearly moved but the
  prediction curve is nearly flat. This is the failure mode `freq_scale`
  augmentation is meant to fix — forcing the model to read spacing instead
  of amplitude/timbre pattern.
]

= Fixing the augmentation regime moved the score

#wip[WIP — the uniform-v2 run is days old]

#v(0.2em)

#align(center, text(size: 1.0em)[
  #table(
    columns: 3,
    align: (left, center, center),
    stroke: 0.5pt + luma(180),
    inset: 7pt,
    table.header([*same architecture, same data*], [*old regime*], [*uniform v2*]),
    [frequency-scale augmentation], [never fired], [on *every* chunk, $alpha in [0.7, 1.3]$],
    [full-envelope validation MSE], [63.7], [#text(weight: "bold")[42.3]],
    [PIT-MAE over all 37 valid clips (rev/s)], [4.72], [#text(weight: "bold")[3.83]],
  )
])

#v(0.5em)

#align(center, text(size: 0.95em)[*Where the error actually moved* --- PIT-MAE (rev/s), per frame regime])

#v(0.2em)

#align(center, text(size: 0.95em)[
  #table(
    columns: 5,
    align: (left, center, center, center, center),
    stroke: 0.5pt + luma(180),
    inset: 6pt,
    table.header(
      [*per-frame regime*],
      table.cell(colspan: 2)[*DREGON*],
      table.cell(colspan: 2)[*Michael's FLY124*],
    ),
    [], [old], [v2], [old], [v2],
    [zero RPS (GT $approx 0$)], [6.19], [6.41], [30.49], [#good[4.99]],
    [warm-up ($0 <$ RPS $< 50$)], [11.02], [15.83], [5.32], [#good[3.29]],
    [free flight (RPS $gt.eq 50$)], [3.03], [3.24], [2.96], [3.11],
  )
])

#v(0.4em)

#align(center, text(size: 0.95em)[
  Uniform scaling costs *DREGON* everywhere (worst at warm-up, 11.0 -> 15.8)
  and still buys *FLY124* almost everything, especially the idle frames
  (30.5 -> 5.0). The total goes down only because FLY124's gain is larger
  than DREGON's loss.
])

#speaker-note[
  This is the answer to "if the augmentation did not teach the model to follow
  the comb, where did it help — and cost?". Computed here: both checkpoints
  (g2_if_transformer = old, g2_if_freqscale_v2 = uniform v2, freq_scale
  probability 1.0 with alpha in [0.7, 1.3] on every post-warmup chunk) run
  over all 37 valid-full clips, per-frame absolute error after per-clip PIT
  alignment, bucketed by the rotor-mean ground truth of that frame. This is
  a genuinely different picture from the earlier "fires as intended" regime:
  DREGON gets uniformly worse now, not just a little (free 3.03 -> 3.24,
  zero 6.19 -> 6.41, warm-up 11.0 -> 15.8, its worst regression). FLY124
  still improves everywhere it did before, most dramatically the idle
  frames (30.5 -> 5.0) and warm-up (5.3 -> 3.3); free flight is a wash
  (2.96 -> 3.11). Net effect over all 37 clips is still an improvement
  (4.72 -> 3.83) because FLY124 carries more frames and a bigger swing, but
  read this plainly: uniform scaling is not a free lunch, it is a trade
  that currently favours one drone over the other. Frame counts: DREGON
  free 4482, warm-up 196, zero 844; FLY124 free 2349, warm-up 1083,
  zero 333 — the DREGON warm-up cell is the thinnest and the least
  trustworthy, but its direction (worse) is consistent with the free and
  zero cells, so it is not just noise.
]

= The uniform regime partially follows the shift

#align(center, image("assets/freqshift_both.png", width: 62%))

#align(center, text(size: 0.72em)[
  This clip: truth rises 80.2 → 81.9 → 88.3 rev/s. Old regime still drifts the
  *wrong* way (80.1 → 80.1 → 80.2, flat). Uniform v2 moves in the *right*
  direction at every shift (80.8 → 80.9 → 86.2, ×1.067 at 10%) --- across a
  12-clip probe the same v2 checkpoint follows *42%* of a 2% shift and *71%*
  of a 10% shift (the old regime followed ~0% either way).
])

#speaker-note[
  Same probe, both models, same clip. Top row spectrogram, middle row the
  old-regime prediction, bottom row uniform v2. The old regime is unchanged
  from before: it still sits near the training-distribution mean and barely
  moves as truth climbs. Uniform v2 is a real change, not a wash — it moves
  the right way at every shift size, more so the larger the shift, matching
  the 12-clip average computed the same way: 42% of a 2% shift followed, 71%
  of a 10% shift, versus essentially 0% under the old regime. So: "does not
  respond" is no longer the right read — "partially responds, and more so
  the larger the shift" is. Why 2%: it is inside the physically plausible
  rotor-speed band, so the ideal response is unambiguous; 10% is shown
  alongside it only because it is out-of-band and the direction of movement
  is easier to see by eye.
]

=

#align(center + horizon)[
  #text(size: 1.6em, weight: "bold")[Novel architecture attempts]
  #v(0.5em)
  #text(size: 1.05em)[
    Inspired by recent results in iterative blind motor speeds annotation
  ]
]

#speaker-note[
  Section break. Everything after this is architecture work, and the thread
  that leads into it is our own blind re-annotation pipeline: an iterative
  filter that reads rotor speeds out of a recording without telemetry. It
  works, and it is far too slow to run inside a network — so the question is
  whether the same filter can be a layer.
]

= How we do iterative RPS trajectories optimization, recap

#grid(
  columns: (1fr, 1fr),
  gutter: 0.9em,
  align(horizon, image("assets/dia_comb.png", width: 100%)),
  align(horizon, image("assets/dia_vk_loop.png", width: 100%)),
)

#text(size: 0.85em)[
  $ J[a] = sum_t | y(t) - sum_m "Re"[a_m (t) c_m (t)] |^2 + sum_m rho_m^2 norm(Delta^p a_m)^2 $
]

#align(center, text(size: 0.78em)[
  A rotor radiates a comb; the gap between teeth is its speed. Explain the
  recording with a few smooth envelopes, then re-read the spacing from the
  residual, and repeat.
])

#speaker-note[
  Recap of the annotation method, in two pictures. Left: the physical fact —
  the quantity we want is the spacing of the comb, not something hidden.
  Right: Vold–Kalman order tracking as an alternating loop. Fix the frequency
  track, i.e. the phasors c_m(t), and the per-harmonic envelopes a_m(t) fall
  out of one global least-squares problem; rho sets how fast an envelope may
  move. Then re-estimate the track from the residual and repeat. Accurate,
  and our reference bar — but it needs a pitch guess, sees the whole clip at
  once, and is orders of magnitude slower than a forward pass.
]

= Vold–Kalman began life as a Kalman filter

#align(center, image("assets/dia_kf_vs_vk.png", width: 66%))

#v(0.1em)

#align(center, text(size: 0.72em)[
  Vold's 1993 algorithm *was* a sequential Kalman filter plus a backward
  smoother: state = one order's complex envelope under an integrated-
  random-walk prior; the tachometer supplies a known carrier
  $H(n) = [e^(i phi(n)), 0, dots]$ in the measurement row.
])

#v(0.1em)

#align(center, text(size: 0.76em)[
  #table(
    columns: 2,
    align: (left, left),
    stroke: 0.5pt + luma(180),
    inset: 5pt,
    table.header([*1993 recursive solver*], [*modern batch solver (what we run)*]),
    [predict–update, one sample at a time], [banded regularised least squares, one shot over the whole record],

    [Riccati covariance recursion], [$O(N)$ sparse linear solve],

    [one order at a time], [all orders jointly],
  )
])

#v(0.1em)

#align(center, text(size: 0.74em, style: "italic")[
  The model never changed --- only the solver did: recursion → batch.
])

#speaker-note[
  Everyone who uses Vold–Kalman today runs the second-generation, batch
  version — but the name is not a coincidence. Vold's 1993 paper really was
  a Kalman filter: the state was one order's complex envelope, driven by an
  integrated-random-walk process model, and a tachometer-derived phase sat
  in the measurement matrix as the known carrier. It ran forward as a
  recursion and backward as an RTS smoother, sample by sample.

  It was abandoned in the mid-1990s for three reasons: it estimated one
  order at a time, so crossing orders leaked into each other; the covariance
  recursion becomes ill-conditioned exactly in the heavy-smoothing regime
  (Q much less than R) that tonal extraction needs; and there was no clean
  way to make the recursion joint across many orders. The fix that survived
  is the batch version everyone uses now: all orders solved jointly in one
  banded least-squares problem, where the smoothness weight lambda plays
  the role Q/R played before. Under Gaussian assumptions that batch solve
  is mathematically the fixed-interval Kalman *smoother* of the very same
  state-space model. Same model, two solvers — the next slide shows why
  it is the same model even in the choice of process noise.
]

= One stochastic model underneath: the Ornstein–Uhlenbeck process

#align(center, image("assets/dia_ou.png", width: 50%))

#align(center, text(size: 0.66em)[
  $
    x_(n+1) = e^(-gamma Delta t) x_n + w quad "(real)", quad
    z_(n+1) = e^((-gamma + i omega) Delta t) z_n + w quad "(complex)"
  $
])

#align(center, text(size: 0.66em, style: "italic")[
  The complex OU process is the canonical stochastic model of a tone at
  frequency $omega$ with linewidth $gamma$.
])

#align(center, text(size: 0.7em)[
  #table(
    columns: 3,
    align: (left, left, left),
    stroke: 0.5pt + luma(180),
    inset: 3pt,
    table.header([*VK-2, batch*], [*Vold 1993, recursive*], [*CKLA, learned*]),
    [demodulate by the known carrier, then globally smooth the residual],
    [carrier in the measurement row, run as a recursion],
    [rotation in the state transition, everything else learned],
  )
])

#speaker-note[
  Both rows of the previous slide, and CKLA, are the same stochastic model
  wearing different clothes. A real Ornstein–Uhlenbeck process is a state
  that fades back toward zero, $x_(n+1) = e^(-gamma Delta t) x_n + w$ — this
  is exactly the state decay in plain linear attention / KLA. A complex OU
  process is a phasor that fades *and* turns,
  $z_(n+1) = e^((-gamma + i omega) Delta t) z_n + w$ — the canonical
  stochastic model of a tone at frequency omega with linewidth gamma: its
  autocovariance is $e^((-gamma+i omega)|tau|)$, a Lorentzian line.

  Three things people built are three factorizations of this one model.
  VK-2 demodulates by a known carrier first and then batch-smooths what's
  left. Vold's 1993 filter kept the rotation inside the measurement row and
  ran a recursion. CKLA puts the rotation inside the state transition itself
  and learns the rest — carrier, bandwidth, everything. Same object, three
  places to put the turning.
]

= Recent work (2026): Attention layer replacement based on Kalman filter

#align(center, image("assets/dia_kla_pipeline.png", width: 74%))

#align(center, text(size: 0.82em)[
  Linear attention rediscovered the Kalman filter --- a decaying state with
  input-dependent, learned gains --- which makes the 1993-style recursion
  attractive again: differentiable, causal, streaming.
])

#align(center, text(size: 0.82em)[
  #table(
    columns: 3,
    align: (center, left, left),
    stroke: 0.5pt + luma(180),
    inset: 5pt,
    table.header([*per token*], [*is*], [*in plain words*]),
    [$macron(a)$], [state decay], [how much of the old belief survives one frame],
    [$macron(p)$], [process noise], [how much the state is allowed to drift on its own],
    [$phi = k^2 lambda_v$], [evidence precision], [how much this frame is worth trusting],
    [$kappa = k lambda_v v$], [evidence], [what this frame actually claims],
  )
])

#speaker-note[
  This is Kalman Linear Attention, KLA, from recent sequence-model work: a
  bank of little Kalman filters, one per state slot, instead of an attention
  matrix. Every quantity on the table is produced by the network from the
  token — k is the key, v the value, lambda_v a learned per-token confidence.
  So the network does not compute an output directly; it computes what the
  evidence is and how much to believe it, and the filter does the rest.
]

= The recursion, and how it is read out

#v(1fr)

#align(center, text(size: 1.15em)[
  $
    lambda arrow.r lambda / (macron(a)^2 + macron(p) lambda) + phi, quad
    eta arrow.r macron(a) eta / (macron(a)^2 + macron(p) lambda) + kappa
  $
  #v(0.7em)
  $ mu = eta / lambda, quad y = sum q mu $
])

#v(0.7em)

#align(center, text(size: 0.95em)[
  In words: fade what you believed, add what this frame is evidence for,
  divide to read out the estimate --- and the query reads it out *gated by the
  layer's own confidence*.
])

#v(1.4fr)

#speaker-note[
  Two lines carry the whole layer. Lambda is precision (how sure the layer is),
  eta is information; both are pushed through the same decay denominator, then
  the new evidence is added. Their ratio mu is the state estimate, and the
  query q reads it out. The precision is explicit, not implicit — an attention
  layer has no equivalent of "I am unsure here". Note this recursion is affine
  in (lambda, eta), which is what makes the next slide's parallel training
  possible.
]

= This replaces attention

#align(center, image("assets/dia_block_swap.png", width: 82%))

#align(center, text(size: 0.9em)[
  Same residual and MLP scaffolding --- only the mixer changes. The recursion is
  associative, so it runs as a parallel prefix (Möbius) scan: linear in
  sequence length, no Python loop at training time.
])

#speaker-note[
  The point of the slide: KLA is a drop-in. Norms, residual, MLP all stay; the
  self-attention sub-layer is swapped for the filter bank. And the cost story
  is favourable — the update is a Möbius transform of the previous state, and
  Möbius transforms compose, so the whole sequence can be resolved by an
  associative scan in log-depth rather than sequentially.
]

= Why the state should rotate

#align(center, image("assets/dia_phasor.png", width: 72%))

#align(center, text(size: 0.85em)[
  A rotor harmonic is not a fading number --- it is a phasor going round at a
  rate set by the rotor speed. A real state can hold its loudness but not its
  phase.
])

#text(size: 0.7em, style: "italic")[This is the complex-OU process from three slides back, put inside a layer.]

#speaker-note[
  Motivation for our change. Plain KLA's state decays: it can remember how
  loud something was, not where in its cycle it is. The physical latent here
  literally rotates. If the memory rotates with it, one state slot can lock
  onto one harmonic and stay coherent with it over many frames instead of
  smearing.
]

= CKLA: one step fades *and* turns

#v(1fr)

#align(center, text(size: 1.15em)[
  $ macron(a) = e^((-gamma + i omega) Delta t) $
  #v(0.6em)
  $ Sigma arrow.r |macron(a)|^2 Sigma + macron(p) $
])

#v(0.5em)

#align(center, text(size: 0.95em)[
  Top: $gamma$ fades the state, $omega$ turns it --- both predicted per frame.\
  Bottom: rotation is unitary, so only $|macron(a)|^2$ enters the variance.
])

#v(0.5em)

#align(center, text(size: 0.9em)[
  The precision recursion is *identical* to plain KLA's. Complex tracking costs
  the same; only the information vector picks up a unit-modulus multiplier.
])

#v(1.4fr)

#speaker-note[
  Make the state complex and the single transition does both jobs. The reason
  this is cheap: second moments do not see a rotation, so the precision
  algebra stays real and unchanged — exactly the recursion from two slides
  ago, with |a-bar| in place of a-bar. That is the whole trick. We call it
  CKLA, complex KLA.
]

= The network steers the filter

#align(center, image("assets/ckla_block_diagram.png", width: 88%))

#align(center, text(size: 0.78em)[
  #table(
    columns: 2,
    align: (left, left),
    stroke: 0.5pt + luma(180),
    inset: 5pt,
    table.header([*1993 filter needed...*], [*CKLA answer*]),
    [a tachometer for the carrier], [$omega_t$ predicted by the network --- the network is the tachometer],

    [a fixed bandwidth $lambda$ / ill-conditioned $Q$-$R$],
    [per-token learned gains --- the streaming twin of iterative bandwidth adaptation],

    [one order at a time (crossings leak)], [many complex state channels + a learned query readout],
  )
])

#speaker-note[
  This is what separates CKLA from a hand-built complex-OU Kalman tracker,
  and it maps directly onto the three reasons the 1993 filter was
  abandoned. It needed a tachometer to supply the carrier — here omega_t is
  predicted from the token, so the network *is* the tachometer. It needed a
  fixed smoothing bandwidth that became ill-conditioned exactly where you
  want heavy smoothing — here the gain is learned per token, the streaming
  analogue of the bandwidth-adaptation tricks modern adaptive VK variants
  use offline. And it could only track one order at a time, so crossings
  leaked — here there are many complex state channels running in parallel,
  read out by a learned query, so channels do not have to share.
]

= Why it must be steerable: phase error accumulates

#align(center, image("assets/dia_phase_error.png", width: 80%))

#align(center, text(size: 0.82em)[
  A fixed, slightly wrong rate de-coheres: the error grows without bound while
  the filter's confidence grows too. Illustration, not a measurement.
])

#align(center, text(size: 0.88em)[
  Stacking is the counter-measure --- an early block locks coarsely, later
  blocks condition their rotation on that belief and correct it.
])

#align(center, text(size: 0.78em, style: "italic")[
  Recursive (1993) → batch (1997–today) → recursive again, with the
  unknowns learned.
])

#speaker-note[
  The failure mode of any fixed-rate tracker. A few percent of rate error is
  not a bounded offset; it integrates, and once the phase error passes half a
  cycle the filter is coherently summing noise. Worse, precision only grows
  with more observations, so it reports high confidence while doing it. Two
  panels, one message: the mistake and the self-assurance grow together. The
  architectural answer is depth — capture first, refine after, instead of
  committing to one rate for the whole clip.

  And that's the arc of this whole section: Vold ran a recursion in 1993,
  the field moved to a batch solve because the recursion couldn't scale to
  many orders, and now we're back to a recursion — but this time the
  carrier, the bandwidth and the per-order separation are all learned
  instead of assumed.
]

= Results — full-envelope validation

#wip[WIP — runs in flight, no conclusions]

#v(0.2em)

#align(center, table(
  columns: 3,
  align: (left, center, left),
  stroke: 0.5pt + luma(180),
  inset: 7pt,
  table.header([*arm, uniform freq-scale v2*], [*val/MSE*], [*note*]),
  [transformer (no CKLA)], [42.3], [matched baseline, same v2 regime],
  [CKLA (gain-fixed), current run], [*41.4*], [still improving],
))

#speaker-note[
  Full-envelope validation MSE, lower is better. Both arms now trained under
  the uniform freq-scale v2 regime (probability 1.0, alpha in [0.7, 1.3] on
  every post-warmup chunk) — internally the CKLA arm is `ckla_pnoise_fs_v2`.
  This is a like-for-like comparison, not the old protocol-B snapshot: CKLA
  edges the matched transformer (41.4 vs 42.3), but both numbers are the
  best checkpoint of runs that have not finished improving, so treat the
  margin as noise until it replicates.
]

= Results — cruise pools

#wip[WIP — updating for uniform v2; numbers below are the previous arm]

#v(0.2em)

#align(center, table(
  columns: 3,
  align: (left, center, center),
  stroke: 0.5pt + luma(180),
  inset: 7pt,
  table.header([*pool (PIT-MAE, rev/s)*], [*CKLA, protocol-B (pre-v2)*], [*for reference*]),
  [DREGON cruise], [2.75], [neural floor 2.481 · VK 0.68–0.74],
  [FLY124 cruise], [1.25], [neural floor 2.33 · VK 1.027],
))

#text(size: 0.78em)[
  PIT-MAE = mean abs. error after permutation-aligning predicted rotor tracks
  to ground truth; neural floor = best prior transformer arm. The current
  checkpoint (`ckla_pnoise_fs_v2`) has not been run over these cruise pools
  yet: do not read these as its numbers.
]

#speaker-note[
  These two numbers are carried over from the earlier gain-fixed / protocol-B
  CKLA arm, computed before the uniform freq-scale v2 regime existed — kept
  here only to show scale against the neural floor and the Vold-Kalman blind
  bars, not as a claim about the current best checkpoint (`ckla_pnoise_fs_v2`).
  That model's cruise-pool accuracy is still being evaluated; do not present
  2.75 / 1.25 as if they were fs_v2's numbers. No conclusion is being drawn
  on this slide until the re-eval lands.
]

= Results — one prediction, one clip

#wip[WIP — snapshot from the pre-v2 arm, illustration only]

#align(center, image("assets/ckla_prediction_overlay.png", width: 44%))

#speaker-note[
  This snapshot is still the earlier gain-fixed / protocol-B checkpoint, one
  DREGON cruise clip covering a takeoff ramp: dashed is ground truth, solid
  is the prediction, permutation-aligned. It has not been regenerated for
  `ckla_pnoise_fs_v2` — shown only to illustrate what the model's output
  shape currently looks like, not as a quantitative claim about the current
  best checkpoint.
]

= The bar: a model that follows the comb

#wip[WIP — single-seed; replicates + cruise eval running]

#align(center, image("assets/ckla_freqshift.png", width: 56%))

#align(center, text(size: 0.62em, fill: luma(90))[
  This one clip: pred ×1.095 at the 10% shift, close to ideal --- the table
  below is the 12-clip probe average, which is what the following
  percentages are computed from.
])

#v(0.2em)

#align(center, text(size: 0.66em)[
  #table(
    columns: 4,
    align: (left, center, center, center),
    stroke: 0.5pt + luma(180),
    inset: 3.5pt,
    table.header([*model*], [$times 1.02$], [$times 1.05$], [$times 1.10$]),
    [transformer, old regime], [#sym.times\1.0006 (3%)], [#sym.times\1.0010 (2%)], [#sym.times\1.0024 (2%)],
    [transformer, uniform v2], [#sym.times\1.0084 (42%)], [#sym.times\1.0276 (55%)], [#sym.times\1.0707 (71%)],
    [CKLA, uniform v2], [#sym.times\1.0135 (68%)], [#sym.times\1.0454 (91%)], [#sym.times\1.0992 (99%)],
  )
])

#align(center, text(size: 0.68em)[
  Under uniform scaling the CKLA layer follows 99% of a 10% shift --- it
  tracks the comb, and out-tracks the matched transformer at every shift
  size. Numbers are single-seed (replicates running); cruise-pool accuracy
  vs the VK floor is still being evaluated.
])

#speaker-note[
  This is the criterion the whole architecture line is aimed at: a layer that
  really tracks the comb must move its prediction when the comb moves. Scale-
  response probe on 12 held-out clips, "following" = (ratio − 1) / (ideal
  ratio − 1) against a shift of ×1.02 / ×1.05 / ×1.10. Any pre-v2 training
  regime, transformer or CKLA, sits at the training-distribution mean —
  effectively 0% following at every shift size. Switching to uniform
  freq-scale augmentation (probability 1.0, alpha drawn uniformly in
  [0.7, 1.3], on every post-warmup chunk, not just some) is what breaks
  that amplitude anchor for both arms, and CKLA benefits more at every
  shift size: 68/91/99% of the 2/5/10% shifts versus 42/55/71% for the
  matched transformer under the identical regime. Probe PIT-MAE over the
  same 12 clips: transformer v2 3.20 rev/s, CKLA v2 3.00 rev/s — CKLA is
  also the more accurate of the two here. All numbers are single-seed and
  the checkpoints are not fully converged, so read this as a promising
  direction, not a result: full following of the comb (100%) is still the
  open bar, and cruise-pool accuracy against the Vold-Kalman floor has not
  been re-evaluated for this regime yet.
]

