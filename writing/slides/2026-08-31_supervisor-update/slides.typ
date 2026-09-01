#import "@preview/touying:0.7.4": *
#import "/writing/templates/typst/slides.typ": hns-slides

#let notes-mode = sys.inputs.at("notes", default: none)

#show: hns-slides.with(
  title: [What the rotor-speed models actually read],
  subtitle: [Synthetic families, the fixed fan, and a scoring architecture],
  author: [Dmitrii Mukhutdinov],
  date: [2026-08-31],
  show-notes: if notes-mode != none { bottom } else { none },
)

#let cbox(body, fill: luma(235), stroke-color: luma(160)) = box(
  fill: fill,
  stroke: 0.7pt + stroke-color,
  inset: 8pt,
  radius: 3pt,
  width: 100%,
  body,
)

#let small(body) = text(size: 0.78em, body)
#let tiny(body) = text(size: 0.68em, body)
#let fig(path, h) = align(center, image(path, height: h, fit: "contain"))
#let todo = text(fill: rgb("#b03000"), weight: "bold")[\[TODO verify\]]

// A box in a flow diagram, plus the arrow between two of them.
#let node(body, fill: rgb("#eef2f8")) = box(
  fill: fill,
  stroke: 0.7pt + luma(140),
  inset: 5pt,
  radius: 2.5pt,
  width: 100%,
  align(center, tiny(body)),
)
#let arrow = align(center, tiny(sym.arrow.b))

// ---------------------------------------------------------------------
= Augmentation makes them react; only synthetic data makes them track

#grid(
  columns: (1.35fr, 1fr),
  gutter: 10pt,
  fig("assets/freq_probe_nophase.pdf", 7.6cm),
  [
    #small[
    #table(
      columns: (1fr, auto, auto),
      stroke: 0.5pt + luma(200),
      inset: 5pt,
      align: (left, right, right),
      table.header([*Training regime*], [*near $alpha = 1$*], [*full range*]),
      [no label-transforming augmentation], [0.16], [0.03],
      [augmentation only (R2)], [0.14], [0.89],
      [augmentation + comb pre-training (R4)], [*1.02*], [*1.04*],
      [ideal], [1.00], [1.00],
    )
    ]
    #v(0.4em)
    #tiny[The probe resamples the input by $alpha$: every harmonic, and hence
    every true speed, moves by that factor. Slope 1.00 reads harmonic positions.]
    #v(0.3em)
    #cbox[
      #small[*Augmentation buys a GLOBAL reaction (0.89) and no LOCAL one
      (0.14). Only comb pre-training buys both.*]
    ]
  ],
)

// ---------------------------------------------------------------------
= What we had before stochastic combs

#grid(
  columns: (1.25fr, 1fr),
  gutter: 10pt,
  [
    #fig("assets/families_row.pdf", 6.4cm)
    #tiny[Real drone noise, the neural generator (`gen_m3_refined_all_perrotor`,
    the checkpoint every curriculum run consumes), and the analytic static comb —
    driven by the SAME rotor-speed trajectory, that of a DREGON cruise clip with
    no played source. Each panel is RMS-normalized under one shared dB scale:
    structure is comparable, absolute level is not.]
  ],
  [
    #small[
    #table(
      columns: (1fr, auto, auto),
      stroke: 0.5pt + luma(200),
      inset: 5pt,
      align: (left, right, right),
      table.header([*Arrangement*], [*all-MAE*], [*val PIT-MSE*]),
      [comb stage 1 $->$ real stage 2 (R4)], [*2.67*], [*17.59*],
      [stochastic stage 1 $->$ real stage 2 (R6)], [3.04], [23.78],
      [one stage, stochastic pooled (R7)], [5.99], [94.97],
      [real stage 2 + 29.4% stochastic, warm (R8)], [5.81], [104.85],
      [one stage, generator pooled (R5)], [---], [147.6],
    )
    ]
    #v(0.4em)
    #small[
    #table(
      columns: (1fr, auto),
      stroke: 0.5pt + luma(200),
      inset: 5pt,
      align: (left, right),
      table.header([*Generated data in the pool*], [*val PIT-MSE*]),
      [real training on its own], [$approx 7.3$],
      [generated recordings added to the real pool], [+27% worse],
      [generated-only training], [17.8--25.4],
      [generated-only + a short real fine-tune], [11.1--14.1],
    )
    ]
    #v(0.4em)
    #cbox[
      #small[*Synthetic data bought coverage, never realism — and it had to
      be kept out of the real stage entirely.*]
    ]
  ],
)

// ---------------------------------------------------------------------
= The stochastic comb family

#grid(
  columns: (1.15fr, 1fr),
  gutter: 10pt,
  cbox(fill: rgb("#f6f6f2"))[
    #small[
    $ S(f, t) = B(f, t) + sum_r sum_k P_(r k)(t) dot L(f - k dot "rps"_r (t); gamma_(r k)) $
    $ L(d; gamma) = 1/pi dot gamma / (d^2 + gamma^2) quad "(Cauchy)", quad gamma_(r k) = gamma_(0 r) + s_r dot k $
    $ 10 log_10 P_(r k)(t) = "harm_mean_db" + "profile_db"[r, k] + h_(r k)(t) $
    ]
  ],
  [
    #small[
    - $h_(r k)$ drifts as a squared-exponential Gaussian process, drawn
      *independently of the speed trajectory* — no amplitude carries speed
      information, by construction.
    - The floor $B$ is a smooth random curve in log frequency, with its own GP
      in level and tilt.
    - Line width grows with $k$: a shaft wandering by 0.6 rev/s widens harmonic
      $k$ to about $0.6 k$ Hz.
    - `harm_coherence`: a rotor breathes as a whole;
      `rotor_similarity`: four rotors of one airframe.
    ]
  ],
)
#v(0.3em)
#align(center, image("assets/stoch_samples.pdf", width: 76%, fit: "contain"))
#v(0.15em)
#tiny[The same trajectory in all four panels: the spacing is identical, and the
timbre, the floor and the line breathing are redrawn each time. RMS-normalized,
one shared dB scale.]
#v(0.3em)
#cbox[#small[*Comb spacing stays the only cue, and the texture stops being a
constant — the narrowness that made every static-comb clip sound the same is
gone.*]]

// ---------------------------------------------------------------------
= Results after stochastic-comb training, split by regime

#v(0.8em)
#set text(size: 1.05em)
#table(
  columns: (auto, auto, 1fr, 1fr, 1fr, 1fr),
  stroke: 0.5pt + luma(200),
  inset: 9pt,
  align: (left, left, right, right, right, right),
  table.header([*Model*], [*Trained on*], [*all-MAE*], [*zero*], [*low (ramps)*], [*flight*]),
  [`r4hb_scv2`], [real], [*2.67*], [*2.87*], [*3.48*], [*2.49*],
  [`stoch_s1g_scv2`], [synthetic], [8.08], [20.27], [16.20], [4.50],
  [`m3abl_comb_unigru128_s1`], [synthetic], [8.30], [4.73], [24.24], [6.00],
  [`stoch_s1h_scv2`], [synthetic], [9.07], [27.98], [26.77], [*2.60*],
)
#v(0.5em)
#set text(size: 1em)
#v(0.4em)
#small[PIT MAE in rev/s on the frozen real split
`DREGON-LM-V4-michaels-valid-full`. One row per MODEL — never a per-regime
best-of, which no single model achieves.]
#v(0.4em)
#cbox[
  #small[*Best synthetic-only is 3.03x the real target overall: 7.06x on
  stopped rotors, 4.66x on ramps, 1.81x at cruise. One arm reaches cruise
  parity alone (2.60 against 2.49); none reaches two cells at once.*]
]

// ---------------------------------------------------------------------
= Synthetic-only, trained to convergence

#v(0.4em)
#table(
  columns: (auto, auto, auto, auto, auto),
  stroke: 0.5pt + luma(200),
  inset: 7pt,
  align: (left, right, right, right, left),
  table.header([*Run*], [*params*], [*epochs*], [*val RMSE min*], [*at epoch*]),
  [`stoch_long_scv2`], [1.5M], [229], [*10.304*], [202],
  [`stoch_long_trxxl`], [38M], [157], [13.207], [31],
  [multi-F0 salience on the same family], [---], [---], [--- #todo], [---],
)
#v(0.5em)
#cbox(fill: rgb("#fdeee6"), stroke-color: rgb("#c0714f"))[
  #small[
  *CAVEAT.* These are *half-real, half-synthetic* validation numbers. They are
  NOT comparable to the synthetic-only figures on the previous slide or to the
  real-only figures anywhere else in this deck. Read them only against each
  other.
  ]
]
#v(0.4em)
#small[
- Training at the 8 s scoring length is worth about 8.5% over the 1 s-trained
  predecessor (11.265).
- The 1.5M run was truncated by an MSE-patience rule while its RMSE was still
  descending, so 229 epochs is a *lower bound* on what convergence costs.
- The multi-F0 salience arms on this family are still in flight. #todo
]

// ---------------------------------------------------------------------
= Every model outputs an evenly-spaced fan around the mean

#grid(
  columns: (1.3fr, 1fr),
  gutter: 10pt,
  [
    #fig("assets/fan_panels.pdf", 7.4cm)
    #tiny[Predicted (solid) against true (dotted) per-rotor speeds, PIT-aligned.
    LEFT: true spread 36.1–50.0, predicted 10.5–11.2 (s.d. 0.15), PIT MAE 10.6.
    RIGHT: true spread 7.0–12.6, predicted 8.2–8.7 (s.d. 0.11), PIT MAE 2.20 —
    the centre is right and the spread is still pinned. The LEFT truth is
    redrawn on the rebuilt synthetic benchmark, whose trajectories now carry
    real telemetry's frame-to-frame jitter; the pinned prediction does not
    move. RIGHT is real data and is unchanged.]
  ],
  [
    #small[`stoch_s1id_scv2`, cruise columns, 120 clips:]
    #tiny[
    #table(
      columns: (auto, auto, auto, auto, auto, auto),
      stroke: 0.5pt + luma(200),
      inset: 4.5pt,
      align: (left, right, right, right, right, right),
      table.header([*true spread*], [0.19], [3.96], [8.37], [12.76], [42.75]),
      [*predicted spread*], [8.96], [10.28], [9.59], [10.74], [*10.81*],
      [PIT MAE], [5.04], [5.86], [7.69], [5.29], [17.55],
    )
    ]
    #v(0.4em)
    #tiny[True spread varies over 42.6 rev/s; predicted spread over 1.85. The
    fan settles at $approx 9.4$ rev/s — this generator's own mean spread. The
    static-comb model emits the same pinned fan.]
    #v(0.3em)
    #cbox[
      #small[*The model learned the marginal distribution of the quantity, not
      the signal that determines it.*]
    ]
    #v(0.3em)
    #tiny[*Honest exception — the fan is RIG-dependent.* Swept over all 37
    validation clips, `r4hb_scv2` pins its spread at 8.3–8.6 rev/s on every
    DREGON clip (within-clip s.d. 0.07–0.14). On the FLY124 clips it does follow
    the true spread (within-clip correlation +0.78 to +0.91; clip 27, true
    12.3–27.5, predicted 15.5–19.5). It has learned two per-rig fan widths.
    Synthetic training locks the fan in everywhere; real training escapes it on
    one rig only.]
  ],
)

// ---------------------------------------------------------------------
= The exception: the fan width is per-rig, not universal

#fig("assets/fan_counterexample.pdf", 6.6cm)

#grid(
  columns: (1fr, 1fr),
  gutter: 14pt,
  [
    #small[*DREGON, cruise clip 20* — true spread 7.0–12.6 rev/s, predicted
    8.2–8.7. The spread trace is flat: the prediction ignores a 5.6 rev/s swing
    in the truth. PIT MAE 2.20.]
  ],
  [
    #small[*Michael's FLY124, cruise clip 27* — true spread 12.3–27.5 rev/s,
    predicted 15.5–19.5, and the two traces move together. PIT MAE 1.90.]
  ],
)

#v(0.4em)
#cbox[
  #small[Same model, same weights, one forward pass each. Across all 37
  validation clips `r4hb_scv2` pins its spread at 8.3–8.6 rev/s on *every*
  DREGON clip (within-clip s.d. 0.07–0.14, correlation with truth $-0.63$ to
  $+0.48$), and tracks the true spread on *every* FLY124 clip (correlation
  $+0.78$ to $+0.91$). It has memorized two per-rig fan widths rather than
  learning one universal one — so the fan is what the model falls back on when
  it cannot resolve the rotors, not something it always does.]
]

// ---------------------------------------------------------------------
= Scaling does not touch it

#grid(
  columns: (1.45fr, 1fr),
  gutter: 12pt,
  [
    #tiny[Static comb, three temporal heads trained to saturation:]
    #small[
    #table(
      columns: (auto, auto, auto, auto),
      stroke: 0.5pt + luma(200),
      inset: 5pt,
      align: (left, left, right, right),
      table.header([*Arm*], [*Head*], [*params*], [*floor (val RMSE)*]),
      [`comb_floor_base`], [2 $times$ 64, 4 heads], [0.141M], [2.535],
      [`comb_floor_deep`], [*4* $times$ 64, 4 heads], [0.241M], [*2.155* (#sym.minus 15%)],
      [`comb_floor_wide`], [2 $times$ *128*, 8 heads], [0.479M], [2.968 (+17%)],
    )
    ]
    #v(0.3em)
    #tiny[Predicted spread stays pinned in all three, over true spreads
    0.20--21.04 rev/s: base 10.4--10.7, deep 10.2--10.6, wide 11.5--12.1.
    Depth buys a better CENTRE for the fan, not per-rotor resolution.]
    #v(0.4em)
    #cbox[
      #small[
      Stochastic family, same axis: the 38M `trxxl` sits *28% worse* than the
      1.5M trunk (13.207 against 10.304) and climbs away from its epoch-31
      minimum. Width is harmful on both families; the arm with the most
      parameters loses.
      ]
    ]
  ],
  [
    #tiny[*How the heads were scaled*]
    #v(0.3em)
    #let lyr(w, c) = box(width: w, height: 0.4cm, fill: c, stroke: 0.6pt + luma(140))
    #let tower(n, w, c, cap) = align(center + bottom)[
      #stack(dir: btt, spacing: 3pt, ..range(n).map(_ => lyr(w, c)))
      #v(0.25em)
      #tiny(cap)
    ]
    #grid(
      columns: (1fr, 1.5fr, 1fr),
      gutter: 4pt,
      align: bottom,
      tower(2, 0.8cm, rgb("#dfe8f5"))[2 $times$ 64 \ base],
      tower(2, 1.6cm, rgb("#f5dfe0"))[2 $times$ 128 \ *wider: +17%*],
      tower(4, 0.8cm, rgb("#dfeee2"))[4 $times$ 64 \ *deeper: #sym.minus 15%*],
    )
    #v(0.5em)
    #tiny[Width = key/value/head dimension per time step, quadratic in
    parameters. Depth = sequential processing steps, linear. The widest arm
    carries 3.4x the parameters of base and loses; the deep arm carries 1.7x
    and wins.]
  ],
)
#v(0.3em)
#cbox[#small[*The model is not capacity-limited; it has found a degenerate
solution the loss rewards.*]]

// ---------------------------------------------------------------------
= What the new salience models change, and why

#grid(
  columns: (1fr, 1fr),
  gutter: 12pt,
  [
    #small[*Original multi-pitch models* (HarmoF0, HPPNet, hFT, and our own
    `multif0_salience` baseline)]
    #v(0.2em)
    #node[CQT / log-frequency spectrogram]
    #arrow
    #node[harmonics gathered as a *shift* by $log_2(k) dot B$ bins]
    #arrow
    #node[trunk]
    #arrow
    #node(fill: rgb("#f7e9e6"))[*ONE* salience map, all rotors]
    #arrow
    #node(fill: rgb("#f7e9e6"))[peak-pick + threshold + Hungarian tracking]
    #tiny[Built for music: a semitone grid, sources of different timbre.]
  ],
  [
    #small[*Ours*]
    #v(0.2em)
    #node[*linear* STFT power, $n_"fft"$ 4096, 3.906 Hz bins]
    #arrow
    #node(fill: rgb("#e6f2ea"))[*GATHER*: read at $k dot r$ for every candidate rate]
    #arrow
    #node[same trunk]
    #arrow
    #node(fill: rgb("#e6f2ea"))[*FOUR* layers, one per rotor, zero at bin 0]
    #arrow
    #node(fill: rgb("#e6f2ea"))[one *CRF* best path per layer — no threshold,
    no assignment]
    #tiny[Built for rotors: 0.13 rev/s pairs, four equal timbres.]
  ],
)

// ---------------------------------------------------------------------
= Why each of the three changes

#grid(
  columns: (1fr, 1fr, 1fr),
  gutter: 8pt,
  cbox[#small[*Why a linear STFT.* On a log grid the pair separation-to-bandwidth
  ratio is $D\/(r(2^(1\/B)-1))$ — *$k$ cancels*, so a pair resolves at every
  harmonic or at none. DREGON's 0.13 rev/s pair then needs a 19 s window. A
  uniform STFT separates *linearly with $k$*: 0.13 rev/s from $k>=31$.]],
  cbox[#small[*Why four layers.* One map cannot hold four rotors whose pairs sit
  inside one bin. Encode the true speeds, decode them back: *8.24 rev/s error on
  a PERFECT target*. Four layers: *2.2e-16*. A stopped rotor also becomes a peak
  at bin 0, not an absence.]],
  cbox[#small[*Why the gather.* A comb at $r$ is a *dilation* of one at $r'$, not
  a translation, so convolutions share no weights on a linear axis. Reading *at*
  $k dot r$ restores that sharing by indexing, and keeps the STFT resolution.]],
)

// ---------------------------------------------------------------------
= Current validation scores, new salience models

#grid(
  columns: (1.25fr, 1fr),
  gutter: 12pt,
  [
    #small[`val/rps_mae`, rev/s, PIT-aligned. Each arm validates on the family it
    trained on; *r4* is the real split.]
    #v(0.3em)
    #tiny[
    #table(
      columns: (auto, auto, auto, auto, auto),
      stroke: 0.5pt + luma(200),
      inset: 4pt,
      align: (left, left, right, right, right),
      table.header([*model*], [*arm*], [*epoch*], [*state*], [*rev/s*]),
      table.hline(),
      [*regressor*], [static comb], [—], [converged], [*3.08*],
      [*regressor*], [stochastic],  [—], [converged], [*4.96*],
      [*regressor*], [real (R4)],   [—], [converged], [*2.67*],
      table.hline(),
      [HarmoF0], [static comb], [39], [running],   [*1.43*],
      [HPPNet],  [static comb], [28], [converged], [*1.51*],
      [hFT],     [static comb], [20], [converged], [19.31],
      [HPPNet],  [stochastic],  [27], [converged], [*4.62*],
      [HarmoF0], [stochastic],  [94], [converged], [5.90],
      [hFT],     [stochastic],  [20], [converged], [66.47],
      [HPPNet],  [real (r4)],   [26], [converged], [5.32],
      [HarmoF0], [real (r4)],   [24], [converged], [12.69],
      [hFT],     [real (r4)],   [—], [never ran], [—],
    )
    ]
    #v(0.3em)
    #tiny[*WIP.* Only `hf0_comb` still trains. Regressor rows:
    `m3abl_comb_scv2_s1`, `stoch_s1id_scv2`, `r4hb_scv2`. The two blocks use
    different metrics. hFT's real arm gave no epoch — it did not run. #todo]
  ],
  [
    #cbox[#small[*Both convolutional ports beat the regressors on the two
    synthetic families. On real data they do not.*]]
    #v(0.3em)
    #tiny[*The monitor is wrong.* `val/bce` increases while `val/rps_mae`
    decreases. BCE scores every bin; the metric scores the peak only. Thus early
    stopping stops the runs too soon, and each `best.ckpt` uses the wrong signal.]
    #v(0.3em)
    #tiny[*hFT failed, and it is not undertrained.* Its comb arm stopped at 19.31.
    Its rate tokens read only their own 32 harmonics, so the four layers cannot
    become different. A `gather-as-bias` variant is not yet run.]
    #v(0.3em)
    #tiny[*The r4 rows are not proper R4 cells.* They start from a stage-1
    checkpoint at epoch 6, which has since improved. #todo]
    #v(0.3em)
  ],
)

