#import "@preview/touying:0.7.4": *
#import "/writing/templates/typst/slides.typ": hns-slides

// Notes-visible build: `make notes` (typst --input notes=1). Default build is clean.
#let notes-mode = sys.inputs.at("notes", default: none)


#show: hns-slides.with(
  title: [Progress update],
  subtitle: [Decomposing the drone noise into harmonic and noise components],
  author: [Dmitrii Mukhutdinov],
  date: [2026-08-18],
  show-notes: if notes-mode != none { bottom } else { none },
)

#let cbox(body, fill: luma(235), stroke-color: luma(160)) = box(
  fill: fill,
  stroke: 0.7pt + stroke-color,
  inset: 8pt,
  radius: 3pt,
  body,
)

= Real update

#text(size: 1.3em)[
  - I did not manage to produce the proper paper draft last week, because I found my results insufficient.
  - But also, I feel that maybe I am getting a bit lost in my own results
  - Next slides are some of the latest results on the iterative-estimation track, but maybe I should take a step back, because I again feel that I am not sure what the paper should be about.
]

= Refined labels: did they save the generator?

#v(1.5em)
#text(size: 1.15em)[
  - Raw DREGON telemetry is off by 0.35 to 0.85 %.
  - We fitted refined trajectories with L-BFGS on the Vold–Kalman (VK)
    residual — the tracker's leftover error.
  - Refined trajectories lock onto the acoustic comb, the harmonic pattern
    in the spectrogram.
]
#v(2em)
#align(center, cbox(text(size: 1.3em)[
  *This week's question:* trained on refined labels, does the generator
  learn sharp harmonics all the way up #sym.arrow.r or only somewhere?
]))
#v(1fr)

#speaker-note[
  Quick recap for context: last deck showed the label bias and the refinement
  fix. This deck asks whether that fix was sufficient for the generator, finds
  it was not, explains why with a measured mechanism, and presents the fix:
  train on decomposed amplitudes instead of rendered audio.
]

= Refined labels help, but only up to k#sym.approx 25

#v(1fr)
#grid(
  columns: (1.1fr, 0.9fr),
  gutter: 1.2em,
  align(horizon)[
    #text(size: 0.85em)[
      *Tooth* = a harmonic's peak power minus the local floor, in dB. A
      generator that learns a sharp line shows a large tooth at that k.
    ]
    #v(0.3em)
    #align(center, image("assets/tooth_def.png", width: 78%))
    #v(0.3em)
    #text(size: 0.82em)[
      Self-referenced comb readout by arm (tooth, dB; null #sym.approx
      #sym.minus 0.8, the tooth of a band with no line):
    ]
    #table(
      columns: (auto, 1fr, 1fr, 1fr, 1fr),
      column-gutter: 8pt,
      align: (left, right, right, right, right),
      stroke: 0.5pt + luma(200),
      inset: 5pt,
      text(size: 0.85em)[*arm*], text(size: 0.85em)[*k1-9*],
      text(size: 0.85em)[*k10-24*], text(size: 0.85em)[*k25-49*],
      text(size: 0.85em)[*k50-80*],
      [raw telemetry], [0.44], [2.64], [0.23], [#sym.minus 0.60],
      [scaled (#sym.times 0.99458)], [0.68], [5.40], [1.40], [#sym.minus 0.02],
      [*refined*], [*1.41*], [*4.22*], [*0.96*], [*0.13*],
    )
  ],
  align(horizon)[
    #text(size: 0.9em)[
      - Refined wins in every band.
      - Only arm above the null at k50-80.
      - Margin there: a tenth of a decibel.
      - Refined labels put lines *on* the comb.
      - They do not sharpen lines above k #sym.approx 25.
      - Label fix: necessary, not sufficient.
    ]
  ],
)
#v(1fr)

#speaker-note[
  Numbers from the instrument-audited, self-referenced readout: each arm's
  floor and excision are pinned to its own carriers, so the comparison is
  fair across arms. Refined wins in every band and is the only arm to clear
  the null past k=25, but the margin at k50-80 is a tenth of a decibel — not
  a usable training signal. The question becomes: why does the ceiling sit
  at k=25 regardless of label quality?
]

= Seeing it: refined labels put lines back into the generator

#align(center + horizon, image("assets/gen_ab_shared.png", height: 84%))

#speaker-note[
  Same free-flight_nosource_room1 chunk, mic 0, from the generator-refined-
  labels campaign (scripts/eval_gen_comb_real.py). This is the visual
  version of the tooth-readout table on the previous slide. Left panels are
  freshly rendered STFTs (16 kHz, n_fft 2048, hop 512, 4-frame power
  average, RMS-matched to the real chunk, one shared color scale from the
  real row). Note: the rotor comb teeth above k about 10 are not legible in
  ANY full-band spectrogram of this recording — that is the phase-
  decoherence physics itself; the quantitative comb comparison is the
  right column.
]

= Same test, model with individual rotor embeddings

#align(center + horizon, image("assets/gen_ab_perrotor.png", height: 84%))

#speaker-note[
  Same chunk and layout, but the generator variant that conditions on
  individual rotor embeddings (gen_r2). Before/after refined labels shows
  the same effect: refined labels restore the comb; the per-rotor variant
  tracks the real line spectrum slightly more closely at mid k.
]

= Cause: the harmonic lines decohere inside the loss's own window

#v(1.2em)
#grid(
  columns: (0.9fr, 1.1fr),
  gutter: 1.2em,
  align(horizon)[
    $ Delta phi_k approx 2 pi k dot sigma dot T_"win" approx 0.24 dot k "rad" $
    #text(size: 0.8em)[phase drift within one 2048-sample window, #sym.sigma #sym.approx 0.6 rev/s]
    #v(0.8em)
    $ Delta phi_25 approx 6 "rad" > 2 pi $
    #text(size: 0.8em)[a full cycle, at k = 25]
    #v(0.8em)
    $ nabla_theta cal(L)_"log-L1" prop log("median"_t |X_k (t)|) - log hat(a)_k $
    #text(size: 0.8em)[the loss fits the smeared median, not the peak]
  ],
  align(top)[
    #image("assets/linewidth_law.png", width: 100%)
  ],
)
#v(1fr)

#speaker-note[
  This is the mechanism, not a guess: order-averaged tooth contrast along the
  refined tracks is 6.76 / 1.36 / 0.13 / 0.01 dB by band — the real acoustic
  comb itself is not STFT-separable above k=25 at 128 ms windows, because the
  phase wander smears the tooth into its neighbouring valleys within one
  window. Any audio-domain spectral loss, at any window length up to 2048
  samples, sees that same washed-out magnitude and correctly matches it. No
  amount of label precision fixes a target the loss cannot resolve.
]


= The fix: fit amplitudes, not waveforms

#v(1.5em)
#grid(
  columns: (1fr, 1fr),
  gutter: 1.5em,
  align(top)[
    $ a_k (t) = |"LPF"(x(t) e^{-i k phi(t)})| $
    #text(size: 0.8em)[demodulated envelope, not rendered audio]
    #v(0.8em)
    $ cal(L) = sum_k (log a_k - log hat(a)_k)^2 $
    #text(size: 0.8em)[compares two amplitudes, never two decohering waveforms]
  ],
  align(top)[
    $ B_k = B_0 = 1 "Hz" quad "(v1, wrong)" $
    #text(size: 0.8em)[one flat bandwidth for every k]
    #v(0.8em)
    $ B_k prop gamma_k = max(0.6 k, gamma_"min") quad "(v4, correct)" $
    #text(size: 0.8em)[bandwidth grows with k — the measured linewidth law]
  ],
)
#v(1fr)

#speaker-note[
  Version legend: v1 = first amplitude targets, flat-bandwidth envelopes.
  v3e = the shipped decomposition used for the results slides later in this
  deck. v4 = the model this deck presents.

  This slide is the pivot. The amplitude-target training path — codec, loss,
  data loader — is built and trains at 25 ms/step on CPU. The v1-target run
  already beats the audio-trained arms on fidelity at every band, including
  the ones the audio loss cannot see at all. But v1 solved every track at one
  fixed 1 Hz bandwidth, so it under-resolved the lines whose true linewidth
  grows with k, and that leakage is why the rendered comb from these targets
  still has no peak-to-floor contrast. The rest of this deck is the fix: a
  decomposition with the right bandwidth, built as one probabilistic model.
]


= Second problem: the old trajectory score prefers wrong trajectories

#align(center, image("assets/trajectory_candidates.png", height: 52%))
#v(0.2em)
#table(
  columns: (auto, 1fr, 1fr, 1fr, 1fr),
  column-gutter: 8pt,
  align: (left, right, right, right, right),
  stroke: 0.5pt + luma(200),
  inset: 4pt,
  text(size: 0.75em)[*old score (VK residual)*], text(size: 0.75em)[*telemetry*],
  text(size: 0.75em)[*refined*], text(size: 0.75em)[*ours (alternation)*],
  text(size: 0.75em)[*L-BFGS blind result*],
  text(size: 0.8em)[DREGON no-source], text(size: 0.8em)[−6.34], text(size: 0.8em)[−7.83], text(size: 0.8em)[*−12.87*], text(size: 0.8em)[−9.90],
  text(size: 0.8em)[DREGON speech], text(size: 0.8em)[−13.15], text(size: 0.8em)[*−13.29*], text(size: 0.8em)[−14.37], text(size: 0.8em)[−12.93],
  text(size: 0.8em)[DREGON whitenoise], text(size: 0.8em)[−15.30], text(size: 0.8em)[−15.99], text(size: 0.8em)[*−16.07*], text(size: 0.8em)[−15.69],
  text(size: 0.8em)[FLY124 w04], text(size: 0.8em)[−6.15], text(size: 0.8em)[−6.13], text(size: 0.8em)[−6.66], text(size: 0.8em)[*−9.80*],
  text(size: 0.8em)[FLY124 w05], text(size: 0.8em)[2532.99], text(size: 0.8em)[962.97], text(size: 0.8em)[*−11.19*], text(size: 0.8em)[149.46],
)
#v(0.2em)
#text(size: 0.8em)[
  The red trajectories are 10–36 rev/s from telemetry, yet the old score
  (lower = better) prefers them on 3 of 5 windows.
]
#v(1fr)

#speaker-note[
  Values are the old per-cell objective (total_per_cell) from
  results/joint_rescore_v4/summary.json. The multistart L-BFGS arm sits
  10-36 rev/s (rms) from telemetry on the FLY windows yet wins under the
  old score on w04, and beats telemetry and refined on the DREGON
  no-source window. FLY124 w05 shows the old score exploding to large
  positive values on non-matching hypotheses — another sign it is not a
  calibrated likelihood. The figure shows the four candidate sets on the
  two illustration windows: on FLY124 the blind result parks one rotor
  near 97 rev/s where telemetry has none, and on DREGON it invents a
  rotor pair near 68 rev/s — the coverage behaviour the old score
  rewards.
]

= The requirement: split real audio into comb and broadband, exactly

#v(0.8em)
#text(size: 1.05em)[
  fit to amplitudes #sym.arrow.r need to extract amplitudes from recording
  #sym.arrow.r need maximum a posteriori (MAP) decomposition
]
#v(0.8em)
#grid(
  columns: (0.9fr, 1.1fr),
  gutter: 1.2em,
  align(horizon)[
    $ x(t) = x_"comb"(t) + x_"broadband"(t) $
    #text(size: 0.8em)[exact split, sample for sample, on DREGON and FLY124/125]
    #v(0.6em)
    $ x_"comb"(t) = sum_(i=1)^4 sum_k x_(i,k)(t) $
    #text(size: 0.8em)[four rotors, interleaved combs, decohering, under one wash]
  ],
  align(center + horizon)[
    #image("assets/spec_dregon_original.jpg", width: 100%)
    #v(0.5em)
    #text(size: 0.85em, style: "italic")[
      One microphone, raw audio, 0-8 kHz.
    ]
  ],
)
#v(1fr)

#speaker-note[
  This is the difficulty in one picture, ahead of the model: four interleaved
  combs riding independent, wandering trajectories, sitting inside broadband
  turbulence that is itself not stationary. A hand-tuned splitting stage
  (v1 through v3) chased this difficulty with a growing pile of special
  cases — masks, block schedules, a separate stochastic-comb stage. v4
  replaces all of that with one probabilistic model.
]

= The v4 model: one Gaussian process, floor plus lines

#v(0.6em)
$ M_c (f,t) = S_c (f,t) + sum_(i,k) H_(c,i,k)(t) dot L_(gamma_k)(f - k dot r_i (t)) $
#text(size: 0.8em)[claimed PSD at mic $c$ = floor $S$ + sum of harmonic lines]
#v(0.6em)
#grid(
  columns: (1.1fr, 0.9fr),
  gutter: 1.2em,
  align(horizon)[
    $ L_(gamma_k)(f) = gamma_k^2 / ((f - k dot r_i (t))^2 + gamma_k^2) $
    #text(size: 0.8em)[unit peak at line center, half-width $gamma_k = max(0.6 k, gamma_"min")$ Hz]
    #v(0.6em)
    $ H_(c,i,k)(t) equiv "peak PSD above floor" $
    #text(size: 0.8em)[mic $c$, rotor $i$, harmonic $k$, per time block — same units as $S$]
    #v(0.6em)
    $ integral L_(gamma_k) (f) d f = pi gamma_k, quad "line power" = pi gamma_k H_(c,i,k) $
    #text(size: 0.8em)[$H$ is the training target we set out to get]
  ],
  align(center + horizon, image("assets/lorentzian_bumps.png", width: 100%)),
)
#v(1fr)

#speaker-note[
  One sentence version: floor plus a comb of Lorentzian lines riding the
  trajectories, with the measured 0.6k Hz linewidth law built in as the
  shape of each line, not fitted per line. The phase-noise structure from
  the tracking work is unchanged and enters through the trajectories. What
  is new is that H — the per-harmonic power — sits inside this one
  likelihood as a parameter to be estimated, not a byproduct of a
  subtraction stage.
]

= Objective, part 1: the Whittle cost of one cell

#v(1.2em)
#grid(
  columns: (0.9fr, 1.1fr),
  gutter: 1.0em,
  align(top)[
    $ "cost"(S; P) = P slash S + log S $
    #text(size: 0.8em)[cost of one cell, claimed $S$ against observed $P$]
    #v(0.8em)
    $ (partial "cost") / (partial S) = 0 quad arrow.r.long quad S = P $
    #text(size: 0.8em)[minimum at $S = P$]
  ],
  align(top, image("assets/whittle_cost.png", width: 100%)),
)
#v(1fr)

#speaker-note[
  This is the Whittle likelihood cost of a single time-frequency cell. The
  U-curve on the left shows the shape: cost is minimized when the claimed
  power matches the observed power, and it rises steeply on both sides. The
  right panel is a worked example — one loud cell, ten units of power. Priced
  as pure floor at unit power it costs about 10 plus log S0; priced as a line
  at its own power it costs about 3.3 plus log S0. The model prefers to
  explain the loud cell as a line, for free, because the cost function says
  so — no threshold, no hand-tuned mask.
]

= Objective, part 2: linewidth and marginalization

#v(1fr)
#grid(
  columns: (1fr, 1fr),
  gutter: 1.2em,
  align(horizon)[
    $ e^(-t slash tau) quad limits(<->)^"Fourier" quad L_gamma (f), quad gamma = 1 slash (pi tau) $
    #text(size: 0.8em)[exponential phase forgetting in time = Lorentzian width in frequency]
    #v(0.8em)
    $ J = sum_(f,t) [P slash M + log M], quad M = S + sum H dot L $
    #text(size: 0.8em)[marginalized objective — $H$ inside $log M$, no free envelope left]
  ],
  align(center + horizon, image("assets/marginalization.png", width: 100%)),
)
#v(1fr)

#speaker-note[
  The connection to the linewidth measurement: a shaft that forgets its phase
  exponentially in time is, by a standard Fourier fact, a Lorentzian-shaped
  line in frequency, and the measured 0.6k Hz law is exactly that width.
  Marginalizing the line envelopes — integrating them out as Gaussian
  processes rather than fitting them as free parameters — removes the
  envelope from the objective entirely and leaves one clean sum over cells,
  each priced by the same cost-of-a-cell rule from the last slide.
]

= Solver 1/2: init and floor/line fit

#v(0.3em)
#text(size: 0.78em)[
  *Init.* $hat(phi.alt)_i [n] = 2 pi Delta sum_(m <= n) hat(r)_i [m]$; warm start $g = log S$ from the masked v3 fit.
]
#v(0.4em)
#text(size: 0.78em, weight: "bold")[Block 1 — floor $S$ and line powers $H$, per (mic $c$, block $b$)]
#v(0.15em)
#text(size: 0.72em)[
  $ overline(P)(f) = op("median")_(t in b) lr(|X(f,t)|)^2 / ln(2) $
  #text(size: 0.85em, style: "italic")[pooled periodogram; median of Exp(1) is $ln 2$]
  #v(0.15em)
  $ A[f,l] = L_(gamma_k) (f - k thin overline(r)_i), quad gamma_k = max(0.6 k, Delta f) $
  #text(size: 0.85em, style: "italic")[truncated Lorentzian design]
  #v(0.15em)
  alternate 2-3 rounds, two starts $g_0 in {"warm", "warm" - 12 "dB"}$, keep lower objective:
  #v(0.1em)
  H-step: $ min_(H >= 0) norm(1/M (overline(P) - S - A H))^2 $
  #text(size: 0.85em, style: "italic")[IRLS-weighted NNLS, signed excess]
  #v(0.1em)
  S-step: damped Newton on $g$, Fisher weight $(S slash M)^2$, banded Hessian $+ lambda_f D_2^top D_2$, trust radius 2 nats
  #v(0.15em)
  $ J_"floor" = sum_f [overline(P) slash M + log M] + lambda_f norm(D_2 g)^2, quad M = e^g + A H $
]
#v(1fr)

#speaker-note[
  Slide 1 of the solver pair. Initialize the phase trajectory by integrating
  the current speed estimate, and warm-start the log-floor from the masked
  v3 fit so the very first iteration is close. Block 1 fits the noise floor
  jointly with every harmonic line's power against a Whittle-type objective:
  H-step is a nonnegative least squares in the signed residual weighted by
  the model spectrum, S-step is a damped Newton step on the log-floor with
  a smoothness penalty. Two different starting floors are tried and the
  lower-objective result is kept, guarding against local minima.
]

#pagebreak()

= Solver 2/2: envelope, trajectory, loop

#v(0.6em)
#grid(
  columns: (1fr, 1fr),
  gutter: 1.2em,
  align(top)[
    #text(size: 0.8em, weight: "bold")[Block 2 — envelope posterior, one banded solve per coupled group]
    #v(0.2em)
    #text(size: 0.78em)[
      $ (E^H W E + op("diag")(rho_k^2 D_2^top D_2) + op("diag")(beta_k)) a = E^H W z $
      #v(0.2em)
      $ W = 1 slash S $ at line positions (clamped $plus.minus 15$ dB)
      #v(0.15em)
      $ rho_k = rho(max(3, 0.6 k) "Hz"), quad beta_k = S_"geo" slash H_k $
      #text(size: 0.85em, style: "italic")[the amplitude prior]
      #v(0.2em)
      comb $= sum op("Re")[a thin e^(j k hat(phi.alt))]$; broadband $= x - $ comb
      #text(size: 0.85em, style: "italic")[exact identity]
    ]
  ],
  align(top)[
    #text(size: 0.8em, weight: "bold")[Block 3 — trajectory correction from coherent harmonics]
    #v(0.2em)
    #text(size: 0.78em)[
      $ Delta hat(phi.alt)_k [j] = arg(a_k [j+1] a_k^* [j]) $
      keep tracks with $kappa_k >= 0.5$
      #v(0.2em)
      $ hat(delta)[j] = (sum_k k^2 kappa_k^2 Delta hat(phi.alt)_k [j] slash (2 pi Delta_e k)) / (sum_k k^2 kappa_k^2) $
      #v(0.15em)
      $arrow.r$ Whittaker–Henderson smooth at 1.5 Hz $arrow.r$ $hat(r) <- hat(r) + hat(delta)$
    ]
  ],
)
#v(0.6em)
#text(size: 0.78em)[
  *Loop.* Blocks 1 → 3, three iterations, harmonic-trust ladder $k <= (3, 12, 80)$.
]
#v(0.3em)
#text(size: 0.75em)[
  *Readout.* $ J_"v4" = sum_(c,f,t) [P slash M + log M] + lambda_theta norm(D_2 theta)^2 + sum_k lambda_psi (k) norm(D_2 psi_k)^2 + lambda_f R(log S) $
]
#v(1fr)

#speaker-note[
  Slide 2 of the solver pair. Block 2 solves for every line's complex
  envelope in one banded linear system per coupled group, weighted by the
  local floor and shrunk toward zero by a prior tied to the line's fitted
  power. The comb reconstruction and the broadband residual sum back to the
  input exactly, by construction. Block 3 turns the envelope's phase
  increments into a rotor-speed correction, power-weighted by harmonic
  concentration, smoothed, and folded back into the trajectory. The whole
  loop runs three times with an increasing harmonic-trust ladder, and the
  readout objective is the same v4 cost used to rank a fit at convergence.
]

= The decomposition on DREGON

#v(1fr)
#grid(
  columns: (0.85fr, 1.15fr),
  gutter: 1.0em,
  align(horizon)[
    #text(size: 0.82em)[
      Numbers from the unified-model decomposition (run of record, all 7
      windows).
    ]
    #v(0.4em)
    #text(size: 0.8em)[
      Retained excess by band (% of harmonic-band energy still in the
      residual — lower is a cleaner split):
    ]
    #table(
      columns: (auto, 1fr, 1fr, 1fr, 1fr),
      column-gutter: 8pt,
      align: (left, right, right, right, right),
      stroke: 0.5pt + luma(200),
      inset: 5pt,
      [], text(size: 0.85em)[*k1-9*], text(size: 0.85em)[*k10-24*],
      text(size: 0.85em)[*k25-49*], text(size: 0.85em)[*k50-80*],
      [DREGON], [6.7%], [9.7%], [15.3%], [7.5%],
    )
    #v(0.3em)
    #text(size: 0.78em, style: "italic")[
      DREGON's absolute tooth contrast at k50-80 is itself #sym.approx 0.01 dB
      (previous slides) — a small residual fraction here is real signal, not
      noise.
    ]
  ],
  align(center + horizon)[
    #image("assets/spec_dregon.jpg", height: 78%)
    #text(size: 0.65em, style: "italic")[
      Top to bottom: original, comb (ALL rotor-locked energy — one model
      channel; its power is the H table), broadband floor. There is no
      separate stochastic channel: the line process is one object.
    ]
  ],
)
#v(1fr)

#speaker-note[
  Original, comb, and broadband panels, 0 to 8 kHz, one microphone, one
  eight-second span. DREGON's comb is the cleanest of the three recordings —
  lowest retained excess in every band except k25-49, where twin-rotor
  crossings cost extra.
]

= DREGON: the comb channel split by rotor

#align(center + horizon)[
  #image("assets/spec_dregon_perrotor.jpg", height: 76%)
  #text(size: 0.65em, style: "italic")[
    The same comb channel, attributed per rotor. Rotors whose speeds stay
    #sym.lt.eq 1 rev/s apart share their energy softly between them — the
    comb-vs-floor split stays sharp.
  ]
]

#speaker-note[
  Per-rotor attribution of the DREGON comb, same window and color scale as
  the previous slide. The twin pair overlaps for most of the window, so
  the model cannot uniquely attribute their shared harmonics — that energy
  divides between the two tracks. The sum over rotors equals the comb
  channel exactly.
]

= The new score ranks the same trajectories correctly

#align(center, image("assets/trajectory_candidates.png", height: 52%))
#v(0.2em)
#table(
  columns: (auto, 1fr, 1fr, 1fr, 1fr),
  column-gutter: 8pt,
  align: (left, right, right, right, right),
  stroke: 0.5pt + luma(200),
  inset: 4pt,
  text(size: 0.75em)[*new score ($J$, per cell)*], text(size: 0.75em)[*telemetry*],
  text(size: 0.75em)[*refined*], text(size: 0.75em)[*ours (alternation)*],
  text(size: 0.75em)[*L-BFGS blind result*],
  text(size: 0.8em)[DREGON no-source], text(size: 0.8em)[−17.5674], text(size: 0.8em)[*−17.5792*], text(size: 0.8em)[−17.5721], text(size: 0.8em)[−17.5748],
  text(size: 0.8em)[DREGON speech], text(size: 0.8em)[−17.4022], text(size: 0.8em)[*−17.4054*], text(size: 0.8em)[−17.4024], text(size: 0.8em)[−17.3504],
  text(size: 0.8em)[DREGON whitenoise], text(size: 0.8em)[−17.2709], text(size: 0.8em)[−17.2739], text(size: 0.8em)[*−17.2762*], text(size: 0.8em)[−17.2563],
  text(size: 0.8em)[FLY124 w04], text(size: 0.8em)[−16.2274], text(size: 0.8em)[*−16.2397*], text(size: 0.8em)[−16.1564], text(size: 0.8em)[−16.1067],
  text(size: 0.8em)[FLY124 w05], text(size: 0.8em)[*−15.9660*], text(size: 0.8em)[−15.9520], text(size: 0.8em)[−15.8584], text(size: 0.8em)[−14.6550],
)
#v(0.2em)
#text(size: 0.8em)[
  Same candidates, scored with $J$: the blind result now ranks *last* on 4
  of 5 windows; refined labels rank first on 3 of 5.
]
#v(1fr)

#speaker-note[
  Values are total_v4_per_cell from results/joint_rescore_v4/summary.json —
  the marginal Whittle likelihood of the original signal under each
  trajectory hypothesis, no extra terms. Compare with the old-score slide:
  the same blind L-BFGS trajectories that won under the profiled VK
  residual now rank dead last on four of five windows; on the DREGON
  no-source window their old 0.126/cell win shrank to a 0.004/cell loss —
  a factor of about 30. The k-scaling stress test (k_max 65) does not
  reopen the gap. This is the fitness-quality result for the unified
  model, in place of the stale v3e FLY decomposition panels.
]

= From the v4 solution to generator targets

#v(0.3em)
$ (hat(S), hat(H), hat(r)) = op("arg min") J $
#text(size: 0.75em)[1. solve the v4 objective]
#v(0.2em)
$ a_(c,i,k)(t) = sqrt(2 pi gamma_k hat(H)_(c,i,k)(t)) $
#text(size: 0.75em)[2. amplitude target per line — RMS sinusoid amplitude from the Lorentzian's integrated power]
#v(0.2em)
$ G(r(t)) arrow.r (log hat(a)_(i,k)(t), log hat(S)(f,t)), quad cal(L) = sum (log hat(a) - log a)^2 + sum (log hat(S)_G - log hat(S))^2 $
#text(size: 0.75em)[3. generator training — plain regression, no audio synthesis, no phase, no MRSTFT]
#v(0.2em)
$ hat(x) = sum_k hat(a)_k (t) cos(k hat(phi) + psi_k (t)) + "noise"(hat(S)) $
#text(size: 0.75em)[4. rendering, only when audio is needed — $psi_k$ injected from the measured 0.6k phase-noise law, not learned]
#v(1fr)

#speaker-note[
  Four-step chain from the v4 fit to what actually trains the generator.
  Step 1 is the decomposition itself. Step 2 turns the fitted line power H
  into an RMS amplitude by integrating the Lorentzian — pi*gamma*H is the
  line's power, and the RMS amplitude of a sinusoid is the square root of
  twice its power. Step 3 is the generator objective: a plain log-domain
  regression against those amplitudes and the floor, nothing about audio
  waveforms or phase. Step 4 is the one place phase enters at all, and only
  at render time, using the measured phase-noise law rather than anything
  the generator learned. The small figure is the pipeline in one line:
  recording, v4 fit, the (H, S) it outputs, the regression targets built
  from them, the generator, and the optional render step.
]

= Performance: CPU is the bottleneck

#v(0.8em)
#table(
  columns: (auto, auto, auto),
  column-gutter: 10pt,
  align: (left, right, left),
  stroke: 0.5pt + luma(200),
  inset: 6pt,
  text(size: 0.85em)[*operation (one 12–16 s window)*],
  text(size: 0.85em)[*CPU wall time*], text(size: 0.85em)[*consequence*],
  [one score evaluation of a trajectory], [#sym.approx 70 s],
  [too slow for an inner loop],
  [one blind L-BFGS refinement], [#sym.approx 600 s], [offline only],
  [full decomposition of the window], [400–2100 s],
  [batch pipeline only],
)
#v(0.4em)
#text(size: 0.9em)[
  Why an evaluation is cheaper than a decomposition: the trajectory is
  *fixed*, so it is *one* inner (floor, line-power, envelope) fit at
  k#sub[hi] #sym.approx 40. The full decomposition alternates that fit
  with trajectory updates to convergence at k#sub[max] = 100 and adds
  the stitching — 6–30#sym.times the work. One recording (7 windows)
  measured 110 min.
]
#v(0.4em)
#text(size: 0.9em)[
  Cost structure: hundreds of *independent* banded solves (one per track)
  #sym.times mics #sym.times windows, plus FFTs — all batchable.
]
#v(0.4em)
#align(center, cbox(text(size: 0.95em)[
  *Hypothetical GPU speedup: 30–100#sym.times.* Batched banded Cholesky +
  batched FFT.
]))
#v(1fr)

#speaker-note[
  Timing sources: 70 s per objective evaluation is the measured wall_s in
  the joint-rescore runs (results/joint_rescore_v4, wall_s 72.6 on FLY124
  w04); 600 s per window is the L-BFGS-on-the-score arm of the refiner
  comparison (results/fvk_arms fig, wall/window 585-612 s on DREGON and
  FLY124); the full-recording decomposition jobs run for hours on a CPU
  node of the cluster. The 30-100x range is an estimate, not a
  measurement: the per-track banded solves are independent and small, so
  they batch well, and the one kernel we already ported (the CKLA scan,
  Triton) gained 230x — but Cholesky factorization is more
  synchronization-heavy than a scan, hence the conservative lower bound.
]

= Next steps

#v(1.5em)
#text(size: 1.2em)[
  - *1. Finsih the v4 experiments* — Michael's decomposition; synthetic experiments
  - *2. Retrain the generator* on restored amplitude targets and see if harmonics are well-recovered
  - *3. Try pseudo-labeling other drone data using this objective?* Should work but would be very very slow.
]
#v(1.5em)
#align(center, cbox(text(size: 1.1em)[
  Honestly - I got a little bit lost in all of these experimental results and I would appreciate
  some guidance in how to select the actual content of the paper.
]))
#v(1fr)

#speaker-note[
  Three concrete next steps, in priority order. The v4 gates are the
  immediate blocker: everything on the results slides is v3e; v4 is designed
  and coded but not yet validated end to end. Once it passes, the amplitude
  targets it produces feed straight into the training path that already
  exists. The blind-labeling use case is the longer-range payoff: any
  unlabeled drone recording becomes trainable data once the objective can
  rank candidate trajectories against it.
]
