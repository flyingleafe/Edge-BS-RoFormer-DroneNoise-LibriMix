#import "@preview/touying:0.7.4": *
#import "/writing/templates/typst/slides.typ": hns-slides

// Notes-visible build: `make notes` (typst --input notes=1). Default build is clean.
#let notes-mode = sys.inputs.at("notes", default: none)

#show: hns-slides.with(
  title: [RPS prediction and drone noise generation progress],
  subtitle: [Attempts to improve generator #sym.dot.c Another approach to RPS prediction #sym.dot.c GP noise model analysis #sym.dot.c Back to basics],
  author: [Dmitrii Mukhutdinov],
  date: [2026-07-18],
  show-notes: if notes-mode != none { bottom } else { none },
)

#let keyline(body) = align(center, text(size: 1.05em, body))

= This week

#v(1em)

+ *Generative model* #sym.dash.em audit + fix the physics-structured generator.
  + Found bugs in array geometry annotations (both real drone datasets).
  + Tried two ways to make the model more expressive: per-rotor timbre, wind noise.
+ *Blind order tracking* #sym.dash.em per-rotor RPS from audio alone.
  + Coupled Vold–Kalman: score a trajectory by how well it explains the audio.
  + Today: made it $approx 10 times$ faster and fixed a blind-seeding bias.
+ *Literature baseline* #sym.dash.em reproduce JASA-GP as a reference.
  + Fit the same GP to our drone setups and compare all four noise sources head-to-head.

#v(0.6em)
Two more threads *just opened*: modern blind speech-enhancement baselines, and a
plan to bring the RPS predictor up to VK's accuracy without VK's cost.

#speaker-note[
  Here is what I did
]

= The problem: what the old generator sounds like

#v(1fr)
#align(center, image("assets/fig_oldgen_problem.png", height: 86%))
#v(1fr)

#speaker-note[
  Before any theory: look (and listen). Real cruise noise vs the old generator,
  both drones. The generated noise misses the mid-frequency harmonic texture —
  the combs above ~1 kHz are washed out. This is the concrete problem the whole
  generator thread attacks.
]

= Motor sound propagation in generator model

#v(1fr)
#text(size: 1.3em)[$ y_m (t) = sum_k frac(r_"ref", r_(m,k)) dot s(t - r_(m,k) slash c) $]
#v(1.2em)

#text(size: 1.1em)[One emitter per rotor, propagated free-field. Three assumptions testable on DREGON:]

#v(0.5em)
#text(size: 1.1em)[
  + array *geometry* correct (distances between rotors and mics are )
  + all four rotors share *the same model*
  + a *coherent, propagating* field is complete
]

#text(size: 1.1em)[Now the question is: why noise generator loses mid-range harmonics?]
#v(1fr)


#speaker-note[
  Problem: why mid-range harmonics are so bad? Let us try to understand
]

= Hypothesis 1 — geometry: a small error is not benign

#figure(image("assets/geo_propagation_phase.png", width: 90%))

#keyline[Symptom: systematically weak mid-frequency harmonics. #h(0.6em) $Delta phi.alt(f) = 2 pi f delta$.]

#speaker-note[
  Hypothesis 1: rotor and mic positions are not precise, so TDOA errors between real and simulated sound are large
  in comparison with mid-range harmonic fequencies
]

= Geometry errors found (huge!) in DREGON annotations

#grid(
  columns: (1fr, 1fr),
  gutter: 1em,
  figure(image("assets/geo_frame_alignment.png", width: 100%)),
  figure(image("assets/dregon_geometry_positions.png", width: 100%)),
)

#keyline[Predicted vs. measured TDOA: $r = -0.56 arrow.r +0.93$ at $183 degree$.]

#speaker-note[
  How we checked: on DREGON, we have single-motor recordings, so can compare TDOA
  estimates from GCC-PHAT with provided mic positions easily enough (showing figures).
  And turns out the microphones should be rotated 180° about z for TDOAs to make sense!
  (show relevant correlation plots + mic + rotor positions 3d plot)
]

= Geometry errors found (silly) in Michael's annotations

#align(center, image("assets/michaels_geometry_positions.png", height: 76%))

#keyline[Mic ring coded *vertical* instead of *horizontal*, found by eye, not by TDOA.]

#speaker-note[
  In Michael's case, the mic array annotation just had a bug (swapped coordinates) so it was
  as if circular array was put vertically, not horizontally. But this was understood just visually,
  TDOA correlation optimization does not work for Michael's because no single-rotor recordings (hard).
  (see plot of mic + rotor positions)
]

= Geometry: fine calibration

#grid(
  columns: (1.15fr, 1fr),
  gutter: 1.2em,
  align(center, image("assets/geo_summary.png", height: 62%)),
  [
    #v(0.5em)
    Minimise the coherence-weighted, phase-wrapped RTF residual over mic
    positions $bold(p)$:
    $
      cal(L)(bold(p)) = frac(
        sum_(r,m,f) gamma^2_(r,m)(f) dot "wrap"[phi.alt^"meas"_(r,m)(f) - phi.alt^"ff"_(r,m)(f; bold(p))]^2,
        sum_(r,m,f) gamma^2_(r,m)(f)
      ) + lambda norm(bold(p) - bold(p)_0)^2
    $
    $ phi.alt^"ff" = -2 pi f (r_m - r_"ref") slash c $
    - $gamma^2$ = magnitude-squared coherence (trust only clean bins; $400$–$800$ Hz band)
    - prior $lambda norm(bold(p)-bold(p)_0)^2$ fixes the *gauge* (translation/rotation) + regularises
    - minimised with *Adam*; synthetic control recovers truth to $0.36$ cm
  ],
)

#keyline[DREGON: phase residual $38.3 degree arrow.r 28.8 degree$, moves $<= 2.2$ cm. #h(0.6em) Michael's: not identifiable.]

#speaker-note[
  Here we say how we can _optimise_ the mic positions further using TDOA correlation maximization,
  using formulas and the figure of mic positions before and after optimisation.
]

= Hypothesis 2 — rotors are individuals

#figure(image("assets/fig_per_rotor.png", width: 92%))

#keyline[Level-normalised *timbre* differs by *6.8 dB RMS*.]

#speaker-note[
  Show rotor spectras figures (individual recordings) at same speeds, demonstrating that
  harmonics and broadband components have different amplitude distributions.
]

= Treating rotors individually: per-rotor sub-embeddings

#v(0.2em)
$ z_r = z_"drone" + delta z_r, quad delta z in RR^(R times d) $

#align(center, image("assets/fig_subembed_schema.png", width: 80%))

#v(0.2em)
#text(size: 0.9em)[
  - shared across drones (identity = airframe layout); zero-initialised — a
    strict generalisation of the shared-code model
  - regularised: $lambda dot bar(delta z)_2^2$ keeps sub-codes small unless the
    data actually needs per-rotor detail
]

#speaker-note[
  Show schema of drone noise generator model (see prev slides) with per-rotor sub-embeddings ADDED and emphasized,
  + with formula on how the losses change (how we keep sub-embedding delta small)
]

= Hypothesis 3 — wind noise confusing the model

#v(0.2em)
#align(center, image("assets/fig_wind_schema.png", height: 52%))

#keyline[Physics fixes where the flow lands; a small learned head fits what it does to a mic.]

#speaker-note[
  Mid-range harmonics were much weaker on DREGON, and DREGON has wind noise, so maybe that is why the model struggles.
  Let's actually add the physically-plausible wind noise model additively on top!
]

= Results: spectrograms

#v(1fr)
#align(center, image("assets/generator_variants_grid_wide.png", width: 99%))
#v(1fr)

#speaker-note[
  Here we display FREE-FLIGHT segments: real and corresponding generations for each model variant -
  old (before modifications), v1 (geometry fix), v2 (per-rotor sub-embeddings), v3 (wind noise model).
  Top row - DREGON, bottom row - Michael's
]

= Results: the scores, and why

#v(0.3em)
#figure(
  table(
    columns: 3,
    align: (left, center, left),
    stroke: 0.7pt,
    inset: 8pt,
    table.header([*variant*], [*free-flight match score* #sym.arrow.t], [*reading*]),
    [OLD (wrong geometry)], [4.51], [pre-fix baseline],
    [*v1 — corrected geometry*], [*5.22*], [*best — geometry fix helps*],
    [v2 #sym.plus per-rotor], [4.82], [best on idle, below v1 in flight],
    [v3 #sym.plus wind], [3.44], [dormant at hover airspeed],
  ),
)

#keyline[#text(size: 0.75em)[A different quantity from the msSTFT #sym.arrow.b distance used later
  (§ four-way comparison) — this one is higher-is-better, not a common scale.]]

#v(0.6em)
- Fixing the geometry errors helped with mid-range harmonics on both DREGON and Michael's - but not quite fully
- Per-rotor sub-embeddings lead to weird broadband suppressions in mid-to-high frequencies - need to investigate
- Wind noise model just doesn't work as expected - most likely, due to stochasticity, it's unreasonable to apply the same deterministic spectral loss to it.

#speaker-note[
  Here we display FREE-FLIGHT segments: real and corresponding generations for each model variant -
  old (before modifications), v1 (geometry fix), v2 (per-rotor sub-embeddings), v3 (wind noise model).
  Top row - DREGON, bottom row - Michael's
  Under each spectrogram of generated noise - mrSTFT loss against the real noise.
  Then we list possible reasons why sub-embeddings and wind noise model worked badly.
]

= Work thread 2: blind per-rotor RPS from audio alone

#v(1fr)
#text(size: 1.1em)[
  $ y(t) = sum_r "Re"[a_r (t) dot e^(i phi.alt_r (t))], quad "residual" = norm(y - hat(y))_2^2 $
]
#v(0.5em)

#text(size: 1.05em)[Instead of picking peaks off a spectrogram, ask: *which RPS trajectory,
  plugged into a generative signal model, best explains the recorded audio?*
  The envelope solve that makes this tractable is the classical Vold–Kalman
  (VK) filter (Vold & Leuridan 1993); everything else is how we *initialise*
  and *iterate* it blindly.]

#v(0.8em)
#text(size: 1.05em)[*The pipeline, in the order it runs:*]
#text(size: 1.0em)[
  + *whiten* the average spectrum (kill the broadband tilt — combs stand out)
  + *scan* for comb bases $f_0$ (teeth $k dot f_0$, band-capped, alias-filtered)
  + *seed* $R = 4$ rotors (dedup; residual re-scan for missed rotors; pair duplication)
  + *capture* — coarse VK iterations (wide bandwidth, few harmonics) lock on from $plus.minus 2$–$3$ rev/s off
  + *refine* — narrow VK iterations (tight bandwidth, more harmonics) polish the trajectory
  + *twin resolution* — spatial dynamic programming over windows disambiguates near-identical rotors
]
#v(1fr)

#speaker-note[
  Main idea: score a candidate trajectory by how well it explains the audio.
  Then the algorithm is: whiten, scan, seed, capture, refine, resolve twins.
  Steps 1-3 happen once; steps 4-5 are the same VK iteration at two coarseness
  levels; step 6 runs across windows. Next slides: each step in detail, then
  the full pseudocode.
]

= Steps 1–2: whiten, then scan for combs

#v(1fr)
#text(size: 1.0em)[
  #grid(
    columns: (1fr, 1fr),
    gutter: 1.5em,
    [
      *Step 1 — whiten.* Average the spectrogram over time, divide out a smooth
      spectral envelope:
      $ W(f) = frac(macron(S)(f), "smooth"(macron(S))(f)) $
      Broadband tilt and room colouration are gone; only *peaks relative to the
      local floor* remain — a comb becomes a row of equal-height teeth.
    ],
    [
      *Step 2 — scan.* For each candidate base $f_0 in [30, 120]$ rev/s, score
      the comb by its mean whitened tooth height, *only in the energetic band*:
      $ S(f_0) = "mean"_(k : k f_0 <= 1.2 "kHz") W(k f_0) $
      Reject *aliases*: a candidate whose teeth are a strict subset of a
      stronger candidate's teeth (e.g. $f_0 slash 3$), or an integer ratio of
      one, is dropped. Output: ranked distinct bases.
    ],
  )
]
#v(0.4em)
#keyline[The band cap matters: uncapped, a small $f_0$ packs all its teeth into the loud low band and can outscore real combs.]
#v(1fr)

#speaker-note[
  Whitening: combs are peaks relative to local floor, so divide the floor out.
  Scan: score every candidate fundamental by mean tooth height in the band
  where harmonics actually live. The band cap (1.2 kHz) is the fix from today:
  without it a junk 30 rev/s candidate wins because all 40 of its teeth sit in
  the loud low-frequency region while a real 91 rev/s comb dilutes into the
  dead high band.
]

= Step 3: seed all four rotors

#v(1fr)
#text(size: 1.0em)[
  + take the top-$R$ *distinct* bases (dedup at $4$ rev/s — closer peaks are one rotor);
  + *if fewer than $R$ found* — *residual re-scan*: mask the teeth of every
    claimed comb in $W$, re-run the scan on what remains. A weak rotor hidden
    behind a strong one becomes the dominant residual comb;
  + *still short?* duplicate seeds at the lowest-scoring used base ($plus.minus 0.1$
    nudge) — a merged *twin pair* spreads energy over two offset combs and
    under-scores; the coupled solve (step 4) splits the duplicates.
]
#v(0.6em)
#figure(
  table(
    columns: 3,
    align: (left, center, center),
    stroke: 0.5pt,
    inset: 6pt,
    table.header([*FLY124 cruise*], [*seeds (rev/s)*], [*note*]),
    [truth], [$73.9, 74.5, 81.3, 91.0$], [one twin pair + two singles],
    [plain scan (band-capped)], [$74.7, 75.7, 91.3, 92.3$], [3 distinct bases + duplicate],
    [+ residual re-scan (R)], [$75.2, 75.3, 82.7, 91.8$], [all 4 distinct bases seeded],
  ),
)
#v(1fr)

#speaker-note[
  Seeding = deciding where the four trackers start. Dedup keeps distinct bases;
  the residual re-scan is the newest arm: subtract (mask) what the seeds already
  explain, scan again — the missed 81.3 rotor is the loudest thing left. The
  duplication fallback covers merged twin pairs; VK's coupling plus a small
  nudge separates them during capture. Table: the real FLY124 example,
  before/after today's fixes.
]

= Steps 4–5, the VK core: fit envelopes, correct frequencies, tighten

#v(0.3em)
#text(size: 0.92em)[
  #grid(
    columns: (1fr, 1fr),
    gutter: 1.5em,
    [
      *Fit envelopes (VK).* Frequencies fixed $arrow.r$ envelopes are *linear*:
      $ y(t) = sum_r "Re"[a_r (t) e^(i phi.alt_r (t))] = bold(C)(t) bold(a)(t) $
      $ bold(a) = arg min_bold(a) norm(y - bold(C) bold(a))_2^2 + "smoothness prior" $
      At audio rate that is $approx$ *16M* unknowns; envelopes are narrowband, so
      demodulate + decimate first:
      $ z_r = "decim"("LP"[y dot overline(c_r)]), quad c_r = e^(i phi.alt_r) $
      $arrow.r$ *200k* unknowns on a $100$ Hz grid.
    ],
    [
      *Correct frequencies (phase slope).* The fitted envelope of harmonic $k$
      rotates at $k dot$ (frequency error):
      $ hat(delta)_r = angle(x_r (t{+}1) overline(x_r (t))) frac(f_s^"env", 2 pi k) $
      fused over harmonics with Fisher weights $k^2 |z_(r,k)|^2$, clipped, added
      to $phi.alt_r$; then re-demodulate.

      *Tighten (anneal).* Capture: few harmonics, wide bandwidth (big basin).
      Refine: many harmonics, narrow bandwidth (precision). $k_max$ grows as the
      track locks.
    ],
  )
]

#speaker-note[
  The inner loop, run at two coarseness levels (capture, then refine): fit
  envelopes by least squares (the VK filter — linear because frequencies are
  frozen), read the frequency error off the envelope phase drift (each harmonic
  k amplifies it k-fold, so high harmonics are precise but only near lock),
  update, re-demodulate, and progressively add harmonics as lock improves.
]

= Step 4 detail: coupled solve — tracks compete for shared energy

#v(0.1em)
#text(size: 0.85em)[
  #grid(
    columns: (1.1fr, 1fr),
    gutter: 1.2em,
    align: (left, left + horizon),
    [
      Couple tracks $r$, $r'$ whenever their carriers can be confused:
      $ |f_r (t) - f_r' (t)| < B_"env", quad B_"env" = "envelope bandwidth" $
      Then the joint envelope solve becomes one linear system per time-block:
      $
        bold(G) bold(a) = bold(b), quad
        bold(G)_(r,r') = cases(
          bold(C)_r^* bold(C)_r & r = r',
          bold(C)_r^* bold(C)_(r') & "coupled",
          0 & "else"
        )
      $
      $bold(G)$ is *block-banded Hermitian* (only nearby tracks couple) $arrow.r$
      cheap Cholesky, not a dense solve.

      #v(0.2em)
      #text(size: 0.85em)[
        ```
        for outer iter = 1..N:
          z_r[T_env,D] = demod+decimate(y, phi_r) # Step 2
          a[R,T_env]   = banded_chol_solve(G, b)  # Step 3
          phi_r        = phase_slope_update(a)    # Step 4
          k_max        = anneal(k_max, iter)      # Step 4
        ```
      ]
    ],
    figure(image("assets/vk_coupling_schematic.png", width: 100%)),
  )
]

#speaker-note[
  Nearby harmonic tracks (e.g. a twin rotor pair a fraction of a rev/s apart)
  overlap in frequency and would otherwise both try to claim the same spectral
  energy. A sparse off-diagonal coupling term between nearby tracks forces them
  to split it (explaining-away) instead. The linear system stays block-banded
  Hermitian — cheap to solve even coupled.
]

= Step 5 detail: frequency update by phase slope, then anneal

#v(0.8em)
#text(size: 1.15em)[
  $
    hat(delta)_r = angle(x_r (t+1) dot overline(x_r (t))) dot frac(f_s^"env", 2 pi k), quad
    "fused across harmonics by Fisher weight" thin k^2 |z_r,k|^2
  $
]
#v(1.2em)

#text(size: 1.05em)[
  #grid(
    columns: (1fr, 1fr),
    gutter: 1.5em,
    [
      - each harmonic $k$ gives its own frequency-drift estimate; fuse them,
        weighted by how much energy/certainty that harmonic actually carries
      - update: $phi.alt_r arrow.l phi.alt_r + hat(delta)_r$, then re-demodulate (Step 2)
    ],
    [
      - $k_max$ is *annealed* upward across outer iterations
      - start: wide, forgiving low-order comb (few harmonics, big capture basin)
      - end: tight high-order comb, once tracks have locked on
    ],
  )
]
#v(1fr)

#speaker-note[
  Phase-slope frequency update: the phase drift of the demodulated envelope
  between consecutive samples gives a frequency correction, per harmonic.
  Fisher-weighted fusion combines all harmonics' estimates by how informative
  they are. k_max annealing widens then narrows the capture basin so the solver
  doesn't get stuck on the wrong harmonic early on.
]

= The full blind-annotation algorithm

#v(0.2em)
#text(size: 0.78em)[
  ```python
  def blind_annotate(audio[M, T], fs, R=4) -> r[R, T_env], confidence:
      # -- steps 1-2: whiten + scan (once) ------------------------------------
      W       = whiten(mean_spectrogram(audio))            # peaks vs local floor
      cands   = [(f0, mean(W[k*f0] for k*f0 <= 1.2kHz))    # comb score
                 for f0 in grid(30..120, step 0.05)]
      cands   = alias_filter(dedup(cands, min_sep=4))      # drop subsets/ratios
      # -- step 3: seed R rotors ----------------------------------------------
      seeds   = top_R_distinct(cands)
      if len(seeds) < R:
          seeds += rescan(W minus teeth(seeds))            # residual re-scan
          seeds += duplicate(lowest_scoring(seeds), ±0.1)  # merged-twin fallback
      r       = constant_tracks(seeds)
      # -- steps 4-5: capture then refine (same loop, two coarseness levels) --
      for cfg in [CAPTURE(bw=7Hz, k=6..12), REFINE(bw=1.5Hz, k=6..30)]:
          for it in 1..N_outer:
              z[r,k,m] = decimate(LP(audio * exp(-i*2π*k*∫r)))     # demodulate
              a        = banded_cholesky(coupled_normal_eqs(z))    # envelopes
              δ[r]     = Σ_k fisher_w(k,z) * phase_slope(a[r,k])   # freq error
              r       += clip(δ, max_step);  k_max = anneal(it)
      # -- step 6: twin resolution across windows -----------------------------
      r = spatial_DP(windowed_candidates(r), mic_weights(geometry))
      return r, comb_confidence(r, W)   # low confidence => stay at init
  ```
]

#speaker-note[
  The whole thing on one slide, in execution order. Steps 1-3 run once: whiten,
  scan, seed (with the residual re-scan for hidden rotors). Steps 4-5 are the
  same alternating loop — envelope solve given frequencies, frequency
  correction given envelopes — run first coarse (capture), then tight
  (refine). Step 6 resolves near-identical twins using spatial evidence across
  windows. The confidence gate refuses to hallucinate when there is no comb.
]

= VK results: telemetry-init refinement and blind re-annotation

#align(center, image("assets/vk_blind_dregon.png", width: 80%))

#figure(
  table(
    columns: 4,
    align: (left, center, center, left),
    stroke: 0.5pt,
    inset: 6pt,
    table.header([*task (blind unless noted)*], [*pooled err#sub[sm] (rev/s)*], [*twins*], [*reading*]),
    [DREGON refine (telemetry-init)], [0.604], [—], [vs. 0.609 without VK refinement],
    [DREGON nosource], [*0.680*], [resolved], [all 4 rotors, no telemetry],
    [DREGON speech-low], [*0.701*], [resolved], [speech interference],
    [DREGON whitenoise], [*0.744*], [not resolved], [auto-knobs arm wins],
    [FLY124 cruise], [*1.03* best-stage / 3.26 final], [—], [next slide],
  ),
)

#speaker-note[
  Main idea here: we definitely can somehow calculate _how good given RPS trajectory set explains the noise_, right?

  THEN IN THE NEXT FEW SLIDES: we step by step, starting from old idea of least-squares VP transform,
  explain in detail, step by step, with formulas, what is our method (Viterbi peak tracking + Vold-Kalman coupled order matching optimization),
  how it works, which steps are involved. An undergrad should be able to follow the presentation and understand everything.
  A final slide should be slide with the results: mean absolute errors for telemetry-init refinement + blind re-annotation on DREGON and Michael's real noise,
  and output RPS trajectories overlaid over real ones (figures).
]

= Blind annotation on FLY124: the failure mode, visibly

#v(0.2em)
#align(center, image("assets/vk_blind_fly124.png", width: 92%))
#v(0.2em)

#keyline[All four rotors tracked at the coarse stage (pooled *1.03*); the
  *refinement* re-captures the weak $approx 81$ rev/s rotor onto the strong
  $91$ comb (final 3.26). Fix in progress: per-track confidence guard between
  stages.]

#speaker-note[
  DREGON's failure mode: the matched-filter scan for candidate combs wasn't
  band-capped, so a harmonic alias (91.3/3 = 30.4, close to a real low rotor
  speed) scored higher than the true comb and seeded the wrong track. Capping
  the scan band and rejecting aliases fixes seeding on both twin pairs.

  Final numbers (rounds 2+3 merged, results/vk_blind_sweep_r2 +
  results/vk_blind_sweep_r3, spatial-DP ladder, band-capped scan): DREGON
  nosource 0.680, speech-low 0.701, whitenoise 0.744 (round-1 read of 22.1 was
  the scan-band bug, not a real SNR floor — the auto-knobs arm wins this
  recording), FLY124 cruise 3.241 (down from 4.0), per-rotor
  [1.27, 1.05, 9.8, 0.85] — only the fast rotor (index 2, analogous to
  DREGON's 81.3) is still missed. Verdict: adopt spatial-DP + auto-knobs
  (vit2dsp ladder), drop the template/completeness/count knob arms (T/C/N),
  no accuracy regression anywhere. A residual re-scan arm targeting that one
  hard rotor is running now.
]

= Speed of algorithm: optimized quite a bit already

#figure(image("assets/vk_speedup_bars.png", width: 65%))

#keyline[Banded Hermitian Cholesky ($2.9 times$) $times$ pair pruning ($1.7 times$) $times$ fixes; results bit-identical.]

Note: full blind annotation requires both phases, and remains quite slow - ~4 seconds for a single second of recording.

#speaker-note[
  Profiled the tracker: refine was 55-58% SuperLU solve time, blind was FFT-bound.
  Switched the coupled normal-equations solve to a banded Hermitian Cholesky
  (zero fill-in, exploits the block structure) and pruned cross-coupling demod
  for track pairs with no shared support or well-separated frequencies. Also
  halved peak RSS. Verified bit-identical to the old solver path on the same
  inputs — this is a pure speed win, no accuracy tradeoff.
]


= Work thread 3: a literature baseline (JASA-GP)

#v(0.5em)
Gaussian process over (mic position, flight condition, time), with a
blade-passing-frequency-informed Fourier kernel:

$ k(t, t') = sum_n sigma_n^2 cos(n Omega (t - t')), quad Omega = "BPF" $

#keyline[Lee et al., *JASA* 159(4):3418, 2026. #h(0.4em) `jasa-flyovers`: NASA 1-Pax quadrotor, 256 ground mics.]

#speaker-note[
  First slide: formulas explaining how GP model works
]

= What is CONA, and how is the synthetic audio computed?

#v(0.2em)
#text(size: 0.95em)[
  *CONA* = physics-based rotor-noise *auralization*: from blade geometry and
  rotor kinematics straight to a pressure waveform at each microphone —
  no recordings involved. Two channels, summed per mic:
]
#v(0.3em)
#text(size: 0.92em)[
  #grid(
    columns: (1fr, 1fr),
    gutter: 1.5em,
    [
      *Tonal — Ffowcs Williams–Hawkings, Farassat 1A.* Each blade element is a
      moving *thickness* + *loading* source; the acoustic pressure is the
      retarded-time integral over the blade surface:
      $
        p'(bold(x), t) = underbrace(p'_T, "thickness: air displaced by the blade") + underbrace(p'_L, "loading: blade-element forces")
      $
      evaluated at emission time $t - r slash c$ per element. Blade-element
      loads come from a BEMT aerodynamic model (chord, twist, polar); rotor
      kinematics prescribe the motion. Deterministic, phase-exact $arrow.r$
      the harmonic comb at multiples of the *blade-passing frequency*
      ($"BPF" = N_"blades" dot "RPS"$).
    ],
    [
      *Broadband — BPM self-noise + Griffin–Lim.* The Brooks–Pope–Marcolini
      model predicts turbulence self-noise (boundary layer, tip, trailing
      edge) as *1/3-octave band levels* per element:
      $ "SPL"_(1/3)(f) = f("boundary-layer params", "Re", "Ma", alpha) $
      Band spectra are interpolated to an STFT magnitude grid; a waveform is
      synthesised by *Griffin–Lim* phase reconstruction (magnitude-only
      $arrow.r$ iterated phase estimate).
    ],
  )
]
#v(0.3em)
#keyline[blade geometry + RPS #sym.arrow BEMT loads #sym.arrow FW-H F1A (tonal) $plus$ BPM #sym.arrow Griffin–Lim (broadband) #sym.arrow free-field sum at each mic. No room, no absorption, no jitter — deviations we must remember.]

#speaker-note[
  CONA is the simulation framework the JASA paper trains its GP on: pure
  physics-based auralization. Left: the tonal path — FW-H with Farassat's
  formulation 1A, thickness + loading integrals at retarded time, driven by
  BEMT blade loads; this produces the exact BPF comb. Right: the broadband
  path — BPM self-noise model gives 1/3-octave levels, Griffin-Lim turns the
  magnitude spectrogram into a waveform. Everything is free-field: no room, no
  atmospheric absorption, no RPS jitter — exactly the gaps we later measure
  against real recordings.
]

= JASA-GP: original data replication

#figure(image("assets/jasa_gp_eval_slim.png", width: 72%), numbering: none, supplement: none)

A faithful replication of the GP rotor-noise model of Lee et al.: `CONA`
auralizes tonal content by FW-H F1A (rigid-blade propagation) plus broadband
content by the Brooks–Pope–Marcolini (BPM) model and Griffin–Lim phase
reconstruction. Original training set: the NASA 1-Pax quadrotor, held-out
flyover $V = 7$ shown above (held-out loudness $9.3 arrow.r 3.0$ dB after 3
alignment fixes).

#speaker-note[
  Second slide: figure + formulas describing CONA framework and how the synthetic audio is being produced by it,
  + illustration of NASA quadrotor and its stats (the original training data described)
]

= Adapting the recipe to our use case

#v(0.5em)

- DREGON and Michael's Matrice-100 geometries reconstructed inside CONA
  (corrected mic-array geometry from earlier in this deck).
- Prescribed *constant* RPS, static-stand protocol, 40–85 rev/s grid.
- 64-mic shell around each drone (denser than our real 8-mic arrays).
- *40-case dataset generated today*: 2 s clips, 44.1 kHz, both drones $times$
  the RPS grid.

#speaker-note[
  Third slide: figures showing DREGON and Michael's drones, which of their characteristics are input into CONA,
  how varying speeds are handled, how the observation points are placed around the drone positions.
]

= GP trained on the CONA data

#grid(
  columns: (1.4fr, 1fr),
  gutter: 1.5em,
  align: (center + horizon, left + horizon),
  figure(image("assets/gp_dregon_overlay.png", width: 100%)),
  [
    #table(
      columns: 3,
      align: (left, center, center),
      stroke: 0.5pt,
      table.header([*split*], [*corr*], [*rel. res.*]),
      [seed holdout], [0.75], [3.7%],
      [rps interp. (60 rev/s)], [0.87], [1.8%],
    )
    #v(0.5em)
    matrice100 (Michael's rig) training queued next.
  ],
)

#speaker-note[
  DREGON GP fit: 9 training RPS values 40-85 rev/s, holding out 60 rev/s for
  interpolation and a seed for held-out-draw generalisation. Both splits fit
  well; interpolation is actually better than the seed holdout because 60 rev/s
  sits comfortably inside the training grid. matrice100 not yet trained.
]

= Results: the four-way comparison

#text(size: 0.95em)[Same ground-truth RPS trajectory, same real microphone position — real,
  CONA, our deep generator, and the GP, compared head to head:]

#figure(image("assets/four_way_spectrograms.png", width: 92%))

#v(0.2em)
#figure(
  table(
    columns: 4,
    align: (left, center, center, center),
    stroke: 0.5pt,
    inset: 6pt,
    table.header([*source*], [*comb err low/mid/high (dB)* #sym.arrow.b], [*msSTFT* #sym.arrow.b], [*verdict*]),
    [deep generator], [7.0 / 8.2 / 9.5], [6.9], [closest by a decisive margin],
    [GP], [39.1 / 26.8 / 19.1], [15.6], [clean but structurally limited],
    [CONA], [50.1 / 52.9 / 52.0], [30.9], [farthest from real],
  ),
)
#text(size: 0.65em)[DREGON free-flight cruise, rotor-mean 80.7 rev/s (75.2–86.5 per rotor).]

#speaker-note[
  Spectrogram comparison: real, best deep generator, CONA auralized over same RPS trajectories, GP generation results over same RPS trajectories.
  For generated data, mrstft loss against real is given under each spectrogram.
]

= Discussion: is the GP bad, or its data?

#v(1fr)

#figure(
  table(
    columns: 2,
    align: (left, left),
    stroke: 0.5pt,
    inset: 10pt,
    table.header([*family*], [*where the gap comes from*]),
    [CONA (structural)],
    [identical-blade FW-H #sym.arrow BPF-only, even-$k$ harmonics only; no jitter linewidth; resolution-limited comb; broadband truncation bug],

    [GP (regression)], [high-harmonic shrinkage; inherits every CONA structural gap it was trained on],
    [deep generator (residual)], [best by far, but still too coherent across mics (the wind-channel target)],
  ),
)

#v(0.8em)
#text(size: 1.05em)[*All three synthetics are far too spatially coherent* — real MSC 0.05,
  deep-generator 0.69, GP 0.03–0.37, CONA 0.11–0.29 (low band). That's the one
  gap none of them have closed yet.]

#v(1fr)

#speaker-note[
  What we can say about the GP generator performance? Is it good or is training data (CONA recordings) just not matching the real one?
]

= Initiated work: wider baselines on noise suppression

#v(0.8em)
#text(size: 1.05em)[Before any RPS-informed claim, establish a *blind* (no-RPS) modern floor:
  two passes per architecture, drone-only vs. category-uniform harmonic noise.]
#v(1em)

#align(center)[#text(size: 1.05em)[
  #table(
    columns: 3,
    align: (left, left, right),
    stroke: 0.7pt,
    inset: 10pt,
    table.header([*arch*], [*family*], [*params*]),
    [Edge-BS-RoFormer], [band-split transformer (in-house SOTA)], [—],
    [TF-GridNet], [dense full+sub-band dual-path], [8.38M],
    [MP-SENet], [parallel magnitude+phase], [1.71M],
    [DCUNet], [complex UNet (continuity anchor)], [—],
    [SGMSE+], [score-based diffusion, from scratch], [65.6M],
    [noisy / Wiener], [trivial anchors], [—],
  )
]]
#v(1em)

#keyline[5 archs $times$ 2 passes; valid sets published; anchor floors done; training runs queued.]

#speaker-note[
  We need to understand now - pivot away from noise suppression completely or not, and for that,
  we need to at least show usefulness of reusing RPS data for noise suppression; but even before that,
  we need proper up-to-date audio-only baselines. Here is the table of which models we scheduled to run
  and the training protocol.
]

= Initiated work: RPS predictor achieving parity with VK optimization

#v(0.3em)
#align(center, image("assets/rps_predictor_quadrant.png", height: 82%))

#speaker-note[
  Main problem with VK optimization (speed) and limits to which we can optimize (numerical estimations);
  quick bullet list of ideas on how to improve the RPS predictor to match VK performance (while still being fast).
]

= Takeaways

#v(1fr)
#text(size: 1.15em)[
  - *Geometry fix:* validated and permanent (v1 best in flight).
  - *Per-rotor + wind:* well-motivated, not yet clear wins; wind's realism gap is
    now *measured* (MSC 0.05 real vs 0.69 deep-gen), not just suspected.
  - *Blind tracking:* now $approx 10 times$ faster and de-biased in seeding;
    blind pipeline final: 0.68–0.74 rev/s DREGON, 3.24 FLY124 (one rotor open).
  - *Deep generator confirmed best-in-class* against a faithful literature baseline
    (comb error 7–10 dB vs 19–53 dB for GP/CONA).
  - *Two programs running* toward the eventual speech-enhancement decision: blind
    SE baselines, and a VK-parity RPS predictor.
]
#v(1fr)

#speaker-note[
  The loop: thread 1's per-rotor template is exactly what thread 2's twin failure
  needs, so the next step serves both. Wind isn't refuted — it's un-testable at
  hover; free-flight airspeed + a coherence-aware loss is what proves it. The
  regime lesson (score flight on flight clips) nearly fooled us via the idle-heavy
  valid set. Headline: the generator's assumptions are now audited against real
  data, one is fixed for good, and tracker and generator inform each other.
]
