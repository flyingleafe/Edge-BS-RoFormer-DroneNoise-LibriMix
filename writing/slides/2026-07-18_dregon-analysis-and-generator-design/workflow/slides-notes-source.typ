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

= This week: the rotor comb, three ways

#v(1em)

+ *Generative model* #sym.dash.em audit + fix the physics-structured generator.
  + Found bugs with array geometry annotations...
  + Attempts to improve the expressivity of the model + wind noise modeling attempt
+ *Blind order tracking* #sym.dash.em per-rotor RPS from audio alone.
  + Can we optimize trajectories __directly__ to simply match the harmonic comb?
  + How good the result could be with that approach?
+ *Literature baseline* #sym.dash.em reproduce JASA-GP as a reference
  + Also fit GP model to our drone setups (approximate) and see results

#speaker-note[
  Here is what I did
]

= A physics-structured generator, answerable to data

#v(0.5em)
$ y_m (t) = frac(r_"ref", r_m) dot s(t - r_m slash c) $
#v(0.5em)

One emitter per rotor, propagated free-field. Three assumptions testable on DREGON:

+ array *geometry* correct
+ four rotors share *one emitter code*
+ a *coherent, propagating* field is complete

#speaker-note[
  Problem: why mid-range harmonics are so bad? Let us try to understand
]

= Assumption 1 — geometry: a small error is not benign

#figure(image("assets/geo_propagation_phase.png", width: 72%))

#keyline[Symptom: systematically weak mid-frequency harmonics. #h(0.6em) $Delta phi.alt(f) = 2 pi f delta$.]

#speaker-note[
  Hypothesis 1: rotor and mic positions are not precise, so TDOA errors between real and simulated sound are large
  in comparison with mid-range harmonic fequencies
]

= Geometry errors found (huge!) in DREGON annotations

#figure(image("assets/geo_frame_alignment.png", width: 74%))

#keyline[Predicted vs. measured TDOA: $r = -0.56 arrow.r +0.93$ at $183 degree$.]

#speaker-note[
  How we checked: on DREGON, we have single-motor recordings, so can compare TDOA
  estimates from GCC-PHAT with provided mic positions easily enough (showing figures).
  And turns out the microphones should be rotated 180° about z for TDOAs to make sense!
  (show relevant correlation plots + mic + rotor positions 3d plot)
]

= Geometry errors found (silly) in Michael's annotations

#speaker-note[
  In Michael's case, the mic array annotation just had a bug (swapped coordinates) so it was
  as if circular array was put vertically, not horizontally. But this was understood just visually,
  TDOA correlation optimization does not work for Michael's because no single-rotor recordings (hard).
  (see plot of mic + rotor positions)
]

= Geometry: fine calibration

#align(center, image("assets/geo_summary.png", height: 64%))

#keyline[DREGON: bundle adjustment, moves $<= 2.2$ cm. #h(0.6em) Michael's: not identifiable.]

#speaker-note[
  Here we say how we can __optimise__ the mic positions further using TDOA correlation maximization,
  using formulas and the figure of mic positions before and after optimisation.
]

= Hypothesis 2 — rotors are individuals

#figure(image("assets/fig_per_rotor.png", width: 74%))

#keyline[Level-normalised *timbre* differs by *6.8 dB RMS*.]

#speaker-note[
  Show rotor spectras figures (individual recordings) at same speeds, demonstrating that
  harmonics and broadband components have different amplitude distributions.
]

= Treating rotors individually: per-rotor sub-embeddings

#v(1em)
$ z_r = z_"drone" + delta z_r, quad delta z in RR^(R times d) $
#v(1em)

- shared across drones (identity = airframe layout)
- zero-initialised — a strict generalisation of the shared-code model

#speaker-note[
  Show schema of drone noise generator model (see prev slides) with per-rotor sub-embeddings ADDED and emphasized,
  + with formula on how the losses change (how we keep sub-embedding delta small)
]

= Hypothesis 3: wind noise confusing the model

#figure(image("assets/fig_wind_schema.png", width: 78%))

#keyline[Physics fixes where the flow lands; a small learned head fits what it does to a mic.]

#speaker-note[
  Mid-range harmonics were much weaker on DREGON, and DREGON has wind noise, so maybe that is why the model struggles.
  Let's actually add the physically-plausible wind noise model additively on top!
]

= What the data said back: generator variants

#figure(
  table(
    columns: 3,
    align: (left, center, left),
    stroke: 0.5pt,
    table.header([*variant*], [*free-flight MR-STFT* #sym.arrow.t], [*reading*]),
    [OLD (wrong geometry)], [4.51], [pre-fix baseline],
    [*v1 — corrected geometry*], [*5.22*], [*best — geometry fix helps*],
    [v2 #sym.plus per-rotor], [4.82], [best on idle, below v1 in flight],
    [v3 #sym.plus wind], [3.44], [dormant at hover airspeed],
  ),
)

#keyline[Free-flight only ($n = 48$); the idle-heavy full set inverts the ranking.]

#speaker-note[
  Here we display FREE-FLIGHT segments: real and corresponding generations for each model variant -
  old (before modifications), v1 (geometry fix), v2 (per-rotor sub-embeddings), v3 (wind noise model).
  Top row - DREGON, bottom row - Michael's
  Under each spectrogram of generated noise - mrSTFT loss against the real noise.
]

= Generator improvements: discussion

+ TBD

#speaker-note[
  Here we list possible reasons why sub-embeddings and wind noise model worked badly.
]

= Work thread 2: Optimizing for best RPS trajectory

#v(0.8em)

Coupled Vold–Kalman, built from the published math (not PyVKF / GPL-3):

- envelope grid, $approx 200$k unknowns (not the full-rate $approx 16$M)
- *sparse coupling* term between nearby tracks

#speaker-note[
  Main idea here: we definitely can somehow calculate __how good given RPS trajectory set explains the noise__, right?

  THEN IN THE NEXT FEW SLIDES: we step by step, starting from old idea of least-squares VP transform,
  explain in detail, step by step, with formulas, what is our method (Viterbi peak tracking + Vold-Kalman coupled order matching optimization),
  how it works, which steps are involved. An undergrad should be able to follow the presentation and understand everything.
  A final slide should be slide with the results: mean absolute errors for telemetry-init refinement + blind re-annotation on DREGON and Michael's real noise,
  and output RPS trajectories overlaid over real ones (figures).
]

// = Coupling: overlapping rotors explain away shared energy

// #figure(image("assets/vk_coupling_schematic.png", width: 82%))

// #speaker-note[
//   Off-diagonal coupling forces overlapping tracks to split shared spectral energy
//   (explaining-away) instead of both claiming all of it. The earlier heuristic comb
//   refinement lacked this and biased tight rotor pairs toward their mean. Hardest
//   exactly on a twin pair — the next slide.
// ]

// = Blind re-annotation: where it works, where it breaks

// #align(center, image("assets/vk_blind_dregon.png", height: 66%))

// #keyline[DREGON *4/4* @ *0.68 rev/s*. #h(0.8em) FLY124 *3/4*; the twin pair shares one comb.]

// #speaker-note[
//   DREGON free-flight, 25 s, no telemetry: all 4 rotors, pooled error 0.68 rev/s.
//   FLY124 cruise (~74/74/81/91 rev/s): 3 recover (91→0.8, 74→1.0, 81→1.3); the 4th
//   fails because two rotors at ~74/74 share one comb — only one track seeds, the
//   other lands on a spurious ~60 Hz peak. The fix is a shared per-rotor comb prior =
//   the thread-1 sub-embeddings. The two threads meet.
// ]

// = The floor: fine RPS jitter is below resolution

// #v(0.8em)

// $0.68$ rev/s is the achievable *floor*, not a tuning gap.

// - the smooth blind answer fits *better* than the true jittery telemetry
//   ($0.99196$ vs $0.99376$)
// - holds in *every band* after stripping wind (ratios $0.87$–$0.81$)

// #speaker-note[
//   The objective's minimum is genuinely not at the true RPS: a flexible envelope
//   re-absorbs ~0.74 rev/s of real jitter, so the smooth answer has the lower data
//   residual even full-band. Concentrating on the comb (strip wind) doesn't rescue
//   it: high-pass >0.5 kHz ratio 0.874, band-pass 1–3 kHz 0.810 — smooth still wins.
//   Jitter is invisible to the comb model, not merely masked. Bounds any audio-only
//   tracker.
// ]

= Work thread 3: a literature baseline (JASA-GP)

#figure(image("assets/jasa_gp_eval_slim.png", width: 70%))

#keyline[Lee et al., *JASA* 2026. #h(0.4em) `jasa-flyovers`, 256 mics. Held-out $V = 7$.]

#speaker-note[
  First slide: formulas explaining how GP model works
]

= JASA-GP: original data replication

#speaker-note[
  Second slide: figure + formulas describing CONA framework and how the synthetic audio is being produced by it,
  + illustration of NASA quadrotor and its stats (the original training data described)
]

= Adapting the recipe to our use case

#speaker-note[
  Third slide: figures showing DREGON and Michael's drones, which of their characteristics are input into CONA,
  how varying speeds are handled, how the observation points are placed around the drone positions.
]

= Results

#speaker-note[
  Spectrogram comparison: real, best deep generator, CONA auralized over same RPS trajectories, GP generation results over same RPS trajectories.
  For generated data, mrstft loss against real is given under each spectrogram.
]

= Discussion

#speaker-note[
  What we can say about the GP generator performance? Is it good or is training data (CONA recordings) just not matching the real one?
]


= Initiated work: wider baselines on noise suppression

#speaker-note[
  We need to understand now - pivot away from noise suppression completely or not, and for that,
  we need to at least show usefulness of reusing RPS data for noise suppression; but even before that,
  we need proper up-to-date audio-only baselines. Here is the table of which models we scheduled to run
  and the training protocol.
]

= Initiated work: RPS predictor achieving parity with VK optimization

#speaker-note[
  Main problem with VK optimization (speed) and limits to which we can optimize (numerical estimations);
  quick bullet list of ideas on how to improve the RPS predictor to match VK performance (while still being fast).
]

// = JASA-GP: getting the replication faithful

// #figure(image("assets/jasa_gp_loudness.png", width: 62%))

// #keyline[After 3 fixes: correlation $0.41 arrow.r 0.70$, loudness $9.3 arrow.r 3.0$ dB.]

// #speaker-note[
//   Out of the box ~4× too quiet. Fixes: (1) a np.roll sign bug mixed phasor signs
//   before GP averaging → exact coefficient-space alignment; (2) per-signal f0 (a
//   pinned comb drifts ~12 bins at the top harmonic); (3) loudness on the AC part
//   only (CONA carries a large inaudible DC offset). Result matches the paper's
//   held-out fit. Deliverable: jasa_gp_interactive.ipynb — plotly 3D, rotor-speed FM,
//   per-point audio, to A/B against our generator.
// ]

= Takeaways

#v(0.5em)

- *Geometry fix:* validated and permanent (v1 best in flight).
- *Per-rotor + wind:* well-motivated, not yet clear wins.
- *Blind tracking:* $0.68$ rev/s floor; the twin gap points at the per-rotor prior.
- *Baseline:* faithful JASA-GP + a listening tool.

#v(0.5em)
*Next:* per-rotor comb prior (closes the twin gap); free-flight airspeed +
coherence-aware loss (activates wind); valid-set re-annotation.

#speaker-note[
  The loop: thread 1's per-rotor template is exactly what thread 2's twin failure
  needs, so the next step serves both. Wind isn't refuted — it's un-testable at
  hover; free-flight airspeed + a coherence-aware loss is what proves it. The
  regime lesson (score flight on flight clips) nearly fooled us via the idle-heavy
  valid set. Headline: the generator's assumptions are now audited against real
  data, one is fixed for good, and tracker and generator inform each other.
]
