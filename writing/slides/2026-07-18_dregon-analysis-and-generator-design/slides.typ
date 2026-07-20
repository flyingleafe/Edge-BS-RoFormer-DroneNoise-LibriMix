#import "@preview/touying:0.7.4": *
#import "/writing/templates/typst/slides.typ": hns-slides

// Notes-visible build: `make notes` (typst --input notes=1). Default build is clean.
#let notes-mode = sys.inputs.at("notes", default: none)

#show: hns-slides.with(
  title: [The rotor comb, three ways],
  subtitle: [A physics-structured generator #sym.dot.c blind order tracking #sym.dot.c a literature baseline],
  author: [Dmitrii Mukhutdinov],
  date: [2026-07-18],
  show-notes: if notes-mode != none { bottom } else { none },
)

#let keyline(body) = align(center, text(size: 1.05em, body))

= This week: the rotor comb, three ways

#v(1em)

+ *Generative model* #sym.dash.em audit + fix the physics-structured generator.
+ *Blind order tracking* #sym.dash.em per-rotor RPS from audio alone.
+ *Literature baseline* #sym.dash.em reproduce JASA-GP as a reference.

#speaker-note[
  The week's recurring object is the *rotor comb*: a rotor at N rev/s radiates a
  tone at N Hz plus harmonics — four rotors, four interleaved combs.

  Why a generator: real drone-corrupted speech is scarce; we need training data
  across 0 to −30 dB SNR, so a faithful ego-noise generator is a real research
  object.

  The threads couple: thread 1 yields a per-rotor emitter template; thread 2's one
  failure mode needs exactly that template; thread 3 is an independent external
  check.
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
  Emitter = harmonic comb + broadband per rotor; propagation = $1\/r$ amplitude +
  $r\/c$ delay per mic. A physics-structured model is a strength: each assumption
  is falsifiable. DREGON is the testbed — constant-speed single-motor recordings +
  a documented array geometry. The next slides audit the three assumptions in
  order. Each needed a correction or extension.
]

= Assumption 1 — geometry: a small error is not benign

#figure(image("assets/geo_propagation_phase.png", width: 72%))

#keyline[Symptom: systematically weak mid-frequency harmonics. #h(0.6em) $Delta phi.alt(f) = 2 pi f delta$.]

#speaker-note[
  A position error is a delay error δ; phase effect ∝ frequency (Δφ = 2π f δ).
  15-sample error at 44.1 kHz (≈340 µs) = 12° at 100 Hz, 122° at 1 kHz, crosses
  180° (full inversion) before 1.6 kHz. Past inversion a coherent subtraction adds
  instead of cancelling. Hence the weak mid-frequency harmonics — the readable
  symptom that prompted the geometry audit.
]

= Geometry: a 180° array-frame mismatch

#figure(image("assets/geo_frame_alignment.png", width: 74%))

#keyline[Predicted vs. measured TDOA: $r = -0.56 arrow.r +0.93$ at $183 degree$.]

#speaker-note[
  Predicted TDOA (shipped positions) vs measured (GCC-PHAT on single-motor cuts)
  anti-correlate (−0.56) — the fingerprint of a frame flip, not random error.
  Sweeping a rigid mic-frame z-rotation peaks at +0.93 at 183° ≈ 180°: micPos and
  rotorsPos sit in frames ~180° apart about z. All downstream work uses the
  corrected frame.
]

= Geometry: fine calibration, and one limit

#align(center, image("assets/geo_summary.png", height: 64%))

#keyline[DREGON: bundle adjustment, moves $<= 2.2$ cm. #h(0.6em) Michael's: not identifiable.]

#speaker-note[
  Bundle adjustment refines each mic within the corrected frame: residual 38.3° →
  28.8°, moves ≤ 2.2 cm; synthetic control recovers to 0.36 cm (optimiser correct;
  leftover ~25° is free-field model mismatch, not a bug). Michael's: a
  vertical→horizontal plane bug fixable from the rig photo, but audio
  self-calibration is not identifiable — all rotors on one side, no angular
  diversity. Both fixes now in get_geometry + the published `frames:` datasets.
]

= Assumption 2 — are the four rotors one source?

#figure(image("assets/fig_per_rotor.png", width: 74%))

#keyline[Level-normalised *timbre* differs by *6.8 dB RMS*.]

#speaker-note[
  Each motor's solo recording at a matched 70 Hz fundamental, read at its nearest
  mic, divided by its own fundamental (removes 1/r level → leaves timbre). Rotor 2
  has a much richer comb; spread 6.8 dB RMS over harmonics 2–12. Caveat: folds in
  residual nearest-mic geometry + real unit-to-unit variation — either way, one
  shared code can't represent four distinct sources.
]

= Assumption 2 — per-rotor sub-embeddings

#v(1em)
$ z_r = z_"drone" + delta z_r, quad delta z in RR^(R times d) $
#v(1em)

- shared across drones (identity = airframe layout)
- zero-initialised — a strict generalisation of the shared-code model

#speaker-note[
  $delta z_r$ = small per-rotor delta, one per rotor. Zero-init → at start exactly
  the old shared-code model, so it can't hurt and only adds capacity. Shared across
  drones because rotor identity is an airframe property. The analysis proves the
  need; the variant scoreboard decides whether it pays off.
]

= Assumption 3 — a wind channel (incoherent flow noise)

#figure(image("assets/fig_wind_schema.png", width: 78%))

#keyline[Physics fixes where the flow lands; a small learned head fits what it does to a mic.]

#speaker-note[
  Flow noise doesn't propagate (no 1/r, no delay) and is spatially incoherent
  (γ²→0), so it can't come from the coherent path — it needs an additive channel.
  A: RPS→airspeed (physics, zero at hover); B: wake flow field (physics, geometric
  in-column gate); C: flow→pressure (the only learned part; incoherent per mic).
  De-risk: the geometric gate predicts DREGON's measured per-mic floor at Spearman
  0.92 vs 1/r 0.74; Michael's out-of-wake predicted near-silent. Exposure comes for
  free from geometry — the generalisation claim.
]

= What the data said back: generator variants

#figure(
  table(
    columns: 3, align: (left, center, left), stroke: 0.5pt,
    table.header([*variant*], [*free-flight MR-STFT* #sym.arrow.t], [*reading*]),
    [OLD (wrong geometry)], [4.51], [pre-fix baseline],
    [*v1 — corrected geometry*], [*5.22*], [*best — geometry fix helps*],
    [v2 #sym.plus per-rotor], [4.82], [best on idle, below v1 in flight],
    [v3 #sym.plus wind], [3.44], [dormant at hover airspeed],
  ),
)

#keyline[Free-flight only ($n = 48$); the idle-heavy full set inverts the ranking.]

#speaker-note[
  MR-STFT = multi-scale spectrogram match, higher better. Free-flight only (RPS ≥
  45, n = 48) because on idle a magnitude metric rewards reproducing silence, which
  inverts the ranking. Geometry (5.22 > 4.51): permanent. Per-rotor (4.82): rotors
  distinct but near-identical in cruise → helps idle/startup, not flight; re-test
  on flight-heavy data. Wind (3.44): dormant at hover, adds an incoherent floor
  that dilutes the fit; needs free-flight airspeed + a coherence-aware loss. Gate
  itself validated at Spearman 0.92. Detail: companion report 2026-07-17.
]

= Thread 2 — blind per-rotor order tracking

#v(0.8em)

Coupled Vold–Kalman, built from the published math (not PyVKF / GPL-3):

- envelope grid, $approx 200$k unknowns (not the full-rate $approx 16$M)
- *sparse coupling* term between nearby tracks

#speaker-note[
  Order tracking = following each comb's frequency over time. PyVKF is GPL-3 and
  solves the full audio-rate system (~16M unknowns per 25 s clip) with frequencies
  as inputs, not tracked. Ours works on a decimated envelope grid (~200k) and adds
  a coupling term — the new part, next slide.
]

= Coupling: overlapping rotors explain away shared energy

#figure(image("assets/vk_coupling_schematic.png", width: 82%))

#speaker-note[
  Off-diagonal coupling forces overlapping tracks to split shared spectral energy
  (explaining-away) instead of both claiming all of it. The earlier heuristic comb
  refinement lacked this and biased tight rotor pairs toward their mean. Hardest
  exactly on a twin pair — the next slide.
]

= Blind re-annotation: where it works, where it breaks

#align(center, image("assets/vk_blind_dregon.png", height: 66%))

#keyline[DREGON *4/4* @ *0.68 rev/s*. #h(0.8em) FLY124 *3/4*; the twin pair shares one comb.]

#speaker-note[
  DREGON free-flight, 25 s, no telemetry: all 4 rotors, pooled error 0.68 rev/s.
  FLY124 cruise (~74/74/81/91 rev/s): 3 recover (91→0.8, 74→1.0, 81→1.3); the 4th
  fails because two rotors at ~74/74 share one comb — only one track seeds, the
  other lands on a spurious ~60 Hz peak. The fix is a shared per-rotor comb prior =
  the thread-1 sub-embeddings. The two threads meet.
]

= The floor: fine RPS jitter is below resolution

#v(0.8em)

$0.68$ rev/s is the achievable *floor*, not a tuning gap.

- the smooth blind answer fits *better* than the true jittery telemetry
  ($0.99196$ vs $0.99376$)
- holds in *every band* after stripping wind (ratios $0.87$–$0.81$)

#speaker-note[
  The objective's minimum is genuinely not at the true RPS: a flexible envelope
  re-absorbs ~0.74 rev/s of real jitter, so the smooth answer has the lower data
  residual even full-band. Concentrating on the comb (strip wind) doesn't rescue
  it: high-pass >0.5 kHz ratio 0.874, band-pass 1–3 kHz 0.810 — smooth still wins.
  Jitter is invisible to the comb model, not merely masked. Bounds any audio-only
  tracker.
]

= Thread 3 — a literature baseline (JASA-GP)

#figure(image("assets/jasa_gp_eval_slim.png", width: 70%))

#keyline[Lee et al., *JASA* 2026. #h(0.4em) `jasa-flyovers`, 256 mics. Held-out $V = 7$.]

#speaker-note[
  Lee et al., JASA 159(4):3418, 2026; jasa-flyovers = 10 CONA quadrotor flyovers,
  V = 1..10 m/s, 256 ground mics. Pipeline: de-Doppler → estimate f0 →
  least-squares Fourier coefficients at k·f0 (24 harmonics) → Matérn-5/2 GP over
  (x, y, V); broadband separate. The literature's strongest comb model — a fair
  baseline and an independent generation route vs our emitter + propagation.
]

= JASA-GP: getting the replication faithful

#figure(image("assets/jasa_gp_loudness.png", width: 62%))

#keyline[After 3 fixes: correlation $0.41 arrow.r 0.70$, loudness $9.3 arrow.r 3.0$ dB.]

#speaker-note[
  Out of the box ~4× too quiet. Fixes: (1) a np.roll sign bug mixed phasor signs
  before GP averaging → exact coefficient-space alignment; (2) per-signal f0 (a
  pinned comb drifts ~12 bins at the top harmonic); (3) loudness on the AC part
  only (CONA carries a large inaudible DC offset). Result matches the paper's
  held-out fit. Deliverable: jasa_gp_interactive.ipynb — plotly 3D, rotor-speed FM,
  per-point audio, to A/B against our generator.
]

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
