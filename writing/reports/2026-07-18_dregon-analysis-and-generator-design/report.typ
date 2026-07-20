#import "/writing/templates/typst/report.typ": report, author-meta

#let fig(path, caption, width: 98%) = figure(image(path, width: width), caption: caption)
#let spread = read("assets/per_rotor_spread.txt").trim()

#show: report.with(
  title: [From DREGON's audio to the generator's structure:\ geometry, per-rotor identity, and a wind channel],
  authors: (
    "Dmitrii Mukhutdinov": author-meta("project"),
  ),
  affiliations: (
    "project": "Harmonic Noise Suppression Project",
  ),
  abstract: [
    The project's drone-noise generator is a physics-structured model: a
    single emitter per rotor, propagated to every microphone by free-field
    $1/r$ attenuation and a delay. Three structural choices in that model are
    not free parameters — they are answerable to what the DREGON recordings
    actually show. This report walks the evidence for each. *(1) Geometry.*
    A geometry error corrupts the model's inter-mic phase in proportion to
    frequency and can silently cancel whole harmonic bands; the symptom
    (systematically weak mid-frequency harmonics) led us to audit the shipped
    array geometry, and DREGON's turned out frame-mismatched by $approx 183 degree$
    — predicted and measured inter-mic delays are anti-correlated ($-0.56$)
    under the shipped frame and strongly correlated ($+0.93$) after the fix,
    with a bundle adjustment then refining individual mic positions to
    centimetre scale. *(2) Per-rotor identity.* DREGON records each of the four
    motors on its own; once the propagation *level* is normalised out, the four
    rotors' *timbre* — their relative harmonic profile — still differs by
    #spread~dB RMS, so a single shared per-drone code cannot represent them.
    This motivates per-rotor sub-embeddings $z_r = z_"drone" + delta z_r$.
    *(3) A wind channel.* Real ego-noise carries flow noise — the incoherent
    pseudo-sound of downwash hitting a diaphragm — that no shared-emitter +
    propagation model can produce; we describe an additive, physics-gated
    wind-wake channel whose geometric gate predicts DREGON's measured per-mic
    flow floor at Spearman $0.92$. We close with how these three became trained
    model variants and what the held-out data said back.
  ],
  keywords: ("drone noise", "generative model", "microphone array geometry",
             "per-rotor conditioning", "wind noise", "DREGON"),
)

#set heading(numbering: "1.")

= A physics-structured generator, answerable to data <sec-intro>

The generator maps a rotor-speed (RPS) trajectory $r(t)$ and a per-drone
conditioning code $z_"drone"$ to multichannel ego-noise. Each rotor is a single
harmonic-plus-broadband *emitter*; its source waveform is then *propagated* to
each microphone $m$ as a free-field point source,
$ y_m (t) = frac(r_"ref", r_m) dot s(t - r_m slash c), $
i.e. $1/r$ amplitude spreading and a pure propagation delay $r_m slash c$. This
is a deliberately physics-structured model, not a black box — which is a
strength, because every structural assumption it bakes in becomes a *testable*
claim about the data. Three such assumptions are the subject of this report:

+ that the *geometry* $(r_m, r_"rotor")$ feeding propagation is correct
  (@sec-geometry);
+ that the four rotors of a drone are well-described by *one shared* emitter
  code (@sec-perrotor);
+ that a *propagating coherent* field is a complete description of ego-noise
  (@sec-wind).

Each turns out to need correction or extension, and DREGON's recordings — which
uniquely include clean, constant-speed *single-motor* recordings and a
documented array geometry — are what let us see it. We take them in turn, then
report what happened when each became a trained variant (@sec-variants).

= Geometry: the symptom, the discovery, the fix <sec-geometry>

== Why a geometry error is diagnostic, not benign

The propagation stage reads distances and delays straight off the 3-D mic and
rotor positions. A constant position error looks, to first order, like a
constant delay error $delta$, and its effect on phase is *frequency-proportional*:
$ Delta phi.alt (f) = 2 pi f delta. $
A small delay error is invisible at low frequency and catastrophic high up
(@fig-geo-phase): a $15$-sample error at $44.1$ kHz ($delta approx 340 mu s$) is
a mere $12 degree$ at $100$ Hz but $122 degree$ at $1$ kHz, and past $180 degree$
the model's predicted phase has *inverted* — so wherever this feeds a coherent
subtraction, that band does not attenuate, it cancels *itself* out. This is
exactly the fingerprint we were seeing: mid-frequency harmonics of the generator
were systematically weak, the signature of a delay term that is right at DC and
wrong several harmonics up. That symptom is what made us doubt the shipped
geometry in the first place.

#fig("assets/geo_propagation_phase.png", [
  A geometry error is a frequency-proportional corruption. *(a)* A fixed
  $15$-sample delay error produces a phase error growing linearly with
  frequency, crossing the $180 degree$ full-inversion line before $1600$ Hz.
  *(b)* The consequence: how much of a unit signal survives an attempted
  coherent cancellation with a mistuned delay — near-total loss of destructive
  interference by $1.2$–$1.5$ kHz. #emph[(From the Stage-0 geometry-calibration study.)]
]) <fig-geo-phase>

== The discovery: a 180° frame mismatch in DREGON

The direct check — do measured time-differences-of-arrival (TDOA, from GCC-PHAT
on the single-motor recordings) agree with what the shipped `micPos`/`rotorsPos`
predict — failed *systematically*, not noisily: the correlation was $-0.56$.
A truly-wrong geometry produces scatter; a strong *negative* correlation is the
fingerprint of a coordinate-frame mismatch. Sweeping a rigid $z$-rotation of the
mic frame against the rotor frame resolves it cleanly (@fig-geo-frame): the
correlation rises to a sharp $+0.93$ at $183 degree$. DREGON's shipped mic and
rotor arrays are expressed in frames that disagree by (almost exactly) a
$180 degree$ rotation about $z$ — a convention mismatch to reconcile, and once
reconciled, predicted and measured delays agree well. All downstream work starts
from this $183 degree$-corrected nominal.

#fig("assets/geo_frame_alignment.png", [
  DREGON's shipped geometry is frame-mismatched by $approx 183 degree$.
  *(a)* Correlation between predicted and measured TDOA vs. an applied mic-frame
  $z$-rotation: sharply peaked at $183 degree$ ($r = 0.93$), strongly *negative*
  at the shipped $0 degree$ ($r = -0.56$). *(b)* Measured vs. predicted TDOA,
  shipped (anti-correlated) vs. corrected (tight around the diagonal).
], width: 92%) <fig-geo-frame>

== The fine fix, and an honest limit

Frame correction is global; it says nothing about individual mic placement
inside that frame. A coherence-weighted phase *bundle adjustment* over the
$400$–$800$ Hz clean band (gauge-fixed by anchoring to the corrected nominal)
refines each mic, cutting the phase residual $38.3 degree arrow.r 28.8 degree$
with only centimetre-scale moves (all $<= 2.2$ cm); a synthetic control with
known ground truth recovers positions to $0.36$ cm, confirming the optimiser is
correct and the residual $approx 25 degree$ is real free-field model mismatch
(no directivity, no near-field), not a bug. The same recipe *fails* on the
second dataset (Michael's DJI Matrice-100 rig): there a plane-coding bug
(vertical ring coded for a horizontal one) is fixable by inspection from the rig
photo, but audio self-calibration is *not* identifiable, because all four rotors
sit clustered on one side of the array — there is no angular diversity to
triangulate against. @fig-geo-summary shows both corrected geometries. The
travelling lesson: geometry can be recovered from audio, but only when the
sources illuminate the array from enough directions.

#fig("assets/geo_summary.png", [
  Corrected 3-D geometry, both drones. DREGON (left): the audio bundle
  adjustment moves mics by $<= 2.2$ cm from the frame-corrected nominal — a
  fine-tune of an already-good geometry. Michael's (right): the only correction
  is the vertical$arrow.r$horizontal plane swap read from the rig photo; audio
  refinement is not identifiable here and is not attempted.
], width: 86%) <fig-geo-summary>

Both fixes are now baked into `dregon.get_geometry` / `michaels.get_geometry`
and the published `frames:` datasets, so every load-time consumer — including
generator training — sees the corrected geometry. Retraining the generator on it
is the subject of @sec-variants.

= Are the four rotors one source? <sec-perrotor>

The generator conditions the emitter on a single per-drone code $z_"drone"$,
implicitly assuming a drone's four rotors are interchangeable — same tone, same
broadband texture, differing only in *position* (which propagation already
handles). DREGON lets us test that assumption directly, because it ships
constant-speed recordings of *each motor spun on its own*. We take each rotor's
single-motor recording at a matched speed ($70$ Hz fundamental), read it at that
rotor's nearest microphone (best isolates its own field), and *normalise each
spectrum by its own fundamental* — dividing out the $1/r$ propagation level so
that what remains is the rotor's *timbre*: the relative heights of its harmonics
and the shape of its broadband floor.

They are not the same (@fig-perrotor). Rotor 2 carries a markedly richer, more
slowly-decaying harmonic comb than rotors 1, 3 and 4, which themselves differ in
the detail of their profiles; across the first twelve harmonics the four rotors'
level-normalised profiles spread by #spread~dB RMS. Some of this folds in
residual geometry (nearest-mic distance is not identical across rotors) and
some is genuine unit-to-unit variation (manufacturing, wear, mounting) — but the
direction of the conclusion is unaffected either way: *a single shared code
cannot represent four measurably different sources.*

#fig("assets/fig_per_rotor.png", [
  DREGON single-motor recordings, level-normalised so only *timbre* remains.
  *(a)* The four rotors' spectra at $70$ Hz fundamental, each divided by its own
  fundamental peak (vertical lines mark harmonics $k dot 70$ Hz). *(b)* The
  per-rotor harmonic profile (level relative to the fundamental) — rotor 2 sits
  well above the rest and the profiles diverge with $k$; inter-rotor spread
  #spread~dB RMS across harmonics $2$–$12$.
]) <fig-perrotor>

This motivates *per-rotor sub-embeddings*: replace the shared emitter code with
$ z_r = z_"drone" + delta z_r, quad delta z in RR^(R times d), $
a learnable per-rotor delta *shared across drones* (rotor identity — position and
manufacturing timbre — is a property of the airframe layout, not of which drone)
and *zero-initialised*, so training starts identical to the shared-code model (a
strict generalisation) and each rotor's code then diverges under its own
gradients. Whether the extra capacity actually pays off is an empirical question
we return to in @sec-variants — the analysis here establishes only that the
*need* is real.

= A wind channel: what propagation cannot produce <sec-wind>

The coherent generator, however well-calibrated, can only produce a
*propagating* field: one emitter per rotor, spread by $1/r$ with an inter-mic
delay, hence *coherent* across microphones (a well-defined phase relationship
between channels). Real ego-noise carries a second thing it structurally cannot:
*flow noise* — the pseudo-sound of a rotor's turbulent downwash striking a
diaphragm directly. Flow noise does not propagate (no $1/r$, no inter-mic delay)
and is *spatially incoherent* (its magnitude-squared coherence $gamma^2$
collapses at low frequency), so it must live in its own additive channel outside
the coherent path.

We add exactly that: a *wind-wake channel* built on the principle that *physics
should decide where the air flows and how fast, while only a small learned head
decides what that flow does to a microphone* (@fig-wind). Three modules in
series feed an additive per-mic mix:

/ A — RPS to airspeed (physics, grey-box): the four rotor speeds are a
  quadrotor's control inputs, so a calibrated rigid-body model turns them into
  the body-frame relative wind $V_"rel"(t)$ that bends the wake. At hover or on
  a static rig $V_"rel" = 0$ and this module is skipped.

/ B — wake flow field (physics): each rotor emits a downwash column along its
  thrust axis, convecting along $hat(c)_r = "normalize"(2 v_i hat(a)_r + V_"rel")$
  (the far-wake $2 v_i$ speed plus freestream); a microphone's local flow speed is
  the induced velocity $v_i$ times a smooth *in-column gate* $g$ (a downstream
  $sigma(s slash "soft")$, a Gaussian radial decay $exp(-rho^2 slash 2(alpha R)^2)$,
  and a mild far-field falloff). Rotors superpose: $U_m (t) = sum_r v_i g$. The
  only free parameters are three interpretable aero constants; everything is
  closed-form and differentiable in the calibrated positions.

/ C — flow to microphone transduction (learned): the *only* part fit from audio —
  how a flow of speed $U$ becomes low-frequency pressure. Dynamic pressure
  $q = 1/2 rho U^2$ sets the level, $U slash ell$ sets a corner frequency, and a
  learned low-pass shapes the band. A slow Ornstein–Uhlenbeck envelope on $U$
  models wake meander (gust intermittency). The noise is realised *independently
  per microphone*, so the channel is incoherent by construction.

#fig("assets/fig_wind_schema.png", [
  The wind-wake channel. Physics (modules A, B, on the fixed calibrated geometry)
  decides *where* the flow lands and *how fast*; a small learned head (module C)
  decides only *what* it does to a diaphragm. The incoherent per-mic output
  $y_"wind"$ is summed onto the coherent generator's field.
], width: 96%) <fig-wind>

Because exposure is a *geometric* function of the calibrated positions, the
channel needs no per-array switches: an array *inside* the wake gets gusts and an
array *above/forward* of the disk gets near-silence, automatically. That is the
design's central generalization claim, and it is exactly what a CPU-only
de-risk (`scripts/wind_wake_validation.py`) checks against DREGON's constant-speed
single-motor recordings: the geometric wake gate predicts the measured per-mic
low-band broadband floor at *Spearman $0.92$* (Pearson $0.97$, $n = 64$
mic-observations) and *beats* a plain $1/r$ proximity control ($0.74$), while
Michael's out-of-wake array (mics above the disk) is predicted near-silent
(max exposure $0.006$ m/s $approx 0$). The gate carries real predictive power
beyond mere closeness, and it generalises across arrays for free.

= From analysis to variants, and what the data said back <sec-variants>

Each of the three findings above became a single-addition variant of the E6
generator backbone, retrained on the corrected-geometry swapped DREGON +
Michael's split and scored by the MR-STFT quality metric on the *free-flight*
subset (mean RPS $>= 45$, $n = 48$) — the regime we care about, since the split
holds out takeoff/idle where a magnitude metric mostly rewards reproducing
silence.

#figure(
  table(
    columns: (auto, auto, auto),
    align: (left, center, left),
    table.header([*variant*], [*flight mrstft* #sym.arrow.t], [*reading*]),
    [OLD (wrong geometry)], [4.51], [pre-fix baseline],
    [*v1 — corrected geometry*], [*5.22*], [best — the geometry fix helps, as physics demands],
    [v2 — + per-rotor sub-emb.], [4.82], [helped on idle, neutral/below v1 in flight],
    [v3 — + wind channel], [3.44], [weakest — wind dormant at hover ($V_"rel"=0$)],
  ),
  caption: [Free-flight MR-STFT (higher is better). Full detail, bar chart and
  real-vs-generated spectrograms are in the companion report
  #emph[Retraining the drone-noise generator on corrected microphone-array geometry]
  (`2026-07-17_generator-corrected-geometry-variants`).],
) <tab-variants>

The data answered each hypothesis on its own terms. *Geometry* (v1) is
vindicated: once idle is excluded, the corrected-geometry model clearly beats the
wrong-geometry one ($5.22$ vs $4.51$), the physically expected result, and the
correction stays. *Per-rotor identity* is real in the data (@sec-perrotor) but
the extra embedding capacity did *not* convert to a flight-metric win (v2 $4.82$,
below v1): it helps model start-up/idle asymmetries between rotors — where subtle
per-rotor differences matter most — but the four rotors are near-identical in
sustained cruise, so on this split it is at best neutral. We would re-test it on a
flight-heavy split before adopting it. *The wind channel* is architecturally
sound and physically validated ($0.92$ gate correlation), but it is dormant at
the hover airspeed this split provides, so it only adds an incoherent floor that
dilutes the coherent fit (v3 $3.44$); its payoff is contingent on free-flight
data and a coherence-aware objective the current single-channel magnitude metric
cannot supply.

= Takeaways <sec-takeaways>

- *A physics-structured generator makes its assumptions falsifiable.* Weak
  mid-frequency harmonics were not a tuning nuisance; they were the readable
  symptom of a geometry frame bug, which the audio then confirmed and fixed.
- *The geometry correction is validated and permanent* — it improves the
  generator on the flight regime and underpins the project's
  propagation/localization goals.
- *The four rotors are measurably distinct sources* (#spread~dB RMS timbre
  spread), so the per-rotor structure is well-motivated; whether it *helps* is
  regime-dependent and not yet a clear win.
- *Flow noise needs its own incoherent channel*; the physics-gated design
  predicts real per-mic flow noise and generalises across arrays, but proving its
  worth needs free-flight airspeed and a coherence-aware loss — the concrete next
  step for that line of work.
