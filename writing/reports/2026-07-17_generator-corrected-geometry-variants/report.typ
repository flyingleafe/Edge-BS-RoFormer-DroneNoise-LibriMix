#import "/writing/templates/typst/report.typ": report, author-meta

#show: report.with(
  title: [Retraining the drone-noise generator on corrected microphone-array geometry],
  authors: (
    "Harmonic Noise Suppression Project": author-meta("project"),
  ),
  affiliations: (
    "project": "Harmonic Noise Suppression Project",
  ),
  abstract: [
    We retrain the RPS-conditioned drone-noise generator on the *corrected* DREGON +
    Michael's microphone-array geometry — the 180° mic-frame fix established by the
    Stage-0 self-calibration study — and evaluate three variants against the old
    wrong-geometry generator by the MR-STFT quality metric on the *free-flight*
    (mean RPS $>= 45$) subset of the swapped DREGON + Michael's validation set:
    (v1) the corrected-geometry baseline, (v2) v1 plus learnable *per-rotor
    sub-embeddings* $z_r = z_"drone" + delta z_r$, and (v3) v1 plus an additive,
    physics-gated *wind-wake channel*. Restricting to free flight is essential: the
    swapped split's ``val_at_start`` holds out the *start* of each recording, which
    is takeoff/idle, and scoring there is dominated by near-silence and swamps the
    effects we care about. On free-flight clips the *corrected-geometry* model (v1,
    mrstft $5.22$) clearly beats the old wrong-geometry generator ($4.51$) — the
    geometry fix helps, as physics demands. Per-rotor sub-embeddings do *not* beat
    plain v1 in flight ($4.82$; they had looked best only on the idle-dominated
    set), and the wind channel is the weakest variant ($3.44$), consistent with its
    flow field being dormant at the zero relative airspeed of hover.
  ],
  keywords: ("drone noise", "generative model", "microphone array", "per-rotor conditioning", "wind noise"),
)

= Introduction

The project's drone-noise generator maps rotor-speed (RPS) trajectories and a
per-drone conditioning code to multichannel ego-noise: a single-rotor harmonic +
broadband emitter produces per-rotor source waveforms, which are then *propagated*
to each microphone by free-field $1\/r$ attenuation and a fractional delay
$tau = r\/c$ (the position-aware generator, `models.generative.positional_harmonic_gen`,
differentiable in position). The propagation stage depends entirely on the
microphone-array geometry.

The Stage-0 self-calibration study found that DREGON's shipped microphone frame is
rotated *180° about $z$* relative to its rotor frame: using the shipped geometry,
free-field propagation predicts inter-mic delays *anti-correlated* with the
measured GCC-PHAT TDOAs ($r = -0.55$), and negating the mic $x, y$ restores
agreement ($r = +0.93$). Michael's array was likewise re-derived as a *horizontal*
ring above the rotors. Both fixes are now applied in code
(`dregon.get_geometry` / `michaels.get_geometry`) and baked into the published
`frames:` datasets, so every load-time consumer — including training — now sees
the corrected geometry.

This report asks a focused question: *with the geometry now corrected, does
retraining the generator — optionally with new architectural innovations —
improve how well it reproduces real ego-noise, measured by valid MSSTFT, relative
to the old wrong-geometry generator?* We train three variants on the "swapped"
DREGON + Michael's split and compare them, plus the old generator, on the same
corrected-geometry validation set.

= Variants

All variants share the E6 winner backbone: a per-drone codebook
(`[dregon, michaels]`, code dim 16), a learnable per-drone RPS-jitter linewidth,
spectral-norm FiLM conditioning, and vicinal $z$-noise regularisation. They differ
by a single architectural change each, so the comparison is a clean ablation.

/ OLD (wrong geom): the previous generator (`e6_noisegen_jitter_latreg_perdrone`),
  trained *before* the geometry fix. This is the baseline the new variants must beat.

/ v1 — corrected geometry: the same architecture, retrained on the corrected
  geometry. Isolates the effect of the geometry fix alone.

/ v2 — per-rotor sub-embeddings: v1 plus a learnable per-rotor delta, so the
  emitter code becomes $z_r = z_"drone" + delta z_r$. The delta $delta z in
  RR^(R times d)$ is *shared across drones* (rotor identity — position and
  manufacturing timbre — is drone-independent) and zero-initialised, so training
  starts identical to v1 (a strict generalisation) and each rotor's code then
  diverges under its own gradients. This lets the four rotors of a drone differ in
  timbre instead of sharing one code.

/ v3 — wind-wake channel: v1 plus an additive, spatially-incoherent flow-noise
  channel — the pseudo-sound of turbulent rotor downwash striking a diaphragm,
  which no shared-emitter + propagation model can produce. Physics places the air
  (a closed-form bent-wake-column gate from the rotor$->$mic vectors decides which
  mics sit in a downwash) and only a small flow$->$microphone transduction is
  learned. Its per-mic noise is realised independently, so the channel is
  incoherent by construction, and it is summed onto the coherent output.

== Wind-channel de-risk

Before wiring the wind channel into training, we validated its central premise —
that geometry alone predicts *where* flow noise lands — on DREGON's constant-speed
single-motor recordings. The geometric wake gate predicts the measured per-mic
low-band broadband floor at Spearman $0.92$ (Pearson $0.97$, $n = 64$
mic-observations), and it *beats* a plain $1\/r$ proximity control ($0.74$),
showing the wake-column structure carries predictive power beyond mere closeness.
Michael's out-of-wake array (mics above the disk) is predicted near-silent
(max exposure $0.006$ m/s $approx 0$) — the design's generalization claim. A
caveat that matters for the results below: this validation, and v3's training,
are in the *hover* regime (relative airspeed $V_"rel" = 0$); the grey-box
quadrotor-dynamics module that would drive the wake in free flight exists but is
not exercised on the swapped split, which lacks reliable airspeed.

= Results

== Scoring on free flight, not idle

The models must be compared on the regime we care about. The swapped split sets
``val_at_start = true``, holding out the *first* slice of each recording — and
DREGON/Michael's recordings *begin* with several seconds of pre-takeoff idle and
ramp-up. Measured on the raw valid set the median mean-RPS is only $15$ for DREGON
and $35$ for Michael's, and most clips are near-silent idle; a magnitude metric
there is dominated by how well each model reproduces *silence*, which is both easy
and uninformative, and it inverts the rankings. We therefore score on the
*free-flight* subset — clips with mean RPS $>= 45$ ($n = 48$ per variant across
both drones) — and illustrate with free-flight clips only.

#figure(
  caption: [Per-variant MR-STFT quality on the free-flight subset (mean RPS $>= 45$,
  $n = 48$). "train best mrstft" is each run's own best epoch (on its own valid, so
  the OLD value — on wrong geometry — is not directly comparable); "flight mrstft"
  is the free-flight score computed here. Higher is better.],
  table(
    columns: (auto, auto, auto, auto),
    align: (left, center, center, left),
    table.header([*variant*], [*epochs*], [*flight mrstft* #sym.arrow.t], [*note*]),
    [OLD (wrong geom)], [\~32], [4.51], [wrong-geometry baseline],
    [*v1 — corrected*], [32], [*5.22*], [best in flight — geometry fix helps],
    [v2 — +per-rotor], [52], [4.82], [helped on idle, not in flight],
    [v3 — +wind], [\~30], [3.44], [weakest — wind dormant at hover],
  ),
) <tab-results>

#figure(
  image("assets/msstft_bars.png", width: 72%),
  caption: [MR-STFT quality on the free-flight subset. The corrected-geometry
  baseline (v1) is best and clearly beats the old wrong-geometry generator; the
  wind variant (v3) is weakest.],
) <fig-bars>

#figure(
  image("assets/spectrograms.png", width: 96%),
  caption: [Real (top) vs. generated, an 8-second *mid-flight cruise* clip
  (RPS $approx 81$) per drone, dB log-STFT to 4 kHz (the repo spectrogram
  convention; the band where the rotor harmonics live). Harmonic stacks are
  clearly visible — evenly spaced horizontal lines from the rotor fundamental up.
  Real noise mixes those harmonics with a broadband floor; v1/OLD reproduce the
  mix; *v2 (+per-rotor)* generates the *strongest, most regular* harmonic lines,
  especially on Michael's — arguably over-tonal relative to the real broadband
  content; *v3 (+wind)* is the flattest and most washed-out. These clips come
  from the training region (the swapped split holds out only takeoff, so no
  held-out sustained-cruise data exists) and are illustrative — the quantitative
  comparison is the held-out free-flight metric above.],
) <fig-spec>

Three findings stand out:

+ *The geometry correction helps.* On free flight the corrected-geometry baseline
  (v1, mrstft $5.22$) is the best model and clearly beats the old wrong-geometry
  generator ($4.51$). This is the physically expected result, and it only becomes
  visible once idle clips are excluded — on the raw (idle-heavy) valid set the two
  were indistinguishable and the ranking even inverted. The earlier impression that
  "corrected geometry made things worse" was an artefact of scoring on takeoff/idle.

+ *Per-rotor sub-embeddings do not help in flight.* v2 ($4.82$) sits *below* plain
  v1 in flight, despite having looked best on the idle-dominated set. Letting each
  rotor own its timbre apparently buys fit on the low-RPS/idle regime (where subtle
  per-rotor differences at start-up matter) but not in cruise; on this data it is at
  best neutral and slightly regresses the flight fit. We would not carry it forward
  without evidence on more flight data.

+ *The wind channel is the weakest.* v3 ($3.44$) is clearly worst and its
  spectrogram is the flattest. This is expected: with $V_"rel" = 0$ at hover the
  physics gate places little flow, so the wind channel is near-dormant and only adds
  an incoherent floor that dilutes the coherent fit. Its value is contingent on
  free-flight airspeed the swapped split does not provide.

= Discussion <sec-discussion>

*The evaluation regime was the whole story.* The single most important lesson here
is methodological: scoring generative fidelity on the swapped ``val_at_start`` set
means scoring on takeoff/idle, where the signal is near-silence and every model
"wins" trivially. That regime inverted the rankings — it made the corrected
geometry look worse than the wrong geometry, which is physically nonsensical — and
produced a misleading illustration clip. Restricting to free flight (RPS $>= 45$)
removes the confound and recovers the sensible ordering. Any future generator
comparison on these recordings must condition on the flight regime, not average
over idle.

*The geometry fix helps, as it must.* Once idle is excluded, v1 (corrected) is the
best model and beats OLD by $0.7$ mrstft. The 180° mic-frame correction changes the
rotor$->$mic distances and delays that drive propagation; on real flight noise —
rich, broadband, multi-tonal — getting that propagation right measurably improves
the fit. (The correction's largest effect is still on *phase/inter-channel* TDOAs,
which Stage-0 quantified at correlation $-0.55 -> +0.93$; the flight-regime
magnitude gain is a welcome, and consistent, second signal.)

*Per-rotor sub-embeddings are regime-dependent — treat with caution.* The reversal
between idle (v2 best) and flight (v2 below v1) is the kind of result that warns
against over-fitting a metric on the wrong distribution. The extra per-rotor
capacity seems to help model start-up/idle asymmetries between rotors but does not
transfer to cruise, where the four rotors are near-identical. We would re-test it on
a flight-heavy split before adopting it.

*Limitations.* (i) $n = 48$ flight clips is modest; the v1-vs-v2 gap ($0.4$) is
suggestive, not decisive, though v1-vs-OLD ($0.7$) and the v3 deficit are clearer.
(ii) The metric is single-channel magnitude; the geometry fix's inter-channel
benefit and the wind channel's incoherence both live in multi-channel structure it
cannot see, so this understates the geometry gain and gives the wind channel no way
to demonstrate its distinctive property. (iii) v3's wind channel is evaluated only
at hover; the free-flight airspeed that would activate it (via the grey-box
dynamics module) is absent from this data. (iv) v1/v3 used gradient accumulation
(batch $16 times 2 =$ effective batch $32$), numerically equivalent to v2's
batch $32$.

*Takeaway.* The *geometry correction* is validated and should stay: it improves the
generator on the flight regime and underpins the project's propagation/localization
goals. *Per-rotor sub-embeddings* are not a clear win and should not be adopted on
this evidence. The *wind channel* is architecturally sound and physically validated
(its gate predicts real per-mic flow noise at Spearman $0.92$), but it cannot pay
off without free-flight data and a coherence-aware objective — the concrete next
step for that line of work.
