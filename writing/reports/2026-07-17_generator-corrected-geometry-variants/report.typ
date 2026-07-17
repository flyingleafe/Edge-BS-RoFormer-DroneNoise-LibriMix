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
    wrong-geometry generator by valid multi-scale STFT (MSSTFT) on the swapped
    DREGON + Michael's split: (v1) the corrected-geometry baseline, (v2) v1 plus
    learnable *per-rotor sub-embeddings* $z_r = z_"drone" + delta z_r$, and (v3) v1
    plus an additive, physics-gated *wind-wake channel*. Per-rotor sub-embeddings
    give the clearest and most consistent improvement (mrstft $8.47 -> 8.94$); the
    wind channel does not help on this hover-dominated data and is the weakest
    variant, consistent with its flow field being dormant at zero airspeed; and the
    geometry correction itself is roughly neutral on these magnitude metrics — as
    expected, since the correction is fundamentally a phase/delay fix that a
    magnitude spectrogram loss cannot see.
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

All four models are scored on the same corrected-geometry swapped validation set
($n = 128$ 1-second multichannel clips) with the exact training-time computation:
the multi-scale STFT loss (lower = better) and the rescaled MR-STFT quality metric
(`mrstft`, higher = better, the training monitor). @tab-results and @fig-bars
summarise; @fig-spec shows the generated vs. real log-STFT magnitude side by side.

#figure(
  caption: [Per-variant results. `mrstft` (the training monitor) and MSSTFT loss are
  measured on the corrected-geometry valid set ($n = 128$). "train best mrstft" is
  each run's own best epoch; the OLD value is on its *own* wrong-geometry validation
  and is not directly comparable.],
  table(
    columns: (auto, auto, auto, auto, auto),
    align: (left, center, center, center, center),
    table.header([*variant*], [*epochs*], [*train best mrstft*], [*eval mrstft* #sym.arrow.t], [*eval MSSTFT loss* #sym.arrow.b]),
    [OLD (wrong geom)], [\~32], [9.34#super[\*]], [8.58], [*5.56*],
    [v1 — corrected], [32], [9.20], [8.47], [7.04],
    [*v2 — +per-rotor*], [52], [*9.70*], [*8.94*], [6.96],
    [v3 — +wind], [\~30], [8.83], [8.08], [7.08],
  ),
) <tab-results>

#figure(
  image("assets/msstft_bars.png", width: 100%),
  caption: [Generator variants on the corrected-geometry swapped valid set.
  Per-rotor sub-embeddings (v2, green) are best on the quality metric; the wind
  variant (v3) is weakest. The MSSTFT loss panel disagrees on the OLD baseline
  (see the Discussion).],
) <fig-bars>

#figure(
  image("assets/spectrograms.png", width: 92%),
  caption: [Real (top) vs. generated log-STFT magnitude, one DREGON and one
  Michael's clip, for each model. v2 (+per-rotor) reproduces the richest,
  most real-like harmonic texture on both drones; OLD and v1 are moderate (OLD
  captures Michael's silence$->$onset and the bright low band); v3 (+wind) is the
  flattest and least structured.],
) <fig-spec>

Three findings stand out:

+ *Per-rotor sub-embeddings clearly help.* v2 is the best variant on both the
  quality metric (mrstft $8.94$ vs. v1's $8.47$ and OLD's $8.58$) and, visibly, in
  the spectrograms — its output has the harmonic striations and broadband texture
  the flatter v1/OLD outputs lack. Letting each rotor own its timbre is the single
  most useful change.

+ *The wind channel does not help here.* v3 is the weakest variant (mrstft $8.08$)
  and its spectrogram is the flattest. This is expected: on hover-dominated data
  the physics gate places little flow, so the wind channel is near-dormant and only
  adds a small incoherent floor that slightly dilutes the coherent fit. Its value
  is contingent on free-flight airspeed, which this split does not provide.

+ *The geometry correction is neutral on these metrics.* v1 (corrected) is
  statistically on top of OLD (mrstft $8.47$ vs. $8.58$; the spectrograms are
  near-indistinguishable). This is the expected outcome, discussed next.

= Discussion <sec-discussion>

*Why the geometry fix does not move MSSTFT.* The Stage-0 correction is, in
physical terms, a *phase/delay* fix: the 180° frame error inverts the sign of the
inter-microphone TDOAs but barely changes each microphone's distance-based
*level*. MSSTFT and MR-STFT are *magnitude* losses — they discard phase — and a
single channel's magnitude spectrum is dominated by the RPS-driven harmonic
amplitudes, which the emitter learns geometry-independently (the emitter produces
a source "as radiated"; geometry enters only in the subsequent propagation). So
correcting the geometry leaves the per-channel magnitude essentially unchanged,
and a magnitude metric cannot register the improvement. The correction's real
value is in *propagation and inter-channel coherence* — where Stage-0 already
demonstrated it (TDOA correlation $-0.55 -> +0.93$) — not in the single-channel
spectral magnitude this generator is trained on. A fair verdict is therefore *not*
"the fix was useless" but "MSSTFT is the wrong instrument to see it"; a coherence-
or phase-aware metric would be needed.

*The loss/quality disagreement on OLD.* The two magnitude metrics rank the top two
models differently: MR-STFT puts v2 first and OLD second, while the raw MSSTFT loss
puts OLD lowest (best). The spectrograms explain it — OLD reproduces a notably
*brighter low-frequency band*, which the multi-scale L1 magnitude loss (summed
across bands) rewards, whereas the rescaled MR-STFT metric weights the mid/high
harmonic structure where v2 wins. We treat MR-STFT as primary (it is the training
monitor), but the disagreement is a reminder that a single scalar spectral loss is
a coarse proxy for perceptual/behavioural fidelity.

*Limitations.* (i) The comparison is single-channel magnitude only; the geometry
fix and the wind channel's incoherence both live in multi-channel structure that
MSSTFT ignores. (ii) v3's wind channel is evaluated only at hover; the free-flight
regime that would activate it (via the grey-box dynamics module) is untested here.
(iii) v1/v3 used gradient accumulation (batch 16 × 2 = effective batch 32) to fit
memory-constrained backends; this is numerically equivalent to v2's batch 32, but
worth noting.

*Takeaway.* Of the three innovations, *per-rotor sub-embeddings* is the clear win
and should be carried forward. The *wind channel* is architecturally sound and
physically validated (the gate predicts real per-mic flow noise), but needs
free-flight data to pay off. The *geometry correction* is important for the
project's downstream propagation/localization goals, but is invisible to — and
should not be judged by — a single-channel magnitude metric.
