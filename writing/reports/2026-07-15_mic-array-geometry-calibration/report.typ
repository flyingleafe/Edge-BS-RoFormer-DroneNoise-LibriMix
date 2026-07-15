#import "/writing/templates/typst/report.typ": report, author-meta

#let fig(path, caption, width: 98%) = figure(image(path, width: width), caption: caption)

#show: report.with(
  title: [Recovering Microphone-Array Geometry from Audio:\ Frame, Indexing, and Position Self-Calibration for Two Drone Datasets],
  authors: (
    "Dmitrii Mukhutdinov": author-meta("project"),
  ),
  affiliations: (
    "project": "Harmonic Noise Suppression Project",
  ),
  abstract: [
    Our physics-based rotor-noise model (free-field $1/r$ + propagation delay)
    needs the true 3-D position of every microphone and rotor; a position
    error corrupts the model's predicted inter-mic phase in proportion to
    frequency, and can silently cancel entire frequency bands. We audited the
    array geometry that ships with two drone-noise datasets and found it
    wrong in both — but recoverable in only one. In *DREGON*, the shipped
    microphone and rotor coordinate frames turn out to disagree by a
    $approx 183 degree$ rotation: predicted and measured inter-mic delays are
    anti-correlated ($-0.55$) under the shipped frame and strongly correlated
    ($+0.93$) after correction. Treating this as an initialisation, a
    coherence-weighted phase bundle adjustment (gauge-fixed by anchoring to
    the corrected nominal) then refines individual mic positions from single-rotor
    recordings, cutting the phase residual $38.3 degree arrow.r 28.8 degree$
    with only centimetre-scale moves; a synthetic control with known ground
    truth confirms the optimiser itself is correct (position error
    $0.36$ cm), so the remaining $approx 25 degree$ is real model mismatch
    (directivity, near-field effects), not optimisation failure. In
    *Michael's* recordings, the array was coded as a *vertical* ring when the
    rig's own top-down photograph plainly shows a *horizontal* one — an easy,
    confident fix by inspection. But the same audio-refinement recipe that
    worked for DREGON *fails* here, and we show why it must: Michael's four
    rotors sit clustered directly below the array with no angular spread
    around it, so the array's orientation and scale are not identifiable
    from inter-mic phase alone. A rigid-ring degeneracy sweep makes this
    precise — every way of lowering the fit residual does so by tilting the
    ring towards edge-on and shrinking its radius, converging on a
    physically impossible $81 degree$-tilted, $5.0$ cm disc, while an
    unconstrained fit simply flies the whole array $2.65$ m away. The
    practical upshot: audio *can* self-calibrate a microphone array, but only
    when the recording geometry gives it something to triangulate against;
    otherwise, trust the photo and file audio refinement as future work.
  ],
  keywords: ("microphone array calibration", "bundle adjustment", "TDOA",
             "GCC-PHAT", "relative transfer function", "gauge freedom",
             "DREGON", "identifiability"),
)

#set heading(numbering: "1.")

= Why positions matter, and why we doubted them

The project's rotor-noise model propagates each rotor's tone to every
microphone as a free-field point source:
$ y_m (t) = frac(r_"ref", r_m) dot s(t - r_m slash c), $
i.e. an amplitude that falls off as $1/r$ and a pure propagation delay
$r_m slash c$. Both terms are read directly off the
3-D microphone and rotor positions. If those positions are wrong, the model's
idea of "how this rotor should sound at this mic" is wrong — not obviously
so, but *systematically*, and the way it fails is diagnostic.

A constant position error looks, to first order, like a constant delay
error $delta$ (seconds). Its effect on phase is *frequency-proportional*:
$ Delta phi.alt (f) = 2 pi f delta. $
A small delay error is invisible at low frequencies and catastrophic at
high ones. @fig-propagation makes this concrete: a $15$-sample error at
$44.1$ kHz ($delta approx 340 mu s$) is a mere $12 degree$ phase error at
$100$ Hz, but $122 degree$ at $1$ kHz — past $180 degree$ the model's
predicted phase has *inverted* relative to the truth, so wherever this
mismatch feeds into anything like a matched filter or a coherent
subtraction, that band doesn't get attenuated, it cancels *itself* out.
This is exactly the failure mode that made us suspect the shipped
geometry in the first place: mid-frequency harmonics of the generative
noise model were systematically weak, the signature of a delay term that
is right at DC and wrong by the time you're several harmonics up.

#fig("assets/fig1_propagation_phase.png", [
  A geometry error is a frequency-proportional corruption. *(a)* A fixed
  $15$-sample delay error produces a phase error that grows linearly with
  frequency, crossing the $180 degree$ (full-inversion) line before $1600$ Hz.
  *(b)* The practical consequence: how much of a unit-amplitude signal
  survives an attempted coherent cancellation with a mistuned delay,
  $abs(1 - e^(i Delta phi.alt (f)))$ — near-zero at low frequency, near-total
  loss of destructive interference by $1.2$–$1.5$ kHz.
]) <fig-propagation>

This report documents auditing and (where possible) repairing the array
geometry for two datasets — DREGON @sec-dregon-frame–@sec-dregon-bundle and
Michael's DJI Matrice-100 rig @sec-michaels — using nothing but the recorded
audio and, in one case, a photograph.

= Reading geometry off the audio: TDOA and the RTF <sec-rtf>

Two audio-only measurements do the work throughout this report, both
standard array-processing tools applied to DREGON's individual-motor
recordings — each spins *one* rotor at a fixed speed while the full array
records, giving a clean, (near-)single-source field.

*Time difference of arrival (TDOA).* For a pair of mics $(a, "ref")$ observing
one source, GCC-PHAT estimates the sub-sample delay
$ hat(tau)_(a,"ref") = op("argmax")_tau, quad "IFFT"[ frac(X_a (f) X_"ref"^*(f), abs(X_a (f) X_"ref"^*(f))) ](tau), $
the phase-transform (PHAT) weighting whitens the cross-spectrum so the
peak is sharp even for a source with an uneven spectrum. This should
match the free-field prediction $(r_a - r_"ref") slash c$ up to the
propagation model's own error.

*Relative transfer function (RTF) and coherence.* For a single source
observed at mic $m$ and a reference mic, the Wiener-optimal estimate of
the transfer-function ratio $H_m slash H_"ref"$ is
$ hat("RTF")_m (f) = frac(S_(x_m x_"ref") (f), S_(x_"ref" x_"ref") (f)), $
the cross-spectrum over the reference auto-spectrum (Welch-averaged). The
free-field model predicts $"RTF"_m^"ff" = (r_"ref" slash r_m) dot
e^(-i 2 pi f (r_m - r_"ref") slash c)$ — magnitude from the $1/r$ law, phase
from the delay. Alongside it we get the magnitude-squared *coherence*
$gamma^2_m (f) in [0,1]$, which is exactly the diagnostic we need: near $1$
where mic $m$ and the reference see the same coherent source, low where
broadband noise, scattering, or a second source dominates and no
single-source linear model — free-field or otherwise — can explain mic $m$'s
signal at that frequency.

#fig("assets/fig2_rtf_coherence.png", [
  Rotor 1, DREGON: measured $abs("RTF")$ (blue) against the free-field
  $1/r$ prediction (orange), per mic, with coherence overlaid (green,
  right axis). All eight mics track the free-field level reasonably in a
  band around $400$–$800$ Hz (shaded); coherence there averages
  $gamma^2 approx 0.71$ versus $0.37$ below (wind and low-frequency room
  noise dominate) and $approx 0.57$–$0.61$ above (multipath/scattering
  decorrelation grows with frequency). This 400–800 Hz window is the "clean
  band" used for the phase-bundle-adjustment objective in
  @sec-dregon-bundle.
]) <fig-rtf>

= Discovery: a 180° frame mismatch in DREGON <sec-dregon-frame>

The first sanity check — do measured TDOAs agree with what the shipped
`micPos`/`rotorsPos` predict — failed outright. Correlating the full
measured TDOA matrix (all four single-motor recordings, GCC-PHAT, against
a fixed reference mic) with the free-field prediction from the *shipped*
geometry gives a correlation of $-0.56$: not weak agreement, but
*systematic disagreement*. That is the fingerprint of a coordinate-frame
mismatch, not measurement noise or a genuinely bad geometry — a
truly-wrong geometry produces scatter, not a strong negative correlation.

Sweeping a rigid $z$-rotation of the mic frame against the (fixed) rotor
frame and re-scoring the correlation resolves it cleanly
(@fig-frame-align a): the correlation rises smoothly from $-0.56$ at
$0 degree$ to a sharp maximum of $+0.93$ at $183 degree$. The shipped
`micPos` and `rotorsPos` arrays are expressed in coordinate frames that
differ by an (almost exactly) $180 degree$ rotation about $z$ — plausibly a
sign or axis-order convention mismatch upstream in how the two arrays were
authored or exported, rather than a physically rotated array. We treat this
cautiously as a *frame convention to reconcile*, not a claim about
DREGON's physical rig: what matters downstream is that with the correction
applied, predicted and measured TDOAs agree well (@fig-frame-align b,
$r=0.93$), which is the standard we hold every geometry in this report to.

#fig("assets/fig3_frame_alignment.png", [
  DREGON's shipped `micPos`/`rotorsPos` are frame-mismatched by
  $approx 183 degree$. *(a)* Correlation between predicted and measured TDOA
  as a function of an applied mic-frame $z$-rotation: sharply peaked at
  $183 degree$ ($r=0.93$), strongly *negative* at the shipped $0 degree$
  ($r=-0.56$). *(b)* Measured vs. free-field-predicted TDOA, shipped
  (red, anti-correlated, scattered around the anti-diagonal) vs. the
  $183 degree$-corrected frame (blue, tight around the diagonal).
]) <fig-frame-align>

All subsequent DREGON work starts from this $183 degree$-corrected nominal
geometry, never the raw shipped arrays.

= Self-calibration by bundle adjustment (DREGON) <sec-dregon-bundle>

Frame correction fixes a *global* error; it says nothing about whether the
individual mic positions inside that frame are exactly right (assembly
tolerances, cable-routing offsets from the CAD nominal, etc.). For that we
fit positions directly to the coherent-band phase.

== Objective

Building on the RTF machinery of @sec-rtf, for each rotor $r$ we restrict
to the $400$–$800$ Hz clean band and use the *phase* residual between the
measured RTF and the free-field prediction, in the free-field sign
convention $phi.alt^"ff" = -2 pi f (r_m - r_"ref") slash c$ (see the
sign-convention caveat in @sec-caveats — `scipy.signal.csd` is the
*opposite* sign, and must be negated before this comparison). Over mic
positions $bold(p) = {p_m}$ and, optionally, rotor positions, minimise the
coherence-weighted, phase-wrapped squared residual
$ cal(L)(bold(p)) = frac(
  sum_(r,m,f) gamma^2_(r,m)(f) dot "wrap"[phi.alt_(r,m)^"meas"(f) - phi.alt_(r,m)^"ff"(f; bold(p))]^2,
  sum_(r,m,f) gamma^2_(r,m)(f)
) + lambda norm(bold(p) - bold(p)_0)^2, $
minimised with Adam. The prior term does two jobs at once. First,
*regularisation*: absolute position is weakly constrained by phase at these
wavelengths (a $2 pi$ ambiguity lurks for any single frequency; only the
*joint* multi-frequency, multi-rotor constraint pins it down at all), so an
unregularised fit can wander. Second, and more fundamentally, it *fixes the
gauge*: relative inter-mic delays determine geometry only up to a global
rigid motion (and reflection) — translating or rotating the whole array
leaves every predicted TDOA unchanged, so without an anchor the optimiser
is free to drift along that whole symmetry group. Anchoring to
$bold(p)_0$ (the $183 degree$-corrected nominal) removes that freedom and
makes "how far did each mic move" a meaningful question. We additionally
strip any *residual* rigid motion left in the anchored solution by Procrustes-aligning
the fit back onto $bold(p)_0$ before reporting per-mic deltas, so the
numbers reflect genuine shape change, not gauge drift the weak prior
didn't fully suppress.

== Does the optimiser work? A synthetic control

Before trusting this on real (noisy, model-imperfect) data, we check it on
synthetic data with known ground truth: place $8$ mics randomly, generate
*exact* free-field phase records (coherence $equiv 1$) for a $4$-rotor
geometry, perturb the mic positions by $approx 1$ cm noise, and refine. The
optimiser recovers the phase objective almost exactly
($8.75 degree arrow.r 0.001 degree$ RMS residual) and the mic positions
back to $0.36$ cm RMSE after removing the residual gauge freedom
(@fig-bundle c) — well below DREGON's own $6$ cm mic spacing. This is the
load-bearing check for everything that follows: *the optimiser is correct*,
so whatever residual remains on real data is a property of the data or the
model, not a bug in the fit.

== DREGON refinement

Running the same objective on the real DREGON single-motor recordings
(speeds $60$/$70$/$80$ Hz, $400$–$800$ Hz band) drops the phase residual
from $38.3 degree$ to $28.8 degree$ overall, and from $34.0 degree$ to
$24.6 degree$ restricted to the high-coherence subset ($gamma^2 >= 0.8$)
(@fig-bundle a). Every mic moves by no more than $2.2$ cm from the
frame-corrected nominal (@fig-bundle b, @tab-dregon-deltas) — plausible
assembly-tolerance-scale corrections, not a sign the nominal geometry was
badly wrong. Broadband TDOA correlation barely moves ($0.89 arrow.r 0.88$):
the delay geometry was already good; what the fit buys is a modest
*phase* improvement inside the clean band.

#fig("assets/fig4_bundle_adjustment.png", [
  Bundle adjustment: real DREGON refinement (a, b) and the synthetic
  ground-truth check (c). *(a)* Coherence-weighted phase residual, before
  vs. after, all bins and the high-coherence subset. *(b)* Per-mic position
  change from the frame-corrected nominal, sorted, all $<=2.2$ cm. *(c)*
  Synthetic-control convergence — residual collapses to near machine
  precision within $approx 500$ Adam steps (the later oscillation is the fixed
  learning rate with no decay circling an already-tiny minimum, not
  divergence); final position error $0.36$ cm.
]) <fig-bundle>

#include "assets/dregon_deltas_table.typ"

Given the optimiser is verified correct on synthetic data, the natural
question is *why does a real, physically-motivated model still leave
$approx 25 degree$–$29 degree$ of residual behind?* Two candidates, both
outside the geometry: the free-field model has no directivity (a real
rotor is not an omnidirectional point source) and no near-field
correction (the mics sit well within a few wavelengths of the rotor at
these frequencies, where the pure $1/r$, plane-wave-phase approximation
starts to break down). Consistent with this, letting the optimiser also
touch magnitude ($abs("RTF")$, via an optional loss term we did not enable
here) does not improve the mean coherence-weighted magnitude error — it is
$2.41$ dB before the phase-only fit and $2.52$ dB after, i.e. essentially
unchanged (phase and magnitude residuals do not covary, as they should not
if the remaining mismatch is physics rather than a positional offset that
a phase fit alone would happen to also fix). We return to this ceiling in
@sec-summary.

= When audio can't help: Michael's, and an honest negative result <sec-michaels>

A second dataset — 8-channel recordings from a DJI Matrice 100 quadrotor,
courtesy of a lab-mate ("Michael's") — exposes a different failure mode
entirely, and a limit worth stating plainly rather than working around.

== The fixable error: wrong plane

`data_processing/michaels.py`'s `get_geometry()` had encoded the 8-mic ring
as *vertical* — lying in the $Y$-$Z$ plane at a constant forward offset $X$.
The rig's own documentation photograph
(`Microphone_Array_Configuration.jpeg`, @fig-michaels-photo c), a top-down
shot with the ring, its numbered mics, and its measurements
hand-annotated, is unambiguous: shot from directly above, the ring reads as
a full circle, not the edge-on ellipse a vertical ring would present from
that angle — and the same photo places all four rotors visibly *below and
around* the ring in the image plane, confirming it's horizontal and
elevated. Correcting the ring to lie in the $X$-$Y$ plane at constant
height ($Z = 0.33$ m, matching the photo's stated $33$ cm mount height) is a
confident fix by inspection, no audio required. @fig-michaels-photo makes
the bug visually obvious: viewed from above, the buggy vertical ring
collapses to a straight line (constant $X$); viewed from the side, the
corrected horizontal ring does the same (constant $Z$) — each geometry is
degenerate exactly in the view orthogonal to its own (wrong or right)
plane.

#fig("assets/fig5_michaels_photo_correction.png", [
  Michael's array: the shipped code drew a *vertical* ring (collapses to a
  line from directly above); the rig photo shows a *horizontal* one.
  *(a)* Top view: the buggy geometry (red) collapses to a vertical line at
  constant $X$; the corrected geometry (blue) is a proper circle. *(b)*
  Side view: the pattern inverts, as it must for two rings in orthogonal
  planes. *(c)* The actual rig photograph the correction is read from —
  numbered mics, a full circle from directly above, rotors visible below.
]) <fig-michaels-photo>

== The hard limit: geometry is not identifiable from this audio

Having fixed the plane, the natural next step is the same audio
self-calibration that worked for DREGON — except it does not work here,
and the reason is structural, not a tuning problem. DREGON's four rotors
sit in a ring of radius $approx 24$ cm around the mic array (itself
$approx 6$–$8$ cm across), one in each quadrant, so each single-motor
recording illuminates the array from a *different* direction — the
angular diversity that makes triangulation possible. Michael's rig is
different: all
four rotors sit *clustered directly below* the mic ring (visible in
@fig-michaels-photo c), and the recordings are noisy in-flight audio (all
four rotors simultaneously, separated only by their harmonic combs via
telemetry-gated frequency-domain gating — not the clean single-source
recordings DREGON provides). With every source in essentially one
direction from the array, inter-mic phase constrains far less: there is a
whole family of array translations, rotations, and even a
sensible-looking *shrink* that leave the predicted phase pattern almost
unchanged, because moving the whole array along the line-of-sight to a
clustered source barely perturbs the relative delays.

We make this precise with a *rigid-ring* reduced model: instead of letting
all 8 mic positions move independently (24 DOF, enough to overfit
anything), parameterise the fit as a rigid ring with only 7 degrees of
freedom — a 3-D translation, a 3-D rotation, and a scalar radius — anchored
to the nominal (photo-derived) pose with prior strength $lambda$, exactly
as in @sec-dregon-bundle's gauge-fixing but now swept across $lambda$ to
see *how* the fit trades residual for physical plausibility.

#fig("assets/fig6_michaels_degeneracy.png", [
  Michael's rigid-ring degeneracy: every relaxation trades physical
  plausibility for residual. *(a)* Anchored $lambda$-sweep — lowering
  $lambda$ (looser anchor) buys lower phase residual only by tilting the
  ring away from horizontal and shrinking its radius; the nominal (no fit)
  residual is $53.4 degree$. Left unconstrained entirely, the ring simply
  translates $2.65$ m away from the drone (resid. $16.5 degree$) — it does
  not need to deform at all to "explain" the phase, because the near-field
  falls out of relevance once the whole array is that far from the source.
  *(b)* The nominal horizontal ring (grey) against the $lambda=10$
  anchored fit (red): tilted $47 degree$ from horizontal, radius shrunk
  from $8.25$ to $5.2$ cm.
]) <fig-michaels-degeneracy>

#include "assets/ring_sweep_table.typ"

The pattern in @tab-ring-sweep is monotone and tells its own story: as
$lambda$ relaxes from $200$ (barely deforms — $1.5 degree$ tilt, radius
$7.8$ cm, residual still $51.4 degree$) down to $0.5$ (residual
$21.9 degree$), the fit buys every degree of residual reduction by tilting
towards edge-on ($1.5 degree arrow.r 81 degree$) and shrinking
($7.8 arrow.r 5.0$ cm) — never by anything resembling "the mics were
slightly misplaced". An $81 degree$-tilted, $5$ cm ring bears no
resemblance to the $8.25$ cm horizontal ring the photograph shows, and there
is no intermediate $lambda$ where the fit both improves substantially *and*
stays physically sane. That is the definition of an *ill-posed* inverse
problem: the objective has a broad, shallow valley along a
tilt-and-shrink direction that has nothing to do with the true correction,
and the data cannot tell the optimiser not to go there.

*Conclusion:* for Michael's array, take the geometry from the photo/spec
(@fig-michaels-photo) and stop there. Audio-based refinement is not a
free upgrade here the way it was for DREGON — attempting it without
recognising the degeneracy would silently swap a known-good geometry for a
confidently-wrong one. Making this observability gap identifiable in
advance (from source/array geometry alone, before running any fit) is
future work.

= What it recovers, and what it cannot <sec-summary>

#fig("assets/fig7_geometry_summary.png", [
  Original vs. refined 3-D geometry, both drones. DREGON (left column): the
  audio bundle adjustment moves mics by at most $2.2$ cm from the
  frame-corrected nominal — a fine-tuning of an already-good geometry.
  Michael's (right column): the only applied correction is the
  vertical$arrow.r$horizontal plane swap, read from the photo; there is no
  audio-refined mic cloud to show, because @sec-michaels shows there
  shouldn't be one.
]) <fig-summary>

Two datasets, two different outcomes, and the contrast is the actual
finding:

- *DREGON* had a global frame bug (fixed by a $183 degree$ rotation,
  identified from broadband TDOA correlation) and small
  ($<=2.2$ cm) residual positional errors within that frame (fixed by
  audio bundle adjustment, verified correct on synthetic ground truth to
  $0.36$ cm). A $approx 25$–$29 degree$ phase residual remains after
  fitting, attributable to free-field model mismatch (no directivity, no
  near-field correction) rather than to remaining position error — the
  magnitude error is unchanged by the phase-only fit ($2.41 arrow.r
  2.52$ dB), which is what you'd expect if the leftover gap is physics, not
  geometry.
- *Michael's* had a plane bug (fixed by inspection from the rig photo,
  no audio needed) and is *not* further refinable from audio: the source
  geometry (all rotors clustered on one side of the array) makes position
  unidentifiable from inter-mic phase, a fact we demonstrated rather than
  assumed via the rigid-ring $lambda$-sweep.

The general lesson travels beyond these two datasets: audio self-calibration
of a microphone array is not something you get for free by having sources
and a phase objective — it requires the *sources* to provide angular
diversity around the array. Checking that (cheaply, before investing in a
bundle-adjustment pipeline) is the practical takeaway; the rigid-ring sweep
here is one way to check it when in doubt.

== A reusable methods caveat <sec-caveats>

One sign-convention trap cost real debugging time and is worth flagging
for reuse: `scipy.signal.csd(x, y)` returns $E[Y dot X^*]$, so for a pure
delay where mic $m$ *lags* the reference by $tau$, its phase is
$+2 pi f tau$ — the *opposite* sign to the free-field convention
$phi.alt^"ff" = -2 pi f (r_m - r_"ref") slash c$ used throughout this report
(and to `gcc_phat_tdoa`'s sign convention). Comparing *magnitudes* only (as
the original Stage-0 RTF validation did) never surfaces this, because
magnitude is sign-blind; it only bites once you compare *phases* directly,
as the bundle-adjustment objective in @sec-dregon-bundle does. The DREGON
path here negates the `estimate_rtf` phase to enter the free-field
convention; the Michael's path sidesteps it by building the cross-spectrum
manually in the correct sign from the start. Anyone reusing
`scipy.signal.csd`-based phase comparisons against a free-field or GCC-PHAT
delay model should check this sign before trusting a phase residual.
