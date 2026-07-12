#import "/writing/templates/typst/report.typ": report, author-meta

#show: report.with(
  title: [Predicting Drone Rotor Speed from Sound:\ Teaching a Model the Whole Flight],
  authors: (
    "Dmitrii Mukhutdinov": author-meta("project"),
  ),
  affiliations: (
    "project": "Harmonic Noise Suppression Project",
  ),
  abstract: [
    A drone's rotors sing: each spinning propeller stamps a comb of harmonic tones
    onto the recorded sound, spaced by the rotor's turning rate. If we can read that
    spacing, we can recover the rotor speed (RPS, revolutions per second) from audio
    alone --- useful for cleaning drone-corrupted speech. We can make *unlimited*
    training audio with a synthetic drone-noise generator, but a model trained on
    synthetic noise must still work on *real* recordings. This report tells the story
    of getting that "sim-to-real" transfer to work across an entire flight --- from
    the rotors sitting still on the ground, through take-off, cruise, and landing.
    Three things had to be fixed: (1) a *contaminated test set* that made a working
    model look broken; (2) synthetic data that only ever showed *cruise*, so models
    never learned take-off, landing, or silence; and (3) a generator that could not
    fall *silent* when the rotors stopped. We also rediscover an old lesson the hard
    way --- data augmentation is not "making the task harder", it is what stops a
    model from memorising quirks of the synthetic sound. The punchline is a
    diagnostic: models trained this way do #emph[not] "collapse to guessing the
    average" --- they genuinely track rotor speed across the whole flight. The real
    bottleneck was simpler --- #emph[training coverage] of the low-speed regimes ---
    and the cheapest fix wins: stop *trimming* the take-off ramp out of the real
    recordings. A model trained on the whole real flight (nothing synthetic) roughly
    quarters the error of the cruise-only baseline. Synthetic full-flight noise still
    helps, and remains the only option for regimes real data cannot supply --- but
    where the real ramp exists, keeping it beats simulating it.
  ],
  keywords: ("drone noise", "rotor speed estimation", "sim-to-real", "data augmentation"),
)

= The problem, in plain terms

A quadrotor makes noise the way an organ pipe does: periodically. Each rotor blade
passes a fixed point once per turn, so a rotor spinning at #box[$f$ turns per second]
radiates tones at #box[$f, 2f, 3f, dots$] --- a *harmonic comb*. The spacing between
the teeth of that comb #emph[is] the rotor speed. So "how fast is the rotor turning?"
becomes "how far apart are the comb's teeth?", a question about the sound's spectrum.

Why care? Drone-recorded speech is buried under this rotor noise. If a downstream
speech-enhancement model knows the rotor speed, it knows exactly where the comb sits
and can notch it out. We predict four rotor speeds (one per rotor) at every instant,
and score with #emph[permutation-invariant] mean-squared error (PIT-MSE): the model's
four numbers are matched to the four true numbers in whichever order fits best, then
squared-differenced. Lower is better; an error of #box[$K$] in MSE means being off by
about #box[$sqrt(K)$] rev/s on average.

The catch is data. Real labelled drone recordings are scarce, but we have a *generator*
that turns any rotor-speed trajectory into realistic multi-microphone drone noise. That
promises unlimited training data --- if a model trained on synthetic noise transfers to
real recordings. This is the classic *sim-to-real* problem, and this report is about
making it work.

= A test set that lied to us

Our first models, trained on synthetic noise, looked like total failures: on the real
validation set they scored PIT-MSE in the *hundreds* with hugely negative #box[$R^2$],
seemingly unable to predict rotor speed at all. That conclusion drove three whole
experiments. It was wrong.

The validation set mixed clips from two drones. One recording (the DJI "FLY124") began
with the drone sitting on the ground *warming up* --- rotors spinning at a low, steady
#box[$approx 36$] rev/s before take-off. The pipeline that cut clips out of the
recordings kept any stretch where the rotors were above a #box[$30$]-rev/s threshold, so
this ground warm-up leaked in and was labelled as if it were flight. A model trained on
cruise-speed synthetic noise (#box[$approx 80$] rev/s) confidently predicted #box[$approx 80$]
on these #box[$36$]-rev/s ground clips, and a handful of such clips dominated the average
error.

The flight-controller logs made this unambiguous: a `flyCState` field flips from
`AssistedTakeoff` (ground warm-up) to `GPS_Atti` (real flight) exactly when the rotor
speed jumps from #box[$36$] to #box[$80$] rev/s. Rebuilding the validation set to keep only
genuine free flight (raising the rotor-speed floor to #box[$50$] rev/s) turned the same
"failed" model's score from #box[$approx 204$] to #box[$approx 20$] --- an order of magnitude,
from a single data-cleaning fix.

#emph[Lesson: before believing a model is broken, check that the yardstick is not.]

= Making synthetic data cover the whole flight

Fixing the test set exposed the real limitation: our synthetic noise only ever depicted
*cruise*. The rotor-speed trajectories were generated as steady hover with small
manoeuvres, so a model trained on them had never seen take-off, landing, warm-up, or a
drone sitting silently on the ground. If we ever want to handle a whole flight, the
training data must contain a whole flight.

We built a #emph[full-flight] trajectory generator that walks through every phase
(@fig-fullflight): the rotors start stopped (zero rev/s), spin up to a warm-up idle,
ramp through take-off to cruise, hold cruise with realistic manoeuvres, then ramp down
through landing back to zero. Windows sliced from these trajectories now visit the whole
range of rotor speeds in the proportions a real flight would.

#figure(
  image("assets/fullflight.png", width: 92%),
  caption: [A synthetic full-flight rotor-speed trajectory: ground (0) #sym.arrow warm-up
    idle #sym.arrow take-off ramp #sym.arrow cruise (with manoeuvres) #sym.arrow landing
    #sym.arrow ground. Earlier synthetic data was only the flat cruise middle.],
) <fig-fullflight>

We also made the *loudness* physical. Rotor noise power scales steeply with tip speed
(roughly as speed to the fifth power), so sound *pressure* scales as roughly speed#super[2.5].
A rotor at #box[$36$] rev/s is far quieter than at #box[$80$], and a *stopped* rotor is
silent. Encoding this means the amplitude cue and the frequency cue tell the same,
physically-consistent story across the flight.

= Teaching the generator to be silent

Here we hit a subtle bug. The synthetic model still would not go *silent* when the rotors
stopped. Measuring its output as we swept the rotor speed down to zero, the loudness
flattened out at a small constant floor instead of vanishing (@fig-silence, "before").

The cause is a numerical artefact of how the harmonic comb is synthesised. Each harmonic
is an oscillator whose phase is the running sum of #box[$2 pi f$]; at #box[$f = 0$] the phase
freezes, so #box[$sin("phase")$] becomes a *constant* (a DC offset) rather than zero. Summed
over harmonics, a stopped rotor emitted a steady pedestal of "sound". A model trained on
this learned that silence does #emph[not] mean zero rotor speed --- exactly the wrong
lesson for the ground phase.

The fix is at the emitter: multiply each rotor's synthesised waveform by a smooth gate
that is #box[$1$] above #box[$10$] rev/s and eases to #box[$0$] as the speed approaches zero
(a #emph[smoothstep]). Now a stopped rotor is #emph[exactly] silent, and the fade in
between is smooth and physical (@fig-silence, "after").

#figure(
  image("assets/silence_fade.png", width: 82%),
  caption: [Generator output loudness versus rotor speed. #emph[Before]: a constant floor
    at low speed (the DC pedestal); the generator can never be silent. #emph[After]: the
    emitter-level smoothstep gate fades output to exact zero by #box[$0$] rev/s, full by
    #box[$10$].],
) <fig-silence>

= Augmentation is not "making it harder"

With the full-flight data and a silent-capable generator, an intuition said: keep the
task #emph[simple] --- just mix synthetic noise with speech, no augmentation --- since
earlier troubles looked like the task being too hard. That intuition was backwards.

Trained this way, the models fit the *synthetic* audio well (training error #box[$approx 5.6$]
rev/s) but under-read *real* cruise badly, predicting #box[$approx 51$] rev/s where the truth
was #box[$approx 80$]. The training-versus-validation gap (@fig-trainval) is the fingerprint
of *overfitting to the synthetic domain*: the model latched onto some texture of the
synthetic cruise noise that real recordings do not share, and defaulted low when that
texture was absent.

Augmentation --- random gain, polarity flips, dropping a microphone channel, time-warping
the noise --- is precisely the cure. It is not extra difficulty; it is what prevents the
model from memorising synthetic-specific quirks, forcing it to rely on the one cue that
survives into the real world: the comb's spacing. Restoring augmentation is a core part
of the final recipe.

#figure(
  image("assets/trainval.png", width: 82%),
  caption: [Without augmentation, synthetic-only training overfits the synthetic domain:
    training error keeps dropping while real-validation error stalls and drifts up --- the
    model memorises synthetic texture instead of the transferable comb-spacing cue.],
) <fig-trainval>

= Do the models actually predict, or just guess the average?

Before comparing recipes, a worry worth killing. When a model scores badly on a set
that is mostly one regime (cruise), a classic failure is that it has quietly learned to
output a single safe number --- roughly the average rotor speed --- and given up on
telling regimes apart. If that were happening here, our whole enterprise would be
pointless.

It is not happening (@fig-tracking). Plotting each model's #emph[mean prediction] against
the #emph[true mean] for each regime, the curves climb from ground to warm-up to cruise
rather than sitting flat at the global average (#box[$approx 49$] rev/s). Every model ---
even the weak cruise-only baseline --- reads #box[$approx 79$] on cruise clips whose truth
is #box[$79$]. The models genuinely respond to rotor speed; they differ only in how well
they handle the low-speed end, where they read too high (baseline) or a little too low
(the full-flight-trained ones).

#figure(
  image("assets/tracking.png", width: 78%),
  caption: [Mean predicted versus mean true rotor speed, per regime (Transformer). All
    three training recipes track the true speed (curves rise with the truth, not flat at
    the #box[$approx 49$] global mean) --- no model has collapsed to guessing the average.
    They differ at the low-speed end: the cruise-only baseline over-reads a near-silent
    drone (ground truth #box[$0$] #sym.arrow predicts #box[$48$]); the full-flight recipes
    pull that down toward zero.],
) <fig-tracking>

= Results: three ways to cover the flight

Given the models can predict, the real question is #emph[coverage]: which training data
teaches the low-speed regimes? We compare three recipes, all scored on the #emph[full]
validation set and broken down by regime:

- #strong[Real-only, cruise-trained] (the baseline): real recordings with time-warp, but
  with the usual #box[$>= 30$]-rev/s filter that #emph[trims the take-off ramp away]. This
  is what the pipeline did by default --- and it means the model never sees low speed.
- #strong[Sim full-flight curriculum]: pre-train on full-flight #emph[synthetic] noise
  (the fixed silent generator + analytic comb, with augmentation), then fine-tune on real.
- #strong[Real full-flight]: the #emph[same] real recordings and recipe as the baseline,
  but keep the whole powered envelope (drop the #box[$30$]-rev/s trim to #box[$0$]) so the
  real take-off ramp becomes training data. Nothing synthetic.

#figure(
  image("assets/regime_comparison.png", width: 98%),
  caption: [Per-regime PIT-MSE on the full validation set (log scale, lower better),
    averaged over the three model sizes. Every recipe handles cruise; the two that see
    low-speed data in training (synthetic curriculum, or the kept real ramp) cut warm-up
    and especially ground error by large factors. Keeping the #emph[real] ramp wins.],
) <fig-results>

#include "assets/results_table.typ"

Three things stand out (@fig-results, @tab-results). #strong[First], everyone is good at
cruise (PIT-MSE #box[$approx 15$--$45$]) --- unsurprising, since cruise is almost all of
what any drone recording contains. #strong[Second], the cruise-only baseline is a disaster
off cruise: on ground clips (a near-silent, landed drone) it confidently predicts
cruise-like speeds, for a PIT-MSE up to #box[$2450$] --- it has simply never seen a slow
rotor. #strong[Third], and the point of the whole report: the best model overall is the
#emph[real full-flight] Transformer, at an aggregate PIT-MSE of #box[$approx 80$] --- it
nearly matches the baseline on cruise (#box[$20$] vs #box[$15$]) while cutting warm-up
(#box[$149$] vs #box[$385$]) and ground (#box[$375$] vs #box[$2450$]) dramatically. It
roughly #emph[quarters] the baseline's overall error, using no synthetic data at all ---
only the real ramp the baseline was throwing away.

The synthetic curriculum also beats the cruise-only baseline (e.g. Transformer aggregate
#box[$132$] vs #box[$338$]), confirming that #emph[any] low-speed coverage helps. But where
the real ramp exists, keeping it beats simulating it. The one regime nobody nails is
#emph[true silence]: a genuinely stopped drone (#box[$0$] rev/s) still draws a #box[$10$--$15$]
rev/s guess from even the best model, because near-silent audio carries almost no
comb to read. That residual is exactly the kind of gap a synthetic generator --- which can
manufacture unlimited perfectly-silent examples --- is best placed to close.

= Takeaways

+ #strong[Audit the metric before the model.] A handful of mislabelled ground clips made
  a working model look like a total failure and sent us down three blind alleys.
+ #strong[Check for mean-collapse explicitly.] A bad aggregate on a lopsided test set
  invites the story "the model just predicts the average". Plotting predicted-vs-true per
  regime refuted it in one figure --- the models track; they were only starved of
  low-speed training data.
+ #strong[Coverage beats cleverness --- and the cheapest coverage was already in hand.]
  The low-speed failure was not sim-to-real and not an unlearnable task; it was a
  #box[$30$]-rev/s filter silently deleting the real take-off ramp. Removing one threshold
  did more than a whole synthetic-data pipeline.
+ #strong[Augmentation reduces the sim-to-real gap; it does not raise difficulty.]
  Removing it to "keep things simple" reintroduced exactly the overfitting we were trying
  to avoid.
+ #strong[Synthetic data earns its keep on the regimes reality cannot supply.] It beats
  the cruise-only baseline and, for true silence --- where no real recording carries a
  readable signal --- it is the only source of clean examples. But it is a complement to
  real coverage, not a substitute for keeping the real data you already have.
