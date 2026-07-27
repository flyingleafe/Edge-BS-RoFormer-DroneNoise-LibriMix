#import "/writing/templates/typst/report.typ": report, author-meta

#show: report.with(
  title: [CKLA: a Kalman-Structured Temporal Head, its Mechanistic Diagnosis, and the Augmentation Regime That Feeds It],
  authors: (
    "Harmonic Noise Suppression Project": author-meta("project"),
  ),
  affiliations: (
    "project": "Harmonic Noise Suppression Project",
  ),
  abstract: [
    The VK-parity campaign (see the 2026-07-24 status report) stalled at a
    3.4x structural gap between the best neural rotor-speed (RPS) predictor
    (protocol MAE 2.481) and the blind Vold-Kalman (VK) reference
    (0.68-0.74 rev/s), with every cheap recipe/front-end/augmentation lever
    exhausted. This report covers two threads that opened since. First, we
    replaced the transformer temporal head with a complex Kalman linear
    attention (CKLA) head - an information-form Kalman filter run as a
    sequence layer, extended with an input-dependent complex rotation meant
    to give it closed-loop frequency tracking, the piece the earlier
    hand-built Kalman tracker (K2) lacked running open-loop. Rather than
    iterating blind on top-line numbers, we then instrumented the trained
    head and asked what it actually computes. Two pathologies fell out:
    accumulator degeneration (the state precision never saturates within an
    8 s clip, so the Kalman gain collapses and the layer degrades into a
    fixed long-horizon averager) and an amplitude anchor shared by *every*
    architecture we have trained (a genuine 2% comb-spacing shift moves
    predictions by about 0.03%, regardless of head). Second, we trace how a
    poorly-designed augmentation experiment produced a false negative for
    the fix to the second pathology, and how re-testing it correctly
    reversed the verdict. Each diagnosis became a one-knob intervention on
    the CKLA head: the pathology fixes take full-envelope validation MSE
    from 85.2 (base CKLA) to 63.0 (+freq-scale augmentation) to 44.8
    (+precision-gain restoration) - the CKLA head now beats the matched
    transformer control (63.7) by 30% under otherwise identical
    conditions. The DREGON-cruise/FLY124-cruise protocol read and two
    further attribution controls are still training as this report is
    written; those cells are marked "updating" with their job handles.
  ],
  keywords: ("Kalman filtering", "linear attention", "RPS prediction", "augmentation", "mechanistic analysis"),
)

// Number headings so in-text "§N" references resolve to real, visible
// section numbers (matches the TOC), and shorten the running header so it
// fits on one line instead of wrapping the full title.
#set heading(numbering: "1.1")
#set page(
  header: context {
    let page-num = here().page()
    if page-num > 1 {
      align(right)[
        #smallcaps[HNS] --- CKLA campaign #h(1fr) #page-num
      ]
    }
  },
)

= Where the VK-parity campaign stood <sec-vk-status>

The project needs an audio-only estimate of each rotor's instantaneous
rotation speed (RPS), because ground-truth telemetry is unavailable at
inference time and only approximately trustworthy even during training (the
DREGON tachometer jitters by roughly 0.6 rev/s). The blind Vold-Kalman (VK)
order tracker gives a slow, non-causal, but accurate reference: pooled error
0.68-0.74 rev/s on DREGON, 1.03 rev/s on FLY124. A fast, causal neural
predictor trained to match it currently manages only 2.481 rev/s on
DREGON-cruise - a 3.4x gap - after the 2026-07-24 report closed out every
cheap lever tried: test-time smoothing, longer context, constant-Q
front-ends, a comb matched-filter front-end, and GP-noise augmentation. The
IF (instantaneous-frequency) channel front-end remains the only front-end
change that helped at all (2.481 vs the 2.62 baseline). Two levers were
still open when that report was written: augmentation strategy, and
architecture. This report covers both, in the order they turned out to be
entangled: the augmentation story first, because a subtle experimental-design
mistake there produced a false negative that later mattered for the CKLA
head's own best result.

= The augmentation regime: dilution, not futility <sec-dilution>

For most of the campaign, training used only mixture-level augmentations -
random gain, random polarity, channel-drop (p=0.5), and a noise time-warp -
applied on top of a two-stage schedule (an unaugmented 50k-sample warmup,
confirmed load-bearing in phase G5: removing it made things worse, best
val/mse 117.9 vs 63.7). Of that family, `random_polarity` is an exact no-op
for magnitude/IF front-ends and `random_gain` is just a log-magnitude
offset; only `channel_drop` and the mild noise time-warp actually changed
anything the predictor could see. Every trained arm overfits the same way -
train keeps falling while validation roughly doubles within about 20 epochs
of the optimum - so phase G6 tried a stronger fix: six *noise-chunk*
transforms (`freq_scale`, `spectral_recolor`, `random_reverb`,
`tooth_dropout`, `spec_mask`, `floor_inject`), sampled one-at-a-time with
probability 0.7 per training chunk, applied to the noise+RPS pair before
mixing.

The G6 bundle failed outright: best protocol MAE 2.982 (IF arm) / 3.370
(baseline arm), both worse than the 2.481 floor, with the transformer arm's
training also destabilized from epoch 0 in the paired G5 cold-start variant.
The post-hoc reading in the 07-24 report closed the entire augmentation
lever class as refuted.

The reading changed once we looked at what the bundle actually delivers to
the model. Of the six transforms, `freq_scale` is structurally different
from the other five: it resamples the noise *and* its RPS label together by
$alpha ~ U(0.75, 1.3)$, so it is the only transform where amplitude and
timbre cues stop predicting the label and comb *spacing* becomes the only
signal left. The other five (recolor, reverb, tooth-dropout, spec-mask,
floor-inject) leave the label untouched - they are regularizers, not
spacing-forcing transforms. Bundling all six behind a single "pick one of
six, then apply at $p=0.7$" gate means `freq_scale` reaches the model on
only about 1 chunk in 9 (@fig-dilution). A transform diagnosed later
(§#ref(<sec-mechanism>, supplement: none)) as the only one that attacks the model's core failure mode was
therefore present at roughly 6x below the dose needed to matter, drowned out
by five transforms that do not touch the pathology at all. "Strong
augmentation is refuted" was the wrong generalization from "a diluted bundle
containing one useful ingredient at ~11% strength is refuted."

#figure(
  image("assets/aug_dilution.png", width: 92%),
  caption: [
    The G6 bundle samples uniformly from six transforms at $p=0.7$ per
    chunk, so `freq_scale` (red) reaches roughly 1 chunk in 9 (left). The
    CKLA-campaign policy used later (§#ref(<sec-mechanism>, supplement: none), `g2_if_freqscale` /
    `ckla_p1_freqscale`) runs `freq_scale` solo at the same $p=0.7$, an
    approximately 6x dose increase (right).
  ],
) <fig-dilution>

= The freq-scale augmentation and the collapse to the mean <sec-freqscale>

The dilution arithmetic alone does not explain *why* `freq_scale` should
matter more than the other five transforms - that came from the mechanistic
analysis of §#ref(<sec-mechanism>, supplement: none) (a scale-response probe run on the trained CKLA and
transformer models, detailed in §A6 of the activation-analysis note). The
result: perturbing an input clip by resampling every frequency by exactly
$times 1.02$ - a real, physically meaningful 2% shift of every rotor's comb -
and the corresponding label shifted by the same factor - moves the trained
model's prediction by about 0.03-0.06%, not the 2% a spacing-reading model
would produce (@fig-freqscale). This holds for *both* architectures tested
(CKLA and the IF transformer), and is not a small effect: the ideal response
is roughly 60x larger than what either model does. The models are not
reading comb spacing at the margin; their outputs are anchored to the RPS
distribution seen in training, and small genuine changes in the true
physical quantity get absorbed rather than tracked.

#figure(
  image("assets/freqscale_illustration.png", width: 96%),
  caption: [
    Left/middle: `freq_scale` at $alpha = 1.02$ shifts every harmonic line
    of a synthetic comb by 2% (illustrative spectra, not measured data).
    Right: the measured response of the two trained models to exactly this
    perturbation, against the ideal (label-consistent) response - both
    models under-respond by roughly 60x.
  ],
) <fig-freqscale>

This is the mechanism `freq_scale` (solo, undiluted) is designed to attack:
by construction it is the only transform in the family that manufactures
genuinely new (audio, RPS) pairs where amplitude/timbre alone cannot predict
the label, forcing the model to use spacing information it otherwise
ignores. The CKLA campaign re-ran it solo at the same per-chunk probability
(0.7) used in the G6 bundle - an effective dose increase of roughly the
1-in-9 to 1-in-1 ratio in @fig-dilution - as one lever in the CKLA ladder
(§#ref(<sec-results>, supplement: none)) and, separately, as a matched-transformer control
(`g2_if_freqscale`, still training as this report is written).

= CKLA: a complex Kalman linear attention head <sec-ckla-arch>

== Preliminaries: filtering as a sequence-mixing primitive

An Ornstein-Uhlenbeck (OU) process is the continuous-time analogue of an
AR(1): a latent state relaxes toward a mean with time constant $1/gamma$
while absorbing Gaussian process noise. Discretized over a fixed step, its
transition is $z_t = macron(a) z_(t-1) + epsilon_t$ with $macron(a) =
e^(-gamma Delta t)$ and $epsilon_t tilde cal(N)(0, macron(p))$. An
information-form Kalman filter tracks the same process not via the state
mean/covariance $(z, Sigma)$ directly but via its *precision-weighted*
dual, the information pair $(eta, lambda) = (Sigma^(-1) z, Sigma^(-1))$: a
representation where fusing independent evidence is addition, and where a
"no information yet" prior is simply $eta = lambda = 0$. Kalman Linear
Attention (KLA, arXiv 2602.10743) observes that a bank of independent
information-form OU filters, one per (state slot, channel), can be run in
parallel across a whole sequence as a linear-attention-shaped layer: each
token contributes evidence $(phi_t, kappa_t)$ (precision and
information), the filter recursion is a first-order linear recurrence
amenable to associative parallel scan, and the precision-weighted readout
$mu = eta \/ lambda$ gives every layer a built-in notion of "how confident
is this state slot right now" that a plain linear-attention layer lacks.

== KLA background (the substrate we extend)

Per token, KLA computes four broadcastable quantities over a state grid of
$N$ slots $times$ $D$ channels: the transition $macron(a)_t in RR$, process
noise $macron(p)_t in RR_(>=0)$, evidence precision $phi_t = k_t^2 dot
lambda v_t$, and evidence information $kappa_t = k_t dot lambda v_t dot
v_t$ (where $k_t$, $v_t$ are the usual attention-style key/value
projections and $lambda v_t$ is a learned per-token evidence-confidence
scalar). The flat information recursion (flat prior, $eta = lambda = 0$
at $t=0$) is

$ "den"_t = macron(a)_t^2 + macron(p)_t lambda_(t-1), quad
  lambda_t = lambda_(t-1) \/ "den"_t + phi_t, quad
  eta_t = macron(a)_t eta_(t-1) \/ "den"_t + kappa_t $

with readout $mu_t = eta_t \/ max(lambda_t, epsilon)$ and $y_t = sum_n
q_t [n] dot mu_t [n, :]$ (a query vector selecting which state slots to
read out, exactly linear-attention shaped). This recursion is associative
in $t$ (a Möbius-transform composition), so it parallelizes as a scan; we
use the *flat* sequential form only (no Fenwick-tree acceleration), since
our clips are 126-250 frames - a plain fp32 sequential loop over that
length is cheap and exact, and avoids a second implementation to validate.

== The complex extension

The extension this campaign bets on: make the transition complex,
$macron(a)_t = e^(-gamma + i omega_t)$, so the latent can *rotate* as well
as decay. The key structural fact that makes this free: the precision
recursion only ever sees $|macron(a)_t|^2$ (rotation is unitary and drops
out of second moments for a complex-Gaussian latent), so the real precision
algebra above is *untouched* - $lambda_t$'s recursion is identical to plain
KLA. Rotation acts only on the information vector $eta_t$, as a
unit-modulus multiplier:

$ "den"_t = |macron(a)_t|^2 + macron(p)_t lambda_(t-1), quad
  lambda_t = lambda_(t-1) \/ "den"_t + phi_t, quad
  eta_t = macron(a)_t eta_(t-1) \/ "den"_t + kappa_t $

(@fig-block-diagram). The rotation rate is made input-dependent, $omega_t =
omega_0 + s dot W_omega h_t$, with $W_omega$ zero-initialized so training
starts as a plain linear-time-invariant complex-OU layer (a rotating LRU
with uncertainty) and input dependence only grows where gradients ask for
it; $omega_0$ is ring-initialized linearly over $[0, pi]$ across slots, and
a learnable per-slot gate $s$ (init 0.1) keeps early rotation excursions
small. This input-dependent rotation is the piece the project's earlier
hand-built Kalman harmonic tracker ("K2", killed for drift collapse) never
had - K2 ran the same complex-OU filter open-loop with oracle rotation; CKLA
lets the rotation itself be driven by learned features, closing the loop.

#figure(
  image("assets/ckla_block_diagram.png", width: 96%),
  caption: [
    `ComplexKLALayer` block: a causal conv + SiLU + QK-norm front-end feeds
    per-token evidence ($phi_t$, $kappa_t$, top) and the transition
    ($macron(a)_t$, bottom, the only complex-valued path) into the flat
    information recursion, then a precision-weighted readout. The real
    precision algebra is identical to plain KLA; only $eta_t$'s update
    carries rotation.
  ],
) <fig-block-diagram>

== Layer scaffolding and model wiring

The layer otherwise mirrors the project's existing `FenwickKLALayer`
scaffolding: causal `conv1d(k=4)` + SiLU, QK L2-normalization, `softplus`
evidence gain, a gated residual, and RMSNorm, minus everything Fenwick
(no bucket/fold machinery - unnecessary at our sequence lengths). $eta$ is
stored as an explicit `(re, im)` pair of real fp32 tensors rather than a
native complex dtype, to avoid complex-autograd slow paths; the scan runs
in fp32 unconditionally regardless of surrounding autocast, the same
discipline used for the real-valued KLA reference. `TemporalCKLAHead`
replaces the transformer temporal head in the project's `SimpleConvV2`
trunk (front-end $arrow$ 6x residual conv encoder $arrow$ frequency-attention
pool $arrow$ temporal head), with 2 `CKLABlock`s of `d_model=128`,
`n_state=16` over time - a parameter budget of 1.84M against the 1.48M
transformer head it replaces. The front-end is `stft_mag_if` throughout
(the one front-end change that beat baseline in the earlier phase).

= What the trained head actually computes <sec-mechanism>

Beating a floor by architecture search alone risks learning the wrong
lesson from a win or a loss. Once `ckla_p1_if` (the P1 real-protocol model,
§#ref(<sec-results>, supplement: none)) was trained, we instrumented it directly - taps on the per-slot
precision $lambda_t$, evidence gain $phi_t \/ lambda_t$, rotation
excursions, and readout mix - on 12 seeded real validation clips (8
DREGON-cruise, 4 FLY124-cruise), alongside the same taps on the P0 synthetic
model and the transformer comparator. Six questions, six answers.

== Accumulator degeneration (the headline pathology) <sec-accumulator>

*Does the Kalman gain adapt within a clip, or does it collapse?* It
collapses. State precision $lambda_t$ climbs monotonically through the
entire 8 s clip to $10^3$-$10^5$ (never saturating - median saturation time
7.2-7.3 s, i.e. essentially the whole clip), so the effective per-step
Kalman gain $phi \/ lambda$ decays to $10^(-7)$-$10^(-4)$ by mid-clip
(@fig-accumulator). Combined with a near-static readout mix (entropy 1.9 of
a possible 4 bits, concentrated on the two or three longest-horizon state
slots, and fluctuating by only $plus.minus 0.02$ over a clip) the layer is
functionally a fixed bank of clip-scale accumulators: new frames nudge the
state by parts-per-million once the clip is a few seconds in. This is not
what "uncertainty-gated filtering" is supposed to buy - it is a leaky
long-horizon average with a static weighting, degrading gracefully rather
than adapting.

#figure(
  image("assets/accumulator_degeneration.png", width: 96%),
  caption: [
    Schematic of the measured pattern (shape matches the reported
    saturation times and gain magnitudes; not raw per-frame data). State
    precision $lambda_t$ never saturates within the 8 s clip (left); the
    effective Kalman gain $phi \/ lambda$ collapses to $10^(-7)$-$10^(-4)$
    within 1-2 s (right).
  ],
) <fig-accumulator>

This single pathology explains two otherwise-puzzling results at once: it
is consistent with the FLY124 win (a fixed long integrator suppresses the
transformer's tendency to overfit per-frame DREGON texture - see the ridge-
probe result below) and with the DREGON loss (no drift bandwidth survives
once the gain has collapsed, so anything requiring within-clip adaptation is
unavailable).

== Precision gating is a regime switch, not an SNR gate

Within cruise segments the evidence precision $lambda v_t$ is close to
constant (coefficient of variation 0.03-0.17, correlation with log-energy or
speech-band fraction $|r| <= 0.22$) - no frame-level "trust this frame more"
behavior. The one place it moves sharply is at flight-regime transitions:
in the one clip containing a takeoff ramp, layer-1 $lambda v$ steps
$0.3 arrow.r 2.0$ exactly at spin-up, with layer-2 doing the inverse. The
gate the layer learned is a coarse idle/cruise switch, not the
frame-by-frame uncertainty weighting the Kalman framing was meant to supply.

== Rotation: null in synthetic, causal on real DREGON <sec-rotation>

The rotation ablation (zeroing $s$, $omega_0$, $W_omega$) was run three
times at increasing realism, and the verdict changed each time. On the P0
synthetic static-comb task at 1 s context, rotation is flatly null - both an
eval-time ablation on the trained model and a train-time control with
rotation disabled from scratch land within 0.2 of each other (best val/mse
21.51 rotation-off vs 21.70 rotation-on). Extending to 4 s native context
changed nothing (ablation delta $plus.minus 0.05$ across all drift/SNR
cells; rotation parameters stayed near their zero-init values). On real
DREGON data, the picture flips: zeroing rotation on `ckla_p1_if` costs
+0.31 PIT-MAE (+9%) on the 8-clip DREGON subset, consistently across 7 of 8
clips even excluding the takeoff-ramp outlier, and layer-2 rotation
excursions correlate with ground-truth RPS *level* at $r=0.82$
(@fig-rotation) - a genuine, if small, closed-loop frequency code. Zeroing
only the imaginary half of the readout costs about half that (+0.17),
splitting the effect roughly evenly between rotation's direct contribution
and its effect on the real channel. On FLY124 both ablations are exact
nulls ($plus.minus 0.02$) - the cross-drone win is not coming from the
complex path.

#figure(
  image("assets/rotation_attribution.png", width: 78%),
  caption: [
    Causal 3-arm rotation attribution on `ckla_p1_if`, 12-clip subset.
    Zeroing rotation costs DREGON +0.31 PIT-MAE (+9%) and is a null on
    FLY124 ($plus.minus 0.02$) - the complex path is real but small, and
    drone-specific.
  ],
) <fig-rotation>

== Where RPS becomes linearly decodable

A ridge probe fit on trunk / block-1 / block-2 activations shows RPS is not
linearly readable from the shared convolutional trunk in either
architecture (MAE 3.8-4.4 rev/s). The transformer makes it almost fully
explicit after one layer (MAE 1.45, no further gain from layer 2); CKLA
refines gradually across both blocks (3.82 $arrow.r$ 2.40 $arrow.r$ 1.95)
and never reaches the transformer's linear readability, even though the two
full models score comparably (2.71 vs 2.78 on this subset). CKLA's final
readout evidently extracts a non-linear code a plain ridge on block-2
features cannot - plausibly a ratio code across the slowly-frozen
accumulator states from §#ref(<sec-accumulator>, supplement: none) - where the transformer's representation is an
explicit per-frame feature.

== The shared amplitude anchor

The scale-response result already previewed in §#ref(<sec-freqscale>, supplement: none) belongs here as a
mechanistic finding, not just an augmentation-design one: both
architectures respond about 0.03-0.06% to a genuine 2% comb-spacing shift,
against an ideal 2% response - neither is reading spacing at the margin.
Against a "CKLA is a scale-faithful comb reader, the transformer is a
timbre reader" hypothesis, the data run the other way: CKLA is *more*
sensitive to spectral recoloring (0.80/0.57 rev/s under $plus.minus$6 dB
tilt vs the transformer's 0.46/0.27) and to a $-6$ dB gain change (3.64
rev/s vs 0.99) - the log-magnitude front-end is not scale-invariant and the
CKLA head amplifies rather than suppresses that. Both models are
amplitude/timbre-pattern readers anchored to the training RPS distribution;
whatever explains CKLA's FLY124 advantage, it is not scale-faithful comb
reading.

= Results ladder <sec-results>

== P0: synthetic drift-tracking gate

Matched-budget arms (CKLA / transformer / GRU temporal heads on an
identical trunk) trained on synthetic 4-rotor harmonic noise with drift and
speech-proxy interference. CKLA reaches a train PIT-MSE of about 3.4 by
epoch 3; the matched transformer needs roughly 23 epochs to reach the same
level (a uni-directional GRU head never got below about 9) - a 7x faster
epoch-to-epoch convergence to an equal-or-lower floor. On a fair rescore
under an identical clean validation split and eval harness, CKLA's best
checkpoint scores MSE 21.7 (RMSE 4.48, R² +0.50) against the transformer's
85.4 (RMSE 9.01, R² $-1.37$) - roughly 4x lower MSE at identical training
data and protocol. On a controlled capture-boundary sweep (drift
aggressiveness $times$ SNR, 16 clips/cell), CKLA sustains a lock (error $<$
2 rev/s) on 38-69% of clips at aggressiveness $<=$ 1 with graceful
degradation to aggressiveness 4; the transformer *never* sustains a lock in
any cell tested (@fig-capture). This is the capability K2 lacked -
locking through drift rather than collapsing - now demonstrated in a
learned, closed-loop layer at matched budget.

#figure(
  image("assets/capture_boundary.png", width: 78%),
  caption: [
    Capture-boundary sweep (P0b): fraction of 16 synthetic clips per cell
    where the model sustains a lock (error $<$ 2 rev/s), against drift
    aggressiveness. CKLA locks and degrades gracefully; the matched
    transformer never sustains a lock in any tested cell.
  ],
) <fig-capture>

== P1: the real protocol, before the diagnosis-driven levers

Trained on the project's E12 two-stage real-data schedule (same
online-mixed v4-michaels stream as every other G-series arm), `ckla_p1_if`
splits the ledger: DREGON-cruise 2.87 against the 2.481 floor (worse by
0.39), FLY124-cruise 1.36-1.39 against the transformer's 2.33 (a 40%
improvement, and the best cross-drone neural score of the campaign, within
0.35 of the blind-VK bar on that pool). A pure head swap at an otherwise
identical front-end and schedule generalizes across drones far better than
attention, at the cost of losing ground on the drone the floor was set on.
This is the split the §#ref(<sec-mechanism>, supplement: none) diagnosis explains: a fixed long-horizon
accumulator averages away DREGON-specific texture the transformer overfits
to (helping FLY124) while giving up the within-clip adaptivity DREGON
apparently rewards (hurting DREGON).

== The levers: each one fixes its diagnosed pathology

Two one-knob interventions followed directly from §#ref(<sec-mechanism>, supplement: none), each targeted at a
named measured failure and each tested on the full-envelope validation
split (not the cruise-only protocol, so these numbers are not directly
comparable to the 2.481/2.87 pair above, but are comparable to each other
and to the matched transformer control):

- *freq-scale augmentation* (§#ref(<sec-freqscale>, supplement: none)) - targets the amplitude anchor by
  forcing spacing-reading, run solo rather than diluted in a six-way
  bundle. Full-envelope val MSE: base CKLA 85.2 $arrow.r$ 63.0.
- *precision-gain restoration* (`p_init`, targets the accumulator
  degeneration of §#ref(<sec-accumulator>, supplement: none) directly by re-initializing the evidence-gain
  parameter closer to 1.0 rather than the small-signal default the layer
  otherwise converges toward) - full-envelope val MSE: 85.2 $arrow.r$
  44.8.

Both levers land, and they compose: the combined arm (`ckla_p1_pnfs`,
freq-scale + precision-gain together) is training now (§#ref(<sec-open-cells>, supplement: none)). Against the
matched IF-transformer control at 63.7 full-envelope val MSE, the single
best CKLA lever (precision-gain, 44.8) is a 30% reduction under otherwise
fully matched conditions - same stream, same front-end, same schedule, only
the temporal head and one initialization constant differ
(@fig-results-ladder). This is the first clean architecture win of the
campaign at matched conditions, on the metric the whole VK-parity effort has
been chasing since the front-end ledger closed.

#figure(
  image("assets/results_ladder.png", width: 98%),
  caption: [
    Left: full-envelope validation MSE across the temporal-head family and
    the two CKLA levers - each lever addresses a named pathology from §#ref(<sec-mechanism>, supplement: none)
    and both reduce loss; the best single lever (precision-gain, 44.8) beats
    the matched transformer control (63.7, dashed line) by 30%. Right: the
    cruise-only vk_eval protocol read for `ckla_p1_if` (pre-lever) against
    the established floor - worse on DREGON, best-in-campaign on FLY124.
  ],
) <fig-results-ladder>

= Open cells and verdict criteria <sec-open-cells>

Four cells were still training as this report was written; each is marked
below with its job handle and the decision it resolves.

- *Cruise-pool `vk_eval` of both levers* (Slurm `20928550`) - reads the
  freq-scale and precision-gain arms on the DREGON-cruise/FLY124-cruise
  protocol used for the 2.481/2.33 floor, the number this whole report has
  been building toward on the full-envelope proxy. *Updating.*
- *Combined lever arm* `ckla_p1_pnfs` (Slurm `20928549`) - freq-scale +
  precision-gain together; decides whether the two interventions compose
  additively or interact. *Updating.*
- *Live-gain rotation control* `ckla_p1_pnoise_norot` (Slurm `20928833`) -
  the goal-deciding attribution cell for the complex-rotation hypothesis:
  does rotation remain causal (as on the un-levered `ckla_p1_if`, §#ref(<sec-rotation>, supplement: none)) once
  the accumulator-degeneration fix is in place, or does the precision-gain
  lever make rotation redundant too? *Updating.*
- *Matched-transformer freq-scale control* `g2_if_freqscale` (omnirun
  `python-b597db`) - does the freq-scale augmentation help the transformer
  head by a comparable margin, or is the benefit CKLA-specific? Needed to
  know whether §#ref(<sec-freqscale>, supplement: none)'s fix is an architecture lever or a general augmentation
  fix that should be folded back into the main recipe. *Updating.*

Immediate next step once these land: Edge-BS-RoFormer, adapted to RPS
prediction, joins the comparison. Its rotary time/frequency embeddings claim
harmonic-line tracking as a structural property - a claim this task can
test directly, and a natural fourth architecture point alongside the GRU,
transformer, and CKLA heads already on the ledger.

#pagebreak(weak: true)
= Appendix: pointers

Design of record: `docs/ckla-design.md`. Prior-art/exploration:
`docs/complex-ou-layer-exploration.md`. Batch doc (ladder, job handles,
raw result excerpts): `docs/experiments/ckla.md`. Activation-analysis note
(the §#ref(<sec-mechanism>, supplement: none) diagnostics in full, A1-A6): `docs/experiments/ckla-activation-analysis.md`.
Augmentation history: `docs/experiments/g1-vk-parity.md` §G5-G7. Layer/model
code: `src/models/ckla.py`. wandb runs referenced: P0 `jcrr4tqe`, P1
`s4u1tb7w`, mechanistic-lever arms `smwulrhf`/`hilihk2v`, P0 rotation
control `08k0ct9x`, P0 4s arm `my6d6emg`.
