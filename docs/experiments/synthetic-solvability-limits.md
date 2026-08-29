# Limits of solving synthetic data

Campaign opened 2026-08-28. The question: how hard is each synthetic family to
solve, and what sets that difficulty. The campaign's premise on opening was
that the analytic static comb is the simplest corner of the stochastic comb
family, reached by setting the family's variances to zero, so that a curriculum
could walk from one to the other by raising those variances.

**That premise is false.** The two families are disjoint in comb contrast, no
exposed parameter closes the gap, and the knobs named "variance" are not the
difficulty axis. What follows is the measurement.

## The observable

Comb contrast is measured as the peak-to-bulk ratio of a spectrum: the 95th
percentile of bin levels minus the median, in dB, inside an octave band. It
assumes nothing about where the lines are, which matters because every
line-position-based estimator is contaminated by the neighbouring rotors' combs.

Two reference points anchor the scale:

- **Filtered white noise reads 5.6 to 6.3 dB.** Bin power is exponential, so
  p95 minus p50 is 10 log10(ln 20 / ln 2) = 6.4 dB with no structure present
  at all. Anything near 6 dB carries no comb.
- **The analytic static comb reads 15 to 25 dB** across every band.

## The stochastic family sits at the noise floor

Realized audio, 8 s windows, cruise frames, by octave band (Hz):

| family | 100 | 250 | 500 | 1k | 2k | 4k |
|---|---|---|---|---|---|---|
| `static_comb` | 25.2 | 14.8 | 17.2 | 16.2 | 20.4 | 18.9 |
| stochastic, full ranges | 7.2 | 7.3 | 8.2 | 7.9 | 7.6 | 7.5 |
| + zero linewidth | 6.7 | 6.4 | 6.7 | 6.8 | 7.2 | 7.2 |
| + floor buried 40 dB | 7.1 | 7.4 | 8.2 | 8.1 | 8.4 | 8.5 |
| + coherent line rendering | 7.9 | 7.9 | 8.9 | 9.1 | 9.7 | 9.7 |
| + bin-integrated Lorentzians | 7.1 | 6.9 | 7.1 | 7.2 | 7.3 | 7.6 |

Every stochastic variant lands within about 2 dB of structureless noise. The
static comb is 8 to 15 dB above the best of them.

## The flatness is in the target spectrum, not the rendering

The model spectrum that the renderer is asked to realize was read directly and
compared against the audio it produced, same parameters:

| | 250 | 500 | 1k | 2k | 4k |
|---|---|---|---|---|---|
| white noise | 5.6 | 6.0 | 6.2 | 6.2 | 6.3 |
| model PSD | 10.6 | 6.4 | 7.5 | 4.5 | 5.0 |
| realized audio | 10.9 | 9.1 | 9.0 | 7.6 | 9.7 |

Above 1 kHz the noiseless target is flatter than white noise. The comb is not
destroyed by the rendering, because it is not in the model spectrum to start
with. The realized audio reads slightly higher than its own target only because
the realization adds its exponential spread on top.

## Why the floor knob does nothing

Band occupancy — the fraction of bins carrying line support — is 56 to 88% for
a four-rotor 80-harmonic comb at 80 rev/s, even with narrow lines. **The comb's
own lines tile the band, so the median bin is a line, not floor.** Contrast
therefore measures the spread of the harmonic amplitude profile, and the
broadband floor never enters it. Burying the floor by 40 dB moves band contrast
by 0.0 dB at every linewidth:

| gamma_slope \ floor | 0 | −6 | −12 | −24 | −40 dB |
|---|---|---|---|---|---|
| 0.00 | 8.3 | 8.5 | 8.5 | 8.5 | 8.5 |
| 0.20 | 7.1 | 7.2 | 7.2 | 7.2 | 7.2 |
| 0.80 | 4.8 | 4.8 | 4.8 | 4.8 | 4.8 |

## The variance knobs are not the difficulty axis

Zeroing `harm_gp_std_db`, `floor_gp_std_db`, `floor_tilt_gp_std`,
`harm_coherence`, `harm_jitter_db`, `rotor_similarity` and `harm_dropout_p`
together moves per-frame line flicker from 3.41 to 3.22 dB and line visibility
from 3.59 to 2.97 dB. Both are inside the measurement's own spread. A curriculum
that raises these knobs from zero to their full ranges is very nearly a no-op.

## `calibrate_floor` holds contrast constant by construction

`calibrate_floor` places the floor `floor_rel_db` under the **peak PSD** of the
median line, and that peak is `profile_db − 10 log10(pi · gamma)`. Narrowing a
line raises its peak, so the solver raises the floor by the same amount. Line
sharpness and floor depth are coupled, which is why zeroing the linewidth alone
made realized contrast slightly **worse** (7.2 against 7.5 dB at 4 kHz).

## A separate defect: point-sampled Lorentzians

`build_psd` evaluates each Lorentzian at the bin centres. A line is 0.5 to 4 Hz
wide against a 7.81 Hz bin, and `k · rps` puts it at an arbitrary sub-bin
position. For one isolated line at n_fft 2048:

| gamma (Hz) | line on a grid point | line between grid points |
|---|---|---|
| 0.05 | 53.4 dB | 14.0 dB |
| 0.50 | 33.4 dB | 13.9 dB |
| 2.00 | 21.4 dB | 13.0 dB |

Below a bin the width stops mattering: 0.05 and 0.5 Hz read the same. The
fraction of a line's power in its peak bin swings from 1.000 to 0.406 with
sub-bin position, which is 3.9 dB of amplitude wobble that tracks the rotor
speed and belongs to no physical process.

The fix is exact and closed-form — the Lorentzian's antiderivative is an
arctangent — and is available as `line_bin_integrate`, default off so every
existing stream and checkpoint is unchanged. **It does not move band contrast**
(8.4 to 8.1 dB), because density and not sharpness is what caps the band. It is
worth having for the spurious wobble it removes, not as a fix for contrast.

## What does move contrast

Two axes, both monotone and both physically meaningful. Band-mean model-PSD
contrast:

| `rolloff_p` | slope 0.0 | slope 0.4 |
|---|---|---|
| 0.4 | 7.7 | 5.5 |
| 1.0 | 8.4 | 5.9 |
| 1.9 | 8.4 | 6.2 |
| 3.0 | 8.8 | 7.0 |
| 5.0 | 10.2 | 8.7 |
| 8.0 | 12.8 | 11.5 |

`rolloff_p` sets how fast the comb decays with harmonic index, and therefore how
many harmonics are strong enough to tile the band. `gamma_slope_hz` sets how
fast a line broadens with its own order. Together they span 5.5 to 12.8 dB.

## Consequences

1. The static comb cannot be reached from the stochastic family. Its minimum
   (about 15 dB) is above the family's maximum (about 12.8 dB).
2. **WITHDRAWN.** This section previously read that the stochastic family's 1
   to 2 dB of comb over noise was a candidate explanation for stochastic-only
   training plateauing at 3.70 PIT-MAE against comb-only's 1.29 — that the
   information was largely absent from the target. The detectability probe below
   refutes it: the comb is present and lockable.
3. A curriculum is constructible on line width. See the ladder section below.

## The comb-only runs never converged

Training-loss histories for the three comb-only arms, read from W&B. The
percent-of-descent measure is useless here because the first epochs fall from
about 2650 to about 4, so every arm "reaches 99% of its descent" by epoch 3.
The informative number is the slope over the last quarter of each run:

| run | epochs | final train loss | last-quarter slope |
|---|---|---|---|
| `m3abl_comb_scv2_s1` | 37 | 4.37 | −0.079 per epoch |
| `m3abl_comb_unigru128_s1` | 59 | 3.06 | −0.041 per epoch |
| `m3abl_comb_transformer_s1` | 21 | 8.35 | −0.136 per epoch |

All three were still descending when they stopped. The transformer — the arm on
which the campaign recorded "comb-only stage 1 kills the transformer" — had the
steepest remaining descent and the fewest epochs of the three. That verdict
measured the stopping rule.

This is a lower bound on the ladder's rung length, not a calibration: rung
length must be longer than 59 epochs, and the comb-floor runs (`comb_floor_base`
/ `_wide` / `_deep`, patience 200, epochs 2000, validated on the comb itself)
are what will set it.


## Contrast is not a difficulty measure

A model-free check: a harmonic-sum estimator sums the log spectrum at k*f0 over
a grid of candidate speeds and takes the peak. No training and no learned prior,
so it measures what is in the signal. Scored as the distance from its single
estimate to the NEAREST of the four true rotor speeds — detectability, not
assignment — on cruise frames.

The first ladder ordered its rungs by peak-to-bulk contrast, r0 highest. The
probe puts them in a different order entirely:

| rung | r0 | r1 | r2 | r3 | r4 | static comb |
|---|---|---|---|---|---|---|
| contrast (dB) | 12.2 | 11.5 | 10.3 | 9.6 | 7.3 | 15 to 25 |
| detectability (rev/s) | 8.68 | 1.04 | 0.32 | 0.26 | 0.81 | 0.14 |

**The highest-contrast rung is by far the hardest.** That rung reached its
contrast through `rolloff_p` 8.0, and a steep rolloff raises peak-to-bulk by
emptying bins — it kills the harmonics rather than sharpening them, leaving
fewer lines to lock onto. Peak-to-bulk measures how far the loud bins stand over
the quiet ones; it says nothing about how many usable lines there are. That
ladder was withdrawn before it ran, and only rung 0 had been submitted.

### The comb is lockable, so absent information is not the explanation

The same probe reads the nearest rotor speed to 0.26 to 1.04 rev/s on rungs r1
to r4, and to 0.16 rev/s with line width zeroed at 100% hit rate. A matched
filter over 60 harmonics accumulates even 1 to 2 dB per bin into a firm peak.
Whatever holds stochastic-only training at 3.70 PIT-MAE, it is not that the comb
is missing from the spectrum. The gap is more likely in separating and assigning
four interleaved combs, which is the actual task and which this probe does not
attempt.

### The probe cannot order the rungs either

Varying one knob at a time, 12 cruise clips each:

| gamma_slope | 0.0 | 0.1 | 0.2 | 0.4 | 0.8 | 1.6 |
|---|---|---|---|---|---|---|
| median (rev/s) | 0.16 | 21.56 | 0.27 | 21.44 | 0.94 | 3.64 |
| hit under 1 rev/s | 100% | 17% | 83% | 33% | 58% | 17% |

Per-clip error is bimodal — the estimator either locks near 0.2 rev/s or fails
near 20 — so the median is unstable and the hit rates are within noise of each
other at this sample size. The probe is good enough to show that a comb is
there and to catch a rung that is badly misdesigned. It is not good enough to
rank rungs, and this campaign does not use it that way.

## The ladder as it now stands

One axis, line width, with `rolloff_p` and `gamma0_hz` held at the family
default so exactly one thing changes:

| rung | r0 | r1 | r2 | r3 | r4 |
|---|---|---|---|---|---|
| `gamma_slope_hz` | 0.0 | 0.0–0.05 | 0.02–0.15 | 0.05–0.4 | 0.05–0.8 |

Rung 4 reproduces the family defaults exactly, so a model finishing the ladder
has arrived at the distribution the transfer campaign's stochastic arms trained
on directly. Line width is also the single structural difference from the static
comb, whose lines have no width at all. Each rung trains until its own
half-real, half-synthetic validation stops improving. The ladder claims no
difficulty ordering in advance — the runs measure it.

## The models predict a fixed fan: mean tracking, confirmed

`scripts/spread_eval.py` buckets a checkpoint's per-frame error by that frame's
own rotor spread, on cruise columns only, and reports the spread the model
itself asserts. A model that tracks four harmonic lines follows the true spread.
A model that has collapsed onto the comb's mean predicts a fixed fan whatever
the rotors do.

`stoch_s1id_scv2` @ `last`, its own policy, 120 clips:

| true spread (rev/s) | 0.19 | 3.96 | 8.37 | 12.76 | 42.75 |
|---|---|---|---|---|---|
| **predicted spread** | 8.96 | 10.28 | 9.59 | 10.74 | **10.81** |
| PIT MAE | 5.04 | 5.86 | 7.69 | 5.29 | 17.55 |
| frames | 5792 | 1216 | 18112 | 49696 | 23680 |

**The true spread varies over 42.6 rev/s and the predicted spread over 1.85.**
The model splays four rotors by about 9 rev/s when they are turning in unison,
and still by only 10.8 when they are genuinely 42.75 apart. The constant it
settled on is about 9.4 rev/s, which is this generator's own mean spread at
aggressiveness 1.0 — the model learned the marginal distribution of the
quantity, not the signal that determines it.

This is the mean-tracking failure stated as a hypothesis when this campaign
opened, now measured: the model predicts the mean plus a memorized evenly-spaced
fan rather than resolving individual lines. It also explains why capacity
scaling did not move the plateau. The model is not capacity-limited; it has
found a degenerate solution the loss rewards.

### It is not specific to the stochastic family

`m3abl_comb_scv2_s1` @ `last`, on the static comb:

| true spread (rev/s) | 0.14 | 4.33 | 8.92 | 11.69 | 21.46 |
|---|---|---|---|---|---|
| predicted spread | 4.16 | 10.10 | 10.28 | 10.08 | 10.40 |
| PIT MAE | 1.87 | 2.11 | 0.97 | 1.21 | 3.88 |
| frames | 4544 | 9248 | 17344 | 82880 | 1408 |

The comb model emits the same pinned fan. It is more accurate everywhere, and
it does adapt in the unison bucket where the stochastic model does not (4.16
against 8.96 at a true spread near 0.15), but it is tracking the same way.

`m3abl_comb_transformer_s1` @ `last`, same policy, makes it three for three:

| true spread (rev/s) | 0.14 | 4.33 | 8.92 | 11.69 | 21.46 |
|---|---|---|---|---|---|
| predicted spread | 2.76 | 9.87 | 10.01 | 9.89 | 10.09 |
| PIT MAE | 7.45 | 3.25 | 1.81 | 2.38 | 3.95 |

The fixed fan is not an artifact of one architecture — a BiGRU trunk and a
transformer head, trained on two different families, all settle on about
10 rev/s. Note also that the transformer's unison bucket carries the CLOSEST
predicted spread of any model (2.76 against a true 0.14) and its WORST error
(7.45): getting the width right does not help when the centre is wrong, so
predicted spread and accuracy are separate failures.

### Spread coverage explains 38% of the comb-to-stochastic gap

The two policies do not produce the same spread distribution. The comb policy
puts 1408 cruise frames above 20 rev/s of spread; the stochastic policy puts
23680 there, 17 times as many. Wide spread is the worst bucket for both models,
so part of the gap is which frames each family generates rather than how hard
each family is:

| | cruise PIT MAE |
|---|---|
| comb model on its own spread distribution | 1.30 |
| stochastic model on its own distribution | 8.67 |
| stochastic model reweighted to the comb's spread distribution | 5.84 |

Reweighting closes 38% of the gap, so the remaining 62% is genuine difficulty.
The comb model's reweighted figure of 1.30 reproduces its known campaign figure
of 1.29, which is the consistency check on these buckets.

**These are cruise columns only** and are not comparable to the campaign's
all-regime numbers: the stochastic arm's all-regime figure is 3.70 against the
8.67 here, because the easy zero- and low-speed regimes pull that average down.
The comb policy is nearly all cruise, which is why its two figures agree.

### What this changes

1. The next lever is whatever breaks the fixed-fan solution, not capacity and
   not more data from the same generator.
2. Rotor spread is a coverage axis in its own right. The real split's flight
   frames average 13.7 rev/s of spread; this generator gives 9.4 at
   aggressiveness 1.0, and 9.4 is exactly what the model learned to emit.
3. The line-width ladder is a test of this: if sharper lines let a model resolve
   rotors individually, its predicted spread should start following the true
   spread at rung 0 and stop following it as the rungs widen. That is a sharper
   read on each rung than PIT-MAE alone, and `spread_eval.py` measures it.

## The peeling probe failed as an instrument

To ask whether four-rotor recovery is achievable at all without a network,
`scripts/comb_peel_probe.py` finds the best f0 by harmonic sum, suppresses that
comb, and repeats four times, against a hand-built control: find ONE f0 and
answer with four speeds evenly spaced by the same 10 rev/s width the networks
converged on. Median PIT MAE over 84 cruise frames per policy, speech at −30 dB:

| policy | peel | fan | true spread |
|---|---|---|---|
| `ladder_r4` (full family) | 18.87 | 6.70 | 10.03 |
| `ladder_r0` (sharp lines) | 14.68 | 4.42 | 9.61 |
| `comb_floor_s1` (static comb) | 4.06 | 3.36 | 10.37 |

**Peeling loses to the fan on every family**, by 12 rev/s on the full one. That
is not a result about the data; it means the peeler is a bad estimator. The
likely mechanism is that the four combs interleave, so suppressing a band around
each `k * f0` removes the neighbouring rotors' lines along with the target's,
and later picks land on aliases. **The identifiability question remains open.**
Peeling does close on the fan where lines are sharp (4.06 against 3.36 on the
static comb, against 18.87 against 6.70 on the full family), which is consistent
with sharpness helping, but a failing estimator cannot carry that claim.

### What does survive: the fan is competitive with the trained networks

The fan control scores 3.36 to 6.70 across the three families. The trained
networks score 1.30 (comb) and 8.67 (stochastic, cruise-only). So a hand-built
heuristic that estimates ONE frequency and pads it with a fixed spread is in the
same range as networks trained for tens of epochs, and on the full stochastic
family it is ahead of one.

**This is suggestive, not a like-for-like comparison.** The fan probe fixes
speech at −30 dB while the network evaluation draws SNR over the policy's full
−30 to 0 dB range, and `ladder_r4` reproduces the family's default ranges while
the networks trained on `stoch_s1id`, which adds line-visibility constraints. The
two numbers are not directly comparable and no ranking should be read from them.
What they do support is the finding above: whatever the networks learned on the
stochastic family, a one-frequency-plus-fixed-fan heuristic reaches it.

## The comb floor exists, and DEPTH moves it while width does not

Three transformer heads trained to saturation on the static comb, validated on
the comb itself at a held-out seed, patience 200 and epochs 2000, all training
at the 8 s length they are scored at:

| arm | head | head params | floor (val RMSE) | at epoch | epochs past min | best MSE |
|---|---|---|---|---|---|---|
| `comb_floor_base` | 2 x 64, 4 heads | 0.141M | 2.535 | 46 | 202 | 8.345 |
| `comb_floor_deep` | **4** x 64, 4 heads | 0.241M | **2.155** | 60 | 201 | **6.205** |
| `comb_floor_wide` | 2 x **128**, 8 heads | 0.479M | 2.968 | 39 | 177 | 11.418 |

All three arms are finished, each having run 177 to 202 epochs past its own
minimum without improving on it. The floors are converged, not snapshots.

**There is a floor.** `base` found 2.535 at epoch 46 and then ran 202 further
epochs without improving on it. The static comb is not solvable to near-zero
precision by this architecture at this scale, which answers the question the
campaign opened with: models cannot reach near-perfect precision on a static
comb, and the training loss does hit a hard limit.

**Depth moves the floor; width moves it the wrong way.** Doubling the temporal
transformer's DEPTH takes the floor from 2.535 to 2.155, a 15% reduction, and
the MSE from 8.345 to 6.205, a 26% reduction. Doubling its WIDTH instead — key,
value and head dimensions — puts the floor at 2.968, 17% WORSE than base.

The two axes were deliberately not parameter-matched, because doubling a
transformer's model dimension is quadratic while doubling its layers is linear.
That asymmetry now works in the result's favour: `wide` carries 3.4x base's head
parameters and loses, while `deep` carries 1.7x and wins. **The binding
constraint is sequential processing depth, not per-time-step representation
capacity, and it is not parameter count** — the arm with the most parameters is
the worst of the three.

Parameter count does not order the result at all. Ranked by head parameters the
arms run 0.141M, 0.241M, 0.479M; ranked by floor they run 2.155 (`deep`), 2.535
(`base`), 2.968 (`wide`). The middle-sized arm wins and the largest loses. In
MSE the spread is wider still: 6.205, 8.345, 11.418 — depth takes 26% off base
and width adds 37%.

**Width does not merely fail to help; it hurts.** Whatever the temporal head
needs in order to read a comb, a richer per-time-step representation is not it,
and paying 3.4x the parameters for one makes the model measurably worse.

### This refutes a prediction made in this campaign

From the fixed-fan measurement — all three architectures emitting a constant
10 rev/s spread whatever the rotors do — this campaign predicted that capacity
would not move the comb floor, and that the limit would prove to be the signal
representation. Depth moved it by 15%. The prediction was wrong, and the runs
settled it rather than the argument.

The fixed-fan finding still stands on its own measurement. What it does not
support is the inference that was drawn from it about capacity.

### A note on the 8 s fix

The cancelled 1 s-trained `base` run reached a BETTER best than its 8 s
replacement (2.372 against 2.535) before diverging. Training at the scoring
length did not improve the best number; it made training stable — the 8 s run
sits flat near 2.7 while the 1 s run ran away to 4.08 and climbing. For a floor
measurement stability is what matters, but the fix should not be credited with
an accuracy gain it did not deliver.

## The rung-length calibration, from the comb-floor runs

The curriculum was specified as running each rung for a number of steps
calibrated on the comb-only experiments. Those experiments have now supplied the
number. Trained to saturation on the static comb, every arm found its floor
early and then never improved on it:

| arm | floor at epoch | epochs past the minimum with no improvement |
|---|---|---|
| `comb_floor_base` | 46 | 202 |
| `comb_floor_deep` | 60 | 201 |
| `comb_floor_wide` | 39 | 80 (still running) |

Saturation on a comb-only task takes **40 to 60 epochs**. Rungs 1 to 4 therefore
carry patience 100 — more than twice the observed requirement, so no arm the
comb runs resemble would be truncated, while patience 200 spent about 200 idle
epochs per arm. Rung 0 was already running at 200 and was left alone rather than
restarted for a saving.

Patience rather than a fixed step count: a wider rung may genuinely need longer
than a sharp one, and the ladder does not assume it knows which. The comb runs
calibrate what patience has to be, which is what the specification asked for.

### The open follow-up

The ladder runs the 1.5M `simple_conv_v2` trunk, chosen as the campaign's
reference. The comb-floor result argues for revisiting that: depth moved the
comb floor 15% while width made it worse, so a depth-scaled temporal head is the
better probe of what the DATA allows, as opposed to what this trunk allows. A
ladder run on the weaker architecture measures the architecture's limit where it
is lower than the data's. Rung 0 is already training on `simple_conv_v2`, so the
chain is left as it is and the depth-scaled ladder is recorded as the natural
second pass rather than a mid-flight change.

## Depth improves the centre; it does not break the fan

The three comb-floor checkpoints, scored by rotor-spread bucket on the comb they
trained on (`best` checkpoint, 100 clips, cruise columns):

| true spread (rev/s) | 0.20 | 4.34 | 8.97 | 11.57 | 21.04 |
|---|---|---|---|---|---|
| `base` predicted spread | 6.07 | 10.54 | 10.74 | 10.62 | 10.39 |
| `deep` predicted spread | 5.23 | 10.19 | 10.61 | 10.37 | 10.44 |
| `wide` predicted spread | 7.50 | 12.06 | 12.12 | 11.97 | 11.49 |
| `base` PIT MAE | 3.45 | 2.50 | 1.60 | 1.58 | 3.33 |
| `deep` PIT MAE | **2.92** | **2.14** | **1.16** | **1.31** | **3.26** |
| `wide` PIT MAE | 3.50 | 2.81 | 2.03 | 1.82 | 3.77 |

**The fixed fan survives depth scaling.** `deep` wins every bucket, and its
predicted spread is as pinned as anyone's: 10.2 to 10.6 across true spreads from
0.20 to 21.04. Depth buys a better CENTRE for the fan, not per-rotor resolution.
The 15% floor improvement is entirely a centre-estimate gain inside the same
degenerate solution.

`wide` shows the same thing from the other side. Its fan is both wider (11.5 to
12.1 against base's 10.4 to 10.7) and worse in every bucket — the extra
per-time-step capacity bought a more confidently wrong fan.

### This reconciles the two results

Earlier in this campaign the fixed-fan measurement was used to predict that
capacity would not move the comb floor. The floor moved, and that prediction was
recorded as refuted. This measurement shows the prediction was right about the
FAN and wrong about the FLOOR, and that the two are separable: architecture
scaling reduces error inside the mean-tracking solution without escaping it.

The consequence for the curriculum is direct. If no architecture axis breaks the
fan — depth improves within it, width degrades within it — then the lever that
might break it is the DATA, which is what the ladder varies. That is now the
ladder's sharpest question, and `spread_eval.py` reads it per rung: does a rung's
model start tracking true spread, or does it just re-centre the same fan?

## Training to convergence on the stochastic family

`stoch_long_scv2`, the 1.5M trunk at patience 200 and epochs 2000, on half-real
half-synthetic validation, trained at 8 s:

| run | epochs | val RMSE min | at epoch | past min |
|---|---|---|---|---|
| `stoch_long_scv2` (1.5M) | 229 | 10.304 | 202 | 27 |
| `stoch_long_trxxl` (38M) | 157 (running) | 13.207 | 31 | 126 |

Its 1 s-trained predecessor reached 11.265 before being cancelled, so training
at the scoring length is worth about 8.5% here — unlike the comb case, where the
same change bought stability rather than accuracy.

**Width scaling hurts on the stochastic family too.** The 38M `trxxl` sits 28%
worse than the 1.5M trunk (13.207 against 10.304) and is climbing away from its
epoch-31 minimum. `trxxl` scales `width` to 384 with an 8 x 512 head, which is
predominantly the axis the comb-floor study found harmful, and it fails the same
way on a different family.

**These are half-real, half-synthetic numbers** and are not comparable to this
campaign's synthetic-only or real-only figures.

### The stopping rule terminated a run that was still improving

`stoch_long_scv2` reached its RMSE minimum at epoch 202 and stopped at 228 with
only 27 epochs past it, far short of patience 200. The arithmetic gives it away:
the monitored metric is MSE, whose minimum was at about epoch 28, and 28 + 200
is 228. MSE patience ended a run whose RMSE was still descending.

This is the fourth time in this campaign that MSE and the error metric the work
is judged on have disagreed, and the first time the disagreement has silently
truncated a run. Any conclusion from this arm about what convergence on the
stochastic family costs is therefore a LOWER bound on the epochs required, not a
measurement of it.

## Why pi_kalman degrades a perfect initialization: twin collisions

`pi_kalman` never refines a trajectory. It estimates a correction `dr(t)` and
adds it (`r_hat += dr`, module docstring step 4), so handed the exact answer it
still estimates a correction from noisy measurements and applies it. There is no
do-nothing outcome in the architecture. Measured on the deterministic static
comb with an EXACT initialization:

| configuration | error injected into a perfect init |
|---|---|
| `k_scaled`, 12 iterations | 0.0279 |
| `k_scaled`, 6 iterations | 0.0161 |
| `fixed` band, 12 iterations | 0.1120 |
| **four rotors** | **0.0160** |
| **one rotor, collision-free by construction** | **0.0006** |

**The bias is cross-rotor collisions.** A single-rotor comb — where a collision
cannot occur — drops the injection by a factor of 27, so about 96% of what the
estimator adds on a four-rotor comb is the two-phasor winding-number bias its own
docstring warns about, leaking past the twin gate. Nyquist crossing is excluded
by arithmetic: at `k_max` 40 and 75 rev/s the top order sits at 3 kHz, far inside
the band.

**Neither trust knob can fix it**, and the reason is structural rather than a
matter of tuning:

| knob | swept | injected error |
|---|---|---|
| `sigma_prior` | 2.0 → 0.02 | 0.0161 → 0.0160 |
| `sigma_process` | 2.0 → 0.25 | 0.0279 → 0.0286 |

`p0 = sigma_prior**2` is the variance of `dr` at the FIRST FRAME ONLY; the state
is re-informed by measurements at every subsequent frame, so across ~500 frames
its influence is negligible. `sigma_process` governs how fast `dr` may wander,
not whether it is zero. **No parameter says "the trajectory I was given is
correct"** — expressing that needs both to reach zero together, which makes the
filter a no-op by construction.

### Consequence for the blind chain

Applying it after the blind Viterbi ladder makes things monotonically worse:

| rung | PIT MAE |
|---|---|
| `vit2dsp` (blind) | 2.744 |
| + `pi_kalman` x1 to x4 | 2.979, 3.186, 3.310, 3.384 |

So on the deterministic comb `pi_kalman` should not be in the chain as
configured. The indicated fix is in the GATING — a wider twin guard, or the
joint two-tone pair mode that resolves the two lines instead of averaging their
rotation — not in the filter. The gap it would unlock is large: oracle-init
refinement reaches 0.028 rev/s on a four-rotor comb and 0.0006 on a single-rotor
one, against a fully blind 2.744.

## The curriculum, running

The ladder walks the stochastic family on line width, the axis measured to move
comb contrast, with `rolloff_p` and `gamma0_hz` held at the family default so
exactly one thing changes. Rung 4 reproduces the family defaults, so a model
finishing the ladder has arrived at the distribution the transfer campaign's
stochastic arms trained on directly.

| rung | `gamma_slope_hz` | val min | at epoch | epochs past min |
|---|---|---|---|---|
| `ladder_r0_scv2` | 0.0 | 16.669 | 39 | 131 |
| `ladder_r0_deep` | 0.0 | **16.142** | 4 | 109 |
| `ladder_r1_scv2` | 0.0 to 0.05 | **15.970** | 18 | 101 |

Each rung warm-starts from the previous one's best checkpoint and is validated
on half the frozen real split and half synthetic drawn from its OWN rung, so the
two curves come out together: the achievable synthetic fit as the family widens,
and whether real-rig transfer follows it.

**Patience 100 held.** The comb-floor runs put saturation at epoch 40 to 60, and
rung 1 stopped at 119 epochs with its minimum at 18 — 101 past it, no truncation.
The calibration the specification asked for is doing its job.

**Two early readings, both provisional at one seed.** Depth helps here as it did
on the comb: `ladder_r0_deep` beats `ladder_r0_scv2` by 3.2% (16.142 against
16.669), consistent with the comb-floor result that depth is the scaling axis
that moves the floor. And rung 1 is BETTER than rung 0 (15.970 against 16.669)
even though its distribution is wider — the ladder is not monotone in difficulty
so far. That is not a contradiction: validation is half real, and a slightly
wider training distribution may transfer to the real half better than the sharp
corner does. Whether that survives the wider rungs is what r2 to r4 measure.

## The joint four-rotor filter: built, tested, refuted

The sequential refiner estimates one rotor at a time, so a demod band shared by
two rotors holds a two-phasor sum whose argument advances at neither line's
rate. The sequential design can only discard such a measurement. The joint
model keeps it: the state becomes the whole correction vector, and a shared
band contributes one DENSE observation row,

    dpsi ~= sum_m w_m * 2 pi k_m dt * dr_rot(m),   w_m = P_m / sum P

the power-weighted mean of the member lines' increments. A clean band gives
back the sequential row exactly. Implementation: `src/tracking/joint_phase_kalman.py`
(4x4 information matrix, matrix Kalman and RTS, per-band excess variance
measured apart on collided and clean frames). Probe: `scripts/joint_kalman_probe.py`.

### The 27x collision cost was a confound

The number that motivated the joint model — 0.0160 rev/s injected into an exact
initialization with four rotors against 0.0006 with one — came from two
DIFFERENT synthesis paths. The one-rotor control was built directly (smooth
trajectory, one channel, its own noise floor) because the pool's frame geometry
fixes four rotor positions. On matched synthesis, where only the rotor count
changes, the ladder is:

| rotors | injected error (rev/s) |
|---|---|
| 1 | 0.0006 |
| 2 | 0.0115 |
| 3 | 0.0431 |
| 4 | 0.0593 |

### The joint model buys nothing

At matched settings the joint filter and the sequential filter agree. Three
weight modes, two band widths, four clips, exact initialization:

| band_b0 | seq | joint/power | joint/hard | joint/drop |
|---|---|---|---|---|
| 0.35 | 0.0593 | 0.0510 | — | 0.0474 |
| 0.15 | 0.0218 | 0.0211 | 0.0216 | 0.0214 |

`drop` gates collided measurements out, which decouples the joint filter into
the per-rotor scalar filters — it agrees with the other two modes, so keeping
the collided measurements is worth nothing either. `guard_hz` is inert (0.0593
at 1.0 Hz against 0.0570 at 0.15 Hz). The apparent 2.8x win in the first run
was entirely a band-width difference: the joint module defaulted to
`band_b0=0.15` and the sequential refiner defaults to 0.35.

**Cross-rotor attribution is refuted as a lever.** The joint filter costs 2x the
runtime and returns the sequential answer.

### The band width is the lever, and it is the whole story

Sweeping `band_b0` on an exact initialization:

| band_b0 | R=1 | R=4 | ratio |
|---|---|---|---|
| 0.35 | 0.0006 | 0.0593 | 98x |
| 0.25 | 0.0006 | 0.0444 | 74x |
| 0.15 | 0.0006 | 0.0218 | 37x |
| 0.08 | 0.0006 | 0.0082 | 14x |
| 0.04 | 0.0006 | 0.0039 | 7x |
| 0.02 | 0.0006 | 0.0022 | 4x |

The one-rotor floor is FLAT — it is not band-limited at all. The four-rotor
error is very nearly proportional to the band width, at about `0.15 * b0`. The
multi-rotor cost is in-band leakage from the other combs, and it is bought back
by narrowing the band, not by modelling the interferers.

### The refiner has an attractor, and it is not the truth

Iterating one call from an exact initialization at `b0=0.35` gives 0.0593,
0.0987 and 0.1258 at 6, 12 and 20 iterations. Repeated one-pass refinement from
a 0.30 rev/s displaced initialization converges to 0.11 and stays. The refiner
therefore has a fixed point at about `0.15 * b0`, approached from below when the
input is better than it and from above when it is worse. This is the mechanism
behind the blind ladder's 2.744 -> 3.384 degradation.

### A trust prior exists but does not escape the wall

`pi_kalman_refine` gained `sigma_trust` (new): a zero-mean pseudo-measurement of
the correction on every frame, `info[j] += 1 / sigma_trust^2`. It is the missing
term — `p0` constrains frame 0 alone, so nothing said the input was worth
anything. Default `None` leaves the filter bit-identical.

It behaves exactly as a prior should, which is to say it helps only when the
initialization really is better than the attractor:

| displacement | no refine | None | 0.3 | 0.1 | 0.03 | 0.01 |
|---|---|---|---|---|---|---|
| 0.10 | 0.0810 | 0.0572 | 0.0543 | **0.0415** | 0.0558 | 0.0739 |
| 0.30 | 0.2429 | **0.1071** | 0.1370 | 0.1617 | 0.2106 | 0.2396 |

### The real wall: the error tail sets capture, the error bulk sets precision

A geometric band anneal (0.35 down by 0.6 per pass) plateaus at 0.1201 rather
than following the attractor down, and an ORACLE capture-respecting anneal
(`b0 = margin * max|error|`, using the truth) runs BACKWARDS: from a 0.02 rev/s
initialization it goes 0.0092 -> 0.0102 -> 0.0127 -> ... -> 0.0296 while the
band it demands grows 0.070 -> 0.204.

The cause is visible in those two columns. At MAE 0.0092 the worst frame is
0.047 — five times the typical one. The band must cover the tail to keep
capture, but running at the tail's width re-injects error across the bulk, which
fattens the tail, which widens the band. **The band width needed for capture is
set by the worst frame; the band width that gives precision is set by the
typical frame; and a constant-in-time band cannot serve both.**

That names the next lever precisely, and it is not a joint state: the band does
not have to be constant in time. The smoother already produces a per-frame
posterior. A per-frame adaptive band — wide where the posterior is loose, narrow
where it is tight — is the one change that attacks the actual constraint.

## Solving the static comb: the search half

With a good initialization the static comb is already solved — 0.0006 rev/s at
one rotor, 0.0022 at four. The blind seed handed to that machinery carried 2.744
rev/s, which is 100 times outside the refiner's own capture range, so every
refinement stage was running blind of its assumptions. The problem is therefore
SEARCH, not precision.

### The Whittle comb score, and why the logarithm is the whole trick

Over a window short enough that a rotor's rate is nearly constant, score a
candidate rate by the log-likelihood of a line-plus-noise model summed over its
harmonics, `S(r) = (1/K) sum_k log(1 + Y(k r) / sigma^2)`.

A plain harmonic SUM does not work, and its failure is instructive: the
half-rate `r / 2` covers every line `r` has AND pockets the noise in the bins
between, so it outscores the truth. Measured on a four-rotor comb, the plain
sum ranked `76.16 / 2 = 38.08` first and the true rates second and fourth. Under
the logarithm an empty bin contributes zero instead of noise, and the same
window ranks the three true rotors first, second and third (errors +0.003,
+0.002, +0.006 rev/s), with the aliases pushed to ranks four through six.

### Peeling, and two corrections that were needed

The strongest comb masks the weaker ones, so each pick's lines are notched out
and the scan repeats. Three further findings, each measured:

* **Rate-space exclusion HURTS.** Barring a neighbourhood of a picked rate to
  stop duplicate picks also forbids the second rotor of a close pair, and rotors
  cross often. Set error rose from 2.01 to 2.58 rev/s. Both exclusion knobs
  default to zero.
* **The octave test must be a RATIO.** Absolute harmonic presence sits on a
  knife edge (0.70 against a 0.75 cut) and flips with the peel order, and it
  cannot reject a MULTIPLE at all, because a multiple's harmonics are a subset
  of the true comb's and are therefore always present. The ratio of odd-harmonic
  to even-harmonic level has neither weakness: a half-rate scores low by
  construction, a true rate scores near one. That fixed the rotor that was
  otherwise never found directly — before it, the third peel was that rotor's
  half-rate in EVERY window.
* **The octave test must be clamped to the search range.** Unclamped doubling
  overshot to 140 and 152 rev/s against a 30-100 band. Those few windows alone
  moved the mean window error from 0.45 to 6.14 while the median stayed at 0.05.

### Where this lands

Per window, four peels, four rotors, no oracle:

| metric | value |
|---|---|
| median set error | **0.046 rev/s** |
| windows within 0.1 | 76% |
| windows within 0.5 | 83% |
| windows worse than 1.0 | 16% |
| mean set error | 1.15 |

The median is 60 times better than the 2.744 rev/s seed it replaces and sits
well inside the refiner's capture range. The mean does not, because the 16% of
failed windows all fail the same way: one rotor's comb is too weak to be found
at all, and a junk candidate takes its place.

**End to end the goal is NOT met.** Stitching windows into tracks and refining
gives 0.88 rev/s in the well-separated regime and 2.6 in the moderate one,
against the 2.744 baseline and the 0.0022 floor. The aggregate is dominated
entirely by the failed windows, and neither a Viterbi chaining nor a
Hungarian-matching tracker recovered them — both scored WORSE than simply
sorting each window's four peels, because a tracker given a window with only
three real rotors has nothing to match the fourth against.

What is solved: the precision, and the per-window search in 83% of windows.
What is not: the windows where a rotor is too quiet to detect, and the temporal
association across them. That is where the next work belongs — most likely by
scoring a rotor's presence over a longer span than one window, so a rotor that
is momentarily weak is carried by its own history rather than re-detected from
scratch each time.

Code: `src/tracking/comb_seed.py` (Whittle score, peeling, octave correction),
`src/tracking/comb_fit.py` (the global coherent objective and its basin, peak at
the truth for K >= 30 and T >= 1 s, half-width 0.09-0.6 rev/s).

## The comb-gram: do not threshold early

The per-window peel scan reached a median set error of 0.046 rev/s but failed in
16% of windows, and those windows set the aggregate. The failure was always the
same: one rotor's comb was too weak to reach the top four peaks in that window.
Extracting peaks and then tracking them cannot recover it, because a tracker
handed a window with three real rotors has nothing to match the fourth against —
both a Viterbi chaining and a Hungarian matching scored WORSE than simply
sorting each window's peaks.

The fix is to stop thresholding early. A rotor that is momentarily weak still
leaves a CONTINUOUS ridge in the score surface, so `seed_from_gram` builds the
whole (window x rate) Whittle surface and takes each track as the best-scoring
smooth path through it, recovering the rotor from its own history rather than
re-detecting it in every window.

Four things had to be right, each found by measurement:

1. **Peel in the spectrum, not in rate space.** Suppressing the surface near a
   found track does not work: a strong rotor's score has sidelobes wider than
   any exclusion narrow enough to keep a close pair apart, so the second path
   lands beside the first. Tracks one and three came back as the same rotor
   (64.80 and 64.81, true rates 64.81 / 71.85 / 78.23 / 84.88). Notching the
   track's comb out of every window's spectrum and rescoring fixes it.
2. **Suppression must be a hard exclusion.** Marking cells just below the
   surface minimum is not enough — one window's score is order 1 while a path
   accumulates tens, so the Viterbi re-uses a suppressed ridge rather than pay
   a transition.
3. **Octave decisions belong to the track, not the window.** A per-window
   majority vote failed on about one rotor in twenty; one track sat at 35.96
   against a true 71.86 and that single track produced a whole 1.82 rev/s
   average. Accumulating the odd-to-even harmonic evidence over every window of
   the track first, and reading the peeled spectra rather than the original
   signal, fixed it.
4. **The window length is 0.25 s and shortening it is catastrophic.** Halving
   it to reduce within-window smear cost far more in resolution than it bought:
   0.033 -> 7.77 rev/s on the separated regime.

### Result

Set error, four rotors, five clips per regime, fully blind:

| regime | median rotor separation | seed | + refinement |
|---|---|---|---|
| separated (distinct set-points) | 4.83 | **0.0374** | 0.0347 |
| moderate (fast slew, apart) | 2.12 | 1.20-2.54 | — |
| crossing (trajectories overlap) | 1.54 | 0.554 | — |

In the separated regime — the realistic one, where a real airframe holds its
rotors at distinct set-points — this is **0.037 rev/s against the 2.744 rev/s
seed it replaces**, a factor of 73, with a worst clip of 0.059. The harder
regimes are not solved: they still lose a clip now and then to two tracks
landing on one rotor.

### Why refinement cannot finish the job yet

Refining the 0.037 seed does not improve it (0.0347 at `b0=0.35`, 0.0377 at
0.02). That is exactly the capture-versus-precision wall measured earlier in
this document: the wide band's attractor is about `0.15 * b0 = 0.05`, which is
WORSE than the seed, and the narrow band's capture is 0.02, which is TIGHTER
than the seed. There is no band that both captures 0.037 and improves on it.
Closing the last factor of twenty to the 0.0022 floor needs the per-frame
adaptive band, which was measured to be worth about 1.5x on its own.

## Reliability: what the regimes were really measuring

The first comb-gram numbers looked unreliable outside the well-separated
regime. Two of the three causes turned out to be defects in the method, and
one was a defect in the BENCHMARK.

### The benchmark defect: independent rotor profiles

The synthetic clips drew an independent harmonic profile per rotor, which
produced a 21.6 dB spread between the loudest and quietest rotor of a clip. The
loud rotor's imperfectly-notched residue then outscored the genuine quiet
rotors, and three of four tracks piled onto one rotor. Rotors on a real airframe
are near-identical. Re-running with ONE shared profile, twelve clips per regime:

| regime | independent profiles | shared profiles (a real airframe) |
|---|---|---|
| separated | 0.321 (worst 1.80) | **0.0208**, 0/12 failures |
| moderate | 1.363 (worst 3.40) | **0.0384**, 0/12 failures |
| crossing | 1.301 (worst 1.83) | 0.215-0.296, 2-4/12 failures |

Two attempts to fix the loud-rotor residue with a wider notch both failed and
are recorded in the code: widening in proportion to the line's sweep helped that
regime (2.54 -> 1.20) and broke the one where rotors pass close (0.55 -> 1.51),
and growing each notch out to its local valley was worse still (0.037 -> 0.97),
because at the first harmonic the rotors are only a few Hz apart.

### The method defect: the smoothness cost must be a hinge

The Viterbi was returning paths that HOP between rotors and alias ridges
collecting each window's maximum: measured on one clip after two peels, the
wandering path scored 1.62 per window against 0.87 for the true rotor, so the
optimizer was right and the objective was wrong. A plain quadratic stiff enough
to stop the hop also blocks a fast rotor's real motion — it improved two regimes
(0.0374 -> 0.0326, 2.54 -> 1.75) and destroyed the third (0.554 -> 5.27).

A hinge fixes it: free up to `slew`, steep past it. Physical slew and a rotor
hop differ by nearly an order of magnitude (about 4 rev/s^2 against 24).

`slew` is then a PHYSICAL parameter — the airframe's maximum rotor acceleration
— and it must cover the rotors' real motion. Set below it the tracker returns
over-spread tracks; that was the whole of the remaining `crossing` failure, whose
true peak slew is 8.75 rev/s^2 against a free band of 6. Matching it is worth
real accuracy: the regime whose true peak is 5.83 scores 0.038 at `slew=6` and
0.092 at `slew=12`.

### A hypothesis of mine that the data refuted

I assumed the `crossing` failures were a genuine identifiability limit — two
rotors at the same speed emit the same comb. They are not. The failing clips
have LESS trajectory coincidence than the passing ones (11.2% of frames with
rotors within 0.3 rev/s, against 15.4%), so the limit is algorithmic.

### Where it stands

With shared rotor profiles and `slew` matched to the airframe, twelve clips per
regime, fully blind, four rotors:

| regime | set error | failures |
|---|---|---|
| rotors hold distinct speeds | **0.021 rev/s** | 0/12 |
| rotors slew fast but stay apart | **0.038 rev/s** | 0/12 |
| rotors fully interleave | 0.21 | 2/12 |

Against the 2.744 rev/s seed this replaces, that is a factor of 130 in the two
regimes that a real airframe produces, with no failures in 24 clips. The third regime — rotors sweeping +-6 rev/s across a 10 rev/s spread, so the
four trajectories interleave continuously — still loses one clip in six.

### The interleaving failure, characterized

Both failing clips of twelve show the same thing, and it is NOT a failure to
follow motion: every track's standard deviation matches its rotor's (3.8-4.2
against 4.2-4.7), so the hinge is doing its job. What fails is IDENTITY at a
crossing.

| clip | symptom |
|---|---|
| 9202 | one rotor (72.53) missed entirely; a spurious track appears at 66.83, below the whole ensemble |
| 9203 | two tracks split the difference between two crossing rotors — means 75.50 against a true 77.08 and 72.52 against a true 70.81, with one track's std too high (5.89) and the other's too low (3.65) |

The cause is that tracks are found GREEDILY, one at a time. Where two rotors
cross, the first path may follow either branch through the crossing, and
whichever it takes, its comb is notched out along a trajectory that is partly
rotor A and partly rotor B. The remaining rotors are then no longer clean single
ridges, so the next path either splits the difference or gives up and takes a
spurious ridge.

The fix this names is joint multi-track assignment — find all R paths at once
under a disjointness constraint, rather than peeling one at a time — which is
the same lesson as the peaks-versus-surface one above, applied to tracks instead
of to windows: do not commit to track 1 before track 2 has had a say.

### Coordinate descent does not repair the greedy commitment

The characterization above names a fix: re-solve each track against the others,
so a path that took the wrong branch through a crossing can take it back. It was
implemented (`n_refine` in `seed_from_gram`) and it does not work.

| n_refine | separated | moderate | crossing | total failures |
|---|---|---|---|---|
| 0 (greedy) | 0.0208, 0/12 | 0.0916, 1/12 | 0.2146, 2/12 | 3/36 |
| 1 | 0.0245, 0/12 | 0.0965, 2/12 | 0.1885, 1/12 | 3/36 |
| 2 | 0.0254, 0/12 | 0.0972, 2/12 | 0.1898, 1/12 | 3/36 |
| 4 | 0.0256, 0/12 | 0.0970, 2/12 | 0.1890, 1/12 | 3/36 |

It converges after ONE pass — 1, 2 and 4 are the same to three decimals — and
moves a failure from one regime to another rather than removing it, at three to
five times the runtime. This is exactly what coordinate descent from a greedy
initialization does: it finds the nearest fixed point, and the nearest fixed
point is inside the basin the greedy sweep already committed to. `n_refine`
defaults to 0.

Escaping the basin needs a genuinely joint search — k-best paths per track and
then an assignment over the combinations — not a better local step. That is the
open item.

### And multi-restart does not repair it either

If coordinate descent cannot leave the greedy basin, the other escape is to
start in a different one. `n_restart` runs the greedy sweep once per band of the
rate range — guaranteeing a restart that begins on each rotor — and keeps the
solution with the least residual energy after all R combs are notched out.

| n_restart | separated | moderate | crossing | total |
|---|---|---|---|---|
| 1 (greedy) | 0.0208, 0/12 | 0.0916, 1/12 | 0.2146, 2/12 | 3/36 |
| 4 | 0.0207, 0/12 | 0.1003, 3/12 | 0.2060, 1/12 | 4/36 |
| 8 | 0.0222, 0/12 | 0.0829, 2/12 | **0.9753**, 2/12 | 4/36 |

More restarts make it WORSE, and that is the informative part. The search is
fine — the JOINT OBJECTIVE is not. Residual energy after notching rewards
notching wherever energy happens to be rather than where combs actually are, so
it can prefer a solution parked on loud junk; given more candidates, a flawed
discriminator simply has more chances to choose one. At 8 restarts the
interleaving regime degrades from 0.215 to 0.975 rev/s with a worst clip of 10.3.

So two candidate fixes have now been implemented and refuted, and between them
they locate the real gap. It is not the local step (coordinate descent converges
in one pass) and not the starting basin (restarts reach the right basins and
then discard them). It is the SCORE that ranks whole solutions. A usable one has
to be the Whittle likelihood of the full R-comb model, normalized so that
notching more bins is not itself rewarded. Both knobs default to off.

### The joint score, third attempt — and what the three attempts prove

Three objectives for ranking whole solutions were implemented and measured. Each
fails in its own direction, and the directions are what identify the right one:

| objective | failure mode |
|---|---|
| residual energy after notching | rewards notching wherever energy is, not where combs are — prefers a solution parked on loud junk |
| mean Whittle evidence per line | rewards DUPLICATES — four tracks stacked on one loud rotor score perfectly, since every line they claim is real (worst clip 10.3) |
| **Whittle evidence over the UNION of covered bins** | neither incentive survives: a duplicate track adds no bins, and covering an uncovered rotor adds all of its lines |

With the union objective and 8 restarts:

| regime | greedy (1 restart) | union score, 8 restarts |
|---|---|---|
| separated | 0.0208, 0/12 | 0.0206, **0/12** |
| moderate | 0.0916, 1/12 | **0.0645**, worst 0.119, **0/12** |
| crossing | 0.2146, 2/12 | 0.2752, 3/12 |

The fast-slew regime is now fully solved — its failure WAS a basin problem, and
a correct joint score plus enough restarts finds the right basin. The
interleaving regime is not. It does not improve under any of the three
objectives at any restart count, and coordinate descent does not touch it
either.

**That is the conclusion worth keeping.** The interleaving failure is neither a
local-step problem, nor a starting-basin problem, nor a scoring problem — all
three have now been tested and eliminated. What is left is the ridge MODEL: one
smooth path per rotor through a score surface, with a hinge cost on rate change.
Where four trajectories interleave and swap order, the evidence for "which ridge
belongs to which rotor" is not present in the surface at all, and no search over
paths in that surface can recover it. Resolving it needs information the surface
does not carry — the phase continuity of the individual lines, which is exactly
what the phase-increment refiner reads and what the seeding stage currently
discards.
