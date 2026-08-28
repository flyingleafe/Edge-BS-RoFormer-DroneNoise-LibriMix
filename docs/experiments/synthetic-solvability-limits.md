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
