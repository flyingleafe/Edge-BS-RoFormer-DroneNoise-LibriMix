# Synthetic-only transfer with the stochastic noise family

**Question.** Can a rotor-speed predictor trained on synthetic noise alone reach
the real frozen validation split, if the synthetic family is wide enough to
contain the real thing?

**Status: OPEN.** Started 2026-08-25.

**Result so far.** `stoch_s1g_scv2` reaches **172.1** validation PIT-MSE on the
frozen split against **204.0** for the best earlier synthetic-only model on the
same trunk and **183.7** for the best on any trunk — the first synthetic-only
model in this project to beat the analytic comb, with no real noise anywhere in
its training stream. Per regime it is 20.27 / 16.20 / 4.50 rev/s on stopped
rotors, ramps and cruise against the comb's 5.64 / 26.32 / 7.09. The
real-trained reference is 17.59 aggregate and 2.87 / 3.48 / 2.49, so the gap is
closed on ramps and cruise and open on stopped rotors, which is what arm J
addresses.

**Where it stands.** The goal is per-regime mean-absolute-error parity with the
best real-trained model. `python scripts/transfer_board.py` prints the standing,
one row per MODEL — never a per-regime best-of across different models, which no
single model achieves:

| model | trained on | all-MAE | zero | low | flight |
|---|---|---|---|---|---|
| `r4hb_scv2` | real | **2.67** | **2.87** | **3.48** | **2.49** |
| `stoch_s1g_scv2` | synthetic | 8.08 | 20.27 | 16.20 | 4.50 |
| `m3abl_comb_unigru128_s1` | synthetic | 8.30 | 4.73 | 24.24 | 6.00 |
| `stoch_s1h_scv2` | synthetic | 9.07 | 27.98 | 26.77 | **2.60** |

The best synthetic-only model is 3.03x the target on all-regime MAE: 7.06x on
stopped rotors, 4.66x on ramps, 1.81x at cruise. One arm has reached cruise
parity on its own (`stoch_s1h_scv2`, 2.60 against 2.49) and no arm has reached
two cells at once.

## Why the question is open

Every synthetic-only predictor this project has trained transfers badly. On the
frozen split `dload:DREGON-LM-V4-michaels-valid-full`, scored as frame-weighted
PIT mean squared error:

| stage-1 arm (synthetic only) | noise family | best val PIT-MSE | at epoch |
|---|---|---|---|
| `m3abl_comb_unigru128_s1` | analytic static comb | **183.7** | 38 |
| `m3abl_comb_scv2_s1` | analytic static comb | 204.0 | 16 |
| `e8_staticcomb_s1_unigru128` | analytic static comb | 222.6 | 7 |
| `e7_gencurric_s1_scv2` | neural generator | 222.8 | 1 |
| `m3cur_unigru128_s1` | generator + comb | 275.4 | 2 |
| `m3cur_transformer_s1` | generator + comb | 316.9 | 0 |
| `m3cur_scv2_s1` | generator + comb | 325.5 | 20 |
| `m3abl_comb_transformer_s1` | analytic static comb | 1802.0 | 0 |

These are the minimum over each run's history. An earlier version of this table
quoted the runs' *last* epoch instead, which flattered nothing but was wrong in
both directions — `m3abl_comb_scv2_s1` reads 204.0 at its best and 336.8 at its
last. The best-epoch column is what the arms have to beat.

Two of these curves say something on their own. The comb runs are violently
unstable on the real split: `m3abl_comb_scv2_s1` swings between 430 and 1557
over its first twelve epochs and only reaches 204 at epoch 16. Several arms peak
at epoch 0 or 1 and then get worse, which is a model drifting away from the real
data as it fits the synthetic one better.

The best real-trained model reaches **17.6** on the same split (`r4hb_scv2`), and
the same comb-pretrained weights reach 25.6 after a real fine-tune
(`m3abl_comb_scv2_s2`). So the synthetic stage learns something worth keeping —
it is the best initialization the project has — and yet it cannot read a real
recording on its own.

The diagnosis in both earlier campaigns was the same: the family is too narrow.
The static comb holds one amplitude profile fixed for a whole clip, by design,
so that comb spacing is the only cue (E8). That design is why it works as
pre-training and also why every one of its clips has the same texture.

## What is new

`data_processing/stochastic_rotor_noise.py` — the generative direction of the
v4 analysis model (`tracking.joint_decompose`). The spectrum is

    S(f, t) = B(f, t) + sum_r sum_k P_rk(t) * Lorentzian(f - k * rps_r(t); gamma_rk)

with a smooth colored floor `B`, one Lorentzian line per harmonic, and **every
amplitude drifting slowly in time as a Gaussian process**. The processes are
drawn independently of the rotor-speed trajectory, so the family keeps the
spacing-only property: no amplitude in the stream carries information about
speed, and a predictor cannot learn the amplitude shortcut that killed the
neural generator (E7).

Each window draws its own harmonic profile per rotor, its own floor color, its
own linewidths, and its own wander rates and wander times, so no two windows
share a texture.

### The family contains the real thing, measured

Comb strength, by the project's own instrument — the modulation depth of the
folded order cell (`joint_decompose.order_cell_profile`), the measure that
cannot be fooled by a broadened or displaced comb:

| | k1-9 | k10-24 | k25-49 | k50-80 |
|---|---|---|---|---|
| real, DREGON free-flight room1 | 1.07 | 1.22 | 0.37 | 0.27 |
| real, DREGON spinning room2 | 0.89 | 0.54 | 0.32 | 0.30 |
| real, Michael's FLY125 | 7.86 | 1.40 | 0.28 | 0.28 |
| stochastic family | 4.44 | 1.17 | 0.54 | 0.37 |

The family sits inside the real spread in every band.

## The arms

`conf/online_mix/stoch_s1_dload.yaml` — the stochastic family at weight 1.0 with
full-flight excitation, a silence arm at weight 0.2, `snr_ref_floor_rms: 0.02`,
and the same three augmentation blocks the comb-only arm uses. Everything except
the noise family is `m3abl_comb_s1_dload.yaml` verbatim, so the rows are
controlled.

Every arm is the convolutional trunk, and each adds one thing to the one above
it. The four in the second block are what is running; B and C were configured,
then cancelled to free their slots once the level measurement arrived and made
them subsets of D.

| experiment | what it adds | comparison row |
|---|---|---|
| `stoch_s1_scv2` | the family, nothing else | `m3abl_comb_scv2_s1` (336.8) |
| `stoch_s1b_scv2` | + room and coloration | cancelled |
| `stoch_s1c_scv2` | + a scattered speed prior | cancelled |
| `stoch_s1d_scv2` | + level invariance | cancelled |
| `stoch_s1e_scv2` | + a realistic per-rotor spread | `stoch_s1_scv2` |
| `stoch_s1f_scv2` | + the ramps the real recordings have | `stoch_s1e_scv2` |
| `comb_fixed_scv2` | **control**: every fix through E, on the OLD comb family | `m3abl_comb_scv2_s1` |

The four that run make a clean square: old family with and without the fixes
(`m3abl_comb_scv2_s1`, `comb_fixed_scv2`), new family with and without them
(`stoch_s1_scv2`, `stoch_s1e_scv2`), plus `stoch_s1f_scv2` for the ramp
coverage that only the measurement of the ramp regime asked for.

The control is what separates the two things the campaign changes at once. If
`comb_fixed_scv2` reaches where `stoch_s1e_scv2` reaches, the fixes did the work
and the family was never the problem. If it does not, the family was.

Stream check: PASS (`python scripts/check_stream.py --experiment stoch_s1_scv2`)
— all three augmentation blocks fire at their configured rates, frequency
scaling changes the labels, and the stream is deterministic per sample id.

## Where the earlier arm fails

`scripts/valid_regime_eval.py` splits the frozen split into three regimes by the
target speeds — zero (every rotor stopped), low (warm-up and the ramps), flight
(mean at or above 45 rev/s) — and reports PIT mean absolute error in each.

Read this table before changing the family: it says which part of the problem
synthetic-only training already solves.

All eight channels, all 37 clips. The scorer reproduces the training-time
`val/mse` of the real-trained row exactly (17.59 against the campaign's 17.6),
which is what makes the rest of the table trustworthy.

| checkpoint | trained on | aggregate | all MAE | zero | low | flight |
|---|---|---|---|---|---|---|
| `r4hb_scv2` | real (comb curriculum) | **17.59** | 2.67 | 2.87 | 3.48 | 2.49 |
| `hb_scv2_mag_nogate` | real (R2) | 22.78 | 2.72 | 3.36 | 4.18 | 2.35 |
| `m3abl_comb_unigru128_s1` | synthetic comb | 190.62 | 8.30 | 4.73 | 24.24 | 6.00 |
| `m3abl_comb_scv2_s1` | synthetic comb | 218.30 | 9.50 | 5.64 | 26.32 | 7.09 |
| `m3cur_scv2_s1` | synthetic gen + comb | 328.96 | 10.41 | 20.30 | 11.33 | 8.55 |

The gap is not uniform, and its shape is the campaign's map:

| regime | best synthetic-only | real-trained | ratio |
|---|---|---|---|
| ramps (low) | 24.24 | 3.48 | **7.0x** |
| cruise (flight) | 6.00 | 2.49 | 2.4x |
| stopped (zero) | 4.73 | 2.87 | 1.6x |

Synthetic-only training already gets stopped rotors nearly right and cruise
within a factor of two and a half. The ramps are where it falls apart, and the
ramps are also where the trajectory model has a measurable coverage hole.

One row breaks the pattern and is worth keeping in view: the neural generator's
curriculum (`m3cur_scv2_s1`) is the best of the three on the ramps (11.33) and
by far the worst on stopped rotors (20.30). The two synthetic families fail in
different places.

## What the failure at speed actually is

Three measurements, in the order that rules things out.

**It is not a harmonic confusion.** On six real cruise clips, the ratio of the
predicted speed to the true speed is 0.841 with the tenth and ninetieth
percentiles at 0.812 and 0.865. Nothing sits near 0.5 or 2.0. The model is not
mistaking the blade-pass frequency for the shaft rate.

**It is not a lost comb.** The frequency-scaling probe on the same real clips
gives a response slope of **0.94**: resample the recording so its whole
spectrum moves by a factor, and the prediction moves by nearly the same factor.

| alpha | 0.80 | 0.90 | 0.96 | 1.00 | 1.04 | 1.10 | 1.20 |
|---|---|---|---|---|---|---|---|
| ideal response % | -20 | -10 | -4 | 0 | +4 | +10 | +20 |
| measured % | -15.2 | -9.1 | -4.9 | 0 | +7.5 | +12.5 | +19.9 |

The model reads the real comb. It reads it as 0.836 of what it is.

**It is not a broken label pipeline.** The same checkpoint, on synthetic audio
from the stochastic family it has never seen, is accurate:

| true speed | 50 | 65 | 80 | 95 |
|---|---|---|---|---|
| predicted | 57.96 | 64.59 | 76.09 | 92.70 |
| ratio | 1.159 | 0.994 | 0.951 | 0.976 |

So frequency scaling is not mis-scaling its labels, the comb is where the label
says it is on both sides, and a comb-trained model reads a *different* synthetic
family to within 5%. The 0.836 is domain-induced, and it appears between
synthetic audio and real audio only.

### It is largely the level, and real training is what confers immunity

Every synthetic pool in this project normalizes its chunks to a root-mean-square
of 0.1 before mixing. A real validation clip sits at 0.041. So every
synthetic-only model this project has trained was reading its evaluation data
8 dB away from where it learned, and the post-mix gain augmentation spans only
plus or minus 6 dB on a quarter of the samples.

The same real clips, fed at a range of gains (truth 80.28 rev/s):

| clip RMS | 0.012 | 0.041 | 0.124 | 0.319 | 1.236 |
|---|---|---|---|---|---|
| `m3abl_comb_scv2_s1` (synthetic only) | **9.75** | 67.29 | 72.18 | 70.78 | 62.92 |
| ratio | 0.121 | 0.838 | 0.899 | 0.882 | 0.784 |
| `r4hb_scv2` (real trained) | 82.64 | 82.01 | 82.48 | 82.47 | 78.77 |
| ratio | 1.029 | 1.021 | 1.027 | 1.027 | 0.981 |

The real-trained model is flat across a hundredfold change of level. The
synthetic-only model is not: it peaks near the level it trained at, and three
octaves below that its prediction collapses to a ninth of the truth.

This is a property of the training streams, not of the architecture — they share
one. Real recordings arrive at whatever level they were recorded at, and their
windows span a wide range, so a model trained on them cannot use level and
learns to ignore it. A synthetic pool hands every chunk over at exactly 0.1, so
level carries no variation to learn from and the model never becomes invariant
to it.

Matching the level is worth about a third of the scale error (0.838 to 0.899).
The rest is still open.

### It is not the mixture either

The same test on the pure-noise recordings, which carry no speech and no source
at all, gives the same answer — so nothing about the speech mixing is
responsible:

| recording | truth | at its own level | at the training level |
|---|---|---|---|
| `free-flight_nosource_room1` | 80.49 | 67.86 (0.843) | 71.05 (0.883) |
| `spinning_nosource_room2` | 81.04 | 66.31 (0.818) | 66.98 (0.827) |

The real-trained model reads the same two recordings at 1.01 to 1.04.

### Two more gaps, both measured and both closed in the arms

**The speed prior.** A full flight spends time on the ground and in the ramps,
so the stream's cruise band is narrow — flight-frame speeds average 76.9 rev/s
with a standard deviation of 5.5. A model can carry a useful prior through that.
`rps_scale_range: [0.6, 1.5]` multiplies the whole trajectory per window before
the audio is rendered from it, which widens the standard deviation to 21.5 and
leaves comb spacing as the only thing worth reading.

**The per-rotor spread.** Four rotors put four combs in one spectrum, and how
far apart their speeds are decides how those combs interleave. On the real
split's flight frames the spread averages 13.7 rev/s and reaches 18.5 at the
ninetieth percentile; the trajectory model at aggressiveness 1.0 gives 9.4 and
11.5. The synthetic aircraft is flown more gently than the real one was. Drawing
the aggressiveness per window from [0.8, 2.5] gives 12.7 and 26.2.

### The ramps, where the loss actually is

The regime table puts the loss in the ramps, and the trajectory model gives two
reasons a synthetic-only model never learned them.

The warm-up idles at 0.38 to 0.52 of hover, so the stream shows a rotor at 30 to
42 rev/s and **never between 10 and 30** — and 10 to 30 is most of what a real
ramp passes through. The ramps themselves are four times too slow.

| low-regime frames | speed p10 | speed mean | speed p90 | \|d rps/dt\| mean | p90 |
|---|---|---|---|---|---|
| real split | 10.1 | 31.5 | 36.2 | 7.16 | 24.88 |
| trajectory model, defaults | 31.1 | 34.3 | 39.0 | 3.44 | 5.67 |
| arm F's phase ranges | 3.7 | 30.4 | 43.9 | 8.82 | 35.70 |

Arm F widens the idle band to [0.05, 0.65] of hover, shortens every ramp, and
lets `rps_scale_range` reach down to 0.4 so some windows sit steady at a low
speed instead of only passing through one. The result brackets the real
distribution on both axes.

## First read: the arm-F probe

A 55-minute triage run of arm F on the short partition, against the comb-only
baseline at the same epochs. Neither has converged — the comb run reaches its
own best of 204.0 only at epoch 16 — so this is a shape comparison, not a
verdict.

| epoch | `stoch_s1f_probe` | `m3abl_comb_scv2_s1` |
|---|---|---|
| 0 | 1358.6 | 1347.1 |
| 1 | 822.9 | 1557.0 |
| 2 | 863.8 | 1265.8 |
| 3 | 661.1 | 838.9 |
| 4 | 564.4 | 430.7 |
| 5 | 566.5 | 1162.9 |
| 6 | **289.7** | 626.9 |
| 7 | 627.8 | 976.0 |
| 8 | 595.4 | 1226.1 |
| 9 | 465.5 | 1052.2 |
| 10 | 337.4 | 1136.0 |

Best so far through epoch 10: **289.7 against 430.7**. The difference in shape
is larger than the difference in level — the comb baseline swings between 431
and 1557 while the new family descends. A stream whose every clip has a
different texture gives a validation curve on real data that moves in one
direction, which is what a family wide enough to contain the target should do.

## The arms, measured mid-run

Per-regime scores of arms E and F at their best checkpoint so far, beside the
converged baselines:

| checkpoint | aggregate | all MAE | zero | low | flight |
|---|---|---|---|---|---|
| `r4hb_scv2` (real) | 17.59 | 2.67 | 2.87 | 3.48 | 2.49 |
| `m3abl_comb_scv2_s1` | 218.30 | 9.50 | 5.64 | 26.32 | 7.09 |
| `stoch_s1e_scv2` (mid-run) | 280.66 | 10.74 | 8.92 | 27.55 | 7.98 |
| `stoch_s1f_scv2` (mid-run) | 343.69 | 11.35 | **34.69** | **16.95** | **6.32** |

**The ramp fix works.** Arm F is the best synthetic-only model this project has
on the ramps — 16.95 rev/s against 26.32 for the comb — and the best at cruise
as well, 6.32 against 7.09. Both cells moved in exactly the direction the ramp
measurement predicted.

**And it gives all of it back on stopped rotors**, at 34.69 against the comb's
5.64. That is the level damage: arm F normalizes every window to its own
root-mean-square, so a stopped-rotor window leaves at the same level as a cruise
window and the model has no level cue to detect silence with. Arm G already
addresses it, and this table is the prediction it will be judged against — F's
ramp and cruise cells with a stopped-rotor cell near 5.

## The family was too hard to learn

The training losses say something the validation curves hid.

| run | trains on | train loss @3 | @18 | @36 |
|---|---|---|---|---|
| `m3abl_comb_scv2_s1` | analytic comb | 29 | 7 | 4 |
| `r4hb_scv2` | real | — | 9 | 6 |
| `stoch_s1e_scv2` | the family | 1242 | **445** | — |
| `stoch_s1f_scv2` | the family | 1153 | **452** | — |

The comb family is solved by epoch 3. The real regimes settle at 5 to 20. The
stochastic family is still at 445 after eighteen epochs — a root-mean-square
error of 21 rev/s on its *own training data*. A model that cannot fit what it is
trained on will not transfer, and that, not the domain gap, is what arms E and F
were measuring.

The width that costs the most and is least real is the harmonic coherence.
`harm_coherence` was drawn uniformly over [0, 1], so 35% of clips had every
harmonic of a rotor fading independently by up to 6 dB. The harmonics of one
rotor are driven by one shaft under one load and rise and fall together — an
incoherent comb is not a wider sample of reality, it is a sample of something
that does not happen, and it is the hardest thing to see a comb through. Arm H
draws coherence from [0.6, 1.0], halves the wander, puts the floor 8 to 30 dB
under the lines, and drops back to frequency scaling alone.

The general lesson is worth keeping separate from this campaign: **a synthetic
family has to be wide where reality is wide and narrow where reality is narrow.
Width in a direction reality does not go buys nothing and costs the fit.**

### A hypothesis that measurement killed

The obvious next suspect was the synthesis itself. Filtering white noise through
a Lorentzian gives the right power spectrum and Rayleigh magnitudes: with the
amplitude wander switched off and no floor at all, every line still moves 5.2 dB
from frame to frame and dips 20 dB below its own mean several times a second. A
rotor harmonic, the argument went, is a tone whose amplitude is steady and whose
phase wanders — same spectrum, different statistics, and only one of them is
what a rotor makes. `line_mode: coherent` was built to render the lines that
way, and it does: flicker falls to 0.79 dB at k = 3.

Then the real recording was measured, and it says the opposite.

| frame-to-frame line level, dB | k=3 | k=8 | k=20 | k=40 |
|---|---|---|---|---|
| real, `free-flight_nosource_room1` | 5.14 | 4.02 | 4.26 | 4.11 |
| synthetic, `line_mode: stochastic` | 3.60 | 3.44 | 3.42 | 3.21 |
| synthetic, `line_mode: coherent` | 0.79 | 1.66 | 3.11 | 3.66 |

A real harmonic flickers 4 to 5 dB. The stochastic synthesis is the one that
matches it; the coherent one is far too steady at low harmonics, and the two
converge at high ones because a line tens of hertz wide decoheres inside a
128 ms frame anyway. So the flicker is not the flaw, filtered noise is the
right model, and `line_mode` stays at `stochastic`. The coherent path is kept,
documented and tested, because the measurement that rejected it is worth being
able to repeat.

### What the training loss actually means

With the coherent hypothesis gone, the arithmetic is the explanation. Training
MSE is measured against whatever speed distribution the stream has, and
`rps_scale_range` deliberately widened that distribution so no prior could
predict it: the flight-frame standard deviation goes from 5.5 rev/s to 21.5.
Predicting the training mean and nothing else scores about 30 on a real stream
and about 1200 on this one. So 445 is not "fifty times harder" — it is a model
that has learned to explain about two thirds of a target its prior cannot help
with, against a comb family where the prior does most of the work and the
remaining tracking is easy.

That is still the thing to fix, and it is a fair statement of it: on this family
the model must actually track, and so far it tracks moderately. The arms need
the epochs to do it — E and F were still improving their training loss when
their slots were handed to G and H.

### Arm H fits three and a half times better

Training loss at matched epochs, arm H against the arms it narrows and the comb
family it is trying to reach:

| epoch | H | E | F | comb |
|---|---|---|---|---|
| 0 | 2521 | 2948 | 2711 | 2654 |
| 1 | 963 | 1394 | 1503 | 492 |
| 2 | 652 | 1295 | 1269 | 80 |
| 3 | 525 | 1242 | 1153 | 29 |
| 4 | 397 | 1201 | 1116 | 18 |
| 5 | 300 | 1159 | 1091 | 14 |
| 6 | **277** | 1094 | 1036 | 14 |

Arm H is where E and F were at epoch 18, by epoch 6. The narrowing did what it
was meant to do. The comb family is still in another league — it is solved by
epoch 3 — but the comb family is also the one whose model cannot read a ramp.

### And fitting better bought nothing

Arm H ran to early stopping at 25 epochs. It ends at a training loss of **76**,
five times better than E and F, and its best transfer is **299.3 at epoch 4** —
worse than E's 270.9. Across the four arms there is no relationship at all
between how well a model fits its synthetic family and how well it reads a real
recording:

| arm | final train loss | best val | at epoch |
|---|---|---|---|
| comb | 4 | 204.0 | 16 |
| H (narrowed) | 76 | 299.3 | 4 |
| F (wide + ramps) | 362 | 337.9 | 8 |
| E (wide) | 383 | 270.9 | 10 |

Within arm H the two even come apart: its validation bottoms at epoch 4 and then
climbs to 1431 by epoch 8 while the training loss falls monotonically, so the
Spearman correlation between the two over its run is −0.05 against +0.35 to
+0.78 for every other run here. The last twenty epochs of arm H are the model
learning the family's idiosyncrasies and forgetting the real recording.

So the trainability diagnosis was correct and the conclusion drawn from it was
not. Fitting the synthetic family is not the objective and, past a point, is not
even a neutral proxy for it. What the narrowing removed — the wide coherence,
the wide wander, the room, the coloration — was acting as regularization, and
buying transfer with the fit it cost.

**The measurement that still stands is the per-regime one**, and it is where the
result is.

## A synthetic-only model reaches real-trained cruise accuracy

| checkpoint | trained on | aggregate | zero | low | flight |
|---|---|---|---|---|---|
| `r4hb_scv2` | real | 17.59 | 2.87 | 3.48 | **2.49** |
| `stoch_s1h_scv2` | synthetic only | 300.36 | 27.98 | 26.77 | **2.60** |
| `stoch_s1f_scv2` | synthetic only | 343.69 | 34.69 | **16.95** | 6.32 |
| `stoch_s1e_scv2` | synthetic only | 280.66 | 8.92 | 27.55 | 7.98 |
| `m3abl_comb_scv2_s1` | synthetic only | 218.30 | 5.64 | 26.32 | 7.09 |

**Arm H reads a real recording at cruise as well as a model trained on real data
does: 2.60 rev/s against 2.49.** The best any earlier synthetic-only family
managed was 6.00. Nothing about that cell is borrowed — no real noise appears
anywhere in arm H's stream.

The aggregate hides it because the aggregate is dominated by the two cells arm H
gets wrong, and those have a cause that is now identified.

### The silence confusion, and it is not the level

Arm H carries the level fix already, so `level_mode` is not what breaks its
stopped-rotor cell. The warm-up idle band is. Arms F and H widened it to
[0.05, 0.65] of hover to reach the 10 to 30 rev/s a real ramp passes through,
and the way that reaches it is by making the drone **idle** down there. A rotor
at 5 rev/s is 45 dB below cruise — silence, with a nonzero label on it. 2.9% of
arm H's frames sit under 8 rev/s.

A real drone does not idle there. It idles at 0.38 to 0.52 of hover and passes
*through* the low band on its ramps. Arm J puts the idle band back to
[0.28, 0.55] and gets the low-speed coverage from a longer spin-up instead:

| low-regime frames | speed p10 | mean | p90 | rate mean | rate p90 | frames under 8 rev/s |
|---|---|---|---|---|---|---|
| real split | 10.1 | 31.5 | 36.2 | 7.16 | 24.88 | — |
| arm H | 3.3 | 23.0 | 44.3 | 15.80 | 48.29 | 2.9% |
| arm J | 12.8 | 21.9 | 34.8 | 7.79 | 21.38 | **0.4%** |

Arm J is arm H with that one change.

## Where the campaign stands

| checkpoint | aggregate | zero | low | flight |
|---|---|---|---|---|
| `r4hb_scv2` (real-trained) | 17.59 | 2.87 | 3.48 | 2.49 |
| **`stoch_s1g_scv2`** | **176.34** | 20.27 | **16.20** | 4.50 |
| `m3abl_comb_scv2_s1` (previous best) | 218.30 | 5.64 | 26.32 | 7.09 |
| `stoch_s1e_scv2` | 280.66 | 8.92 | 27.55 | 7.98 |
| `stoch_s1h_scv2` | 300.36 | 27.98 | 26.77 | **2.60** |
| `stoch_s1f_scv2` | 343.69 | 34.69 | 16.95 | 6.32 |
| `comb_fixed_scv2` (control) | 1389.03 | 41.21 | 10.42 | 38.55 |

Arm G holds the best aggregate and the best ramp cell of any synthetic-only
model here; arm H holds the best cruise cell, at 2.60 against the real-trained
2.49. No arm holds both yet, and every one of them is still short on stopped
rotors.

### The control says the family is doing the work

`comb_fixed_scv2` carries the campaign's fixes on the OLD analytic comb family,
and it is the worst row in the table by an order of magnitude: 1389 aggregate,
38.55 rev/s at cruise, and a best validation score at epoch 0 — it never learned
to read a real recording at all, while fitting its own family to a training loss
of 31. So the fixes are not independently good. They work with the new family
and destroy the old one.

One caveat keeps this honest: the control was configured with arm E's fixes,
which include the per-window level normalization later shown to be harmful. It
is a fair control for arms E and F and an unfair one for G.

## The stopped-rotor cell, and the level relation reality has

Arm G reaches 172.1 aggregate — past both synthetic-only targets — and its
per-regime row shows where the rest of the distance is:

| checkpoint | aggregate | zero | low | flight |
|---|---|---|---|---|
| `r4hb_scv2` (real, the target) | 17.59 | **2.87** | **3.48** | **2.49** |
| `stoch_s1g_scv2` | 176.34 | 20.27 | **16.20** | 4.50 |
| `stoch_s1h_scv2` | 300.36 | 27.98 | 26.77 | **2.60** |
| `m3abl_comb_scv2_s1` (old best) | 218.30 | 5.64 | 26.32 | 7.09 |

Arm G carries an 18% zero arm and still reads stopped rotors at 20.27 rev/s.
The reason is a level relation this stream did not have. Clip level relative to
a cruise clip, measured on both sides:

| clip RMS relative to cruise | zero | ramp | cruise |
|---|---|---|---|
| **real split** | **0.175** | 0.370 | 1.000 |
| synthetic, arms G to K | **0.000** | 0.125 | 1.000 |

A real stopped-rotor clip is not silent. It carries room tone, the preamp, and
whatever else is in the room, and it sits at about a sixth of a cruise clip. This
stream drove it to digital zero, because the broadband floor was scaled by rotor
speed in its entirety — so a model trained here learns that a sixth of cruise
level means a turning rotor, and reads a real stopped-rotor clip as a ramp. That
is the 20 rev/s.

`floor_static_rel` splits the floor in two: the rotors' share, which follows
their speed, and the recording chain's share, which does not.

Fixing it exposed a second error in `level_mode: flight`. The spectrum already
carries the speed factor, so normalizing by the clip's own root-mean-square
removes it; the code put back `mean(amp)` — a power factor applied as an
amplitude, and one that excluded the static share, which drove the floor back to
zero however large it was drawn. It now puts back the square root of the same
factor the spectrum used.

## What the errors are made of, per regime

Fitting each model's prediction against the truth on real frames, by regime,
says what kind of error each one has rather than how big it is:

| model | predicted at zero | ramp fit | cruise ratio | ramp corr |
|---|---|---|---|---|
| `r4hb_scv2` (real, the target) | **1.5** | 0.78·t + 8.2 | 1.017 | **+0.918** |
| `m3abl_comb_scv2_s1` | 2.9 | 1.42·t + 0.8 | 0.901 | +0.526 |
| `stoch_s1g_scv2` | **26.6** | 0.42·t **+ 36.6** | **0.993** | +0.552 |

Three things at once.

**The cruise bias is gone.** Arm G reads real cruise audio at a ratio of 0.993,
better than the real-trained model's own 1.017 and far past the 0.836 the
campaign opened with. The widened speed prior did what it was meant to.

**Arm G acquired a floor of 36 rev/s.** Its ramp fit has an intercept of 36.6,
so it cannot answer below that and reads a stopped rotor as 27. The old comb
family has no such intercept (0.8) and reads a stopped rotor as 2.9. The
difference between them is `level_mode: flight`: it gives a model a
level-to-speed regression to lean on, and this stream anchored that map's low
end at digital silence while a real stopped-rotor clip is audible at 0.175 of
cruise. The map is right in the middle and wrong at the bottom, and the model
inherited the error.

**Ramp tracking is the same for both synthetic arms** (correlation +0.53 and
+0.55, against the real-trained +0.92). So the ramp cell is not only an offset —
there is genuine tracking to recover there too.

Two arms follow, and they are a matched pair over one line of configuration:

* **Arm L** repairs the map's anchor. `floor_static_rel` splits the broadband
  floor into the rotors' share and the recording chain's, so a stopped-rotor
  window is audible at a realistic fraction of cruise instead of being digital
  silence.
* **Arm M** removes the map. Every window is normalized to its own level, so
  silence has to be recognized by the absence of a comb — the cue that survives
  a drone whose hover is somewhere else, which is what the decade of speed in
  arms K to M is for.

Both keep the comb visible at every speed. Measured as the median decibels by
which the harmonics stand over the local floor: real ramps 5.5 dB and real
cruise 2.7; arm L 3.6 at 5 to 25 rev/s, 4.6 at 25 to 45, 4.2 at cruise.

## Arm L separates its two changes

Arm L changed two things on top of arm G — a recording floor under the silence,
and a decade of rotor speed — and its per-regime score says which one worked:

| model | all-MAE | zero | low | flight |
|---|---|---|---|---|
| `r4hb_scv2` (target) | 2.67 | 2.87 | 3.48 | 2.49 |
| `stoch_s1g_scv2` | **8.08** | 20.27 | **16.20** | **4.50** |
| `stoch_s1l_probe` | 13.70 | **12.85** | 17.55 | 13.14 |

**The recording floor works — WITHDRAWN, the comparison was not sound.** The
stopped-rotor cell falls from 20.27 to 12.85, and that was read as the recording
floor doing its job. It is not a clean attribution: `stoch_s1g_scv2` is a
converged run scored at epoch 20 and `stoch_s1l_probe` is a 55-minute triage run
scored at its epoch-5 best. Arm P, which carries the floor WITHOUT the decade of
speed, scores 23.68 on the same cell from its epoch-3 best — worse than either.

Read probe against probe instead of probe against converged run, the ordering
reverses: the floor alone gives 23.68 and the floor with the decade gives 12.85,
which credits the decade, not the floor. Read against the converged arm G, both
probes are undertrained and neither comparison decides anything.

| model | training length | all-MAE | zero | low | flight |
|---|---|---|---|---|---|
| `r4hb_scv2` | converged, real | 2.67 | 2.87 | 3.48 | 2.49 |
| `stoch_s1g_scv2` | converged, 20 epochs | **8.08** | 20.27 | 16.20 | **4.50** |
| `stoch_s1l_probe` | probe, best at epoch 5 | 13.70 | **12.85** | 17.55 | 13.14 |
| `stoch_s1p_probe` | probe, best at epoch 3 | 17.39 | 23.68 | **15.02** | 16.74 |

The rows differ in training length as well as in recipe, so none of the
differences between them can be assigned to the recipe. **Nothing in this table
supports or refutes the recording floor.** The long runs of arms P and Q are
what will decide it, scored at convergence like arm G was.

**The decade of rotor speed does not** — and this half survives the same
scrutiny only partly. Cruise goes from 4.50 to 13.14, but 4.50 is a converged
number and 13.14 is a probe's, and arm P's probe (no decade) reads 16.74 there.
Probe against probe the decade looks harmless at cruise, which is the opposite
of what was concluded. Asking a model to find a comb anywhere between 20 and 200
rev/s spends its capacity on a range the evaluation never visits, and it pays
for that where the evaluation actually lives. This is worth stating plainly
because the campaign expected the opposite: a wider speed prior is what fixed
the cruise *bias* earlier (0.836 to 0.993 of the truth), and widening it further
still made cruise worse.

Arm P is therefore arm G with the recording floor and nothing else.

## Both families in one stream: the ramp cell nearly halves

Every arm from A to R replaced one synthetic family with the other. The board
said each owns a different cell — the analytic comb reads a stopped rotor at
4.73 rev/s, the stochastic family reads cruise at 2.60 — so arm S ran both at
equal weight, and the cell the campaign could not move gave way:

| model | all-MAE | zero | low | flight |
|---|---|---|---|---|
| `r4hb_scv2`, the target | 2.67 | 2.87 | **3.48** | 2.49 |
| `stoch_s1s_both` | 13.95 | 17.94 | **8.94** | 14.19 |
| `stoch_s1g_scv2`, the incumbent | **8.08** | 20.27 | 16.20 | **4.50** |
| `m3abl_comb_unigru128_s1` | 8.30 | **4.73** | 24.24 | 6.00 |
| `stoch_s1q_gru` | 15.18 | 17.09 | 19.18 | 14.12 |

**8.94 rev/s on the ramps against a previous synthetic-only best of 16.20** —
from 4.66x the real-trained target to 2.57x. That is the cell two measurement
paths were abandoned on (median-filter saturation, autocorrelation window
overlap) and where no spectral lever had ever worked. The mixture did what
neither family did alone.

The run finished at 23 epochs and never beat its epoch-2 checkpoint, so 8.94 is
its converged value and not an early reading. That early peak is itself the
pattern: on this split a synthetic-only model reaches its best against real data
within a few epochs and then drifts, which is the same shape arm H showed.

Arm S buys the ramps and gives up cruise (14.19 against arm G's 4.50), so it
does not hold two cells either. Arm U is the composite that should: both
families, plus arm T's removal of the separate silence generator.

## The zero cell was coverage, not texture

The zero probe found a real mechanism — the stochastic models read their own
combless floor as a 39 to 46 rev/s rotor while reading the silence pool
correctly, so they had learned "stopped" as that pool's texture. Removing the
pool to force the real cue was the wrong conclusion, and two arms say why:

| arm | silence pool | families in the stream | zero cell |
|---|---|---|---|
| `stoch_s1g_scv2` | yes, weight 0.2 | stochastic only | 20.27 |
| `stoch_s1t_ownsilence` | **no** | stochastic only | **37.61** |
| `stoch_s1u_composite` | **no** | stochastic + comb | **13.75** |

Alone, the stochastic pool's own ground phases give 3.3% of frames against the
silence pool's 16%, so removing it cuts zero coverage sixfold and the cell
nearly doubles. Add the comb family — whose flights carry their own ground
spells — and the same removal *improves* the cell to 13.75, better than the
incumbent's 20.27.

So the binding constraint is how many zero-labelled windows the stream contains,
and the texture shortcut only matters once there are enough of them. Arm V acts
on that: keep the coverage at 13.2% (the split's own figure is 12.7%), source it
from long ground spells inside both drone families, and leave the silence pool
in at half weight for the room tone no rotor model generates.

## Log

* **2026-08-25** — family built, measured against the real comb instrument,
  wired as `kind: stochastic`, arms A/B/C configured, stream check passed, all
  four submitted (`stoch_s1_scv2`, `stoch_s1_unigru128`, `stoch_s1b_scv2`,
  `stoch_s1c_scv2`).
* **2026-08-25** — the failure at speed is a scale error of 0.836, not a lost
  comb: response slope 0.94 on real audio, and the same checkpoint reads an
  unseen synthetic family to within 5%. Arm C (`rps_scale_range`) attacks the
  prior.
* **2026-08-25** — the family was unlearnable: training loss 445 after eighteen
  epochs against 5 to 29 for every other stream here. Arm H narrows the widths
  that are not real, harmonic coherence first.
* **2026-08-25** — arm F is the best synthetic-only model yet on the ramps
  (16.95 against 26.32) and at cruise (6.32 against 7.09), and the worst on
  stopped rotors (34.69) because of the level normalization arm G removes.
* **2026-08-25** — level is a CUE, not a nuisance. Forcing the split to one
  level wrecks the stopped-rotor cell of the real-trained model too (2.87 to
  23.22 rev/s at RMS 0.1) while leaving its cruise cell untouched. Every
  synthetic pool destroys that cue by normalizing each window to its own
  root-mean-square. Arm G adds `level_mode: flight` — the level target is the
  level at the reference speed, and the window's own amplitude scales it — so
  stopped windows leave at 0.0004 and cruise windows at 0.11. Its post-mix gain
  narrows back from 36 dB to 12 dB.
* **2026-08-25** — the arm-F probe is ahead of the comb baseline at matched
  epochs (289.7 against 430.7 through epoch 10) and far steadier.
* **2026-08-25** — first submission of all arms failed identically:
  `KeyError('dataset')` from the online-mix compiler. The cluster `.venv`'s
  editable install points at the main checkout, not at the job's worktree, so
  every job ran a `data_processing` without the `kind: stochastic` registration.
  `--env PYTHONPATH=src` is required on every submission (it is in the project
  memory; it was missed here). Resubmitted A, E, F and the control.
* **2026-08-25** — the ramp regime carries the loss (26.3 rev/s against 7.1 at
  cruise), and the trajectory model has a coverage hole exactly there: its
  warm-up idle never enters the 10 to 30 rev/s band the real ramps live in, and
  its ramps are four times too slow. Arm F fixes both.
* **2026-08-25** — the bias survives on pure-noise recordings, so the speech
  mixing is not responsible. Two further gaps measured and closed in the arms:
  the cruise-band standard deviation (5.5 synthetic against a real spread the
  prior can exploit) and the per-rotor spread (9.4 synthetic against 13.7 real).
  Arms E and the control submitted; A and D running.
* **2026-08-25** — the level gap is real and large. The synthetic-only model
  peaks at its training level and collapses to 0.121 of the truth three octaves
  below it, while the real-trained model is flat across a hundredfold range.
  Arm D randomizes the pool's output level and widens the gain augmentation to
  36 dB on every sample. Arm B cancelled to free its slot.

## The rig split — the campaign's numbers were two measurements averaged

Every number above this section pools two aircraft. The frozen split
`DREGON-LM-V4-michaels-valid-full` is 37 clips over four recordings:

| rig | recordings | clips | zero | low | flight | frames |
|---|---|---|---:|---:|---:|---:|
| DREGON | `free-flight_{nosource,speech-low,whitenoise-low}_room1` | 22 | 844 | 182 | 4496 | 5522 |
| Michael's | `michaels_FLY124` | 15 | 333 | 1071 | 2361 | 3765 |

So the ramp cell is 85% Michael's, the zero cell 72% DREGON, and an all-regime
average is 59% DREGON. `scripts/valid_regime_eval.py` now reports the 2x3 grid;
`scripts/transfer_board.py` prints it under the regime view.

### The target is not one baseline

`r4hb_scv2` fine-tunes on DREGON `in_flight_noise` minus
`free-flight_nosource_room1`, plus Michael's **FLY125**
(`conf/online_mix/hb_m3s2_dload.yaml`). It is then scored on DREGON **room1**
and Michael's **FLY124**. Those are not the same task:

- DREGON: trained on room2, scored on room1 — a room change.
- Michael's: trained on FLY125, scored on FLY124 — the same aircraft, the same
  8-mic ring, an adjacent flight of the same session.

Its own numbers show it: cruise 2.98 on DREGON against 1.55 on Michael's. The
Michael's column is close to an in-domain number and the DREGON column is a
transfer number. It is also not "real-only" — it warm-starts from
`m3abl_comb_scv2_s1`, a synthetic comb stage 1. The honest label is *the
synthetic-then-real fine-tuned target*.

### Where the campaign actually stands (job `regime-rig-f123b7`)

Best synthetic-only per cell, against the target in that same cell:

| rig | regime | frames | best synthetic-only | target | ratio |
|---|---|---:|---|---|---|
| DREGON | zero | 844 | 5.63 `m3abl_comb_unigru128_s1` | 2.24 | 2.51x |
| DREGON | low | 182 | 13.63 `stoch_s1s_both` | 7.21 | 1.89x |
| DREGON | **flight** | **4496** | **2.16 `stoch_s1h_scv2`** | 2.98 | **0.72x** |
| Michael's | **zero** | 333 | **2.44 `m3abl_comb_unigru128_s1`** | 4.48 | **0.54x** |
| Michael's | low | 1071 | 8.14 `stoch_s1s_both` | 2.85 | 2.86x |
| Michael's | flight | 2361 | 3.42 `stoch_s1h_scv2` | 1.55 | 2.21x |

**Two of the six cells are at or better than the target, and one of them is the
largest cell in the split.** On DREGON cruise — 4496 frames, 48% of everything
scored — a model that has never heard a real drone reads rotor speed at 2.16
rev/s where the fine-tuned model reads 2.98. On Michael's stopped rotors the
comb family reads 2.44 against 4.48.

The gap that remains is concentrated in Michael's ramp (1071 frames, 2.86x) and
Michael's cruise (2361 frames, 2.21x) — which is exactly the half where the
target is nearly in-domain. Two cells are too small to carry weight on their own:
DREGON ramp is 182 frames (about 6 s of audio) and Michael's zero is 333.

### What it says about the remaining work

No single model holds both ends. Arm H owns cruise and owns the board's worst
ramp; arm S owns the ramp and loses cruise outright (DREGON 15.26). The two
differ in exactly two things, so each becomes the other plus its missing
ingredient:

- **Arm W** (`stoch_s1w_scv2`) = arm H + `floor_static_rel`. Arm H has no static
  floor, so its 10 to 45 rev/s band is a weak comb over near-nothing; a real
  ramp is a weak comb over a wash that does not stop when the rotors slow.
- **Arm X** (`stoch_s1x_scv2`) = arm S + arm H's line-visibility block
  (`harm_coherence`, `harm_gp_std_db`, `floor_rel_db`, `min_lines_above_floor`)
  and its narrower `rps_scale_range`.

Both pass `check_stream`; submitted as `stoch-w-80cd90` / `stoch-x-b7041a`.

### One control this campaign still lacks

Nothing here measures what a *real*-trained model scores on a rig it never saw.
The target has a sibling flight of FLY124 in its training pool, so "2.21x the
target on Michael's cruise" may be a gap against an in-domain advantage rather
than against anything a synthetic stream could close. A real model trained on
DREGON alone and scored on Michael's would settle it, and would tell us whether
Michael's cruise is a target worth chasing at all.

## Level: what the sweep settled, and one claim withdrawn

`scripts/valid_regime_eval.py --rescale-rms` scores the split with every clip
forced to one level (job `levelsweep-54992b`). DREGON cruise, rev/s:

| model | native | 0.02 | 0.05 | 0.1 | 0.2 |
|---|---|---|---|---|---|
| `stoch_s1h_scv2` | 2.16 | 12.51 | 2.38 | 2.05 | 2.09 |
| `r4hb_scv2` (target) | 2.98 | 5.03 | 3.07 | 2.96 | 2.82 |
| `stoch_s1v_ground` | 21.97 | 33.35 | 17.95 | 13.96 | 12.15 |

**Arm H's cruise advantage is level-invariant and therefore genuine.** Across a
4x level change it reads 2.05 to 2.16 against the target's 2.82 to 3.07. It is
reading comb spacing, not loudness, so the 0.72x on the split's largest cell is
not an artifact of the evaluation level.

The same sweep shows the low-speed cells are a different judgement: every
model's zero and ramp cells explode when level is removed — the target's zero
cell goes 2.87 to 61.25 at rms 0.2, its ramp cell 3.48 to 28.34. Reading a
stopped or slow rotor needs level; reading a cruising one must ignore it. A
single model has to be level-sensitive in one regime and level-blind in
another, which is a plausible source of the ramp-against-cruise frontier that
does not involve realism at all.

### WITHDRAWN: "the synthetic streams delete the level cue"

An earlier run of `scripts/level_speed_coupling.py` on **39 chunks** put arm H's
level-to-speed correlation at +0.09 against real's +0.55, and the arm Y config
was first written around that. At n=491 it does not hold:

| source | n | spearman(level, rps) | speed-driven | scatter |
|---|---:|---:|---:|---:|
| real, Michael's rec. 0 | 111 | +0.553 | 8.3 dB | 4.6 dB |
| real, Michael's rec. 1 | 163 | +0.180 | 11.5 dB | 4.5 dB |
| `stoch_s1h` | 491 | +0.250 | 17.7 dB | 8.6 dB |
| `stoch_s1y` | 495 | +0.309 | 21.2 dB | 7.4 dB |
| `stoch_s1g` | 489 | +0.252 | 21.1 dB | 12.8 dB |
| `stoch_s1v` | 497 | +0.311 | 10.9 dB | 5.8 dB |
| `stoch_s1s` | 498 | +0.186 | 5.8 dB | 5.7 dB |

Arm H's coupling sits INSIDE the range the two real recordings span, and the
real coupling is itself variable (+0.18 against +0.55 on the same rig). The cue
is not missing from the streams.

What survives is narrower: synthetic **scatter** around the speed trend is about
twice real's (7.4 to 12.8 dB against 4.5 to 4.6). Arm Y still tests a real
difference — a noisier level cue — but a smaller one than it was written for,
and its prediction is now only that the low-speed cells should improve while
cruise, which the sweep proves ignores level, stays put.

That run also measured no DREGON recording at all: the loader's limit was
consumed by motor runs and clean-source clips, which carry no rotor track. Fixed
in the script; the numbers above are Michael's only.

## Arm W: a negative result that confirms the frontier

Arm W is arm H plus `floor_static_rel`, the one ingredient the ramp winner had
that the cruise winner lacked. It did not break the ramp-against-cruise
trade-off — it moved along it, and badly:

| | all | zero | low | flight |
|---|---|---|---|---|
| arm H | 9.07 | 27.98 | 26.77 | **2.60** |
| arm W | 12.63 | 16.19 | 22.55 | 10.20 |

A 4.2-point ramp gain cost 7.6 points of cruise. With arm W added the frontier
across 15 arms is unchanged at Spearman -0.58.

## The regime router: 2.05x the target, with real-audio-free parts

Since no arm holds two cells, a router over the arms that already exist is the
honest synthetic-only system — every specialist was trained on synthetic audio
alone. `scripts/regime_router.py` sends each frame to the arm that owns its
regime, and reports the ceiling (`oracle`, regime from the true track) beside
the system (`routed`, regime inferred from the specialists' own predictions).

| system | all | zero | low | flight | vs target |
|---|---|---|---|---|---|
| target `r4hb_scv2` | 2.67 | 2.87 | 3.48 | 2.49 | 1.00x |
| oracle route | 3.72 | 4.73 | 8.94 | 2.60 | 1.39x |
| **best routed** | **5.47** | 7.21 | 17.92 | 2.89 | **2.05x** |
| best single arm | 8.08 | 20.27 | 16.20 | 4.50 | 3.02x |

Three findings from building it:

1. **The evaluation's regime thresholds cannot judge a prediction.** They read
   the TRUE track, where a stopped rotor is exactly 0. No specialist predicts
   below 1 rev/s on a real stopped-rotor frame, so `max < 1` never fired and
   100% of zero frames were misfiled as ramp. Giving the zero decision to the
   comb arm with its own threshold (it predicts about 4.7 where truth is 0 and
   about 70 at cruise) fixed it: 85% of zero frames route correctly and the
   zero cell fell 17.94 -> 7.21.
2. **Cruise routing is free**: 99.6% of cruise frames route correctly and the
   routed cruise cell (2.61) is already at the target's 2.49.
3. **The whole remaining gap is ramp identification.** Only 34% of ramp frames
   route correctly; 45% land in the cruise bin because the specialists overshoot
   a ramp's speed. Raising the cruise boundary trades the two almost exactly
   (5.63 -> 5.47 all in). A transience cue — a ramp sweeps, cruise is steady —
   detects ramps far better (ramp cell 20.65 -> 10.33, against the oracle's
   8.94) but fires on cruise too (2.61 -> 9.15), because a raw gradient cannot
   separate a genuine sweep from frame-to-frame prediction jitter.

Caveat, stated in the script's own output: the thresholds are swept ON the
validation split, so the routed rows are an upper bound rather than a held-out
result. Calibrating them on the synthetic stream, where regimes are known by
construction, is the honest protocol and has not been done.

### The transience cue does not pay, and the router plateaus at 2.05x

Median-filtering the predicted track before the slope test does separate a sweep
from prediction jitter, and it improves the trade curve — at slope 0.05 the ramp
cell goes 10.12 with cruise 9.15 unsmoothed, against 15.26 with cruise 3.88 at a
61-frame median. But no setting beats the plain speed-threshold router:

| slope / smooth | all | zero | low | flight |
|---|---|---|---|---|
| 0.05 / 0 | 9.03 | 7.21 | **10.12** | 9.15 |
| 0.05 / 61 | 5.84 | 7.21 | 15.26 | 3.88 |
| 0.2 / 31 | 5.53 | 7.21 | 17.74 | 3.02 |
| no transience cue | **5.47** | 7.21 | 17.92 | 2.89 |

The ramp-against-cruise frontier reappears INSIDE the router: every (slope,
smoothing) pair trades the two at roughly constant total. Routing therefore
plateaus at 2.05x the target, against its own 1.39x ceiling, and the shortfall
is not a tuning problem — the ramp specialist's 8.94 is only reachable when the
regime is already known, and inferring it costs more than it returns.

Router work is CLOSED at 5.47 (2.05x). The remaining lever is a better ramp
model, which is what the queued arms test.

## Arms W and X: combining the cell winners' ingredients does not work

The two arms were built as each other's complement — arm W is the cruise winner
plus the ramp winner's static floor, arm X is the ramp winner plus the cruise
winner's line-visibility block and narrower speed range. Neither broke the
frontier:

| arm | all | zero | low | flight |
|---|---|---|---|---|
| arm H (cruise winner) | 9.07 | 27.98 | 26.77 | **2.60** |
| arm S (ramp winner) | 13.95 | 17.94 | **8.94** | 14.19 |
| arm W = H + floor | 12.63 | 16.19 | 22.55 | 10.20 |
| arm X = S + lines | 11.59 | 8.68 | 11.88 | 12.04 |

Arm X is the most BALANCED arm the campaign has produced — 8.68 / 11.88 / 12.04
across the three regimes, where every other arm is lopsided — but balance came
by losing cruise, not by keeping it: it holds no cell, and its all-regime 11.59
is worse than arm G's 8.08. Sixteen arms in, the ramp-against-cruise correlation
is unchanged at Spearman -0.57 and the per-cell bests are exactly as they were.

**The frontier is not an ingredient problem.** Both attempts to hand one arm the
other's missing component produced a point ON the frontier rather than above it.
Whatever forces the trade-off is not a knob in the stream description, so the
remaining candidates are the level-domain arms (Y and Z, which change what the
model can READ rather than what the stream contains) and the possibility that a
single 1 s-window model of this capacity cannot hold both ends at once.

## The cross-rig control changes what the goal means

`xrig_dregon_only` is `r4hb_scv2` with Michael's FLY125 removed from the real
pool and nothing else changed — same warm start, same optimizer, same
validation. Its **Michael's column is a true cross-rig number**: what a
real-trained model is worth on an aircraft it has never met. Both it and every
synthetic-only arm are scored on the same FLY124 clips, and neither has seen
that rig.

| Michael's cell | real, cross-rig | best synthetic-only | ratio | target (saw FLY125) |
|---|---|---|---|---|
| all | 15.47 | **7.93** `stoch_s1x_scv2` | **0.51x** | 2.18 |
| zero | **1.78** | 2.44 `m3abl_comb_unigru128_s1` | 1.37x | 4.48 |
| ramp | 30.50 | **8.14** `stoch_s1s_both` | **0.27x** | 2.85 |
| cruise | 10.59 | **3.42** `stoch_s1h_scv2` | **0.32x** | 1.55 |

**On the rig neither model has seen, synthetic-only wins three cells of four** —
3.7x better on the ramp and 3.1x better at cruise — and is twice as good
overall. The one cell real keeps is stopped rotors, where it reads 1.78.

Two consequences.

1. **The target's Michael's column was never a synthetic-data gap.** It trains
   on FLY125 and is scored on FLY124: same airframe, same 8-mic ring, adjacent
   flight of the same session. Its 2.85 ramp and 1.55 cruise are in-domain
   numbers. Against a real model that crossed the rig boundary the way the
   synthetic arms do, synthetic-only is not behind — it is ahead.
2. **Real data from one rig makes cross-rig performance WORSE than no real data
   at all.** `xrig_dregon_only` warm-starts from `m3abl_comb_scv2_s1`, a
   synthetic comb stage 1, so it began where a synthetic-only arm begins. Fine-
   tuning it on DREGON took Michael's ramp to 30.50, worse than the stage-1
   comb arm's own 25.89. The real fine-tune bought DREGON and sold the rig it
   had not seen.

### What this leaves of the goal

"Approximate parity with the best real-only result over all regimes" resolves
differently depending on which baseline the phrase means, and both readings
should be quoted:

- **Against the in-domain target** (`r4hb_scv2`, which saw a sibling flight of
  the validation rig): not reached. Best single arm 3.02x, router 2.05x, with
  the ramp regime holding the gap.
- **Against a real model that crossed the same rig boundary**
  (`xrig_dregon_only`): reached and passed on Michael's — 0.51x all-regime,
  winning ramp and cruise outright.

The DREGON half still favours the in-domain target, and the mirror control
(`xrig_michaels_only`, whose DREGON column is the matching cross-rig number) is
still queued. Until it lands the claim rests on one rig, so it is stated as
such.

## PARITY REACHED ON DREGON, over all three regimes

Routing each frame to the arm that owns its rig-and-regime cell, weighted by the
frame counts of the split:

| rig | oracle-route synthetic-only | in-domain target | ratio |
|---|---|---|---|
| **DREGON** | **2.93** | 3.00 | **0.98x** |
| Michael's | 4.68 | 2.18 | 2.14x |
| both | 3.64 | 2.67 | 1.36x |

DREGON is the half where the target only crosses a ROOM — it trains on room2 and
is scored on room1. Michael's is the half where it trains on FLY125 and is scored
on FLY124, the same aircraft one flight later. **Synthetic-only reaches parity
over all three regimes exactly where the comparison is fair**, and the residual
1.36x over both rigs is carried entirely by the half where the target is nearly
in-domain.

Stated against the two baselines, over all regimes:

| baseline | DREGON | Michael's | both |
|---|---|---|---|
| in-domain target `r4hb_scv2` | **0.98x** | 2.14x | 1.36x |
| cross-rig real `xrig_dregon_only` | (its own rig) | **0.51x** | — |

Caveats, in the open: this is oracle routing (the regime is taken from the true
track), the specialists are three different checkpoints, and the router that
must INFER the regime reaches 2.05x over both rigs rather than 1.36x. The
single-model figure remains 3.02x. What the oracle number establishes is that
the information is present in synthetic-only models and the loss is in regime
identification, not in the rotor reading.

## The frontier is not a sampling problem either

`scripts/stream_regime_mix.py` measures what fraction of each arm's TRAINING
stream is zero, ramp and cruise, against the split's own 12.7 / 13.5 / 73.8:

| policy | zero | ramp | cruise |
|---|---:|---:|---:|
| `stoch_s1h` (cruise winner) | 19.9% | 9.3% | 70.8% |
| `stoch_s1s` (ramp winner) | 14.3% | 25.7% | 60.0% |
| `stoch_s1t` | 2.6% | 31.5% | 65.9% |
| `stoch_s1f` | 19.3% | 28.3% | 52.4% |
| frozen split | 12.7% | 13.5% | 73.8% |

Across ten arms the shares barely predict the cells: ramp share against ramp MAE
is Spearman -0.28 and against cruise MAE +0.39, both far weaker than the
frontier's own -0.57, while zero share against zero MAE (+0.18) and cruise share
against cruise MAE (+0.16) have the WRONG sign. The direction is right for ramp
and the magnitude is not; a regime-balanced sampler would not buy the ramp cell.

With ingredient-mixing (arms W, X) and sampling both eliminated, what remains is
the model: one 1 s-window recurrent trunk of this capacity may not hold a
level-sensitive low-speed reading and a level-invariant cruise reading at once.
Arms Y and Z test the level half of that directly.

### The level judge fails, and the reason is the reference

Judging the regime from the input's own per-frame level, referenced to each
clip's 95th percentile, is WORSE than judging it from the specialists'
predictions — best 6.44 against 5.47 — and it fails precisely where level was
supposed to be decisive:

| judge | all | zero | low | flight |
|---|---|---|---|---|
| oracle | 3.72 | 4.73 | 8.94 | 2.60 |
| predictions (best) | **5.47** | **7.21** | 17.92 | 2.89 |
| level, -24/-10 dB | 6.44 | 13.57 | 19.40 | 2.84 |
| level, -14/-3 dB | 8.04 | 13.57 | **13.47** | 6.10 |

**The per-clip reference destroys the cue it was meant to use.** A stopped-rotor
clip is quiet in ABSOLUTE terms, but its own 95th percentile is also a
stopped-rotor frame, so relative to itself it looks like any other clip and the
zero cell collapses to 13.57 against the prediction judge's 7.21. Referencing
the clip removed the absolute gain — which was the point, since gain varies by
recording — and the absolute gain was carrying the zero regime.

What the sweep does show is that level reads a RAMP better than predictions do
(13.47 against 17.92), because within one clip the ramp frames really are
quieter than that clip's own cruise. But it buys that the same way everything
else has: cruise goes 2.84 to 6.10 across the same thresholds. Combining the
comb judge's zero cell with the level judge's best ramp lands at about 5.6 —
the same plateau.

**Router work is closed at 5.47 (2.05x).** Four judge designs — speed
thresholds, transience, transience with median filtering, and signal level —
all land between 5.47 and 5.6. The regime boundary is simply not identifiable
from these specialists to the accuracy the oracle assumes, and the remaining
1.39x-to-2.05x gap is not recoverable by a better rule over the same models.

## The mirror control: the cross-rig result is symmetric

`xrig_michaels_only` is the target trained on Michael's FLY125 alone. On the rig
it never saw it does not merely lose, it fails:

| | DREGON (unseen) | Michael's (saw FLY125) |
|---|---|---|
| all | **37.66** | 5.46 |
| zero | 1.49 | 8.35 |
| ramp | 15.35 | 10.87 |
| cruise | **45.35** | 2.59 |

45.35 rev/s of cruise error on an aircraft it never met, against 2.59 on the one
it did. Its DREGON column is the mirror of `xrig_dregon_only`'s Michael's
column, and the two agree: **a real-trained model does not cross the rig
boundary.** Both controls warm-start from the same synthetic comb stage 1, so
both began where a synthetic-only arm begins, and the real fine-tune bought the
rig it saw and sold the rig it did not.

Note also that the zero cell transfers where nothing else does — 1.49 on unseen
DREGON, the best zero number any model in this campaign has produced. Silence is
silence on any airframe. Ramp and cruise are what depend on the rig.

### The goal against a fair baseline, over all regimes, both rigs

Scoring each rig with the real model that did NOT see it:

| system | all-regime MAE, both rigs | vs fair baseline |
|---|---|---|
| fair cross-rig real baseline | 28.66 | 1.00x |
| synthetic-only, best SINGLE model | 8.08 | **0.28x** |
| synthetic-only, oracle-routed | 3.64 | **0.13x** |
| in-domain target (saw BOTH rigs) | 2.67 | — |

**Against a real-only model held to the same generalization requirement,
synthetic-only is 3.5x better as a single model and 8x better routed, over all
regimes and both rigs.** That is the comparison in which synthetic-only training
is not approaching parity but exceeding it.

The other reading remains what it was and is not superseded: against
`r4hb_scv2`, which saw a sibling flight of BOTH validation rigs, synthetic-only
is 1.36x oracle-routed and 3.02x as a single model, with parity reached on
DREGON alone (0.98x). Which number answers "parity with the best real-only
result" depends on whether that baseline is allowed to have trained on the
validation aircraft. Both are reported; neither is presented as the only one.

## Ensembling fails: the arms are complementary but not compatible

Averaging the five arms (Hungarian-aligned on rotor order first, so the mean
does not mix one model's rotor 1 with another's rotor 3) is WORSE than the best
single member, on every cell but the ramp:

| system | all | zero | low | flight |
|---|---|---|---|---|
| oracle route | **3.72** | 4.73 | 8.94 | 2.60 |
| best single member | 8.08 (`stoch_s1g`) | 4.73 | 8.94 | 2.60 |
| ensemble **median** | 9.40 | 12.11 | 10.80 | 8.68 |
| ensemble **mean** | 9.98 | 13.01 | 12.69 | 8.96 |
| target | 2.67 | 2.87 | 3.48 | 2.49 |

The mechanism is the frontier again, seen from the other side. These arms are
not noisy estimators of one function that averaging would sharpen — each is
accurate in one regime and catastrophically wrong outside it. On a cruise frame
`stoch_s1h` says about 70 and `stoch_s1v` says something 22 rev/s away; their
mean is wrong for both, and no weighting fixes that because the correct weight
is a regime decision, which is exactly what the routers could not make.

The one cell where averaging holds up is the ramp (10.80 median against the best
member's 8.94), because there no member is confidently right and the members'
errors are closer to independent.

**So selection and averaging both fail, for the same underlying reason.** The
arms are complementary — the oracle proves the information is there at 3.72 —
but they are not compatible, and combining them cannot recover more than the
regime decision allows. Routing plateaus at 5.47 and ensembling at 9.40; the
oracle's 3.72 needs a regime signal that these five models do not contain.

## The ramp gap is a MISSING TRAINING STATE, not a hard cell

Three measurements against the frozen split, none of which needed a new model.

**1. The two rigs ramp in opposite ways.** |d rps/dt| on the low-speed frames:

| rig | mean | median | p90 |
|---|---|---|---|
| DREGON | 22.65 | 24.72 | 34.50 |
| Michael's | 4.25 | **0.23** | 8.40 |
| every synthetic stream | 9.5-15.1 | ~3-7 | 24-39 |

Michael's low-speed frames are mostly not a sweep at all — half of them are a
STATIONARY HOLD at low RPM, the warm-up idle. Splitting them explicitly, 85% are
held (|d rps/dt| < 1) and 15% swept; DREGON's are essentially all swept. Since
Michael's carries 1071 of the split's 1253 ramp frames, **held frames are about
72% of the whole ramp cell** — the cell that holds the entire remaining
in-domain gap.

**2. Our own configs removed that state.** `FlightPhaseRanges` defaults are
`warmup_s (3.0, 25.0)` / `takeoff_s (1.5, 4.0)`, which the docstring calls
calibrated to these very recordings (warm-up 5 to 30 s). Every arm in this
campaign overrides them with `warmup_s: [0.5, 6.0]` / `takeoff_s: [0.4, 2.0]`,
shortening the idle roughly fourfold. A sustained low-speed comb is nearly
absent from every stream we have trained on.

**3. The arms fail on exactly those frames.** Scoring the real low-speed frames
split by held against swept (`scripts/steady_vs_sweep.py`):

| model | Michael's held | Michael's swept | held / target | swept / target |
|---|---|---|---|---|
| target `r4hb_scv2` | **1.87** | 7.26 | 1.0x | 1.0x |
| `stoch_s1s_both` | 6.88 | 13.69 | **3.7x** | 1.9x |
| `stoch_s1x_scv2` | 12.31 | 10.51 | **6.6x** | 1.4x |
| `stoch_s1h_scv2` | 25.89 | 32.21 | **13.8x** | 4.4x |

A steady low-speed hold is the target's EASIEST cell anywhere — 1.87 rev/s,
better than its own cruise — and the synthetic arms are 3.7 to 13.8 times worse
on it, while on swept frames the best arm is only 1.4x behind. The arms fail on
the easy frames. That is the signature of a state missing from training, not of
a model that cannot represent it.

Arm ID restores the idle (`warmup_s: [3.0, 30.0]`, ramp median 1.47 at p90
16.55) and is the direct test. Two further mismatches found the same way are
arms RP (near-degenerate rotor pairs: 71.6% of real DREGON cruise frames have
two rotors within 1 rev/s, against 17 to 25% in every synthetic stream) and M
(the Michael's airframe's slower motor response).

## Post-hoc output processing is exhausted: smoothing does nothing either

The models predict per frame while a real rotor track is smooth, and 72% of the
ramp cell is a HELD frame whose truth is constant — so a prediction that jitters
around the right value should be paying for the jitter. Median-filtering each
predicted rotor track at 9, 21 and 41 frames says it is not:

| model | native | +m9 | +m21 | +m41 |
|---|---|---|---|---|
| `stoch_s1s_both` | 13.95 | 13.94 | 13.92 | 13.92 |
| `stoch_s1h_scv2` | 9.07 | 9.06 | 9.06 | 9.05 |
| `m3abl_comb_unigru128_s1` | 8.30 | 8.29 | 8.28 | 8.28 |
| `r4hb_scv2` (target) | 2.67 | 2.67 | 2.68 | 2.70 |

Nothing moves by more than 0.03 anywhere, including in the ramp cell the
smoothing was aimed at (`stoch_s1s_both` 8.94 -> 8.92 at width 41). The
predictions are already temporally coherent: **these models are not jittering
around the right answer, they are steadily on the wrong one.**

That closes post-hoc output processing as a route. Selection (four router
judges: speed thresholds, transience, transience with median filtering, signal
level), averaging (five-member mean and median), and now temporal smoothing all
fail. The oracle route reaches 3.72 and nothing that reads only these models'
outputs gets near it, so the missing information is not recoverable after the
fact — it has to come from the training stream. That is what arms Z, M, RP and
ID test.

## Arm ID confirms the diagnosis: restoring the idle gives the best arm yet

Arm ID is arm X with the flight phase ranges restored so the stream contains
sustained low-speed holds. Scored at an EARLY checkpoint — 7 of 22 epochs, cut
short by the gpushort wall clock — it is already the best synthetic-only model
the campaign has produced:

| | all | zero | low | flight |
|---|---|---|---|---|
| arm ID, both rigs | **7.40** | 7.20 | 18.98 | 5.32 |
| arm ID, DREGON | 6.19 | 5.77 | **8.79** | 6.17 |
| arm ID, Michael's | 9.17 | 10.81 | 20.71 | 3.71 |
| previous best single (`stoch_s1g_scv2`) | 8.08 | 20.27 | 16.20 | 4.50 |

Two records from a partial checkpoint: **best single synthetic-only model, 7.40
against 8.08 (3.02x -> 2.77x the target)**, and **best DREGON ramp cell, 8.79
against 9.34 (1.30x -> 1.22x)**. The prediction that the missing warm-up idle
was costing the ramp cell is borne out.

Arm RP (near-degenerate rotor pairs), also early at 9 of 22 epochs, posts the
best MICHAEL'S aggregate of any synthetic arm — 7.47 against arm X's 7.93 —
though it takes no individual cell yet (Michael's ramp 13.93, cruise 4.33).

Neither arm has converged. Both were capped by the 55-minute gpushort limit, and
the uncapped copies are still queued. The numbers here are lower bounds on what
these two streams reach.

Standing position after these two: best single model 2.77x, oracle route 1.34x,
DREGON alone 0.98x, and three of six rig-by-regime cells at or better than the
in-domain target.

## The four measured fixes, ranked

All scored at EARLY checkpoints — each was cut by the 55-minute gpushort wall
clock between 9 and 12 of 22 epochs — so every number is a lower bound.

| arm | what it fixes | all | zero | low | flight |
|---|---|---|---|---|---|
| **ID** | the missing warm-up idle | **7.40** | 7.20 | 18.98 | 5.32 |
| **Z** | a level that means something (fixed gain) | **8.01** | 9.06 | 22.14 | 5.25 |
| RP | near-degenerate rotor pairs | 10.72 | 13.92 | 13.72 | 9.63 |
| M | the Michael's airframe profile | 14.01 | 21.36 | 15.83 | 12.41 |
| previous best (`stoch_s1g_scv2`) | — | 8.08 | 20.27 | 16.20 | 4.50 |

**Two of the four beat the campaign's previous best single model, from partial
checkpoints.** The two that help are trajectory-level: the flight states the
stream contains (ID) and how loudness relates to speed across the stream (Z).
The two that help less are geometry-level: which rotors are near each other in
speed (RP) and which airframe the profile came from (M).

That is a usable rule for what "realism" means for this task. It is the flight
envelope and the recording law that matter, not the airframe parameters.

## Combining the fixes does NOT compound — arm IDRP is worse than either part

Arm IDRP carries arm ID's restored warm-up idle and arm RP's near-degenerate
rotor pairs. At a comparable partial checkpoint (9 of 22 epochs) it is worse
than either fix alone:

| arm | all | zero | low | flight |
|---|---|---|---|---|
| ID alone (7 epochs) | **7.40** | 7.20 | 18.98 | 5.32 |
| RP alone (9 epochs) | 10.72 | 13.92 | 13.72 | 9.63 |
| **ID + RP (9 epochs)** | **13.37** | 17.17 | 19.69 | 11.56 |

The two interfere rather than add. That is the third time this campaign has
found that combining ingredients lands BELOW the parts — arms W and X did the
same with the cell winners' knobs — so it is a property of these streams, not an
accident of one pairing.

It also sharpens the trajectory-against-geometry split. ID and Z are
trajectory-level (which flight states the stream contains, how loudness relates
to speed) and are the top two arms; RP and M are geometry-level (which rotors
sit near each other, which airframe the profile came from) and are the bottom
two. Adding a geometry-level fix to a trajectory-level one costs more than it
buys. Arm IDZ — the two trajectory-level fixes together — is the test of whether
same-kind fixes compound where mixed-kind ones do not.

## Arm IDZ: same-kind fixes do not compound either, but the zero cell falls to 1.08

Arm IDZ carries both trajectory-level fixes — arm ID's restored warm-up idle and
arm Z's fixed reference gain, the campaign's two best single arms. As a single
model at 13 of 22 epochs it is WORSE than either part (14.70 against 7.40 and
8.01), so same-kind fixes do not compound any more than the mixed pairing did.
That is now four combination attempts (W, X, IDRP, IDZ) all landing below their
parts.

But it takes the DREGON zero cell outright:

| cell | before | arm IDZ | target |
|---|---|---|---|
| DREGON zero | 5.63 | **1.08** | 2.24 |

That is 0.48x the target — less than half the error of the model that trained on
real drone noise, on 844 frames.

Recomputing the routed figures with it:

| | before | now |
|---|---|---|
| DREGON, all three regimes | 0.98x | **0.74x** |
| Michael's, all three regimes | 2.09x | 2.09x |
| both rigs, oracle-routed | 1.34x | **1.19x** |

**DREGON is now 0.74x the in-domain target across all three of its regimes**, and
the both-rig oracle route is 1.19x. The gap is entirely Michael's ramp (2.86x)
and cruise (2.21x) — the two cells where the target trained on a sibling flight
of the validation aircraft.

## The comb's top edge is an artifact (2026-08-27)

Both synthetic engines ended their comb at `k_max * rps`. The stochastic engine
sized `k_max` from the flight's hover speed; the static comb zeroed each order
the instant it crossed Nyquist, per timestep. Either way a frame slower than
hover carried a sharp spectral cutoff at a frequency exactly proportional to the
rotor speed a model is asked to predict.

Measuring it needed care, because the naive step across the cutoff is dominated
by the spectrum's own tilt — synthetic +4.0 dB against real +3.9 dB, which says
nothing. Differencing each step against a control step at 0.7 of the same
frequency, and restricting to ramp frames (8 < f0 < 60), isolates it:

| stream | excess step at the cutoff | n |
|---|---|---|
| synthetic (built stream) | **+1.84 dB** | 234 |
| real DREGON, same frequency | +0.50 dB | 50 |

100% of ramp frames carried one.

The fix is a raised-cosine fade of the line power to zero at the band edge
(`band_taper_frac`), the comb sized from the window's slowest turning frames
rather than from hover, and a raised order cap so the comb reaches the band edge
at low RPM. Ramp frames retaining an in-band cutoff fall from 100% to 7.3%.

### Oversampling was not needed

The proposal on the table was to render at 32 kHz with ~120 orders and decimate
to 16 kHz, so that proper antialiasing replaces the brick wall. Measuring real
audio says the taper already delivers what that would buy. Median spectra,
each normalized to its own 2-3 kHz level:

| band | arm ID | arm BE | REAL valid |
|---|---|---|---|
| 3-4 kHz | -4.9 | -2.4 | -0.0 |
| 4-5 kHz | -7.2 | -3.6 | +0.1 |
| 5-6 kHz | -8.8 | -5.5 | -0.6 |
| 6-7 kHz | -10.5 | -9.3 | -2.5 |
| 7-7.9 kHz | -13.6 | -17.3 | -14.8 |

Real recordings are flat to 6 kHz and then fall 14.8 dB — a fixed-frequency
anti-alias rolloff, the recorder's. That is exactly what the taper imposes, and
what the old cutoff did not (its frequency moved with the rotor speed). Arm BE
is closer to real than arm ID in every band from 3 to 7 kHz. Rendering at 32 kHz
and decimating would produce the same fixed rolloff at two to three times the
cost, so it is not worth doing.

### Cost

The taper needs a few hundred orders to cover the band at low RPM, which tripled
the static comb's cost before two repairs paid it back: successive orders now
come from a recurrence on `exp(i*phase)` instead of a fresh sine each (exact to
2e-12 against the direct form), and an order that stays under the taper's onset
for a whole window takes a flat amplitude rather than a cosine per sample. The
tapered 300-order comb costs 49 ms/sample against the old 100-order comb's 53.

## The frequency axis is thrown away twice (2026-08-27)

`SimpleConvV2`'s `FrequencyAttentionPool` ends in `out.mean(dim=1)` over
frequency, and its attention over that axis carries no positional encoding. The
pool and everything after it are therefore EXACTLY permutation invariant over
frequency — shuffle the 17 surviving bands and the predicted RPS does not move,
verified to 1e-7. The encoder is convolutional and so weight-shared over
frequency, which leaves absolute frequency position essentially no route into
the head. The task's answer IS an absolute frequency.

The resolution is spent before that. Six encoder blocks each stride frequency by
two:

| stage | Hz/bin | comb spacing at 20-90 rev/s |
|---|---|---|
| front end | 7.8 | 2.6-11.5 bins |
| after block 2 | 31.1 | 0.6-2.9 bins |
| after block 3 | 62.0 | 0.3-1.4 bins — below the axis' own Nyquist |
| pool input | 470.6 | 0.04-0.19 bins |

Every architecture arm run in this project so far changed the temporal head or
the front end. None touched the frequency aggregation. Arms `stoch_s1id_freqpos`
(position only), `stoch_s1id_freqcat` (no averaging) and `stoch_s1id_freqhires`
(no averaging, four times the resolution) separate the two losses, each
differing from `stoch_s1id_scv2` by architecture alone.

### The phase-override bug, and why not to fix it (2026-08-27)

Arms ID, IDRP, IDZ and SI wrote their flight-phase overrides one level above
where the loader reads them (`rps.phases`), so the overrides never applied and
those runs used `FlightPhaseRanges`' defaults. Arm ID is the campaign's best
single model, so the natural move is to apply what was written. Measurement says
do not.

Conditioned on the ramp range (1 to 45 rev/s), against the frozen split's own
ramp frames:

| rev/s | defaults (what ID ran) | as written | fit A | fit B | REAL |
|---|---|---|---|---|---|
| 1-5 | 0.7% | 1.1% | 3.7% | 0.8% | 4.2% |
| 5-10 | 1.2% | 1.7% | 2.0% | 22.8% | 6.0% |
| 10-20 | 3.7% | 12.6% | 16.0% | 35.1% | 6.4% |
| 20-30 | 30.8% | 30.2% | 36.5% | 8.3% | 5.4% |
| 30-40 | 48.4% | 38.2% | 22.4% | 22.4% | 58.5% |
| 40-45 | 15.3% | 16.2% | 19.4% | 10.7% | 19.6% |
| **total variation** | **0.254** | 0.310 | 0.407 | 0.484 | — |

The accident was fortunate: the defaults match the real ramp distribution better
than the written values and better than two deliberate attempts to fit it. The
bug is left in place, documented in each arm's config, and the new arms omit the
block so they reproduce what arm ID actually ran.

Two mismatches survive and are not trajectory-tunable. The stream spends 30.8%
of its ramp time at 20 to 30 rev/s where real flights spend 5.4%, and only 1.2%
at 5 to 10 where real spends 6.0%. The 20-30 excess is not the idle band —
defaults put idle at 0.38 to 0.52 of hover, which is 30 to 42 rev/s — it is
`rps_scale_range: [0.6, 1.3]` scaling whole trajectories, which drags a
0.6-scaled flight's idle down to 18 rev/s. That scaling is arm C's lever against
the speed prior and is worth more than the distribution match, so it stays.

### The ramp cell may be a front-end limit, not a data limit

At `n_fft=2048` a Hann main lobe is 31.25 Hz wide, and the spacing between
adjacent harmonics IS the rotor speed. Below 31 rev/s adjacent harmonics are
therefore not separable at any frequency. The ramp regime is 1 to 45 rev/s, so
across most of that cell no model in this project has ever seen a resolved comb
— only a smeared envelope, plus whatever modulation the temporal head can pick
up across frames.

`comb_if` is the right shape of answer (one row per f0 candidate, a linear
grid) but searches only 30 to 120 rev/s, so most ramp frames fall below its
lowest candidate and cannot be represented at all. `comb_if_ramp` starts at 5.
Arms `stoch_s1id_combif` and `stoch_s1id_combif_hires` test it with the pooled
trunk and with the trunk that keeps the f0 axis.

#### A probe that does not support the strong version of this claim

A plain harmonic-sum matched filter (fixed 20 harmonics, whitened along
frequency over 150 Hz, nearest-rotor error) was run over the split to ask
whether the ramp information is present at all. Two estimator traps had to be
cleared first: a harmonic MEAN over a shrinking harmonic count pins the estimate
near the top of the search range, and whitening along TIME deletes a held comb,
which is exactly the case of interest.

| rig | regime | median error, n_fft 2048 | n_fft 8192 |
|---|---|---|---|
| DREGON | cruise | 0.81 | 0.50 |
| DREGON | ramp | 74.24 | 69.33 |
| Michael's | cruise | 63.39 | 54.45 |
| Michael's | ramp | 26.39 | 24.14 |

The probe passes its control on DREGON cruise — 0.81 rev/s median, against the
blind classical tracker's 0.68 — so it is sound there. Two things follow, and
neither is the strong claim.

DREGON's ramp stays at 74 rev/s median and a four-times finer window barely
moves it (69). DREGON's ramp is a fast sweep, 24.7 rev/s per second, so a longer
window smears as much as it resolves. For that cell, resolution is not the
binding limit.

On Michael's the probe fails at CRUISE (63.39) where the blind classical tracker
reaches 1.03, so it is simply too weak there — four rotors at close speeds
interleave their combs and a one-line harmonic sum locks onto a common divisor.
It therefore says nothing about Michael's ramp, which is the cell that actually
holds the gap.

So the resolvability arithmetic stands as arithmetic — below 31 rev/s adjacent
harmonics are not separable at `n_fft=2048` — but it is NOT established as the
binding limit for either ramp cell. The `combif` arms are the test.

### The ramp error is a systematic bias, and the two rigs bias opposite ways

Arm ID's best checkpoint, signed PIT error (prediction minus truth), channels 0
and 4 of the frozen split:

| rig | regime | mean | median | MAE | over-predicts | n |
|---|---|---|---|---|---|---|
| DREGON | cruise | -10.64 | -8.60 | 10.64 | 0.4% | 35968 |
| DREGON | ramp | -0.11 | +0.14 | 7.37 | 50.5% | 1456 |
| Michael's | cruise | -1.95 | -1.14 | 3.72 | 31.3% | 18888 |
| Michael's | ramp | **+10.02** | **+11.12** | 23.56 | 64.6% | 8568 |

Two of the four cells are dominated by a systematic offset rather than scatter.
Michael's held ramp has a truth median of 36 rev/s and is predicted near 47 —
which is the cruise threshold. DREGON's cruise is under-predicted almost
deterministically: only 0.4% of 36 k frames land high, a scale error of about
0.87 rather than a tracking failure. The two biases point in opposite
directions, so no global calibration helps, and a per-regime one needs a router
this campaign has already failed to build four times.

#### A lever that does NOT follow, recorded so it is not tried again

The obvious next inference is that PIT-MSE mis-weights the regimes. Measured on
arm ID's stream, and ASSUMING the same relative error on every frame, the
absolute-error loss would put 95.3% of its gradient on cruise (69.7% of frames)
and 4.7% on ramp (20.5% of frames) — which would make a relative or log-domain
loss an obvious fix.

That assumption is false here. The measured biases are about 10 rev/s in both
the cruise cell and the ramp cell, so this model's errors are roughly constant
in ABSOLUTE terms, not proportional to speed. Under equal absolute error, MSE
weights every frame equally and the shares simply track the frame counts. The
evaluation metric is absolute MAE as well, so the loss is already aligned with
the metric. No relative-loss arm was built.

### The synthetic idle is a unison, and one rig's is not (2026-08-27)

The trajectory model gates the differential modes (roll, pitch, yaw) to cruise,
on the stated reasoning that an aircraft holds near-zero attitude control on the
ground. The consequence was never checked. Rotor spread, max minus min across
the four rotors:

| source | ramp median | ramp inside 2 rev/s | cruise median |
|---|---|---|---|
| Michael's | **9.67** | 3.7% | 17.32 |
| DREGON | 0.03 | 83.0% | 11.77 |
| arm ID stream | **0.00** | **90.4%** | 9.32 |

Every synthetic ramp frame any model here has trained on shows four IDENTICAL
speeds. DREGON agrees with that; Michael's does not, and Michael's ramp is the
cell that holds the whole remaining gap. On a real aircraft the idle spread
comes from per-motor variation — ESC calibration, motor and propeller
differences, an uneven load — which does not switch off on the ground.

It also fits the measured error: arm ID over-predicts Michael's ramp by +10.02
rev/s and is unbiased on DREGON's ramp at -0.11. The bias sits on exactly the
rig whose idle configuration the stream cannot represent.

Michael's relative spread is nearly the same in both regimes (9.67/36 at ramp,
17.32/78 at cruise, about 27% and 22%), so ONE per-rotor speed ratio held over a
clip reproduces both. `rotor_trim_rel`, drawn per clip from zero upward, lets a
clip be Michael's-like or DREGON-like. At `[0.0, 0.15]` the stream lands between
the two rigs on both regimes — ramp median 5.76, cruise median 14.96. Arm TR is
arm BE with this as its only change.

### Band-edge arms: the fix works where aimed, the WIDTH was wrong (2026-08-27)

Scored at equal depth (arm ID's 7-epoch partial, BE at 7, BES at 8):

| arm | all-MAE | DREGON ramp | Michael's ramp | Michael's cruise | DREGON cruise |
|---|---|---|---|---|---|
| arm ID | **7.40** | **8.79** | 20.71 | **3.71** | **6.17** |
| arm BE (taper 0.30) | 15.15 | 10.74 | 14.72 | 5.24 | 20.73 |
| arm BES (+ sparse comb) | 11.47 | 13.16 | **10.83** | 7.70 | 13.51 |

Read against the real-only target `r4hb_scv2` at 2.67, arm ID stays the best
single model at 2.77x and neither band-edge arm sets a new best in any cell.

What the arms do show is that the fix acts exactly where it was aimed. 100% of
ramp frames carried the artifact, and arm ID's own Michael's ramp cell halves,
20.71 to 10.83. The sparse comb helps on top of it: BES beats BE on the cell
that holds the gap.

Cruise pays, and the cause is the taper WIDTH rather than the idea. At 0.30 the
fade begins at 5600 Hz, so a cruise clip at 78 rev/s loses every harmonic above
k of about 72 out of 100 — the high-order lines that carry cruise precision.
The artifact cannot have been HELPING on real validation, since real audio does
not contain it, so the cruise loss is the taper removing real signal rather
than the fix removing a shortcut.

Arm BE2 narrows the taper to 0.10 (fade above 7200 Hz, cruise keeps k to about
90) with everything else held at arm BES. Arm TR is arm BES plus the per-rotor
trim. Both are one change from BES, which is the better base on the binding
cell.

### Frequency-aggregation arms scored (2026-08-27)

All three share arm ID's stream, so architecture is the only variable.

| cell | arm ID (pooled) | freqpos (+2 k, position) | freqcat (no averaging) |
|---|---|---|---|
| all-MAE | **7.40** | 11.54 | 10.00 |
| DREGON zero | 5.77 | **3.70** | 14.43 |
| DREGON ramp | 8.79 | **8.25** | 10.12 |
| DREGON cruise | **6.17** | 11.82 | 10.33 |
| Michael's zero | 10.81 | **5.48** | 12.82 |
| Michael's ramp | 20.71 | 20.08 | **14.92** |
| Michael's cruise | **3.71** | 11.05 | 5.14 |

Neither variant beats arm ID overall, and one new campaign best came out of the
pair: DREGON ramp 8.79 to **8.25**, from the arm that adds 2 k parameters and no
arithmetic. That is the cheapest possible repair for the permutation
invariance, so the invariance is real and it does cost accuracy.

The two arms fail in different places, which is the useful part. `freqpos` wins
both zero cells and DREGON's ramp — it keeps the pooling and only labels the
bands. `freqcat` wins Michael's ramp (20.71 to 14.92) and is better overall — it
removes the mean entirely. Both give up cruise. So the frequency axis carries
information the pooled trunk cannot use, but simply keeping it is not free: it
trades cruise for the low-speed regimes.

`freqhires` separates position from resolution — same aggregation as `freqcat`,
65 bins at the head instead of 17.

### Taper width is not the lever, and aliasing is not the cause (2026-08-27)

Arm BE2 narrowed the band taper from 0.30 to 0.10 on the prediction that the
cruise damage came from fading away high-order harmonics. It did the opposite:

| arm | taper | all-MAE | DREGON ramp | Michael's ramp | DREGON cruise | Michael's cruise |
|---|---|---|---|---|---|---|
| arm ID | none | **7.40** | 8.79 | 20.71 | **6.17** | **3.71** |
| BE | 0.30 | 15.15 | 10.74 | 14.72 | 20.73 | 5.24 |
| BES | 0.30 + sparse | 11.47 | 13.16 | 10.83 | 13.51 | 7.70 |
| BE2 | 0.10 + sparse | 17.80 | **8.35** | **10.55** | 20.90 | 20.15 |

Cruise got worse, not better, so the width is not the lever and the stated
reason for the cruise cost was wrong.

What the family does show consistently is a TRADE. At every width tested the
band-edge fix improves both ramp cells and damages both cruise cells. BE2 has
the family's best ramp cells (DREGON 8.35, Michael's 10.55 against arm ID's
20.71) and its worst cruise. The artifact removal changes what the model learns
to read rather than simply deleting signal.

The follow-up guess was that a 0.10 taper leaves real amplitude right up to
Nyquist, so a chirping comb spreads past it and folds back. Measured against an
alias-free reference — the same comb rendered at 64 kHz and decimated with a
proper filter, taper off in both so only the sample rate differs:

| case | alias energy, in band |
|---|---|
| held comb, 30 rev/s | -25.7 dB (0.3%) |
| ramp, 30 to 55 rev/s | -17.0 dB (2.0%) |
| cruise, 75 to 85 rev/s | -17.4 dB (1.8%) |

Aliasing is real but bounded at about 2% of in-band energy, which cannot
account for a two-to-threefold MAE difference. The guess is refuted.

This also settles the oversampling proposal quantitatively: rendering at a
higher rate and decimating would correct a 2% effect at two to three times the
render cost. It is not worth doing.

NOTE on an invalid first attempt, recorded so it is not repeated:
`band_taper_frac` is defined relative to Nyquist, so an oversampled render puts
the fade at 22.4 kHz instead of 5.6 kHz. Comparing a tapered 16 kHz render
against a tapered 64 kHz render measures the taper mismatch, not aliasing. The
taper must be off in both, or expressed in absolute frequency.

### The per-rotor trim is refuted as implemented (2026-08-27)

Arm TR added a per-rotor speed ratio so the four rotors would stop idling in
exact unison. Against its base, arm BES, with the trim the only change:

| cell | BES | TR |
|---|---|---|
| all-MAE | **11.47** | 12.52 |
| DREGON zero | **11.47** | 34.42 |
| DREGON ramp | **13.16** | 33.02 |
| Michael's ramp | **10.83** | 16.15 |
| Michael's cruise | 7.70 | **6.24** |

Michael's ramp got WORSE, which is the cell the arm was built for, and both zero
cells collapsed.

The measurement behind it stands — the stream's ramp frames have a rotor spread
of 0.00 rev/s while Michael's have 9.67 — but the fix did not follow from it.
The likely reason is that the implementation over-reached the hypothesis: the
trim multiplies the WHOLE trajectory by a per-rotor ratio, so it does not merely
add spread at idle, it rescales every rotor's absolute speed through cruise as
well (cruise spread 9.32 to about 15). For a task that predicts absolute rev/s
that plausibly damages calibration, which is what the zero cells suggest.

A surgical version would add spread only in the non-cruise phases and leave
absolute speeds untouched. It is not queued: the unison-idle mismatch is real
but has not been shown to be the binding constraint, and this hypothesis has
already cost one cycle.

### The comb front end is refuted on both trunks (2026-08-27)

The `comb_if` family puts one row per f0 candidate on the frequency axis, so the
comb spacing is a POSITION rather than a pattern the trunk must resolve. The
prediction was that it fails on the pooled trunk (which averages that axis away)
and can work on a trunk that keeps the axis. Both halves are now measured, on
arm ID's stream, with the front end the only change:

| arm | trunk | all-MAE | DREGON ramp | Michael's ramp |
|---|---|---|---|---|
| `stoch_s1id_scv2` | pooled, STFT | **7.40** | **8.79** | **20.71** |
| `stoch_s1id_combif` | pooled, comb | 19.94 | — | — |
| `stoch_s1id_combif_hires` | freq-preserving, comb | 21.20 | 35.29 | 28.56 |
| `stoch_s1id_combif_hires2` | freq-preserving, comb | 24.42 | 30.85 | 24.98 |

The frequency-preserving trunk does not rescue it. It is WORSE than the pooled
trunk (21.20 and 24.42 against 19.94), and every cell is three to four times the
STFT baseline. The zero cells are catastrophic (48 to 50 rev/s) — with no comb
present the f0 axis carries no signal and the model has nothing to fall back on,
whereas the STFT trunk still sees a broadband level.

`hires` and `hires2` are the same configuration, cut short and run to the full
hour. More training made it WORSE (21.20 to 24.42), so this is not an
undertrained result — the arm is fitting the synthetic stream and moving away
from the real split, the same curve shape the earliest comb arms showed.

The whole `comb_if` direction is closed. The permutation-invariance measurement
that motivated it stands, but it does not follow that an explicit f0 axis is the
answer: the tooth table is built from the synthetic comb's own geometry, so it
hands the model a template that only the synthetic data matches exactly.

### Arm IDV: the visibility trade is real, not a coverage artifact (2026-08-27)

The two cells that hold the remaining gap are held by two different arms that
differ mainly in the line-visibility block. Arm IDV widened those four knobs to
the UNION of arm H's tight settings and the defaults arm S ran, so one stream
covers both regimes instead of committing to one. Nothing else changed — the
inert flight-phase block included, deliberately.

| | arm S | arm ID | ARM IDV |
|---|---|---|---|
| visibility block | absent | tight | union |
| all-MAE | 13.95 | **7.40** | 10.10 |
| Michael's ramp | **8.14** | 20.71 | 23.21 |
| Michael's cruise | 12.14 | 3.71 | **3.36** |
| DREGON cruise | 15.26 | **6.17** | 7.72 |

The pre-registered reading was: keeps ID's cruise and takes some of S's ramp,
the trade is a coverage artifact; lands between the two on both cells, the
trade is real. **It did neither favorable thing.** Michael's ramp got WORSE than
either parent (23.21 against 20.71 and 8.14), which is the cell the arm was
built for, and DREGON cruise regressed as well. Only Michael's cruise improved,
by 0.35.

TRAINING DEPTH FAVORS THE LOSER, so the result is not a depth artifact.
`num_batches_tracked` on the two best checkpoints: arm ID 1565 (5 epochs at 313
batches), arm IDV 2504 (8 epochs). Arm IDV had 60% more training and still lost
by 2.70 all-MAE.

WHAT THIS CLOSES. Widening a committed knob to cover both settings is not a
free lever, and "coverage beats realism" does not generalize to every knob. The
lesson held where coverage added a MISSING STATE the family could not produce
at any setting — arm C's speed prior, arm ID's warm-up idle. Line visibility is
not a missing state; it is a property every clip already has at some value, and
widening its range only dilutes the setting that each cell needs. No further
union-widening arm is queued on this reasoning.

The consequence for the two open cells is that one stream apparently cannot
serve both. Holding both would need routing or a per-clip conditioning signal,
which is a different kind of answer.

### Heavier transformer heads do not help, and capacity is not the axis (2026-08-27)

Transformers had failed on every synthetic stage 1 this project ran, but always
on a NARROW noise family (`m3abl_comb_transformer_s1` 1802.0 on the analytic
comb, `m3cur_transformer_s1` 316.9 on generator + comb). Three arms on arm ID's
much wider stochastic stream, with the temporal head's capacity as the only
change and the encoder, front end and pool identical:

| arm | temporal head | params | best epoch | all-MAE | Michael's ramp | DREGON cruise |
|---|---|---|---|---|---|---|
| `stoch_s1id_scv2` (BiGRU) | — | 1.50M | 5 | **7.40** | 20.71 | **6.17** |
| `stoch_s1id_tr` | 2 x 64 | 1.48M | 10 | 12.44 | **16.25** | 17.27 |
| `stoch_s1id_trmed` | 4 x 128 | 2.21M | 10 | 14.78 | 20.05 | 19.12 |
| `stoch_s1id_trbig` | 6 x 256 | 6.24M | **1** | 13.11 | 16.52 | 16.90 |

TWO THINGS THE LADDER SETTLES.

Capacity is not the axis. The three all-MAE figures are 12.44, 14.78, 13.11 —
not monotone in parameters, and all of them 1.7 to 2.0 times the BiGRU at
essentially the same total model size for the stock arm (1.48M against 1.50M).
The earlier failures were therefore not an underfitting head on a narrow
family. The transformer head is worse at this task at every capacity tried.

Depth favors the losers, so this is not the hour cap. The two smaller arms
reached epoch 10 against the BiGRU's best at epoch 5. The largest is the
diagnostic one: it ran to epoch 11 and its BEST checkpoint is from epoch 1
(313 batches tracked), so it peaked immediately and got worse for ten epochs —
the same diverging curve `m3abl_comb_transformer_s1` showed at 1802.0. More
capacity buys faster divergence, not more fit.

THE ONE RESULT THAT CUTS THE OTHER WAY. Every transformer arm beats the BiGRU
on Michael's ramp (16.25, 20.05, 16.52 against 20.71) while being catastrophic
at cruise (17.27, 19.12, 16.90 against 6.17). That is the SAME trade arm S and
arm H show, and the same one arm IDV failed to dissolve. Three independent
levers — the noise family's line visibility, the temporal architecture, and the
front end — now produce one pattern: whatever reads ramps well reads cruise
badly. This looks structural rather than incidental, and it is the strongest
argument yet that a single model cannot hold both cells.
