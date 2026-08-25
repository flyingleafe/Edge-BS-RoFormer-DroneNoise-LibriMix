# Synthetic-only transfer with the stochastic noise family

**Question.** Can a rotor-speed predictor trained on synthetic noise alone reach
the real frozen validation split, if the synthetic family is wide enough to
contain the real thing?

**Status: OPEN.** Started 2026-08-25.

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

## Log

* **2026-08-25** — family built, measured against the real comb instrument,
  wired as `kind: stochastic`, arms A/B/C configured, stream check passed, all
  four submitted (`stoch_s1_scv2`, `stoch_s1_unigru128`, `stoch_s1b_scv2`,
  `stoch_s1c_scv2`).
* **2026-08-25** — the failure at speed is a scale error of 0.836, not a lost
  comb: response slope 0.94 on real audio, and the same checkpoint reads an
  unseen synthetic family to within 5%. Arm C (`rps_scale_range`) attacks the
  prior.
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
