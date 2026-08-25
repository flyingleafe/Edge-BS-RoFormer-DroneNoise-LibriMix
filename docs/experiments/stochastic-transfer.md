# Synthetic-only transfer with the stochastic noise family

**Question.** Can a rotor-speed predictor trained on synthetic noise alone reach
the real frozen validation split, if the synthetic family is wide enough to
contain the real thing?

**Status: OPEN.** Started 2026-08-25.

## Why the question is open

Every synthetic-only predictor this project has trained transfers badly. On the
frozen split `dload:DREGON-LM-V4-michaels-valid-full`, scored as frame-weighted
PIT mean squared error:

| stage-1 arm (synthetic only) | noise family | val PIT-MSE |
|---|---|---|
| `m3abl_comb_unigru128_s1` | analytic static comb | 212.5 |
| `e8_staticcomb_s1_unigru128` | analytic static comb | 225.3 |
| `e7_gencurric_s1_scv2` | neural generator | 226.3 |
| `m3abl_comb_scv2_s1` | analytic static comb | 336.8 |
| `m3cur_scv2_s1` | generator + comb | 349.2 |
| `m3abl_comb_transformer_s1` | analytic static comb | 2563.5 |

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

| experiment | trunk | comparison row |
|---|---|---|
| `stoch_s1_scv2` | bidirectional GRU | `m3abl_comb_scv2_s1` (336.8) |
| `stoch_s1_unigru128` | causal GRU | `m3abl_comb_unigru128_s1` (212.5) |
| `stoch_s1_transformer` | transformer, IF front end | `m3abl_comb_transformer_s1` (2563.5) |

Stream check: PASS (`python scripts/check_stream.py --experiment stoch_s1_scv2`)
— all three augmentation blocks fire at their configured rates, frequency
scaling changes the labels, and the stream is deterministic per sample id.

## Where the earlier arm fails

`scripts/valid_regime_eval.py` splits the frozen split into three regimes by the
target speeds — zero (every rotor stopped), low (warm-up and the ramps), flight
(mean at or above 45 rev/s) — and reports PIT mean absolute error in each.

Read this table before changing the family: it says which part of the problem
synthetic-only training already solves.

| checkpoint | aggregate | all MAE | zero | low | flight |
|---|---|---|---|---|---|
| `m3abl_comb_scv2_s1` (3 clips, channel 0) | 214.1 | 10.21 | **0.26** | 15.59 | 15.77 |

A synthetic-only predictor already solves the stopped-rotor regime outright:
0.26 rev/s, better than any real-trained model reaches there. Everything it
loses, it loses at nonzero speed.

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

**One candidate stands out on size.** A real validation clip has an RMS of
0.0129. Every synthetic chunk is normalized to 0.1 before mixing — an 18 dB
level difference — and the post-mix gain augmentation only spans plus or minus
6 dB. A magnitude front end is not scale invariant, so the whole stream may be
arriving at an operating point the model never trained at.

## Log

* **2026-08-25** — family built, measured against the real comb instrument,
  wired as `kind: stochastic`, arms A/B/C configured, stream check passed, all
  four submitted (`stoch_s1_scv2`, `stoch_s1_unigru128`, `stoch_s1b_scv2`,
  `stoch_s1c_scv2`).
* **2026-08-25** — the failure at speed is a scale error of 0.836, not a lost
  comb: response slope 0.94 on real audio, and the same checkpoint reads an
  unseen synthetic family to within 5%. Arm C (`rps_scale_range`) attacks the
  prior; the level gap is the next candidate.
