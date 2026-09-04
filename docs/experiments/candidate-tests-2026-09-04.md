# Testing the two shortlisted architectures (2026-09-04)

Status: IN PROGRESS. The shortlist comes from
`docs/rps-tracking-architecture-candidates.md` (2026-09-03): C1, the slot-comb
CRF with a learned partial-observation emission, and C2, the HG-CKLA refiner
with three fixes. This document records the test protocol, every arm, and the
verdict on each candidate.

## Protocol

**Test set.** The frozen real split `dload:DREGON-LM-V4-michaels-valid-full`:
37 clips of 8 s at 16 kHz with 8 microphones (DREGON room 1 and FLY124).
Phases by the label: ground (all rotor means below 1 rev/s), ramp (mean rotor
range above 15 rev/s), cruise. Rigs: DREGON and FLY124. FLY124 is not in any
training pool (FLY125 is), so its numbers are cross-drone.

**Metric.** Per-clip PIT MAE in rev/s on the prediction's frame grid
(`experiments.rps_bench.pit_mae`), then the mean over clips of a phase or rig.
This is the metric of the error-profile campaign, so the trained regressors
and salience ports of `docs/experiments/rps-error-profile.md` are read on the
same table. Also reported: the fraction of cruise rotor-frames within 10 % of
the truth, of half the truth and of twice the truth (the octave readout).

**Synthetic parts.** The salv2 comb and stochastic validation parts
(`conf/data/salv2_comb_nomix.yaml`, `salv2_stoch_nomix.yaml`), mono, 8 s,
256 clips each, scored with the same metric, as the control that a real-trained
emission keeps its synthetic performance.

**Reference rows** (from the synthesis doc, same clips and metric):

| row | DREGON cruise | FLY124 cruise | ramp | ground | all |
|---|---|---|---|---|---|
| best trained regressor `r4hb_scv2` | 2.28 | – | 5.11 | 1.60 | 2.74 |
| zero-parameter slot decoder, 8 mics, 15-bin floor (P1c) | 1.49 | 21.6 | – | – | 19.3 |
| HG-CKLA v1 one pass behind `r4hb_scv2` (P4) | 2.09 | – | 5.04 | 1.54 | 2.59 |

## C1: the slot-comb CRF with a learned emission

**Model.** `models.comb_slots.SlotCombNet(emission="partial")`: the P1c corner
(k_max 40, grid 30-100 rev/s at 0.1, 15-bin floor, 8 mics power-averaged,
peel + Viterbi + relocation, octave move off) plus four learnable parts, all
initialized at the corner so an untrained model reproduces it:

- `reliability`: a per-(order, channel, rate, frame) weight from a 7-input MLP
  over the reading, its rate neighbours, a short time average, the floor
  level, the order fraction and the channel kind, times a per-order weight and
  a per-channel gate. It replaces the mean over harmonics.
- `channels`: the 8 per-mic readings enter as candidates next to the
  power-mean channel (gates start closed).
- `empty_tooth`: a hinge charge on predicted teeth that land on the floor, the
  two-sided octave term.
- `floor_mix`: a geometric mixture of running-median floors at 15, 31 and 61
  bins with learned weights.

**Training.** The CRF negative log-likelihood of the true trajectories through
the deployed decoder (`SlotCombNet.loss`), on 2 s crops of two online-mix
streams: real 8-mic windows of the honest pool with the silence arm
(`conf/online_mix/slot_real_dload.yaml`) and the partial-comb stochastic
family (`conf/online_mix/slot_partial_dload.yaml`). Crops keep only frames
where every rotor is inside the grid (30-100 rev/s); rates below 30 rev/s and
stopped rotors are out of scope for the loss. Selection is on a fixed set of
48 real windows drawn from the training pool with another seed, never on the
test split. Trainer: `scripts/train_slot_real.py`, chained on gpushort with
`scripts/chain_cmd.sh`.

**Arms.**

| arm | parts | data |
|---|---|---|
| A0 | none (the corner) | – |
| A1 | all four | real + partial |
| A2 | all minus `channels` | real + partial |
| A3 | all minus `empty_tooth` | real + partial |
| A4 | all minus `floor_mix` | real + partial |
| A5 | all minus `reliability` (per-order weight and channel gates only) | real + partial |
| A6 | all four | real only |

**Gate.** A1 must not lose DREGON cruise against A0 (1.49) while it moves
FLY124 cruise and the ramp; a trade of DREGON for FLY124 is the kill criterion
of the synthesis doc (the two rigs would need different harmonic caps).

**Results.** _Pending._

## C2: the HG-CKLA refiner with the three fixes

**Model.** `models.hg_ckla.HGCKLARefiner` with `state_features`,
`kalman_gain` and `smoother` on (`conf/model/hg_ckla_refiner_v2.yaml`):
the cell sees the state's own differences, the gain comes from a carried
scalar variance with a measurement variance read from the phasor scatter, and
a backward RTS pass smooths the error state.

**Training.** `conf/experiment/hb_hgckla_ref_v2.yaml`: the corrupted-label task
of v1 (`(audio, corrupt(GT)) -> GT`, plain MSE) on the honest pool, 40 epochs
ceiling. The v1 run was flat from epoch 1 to 21 at 113 s per epoch on an A100.

**Tests** (`experiments.refiner_bench`, the P4/M5 harness): M1 corrupted
labels, M2 behind `r4hb_scv2` and behind the best C1 arm, M3 oracle drift
(the fixed-point test), M4 capture at 0.5-8 rev/s, M5 three passes.

**Gate.** A fixed point at the truth: oracle drift at cruise well below v1's
0.41 rev/s, and passes 2 and 3 not worse than pass 1. The comparison row is
one classical `pi_kalman` pass (`experiments.classical_pass`, the flagship
protocol row) from the same initializations.

**Training.** 18 epochs on an A100 (early stop, patience 8), 2 minutes per
epoch, best at epoch 9. On its own validation (the corrupted-label task) v2
matches v1 and does not beat it: per-frame MAE 1.001 against v1's 1.005.

**Results** (`results/refiner_bench/hb_hgckla_ref_v2/REPORT.md`, per-frame
protocol: 296 mono frames, PIT MAE in rev/s):

| test | phase | v1 (P4) | v2 |
|---|---|---|---|
| M2 one pass behind `r4hb_scv2` | cruise | 2.28 -> 2.09 (-8 %) | 2.28 -> **1.85 (-19 %)**, 1 % of frames hurt |
| M2 one pass behind `r4hb_scv2` | all | 2.74 -> 2.59 | 2.74 -> 2.41 |
| M5 passes 1 / 2 / 3 behind `r4hb_scv2` | cruise | 2.09 / 2.12 / 2.17 (walks out) | **1.85 / 1.79 / 1.73** (keeps improving) |
| M3 oracle drift (own floor) | cruise | 0.41 (median 0.39, p90 0.59) | **0.55** (median 0.55, p90 0.68); signed -0.19 % |
| M5 passes from the oracle | cruise | 0.41 / 0.61 / 0.73 (walks out) | 0.55 / 0.75 / 0.88 (walks out) |
| M2 one pass behind the C1 arm A4 | cruise | 5.66 -> 5.63 (-0.5 %) | 5.66 -> 5.68 (+0.4 %), 25 % of frames hurt |
| M2 one pass behind A4 | DREGON cruise (per clip, classical pass) | 0.90 -> 0.93 | – |
| M4 pull at +1 / +2 / +4 rev/s | cruise | ~40 % fixed | 66 % / 57 % / 37 % |
| M1 corrupted labels | all | 1.16 -> 1.00 | 1.16 -> 1.00 |

(The v1 column is the same harness re-run in `results/refiner_bench/hb_hgckla_ref_behindA4/`; it reproduces P4.)

Behind the C1 seed neither refiner adds anything: A4's cruise error is made of
octave and assignment failures on FLY124, which lie outside any refiner's
capture, and its DREGON error sits at the label floor. So the refiner
question only matters behind a regressor seed.

Reading. The three fixes change the refiner's character: the carried gain
pulls harder (66 % of a 1 rev/s offset in one pass against 40 %) and the pass
behind a real seed now converges under iteration instead of walking out. But
the fixed point is still not the truth: from a perfect initialization v2
moves 0.55 rev/s at cruise, more than v1's 0.41, and iterating from the truth
walks out to 0.88. The signed part of that drift is -0.19 % at cruise, the
direction of the DREGON label bias, so a small part of the "drift" is the
acoustic truth; the rest is scatter. On the corrupted-label task the two
versions tie, which is why the training loss could not select for the
fixed-point property. Verdict: C2 v2 is the better coarse puller behind a
regressor (2.3x the one-pass gain, safe to iterate) and is NOT a precision
stage below 0.5 rev/s. The gate "oracle drift well below 0.41" fails.

**The classical pass on the same clips** (`experiments.classical_pass`,
per-clip protocol, 8 mics): behind `r4hb_scv2` cruise 2.24 -> 2.16 (-3 %),
from the oracle 0.35 at cruise (0.29 DREGON, 0.43 FLY124). Behind the A4 slot
decoder (below): DREGON cruise 0.904 -> 0.932 -> 0.954 -> 0.973 over three
passes, FLY124 13.27 -> 13.24, so the classical refiner has nothing to add to
a seed that already sits at the label floor, and it walks away from it slowly.

## Conclusions

_Pending._
