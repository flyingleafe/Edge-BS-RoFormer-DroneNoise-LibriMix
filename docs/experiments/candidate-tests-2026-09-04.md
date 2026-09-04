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

**Results.** _Pending._

## Conclusions

_Pending._
