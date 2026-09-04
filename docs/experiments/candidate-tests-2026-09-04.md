# Testing the two shortlisted architectures (2026-09-04)

Status: CLOSED 2026-09-04 (A3b, the guarded rerun of A3, lands last; see its row). The shortlist comes from
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

**Results** (per-clip PIT MAE in rev/s on the frozen split at the checkpoint
the selection set picked; "sel" is the selection metric on 48 held-out real
windows; the FLY124 columns give the fraction of cruise rotor-frames within
10 % of the truth / within 5 % of half / within 20 % of double; source: the
job logs, `results/slot_real/<arm>/report.json` where pulled):

| arm | parts | data | steps | sel | ground | ramp | cruise | DREGON cruise | FLY124 cruise | all | med | FLY 1x / half / 2x |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A0 | none (the corner) | – | 0 | 13.00 | 67.6 | 29.1 | 9.21 | 1.49 | 21.56 | 19.28 | 16.1 | 0.40 / 0.25 / 0.09 |
| A1 | all four | both | 1500 | 5.98 | 68.1 | 33.7 | 7.52 | 0.91 | 18.10 | 19.02 | 9.8 | 0.48 / 0.17 / 0.10 |
| A2 | no `channels` | both | 1500 | 5.74 | 67.3 | 33.8 | 6.81 | 1.04 | 16.03 | 18.44 | 9.8 | 0.56 / 0.18 / 0.09 |
| A3b | no `empty_tooth` (guarded rerun) | both | 1500 | _pending_ | | | | | | | | |
| A4 | no `floor_mix` | both | 1500 | 5.09 | 67.9 | 31.0 | 5.66 | 0.90 | 13.27 | 17.18 | 3.9 | 0.58 / 0.075 / 0.04 |
| A4 | no `floor_mix`, extended | both | 3000 | 4.17 | 64.6 | 32.0 | 5.10 | 0.90 | 11.80 | 16.62 | 4.1 | 0.59 / 0.025 / 0.13 |
| A5 | no `reliability` | both | 1500 | 8.51 | 62.8 | 23.8 | 6.52 | 0.86 | 15.58 | 15.87 | 9.9 | 0.54 / 0.17 / 0.13 |
| **A6b** | all four | **real only** | 1500 | **3.14** | 60.9 | 31.1 | **4.94** | **0.76** | **11.63** | **15.93** | 4.7 | 0.55 / 0.000 / 0.10 |
| A7 | all four, octave charge ON at init | both | 1500 | 6.05 | 67.4 | 29.9 | 6.47 | 0.91 | 15.36 | 17.48 | 6.9 | 0.56 / 0.125 / 0.075 |
| A6 (first run, diverged at step 400) | all four | real | 300 | 8.22 | 75.4 | 37.0 | 10.57 | 0.92 | 26.02 | 22.57 | 12.8 | 0.31 / 0.19 / 0.12 |
| A4 at 1500 with the octave move ON at decode | – | – | – | – | 69.1 | 32.4 | 10.50 | 0.86 | 25.91 | 20.98 | 20.6 | 0.21 / 0.36 / 0.10 |

The checkpoint-to-checkpoint scatter of FLY124 cruise inside one run is
about +-3 rev/s (two or three clips change octave state between validations),
so differences below that between arms are not readable; DREGON cruise is
stable to +-0.1.

Notes.

- Every trained arm takes DREGON cruise from 1.49 to 0.86-1.04 and FLY124
  cruise from 21.6 to 13-16, so the gate holds: no arm trades one rig for the
  other. On DREGON the signed error of A4 is -0.48 % (median over rotor-
  frames), which is the known telemetry bias; A4's DREGON error is at the
  label floor (W7) and cannot be scored lower against these labels.
- What A4 learned (`results/slot_real/A4/best.pt`): the per-mic channel gates
  stayed closed (sigmoid(b) 5e-4 for every mic, 1.0 for the power mean), the
  empty-tooth charge stayed off (lambda 7e-4 from a softplus(-8) start), the
  per-order weights barely moved (1.01 at k 1-9, 0.98 at 10-24, 0.89 at
  25-40), and the work is in the convex warp of the readings (knot slopes
  0.54 / 0.34 / 0.34 / 0.26 / 0.13 / -0.09 / 0 / 0 at z = 0..12) and the
  reliability network's modulation. So the emission learned to amplify strong
  readings over weak ones, a soft order statistic, and not an explicit octave
  rule or a harmonic cap.
- The selection curve is not converged at 1500 steps (A4: 13.0 -> 15.0 at
  step 200 -> 5.09 at 1500, still falling). Extended to 3000 steps A4 reaches
  4.17 and is flat from step 2400 on; FLY124 cruise moves 13.3 -> 11.8, the
  half-rate fraction 0.075 -> 0.025, DREGON stays at 0.90.
- The remaining FLY124 error of A4 has two parts: the two 36 rev/s hover clips
  are read at exactly twice the rate (median ratio 2.00; 7.3 of the 13.3 mean
  come from these two clips), and on the 80 rev/s clips one rotor of four
  sits off the truth ("other" ratio 0.21-0.26 of rotor-frames). The
  empty-tooth term cannot charge a multiple (its teeth all land on true
  lines); the decoder's coverage-judged octave move is the rule for that
  case, and switching it on costs 13.3 -> 25.9 (half-rate fraction 0.075 ->
  0.36), the same failure P1c measured on the untrained corner. The multiple
  needs a discriminator that real audio does not give the union rule.
- The mono synthetic parts get WORSE under every trained arm (comb 50 -> 57-60,
  stoch 39 -> 45; means dominated by stopped-rotor and below-grid frames), so
  the real-window emission does not keep the synthetic performance; the
  parts are out of scope for this test and are reported as the control they
  were meant to be.
- Two arms (A3, A6) went non-finite near step 400 (reliability + channels +
  floor_mix; the reliability factor `1 + MLP` is unbounded below, so the
  weight sum can reach zero and the CRF overflows). The trainer now skips
  non-finite steps and restores the last best head; both arms were rerun
  (A3b, A6b). A6b met no non-finite step at all, so the event is rare and
  data-order dependent, not systematic.
- Data: real windows alone (A6b) beat real + partial-comb (A1, same parts,
  same steps) on every column: DREGON 0.76 against 0.91, FLY124 11.6 against
  18.1, cruise 4.9 against 7.5, selection 3.14 against 5.98. The partial-comb
  synthetic family does not help the emission and costs it; the emission is a
  real-audio object.
- The explicit octave charge: left at its softplus(-8) start it stays off
  (A4: lambda 7e-4); started ON it stays on and strengthens (A7: lambda
  0.69 -> 1.02, tau 1.09) and the result is no better than without it (A7
  15.4 against A4 13.3 and A1 18.1 on FLY124, DREGON 0.91 in both). The
  half-rate reading disappears in every trained arm (FLY124 half fraction
  0.25 -> 0.00-0.13) with or without the charge; what the charge was built
  for is already handled by the learned reweighting of the readings.
- The floor mixture never moves (weights 0.999 on 15 bins in A1, A7) and the
  per-mic channel gates never open (5e-4 in every arm that had them): of the
  four parts only `reliability` (the warp plus the MLP modulation) and the
  per-order weights carry the gain, and A5 shows that the per-order weights
  with the warp already carry most of it.
- Residual anatomy of the best arm A6b (`results/slot_dump/A6b/`): on the
  eight 80 rev/s FLY124 cruise clips the error is 4.0-5.1 rev/s per clip with
  58-76 % of rotor-frames at the truth and 24-42 % at a non-octave value,
  that is one to one-and-a-half rotors of four on a decoy or a duplicate; the
  two 36 rev/s hover clips are read at twice the rate (median ratio 2.02-2.04)
  and alone contribute 7.9 of the 11.6 mean. Without those two clips FLY124
  cruise is about 4.7. DREGON cruise reads -0.43 % (median signed error), the
  telemetry bias, with every rotor-frame within 10 %. Ramp clips split into
  frames below the 30 rev/s grid or at zero (which the model cannot express)
  and in-grid frames that are off by 6-8 (DREGON) or 15-43 (FLY124, the
  double-rate reading again at the 30-40 rev/s holds).

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

**C1 passes its gate and is the architecture to carry forward.** A 135-
parameter emission trained through the CRF on 2 s real 8-mic windows takes
the zero-parameter corner from 1.49 to 0.76 on DREGON cruise (every
rotor-frame within 10 %, signed error -0.43 % = the telemetry bias, so it is
at the label floor W7) and from 21.6 to 11.6 on FLY124 cruise, an unseen rig,
with the half-rate reading gone (0.25 -> 0.00). No arm trades one rig for the
other, so the kill criterion of the synthesis doc (rig-specific harmonic caps)
does not fire. Against the trained neural models of the error-profile
campaign on the same clips, C1 is 3x better on DREGON cruise (0.76 against
2.28 for the best regressor) and is the first learned model that reads the
unseen rig at all (11.6 against 45 for the regressors).

**What C1 learned is a soft order statistic, not the rules that were
written for it.** The per-mic channels, the floor mixture and the explicit
empty-tooth charge were each either left unused or used without effect; the
gain lives in a convex warp of the floor-relative readings plus a small
reliability modulation and per-order weights. The design requirement
"reliability over (order, channel) in place of the mean" is confirmed; the
requirements "learned floor width", "per-channel pooling" and "explicit octave
term" are not needed on this split.

**What C1 still gets wrong, and what it needs next.** (1) The double-rate
reading at 30-40 rev/s holds (two FLY124 cruise clips, most FLY124 ramp
frames): a multiple cannot be charged by empty teeth and the coverage rule
picks half rates on real audio (octave move ON: 13.3 -> 25.9). The grid's low
edge at 30 rev/s is part of this: a 36 rev/s truth sits 6 grid-steps from the
edge and its double has the full grid to win in. Extending the grid below 30
with a level-aware zero decision is the next change. (2) One rotor of four on
a decoy or a duplicate on the 80 rev/s FLY124 clips (24-42 % of rotor-frames):
an assignment failure, W1, the phase-continuity problem the CRF campaign
named. (3) Below-grid and zero frames (ramps, ground) are out of scope of the
loss and the grid, which is why ramp and ground did not move in any arm.

**C2 fails its gate and is closed as a precision stage.** The three fixes
make HG-CKLA a 2.3x stronger one-pass puller behind a regressor (cruise 2.28
-> 1.85, and 1.73 after three passes instead of walking out) but its fixed
point is still not the truth (oracle drift 0.55 at cruise, worse than v1's
0.41 and the classical pass's 0.35, and iterating from the truth walks out to
0.88). Behind C1 neither refiner version nor the classical pass changes
anything, because C1's remaining error is assignment and octave, outside any
refiner's capture, and its DREGON error is at the label floor. The refiner
question is therefore moot behind C1; it only matters behind a regressor
seed, where v2 is the one to use, once.

**Closed by this campaign:** the partial-comb synthetic family as training
data for the emission (real windows alone are better); the per-mic channels
as emission candidates (gates never open); the learned floor mixture (never
moves); the explicit empty-tooth charge (used when started on, no effect);
the coverage-judged octave move on real audio (costs 2x); HG-CKLA as a
precision stage (v1 and v2 both drift from the truth and walk out under
iteration from it); any refiner behind C1 on this split.

**Costs.** C1: 135 parameters, 1.2-1.9 s per training step on an A100 at
batch 2 (2 s crops, 8 mics), 0.8 s per 8 s clip at decode (classical corner)
and about 2 s with the partial emission on GPU (47-79 s on CPU). C2: 222k
parameters, 2 minutes per epoch, 18 epochs.

**Pending at closing time:** A3b (the guarded rerun of the no-`empty_tooth`
arm); its row completes the ablation but cannot change the conclusions above,
because A4 and A7 already bracket the charge's effect.
