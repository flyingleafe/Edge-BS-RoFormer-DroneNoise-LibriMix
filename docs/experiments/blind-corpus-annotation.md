# Blind annotation beyond DREGON/Michael's: AVQ pilot + corpus plan

**Date**: 2026-08-12 · **Status**: pilot DONE (branch `blind-corpus`, head
11365db — scripts/blind_corpus{,_report,_triage}.py + committed artifacts in
results/blind_corpus/). Cluster follow-ups queued behind the omnirun daemon
disk outage: `blind-avq-8ch` (the run that matters), `blind-avq-dregon-calib`,
`blind-avq-dregon-motor`.

## Label-free quality instruments (no telemetry exists on these corpora)

F_VK of the blind trajectory vs six fixed-cell perturbed siblings (±0.3/±1.0
rev/s wells, r/2, 2r), ridge concentration vs off-comb half-integer null and a
time-reversed mismatch partner, plus the ladder's own guards. Load-bearing
design point: r/2 is a SUPERSET of r's harmonics, so least-squares fits never
prefer r — `fvk_ratio_half` ≈ 1 by construction; the octave verdict comes from
the ridge and the F_VK alias counter-term.

## Pilot verdicts (AVQ ego-noise, 84/84 units, job blind-avq-mono)

- BEST: S1_seq1 w000 @ 8 s — all four instruments positive (+2.52 dB
  clearance), twin-pair rate pattern; works because the comb is stationary
  and the window stays coherent.
- WORST: DREGON hovering w000 (pre-takeoff: a perfect negative control — no
  well at all); S1_seq1 w001 @ 20 s (sub-harmonic capture: strongest ridges
  explained as THIRD harmonics; clearance below the off-comb null).
- Headline instability: the same window annotates 52/57 rev/s at 8 s and
  78/113 at 20 s — its v(b)/v(2b)=1.41 sits exactly on the 1.4 halving
  threshold.
- Median mono window: half-margin NEGATIVE, F_VK(2r)/F_VK(r)=1.14 — octave
  unresolved on more than half. **Retroactive finding: the published
  AVQ-egonoise-vkrps pseudo-labels are mono-seeded (C=1 seed collapse:
  DREGON PIT-MAE 1.81→18.76) and gated on confidence alone — this explains
  the recorded negative result of the arm trained on them. AVQ must be
  annotated from the 8-channel dataset.**

## Corpus plan

Order: DREGON single-motor bench → DREGON room2 (proxy telemetry —
calibrates all thresholds) → AVQ 8-ch ego → DroneAudioSet → AVQ speech →
SPCUP19 → MIMII/HUST/KAIST subsampled → refuse mono far-field by default.
Recipe: blind_fullrange where rates ramp; vit2dsp where steady; n_rotors=1
for bench. Acceptance = three label-free gates (VK confidence ≥ 0.02, ridge
clearance over off-comb null > 0, ridge(r) > ridge(r/2) AND F_VK(2r)/F_VK(r)
clearly > 1); failures emit NaN. Main missing instrument: a PER-ROTOR octave
test (the global bpf_octave_ratio sits on its threshold for AVQ). Cost: ~45
CPU-h for classes B/C/D/F (one uni-cpu job, <4 h wall).

Actionable extras: enabling the F_VK `alias_penalty=1.0` widened the
half-margin 3.3 % → 14 % on the validated window — the counter-term is the
right octave lever and should be promoted from opt-in to default WITH
per-rotor granularity (the global bpf_octave_ratio is what flips with window
length).

Defects found: `tracking.pipelines.energy_bridge` raises on near-idle
windows (`pipelines.py:1008`, empty c_grid → argmax; upstream fix owed); VK banded system needs 1.53 GiB
per group at 20 s/k=40 and GROWS on the r/2 sibling (bounded centre-crop
scoring added); gridrun workers must be module-level callables.

## 8-channel AVQ run (blind-avq-8ch, 187/187 units, 2026-08-13)

The full 8-channel ego-noise pass does NOT rescue AVQ: 175/187 windows are
octave-suspect (`fvk_ratio_double` < 1.2 or negative half-margin), 103/187
have ridge clearance < 1 dB, and only 2 windows pass the clean gate
(non-suspect AND clearance >= 1 dB):

- BEST `S1_seq2__w000` — clearance 2.74 dB, half-margin +0.43,
  F_VK(2r)/F_VK(r) = 1.23, rates 83/87/88/98 rev/s.
- Runner-up `S2_seq1__w011` — clearance 1.12 dB, half-margin +0.20,
  ratio 1.83, rates 74/75/90/93 rev/s.
- WORST `S2_seq4__w001` — clearance −1.31 dB (below the off-comb null),
  half-margin −0.69, ratio 1.01: the half-rate comb reads BETTER than the
  annotation. The whole S2_seq4 and S2_seq7 recordings sit in this regime.

Median clearance per recording spans 0.05–2.23 dB; no recording has more
than one clean window. Conclusion: the ambiguity is in the signal (weak odd
harmonics at 74–115 rev/s carriers), not in the channel count. AVQ
pseudo-labels therefore need the per-rotor octave instrument (the missing
lever named above) plus acceptance restricted to the clean-gate windows;
mass annotation with the current gates would inherit a ~94 % octave-suspect
rate. The mono run's per-unit artifacts were lost to the daemon disk crisis
(summary survives above); this 8-ch run supersedes it in any case.

Defect (2026-08-13): the corpus plan's "n_rotors=1 for bench" was not
runnable — the vit2dsp ladder is a 4-track unit (two twin pairs + the
spatial joint 2-rotor Viterbi) and `blind_fullrange` embeds it, so both
arms raise on a single track. New `seedvk` arm (blind seed + one coupled-VK
pass, both stages track-count agnostic) is the bench recipe; smoke window
`motor_Motor1_70__w000` annotates 68.5 rev/s with 8.42 dB ridge clearance
and +1.83 dB half-margin at 29 s/window. Branch `blind-corpus` @ 3ffbc26.

## Single-rotor bench (blind-dregon-motor3, seedvk arm, 13/13 units)

The bench validates the octave instruments against physics. All windows
annotate with high confidence (clearance 1.6–10.4 dB, none low-clearance),
and the rates are self-consistent per throttle: 68.4–68.7 rev/s at 70 %
(Motor1 and Motor3 agree), 87.8–88.1 at 90 %, 68.7 with all four motors at
70 %.

The calibration datum: **Motor1_50 annotates 98.0–98.1 rev/s — the DOUBLE
of the plausible ~49 rev/s at 50 % throttle — and the half-margin flags
exactly these three windows at −3.0 to −3.5 dB** (the half-rate comb reads
better than the annotation) while every correctly-annotated window reads
+0.2 to +2.1. At low throttle the shaft fundamental is too weak and the
seed locks onto the second harmonic; the ridge(r) vs ridge(r/2) margin is
the instrument that catches it. Threshold reading: half-margin < 0 ⇒ halve
— which retroactively implies the many negative-half-margin AVQ windows
(median −0.20) are likely doubled annotations, not noise.

`fvk_ratio_double` stays > 1 on the doubled windows (the double of a
doubled annotation is 4× truth — trivially worse), so the F_VK double
ratio alone cannot catch a doubling; the pair of instruments is needed.

## Room2 calibration with the command reference (blind-dregon-calib3, 17/17)

Two prep defects hid the reference on the first two runs: room2 stores the
command track as `motors_command` (not `rps` — fixed with the
`PUBLISHED_RPS_KEYS` chain), and DREGON stamps are Unix-epoch absolute
while prep windows are audio-relative seconds (fixed by re-referencing to
the audio start). With the join in place, the 17 windows split three ways
against telemetry:

- 7 windows CORRECT (all at ~80 rev/s cruise): PIT-MAE 2.1–7.2 rev/s,
  blind/ref rate ratio 0.993–1.026.
- 5 windows HALVED (blind ≈ 39 rev/s on an ≈ 80 rev/s truth): ratio
  0.48–0.51, MAE ≈ 40 — sub-harmonic capture at cruise.
- 5 windows on the takeoff ramp (w000 of each recording): the annotation
  tracks neither the rate nor its half (ratio 0.67–0.93, MAE 33–40).

The calibration finding: **`fvk_ratio_double` separates the populations
PERFECTLY on this set** — correct windows read 1.079–1.214, halved and
ramp windows 1.024–1.054. Threshold **1.065** classifies 17/17. The
existing suspect gate (< 1.2) is directionally right but flags 4 correct
windows; 1.065 is the calibrated cut.

Combined acceptance rule (bench + room2):

- `fvk_ratio_double` < 1.065 — do not trust; if the ridge at 2r clears,
  the annotation is likely HALVED (double it and re-score); on a ramp
  window it simply fails.
- half-margin ≤ −1.5 dB — annotation likely DOUBLED (bench Motor1_50 reads
  −3.0 to −3.5); halve it and re-score. Mild negatives (to ≈ −0.8) occur
  on CORRECT cruise windows and must not trigger halving.
- otherwise accept.

Applied retroactively to AVQ 8-ch: median `fvk_ratio_double` 1.044 < 1.065
— most AVQ windows fail the calibrated gate too (consistent with the 94 %
suspect rate), and the clean tail (ratio ≥ 1.2) is exactly the windows the
per-unit triage already surfaced.

Caveat: the blind ladder ties the four tracks to a near-common rate while
the command reference spreads 75–86 rev/s across rotors, so the 2–4 rev/s
PIT-MAE on correct windows is dominated by per-rotor mismatch, not by
common-mode error.

## Full single-motor bench (blind-bench-all, seedvk arm, 46/46 units)

The first bench pass saw 5 of the 21 DREGON bench recordings. The full pass
sees all of them: `motor_Motor{1-4}_{50,60,70,80,90}` plus `motor_allMotors_70`,
43 single-motor windows and 3 combined ones. The throttle setpoint is in the
recording name, so this corpus has an EXTERNAL physical reference that is not
acoustic and not a label: rotor speed must increase monotonically and smoothly
with throttle.

Three readings, all on the 46 windows:

- The annotation reads 68.5 / 78.4 / 88.2 rev/s at 70 / 80 / 90 % throttle,
  and 98.1 / 117.6 rev/s at 50 / 60 %. The last two are the DOUBLE of the
  linear extrapolation of the first three.
- **The half-margin separates the two populations by SIGN, 46 of 46.** Doubled
  windows read -5.20 to -0.97 dB. Correct windows read +0.23 to +4.16 dB.
- Neither other instrument separates them. Ridge clearance overlaps (doubled
  0.53-2.78 dB against correct 2.35-13.32 dB), and `fvk_ratio_double` overlaps
  fully (1.135-2.076 against 0.633-2.171).

**Threshold correction.** The 13-window bench pass put the doubling cut at
-1.5 dB. With 46 windows the cut is **0 dB** on a SINGLE-TRACK annotation: two
doubled windows read -0.97 and -1.44 and the -1.5 dB rule misses both. The
-1.5 dB value stays correct for the GLOBAL margin of a 4-track annotation,
which is a mean over four rotors and is therefore diluted. Read the cut per
track, not per window.

**The physics acceptance test.** Halve every window whose half-margin is
negative, then fit rate against throttle over the 43 single-motor windows:

    rate = 0.975 * throttle_percent + 0.37 rev/s,  R^2 = 0.99788,
    residual RMS 0.63 rev/s

Per-throttle means are 49.11 / 58.98 / 68.53 / 78.37 / 88.17 rev/s and the fit
predicts 49.13 / 58.88 / 68.63 / 78.38 / 88.13. The residual inside one
throttle (0.45-0.82 rev/s) is motor-to-motor spread: Motor2 runs slow and
Motor4 runs fast at every setpoint. So the corrected annotation of this corpus
is right on all 43 windows, and one non-acoustic instrument says so. The
cross-window repeatability is 0.004-0.135 rev/s.

Job `blind-bench-all-0cb28a`, 46 units, 3.5 min wall on 14 workers.

## KAIST rotating machine — the known-rate control (blind-kaist-c36d84, 20/20)

`KAIST-rotating-acoustic` is 5 mono 51.2 kHz bench recordings of a rotating
testbed at a dataset-stated 3010 RPM (50.167 rev/s), with a bearing fault and a
severity in each file name. It is the only corpus here with an absolute rate
statement and no drone in it, so it tests the annotator outside its design
domain.

- `0Nm_BPFI_10` annotates **50.298 rev/s on all 4 windows** — 0.26 % above the
  stated nominal — with 11.0-11.6 dB ridge clearance, +4.3 to +5.0 dB
  half-margin, and a within-window range of 0.037 rev/s. This is the cleanest
  annotation in the whole campaign.
- `0Nm_BPFI_03` annotates 100.62 rev/s (the double) on all 4 windows. The
  clearance falls BELOW the off-comb null (-0.26 to -0.02 dB) and catches it,
  but the half-margin does NOT (+0.06 to +0.75).
- `0Nm_BPFO_03` and `0Nm_BPFO_10` annotate 91.33 and 91.70 rev/s with 4.7-5.9 dB
  clearance and a positive half-margin. That rate is 1.82 times the nominal and
  is not a simple multiple of it, and it repeats across two severities of the
  same fault. The annotation locked onto a real periodicity that is not the
  shaft. Both windows PASS the calibrated gate, so this is a FALSE ACCEPT.
- `0Nm_Normal` annotates 99.87 rev/s (doubled, caught by a negative clearance)
  on 3 windows and 119.15 rev/s on one, at the top edge of the 30-120 rev/s
  seed scan.

Reading: on a machine with strong non-shaft periodicities the octave
instruments are not enough, because the competing comb is not an octave of the
truth. The corpus plan therefore keeps industrial rotating machines in a
separate class with a rate prior, and never accepts one on the acoustic gates
alone.

## Room2 again, with the per-rotor instrument (blind-room2-pr-2fc857, 17/17)

The room2 pass was repeated on the same 17 windows to test the NEW per-rotor
octave instrument against the command reference. It reproduces the calibration
exactly: 7 correct windows (blind 78-83 rev/s against a reference of 78-83,
PIT-MAE 2.1-7.2 rev/s), 5 halved windows (blind 39-42 on an 80 rev/s truth) and
5 takeoff-ramp windows (`w000` of each recording).

- `fvk_ratio_double` reads 1.079-1.214 on the correct windows and 1.024-1.054
  on the halved and ramp ones. The 1.065 cut classifies 17 of 17 a second time,
  on an independent run of the same protocol.
- The per-rotor half-margin does NOT false-flag a correct window. Its minimum
  over the four rotors reads -0.90 to -0.26 dB on the 7 correct windows, so the
  -1.5 dB cut has 0.6 dB of headroom.

**The two cuts are not the same number, and the reason is physical.** On the
single-motor bench a doubled track reads -0.97 dB or lower and a correct track
reads +0.23 dB or higher, so the cut is 0. On a quadrotor the four combs sit
within 10 rev/s of each other, so a halved carrier still collects teeth that
belong to the neighbours and the margin is compressed toward zero. Use 0 for a
one-source annotation and -1.5 for a rotor inside a four-rotor comb.

Limit of the instrument: it needs the halved carrier to stay above the fitness
admission rate. On a window whose annotation is already halved (39 rev/s) the
test carrier is 19.5 rev/s, no cells are admitted, and the reading is empty.
Those windows are the ones `fvk_ratio_double` rejects anyway.

## SPCUP19 single-rotor takes (blind-spcup-single-9ab55c, 8/8)

The AGH team's `ego-noise/single_rotors` set is 8 short mono takes of one rotor.
Every window annotates with a positive half-margin (+0.52 to +1.81) and 2.8 to
5.9 dB of ridge clearance, at 66.3 / 77.7 / 79.7 / 96.6 / 96.8 / 97.4 / 97.8 /
113.8 rev/s. There is no telemetry, so the readings are the only judge.

Two of the eight fail the `fvk_ratio_double` gate (1.020 and 1.021), and they
are the two SLOWEST takes. A ratio near 1 says the odd harmonics carry little
energy, which on a correct annotation should not happen. The two candidate
explanations are a weak odd-harmonic set on a small rotor, and an annotation an
octave low. The second one is worth a test, because this rig is small: the top
take already reads 113.8 rev/s against a seed scan that stops at 120 rev/s
(`SeedConfig.scan_hi`), so the scan ceiling may be binding for this drone.
Widening the scan is not free — the calibrated configs are frozen — so it needs
its own calibration pass and is listed as an open item, not a fix.

## The corpus plan (2026-08-14 revision)

### What is annotatable

Every raw source is a `src/data_processing/sources/` entry; only the ones
published in the `tdframe-v1` layout can be streamed by the driver.

| Class | Corpus | Size | Arm | Tracks | Regime |
|-------|--------|------|-----|--------|--------|
| A | `DREGON-frames` motor bench | 21 recordings, 46 windows | `seedvk` | 1 | static bench, constant throttle |
| A2 | `KAIST-rotating-acoustic` | 5 recordings, 20 windows | `seedvk` | 1 | machine bench, stated 3010 RPM |
| B | `DREGON-frames` in flight | 10 recordings, ~35 windows | `fullrange` | 4 | hover to free flight, telemetry |
| B2 | `michaels-frames` FLY124/125 | 2 recordings, ~75 windows | `fullrange` | 4 | free flight, measured telemetry |
| C | `SPCUP19-egonoise` static / hover / single rotor | ~60 windows | `fullrange` / `seedvk` | 4 / 1 | 10 different rigs |
| D | `DroneAudioSet` drone-only | 2313 clips, sample ~96 | `vit2dsp` | 4 | rig mounted, static, 2 throttles |
| E | `AVQ` | 12 recordings, 187 windows | `fullrange` | 4 | onboard, free flight |
| F | `new-drone-noises` | 103 DJI flights | not blind | 4 | free flight, RPM in the DatCon logs |
| G | refused | `AeroSonicDB`, `drone_audio`, `drone-detection-samples`, `HornBase`, `MIMII-DG`, MIMII slider/valve | — | — | mono far field, 1 s clips, or not a rotating source |

Class F is the biggest prize and is NOT a blind-annotation task. The 103
flights carry per-rotor RPM in their flight logs, and the alignment procedure
that calibrated FLY124 and FLY125 exists (`scripts/michaels_calib/`). Building
that source gives real labels, not pseudo-labels.

MIMII fan and pump are held back for one reason: an industrial fan turns below
the seed's 30 rev/s floor, so the scan must be widened and re-calibrated first.

### Order, and what each step buys

1. **A and A2 first.** Both have a non-acoustic reference (the throttle
   setpoint, the stated RPM), so they calibrate the octave instruments against
   physics. DONE — see the two sections above.
2. **B and B2 second.** Both have telemetry, so they calibrate the gates
   against a label. B is DONE for room2; room1 (which carries
   `motors_measured`, not a command) and michaels are open.
3. **C third**, because static and hover windows on 10 different rigs test
   whether the calibration transfers off the DREGON and DJI airframes. The
   single-rotor takes are DONE.
4. **D fourth.** It is the largest clean multichannel drone corpus and its
   throttle is constant, so one rate per clip is enough. Sample the design
   cells rather than the clips.
5. **E last**, because AVQ is the hardest case in the corpus and its verdict
   depends on the per-rotor instrument.

### Acceptance gates

Four gates, all label-free. A window is accepted only if it passes all four.

- **G1, octave low or ramp**: `fvk_ratio_double` >= 1.065. Calibrated on DREGON
  room2 twice, 17 of 17 both times.
- **G2, doubling**: the half-margin. Cut 0 dB for a one-source annotation
  (bench, 46 of 46), -1.5 dB for a rotor inside a four-rotor comb (room2, 0.6
  dB of headroom). A window that fails G2 is HALVED and re-scored, not
  discarded: the bench shows the halved value is the right one.
- **G3, weak comb**: ridge clearance over the off-comb null > 1 dB. This is the
  gate that catches the KAIST doubled windows, which G2 misses.
- **G4, continuity**: consecutive windows of one recording must agree inside
  their overlap to better than 1 rev/s, and must not jump by a factor near 2 or
  0.5. No label is needed and no extra demodulation is needed.

### Compute

Measured, per window, on `uni-cpu`: four-track `fullrange` on 8 channels and a
20 s window costs 410 s of CPU (ladder 275, F_VK 120, ridge 10, per-rotor
octave 15). One-track `seedvk` costs 60 s on 8 channels and 8 s on one channel.

Class totals: A 0.8, A2 0.04, B 4.0, B2 8.5, C 7, D 11, E 21 CPU-hours. The
whole plan is about 52 CPU-hours, which is one `uni-cpu` job of 16 cores and
3.5 hours of wall time. The driver is restartable per unit
(`utils.gridrun`), so a partial run is never lost.

### Expected failure modes, and what each one needs

1. **Doubling at low throttle** (bench 50 and 60 %, KAIST). The shaft
   fundamental is too weak and the seed locks the second harmonic. Caught by
   G2, corrected by halving.
2. **Sub-harmonic capture at cruise** (room2, 5 of 17). Caught by G1, not
   correctable — the window is refused.
3. **Ramp windows.** Both octave instruments fail because no single rate fits.
   Refuse, or cut the recording so the ramp is its own window.
4. **A non-shaft comb wins** (KAIST BPFO, 8 windows). No acoustic gate catches
   it, because the competing comb is real and is not an octave. Needs a rate
   prior per rig class.
5. **The rate is outside the 30-120 rev/s seed scan.** Small racing rotors and
   industrial fans both sit outside it. Needs a widened scan and its own
   calibration pass.
6. **Mono far field.** The comb decoheres and the source count is unknown.
   Refused by default (class G).
