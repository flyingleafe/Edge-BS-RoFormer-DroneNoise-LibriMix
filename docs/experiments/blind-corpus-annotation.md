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
