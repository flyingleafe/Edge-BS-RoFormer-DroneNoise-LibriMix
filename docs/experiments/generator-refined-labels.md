# Generator label A/B on real DREGON: refined telemetry vs original telemetry

**Date**: 2026-08-12 · **Branch**: `tracking-opt` (commits 7665afa..0f5749f) ·
**Status**: DONE — verdict below.

The real-data continuation of the synthetic phase-7 batch
(`generator-label-sensitivity.md`). Phase 7 proved on synthetic combs that a
constant RPS label bias alone collapses the generator's high-k lines. This
batch tests the mechanism on real DREGON audio: train the positional harmonic
generator on DREGON with (a) original telemetry, (b) telemetry × 0.99458 (the
phase-7 constant fix), (c) L-BFGS-refined trajectories, each ± per-rotor
sub-embeddings, and read the harmonic combs of the trained models against the
real recording.

## Refined labels

`scripts/refine_dregon_rps.py` runs `tracking.fitness_vk.optimize_trajectory`
(L-BFGS on the profiled coupled-VK residual, cubic B-spline, k ladder 5→40,
4 microphones) window-by-window (16 s, hop 12 s) over
`free-flight_nosource_room1` — the ONE DREGON recording with `motors_measured`
that the generator stream uses. Acceptance is per rotor (|move| ≤ 3 rev/s and
the window objective must improve); rejected rotors keep telemetry; the stitch
cross-fades overlaps. Sidecar: `src/data_processing/refined_labels/`.

- Stitched cruise shift **−0.554 %** (per-rotor −0.25/−0.61/−0.70/−0.68) —
  inside the 6d ridge CI [−0.877, −0.533] and near the L-BFGS oracle −0.596.
- Per-rotor rejections: rotor 1 in the takeoff window, rotor 0 at 41.5 s
  (alias capture, −5.95 %). Every window improves the F_VK objective.
- Jobs: `refine-dregon-rps4-318b3f` (uni-cpu, 2 workers; 5 workers OOM at
  48 GB — the k=40 autograd is ~12 GB/worker even at 4 mics), stitch local.
- Loader knobs (`noise_rps_dataset`): `dregon_rps_override_dir` (sidecar) and
  `dregon_rps_scale` (constant). Stamps outside the sidecar span, stamps with
  zero telemetry, and stamps within one grid step of a motor stop keep the
  original telemetry — the 0.032 s grid smears the sub-frame shutdown step
  (a stopped motor read 75 rev/s before this guard).

## Arms

All arms: DREGON-only 8-mic stream (`noise_rps_dregon_stream_multimic*`,
`channel_policy: all`), `multiscale_stft` loss, `gen_v1_recal_mm`
hyperparameters, `drone_names: [dregon]`. The mic-position → audio-channel →
propagation chain was audited end to end before training: no permutation
anywhere; DREGON geometry (the 180° flip) is TDOA-validated.

| arm | labels | embeddings | job | epochs |
|---|---|---|---|---|
| `gen_r1_orig` | telemetry | per-drone | `gen-r1-orig-9cba41` | 11 |
| `gen_r1_scaled` | × 0.99458 | per-drone | `gen-r1-scaled-f5a043` | — |
| `gen_r1_refined` | refined | per-drone | `gen-r1-refined-a1e433` | — |
| `gen_r2_orig_perrotor` | telemetry | + per-rotor δz | `gen-r2-orig-perrotor-7f9ce8` | — |
| `gen_r2_refined_perrotor` | refined | + per-rotor δz | `gen-r2-refined-perrotor-d118a4` | 15 |

## Readout

`scripts/eval_gen_comb_real.py` (new): per-k comb metrics against the real
recording, 14 × 4 s free-flight chunks × 8 mics. Anchors: estimator null
−0.78 dB (a peak-to-floor there = no measurable tooth); real audio along the
refined tracks: +1.61 / +0.90 / −0.79 / −1.05 dB per band (real teeth above
k≈25 decohere over 4 s — the model renders are deterministic, so PTFgen reads
the model's line structure with more sensitivity than PTFreal has for the
recording). Each arm is conditioned on its own training labels; fidelity
(dLogMag) is measured along the refined tracks for every arm.

Peak-to-floor of the GENERATED audio (dB; null −0.78):

| arm | k1-9 | k10-24 | k25-49 | k50-80 |
|---|---|---|---|---|
| `gen_r1_orig` | 0.43 | 1.69 | **−0.78 (null)** | −1.06 (null) |
| `gen_r1_scaled` | 0.66 | 4.49 | 0.33 | −0.76 (null) |
| `gen_r1_refined` | 1.41 | **4.22** | **0.96** | **+0.13** |
| `gen_r2_orig_perrotor` | −0.56 | 0.66 | −0.91 | −1.18 |
| `gen_r2_refined_perrotor` | 0.50 | 0.83 | 0.13 | −0.02 |
| `gen_v1_recal_mm` (old, mixed drones) | 1.12 | −0.17 | −1.08 | −1.23 |

Fidelity along the refined tracks (comb-masked |Δ log-mag|, dB, lower =
better): `gen_r1_refined` is best in ALL four bands (8.50/8.24/8.68/8.70);
`gen_r1_scaled` is WORST at k10-49 (9.22/9.79) — its lines are strong but sit
off the true comb. mrstft anti-correlates with comb quality across arms
(scaled has the lowest mrstft and the sharpest mid-k lines), as the wind
post-mortem predicted.

## Verdict

1. **Label precision was a real and dominant cause of the mid/high-k washout.**
   With original telemetry the DREGON-only generator has NO measurable tooth
   above k≈22. With refined telemetry it holds teeth above the null through
   k=80, and it is the only arm that does. The k-resolved ladder is
   refined > scaled > original from k≈18 upward.
2. **A constant scale is not enough** — the generative counterpart of the 6d
   "not a pure scale" verdict: the scaled arm matches refined at k10-24, dies
   at k50-80, and its lines sit measurably off the acoustic comb (worst
   dLogMag at k10-49).
3. **Per-rotor sub-embeddings hurt line sharpness in both label conditions**
   (r2 ≪ r1 at k10-24: 0.83 vs 4.22 refined, 0.66 vs 1.69 orig). Consistent
   with the 07-17 finding that per-rotor deltas are not a win; do not adopt.
4. The old DREGON+Michael's 8-mic model (`gen_v1_recal_mm`) is the weakest of
   all six at k ≥ 10, so the mixed-drone setup AND the labels both contributed
   — but the labels bind harder: even DREGON-only training washes out above
   k≈22 on original telemetry.

Caveats: one seed per arm; early stopping selects on the idle-heavy
`val_at_start` split (both members of each pair share the convention, and the
ladder replicates across the r1 and r2 pairs); the eval chunks are training
audio (a fitting-capacity comparison between arms, not generalization —
DREGON has one instrumented recording, so a held-out comb readout does not
exist at cruise).

Artifacts: `results/gen_comb_real/` (per_k.csv, summary.csv, per-k plots,
spectrogram illustrations), sidecar + report in
`src/data_processing/refined_labels/`.

## CORRECTION (2026-08-12, instrument audit)

A user challenge ("real PTF below the null sounds like an error") exposed two
readout defects; probes in `results/perrotor_probe/{self_ref_ptf,
real_comb_vs_window,real_order_avg_contrast}.json`:

1. **Below-null real PTF = floor contamination.** The ±0.5 f0 floor slots
   catch the smeared tails of the rotor's own line (nothing excises the own
   line from its floor) and, for k ≈ 33-84, the twin rotor's line
   (k·Δf ≈ 0.4-0.5 f0), which the ±15.6 Hz excision misses once displaced or
   smeared. A line-free band reads AT the null; these floors were inflated.
2. **The high-k cross-arm ladder was partly the same artifact.** The paired
   reading pins floor slots and excision to the REFINED carriers, so the
   scaled arm's displaced twin lines (k·f0·0.55 % > 15.6 Hz above k ≈ 40)
   landed raw in its floor slots. Self-referenced re-scores (floors and
   excision on each arm's own carriers): refined 1.41/4.22/0.96/0.13,
   scaled 0.68/5.40/1.40/−0.02, orig 0.44/2.64/0.23/−0.60 per band.
   **Withdrawn**: "refined is the only arm with teeth through k=80".
   **Standing**: raw telemetry washes out the comb from k≈10 up (2.6 dB at
   k10-24 vs 4.2-5.4 for corrected labels; 0.2 vs 1.0-1.4 at k25-49), and the
   refined arm alone places its lines on the acoustic comb (fidelity readout,
   which is band-only and floor-free).

**Where real teeth actually end (this recording, 16 kHz).** Order-averaged
tooth contrast along refined tracks (maximal time integration): 6.76 dB
(k1-9), 1.36 (k10-24), 0.13 (k25-49), 0.01 (k50-80). Teeth above k≈25 are not
STFT-separable at 128 ms windows — smear fills the inter-tooth valleys —
although comb-locked energy exists there (ridge/F_VK instruments). Since
every MultiScaleSTFT scale is ≤ 2048 samples, **no arm receives a training
signal for sharp teeth above k≈25 on this data**: all arms sitting at the
null at k50-80 is the correct fit to what the loss can see, not a residual
label failure.
