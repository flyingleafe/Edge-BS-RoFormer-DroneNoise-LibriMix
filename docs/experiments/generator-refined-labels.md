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

## Full-dataset arms (gen_m1_refined / gen_m2_refined_perrotor, 2026-08-12)

DREGON+Michael's 8-mic stream with the refined DREGON sidecar,
`checkpoint_every=1`, comb-aware selection over all epochs (DREGON chunks,
paired reading; curves `results/perrotor_probe/m_arms_epoch_curves.json`):

- `gen_m1_refined` (per-drone only), comb-best ep0: 4.68/0.99/0.67/−0.20 dB —
  mixed-drone training still costs DREGON mid-k sharpness (DREGON-only r1
  held 4.22 at k10-24).
- `gen_m2_refined_perrotor`, comb-best ep14: 3.78/2.59/−0.06/**+1.05** dB —
  the best high-k combs of any arm in the campaign; with two drones the
  per-rotor deltas earn their keep.
- The monitor pathology reproduced a third time: corr(val mrstft, high-k
  comb) = −0.37 (m2); best-by-mrstft picks worse epochs in both runs.

Production pick: `r2://ml-data/artifacts/gen_m2_refined_perrotor/checkpoints/`
`ep14_mrstft_2.1149.ckpt` (registered in the generator lab as
"deep/m2 full data + per-rotor dz, refined"). Caveat: the comb readout covers
DREGON only; Michael's-side quality needs its own instrument before this
checkpoint is used for FLY-drone work.

## gen_m3_refined_all_perrotor: refined labels on BOTH rigs (2026-08-20)

The FLY124/FLY125 refined sidecars are wired into the training stream via
`michaels_rps_override_dir` (commit 52e9a26), so no arm of the data carries
unrefined labels. Same recipe as gen_m2 otherwise; 39 epochs on gpushort
(job `gen-m3-refined-all-ce7ea5`); every epoch checkpoint on R2.

**Checkpoint-selection sweep** (all 39 epochs x both rigs, per-harmonic comb
readout via the extended `scripts/eval_gen_comb_real.py --rigs
dregon,michaels`; curves + notes: `results/gen_m3_sweep/`):

- The monitor (val mrstft, mode max) picks **ep30** (6.739). On the comb
  readout ep30 ranks 17th-35th of 39 in every cut.
- Comb-best on the HELD-OUT chunks (`--split-filter valid,boundary`): **ep19**
  (epochs 16-19 cluster). Comb-best on in-flight audio: **ep7-9**.
- The monitor pathology reproduces a fourth time and is now measured on
  in-flight audio: Spearman(monitor, |dLogMag| k10-80) = **+0.35** (p 0.03;
  michaels alone +0.47) — a better monitor score means a worse comb.
- **The held-out split is warm-up audio.** `val_at_start` (first 10 %) of all
  three recordings is pre-takeoff/ground idle, so the training monitor was
  scored mostly on ground-idle audio (teeth 1.6-3.6 FFT bins apart there vs
  9-10 in flight). Any future generator run needs either a flight-time valid
  split or an in-flight instrument for selection.
- `ptf_delta_db` rises monotonically with the monitor (dregon +0.24 dB at ep7
  -> +1.12 at ep36; michaels +2.14 -> +3.54): late epochs build an over-sharp
  comb on a too-low floor. The michaels comb is 2-3.5 dB too peaky at EVERY
  epoch — the first Michael's-side comb reading (previously uninstrumented).

Production pick: pending the visual/audial A/B (real vs ep19 vs ep9 vs ep30)
the user requested before any downstream use.

## m3cur curriculum verdict (2026-08-21/22)

All six runs finished on the frozen full-envelope valid
(`DREGON-LM-V4-michaels-valid-full`, monitor val/mse, best-over-epochs):

| arch | s1 (gen-only) | s2 (curriculum) | real-only control | dMSE |
|---|---|---|---|---|
| scv2 | 325.5 | **28.4** / MAE 3.16 | 52.5 / 3.98 | **−46 %** |
| transformer-IF | 316.9 | 38.6 / 3.41 | 42.3 / 3.76 | −9 % |
| uni_gru128 | 275.4 | 51.4 / 4.14 | 59.2 / 4.24 | −13 % |

The generated-first curriculum (gen_m3 ep30 + static comb 50/50, full-flight
excitation, freq_scale p=1.0 + time-warp + gain/polarity from sample 1, then
the identical real stage) beats its schedule-matched real-only control on
every architecture. Caveat: the transformer's non-schedule-matched v1-range
arm (g2_if_freqscale, 37.6) ties its curriculum row.

**Where the gain lives** (per-frame PIT regime probe, both metrics in
`results/m3cur_regime_probe/`): stopped rotors improve on every arch (scv2
silence MAE 11.8 → 4.8 rev/s, −70 % MSE — nearly the whole aggregate gain);
ramps/warm-up mixed (unigru −30 %, transformer +21 %); mid-flight ±10 %.
The curriculum buys coverage of the regimes real data lacks (silence above
all), at ~zero cruise cost. Ablations m3abl_{comb,gen,mixed} (see below in
the m3abl doc stubs) test whether the neural generator, the comb, and the
staging are each necessary; first signal: comb-only s1 flatlines the
transformer (train 9.7, real-valid ~2500 — no transfer) while unigru/scv2
comb-only s1 beat the mixed-pool s1 readout (183.7/204.0 vs 275/326).
