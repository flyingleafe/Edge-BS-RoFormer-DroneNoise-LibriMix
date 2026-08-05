# Work inventory since last slides

- generated: 2026-08-03T22:15:21+01:00
- boundary artifact: writing/slides/2026-07-27_dcunet-generalization
- boundary commit: 6fa7886 2026-07-28 Deck: VK reference rows, speed/accuracy final slide, param counts
- HEAD: 4c07be1 2026-08-03 Stage-C/D iteration test: k-scaled bands, IAVKF adaptation, WP18 weights

> Numbers in docs below may predate later fixes — always cross-check the
> newest report/doc before quoting. Sync results before analysis (Rule 5).

## Commits (newest first)

```
4c07be1 Stage-C/D iteration test: k-scaled bands, IAVKF adaptation, WP18 weights
fffca77 GOALS: the likelihood line is not fruitful as applied
cf8e58a Generator lab: fix the two bugs the smoke run actually found
add8c73 Fix the generator lab's real bugs and add a headless smoke driver
90bd241 GOALS: record the wind-noise-channel objective and why it is still unresolved
27836c2 Review fixes on the generator PR: lint, format, and its own type debt
4c84259 Sweep the spatial weight: the 0.05 arms were degenerate, not informative
cf841df Probe the hybrid arms' wind share before trusting the refutation
4eb12ca Wake gating refuted on the fair test; wind channel does not help
4398a9c Hybrid objective: keep the coherent mean, add the coherence term
0f0cb3e Generator lab: one notebook for every noise generator
3861ed6 Replace the ill-posed spatial control with a wake-model control
281566c Register the spatial arms in the eval
a7841e5 spatial_stats returns a superset, so the metrics still have audio
9926cb8 Fix the two defects the spatial validator caught
cec15a7 Wire the spatial likelihood: model, codec, task contract, configs
364d97d Spatial (cross-microphone) likelihood: the objective that can see wind
321bce1 The wind channel is inert: the per-microphone likelihood cannot see it
8c51cba Probe the wind channel's share of predicted variance
92289e1 Add a probe for what the wind channel actually learned
36f1c4f First results: the likelihood objective wins decisively
230e000 Fix the missed noise_psd rename in the eval script
f8ef427 Add the magnitude-loss multi-observer baseline (the missing cell)
fd69681 Teach the config validator about optim.monitor: val_loss
949a1a7 Record the discarded attempt-1 runs as evidence for the metric bias
804403c Monitor the objective, not a metric that is biased against it
58a8a45 Score the generator variants on all microphones, not just channel 0
fee4c13 Document the likelihood objective and the channel-policy choice
122652e Record how the likelihood was made trainable in the batch doc
18ea4cf Make the spectral likelihood trainable: power interface, warm start, beta-NLL
37b32c8 Declare the distributional outputs in the task contract
1c2c07d Fit the stochastic branches by likelihood, not by realization
4b57c57 Michael's telemetry recalibration: port constants, repin frames, retrain arms
1837686 Review pass on the data-pipelines refactor: close the migration gaps
f84bbe4 Post-merge: restore load_michaels_timeframe, fix raw-root paths, pre-commit
06c642c WP21: visibility correction + session resume state (goal, jobs, ladder)
eb7c7b4 WP21: exact per-rotor lattice DP — runs 1-2, masking defects and fixes
d3d4387 rotor_dp: min surviving-teeth floor + probe --k-max
ce151b2 WP20 closure: joint beam bit-identical under shared peaks + comb price; line closed
1ec48b6 rotor_dp: dilate the claim mask to the Hann mainlobe (+-2 bins)
711aa13 rotor_dp: exact single-rotor Viterbi lattice + claim-masked residual emission
22ff656 jb_probe: cost gate takes a beam config, and the decisive shared-peaks arms
63ba51d jb: claim_q fixes the objective on 2 of 6 windows — and the search cannot use it
88469a5 jb: the price of a comb is `claim_q`, and normalisation provably cannot be it
c2a503c jb: "score what each rotor uniquely explains" is structurally wrong — dropped
c05ce5a jb: absolute (MAD) normalisation — the last term the gate says is still wrong
50b7560 WP20 doc: the pooling sweep numbers and the QUALITY x SHARE union
e306036 jb: quantile pooling in the union, and the cost gate that will judge it
4804ddb jb: the objective is wrong, not the search — measured on 6/6 windows
bb624f3 jb_probe: tell a broken objective from a losing search, and measure the emission ceiling
3364e75 WP19: union-comb emission fixes the subharmonic collapse; tracker still loses
cf18866 Paper: tighten the WP18 correction in the abstract
a9c366c jb_sweep: --build-preps, without which every cluster unit dies silently
d4de7e2 Paper: WP18 — the phase-noise model, measured (floor refuted)
3a43156 joint_beam: union-comb emission + assignment-level rotor-band prior
d6f6426 WP18 measured: the weight shape survives, the rank-one floor does not
2a6e278 WP18: fix the cluster path — manifest call, module-name collision, DREGON pull
46e24b9 WP18: gate the saturation report on a resolved common term; cut demod memory
b76b4df WP17: add the fullrange_init control — the shared-shape constraint was never the binding one
8dd7ca7 WP18: measure the across-harmonic covariance of the rate opinions
576e2ea Paper: WP15 re-score — the chain ties on DREGON, wins on FLY124 cruise
4872775 WP17 measured: the OU prior is the right form but 100x too weak, and the seedless joint search collapses onto comb subharmonics
6fe5127 WP15 re-scored: seed flip fixed (w03 7.273 -> 0.978), M2 gate lands
b1aa3b0 Record the WP10 anti-circularity controls (paper Table V source)
6891a34 Paper: anti-circularity controls + promote the IAVKF finding
09badda Paper: restructure around capture range; honest negative result
36bf6db WP17: joint 4-rotor beam-search tracker with an OU control-mode prior
325944d arm R: dedup the ACCEPTED residual candidates against each other
f940c35 WP16: the joint-tracker mode prior is not supported by the telemetry
a14c84b arm R: rotor-band SPAN bound on residual combs — fixes the FLY124 w03 seed flip
6788415 refine lab: M2 proposal dump + move gate; refine_gate_probe diagnostics driver
5968454 beat-vk: re-scored on the corrected protocol; the FLY124 loss is a seeding flip
7d28379 beatvk_rescore: assemble the seed-swap build AFTER the grid, not before
1c06d02 beatvk_rescore: add the seed-swap control build
d3291bc rps_refine_lab: namespace the blind_seed cache by prep build
2129c47 beatvk-valid-raw republished on the corrected FLY124 labels + re-score driver
d2480fb Validate the refit scales: FLY124 value residual -0.178 -> -0.054 rev/s
efda3bf michaels-frames republished at the refit scales + docs supersession
13abb43 michaels: rev/s scale refit on 13 windows (FLY124 1.00698, FLY125 1.00706)
19b034e WP14: per-rotor offset vs lag settled — one global scale, FLY124 magnitude 0.14pp high
752bc22 michaels_calib: verdict tables for the per-rotor offset-vs-lag sweep
c58ccc3 michaels_calib: per-rotor offset-vs-lag sweep over ALL cruise windows
0b394cf Michael's calibration validated on the cluster (job python-0445f6)
8d91e4c beat-vk: flag every FLY124 scoreboard column as stale after the telemetry recalibration
40a13d6 Docs: michaels calibration blast radius + online-mixing note
8856b0b Republish michaels-frames with the calibrated telemetry (@a7a951b94808)
e37e09a Michael's telemetry: apply the measured timing + rev/s calibration
f700a5d Michael's telemetry calibration: cluster sweep + RPS refinement campaign
db50753 CKLA triton tests: isolate CUDA equivalence in a subprocess
7612787 refactor: data layer — dload pipelines, sources registry, pipeline-native mixing
b146e9b Parallel CKLA scan demoted to opt-in; NaN-pred guard in PIT metric align
efe491a CKLA: fused Triton scan+readout kernel (adapted from kla-loglinear flat KLA)
e6eee77 CKLA: parallel associative scan (Mobius lambda pass + linear eta pass)
67b995a 8s ft: batch 16->32 (CKLA scan is launch-bound at T=250; batch is ~free throughput)
27aa0f2 data: all derived datasets as dload pipelines; delete the bespoke creators
580d06c data: uniform sources/ registry — DREGON + michaels as ordinary entries
833fd3d data: delete dead dataset code (external_recordings, dataset.py, v3 creator)
8cfb7a1 8s arm attempt 2: warm-start ft experiment + 8s/S3bc results in beat-vk doc
9c008a8 Paper: full expansion with VK-improvement program (12 pp)
8b630fe vk_eval: register ckla_phonly_long_best (converged phase-only run)
a02d855 neural_reanchor: bases+detrended-neural arms (negative result, tool kept)
7f546b1 Beat-VK: VK-improvement program conclusion (performance + ceiling, both delivered)
b894ed3 Beat-VK doc: fullrange v2 + pi_kalman rows, displaced-comb finding
df231a0 blind_fullrange v2: rumble-band energy bridge + idle re-scan + DP trust gates
f5fb0e3 pi_kalman: per-iteration band_hz schedules + DREGON displaced-comb finding
a8ac6dc pi_kalman: pair-coupled joint twin mode + protocol runner
8379e97 pi_kalman: phase-increment ML frequency tracker (per-harmonic random-walk model)
3ff80ea Ladder: S3b/S3c decoherence-decomposition rungs + coherence-time/off-comb metrics; beamform lock probe
72db744 blind_fullrange arm: coarse full-range Viterbi + BPF octave check + energy bridge
dc3555f Beat-VK: full fixed-protocol scoreboard (VK arms + neural rows)
95de7f9 ckla_phaseonly_8s: 8s-window arm (anchor-integration lever, next increment after 4s win)
2aafecc iter_warp: iterated angular-resampling refinement (generalized demodulation)
a4e88ca vk_phase_validation: S4 falls back to dload-capable loader without local DREGON
5afc392 Archive neural-VK-parity goal (paused for VK tracking improvement program)
c86dfe9 vk_phase_validation: phase-recovery ladder S0-S4 (synthetic -> motor -> free-flight)
3233729 vk_eval registry: fs_v2 scoreboard arms (scv2, unigru128)
31adb07 beatvk_vk_arms: VK tracker rows driver for the fixed raw protocol
fc1bbe9 Fixed raw validation protocol: beatvk-valid-raw dataset + unified scorer
95e0c80 Scoreboard arms: unigru128/scv2 under fs_v2 regime + e12_unigru128 registry key
035bdd7 vk_eval registry: beat-vk retrain arms (avq, 4s)
89471ed RPSMSELoss: promote mixed AMP dtypes before F.mse_loss (fp16 pred vs fp32 target backward crash)
dafb7af Conditional RPS refiner: CKLA cond model + corruption sampler + eval hook
1f00968 Beat-VK R2: AVQ-egonoise-vkrps pseudo-label dataset + phaseonly avq/4s arms
2886bbc Neural-seeded VK: pilot script + --seed neural mode in vk_blind_sweep (R1)
fa5053f vk_pseudolabel: blind-VK pseudo-label annotator for unannotated egonoise (R2)
cbdc275 Beat-VK campaign design doc (R1 hybrid / R2 pseudo-labels / R3 SSL / R4 convergence)
```

## Experiment configs (conf/experiment/)

```
  A	conf/experiment/ckla_phaseonly_4s.md
  A	conf/experiment/ckla_phaseonly_4s.yaml
  A	conf/experiment/ckla_phaseonly_8s.md
  A	conf/experiment/ckla_phaseonly_8s.yaml
  A	conf/experiment/ckla_phaseonly_8s_ft.md
  A	conf/experiment/ckla_phaseonly_8s_ft.yaml
  A	conf/experiment/ckla_phaseonly_avq.md
  A	conf/experiment/ckla_phaseonly_avq.yaml
  A	conf/experiment/ckla_refiner.md
  A	conf/experiment/ckla_refiner.yaml
  M	conf/experiment/f2_dcunet_avq_survey.md
  A	conf/experiment/gen_h1_hybrid_wind.md
  A	conf/experiment/gen_h1_hybrid_wind.yaml
  A	conf/experiment/gen_h2_hybrid_uniform.md
  A	conf/experiment/gen_h2_hybrid_uniform.yaml
  A	conf/experiment/gen_hu_lo.md
  A	conf/experiment/gen_hu_lo.yaml
  A	conf/experiment/gen_hw_lo.md
  A	conf/experiment/gen_hw_lo.yaml
  A	conf/experiment/gen_s1_spatial_nowind.md
  A	conf/experiment/gen_s1_spatial_nowind.yaml
  A	conf/experiment/gen_s2_spatial_wind.md
  A	conf/experiment/gen_s2_spatial_wind.yaml
  A	conf/experiment/gen_s3_spatial_uniform.md
  A	conf/experiment/gen_s3_spatial_uniform.yaml
  A	conf/experiment/gen_v1_recal.md
  A	conf/experiment/gen_v1_recal.yaml
  A	conf/experiment/gen_v1_recal_mm.md
  A	conf/experiment/gen_v1_recal_mm.yaml
  A	conf/experiment/gen_v2_recal.md
  A	conf/experiment/gen_v2_recal.yaml
  A	conf/experiment/gen_w1_lik_nowind.md
  A	conf/experiment/gen_w1_lik_nowind.yaml
  A	conf/experiment/gen_w2_lik_wind.md
  A	conf/experiment/gen_w2_lik_wind.yaml
  A	conf/experiment/gen_w3_lik_nowind_mm.md
  A	conf/experiment/gen_w3_lik_nowind_mm.yaml
  A	conf/experiment/gen_w4_lik_wind_mm.md
  A	conf/experiment/gen_w4_lik_wind_mm.yaml
  A	conf/experiment/scv2_fs_v2.md
  A	conf/experiment/scv2_fs_v2.yaml
  A	conf/experiment/unigru128_fs_v2.md
  A	conf/experiment/unigru128_fs_v2.yaml
```

## Docs (docs/) — excerpts for added files

- MODIFIED: docs/data-and-artifacts.md
### ADDED: docs/experiments/beat-vk.md
```
# Beat-VK Campaign: Neural(-Hybrid) RPS Prediction Past the Blind VK Reference

**Status:** active — goal set 2026-07-28 · **Prior context:** `docs/experiments/ckla.md`
(readout-family results), `docs/experiments/g1-vk-parity.md` (bars),
`docs/vk-order-tracking-design.md` (blind VK pipeline §7),
`docs/experiments/rps-trajectory-refinement.md` (stages A–D lessons).

## Goal

Beat the blind-VK reference on the DREGON + Michael's validation sets with an
architectural idea (CKLA or otherwise). Allowed levers: training regimes,
generated data, self-/unsupervised use of unannotated drone noise. Bars
(guarded blind VK, 20 s windows): **DREGON cruise pooled 0.688**, **FLY124
cruise 1.027** (err_sm rev/s). Same-clip telemetry-init VK reference (8 s clip
protocol, `ref_mae_vk_rec` in `scripts/rps_predictor_vk_eval.py`): **0.729 /
0.282**.

## Gap analysis (2026-07-28 cruise eval, vk-cruise-cpu-3f32b1)

Best neural (phase-only CKLA, chmean best-arm): dregon_cruise **2.79/2.84**
(two seeds), fly124_cruise **1.29/1.33**. Decomposition:

- FLY124: most clips 0.7–1.3 MAE — several already beat the 1.027 bar; the
  pooled number is dragged by two clips. Gap ≈ 25%.
- DREGON: a **uniform ~2.2–3 floor across all free-flight cruise clips**
  (median 2.26; not outlier-driven) + two bad low-RPS/sweep clips (5.1, 6.0).
  Relative error 2.8% vs VK 0.86% at ~80 rev/s.
- DREGON GT itself carries ±0.6 rev/s fast jitter (refinement audit); VK's
  0.688 is near the audio-information floor. The neural gap is **precision**
  (sub-bin frequency reading, per-rotor phase tracking), not coverage: the
