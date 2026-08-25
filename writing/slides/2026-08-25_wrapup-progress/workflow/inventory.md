# Work inventory since last slides

- generated: 2026-08-25T01:34:36+01:00
- boundary artifact: (explicit --since)
- boundary commit: da5d1b3 2026-08-18 v4 fallback: its own band law — the v2 schedule cannot narrow
- HEAD: ac3e426 2026-08-25 TODO: ebsrof-R2, CKLA/KLA-R2 paper rows, HG-CKLA build, slides now

> Numbers in docs below may predate later fixes — always cross-check the
> newest report/doc before quoting. Sync results before analysis (Rule 5).

## Commits (newest first)

```
ac3e426 TODO: ebsrof-R2, CKLA/KLA-R2 paper rows, HG-CKLA build, slides now
8a2c998 Design doc: HG-CKLA - the harmonic-gather CKLA cell (pi_kalman as a layer)
1455bfc Leaderboard: hb_gru_ssq row - the gated 3x3 HB grid is complete
60b61f9 Leaderboard: hb_tr_ssq row (transformer trunk complete)
b466fb4 Leaderboard: blind-tracker rows (both refusal conventions) + hb_gru_if
e0319e2 Leaderboard: hb_tr_if row
9e4b45d Paper: qualitative per-regime figure generator (make_figures.py)
4786f8a Leaderboard: hb_gru_mag and hb_tr_mag rows
8326367 Regime-matched reruns: R2-R5 at fixed architecture (11 experiments)
e927730 Blind-tracker leaderboard row driver (annotate + score on valid-full)
2092d58 Salience zero-path fix + widened narrow-SR arm on the hb regime
00ab2b4 Paper: tag the generated-data experiments with regime names (R3-R5)
9c35d85 Paper: narrative restructure to the six-bullet arc
d739601 Paper: all citation markers resolved, bibliography enabled
a395699 Leaderboard: first HB rows (scv2 trio) with per-regime probe
2c0cfd0 Paper: splits + regime taxonomy (sec:splits) and architecture-search section
9522f23 FLY103/FLY108 fine calibration baked in; michaels-test-frames published
aef6158 Test-set groundwork: FLY103/FLY108 coarse calibration + michaels-test wiring
d741d34 Correct the new-drone-noises coverage note (FLY103/FLY108, not 103-of-108)
8112650 Archival report PDF (force-added past the built-PDF ignore)
5906b95 Unified baseline eval: classical RPS restored + salience on the hb regime
42ba40d Ablation 3 closed: scv2 mixed final number 93.6 (completed rerun)
0f92a9e HB campaign: honest base regime, voicing gates, front-end grid
59442b6 regime probe: comb-only s2 rows + reading in the batch doc
2b578a7 OT protocol row + mixed-ablation verdicts into paper and docs
a98416f generated-noise pool: snapshot the ready flags before nonzero
9184e99 OT baseline: full-protocol gridrun evaluator (clip x channel units)
9347066 OT baseline adaptation: band floor 150 Hz + 1 s frames; cruise 21->14.4 rev/s
5442cfb OT baseline experiment doc
c6cb38a OT multi-pitch baseline: reimplementation of Bjorkman & Elvander (TSP 2026)
a065367 Paper: ablation-1 verdict in section 8 (comb suffices; generator unnecessary)
b9930b2 Ablation 1 verdict: the analytic comb alone matches/beats the generator mix
8b3785c Paper section 8 resolved: the curriculum result table + regime attribution
35b8b61 m3cur curriculum verdict: beats real-only on all three archs; regime attribution
bcbe108 Wrap-up paper: LaTeX project from the v0.2 draft (prose verbatim)
cab9701 m3abl ablations: comb-only / generator-only pretrain + mixed non-curriculum
f90ff6a m3cur curriculum: gen_m3 generated stage + real fs_v2 stage, no warm-up
91c73a6 gen_m3 sweep findings: monitor picks comb-poor epochs; valid split is warm-up audio
eeb2a85 Comb eval: multi-rig chunks (--rigs dregon,michaels), split filter, per-rig labels
52e9a26 gen_m3: refined labels on BOTH rigs via michaels_rps_override_dir
84eec69 Backfill E7-E12 conclusions; wrap-up inventory; amplitude-arm doc debt
```

## Experiment configs (conf/experiment/)

```
  M	conf/experiment/e10_full_scv2.md
  M	conf/experiment/e10_full_transformer.md
  M	conf/experiment/e10_full_unigru128.md
  M	conf/experiment/e10_noisegen_fullrange.md
  M	conf/experiment/e11_full_aug_scv2.md
  M	conf/experiment/e11_full_aug_transformer.md
  M	conf/experiment/e11_full_aug_unigru128.md
  M	conf/experiment/e11_full_ft_warp_scv2.md
  M	conf/experiment/e11_full_ft_warp_transformer.md
  M	conf/experiment/e11_full_ft_warp_unigru128.md
  M	conf/experiment/e11_real_warp_scv2.md
  M	conf/experiment/e11_real_warp_transformer.md
  M	conf/experiment/e11_real_warp_unigru128.md
  M	conf/experiment/e12_real_fullflight_scv2.md
  M	conf/experiment/e12_real_fullflight_transformer.md
  M	conf/experiment/e12_real_fullflight_unigru128.md
  M	conf/experiment/e7_gencurric_s1_scv2.md
  M	conf/experiment/e7_gencurric_s1_transformer.md
  M	conf/experiment/e7_gencurric_s1_unigru128.md
  M	conf/experiment/e7_gencurric_s2_scv2.md
  M	conf/experiment/e7_gencurric_s2_transformer.md
  M	conf/experiment/e7_gencurric_s2_unigru128.md
  M	conf/experiment/e8_staticcomb_s1_scv2.md
  M	conf/experiment/e8_staticcomb_s1_transformer.md
  M	conf/experiment/e8_staticcomb_s1_unigru128.md
  M	conf/experiment/e9_hard_scv2.md
  M	conf/experiment/e9_hard_scv2_ft_real.md
  M	conf/experiment/e9_hard_transformer.md
  M	conf/experiment/e9_hard_transformer_ft_real.md
  M	conf/experiment/e9_hard_unigru128.md
  M	conf/experiment/e9_hard_unigru128_ft_real.md
  A	conf/experiment/gen_a1_amp.md
  A	conf/experiment/gen_a1_amp_render.md
  A	conf/experiment/gen_a1_amp_v2.md
  A	conf/experiment/gen_a2_amp_perrotor.md
  A	conf/experiment/gen_a2_amp_perrotor_render.md
  A	conf/experiment/gen_a2_amp_perrotor_v2.md
  A	conf/experiment/gen_c1_amp_combined.md
  A	conf/experiment/gen_c1_amp_combined_render.md
  A	conf/experiment/gen_c2_amp_combined_perrotor.md
  A	conf/experiment/gen_c2_amp_combined_perrotor_render.md
  A	conf/experiment/gen_m3_refined_all_perrotor.md
  A	conf/experiment/gen_m3_refined_all_perrotor.yaml
  A	conf/experiment/hb_gru_if.md
  A	conf/experiment/hb_gru_if.yaml
  A	conf/experiment/hb_gru_mag.md
  A	conf/experiment/hb_gru_mag.yaml
  A	conf/experiment/hb_gru_ssq.md
  A	conf/experiment/hb_gru_ssq.yaml
  A	conf/experiment/hb_sal_bp.md
  A	conf/experiment/hb_sal_bp.yaml
  A	conf/experiment/hb_sal_multif0.md
  A	conf/experiment/hb_sal_multif0.yaml
  A	conf/experiment/hb_sal_multif0_nsr.md
  A	conf/experiment/hb_sal_multif0_nsr.yaml
  A	conf/experiment/hb_scv2_if.md
  A	conf/experiment/hb_scv2_if.yaml
  A	conf/experiment/hb_scv2_mag.md
  A	conf/experiment/hb_scv2_mag.yaml
  A	conf/experiment/hb_scv2_mag_nogate.md
  A	conf/experiment/hb_scv2_mag_nogate.yaml
  A	conf/experiment/hb_scv2_ssq.md
  A	conf/experiment/hb_scv2_ssq.yaml
  A	conf/experiment/hb_tr_if.md
  A	conf/experiment/hb_tr_if.yaml
  A	conf/experiment/hb_tr_mag.md
  A	conf/experiment/hb_tr_mag.yaml
  A	conf/experiment/hb_tr_ssq.md
  A	conf/experiment/hb_tr_ssq.yaml
  A	conf/experiment/m3abl_comb_scv2_s1.md
  A	conf/experiment/m3abl_comb_scv2_s1.yaml
  A	conf/experiment/m3abl_comb_scv2_s2.md
  A	conf/experiment/m3abl_comb_scv2_s2.yaml
  A	conf/experiment/m3abl_comb_transformer_s1.md
  A	conf/experiment/m3abl_comb_transformer_s1.yaml
  A	conf/experiment/m3abl_comb_transformer_s2.md
  A	conf/experiment/m3abl_comb_transformer_s2.yaml
  A	conf/experiment/m3abl_comb_unigru128_s1.md
  A	conf/experiment/m3abl_comb_unigru128_s1.yaml
  A	conf/experiment/m3abl_comb_unigru128_s2.md
  A	conf/experiment/m3abl_comb_unigru128_s2.yaml
  A	conf/experiment/m3abl_gen_scv2_s1.md
  A	conf/experiment/m3abl_gen_scv2_s1.yaml
  A	conf/experiment/m3abl_gen_scv2_s2.md
  A	conf/experiment/m3abl_gen_scv2_s2.yaml
  A	conf/experiment/m3abl_gen_transformer_s1.md
  A	conf/experiment/m3abl_gen_transformer_s1.yaml
  A	conf/experiment/m3abl_gen_transformer_s2.md
  A	conf/experiment/m3abl_gen_transformer_s2.yaml
  A	conf/experiment/m3abl_gen_unigru128_s1.md
  A	conf/experiment/m3abl_gen_unigru128_s1.yaml
  A	conf/experiment/m3abl_gen_unigru128_s2.md
  A	conf/experiment/m3abl_gen_unigru128_s2.yaml
  A	conf/experiment/m3abl_mixed_scv2.md
  A	conf/experiment/m3abl_mixed_scv2.yaml
  A	conf/experiment/m3abl_mixed_transformer.md
  A	conf/experiment/m3abl_mixed_transformer.yaml
  A	conf/experiment/m3abl_mixed_unigru128.md
  A	conf/experiment/m3abl_mixed_unigru128.yaml
  A	conf/experiment/m3cur_scv2_s1.md
  A	conf/experiment/m3cur_scv2_s1.yaml
  A	conf/experiment/m3cur_scv2_s2.md
  A	conf/experiment/m3cur_scv2_s2.yaml
  A	conf/experiment/m3cur_transformer_s1.md
  A	conf/experiment/m3cur_transformer_s1.yaml
  A	conf/experiment/m3cur_transformer_s2.md
  A	conf/experiment/m3cur_transformer_s2.yaml
  A	conf/experiment/m3cur_unigru128_s1.md
  A	conf/experiment/m3cur_unigru128_s1.yaml
  A	conf/experiment/m3cur_unigru128_s2.md
  A	conf/experiment/m3cur_unigru128_s2.yaml
  A	conf/experiment/r2hb_gru_nogate.md
  A	conf/experiment/r2hb_gru_nogate.yaml
  A	conf/experiment/r2hb_tr_nogate.md
  A	conf/experiment/r2hb_tr_nogate.yaml
  A	conf/experiment/r3hb_gru.md
  A	conf/experiment/r3hb_gru.yaml
  A	conf/experiment/r3hb_scv2.md
  A	conf/experiment/r3hb_scv2.yaml
  A	conf/experiment/r3hb_tr.md
  A	conf/experiment/r3hb_tr.yaml
  A	conf/experiment/r4hb_gru.md
  A	conf/experiment/r4hb_gru.yaml
  A	conf/experiment/r4hb_scv2.md
  A	conf/experiment/r4hb_scv2.yaml
  A	conf/experiment/r4hb_tr.md
  A	conf/experiment/r4hb_tr.yaml
  A	conf/experiment/r5hb_gru.md
  A	conf/experiment/r5hb_gru.yaml
  A	conf/experiment/r5hb_scv2.md
  A	conf/experiment/r5hb_scv2.yaml
  A	conf/experiment/r5hb_tr.md
  A	conf/experiment/r5hb_tr.yaml
```

## Docs (docs/) — excerpts for added files

- MODIFIED: docs/experiments/beat-vk.md
- MODIFIED: docs/experiments/e10-full-flight.md
- MODIFIED: docs/experiments/e7-gen-curriculum.md
- MODIFIED: docs/experiments/e8-static-comb.md
- MODIFIED: docs/experiments/e9-hard-combined.md
- MODIFIED: docs/experiments/generator-refined-labels.md
### ADDED: docs/experiments/honest-base-frontends.md
```
# Honest base regime + front-end grid (HB campaign)

Status: DESIGNED 2026-08-24, runs pending.

## Motivation

The zero-RPS regime is the largest weakness of the real-only predictors.
The diagnosis (2026-08-24 probes, `docs/experiments/generator-refined-labels.md`
§ regime probe) has three parts:

1. **Coverage.** The real training pool holds 421.4 s of noise. The regime
   split of training time is 6.25% zero / 8.11% ramps / 85.63% flight.
   Only 26.4 s of unique zero material exists, almost all from one room.
   The validation split is 12.7% zero frames — double the training share.
2. **Level confound.** The mixer scales speech relative to the power of the
   noise chunk. A zero chunk has ~40 dB less power than a flight chunk, so
   its mixture is near-silent overall. "Quiet in, zero out" is a winning
   shortcut on the training set, and it breaks on validation zeros that
   carry content (the 41–50 Hz rumble clip, transitions).
3. **Output head.** The plain linear head has no off state. Under MSE an
   uncertain model outputs the conditional mean — the observed 10–45 rev/s
   drift on zero frames (57% of real-only scv2 zero-frame values; no model
   hallucinates flight speeds there).

The zero-labeled spans of the real pool are label-honest: their audio RMS is
1–6% of flight RMS and the 40–350 Hz rotor band holds 3–9% of the energy.
The problem is what they lack — level diversity and content — not what they
contain.

## Design
```
### ADDED: docs/experiments/otmp-baseline.md
```
# OT multi-pitch baseline (Björkman & Elvander, TSP 2026) on drone rotor speeds

**Status:** implemented, smoke-tested, adapted — **Date:** 2026-08-23

**Verdict:** the adaptation is worth 1.5x (cruise PIT-MAE 21.1 -> 14.4 rev/s
with `adapted_drone_config()`) and the method still does not work on this
signal. Quote it as a classical-baseline floor. Do not schedule a
full-protocol run. See "Adaptation probes" below.

Reimplementation of arXiv 2508.02471 ("Inverse Harmonic Clustering for
Multi-Pitch Estimation: An Optimal Transport Approach", stochastic
estimator) as the classical multi-pitch baseline for the wrap-up paper
(§4.1). Code: `src/experiments/otmp_baseline/` (commit c6cb38a); every
implementation choice the paper leaves open is marked `[choice]` in source.

## Fidelity to the paper

Monte-Carlo self-test (their Sec VIII-A, 4 pitches 176/197/240/272 Hz,
5 dB, Table I parameters): **GER 28 %** vs the paper's 8–10 %; median
deviation of FOUND pitches 9.2 cents; 18.4 s/draw. The gap is a detection
failure of few-harmonic pitches (harmonic count uniform 3..10): the
group-sparsity term prices a new pitch column above the transport cost of
absorbing 3 partials into a neighbour. Probes: pitch-grid density matters
(1 Hz grid best of those tried; paper does not state G); more iterations
past ~800 hurt (sparsity keeps eroding weak components after the ranking
settles).

## Out-of-the-box drone result (paper Table II params, grids adapted only)

Frozen valid clips, channel 0, 0.5 s frames: PIT-MAE 38.3 rev/s (cruise
```
### ADDED: docs/experiments/unified-baseline-eval.md
```
# Unified baseline evaluation on the frozen validation split

Status: IN PROGRESS 2026-08-24.

## Motivation

Every baseline family in this project was scored on a different set: the
May-2026 classical baselines on 10 DREGON-LM test samples, the June salience
baselines on `DREGON-LM-V4/valid`, the OT multi-pitch baseline and the
neural models on `DREGON-LM-V4-michaels-valid-full`. The numbers are not
comparable. This campaign scores every family once, on one split, with one
protocol.

## Protocol (frozen)

- Split: `dload:DREGON-LM-V4-michaels-valid-full` — 37 clips x 8 channels.
- Grid: per-frame predictions on the 2048/512 STFT grid at 16 kHz.
- Matching: per-frame Hungarian PIT on |pred - target|.
- Regimes from the target: zero (max rotor < 1 rev/s), flight (mean >= 45),
  low (the rest). Report MAE and MSE per regime and overall.
- This is the exact protocol of `results/m3cur_regime_probe/regime_probe.py`.

## Rows

| Family | Method(s) | Source of numbers | Status |
|---|---|---|---|
| Classical (May 2026) | PYIN, cepstral, HPS, matched filter, NMF | `src/experiments/classical_rps/valid_eval.py` (restored from `00753c4`; report `writing/reports/2026-05-29_classical-baselines/`) — local run 2026-08-24 | done |
| OT multi-pitch | Björkman & Elvander 2026 (adapted config) | `results/otmp_protocol/` (run 2026-08-23) — re-aggregated under this table | done, merge |
| Salience (June ckpts) | multif0_salience, basic_pitch_salience | old V4-trained checkpoints, if loadable from the zoo | optional |
| Salience (retrained) | `hb_sal_multif0`, `hb_sal_bp` | retrained on the hb online regime (same data + augmentations as the HB grid) | pending training |
```
### ADDED: docs/pikalman-ckla-design.md
```
# HG-CKLA: the harmonic-gather CKLA cell (CKLA as a true pi_kalman pass)

Status: DESIGN, 2026-08-25. No implementation yet. Companion to
`docs/ckla-design.md` (the original CKLA layer) and
`src/tracking/phase_increment_tracker.py` (the classical algorithm this
mirrors). Part of the neural-RPS program (seed -> annealed refinement ->
heads).

## 1. Motivation

The CKLA campaign built a complex-OU Kalman linear-attention layer and used
it as a drop-in replacement for the Transformer temporal head: conv trunk
over log-magnitude(+IF) features -> frequency-attention pool -> CKLA
sequence mixer over pooled 128-dim frame vectors. The intent was "a layer
that does what one pi_kalman pass does". The built thing cannot do that,
for one structural reason: **pi_kalman's measurements are conditioned on
its own state** — it reads the spectrogram *at the harmonic positions its
current estimate predicts* — while the CKLA head receives measurements that
are state-independent, because the frequency pool collapses the spectral
axis before the recurrence ever sees it. After the pool there is no "at
harmonic k of rotor r" left to read. A sequence mixer over pooled features
can filter; it cannot implement an extended-Kalman measurement update.

This document specifies the smallest architectural change that makes the
original intent true: move the measurement inside the recurrence, as a
differentiable gather at state-predicted harmonic positions.

## 2. What pi_kalman actually computes

From `phase_increment_tracker.py`, per outer iteration and rotor (numbered,
```

## Writing artifacts created/updated in the window

### writing/reports/2026-05-29_classical-baselines

## Code changes (summary)

```
 tests/models/test_voicing_gate.py                  | 124 +++
 tests/scripts/test_blind_valid_row.py              | 117 +++
 40 files changed, 7421 insertions(+), 299 deletions(-)
     10 src/experiments
      9 src/data_processing
      7 src/models
      3 scripts/michaels_calib
      1 scripts/eval_gen_comb_real.py
      1 scripts/blind_valid_row.py
```

## Untracked candidates (not yet committed)

```
  conf/experiment/hb_ckla.md
  conf/experiment/hb_ckla.yaml
  conf/experiment/hb_ebsrof.md
  conf/experiment/hb_ebsrof.yaml
  conf/experiment/hb_ebsrof_lowlr.md
  conf/experiment/hb_ebsrof_lowlr.yaml
  conf/experiment/hb_fkla.md
  conf/experiment/hb_fkla.yaml
  writing/papers/2026-08_wrapup/figures/
  writing/slides/2026-08-18_decomposition-for-amplitude-targets/
```

## Prep notes found (read these fully — often a ready-made narrative seed)

- writing/slides/NEXT-DECK-experiment-inventory.md
