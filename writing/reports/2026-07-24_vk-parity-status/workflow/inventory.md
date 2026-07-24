# Work inventory since last report

- generated: 2026-07-24T11:24:45+01:00
- boundary artifact: writing/reports/2026-07-22_se-blind-baselines
- boundary commit: 2b8a054 2026-07-23 F1 SE-baselines: SGMSE+ result — from-scratch score diffusion is non-viable
- HEAD: b2590f6 2026-07-24 4-way comparison: matrice100 half complete — DEEP wins on both drones

> Numbers in docs below may predate later fixes — always cross-check the
> newest report/doc before quoting. Sync results before analysis (Rule 5).

## Commits (newest first)

```
b2590f6 4-way comparison: matrice100 half complete — DEEP wins on both drones
3990b9a G1 phase-B result: longer native context refuted as the parity lever
0aa4b67 VK-parity eval: register the four G1 phase-B checkpoints
d7b2b63 VK design doc §7.5: blind-annotation criterion closed (FLY124 1.03 pooled)
b4a3eee G1: add the missing conf/data entries for the 4s/8s full-flight streams
23620b2 Slides rework: speaker-notes-driven rebuild of the 07-18 deck
6277fb0 Blind per-track stage guard: revert ladder stages that destroy a track
e72324c G1 VK-parity: phase-A smoothing eval script + phase-B long-context configs
ab3a4ed Arm R (residual re-scan): recover combs shadowed by stronger neighbours
0d5d59c Sweep: pin repo src/ over stale editable install; provenance banner; --recordings
47c0e6f Slides 2026-07-18: hand-edited speaker notes = rework instructions
41a941a Four-way noise comparison: real / CONA / deep-gen / GP on shared GT RPS (dregon)
6279962 vk_bench job runner: pin PYTHONPATH to the job worktree's src
1ca7581 VK tracker CPU fast paths: banded Hermitian solver, pair pruning, FIR flag
a894501 Blind seeding v2 fixes: band-capped scan, spatial-DP ladder, gated T
e6288ff Blind seeding v2 (design §7): T/C/N/K arms + sweep runner
1834a11 GP ego-noise trainer: per-drone JASA GP on the CONA rps sweep
0259a78 vk_bench: --cases filter for partial resubmission
612fc6d vk_bench: drop unused destructured names flagged by pyright
3c3df54 VK bench: profiling/regression benchmark for vk_track (phase 1 fast-inference)
e156ce4 VK design §7: blind seeding v2 (shared-comb matched filter, alias rejection, count prior, auto-knobs)
621c6a6 Plan: add literature-evidence appendix (Gulli deltas, empty novelty niches)
```

## Experiment configs (conf/experiment/)

```
  A	conf/experiment/g1_transformer_4s.md
  A	conf/experiment/g1_transformer_4s.yaml
  A	conf/experiment/g1_transformer_8s.md
  A	conf/experiment/g1_transformer_8s.yaml
```

## Docs (docs/) — excerpts for added files

### ADDED: docs/experiments/g1-vk-parity.md
```
# G1 — VK-parity training arms (longer chunks)

Campaign criterion 2.3: bring an audio-only neural RPS predictor to parity
with the best blind VK tracker on the SAME evaluation clips
(`results/vk_eval/vk_valid_comparison.csv` protocol; blind-VK bars: DREGON
free-flight cruise pooled ~0.68-0.74 rev/s, FLY124 cruise 3.24).

## Phase A result (test-time smoothing, no training)

`scripts/rps_predictor_vk_eval.py` evaluated the E12 real-full-flight
checkpoints (+ C11 DREGON+FLY125 scv2) with sliding-window stitching and
2-20 s moving-average / running-median aggregation, single-mic (protocol)
and 8-mic-averaged inputs. Outcome: smoothing helps but saturates well short
of the bar on DREGON cruise (see `results/rps_predictor_vk_eval/`); the
neural error is systematic within a window, not zero-mean jitter. FLY124
cruise is already below the blind-VK 3.24 bar without any smoothing.

## Phase B hypothesis

E12 trained on 1 s chunks (`duration_s: 1.0`) but the protocol evaluates 8 s
clips; VK integrates over the whole trajectory. Give the model native
context: same recipe, `duration_s` 4/8, batch size scaled down to fit a
T4/P100 16 GB.

## Arms

| experiment | chunk | batch | policy |
|---|---|---|---|
| `g1_transformer_4s` | 4 s | 8 | `conf/online_mix/g1_real_fullflight_4s_dload.yaml` |
| `g1_transformer_8s` | 8 s | 4 | `conf/online_mix/g1_real_fullflight_8s_dload.yaml` |
```
- MODIFIED: docs/three-track-plan-2026-07-20.md
- MODIFIED: docs/vk-order-tracking-design.md

## Writing artifacts created/updated in the window

### writing/slides/2026-07-18_dregon-analysis-and-generator-design
```
= This week
= The problem: what the old generator sounds like
= Motor sound propagation in generator model
= Hypothesis 1 — geometry: a small error is not benign
= Geometry errors found (huge!) in DREGON annotations
= Geometry errors found (silly) in Michael's annotations
= Geometry: fine calibration
= Hypothesis 2 — rotors are individuals
= Treating rotors individually: per-rotor sub-embeddings
= Hypothesis 3 — wind noise confusing the model
= Results: spectrograms
= Results: the scores, and why
= Work thread 2: blind per-rotor RPS from audio alone
= Steps 1–2: whiten, then scan for combs
= Step 3: seed all four rotors
= Steps 4–5, the VK core: fit envelopes, correct frequencies, tighten
= Step 4 detail: coupled solve — tracks compete for shared energy
= Step 5 detail: frequency update by phase slope, then anneal
= The full blind-annotation algorithm
= VK results: telemetry-init refinement and blind re-annotation
= Blind annotation on FLY124: the failure mode, visibly
= Speed of algorithm: optimized quite a bit already
= Work thread 3: a literature baseline (JASA-GP)
= What is CONA, and how is the synthetic audio computed?
= JASA-GP: original data replication
```

## Code changes (summary)

```
 .../gp_rotor_noise/train_egonoise_gp.py            |  603 ++++++++++++
 tests/test_vk_blind_seeding.py                     |  265 +++++
 14 files changed, 4254 insertions(+), 116 deletions(-)
      4 scripts/vk_bench_cases
      2 src/data_processing
      1 src/experiments
      1 scripts/vk_validation.py
      1 scripts/vk_blind_sweep.py
      1 scripts/vk_blind_annotation.py
      1 scripts/vk_bench.py
      1 scripts/vk_bench_opt_job.sh
      1 scripts/rps_predictor_vk_eval.py
```

## Untracked candidates (not yet committed)

```
  (none)
```

## Prep notes found (read these fully — often a ready-made narrative seed)

