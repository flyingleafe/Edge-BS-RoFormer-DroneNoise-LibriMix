# Work inventory since last slides

- generated: 2026-07-21T02:00:34+01:00
- boundary artifact: (explicit --since)
- boundary commit: 5c552db 2026-07-13 Slide-prep note: experiment inventory since the gp-rotor-noise deck
- HEAD: 47c0e6f 2026-07-21 Slides 2026-07-18: hand-edited speaker notes = rework instructions

> Numbers in docs below may predate later fixes — always cross-check the
> newest report/doc before quoting. Sync results before analysis (Rule 5).

## Commits (newest first)

```
47c0e6f Slides 2026-07-18: hand-edited speaker notes = rework instructions
41a941a Four-way noise comparison: real / CONA / deep-gen / GP on shared GT RPS (dregon)
6279962 vk_bench job runner: pin PYTHONPATH to the job worktree's src
1ca7581 VK tracker CPU fast paths: banded Hermitian solver, pair pruning, FIR flag
a894501 Blind seeding v2 fixes: band-capped scan, spatial-DP ladder, gated T
3201298 F1: per-checkpoint eval driver (per category/SNR) + table assembly
3e8905a SGMSE+ port (65.6M NCSN++/OUVE) + bespoke score-matching training loop
f5a00eb F1: noisy+Wiener anchor eval script + audio_pool/SE-target docs
d72e6d8 SE blind-baselines (F1): infra + ports + valid sets + configs
e6288ff Blind seeding v2 (design §7): T/C/N/K arms + sweep runner
1834a11 GP ego-noise trainer: per-drone JASA GP on the CONA rps sweep
0259a78 vk_bench: --cases filter for partial resubmission
612fc6d vk_bench: drop unused destructured names flagged by pyright
3c3df54 VK bench: profiling/regression benchmark for vk_track (phase 1 fast-inference)
e156ce4 VK design §7: blind seeding v2 (shared-comb matched filter, alias rejection, count prior, auto-knobs)
621c6a6 Plan: add literature-evidence appendix (Gulli deltas, empty novelty niches)
be2688a SE blind-baselines plan + three-track plan; koopman survey; 2026-07-18 writeups
736a639 Report: proper 8s mid-flight harmonic spectrograms
4e32f5a Report fix: score on free-flight, drop misleading loss column
5be4075 Report: generator retraining on corrected mic-array geometry (3 variants)
97b25a8 v1/v3: batch 16 x grad_accum 2 (effective batch 32) to fit any backend
18d6414 Add noise-gen variant eval: valid MSSTFT + real-vs-generated spectrograms
e570d80 Fix wandb: log val media under samples/ so val/loss chart is visible
400c092 Generator variant 3: additive wind-wake channel integration
f2a37f2 Generator variant 2: per-rotor sub-embeddings (z_r = z_drone + δz_r)
006664e Generator variant 1: corrected-geometry retrain (swapped split, streamed)
bb9356f Wind-wake flow-noise channel: model + CPU pre-training de-risk
d42bf23 Fix JASA-GP phase/frequency alignment (major fidelity gain)
f8b96fb Lean jasa-flyovers loader: subset speeds, skip audio, float32 (fix OOM)
8cd4efb Strip embedded widget/output metadata from JASA-GP notebook (5.4MB->10KB)
ccaa3fc Interactive 3D JASA-GP listening notebook + anywidget dep
08b8b59 JASA GP rotor-noise: faithful jasa-flyovers model + train/eval entry
3a4a82e Repin DREGON-frames/michaels-frames to corrected-geometry versions
00b86c1 Correct DREGON & Michael's mic-array geometry; add Stage-0 self-calibration study
a235278 Coupled Vold-Kalman order tracking: module, validation, blind annotation, SPCup
7e1771d writeup agent
```

## Experiment configs (conf/experiment/)

```
  A	conf/experiment/f1_dcunet_a.md
  A	conf/experiment/f1_dcunet_a.yaml
  A	conf/experiment/f1_dcunet_b.md
  A	conf/experiment/f1_dcunet_b.yaml
  A	conf/experiment/f1_edge_bs_rof_a.md
  A	conf/experiment/f1_edge_bs_rof_a.yaml
  A	conf/experiment/f1_edge_bs_rof_b.md
  A	conf/experiment/f1_edge_bs_rof_b.yaml
  A	conf/experiment/f1_mpsenet_a.md
  A	conf/experiment/f1_mpsenet_a.yaml
  A	conf/experiment/f1_mpsenet_b.md
  A	conf/experiment/f1_mpsenet_b.yaml
  A	conf/experiment/f1_sgmse_a.md
  A	conf/experiment/f1_sgmse_a.yaml
  A	conf/experiment/f1_sgmse_b.md
  A	conf/experiment/f1_sgmse_b.yaml
  A	conf/experiment/f1_tfgridnet_a.md
  A	conf/experiment/f1_tfgridnet_a.yaml
  A	conf/experiment/f1_tfgridnet_b.md
  A	conf/experiment/f1_tfgridnet_b.yaml
  A	conf/experiment/gen_v1_corrected.md
  A	conf/experiment/gen_v1_corrected.yaml
  A	conf/experiment/gen_v2_perrotor.md
  A	conf/experiment/gen_v2_perrotor.yaml
  A	conf/experiment/gen_v3_wind.md
  A	conf/experiment/gen_v3_wind.yaml
```

## Docs (docs/) — excerpts for added files

- MODIFIED: docs/AGENTS.md
### ADDED: docs/experiments/f1-se-blind-baselines.md
```
**Status:** in progress · 2026-07-20 – present · plan:
[`docs/se-baselines-plan.md`](../se-baselines-plan.md) (Track 1 M1.1–M1.4 of
`docs/three-track-plan-2026-07-20.md`)

# F1 — SE blind (no-RPS) baselines on our data

## Motivation

Establish the **blind** (no-RPS) speech-enhancement floor on our harmonic-noise
data with modern architectures, so every later RPS-informed claim is measured
against an honest, strong no-side-information baseline. Two training passes over
the same architecture set, both scored on the same two fixed validation sets:

- **Pass A (`f1_<arch>_a`) — drone noises only**: the drone-focused floor.
- **Pass B (`f1_<arch>_b`) — all harmonic noises, category-uniform**: does
  *diverse* harmonic noise help models on drone noise (transferable harmonic
  structure) or hurt (capacity dilution)? Pass B − Pass A on `SE-valid-drone`
  answers the diversity question; the per-category breakdown on
  `SE-valid-harmonic` shows which categories transfer.

## Architecture set

| Arch | Family | Source | Params |
|---|---|---|---|
| `edge_bs_rof` (reuses `a1_edge_bs_rof_fa` model) | band-split transformer | in repo (Paper 1) | — |
| `dcunet` (reuses `a1_baseline_dcunet` model) | complex UNet | in repo | — |
| `tfgridnet` (`f1_tfgridnet`) | dense full+sub-band dual-path | port (ESPnet V1, mid-size) | 8.38 M |
| `mpsenet` (`f1_mpsenet`) | parallel magnitude+phase | port (yxlu-0102/MP-SENet, generator-only) | 1.71 M |
| `sgmse` (`f1_sgmse`) | score-based diffusion (generative) | port (sp-uhh/sgmse), trained from scratch | — |
| noisy input, Wiener | floors | trivial anchors | — |
```
### ADDED: docs/koopman-and-order-tracking-ideas.md
```
# Koopman operators & Vold–Kalman literature — ideas

**Status:** literature survey, no implementation commitment · **Date:** 2026-07-18

Consolidates a multi-session literature exploration (via the `bib` bibliography
MCP/CLI — all papers below are tagged `harmonic-noise-suppression` there;
`bib search "..."` or `search_library` will resurface them with full
abstracts). Motivating question: **can we get a single latent state of the
"rotating + buzzing" system (motor+propeller) that's conditionable by either
audio or RPS and can predict either, both directions, in one model** — rather
than a one-way RPS→noise generator or a one-way audio→RPS estimator bolted
together. Two literature branches were explored for this; a third
(Vold–Kalman) turned out to double as a concrete, already-being-implemented
mechanism (see `docs/vk-order-tracking-design.md`) rather than just background
reading.

---

## 1 · Candidate architectures for the bidirectional audio↔RPS latent

Ranked by how directly each matches "one shared latent, either modality in,
either modality out":

1. **Multimodal Mixture/Product-of-Experts VAE** — Shi et al.,
   *"Variational Mixture-of-Experts Autoencoders for Multi-Modal Deep
   Generative Models"* (2019, arXiv:1911.03393). The literal pattern: per-
   modality encoders feed a **shared latent** via PoE/MoE combination; a
   decoder per modality reconstructs from that shared z. Supply audio-only,
   RPS-only, or both; infer/generate the other. Originally vision+language,
   not audio — the framework is domain-agnostic. **Bonus**: PoE handles
```
### ADDED: docs/se-baselines-plan.md
```
# SE Blind-Baselines Plan — execution doc for the baseline-running agent

**Status:** ready to execute · **Date:** 2026-07-20 · **Parent:**
`docs/three-track-plan-2026-07-20.md` (Track 1, M1.1–M1.4). This doc is
self-contained: everything needed to run the baseline program is here.

## Objective

Establish the **blind** (no-RPS) speech-enhancement floor on our data with
modern architectures, via **two training passes** over the same architecture
set:

- **Pass A — drone noises only**: the drone-focused floor.
- **Pass B — all harmonic noises, uniformly weighted by category**: does
  *diverse* harmonic noise help models on drone noise (transferable harmonic
  structure) or hurt (capacity dilution)?

Both passes are evaluated on the **same two fixed validation sets**, so
Pass B vs Pass A on the drone valid answers the diversity question directly,
and the per-category breakdown shows which categories transfer. These floors
gate every later RPS-informed claim.

## Architecture set (5 + anchors)

| # | Model | Family | Source | Notes |
|---|---|---|---|---|
| 1 | Edge-BS-RoFormer | band-split transformer | in repo (Paper 1) | current in-house SOTA on DN-LM |
| 2 | TF-GridNet | dense full+sub-band dual-path | port (ESPnet) | use a mid-size config; heavy — budget accordingly |
| 3 | MP-SENet | parallel magnitude+phase | port (github) | fallbacks if port disappoints: CMGAN, DB-AIAT, dual-branch Mamba (all in bibliography) |
| 4 | DCUNet | complex UNet | in repo | 2023 benchmark winner; continuity anchor to both prior papers |
```
### ADDED: docs/three-track-plan-2026-07-20.md
```
# Three-track plan — 2026-07-20

**Status:** initial proposal (user review pending) · supersedes the "bridge to SE
now" recommendation from the 2026-07-20 goal review after user objections were
upheld by evidence sweeps (see memory: `goal-review-2026-07-20-state-of-project`).

Three goals, run simultaneously by interleaving compute: GPU trains baselines
(Track 1) while CPU-bound VK work (Track 2) and GP fitting (Track 3) proceed;
switch tracks whenever a training run is in flight.

---

## Track 1 — Blind-SE floor, then the oracle question

### Why this order
No credible oracle-RPS conditioning gain exists (own b1/b2 replication: DCUNet
+1.73 dB SI-SDR only at −30…−20 dB with eSTOI/PESQ down, DCCRN −2.30 dB;
Gulli et al. 2025 concentrates its gains at −30/−20 dB and loses STOI/PESQ at
−10/0 dB — same shape). Any RPS-informed claim is meaningless without a strong
*modern blind floor* measured on our data first.

### M1.1 — Infra (CPU/dev work, ~days)
- SE-target mode in `OnlineMixIterableDataset`: yield `(mixture, clean_speech)`
  instead of `(audio, rps_target)`; speech source, SNR-controlled mixing,
  augmentations, and E5 time-warp already exist in the pipeline.
- Plain-audio noise-source kind (no RPS/telemetry track required) so MIMII /
  AeroSonicDB / HornBase / DroneAudioSet can serve as noise pools; per-category
  weight support (extends E9's weighted MixedNoisePool).
- Fixed deterministic SE validation sets: (a) **drone valid** — held-out drone
  noise × held-out LibriSpeech, SNR grid −30…0 dB in 5 dB steps; (b)
```
### ADDED: docs/vk-order-tracking-design.md
```
# Coupled Vold–Kalman Order Tracking — Design

**Status:** design approved, implementation in progress · **Date:** 2026-07-14
**Goal (session):** (1) re-annotate the current valid set (DREGON + Michael's)
convincingly better than the RPS predictors alone — or decisively prove the
method cannot; (2) produce SPCup annotations whose per-rotor tracks visibly
ride the harmonic peaks on spectrogram overlays.

**Predecessor post-mortem:** `docs/experiments/rps-trajectory-refinement.md`.
Stages B+C failed because their objective (comb-sampled log-magnitude,
per-rotor, uniformly weighted) is non-attributive — rotors never compete for
spectral energy — so tight rotor pairs (DREGON: ~0.65 rev/s apart) bias the
argmax toward the pair mean (twin capture, −0.44 rev/s). Stage D (phase-slope
demodulation, Fisher weights k²|z|²) was the only unbiased stage. This design
replaces the heuristic comb score with a generative residual functional whose
minimizer is the Vold–Kalman (VK) filter, and keeps stage D's phase-slope idea
as the frequency update.

**Off-the-shelf survey:** PyVKF (github.com/CyprienHoelzl/PyVKF, port of
van der Seijs' MATLAB) is a faithful 2nd-gen VK but (a) GPL-3 — cannot be
vendored, (b) solves the full `T·M` complex sparse system at audio rate —
intractable at 16 kHz × 25 s × (4 rotors × ~40 harmonics) ≈ 16M unknowns,
(c) frequencies are *inputs*; no tracking loop. MATLAB Order Tracking Toolbox
rejected for obvious reasons. We therefore implement from the published math
(Vold & Leuridan 1993; Tuma 2005 bandwidth formula) with the three changes
below. PyVKF is used only as a numerical cross-check oracle in scratchpad.

---

## 1 · The functional
```

## Writing artifacts created/updated in the window

### writing/reports/2026-07-15_mic-array-geometry-calibration
```
= Why positions matter, and why we doubted them
= Reading geometry off the audio: TDOA and the RTF <sec-rtf>
= Discovery: a 180° frame mismatch in DREGON <sec-dregon-frame>
= Self-calibration by bundle adjustment (DREGON) <sec-dregon-bundle>
== Objective
== Does the optimiser work? A synthetic control
== DREGON refinement
= When audio can't help: Michael's, and an honest negative result <sec-michaels>
== The fixable error: wrong plane
== The hard limit: geometry is not identifiable from this audio
= What it recovers, and what it cannot <sec-summary>
== A reusable methods caveat <sec-caveats>
```
### writing/reports/2026-07-17_generator-corrected-geometry-variants
```
= Introduction
= Variants
== Wind-channel de-risk
= Results
== Scoring on free flight, not idle
= Discussion <sec-discussion>
```
### writing/reports/2026-07-18_dregon-analysis-and-generator-design
```
= A physics-structured generator, answerable to data <sec-intro>
= Geometry: the symptom, the discovery, the fix <sec-geometry>
== Why a geometry error is diagnostic, not benign
== The discovery: a 180° frame mismatch in DREGON
== The fine fix, and an honest limit
= Are the four rotors one source? <sec-perrotor>
= A wind channel: what propagation cannot produce <sec-wind>
= From analysis to variants, and what the data said back <sec-variants>
= Takeaways <sec-takeaways>
```
### writing/slides/2026-07-13_rps-synthetic-data-status
```
= Where we left off (July 6)
= Generator fix 1: harmonic linewidth
= Generator fix 1, in detail: linewidth, per rotor (1)
= Generator fix 1, in detail: linewidth, per rotor (2)
= Does it actually look right? Real vs. generated
= Generator fixes 2--3: silence + full flight
= What makes interpolation work: two regularizers
= Interpolating drone textures (1/2)
= Interpolating drone textures (2/2)
= Time-warp augmentation
= But: still not better than the best real-data models
= What is the "analytic static comb" (E8)?
= Training data: what's actually in each recipe
= Per-regime evaluation: setup
= Per-regime results: where predictors fail
= What the predictions actually look like: cruise
= What the predictions actually look like: warm-up
= What the predictions actually look like: ground
= What the predictions actually look like: full flight
= Sim curriculum predictions are twitchier, not just wronger
= Mean-tracking sanity check
= Conclusion
= Bonus: RPS label refinement -- the idea
= Bonus: two ways to read the spectrogram
= Bonus: validating against a hidden truth
```
### writing/slides/2026-07-18_dregon-analysis-and-generator-design
```
= This week: the rotor comb, three ways
= A physics-structured generator, answerable to data
= Assumption 1 — geometry: a small error is not benign
= Geometry errors found (huge!) in DREGON annotations
= Geometry errors found (silly) in Michael's annotations
= Geometry: fine calibration
= Hypothesis 2 — rotors are individuals
= Treating rotors individually: per-rotor sub-embeddings
= Hypothesis 3: wind noise confusing the model
= What the data said back: generator variants
= Generator improvements: discussion
= Work thread 2: Optimizing for best RPS trajectory
= Work thread 3: a literature baseline (JASA-GP)
= JASA-GP: original data replication
= Adapting the recipe to our use case
= Results
= Discussion
= Initiated work: wider baselines on noise suppression
= Initiated work: RPS predictor achieving parity with VK optimization
= Takeaways
```

## Code changes (summary)

```
 tests/test_wind_wake_gen.py                        |  309 ++
 tests/training/test_val_logging.py                 |    6 +-
 52 files changed, 14927 insertions(+), 40 deletions(-)
     13 src/models
      7 src/data_processing
      4 scripts/vk_bench_cases
      3 src/experiments
      1 src/utils
      1 src/training
      1 scripts/wind_wake_validation.py
      1 scripts/vk_validation.py
      1 scripts/vk_spcup.py
      1 scripts/vk_blind_sweep.py
      1 scripts/vk_blind_annotation.py
      1 scripts/vk_bench.py
      1 scripts/vk_bench_opt_job.sh
      1 scripts/train_sgmse.py
      1 scripts/f1_tables.py
```

## Untracked candidates (not yet committed)

```
  writing/slides/2026-07-18_dregon-analysis-and-generator-design/workflow/slides-notes-source.typ
```

## Prep notes found (read these fully — often a ready-made narrative seed)

- writing/slides/NEXT-DECK-experiment-inventory.md
