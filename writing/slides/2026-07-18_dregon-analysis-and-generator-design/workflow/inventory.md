# Work inventory since last slides

- generated: 2026-07-18T01:25:38+01:00
- boundary artifact: (explicit --since)
- boundary commit: 7e1771d 2026-07-14 writeup agent
- HEAD: 736a639 2026-07-17 Report: proper 8s mid-flight harmonic spectrograms

> Numbers in docs below may predate later fixes — always cross-check the
> newest report/doc before quoting. Sync results before analysis (Rule 5).

## Commits (newest first)

```
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
```

## Experiment configs (conf/experiment/)

```
  A	conf/experiment/gen_v1_corrected.yaml
  A	conf/experiment/gen_v2_perrotor.yaml
  A	conf/experiment/gen_v3_wind.yaml
```

## Docs (docs/) — excerpts for added files

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

## Code changes (summary)

```
 tests/test_wind_wake_gen.py                      |  309 +++
 tests/training/test_val_logging.py               |    6 +-
 21 files changed, 7956 insertions(+), 26 deletions(-)
      5 src/models
      3 src/data_processing
      2 src/experiments
      1 src/training
      1 scripts/wind_wake_validation.py
      1 scripts/vk_validation.py
      1 scripts/vk_spcup.py
      1 scripts/vk_blind_annotation.py
      1 scripts/eval_noise_gen_variants.py
```

## Untracked candidates (not yet committed)

```
  writing/reports/2026-07-18_dregon-analysis-and-generator-design/
  writing/slides/2026-07-18_dregon-analysis-and-generator-design/
```

## Prep notes found (read these fully — often a ready-made narrative seed)

- writing/slides/NEXT-DECK-experiment-inventory.md
