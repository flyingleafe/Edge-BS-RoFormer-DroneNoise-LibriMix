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
| 5 | SGMSE+ | score-based diffusion (generative) | port (sp-uhh/sgmse) | trained FROM SCRATCH. NOTE: prior Diffusion-Buffer batch (a2) was eval-only with a chunking mismatch — nothing reusable; SGMSE+ replaces it as the citation-standard generative baseline |
| — | noisy input, Wiener | floors | trivial | anchors in every table |

Ruled out by prior in-house results — do not re-run: DPTNet/DPRNN/SepFormer
(dual-path without band structure; DPTNet −25 dB vs Edge-BS-RoFormer in
Paper 1), FullSubNet (superseded by TF-GridNet), SEGAN/MetricGAN GAN line
(lost in the 2023 12-model benchmark).

## Data

- **Speech:** `dload:librispeech`. ⚠ Known corrupt flac in the R2 copy
  (`669-129061-0001`, truncated) — the loader has a skip workaround; do not
  "fix" by re-downloading.
- **Pass A noise (uniform over sub-datasets):** DREGON + Michael's noise pools
  (existing Frame sources), `DroneAudioSet`, `SPCUP19-egonoise`,
  `zenodo_drone_noises`, `new-drone-noises`, `drone_audio`.
- **Pass B noise (uniform over CATEGORIES, then uniform within):**
  1. drones (= Pass A pool), 2. MIMII fan, 3. MIMII pump, 4. MIMII valve,
  5. MIMII slider, 6. MIMII-DG, 7. AeroSonicDB (propeller aircraft),
  8. motors (HUSTmotor + KAIST-rotating-acoustic), 9. horns (HornBase).
  Category-uniform weighting is essential — MIMII is 258 GiB and would
  otherwise dominate.
- **Format:** 16 kHz mono (random mic channel per clip from multichannel
  sources), chunk ≈ 2 s, SNR ~ U(−30, 0) dB at train.
- **Augmentations:** random_gain, random_polarity + **time-warp** (α ≤ 1.12,
  E5 recipe — warp the noise clip; no RPS pair to co-warp here).

## Fixed validation sets (build once, publish, pin)

- `SE-valid-drone`: held-out drone noise clips × held-out LibriSpeech
  speakers, SNR grid {−30, −25, −20, −15, −10, −5, 0} dB, deterministic seed,
  ≥50 mixtures per SNR point.
- `SE-valid-harmonic`: same protocol per Pass-B category (held-out clips).
- Held-out means: file-level split within each noise dataset, never reusing a
  source recording between train and valid.
- Publish both via dload (`scripts/publish_frame_datasets.py` pattern), pin in
  `dload.lock`.

## Metrics & reporting

Per-SNR SI-SDR / eSTOI / PESQ for every model, both valids, with noisy +
Wiener anchors in every table. **No aggregate-only claims** — the Gulli-paper
lesson: aggregates and percentage deltas at extreme SNR mislead; per-SNR
absolute numbers only. `eval.py` is the single evaluation entry point.

## Algorithm — straightforward steps

0. **Sanity (worktree):** `.env` present; `.venv` works
   (`.venv/bin/python -c "from utils.paths import get_data_root; print(get_data_root())"`
   must print the MAIN checkout, not the worktree); `dload ls` succeeds.
1. **Infra — SE-target mode** in
   `src/data_processing/online_mixing.py::OnlineMixIterableDataset`: a task
   switch so the stream yields `(mixture, clean_speech)` instead of
   `(audio, rps_target)`; clean target = the gain-scaled speech signal as
   mixed. Unit test: shapes, SNR of returned pair matches the drawn SNR.
2. **Infra — plain-audio noise source** (`kind: audio_pool`): dload-backed
   audio dataset as a noise pool — random file, random channel, resample to
   16 kHz, loop/pad to chunk; no telemetry required. Plus per-source `weight`
   support in the mixed pool (extends E9's MixedNoisePool). Unit tests.
3. **Valid sets:** `scripts/build_se_valid.py` per the spec above → publish
   via dload → pin. Record pins in the batch doc.
4. **Configs:** `conf/online_mix/se_drone_only.yaml` +
   `se_all_harmonic.yaml`; `conf/data/se_baselines_{a,b}.yaml`;
   experiments `conf/experiment/f1_<arch>_{a,b}.yaml` (new `f1` batch).
   ⚠ Pre-commit hook: every `conf/experiment/*.yaml` needs a sibling `.md`
   with `batch:` pointing at an existing batch doc — create
   `docs/experiments/f1-se-blind-baselines.md` FIRST.
5. **Model ports** (use the `reimplement-model` skill; read
   `src/tasks/AGENTS.md` for the task contract first): TF-GridNet (mid-size),
   MP-SENet, SGMSE+. Native 16 kHz (do not resample datasets to match paper
   rates — adapt the models). SGMSE+ is a different training paradigm
   (score-matching objective + reverse-SDE sampling at eval) — expect to add
   a task/loss variant to the training loop; schedule it LAST so the four
   discriminative baselines are never blocked on it.
6. **Smoke runs:** ~500 steps of each `f1_<arch>_a` locally or on gpushort;
   verify loss decreases, wandb logs mixture/enhanced/clean audio samples,
   eval runs end-to-end on `SE-valid-drone`.
7. **Pass A launch:** for each arch:
   `omnirun submit --backend <backend> --gpus 1 --time <t> --yes -- python train.py experiment=f1_<arch>_a`.
   ⚠ omnirun requires a clean, PUSHED HEAD — commit and push the branch
   before submitting. Backends: gpushort (≤1h chunks; rely on
   resume-from-checkpoint), colab/kaggle for long runs, apocrita-long for
   TF-GridNet. `.env` ships with the job automatically.
8. **Pass A eval:** `eval.py` on both valids per arch; assemble the per-SNR
   floor table; sanity-check against anchors (every model must beat Wiener at
   ≥ −10 dB or something is wrong with the run, not the field).
9. **Pass B launch + eval:** same as 7–8 with `f1_<arch>_b`.
10. **Analysis & writeup:** (a) floor tables both passes × both valids;
    (b) diversity verdict: ΔSI-SDR/ΔeSTOI (Pass B − Pass A) on
    `SE-valid-drone` per arch per SNR; (c) per-category transfer table on
    `SE-valid-harmonic`; (d) update `docs/experiments/f1-se-blind-baselines.md`
    with motivation/results/conclusion; (e) full Typst report via the
    `writeup` skill.

Optional arm (only if the diversity result is interesting): Pass B′ holding
one full category out (e.g. AeroSonicDB) → zero-shot cross-category
generalization figure (C3 evidence for the dissertation).

## Compute notes

10 main runs (5 archs × 2 passes) + smokes. Rough per-run budget: DCUNet /
MP-SENet ~hours-class; Edge-BS-RoFormer similar; TF-GridNet the heaviest
(mid-size config + apocrita-long); SGMSE+ long (diffusion training) — start
it early once its port lands. Checkpoints upload to R2/wandb per the training
loop (`best.ckpt` + `last.ckpt`); results sync back via `omnirun pull`.

## Known gotchas (inherited project knowledge)

- Worktree: `data/`, `datasets/`, `results/` live in the MAIN checkout;
  `utils.paths` + `DATA_ROOT` in `.env` handle this (verify in step 0).
- omnirun invocation: `uvx -p 3.12 --from ~/Projects/omnirun omnirun ...`;
  if SSH auth hangs, `omnirun backends check` revives the ControlMaster.
- kaggle: ~1 MB kernel source cap (slim-snapshot clone recipe);
  "cancelled" kaggle jobs silently hold GPU slots — verify via wandb, not
  `omnirun status`.
- `amp: false` for complex-valued models (ComplexHalf gaps in torch).
- Log val audio under `samples/` in wandb so the `val/loss` chart stays
  visible (E-series lesson).
