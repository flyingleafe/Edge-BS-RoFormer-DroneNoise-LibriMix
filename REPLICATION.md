# Replicating the historical experiment catalogue

This document maps every historical experiment (Paper 1, Paper 2, the RPS-prediction
line, and the noise-generation line) onto the unified Hydra framework
(`docs/refactor-unified-framework.md`). For each family: the historical entry
point/command, the new-framework command, the dataset needed (+ how to build
it), where the original results live, and replicability status/caveats.

All new config files live under `conf/data/`, `conf/model/`, `conf/loss/`,
`conf/metrics/`, `conf/experiment/`. Legacy per-model YAMLs (`configs/1_*.yaml`
… `configs/14a_*.yaml`) remain valid and are referenced via
`model.legacy_config_path` + `model.model_type` (the `instantiate_model`
dispatch in `src/training/config.py`).

**Scope note on `train.py`/`final_valid.py`/`train_rps_predictor.py`/
`train_noise_generation.py`**: these historical entry-point scripts are
deleted on this branch (`docs/refactor-unified-framework.md` § "Execution
waves"). Every "historical command" below is dead as written — it documents
*intent and hyperparameters*, not something runnable today. The
"new-framework command" column is the replacement.

## Composition check

`scripts/check_experiment_configs.py` composes every `conf/experiment/*.yaml`
against `conf/config.yaml` + the registered `RootConfig` structured schema
(`training.config.register_configs`), via Hydra's `compose()` API, then fully
resolves the result (`OmegaConf.to_container(..., resolve=True,
throw_on_missing=True)`). This catches missing-mandatory-value, wrong-type,
and unknown-key errors — everything `@hydra.main` would catch at compose
time — **without** instantiating any dataset/model/loss (no training data
lives on this machine).

```
uv run python scripts/check_experiment_configs.py
```

Result as of this writing: **48/48 experiments compose OK** (all families
below, including the pre-existing `rps_simple_conv_v2_v4` and the C7/C8/C9/
E2/E3 experiments added to close the "needs follow-up wiring" gaps).

This is *not* a substitute for real spec validation. On a data-ful machine,
run the actual per-experiment check, which also runs a one-batch CPU smoke
forward pass (`training.validate.validate_config`):

```
python train.py experiment=<name> validate_only=true
```

## A1 — Paper 1: Edge-BS-RoFormer + baselines (DN-LM)

| | |
|---|---|
| **Dataset** | DN-LM (DroneNoise-LibriMix). Per-sample `sample_NNNNN/{mixture.wav,vocals.wav,noise.wav}` + per-split `metadata.json`. **No RPS label.** |
| **Build** | `python create_dataset.py --speech_dir <LibriSpeech train-clean-100> --noise_dir <drone-audio source> --output_dir datasets/DN-LM --train_samples 6480 --valid_samples 720 --sample_duration 1.0 --sample_rate 16000 --snr_min -30 --snr_max 0 --seed 42` (README.md's historical sample counts; `replicate_paper.py`'s current default is `DATASET_TRAIN_SAMPLES=64800` — a 10× discrepancy between the README and the live reproduction script; treat 64800 as current-of-record if reproducing via `replicate_paper.py`). Noise source moved from a local dir to HF `geronimobasso/drone-audio-detection-samples` + a Zenodo `all_drone_noises.zip` mirror over time — check `create_dataset.py --help` for the current `--noise_dir` syntax (`hf:`/`hf-local:` prefixes supported). |
| **conf/data** | `conf/data/dn_lm.yaml`, via the **new** `data_processing.frame_datasets.DNLMFrameDataset` (mirrors `DregonLMFrameDataset`; emits `{mixture, target, meta}` — no `rps`). |
| **Results** | README.md's own Results table (SI-SDR/PESQ vs DCUNet/HTDemucs at −15 dB SNR; Jetson AGX Xavier edge deployment: RTF 0.325, 8.5 MB, <500 MB runtime). `replicate_paper.sh`/`.py` (repo root) are the reproduction drivers, results under `results/<subdir>_<githash>/`. |

| Variant | Historical command | New-framework command |
|---|---|---|
| 1_Nothing (ablation floor) | `train.py --model_type edge_bs_rof --config_path configs/1_Nothing.yaml --data_path datasets/DN-LM/train --valid_path datasets/DN-LM/valid` | `python train.py experiment=a1_edge_bs_rof_nothing` |
| 2_FA (+ flash-attn) | same, `configs/2_FA.yaml` | `python train.py experiment=a1_edge_bs_rof_fa` |
| 3_FA_RoPE(48) | same, `configs/"3_FA_RoPE(48).yaml"` | `python train.py experiment=a1_edge_bs_rof_fa_rope48` |
| 3_FA_RoPE(64) — **headline model** | same, `configs/"3_FA_RoPE(64).yaml"`, `--device_ids 0 1` | `python train.py experiment=a1_edge_bs_rof_fa_rope64` |
| DCUNet baseline | `--model_type dcunet --config_path configs/5_Baseline_dcunet.yaml` | `python train.py experiment=a1_baseline_dcunet` |
| DPTNet baseline | `--model_type dptnet --config_path configs/7_Baseline_dptnet.yaml` | `python train.py experiment=a1_baseline_dptnet` |
| HTDemucs baseline | `--model_type htdemucs --config_path configs/8_Baseline_htdemucs.yaml` | `python train.py experiment=a1_baseline_htdemucs` |

**Replicability status: config-complete, blocked on data.** DN-LM does not
exist on this machine; `DNLMFrameDataset` is new code (untested against real
data — CPU-composed only, see Composition check above). **Loss caveat**: the
historical ZFTurbo-style trainer used an implicit multi-instrument
SI-SDR-scheduled loss; no `conf/loss` entry replicates that exactly. This
encoding uses `conf/loss/masked_mse.yaml` (quantile-masked MSE,
`target_key=target`, matching `DNLMFrameDataset`'s emitted key) + monitor
`si_sdr` (`conf/metrics/separation_basic.yaml`) — a reasonable but
**documented deviation**, not a verified-equivalent reproduction of the
paper's exact training loss.

## A2 — Diffusion Buffer BBED

| | |
|---|---|
| **Dataset** | Same DN-LM as A1 (`conf/data/dn_lm.yaml`). |
| **Model** | `configs/9_Diffusion_Buffer_BBED.yaml` (`model_type: diffusion_buffer`) — paper-exact hyperparams (n_fft/hop=510/256, differs from A1's 2048/512; `buffer_size=20`; SDE `bbed`, c=0.08, k=2.6). |
| **New-framework command** | `python train.py experiment=a2_diffusion_buffer_bbed` |
| **Results** | No standalone report; `docs/diffusion-buffer-paper.md` (paper transcription) + `docs/diffusion-prompt.md` (implementation notes + repo config mapping). |

**Replicability status: config-complete, but historically eval-only in this
repo.** `replicate_paper.py`'s `EVAL_ONLY_MODELS` treats this as a
pretrained-checkpoint import (from `sp-uhh/Diffusion-Buffer` upstream), not a
from-scratch training target — no training step was ever defined for it
here. **Known chunking mismatch** (`docs/diffusion-prompt.md`): DN-LM's
native 1 s/16000-sample clips don't match the paper's K=128-frame (~2 s)
requirement at hop=256; this was never resolved in-repo. Treat
`a2_diffusion_buffer_bbed` as a from-scratch training config that has never
actually been run start-to-finish — use with caution, prefer importing the
upstream checkpoint for eval-only work.

## B1 — DCUNet/DCCRN + encoder-side RPS fusion on DREGON-LM (Gulli et al. EURASIP 2025 replication)

| | |
|---|---|
| **Dataset** | DREGON-LM (mono), `datasets/DREGON-LM/{train,valid}`. |
| **Build** | `python create_dregon_librimix.py` (no flags). |
| **conf/data** | `conf/data/dregon_lm_v1.yaml`. |
| **Results** | `experiments/eval_dccrn_baseline_dregon.yaml` → `results/02b92aae2f89/training/best_model.ckpt`; `experiments/eval_dccrn_rps_dregon.yaml` → `results/85eddadc0e33/training/best_model.ckpt`; `experiments/eval_dcunet_baseline_dregon.yaml` → `results/dcunet_baseline_dregon/model_dcunet_ep_119_sdr_1.2846.ckpt`; `experiments/eval_dcunet_rps_dregon.yaml` → `results/dcunet_rps_dregon/model_dcunet_ep_124_sdr_1.2997.ckpt`. **These paths are stale/historical** — re-verify existence before reuse (`experiments/AGENTS.md` gotcha). |

| Variant | Legacy config | New-framework command |
|---|---|---|
| DCUNet + RPS, bottleneck (6-layer) — dataset ambiguous, see caveat | `configs/5a_DCUNet_RPS_bottleneck.yaml` | `python train.py experiment=b1_dcunet_rps_bottleneck_5a` |
| DCUNet + RPS, GRU (5-layer) — dataset ambiguous | `configs/5b_DCUNet_RPS_gru.yaml` | `python train.py experiment=b1_dcunet_rps_gru_5b` |
| DCUNet + RPS, hierarchical (5-layer) — dataset ambiguous | `configs/5c_DCUNet_RPS_hierarchical.yaml` | `python train.py experiment=b1_dcunet_rps_hierarchical_5c` |
| RPS-DCUN6-P (bottleneck, on DREGON) | `configs/6a_DCUNet_RPS_DREGON_bottleneck.yaml` | `python train.py experiment=b1_dcunet_rps_dregon_bottleneck` |
| RPS-DCUN5 (GRU, on DREGON) | `configs/6b_DCUNet_RPS_DREGON_gru.yaml` | `python train.py experiment=b1_dcunet_rps_dregon_gru` |
| RPS-DCUN5-H (hierarchical, on DREGON) | `configs/6c_DCUNet_RPS_DREGON_hierarchical.yaml` | `python train.py experiment=b1_dcunet_rps_dregon_hierarchical` |
| DCUNet + bottleneck RPS (5-layer variant) | `configs/7a_DCUNet_RPS_DREGON.yaml` | `python train.py experiment=b1_dcunet_rps_dregon` |
| DCUNet baseline, no RPS | `configs/7b_DCUNet_baseline_DREGON.yaml` | `python train.py experiment=b1_dcunet_baseline_dregon` |
| DCUNet + RPS cond. + aux RPS head (weight 0.5) | `configs/7c_DCUNet_RPS_PredRPS_DREGON.yaml` | `python train.py experiment=b1_dcunet_rps_predrps_dregon` |
| DCCRN baseline, no RPS | `configs/10a_DCCRN_baseline_DREGON.yaml` | `python train.py experiment=b1_dccrn_baseline_dregon` |
| DCCRN + GRU-based RPS fusion | `configs/10b_DCCRN_RPS_DREGON.yaml` | `python train.py experiment=b1_dccrn_rps_dregon` |
| DCCRNLite + RPS (embedded variant) | `configs/10c_DCCRNLite_RPS_DREGON.yaml` | `python train.py experiment=b1_dccrn_lite_rps_dregon` |
| DCCRN + RPS cond. + aux RPS head (weight 2.0) | `configs/10d_DCCRN_RPS_PredRPS_DREGON.yaml` | `python train.py experiment=b1_dccrn_rps_predrps_dregon` |
| DCUNet encoder + RPSPredictionHead, RPS-only loss | `configs/11a_DCUNet_RPSOnly_DREGON.yaml` | `python train.py experiment=b1_dcunet_rpsonly_dregon` |
| DCCRN encoder + RPSPredictionHead, RPS-only loss | `configs/11b_DCCRN_RPSOnly_DREGON.yaml` | `python train.py experiment=b1_dccrn_rpsonly_dregon` |
| Standalone RPS predictor (legacy `rps_predictor` model_type) | `configs/11c_RPSPredictor_DREGON.yaml` | `python train.py experiment=b1_legacy_rps_predictor_dregon` |

**Replicability status: config-complete, blocked on `datasets/DREGON-LM`
existing locally.** `experiments/*.yaml` (legacy reproducibility records) map
1:1 onto most of the above; `experiments/AGENTS.md` documents that schema
(not machine-consumed, translate manually). **Caveat — 5a/5b/5c dataset is
ambiguous**: these three configs have no `data_path` field at all (unlike
the `_DREGON`-suffixed 6-series), suggesting they predate the DREGON dataset
entirely (possibly an earlier synthetic-RPS DN-LM-style dataset). Best-effort
mapped to `dregon_lm_v1` here; likely superseded in practice by 6a/6b/6c —
treat 5a-c's specific numeric results (if any exist) with caution.
**Composite loss weights for PredRPS variants** (`conf/loss/masked_mse_plus_pit_rps_w0p5.yaml`,
`..._w2.yaml`) are read off inline comments in the legacy YAMLs, not a
separately-documented field — re-verify against the legacy file's
`training.rps_aux_weight` before trusting exactly.

## B2 — Refactored DCUNet/DCCRN, decoder-only RPS + auxiliary RPS head

Design doc: `docs/dcunet-refactored.md` (read in full before using this
family) — `EncoderModule` (no RPS) + `DecoderModule` (RPS optional,
`bottleneck`/`hierarchical` fusion). Same dataset as B1
(`conf/data/dregon_lm_v1.yaml`, `datasets/DREGON-LM/{train,valid}`).

| Variant | Legacy config | New-framework command |
|---|---|---|
| DCUNetRefactored baseline, no RPS | `configs/12a_DCUNetRefactored_baseline.yaml` (= existing `conf/model/dcunet.yaml`) | `python train.py experiment=b2_dcunet_refactored_baseline` |
| Decoder bottleneck RPS fusion | `configs/12b_DCUNetRefactored_decoder_bottleneck.yaml` | `python train.py experiment=b2_dcunet_refactored_decoder_bottleneck` |
| Decoder hierarchical RPS fusion | `configs/12c_DCUNetRefactored_decoder_hierarchical.yaml` | `python train.py experiment=b2_dcunet_refactored_decoder_hierarchical` |
| Decoder bottleneck + aux RPS head (weight 0.1) | `configs/13a_DCUNetRefactored_PredRPS_bottleneck.yaml` | `python train.py experiment=b2_dcunet_refactored_predrps_bottleneck` |
| Decoder hierarchical + aux RPS head (weight 0.1) | `configs/13b_DCUNetRefactored_PredRPS_hierarchical.yaml` | `python train.py experiment=b2_dcunet_refactored_predrps_hierarchical` |
| DCCRNRefactored, decoder bottleneck + aux RPS head | `configs/13c_DCCRNRefactored_PredRPS_bottleneck.yaml` | `python train.py experiment=b2_dccrn_refactored_predrps_bottleneck` |
| DCCRNRefactored, decoder hierarchical + aux RPS head | `configs/13d_DCCRNRefactored_PredRPS_hierarchical.yaml` | `python train.py experiment=b2_dccrn_refactored_predrps_hierarchical` |

Historical commands (dead, scripts deleted), for reference:
```
python train.py --model_type dcunet_refactored --config_path configs/12c_DCUNetRefactored_decoder_hierarchical.yaml \
  --results_path results/dcunet_refactored_hierarchical --data_path datasets/DREGON-LM/train --valid_path datasets/DREGON-LM/valid
python final_valid.py --model_type dcunet_refactored --config_path configs/12c_... \
  --start_check_point results/.../best_model.ckpt --valid_path datasets/DREGON-LM/valid --metrics si_sdr sdr pesq stoi
```

**Replicability status: config-complete, blocked on data.** `b2_dcunet_refactored_baseline`
reuses the pre-existing `conf/model/dcunet.yaml` (no duplicate model file
created). **Caveat**: 13c/13d's exact `rps_aux_weight` is not separately
documented in the legacy config comments; assumed identical to 13a/13b's 0.1
(`conf/loss/masked_mse_plus_pit_rps_w0p1.yaml` used for all four) — verify
against the legacy YAML before trusting exactly.

## C1 / C3 / C6 — first RPS predictor + 10-variant SimpleConv arch-zoo sweep (DREGON-LM v1)

These three labels (from the original recon) refer to the **same historical
sweep event**: `.pi/checkpoints/autoresearch-simpleconv-variants.md` +
`writing/reports/2026-05-30_simpleconv-variants/report.typ`, on the original
`datasets/DREGON-LM` (v1) dataset. `simple_conv` is both the literal "first"
RPS predictor trained (C1) and the sweep's baseline (C3/C6).

| | |
|---|---|
| **Dataset** | `datasets/DREGON-LM` (mono), same as B1/B2 — `conf/data/dregon_lm_v1.yaml`. |
| **Historical command** | `train_rps_predictor.py --model <key> --epochs 200 --patience 15 --batch_size 16 --lr 0.001 --weight_decay 0.0001 --grad_clip 5.0 --save_path results/rps_exp_<dir>` |
| **New-framework command** | `python train.py experiment=c1_c3_c6_simpleconv_arch_sweep model=<key>` |
| **10-key list** | `simple_conv` (default), `simple_conv_bigru`, `simple_conv_bigru_v2`, `simple_conv_v2`, `simple_conv_tcn`, `simple_conv_magphase_bigru`, `simple_conv_attn_pool`, `simple_conv_wide`, `simple_conv_multiscale`, `simple_conv_se_next` — every key has a `conf/model/<key>.yaml`. |
| **Results** | `writing/reports/2026-05-30_simpleconv-variants/report.typ`. Best: `simple_conv_v2` (MSE 2.61, R² 0.951); Pareto-optimal `simple_conv_bigru` (663K params, 99.4% of v2's score at 44% of params); `simple_conv_se_next` actively harmful (R² 0.688, unstable). |

**Replicability status: config-complete, blocked on data.**
**Caveat — dataset-flawed**: this dataset was later flagged "trivially easy"
(short/measured RPS, train/valid overlap risk) once DREGON-LM-V2 existed for
comparison — kept here for historical value and as the C4 comparison
baseline, not recommended for new work (see C4 below).

## C2 — Classical (non-learned) RPS baselines

Not a `train.py` experiment — a comparison script against fixed,
non-neural signal-processing trackers (`pyin_single_f0`, `cepstral_tracker`,
`hps_tracker`, `matched_filter_tracker`, `nmf_tracker`; root
`classical_rps_predictors.py`) vs. an existing learned checkpoint.

- **Reproduce**: `python writing/papers/classical_baselines_report/build.py`
  (uses `tasks.rps_prediction.{evaluate,load_input_set,load_predictor}`,
  checkpoint `results/rps_predictor/best.pt`, data
  `datasets/DREGON-LM-test/valid`, single-rotor recordings from
  `data/DREGON/DREGON_individual_motors_recordings`). Interactive variant:
  `notebooks/classical_baselines_comparison.ipynb`.
- **Results**: `writing/papers/classical_baselines_report/{main.tex,figures/}`.
  Classical methods MSE 870–1380 vs. SimpleConv; NMF is the only
  true multi-pitch classical method but still mixed results.
- **Status**: script-based, out of scope for `conf/experiment/*.yaml` — no
  train.py routing needed or attempted.

## C4 — DREGON-LM-V2 motor-combo-fraction sweep

| | |
|---|---|
| **Dataset** | DREGON-LM-V2-{0,2.5,5,20}pct, `datasets/DREGON-LM-V2-<pct>/{train,valid}` — 6000 train/600 valid, 3 s, 8-mic, command RPS. |
| **Build** | `python create_dregon_librimix.py --speech_dir data/librispeech/LibriSpeech/train-clean-100 --dregon_dir data/DREGON --output_dir datasets/DREGON-LM-V2-<pct> --num_train 6000 --num_valid 600 --motor_combo_fraction <0.0\|0.025\|0.05\|0.20>` |
| **conf/data** | `conf/data/dregon_lm_v2_{0pct,2p5pct,5pct,20pct}.yaml` (`channel: 0` — 8-mic file, mono model, no channel-flattening trick). |
| **Historical command** | `train_rps_predictor.py --model simple_conv_bigru_v2 --data_root datasets/DREGON-LM-V2-2.5pct --epochs 500 --patience 30 --batch_size 96 --lr 1e-3` |
| **New-framework command** | `python train.py experiment=c4_dregon_v2_motorcombo_sweep` (default: 2.5pct, best); override `data=dregon_lm_v2_0pct` / `dregon_lm_v2_5pct` / `dregon_lm_v2_20pct` for the other points. |
| **Results** | `.pi/checkpoints/dregon-lm-v2-rps-training.md`. PIT MSE: 0%→117.3 (collapses, no PIT anchor), 2.5%→**56.70** (best), 5%→65.9, 20%→71.1 — U-shaped. Checkpoint `results/rps_predictor_v4_2.5pct/simple_conv_bigru_v2/best_simple_conv_bigru_v2.pt`. Cross-eval vs. the "OLD" (v1) checkpoints: `results/rps_cross_eval/validation_metrics.json` (63–123× degradation off-domain — this is the "dataset-flawed" finding for C1/C3/C6 above). |

**Replicability status: config-complete, blocked on data.**

## C5 — DREGON-LM-V3 baseline

| | |
|---|---|
| **Dataset** | `datasets/DREGON-LM-V3` — 6000/600, 1 s @16kHz mono, SNR [-30,0]. |
| **Build** | `python create_dregon_librimix_v3.py` (superseded script; if absent, closest modern equivalent: `create_dregon_librimix.py --duration 1.0 --num_train 6000 --num_valid 600`). |
| **conf/data** | `conf/data/dregon_lm_v3.yaml`. |
| **Historical command** | `train_rps_predictor.py --model simple_conv --data_root datasets/DREGON-LM-V3 --no_pit_loss --epochs 200 --patience 30 --batch_size 128` |
| **New-framework command** | `python train.py experiment=c5_simpleconv_dregon_v3` |
| **Results** | `.pi/checkpoints/dregon-lm-v3-baseline.md`. Val MSE 227.0 (RMSE 15.1), MAE/clip 8.14 RPS. Checkpoint `results/rps_predictor_comparison/best_simple_conv.pt`. wandb: `flyingleafe/rps-prediction/runs/ivbyimpe` (legacy project, distinct from the new framework's `harmonic-noise-suppression` project). |

**Replicability status: config-complete, blocked on data. Documented
deviation**: historical training used plain (non-PIT) MSE (`--no_pit_loss`);
`c5_simpleconv_dregon_v3` uses `pit_mse` instead — current best practice per
the C9 finding that PIT is the correct objective, not a workaround (see C9
below). No plain-MSE-on-RPS loss class exists in `src/losses/`; adding one
was judged out of scope (would replicate an objective already superseded by
better practice). **R² is degenerate** on this dataset (1 s clips → near-
constant RPS → SS_total≈0); trust MSE/MAE.

## C7 / C8 — Salience-map RPS baselines (multif0/basic_pitch)

The gap this section used to describe (**"needs follow-up wiring"**) is
closed: the BCE-target-from-RPS derivation now lives declaratively in
`losses.SalienceRPSBCELoss` — it owns the salience-grid parameters
(`fmin`/`n_octaves`/`over_sample`/`n_bins`/`bins_per_octave` for the
log-spaced CQT grid, or `out_fmin`/`out_fmax`/`out_bins` for a decoupled
linear super-resolution grid) and derives `target["salience"]` from
`target["rps"]` on every call, through the exact same core math the model's
own `SalienceRPSPredictor.salience_target_from_frame_rps` uses — both now
funnel through `models.multif0.utils.salience_target_from_resampled_rps`
(extracted so the two never drift; see `tests/models/test_salience_rps.py`
for the equivalence proof). `metrics.SalienceBCEMetric` mirrors the loss for
`optim.monitor` to track BCE-on-validation (rather than the `monitor: loss`
fallback, which tracks *train* loss — see `training/loop.py::run_training`).

| | |
|---|---|
| **Dataset** | `datasets/DREGON-LM-V4`, 8-channel-as-extra-batch-item — `conf/data/dregon_lm_v4_8ch_flat.yaml` (`DregonLMFrameDataset(flatten_channels=True)`, new C9 flag, shared with C9 below). Matches the historical training loop's `_flatten_channels` call, confirmed by `.pi/checkpoints/salience-baselines-dregon-v4-report.md` ("30 clips x 8 channels (channels flattened into the batch via _flatten_channels)"). |
| **Historical command (baseline)** | `train_rps_predictor.py --model {multif0_salience,basic_pitch_salience} --data_root datasets/DREGON-LM-V4 --device cuda:0 --epochs 200 --patience 15` (`--salience_blur_bins` default 2, `--bce_pos_weight` default 0 = auto). |
| **New-framework command (baseline)** | `python train.py experiment=c7_multif0_salience` / `python train.py experiment=c8_basic_pitch_salience` |
| **Historical command (narrow + super-res, `.pi/checkpoints/salience-narrow-superres-experiment.md`)** | multif0: `--hcqt_fmin 55 --hcqt_n_octaves 1 --hcqt_over_sample 10 --hcqt_harmonics 1 2 3 4 --superres_out --out_fmin 55 --out_fmax 110 --out_bins 360 --salience_blur_bins 2`; basic_pitch: `--bp_fmin 55 --bp_bins_per_semitone 4 --bp_n_contour_semitones 12` + the same `--superres_out` flags. |
| **New-framework command (narrow + super-res)** | `python train.py experiment=c7_multif0_salience_narrow_sr` / `python train.py experiment=c8_basic_pitch_salience_narrow_sr` |
| **conf** | Models: `conf/model/{multif0_salience,basic_pitch_salience}{,_narrow_sr}.yaml`. Losses/metrics: `conf/{loss,metrics}/salience_bce_{multif0,basic_pitch,narrow_sr}.yaml` (one pair per grid, since the BCE target's bin count/positions are grid-specific). |
| **Results** | `.pi/checkpoints/salience-baselines-dregon-v4-report.md` (baseline: multif0 RMSE 6.30/R² 0.19, basic_pitch RMSE 23.24/R² −16.21 — via the *old* eval pipeline's Hungarian-tracked `predict_rps()`, not reproduced by the new framework yet, see caveat below). |

**Replicability status: config-complete, blocked on data.** **Caveat —
`multif0_salience`'s baseline `fmin`**: the historical headline checkpoint
was trained at `fmin=32.7` (C1, 3 harmonics), not the model class's own code
default (`fmin=27.5`, 4 harmonics) — `conf/model/multif0_salience.yaml`
matches the checkpoint-compatible `32.7`, documented inline. **Caveat — no
RPS-space eval metric**: `predict_rps()`'s `sigmoid → salience_to_rps_segmented`
(Hungarian tracking) → PIT-MSE/RMSE/MAE/R² comparison to the SimpleConv
family (what the historical `results/dregon_v4_eval/salience_baselines_final_valid.json`
numbers above measure) is still not wired into `MetricSuite` — `optim.monitor`
here is the BCE objective itself (`metrics.SalienceBCEMetric`), not an
RPS-space metric. Wiring a `predict_rps()`-based metric is a natural
follow-up but out of scope here (it needs per-sample, not per-batch,
tracking — `MetricSuite` evaluates one Frame pair at a time).

## C9 — 8-channel retrain / channel-generalization finding

Closed: `DregonLMFrameDataset` gained `flatten_channels: bool = False`
(`src/data_processing/frame_datasets.py`) — when true, each multichannel
sample expands into one mono-view Frame per mic (`len(dataset) = n_samples *
n_channels`), broadcasting the recording's single RPS target across all mic
views and recording which mic each item came from in `meta.channel`. This
reproduces the legacy `train_rps_predictor.py::_flatten_channels`
`(B, C, T) -> (B*C, T)` batch-flatten trick at the *data* level instead of
inside the training loop.

| | |
|---|---|
| **Dataset** | `datasets/DREGON-LM-V4` — `conf/data/dregon_lm_v4_8ch_flat.yaml` (shared with C7/C8 above). |
| **New-framework command** | `python train.py experiment=c9_simple_conv_v2_8ch` |

**Replicability status: config-complete, blocked on data.** The
single-channel path (`channel: 0`) remains covered by the pre-existing
`rps_simple_conv_v2_v4` experiment and every V4-based experiment; this adds
the 8-channel-as-batch alternative alongside it, not in place of it.

## C10 — 26-variant SimpleConv architecture sweep (offline + online-mix) + clipped-GRU follow-up

Source: `autoresearch/20260617-012233-dregon-lm-v4-michaels-simple-conv-v2/{experiments.md,leaderboard.md}`,
`writing/reports/2026-06-19_rps-arch-sweep-v4-michaels/report.typ`.

| | |
|---|---|
| **Dataset (offline)** | `datasets/DREGON-LM-V4-michaels/{train,valid}` — `conf/data/dregon_lm_v4_michaels.yaml`. |
| **Dataset (online-mix)** | streamed via `configs/online_mix_v4_michaels_train_no_room1_gpfs.yaml`, fixed valid = same as offline — `conf/data/online_mix_v4_michaels.yaml` (pre-existing). |
| **Build** | `python create_dregon_librimix.py --multichannel --real_valid --max_non_overlapping --output_dir datasets/DREGON-LM-V4-michaels ... --train_noise_sources "dregon-split:in_flight_noise,michaels:FLY125" --valid_noise_sources "dregon-id:free-flight_nosource_room1,...,michaels:FLY124"` (full command in `conf/data/dregon_lm_v4_michaels.yaml`'s header comment). |
| **New-framework command (offline)** | `python train.py experiment=c10_arch_sweep_offline model=<key>` |
| **New-framework command (online)** | `python train.py experiment=c10_arch_sweep_online model=<key>` |
| **Winner (named)** | `python train.py experiment=c10_uni_gru128_online` (mirrors the worked example in `docs/refactor-unified-framework.md`) |
| **Clipped-GRU follow-up** | `python train.py experiment=c10_arch_sweep_online model=simple_conv_v2_uni_gru grad_clip=0.5` (or `..._offline`, any `uni_gru*` key) |
| **26-key list** | `simple_conv_v2`, `simple_conv_v2_smol_causal_tcn`, `simple_conv_v2_gru96`, `simple_conv_v2_dwt`, `simple_conv_v2_smol_tcn`, `simple_conv_v2_multires`, `simple_conv_v2_dual_pool`, `simple_conv_v2_magphase`, `simple_conv_v2_tcn`, `simple_conv_v2_smol_bigru`, `simple_conv_v2_uni_gru96_norm_do03`, `simple_conv_v2_causal_tcn`, `smolnet_rps_tcn`, `simple_conv_v2_local_attn`, `simple_conv_v2_uni_gru128_norm`, `simple_conv_tcn`, `simple_conv_v2_uni_gru128`, `simple_conv_v2_transformer`, `smolnet_rps_causal_tcn`, `simple_conv_v2_uni_gru96_norm_do02`, `simple_conv_v2_causal_gru`, `simple_conv_v2_uni_gru64_norm_do03`, `smolnet_rps_simple_head`, `simple_conv_v2_uni_gru128_norm_do03`, `simple_conv_v2_uni_gru`, `simple_conv_v2_causal_gru96` — every key has a `conf/model/<key>.yaml`. |
| **Results** | **On HPC scratch, not synced to the repo**: `/gpfs/scratch/acw592/results/autoresearch/20260617-012233-dregon-lm-v4-michaels-simple-conv-v2/` (offline), `/gpfs/scratch/acw592/results/autoresearch/20260618-v4-michaels-online-mix-200ep-aug50k-gpushort/` (online), clipped-GRU subfolder alongside. Offline best: `simple_conv_v2` PIT MSE **7.89**, R² 0.82 (best R²: `simple_conv_v2_smol_causal_tcn`, 8.38/0.83). Online best: `simple_conv_v2_uni_gru128` PIT MSE **7.33**, R² 0.82. Clipped follow-up: offline only `uni_gru128` trains cleanly (10.40/0.78), online `uni_gru` best-clipped (8.54/0.80, still below the unclipped winner). |

**Replicability status: config-complete, blocked on data + results not
synced from HPC scratch.** Naming caveat: `src/data_processing/AGENTS.md`'s
own composable-noise-pool worked example uses `--output_dir
datasets/DREGON-LM-V5` for what is otherwise the "V4-michaels" recipe —
verify the actual `--output_dir` used historically before re-running matches
`DREGON-LM-V4-michaels` exactly.

## C11 — FLY124/FLY125 cross-drone

Source: `writing/reports/2026-06-15/report.typ`.

**Step 1 (zero-shot cross-drone eval, no retrain) — needs follow-up wiring**,
see below.

**Step 2 (retrain on DREGON+FLY125)**:

| | |
|---|---|
| **Dataset** | Same `datasets/DREGON-LM-V4-michaels` as C10 offline. |
| **Model** | `simple_conv_v2` (not swept). |
| **New-framework command** | `python train.py experiment=c11_dregon_fly125_retrain` |
| **Results** | `results/fly125_simpleconvv2_eval/metrics.json` (script gitignored/not found in repo — `run_eval.py` referenced only in user memory notes, path unverified). FLY124 cross-drone: PIT RMSE 7.96→**1.63** Hz, R² median 0.52→**0.96**. DREGON-LM-V4 in-domain: PIT RMSE regresses 1.62→2.77 Hz (attributed to an early-stopping artifact at epoch 20, not a capacity tradeoff). |

**Replicability status: config-complete for step 2, blocked on data.**
`c11_dregon_fly125_retrain` composes identically to `c10_arch_sweep_offline`'s
default (same dataset, same default model) but is kept as a separately-named
experiment to match the historical wandb-run/results-dir naming.

## E1 — single-observer noise-gen (harmonic oscillator bank) — **not replicated, superseded**

`DroneNoisePlusFilterGen` (`src/models/generative/filtered_noise.py`) is
explicitly marked "port-only" in `src/models/AGENTS.md` — never wired to
*any* training entry point in this repo, unified or legacy. No
`.pi/checkpoints/*.md` note documents a completed run; no `results/noise_gen/`
directory exists in the current worktree. Likely superseded by E2/E3 in
practice. **Not routable** (see "Needs follow-up wiring" — same three gaps
as E2/E3, plus the model class isn't even in
`models.registry.NOISE_GEN_MODEL_REGISTRY`). Recorded here as a dead end,
not actively ported.

## E2 / E3 — positional multi-observer noise-gen (original + swapped-split/smoothness)

The three gaps this section used to describe (**"needs follow-up wiring"**)
are closed:

1. **Codec/model signature fix** (`tasks.codecs.NoiseGenerationCodec`): now
   computes `rel_pos = tasks.noise_generation.geometry_to_rel_pos(mic_pos,
   rotor_pos)` (extended to a batched-tensor path, `(B,M,3)/(B,R,3) ->
   (B,M,R,3)`, differentiable, alongside the original unbatched-numpy path)
   instead of passing `mic_pos`/`rotor_pos`/`drone_id` straight through as
   kwargs the model's `forward(rps, rel_pos, z=None, ...)` never accepted.
2. **Frame dataset adapter**: `data_processing.frame_datasets.NoiseGenFrameDataset`
   wraps the existing `noise_rps_dataset.NoiseRPSDataset`/
   `build_noise_rps_datasets` (DREGON `in_flight_noise` + Michael's, reused
   verbatim) and attaches `mic_pos`/`rotor_pos` geometry + a `meta.drone`
   name (`"dregon"`/`"michaels"`, straight from the inner dataset's chunk
   `origin`). **Caveat**: `NoiseRPSDataset` already reduces each draw to one
   selected audio channel without reporting which physical index was
   picked, so this adapter only supports `channel_policy="first"` — training
   is against a single microphone, not the full 8-mic array the historical
   online trainer rendered jointly (native multi-observer). Extending
   `NoiseRPSDataset` to report the drawn channel index is the natural
   follow-up for full multichannel noise-gen training.
3. **Per-drone conditioning**: the deleted `train_noise_generation.py` kept
   `tasks.noise_generation.DroneCodebook` fully external to "the model" (its
   own optimizer param group, its own bundle-file entry). The unified
   `training.loop.run_training` has a narrower single-model contract — one
   `optimizer = get_optimizer(model, ...)` over `model.parameters()`, one
   checkpoint = `model.state_dict()` — so `models.registry.build_noise_gen_model(...,
   cond_dim=16, drone_names=[...])` now returns a composite
   `_CodebookConditionedNoiseGen` (generator + a trainable `DroneCodebook`
   submodule) instead; `NoiseGenerationCodec(conditioned=True)` resolves
   each sample's `drone_names` from `meta.drone` and passes them through
   (`model(rps, rel_pos, drone_names)`), and the model's own codebook
   resolves `z`. E3's smoothness regularisers additionally needed
   `losses.SmoothnessPenalty` generalized with a `series_dims`/`series_time`
   override (default unchanged) so the same Frame adapter can target
   `harm_amps`/`noise_amps` (exposed via `NoiseGenerationCodec(return_dict=True)`
   as extra pred entries, not just `rps_pred`), and `conf/loss/multiscale_stft.yaml`
   (+ `_smoothness` composite variant) for `losses.MultiScaleSTFTLoss`.

| | |
|---|---|
| **Dataset** | DREGON `in_flight_noise` + Michael's, via `NoiseGenFrameDataset` — `conf/data/noise_rps_dregon_michaels{,_swapped}.yaml`. **Caveat — not a byte-for-byte split reproduction**: these wrap the offline time-holdout `NoiseRPSDataset` (pools all `in_flight_noise` recordings together, does not select by room/recording id), not the historical *online* per-frame-geometry streaming slicer (`configs/noise_gen_online_dregon_michaels{,_swapped}.yaml` — all 8 mics jointly, DREGON room1-vs-room2 split explicitly). `..._swapped.yaml`'s `val_at_start=true` flips which *end* of each pooled recording's time axis is held out — an approximation of the historical room-swap, not the same split. |
| **Model** | `positional_harmonic_gen`, `cond_dim=16`, `drone_names=[dregon, michaels]` — `conf/model/positional_harmonic_gen_conditioned.yaml` (unconditioned `cond_dim=0` variant also exists: `conf/model/positional_harmonic_gen.yaml`, for a single-drone dataset). |
| **Historical command (E2, `.pi/checkpoints/noise-generation-online-dregon-michaels.md`)** | `train_noise_generation.py --online_config configs/noise_gen_online_dregon_michaels.yaml --cond_dim 16 --device cuda:0 --epochs 200 --patience 20 --batch_size 32 --duration_s 1.0 --n_harmonics 100 --samples_per_epoch 6000 --num_valid 256 --num_workers 8` |
| **New-framework command (E2)** | `python train.py experiment=e2_noise_gen_dregon_michaels` |
| **Historical command (E3, `.pi/checkpoints/noise-gen-swapped-smoothness-random-phase.md`)** | same + `--online_config configs/noise_gen_online_dregon_michaels_swapped.yaml --harm_smooth_weight 1e-2 --noise_smooth_weight 1e-2` (random-phase training is automatic, no flag). |
| **New-framework command (E3)** | `python train.py experiment=e3_noise_gen_swapped_smoothness` |
| **Results** | E2/E3 GPU runs were never completed start-to-finish in either the old or new framework (both handoff docs are "needs a real GPU run"/"CPU-smoke verified" only) — no numeric results to compare against. The `results/noise_gen_dregon_michaels_swapped/best_positional_harmonic_gen.pt` bundle E4 depends on is a *pre-existing* artifact from an earlier ad-hoc GPU run outside this framework, not reproduced here. |

**Replicability status: config-complete, blocked on (a) real training data
and (b) the offline-dataset/split deviations documented above** — treat
`e2_noise_gen_dregon_michaels`/`e3_noise_gen_swapped_smoothness` as
"the intended experiment now actually runs end-to-end" (proven via
`tests/training/test_validate.py::test_validate_config_passes_a_conditioned_noise_generation_pipeline`,
a synthetic-data smoke test through the full task/codec/model/loss/metric
pipeline), not as a verified numeric reproduction of either historical run.

## E4 — generated-noise-augmented RPS training

Trains an **RPS predictor** (`simple_conv_v2`) using a frozen, pretrained
noise generator (`PositionalHarmonicNoiseGen`, from E3) as a live
data-augmentation source — not a noise-gen training experiment itself.

| | |
|---|---|
| **Dataset** | Same real sources as `online_mix_v4_michaels` (DREGON minus room1, Michael's FLY125, LibriSpeech) + one `kind: generated` source (`configs/online_mix_generated_augment_example.yaml`). |
| **conf/data** | `conf/data/online_mix_generated_augment.yaml` (augmented arm), `conf/data/online_mix_v4_michaels_no_aug.yaml` (A/B baseline arm, `configs/online_mix_v4_michaels_train_no_room1_no_aug.yaml`). |
| **Historical command** | `train_rps_predictor.py --model simple_conv_v2 --data_root datasets/DREGON-LM-V4-michaels --online_mix --mix_config configs/online_mix_generated_augment_example.yaml --samples_per_validation 5000 --pit_loss --epochs 200 --patience 50 --batch_size 16 --num_workers 6` |
| **New-framework command** | `python train.py experiment=e4_generated_noise_augment` (augmented) / `python train.py experiment=e4_no_aug_baseline` (baseline arm) |
| **Results** | None yet — status as of the last checkpoint note (2026-07-02) is "code done + CPU-smoke verified (`tests/data_processing/test_generated_noise.py`, 42 pass); needs a real GPU run." |

**Replicability status: config-complete, blocked on (a) real training data,
and (b) a hard prerequisite artifact** — the noise-gen checkpoint at
`results/noise_gen_dregon_michaels_swapped/best_positional_harmonic_gen.pt`
(E3's swapped-split-only run) must exist and sync to wherever this is
re-run; its `n_harmonics=100` must match the policy YAML exactly (silent
shape mismatch otherwise). The generated stream is not seed-reproducible
(`refresh: true`) — never route it into validation (already enforced: both
`conf/data` files keep `valid:` on the fixed real split).

## F1 — synthetic RPS trajectories (generation-only script artifact)

Not a `train.py` experiment. `src/data_processing/rps_synthesis.py` models
quadrotor rotor speeds as a linear mixer over four control-mode OU/intermittent
processes, calibrated against real DREGON telemetry. It is a **direct
dependency of E4** (`GeneratedNoisePool` calls
`generate_intermittent_batch` for the synthetic RPS excitation/label).

- **Reproduce**: `cd writing/reports/2026-06-30_synthetic-rps-trajectories && make all` (runs `prepare.py` → figures/CSVs → typst compile), or directly:
  ```python
  from data_processing.rps_synthesis import generate_intermittent_batch
  batch = generate_intermittent_batch(64, duration=8.0, fs=100.0, drone_profile=0.5, aggressiveness=1.5, rng=0)
  ```
- **Results**: `writing/reports/2026-06-30_synthetic-rps-trajectories/report.typ`.
- **Status**: script-based, no model/loss to route — nothing to port beyond what E4 already needs.

---

## Needs follow-up wiring

Gaps found while encoding the catalogue above, deliberately **not** forced
into a `conf/experiment/*.yaml` (per the task ruling: document, don't fake a
config that can't actually train). (E1/E2/E3 noise-generation training,
C7/C8 salience-map baselines, and C9 8-channel training were previously
listed here — all three are now closed, see their own sections above.)

### C11 step 1 — zero-shot FLY124 cross-drone eval (no retrain)

This is an `eval.py`-only exercise (load the DREGON-only checkpoint, run it
against FLY124-derived clips) with no dedicated `conf/data/fly124_eval.yaml`.
The historical eval script (`results/fly124_simpleconvv2_eval/run_eval.py`)
is gitignored and not present in this worktree, so its exact input directory
convention could not be verified — creating a data config from an unverified
path was judged worse than leaving the gap explicit. Once the FLY124 eval
clip directory convention is confirmed, this is a small addition: a
`conf/data/fly124_eval.yaml` (valid-only, `DregonLMFrameDataset` over
whatever that directory is) + `python eval.py experiment=rps_simple_conv_v2_v4
data=fly124_eval checkpoint=<dregon-only ckpt>`.

### A1 — DN-LM sample-count discrepancy

README.md's historical `create_dataset.py` invocation uses
`--train_samples 6480 --valid_samples 720`; the current `replicate_paper.py`
driver defaults to `DATASET_TRAIN_SAMPLES=64800` (10× larger). Not resolved
here — pick one deliberately before a real run and note which in the run's
wandb tags/notes.
