# configs/ — Model Configuration Files

YAML configuration files that define model architectures, training hyperparameters, and dataset parameters. Loaded by `train.py` via `ml_collections.ConfigDict` (or `OmegaConf` for HTDemucs).

## Why this directory exists

Separates experiment configuration from code. Each config YAML is a self-contained experiment specification that can be version-controlled and referenced by experiment YAMLs.

## Naming Convention

Config files follow a numbered prefix pattern reflecting experiment history:

| Pattern | Meaning |
|---------|---------|
| `1_Nothing.yaml` | Baseline (no model) |
| `2_FA.yaml` | Full attention variant |
| `3_FA_RoPE(64).yaml` | Edge-BS-RoFormer (Paper 1) |
| `5_Baseline_dcunet.yaml` | DCUNet baseline (DN-LM) |
| `5a-c_DCUNet_RPS_*.yaml` | DCUNet + RPS variants (DN-LM) |
| `6a-c_DCUNet_RPS_DREGON_*.yaml` | DCUNet + RPS variants (DREGON-LM) |
| `7*_*.yaml` | DREGON experiments (DCUNet, DPTNet) |
| `8_Baseline_htdemucs.yaml` | HTDemucs baseline |
| `9_Diffusion_Buffer_BBED.yaml` | Diffusion Buffer |
| `10a-d_DCCRN_*.yaml` | DCCRN variants (DREGON-LM) |
| `11a-c_*.yaml` | RPS-only predictor experiments |
| `12a-c_DCUNetRefactored_*.yaml` | DCUNetRefactored variants |
| `13a-d_*.yaml` | Refactored models + auxiliary RPS prediction |
| `test_cpu_*.yaml` | CPU-only test configs |
| `online_mix_*.yaml` | Online-mixing RPS dataloader policies consumed by `train_rps_predictor.py --online_mix --mix_config ...` |

## Key Config Fields

| Field | Purpose | Example |
|-------|---------|---------|
| `model` | Model architecture params | `hidden_size`, `n_heads`, `rope` |
| `training` | Training params | `lr`, `batch_size`, `epochs` |
| `training.lr` | Learning rate | `3.0e-4` |
| `training.batch_size` | Batch size | `4` |
| `use_rps` | Enable RPS conditioning | `true` / `false` |
| `dcunet_rps_fusion` | RPS fusion strategy | `bottleneck` / `gru` / `hierarchical` |
| `predict_rps` | Enable auxiliary RPS prediction | `true` / `false` |
| `load_rps` | Load RPS data in dataset | `true` / `false` |

## Online-mixing configs

These are not model-architecture configs. They define the online RPS dataloader
policy for `data_processing/online_mixing.py` and are loaded with OmegaConf by
`train_rps_predictor.py` when `--online_mix --mix_config <path>` is passed.

Current durable configs:

| Config | Use |
|--------|-----|
| `online_mix_v4_michaels_train_no_room1.yaml` | V4-Michaels online training with DREGON `free-flight_nosource_room1` excluded, Michael's `FLY125` train-only, original LibriSpeech sources, SNR uniform `[-30, 0]`, and staged augmentations after 50k samples. |
| `online_mix_v4_michaels_train_no_room1_no_aug.yaml` | Same source/noise split and random mixtures, but no augmentations at any point. |
| `online_mix_generated_augment_example.yaml` | Adds a `kind: generated` noise source (a trained `PositionalHarmonicNoiseGen` rendered live on GPU) alongside real DREGON + Michael's noise. Template for augmenting with unlimited synthetic rotating noise + exact RPS labels. See `data_processing/AGENTS.md` § "Generated noise source". |

Important fields:
- `sources.noise`: aligned rotating-noise pools. DREGON and Michael's loaders
  return `TimeFrame`s; keep validation leakage guards here.
- `sources.speech`: unaligned clean speech/audio pool. Use original
  `data/librispeech/LibriSpeech/train-clean-100/**/*.flac`, not generated
  `datasets/.../train/**/vocals.wav`.
- `sources.speech[].cache.mode: packed_int16`: builds/reuses a PCM16 packed
  source cache behind the same loader interface.
- `sources.speech[].cache.dir`: defaults via
  `${oc.env:ONLINE_MIX_SOURCE_CACHE_DIR,.cache/online_mix_sources}`. Set
  `ONLINE_MIX_SOURCE_CACHE_DIR` in `.env` on machines where the cache belongs on
  another partition.
- `policy.stages`: optional sample-index curriculum. `until: 50000` means first
  50k global samples; `until: null` is the final stage.

Minimal launch pattern:

```bash
python train_rps_predictor.py \
  --model simple_conv_v2 \
  --device cuda:0 \
  --epochs 200 --patience 50 --batch_size 16 \
  --data_root datasets/DREGON-LM-V4-michaels \
  --online_mix \
  --mix_config configs/online_mix_v4_michaels_train_no_room1_no_aug.yaml \
  --samples_per_validation 5000 \
  --pit_loss
```

## Creating a New Config

1. Copy the closest existing config as a starting point
2. Modify model, training, and dataset fields
3. Use the next available number prefix
4. Reference it from an experiment YAML in `experiments/`

## Gotchas

- HTDemucs uses `OmegaConf` instead of `ml_collections.ConfigDict` — different loading path in `utils.py`
- `use_rps` and `load_rps` are separate flags — both must be `true` for RPS experiments
- Config file paths with parentheses (like `3_FA_RoPE(64).yaml`) need escaping on the command line