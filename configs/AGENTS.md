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

## Creating a New Config

1. Copy the closest existing config as a starting point
2. Modify model, training, and dataset fields
3. Use the next available number prefix
4. Reference it from an experiment YAML in `experiments/`

## Gotchas

- HTDemucs uses `OmegaConf` instead of `ml_collections.ConfigDict` — different loading path in `utils.py`
- `use_rps` and `load_rps` are separate flags — both must be `true` for RPS experiments
- Config file paths with parentheses (like `3_FA_RoPE(64).yaml`) need escaping on the command line