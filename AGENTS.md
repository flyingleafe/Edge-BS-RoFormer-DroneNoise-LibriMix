# AGENTS.md — Harmonic Noise Suppression

Guidance for AI agents working with this repository.

## Repository Overview

Research codebase for **speech enhancement under ultra-low SNR drone noise**, with two research directions:

1. **Paper 1 (published)**: Edge-BS-RoFormer — band-split RoPE transformer for UAV speech enhancement on the DN-LM dataset.
2. **Paper 2 (ongoing)**: RPS-conditioned speech enhancement — using Rotor Per-Second (motor speed) information to improve denoising, evaluated on the DREGON-LM dataset with DCUNet and DCCRN models.

### Key Components

- **Models**: Edge-BS-RoFormer (proposed), DCUNet, DCCRN, DPTNet, HTDemucs, Diffusion Buffer (BBED)
- **Datasets**: DN-LM (DroneNoise-LibriMix), DREGON-LM (DREGON-LibriMix with real motor telemetry)
- **RPS conditioning**: Rotor speed features fed to denoising models via bottleneck/GRU/hierarchical fusion
- **Auxiliary RPS prediction**: FPN-style head predicts rotor speeds from encoder features (multi-task learning)
- **Experiment orchestration**: `postdoc` CLI for job submission, GPU scheduling, and result tracking
- **Training**: PyTorch pipeline with W&B logging, configurable models and losses
- **Evaluation**: SI-SDR, SDR, PESQ, STOI, ESTOI metrics with per-SNR breakdown

## Environment Setup

- **Python**: 3.12+ (see `.python-version`)
- **Package manager**: `uv` (recommended), dependencies in `pyproject.toml` and `requirements.txt`
- **Nix**: `flake.nix` provides dev shell with Python 3.12, uv, C++ libs, graphviz
- **direnv**: `.envrc` activates nix shell; use `direnv exec .` prefix or activate the venv directly
- **GPU**: CUDA-capable GPU required for training; local scheduler supports multi-GPU

## Project Structure

```
harmonic-noise-suppression/
├── configs/                        # Model configuration files (YAML)
│   ├── 3_FA_RoPE(64).yaml         # Edge-BS-RoFormer (Paper 1)
│   ├── 5_Baseline_dcunet.yaml     # DCUNet baseline (Paper 1, DN-LM)
│   ├── 5a-c_DCUNet_RPS_*.yaml     # DCUNet + RPS variants (DN-LM)
│   ├── 6a-c_DCUNet_RPS_DREGON_*.yaml  # DCUNet + RPS variants (DREGON-LM)
│   ├── 7_Baseline_dptnet.yaml     # DPTNet baseline
│   ├── 7a_DCUNet_RPS_DREGON.yaml  # DCUNet + RPS bottleneck (DREGON-LM)
│   ├── 7b_DCUNet_baseline_DREGON.yaml # DCUNet baseline (DREGON-LM)
│   ├── 7c_DCUNet_RPS_PredRPS_DREGON.yaml # DCUNet + RPS + auxiliary RPS pred
│   ├── 8_Baseline_htdemucs.yaml   # HTDemucs baseline
│   ├── 9_Diffusion_Buffer_BBED.yaml # Diffusion Buffer
│   ├── 10a_DCCRN_baseline_DREGON.yaml # DCCRN baseline (DREGON-LM)
│   ├── 10b_DCCRN_RPS_DREGON.yaml  # DCCRN + RPS (DREGON-LM)
│   ├── 10c_DCCRNLite_RPS_DREGON.yaml # DCCRNLite + RPS
│   ├── 10d_DCCRN_RPS_PredRPS_DREGON.yaml # DCCRN + RPS + auxiliary RPS pred
│   └── test_cpu_dregon_*.yaml     # CPU test configs
├── models/                         # Model implementations
│   ├── edge_bs_rof/               # Edge-BS-RoFormer (band-split RoPE transformer)
│   ├── dcunet.py                  # DCUNet + RPS conditioning + RPSPredictionHead
│   ├── dccrn.py                   # DCCRN (complex conv recurrent) + RPS conditioning
│   ├── dptnet/                    # DPTNet baseline
│   ├── demucs4ht.py               # HTDemucs baseline
│   └── diffusion_buffer.py        # Diffusion Buffer (BBED)
├── src/postdoc/                    # Experiment orchestration package
│   ├── cli.py                     # Typer CLI: postdoc job/results commands
│   ├── config.py                  # PostdocConfig dataclass from postdoc.yaml
│   ├── experiment.py              # Experiment YAML loading and config resolution
│   ├── run_job.py                 # Job runner: train→eval subprocess orchestration
│   ├── context.py                 # Wires storage/scheduler/tracker from backend config
│   ├── interfaces/                # ABCs: Scheduler, StorageBackend, JobTracker
│   └── backends/
│       ├── local/                 # LocalScheduler (GPU alloc), LocalStorage (disk)
│       └── cloud/                 # Stub — NotImplementedError
├── experiments/                    # Experiment definitions for postdoc
│   ├── _example.yaml              # Reference experiment format
│   ├── train_*.yaml               # Training experiments
│   └── eval_*.yaml                # Eval-only experiments
├── data_processing/
│   └── dregon.py                  # DREGON dataset loading and RPS processing
├── notebooks/
│   ├── analyze_results.ipynb      # Result analysis (Paper 1)
│   ├── rps_experiment_results.ipynb # RPS experiment analysis
│   ├── inspect_dregon_librimix.ipynb # DREGON-LM dataset inspection
│   ├── explore_data.ipynb         # General data exploration
│   └── visualize_models.ipynb     # Model architecture visualization
├── docs/
│   ├── debug-training-loop.md     # Training loop debugging guide
│   ├── diffusion-buffer-paper.md  # Diffusion buffer paper notes
│   ├── diffusion-prompt.md        # Diffusion model implementation prompt
│   └── superpowers/               # Postdoc platform architecture specs
│       ├── specs/                 # Design documents
│       └── plans/                 # Implementation plans
├── slides/                        # Slidev presentations (gitignored)
├── .cursor/skills/                # Cursor agent skills
│   ├── generate-model-comparisons/
│   ├── generate-slidev-presentation/
│   └── examine-presentation-slides/
├── train.py                       # Training script (all model types)
├── valid.py                       # Validation during training
├── final_valid.py                 # Final evaluation with metrics + per-SNR breakdown
├── dataset.py                     # MSSDataset loader (DN-LM and DREGON-LM)
├── metrics.py                     # SDR, SI-SDR, and training metric implementations
├── utils.py                       # Model loading, audio I/O, inference (demix)
├── data_utils.py                  # Download/unpack helpers, audio file discovery
├── create_dataset.py              # DN-LM dataset creation (LibriSpeech + DroneAudioDataset)
├── create_dregon_librimix.py      # DREGON-LM dataset creation (DREGON + LibriSpeech)
├── train_rps_predictor.py         # Standalone RPS predictor training
├── generate_rps_samples.py        # Generate RPS prediction samples for visualization
├── generate_comparison.py         # Publication-ready comparison plots and tables
├── plot_per_snr.py                # Per-SNR metric comparison plots
├── plot_rps_samples.py            # RPS prediction visualization
├── plot_rps_training.py           # RPS predictor training curve plots
├── replicate_paper.py             # One-click Paper 1 replication (Python)
├── replicate_paper.sh             # One-click Paper 1 replication (shell)
├── rps_experiment.sh              # RPS experiment automation (dataset + training)
├── sync_results.sh                # Sync results from vast-server
├── postdoc.yaml                   # Postdoc platform configuration
├── pyproject.toml                 # Project metadata, dependencies, postdoc entrypoint
└── flake.nix                      # Nix dev shell definition
```

## Model Types

Model type keys used in `train.py --model_type` and config files (defined in `utils.py:get_model_from_config`):

| Key | Model | Notes |
|-----|-------|-------|
| `edge_bs_rof` | Edge-BS-RoFormer (BSRoformer) | Band-split RoPE transformer (Paper 1 proposed) |
| `mel_band_roformer` | MelBandRoformer | Mel-band variant of roformer |
| `dcunet` | DCUNet | Deep Complex U-Net, supports RPS conditioning |
| `dccrn` | DCCRN | Deep Complex Conv Recurrent Network, supports RPS |
| `dptnet` | DPTNet | Dual-path transformer |
| `htdemucs` | HTDemucs | Hybrid transformer Demucs |
| `diffusion_buffer` | DiffusionBufferModel | BBED diffusion-based enhancement |

## Datasets

### DN-LM (DroneNoise-LibriMix) — Paper 1

- **Sources**: LibriSpeech `train-clean-100` + DroneAudioDataset
- **Duration**: 2 hours total, 1-second samples at 16 kHz mono
- **SNR range**: -30 dB to 0 dB
- **Split**: 6480 train / 720 valid
- **Structure**: `datasets/DN-LM/{train,valid}/sample_XXXXX/{vocals.wav, noise.wav, mixture.wav}`
- **Creation**: `python create_dataset.py --speech_dir ... --noise_dir ... --output_dir datasets/DN-LM`

### DREGON-LM (DREGON-LibriMix) — Paper 2

- **Sources**: DREGON dataset (real UAV flight recordings with motor telemetry) + LibriSpeech
- **Key difference**: Contains real rotor speed data (`rps.npy`) per sample — 4 rotors at ~929 Hz
- **Sample duration**: 8.224 seconds (131584 samples at 16 kHz)
- **Structure**: `datasets/DREGON-LM/{train,valid}/sample_XXXXX/{vocals.wav, noise.wav, mixture.wav, rps.npy}`
- **Creation**: `python create_dregon_librimix.py` (downloads DREGON from HuggingFace via `datasets` library)

## Workflows

### Result Analysis

**⚠️ CRITICAL: Always sync evaluation results from the remote server before analysis.**

```bash
direnv exec . ./sync_results.sh
```

Syncs from `vast-server:harmonic-noise-suppression/results/evaluation` to local `results/evaluation`.

### Paper 1 Replication (Edge-BS-RoFormer)

```bash
direnv exec . ./replicate_paper.sh all    # Full: setup → download → dataset → train → eval
direnv exec . ./replicate_paper.sh train  # Train all Paper 1 models
direnv exec . ./replicate_paper.sh eval   # Evaluate all Paper 1 models
```

### Individual Model Training

```bash
python train.py \
    --model_type edge_bs_rof \
    --config_path configs/3_FA_RoPE\(64\).yaml \
    --results_path results/edge_bs_roformer \
    --data_path datasets/DN-LM/train \
    --valid_path datasets/DN-LM/valid \
    --dataset_type 1 \
    --device_ids 0
```

### Individual Model Evaluation

```bash
python final_valid.py \
    --model_type edge_bs_rof \
    --config_path configs/3_FA_RoPE\(64\).yaml \
    --start_check_point results/edge_bs_roformer/best_model.ckpt \
    --valid_path datasets/DN-LM/valid \
    --store_dir results/evaluation \
    --device_ids 0 \
    --metrics si_sdr sdr pesq stoi
```

### RPS Experiments (Paper 2)

```bash
direnv exec . ./rps_experiment.sh create-dataset  # Create DREGON-LM
direnv exec . ./rps_experiment.sh train            # Train DCUNet±RPS in parallel
direnv exec . ./rps_experiment.sh eval             # Evaluate both
```

Or via postdoc:

```bash
postdoc job submit experiments/train_dccrn_rps_dregon.yaml
postdoc job submit experiments/eval_dcunet_rps_dregon.yaml
```

## Postdoc Experiment System

`postdoc` is an experiment orchestration CLI installed via `pyproject.toml` (`postdoc = "postdoc.cli:app"`). Source in `src/postdoc/`.

### Architecture

```
postdoc CLI (Typer)
    │
    ├─ job submit   → JobTracker (SQLite) → Scheduler (LocalScheduler)
    │                                            │
    │                                     GPU allocation + subprocess
    │                                            │
    │                                       run_job.py
    │                                      ┌─────┴─────┐
    │                                   train.py   final_valid.py
    │
    ├─ job list/status/logs/resume/cancel
    └─ results show/compare
```

### CLI Commands

```bash
postdoc job submit <experiment.yaml...>    # Submit experiment(s)
postdoc job list [--state <state>]         # List jobs
postdoc job status <job_id>                # Job status + metrics
postdoc job logs <job_id> [--tail]         # View job logs
postdoc job resume <job_id> [--set k=v...] # Resume failed job from best checkpoint
postdoc job cancel <job_id>                # Cancel running job
postdoc results show <job_id>              # Show job results
postdoc results compare <job_id...>        # Compare multiple jobs
```

### Platform Config (`postdoc.yaml`)

```yaml
backend: local            # "local" or "cloud" (cloud is unimplemented)
local:
  gpus: 2                 # Number of GPUs for scheduling
  results_dir: results    # Root output directory
wandb:
  project: harmonic-noise-suppression
  entity: flyingleafe
```

### Experiment YAML Format (`experiments/*.yaml`)

**Training experiment:**
```yaml
model:
  type: dcunet                              # Model type key
  base_config: configs/5_Baseline_dcunet.yaml  # Base config to merge into
  overrides:                                # Dotted-key overrides
    training.lr: 3.0e-4
    training.batch_size: 4
dataset:
  name: DN-LM
eval:
  metrics: [estoi, si_sdr, pesq]
wandb:
  tags: [dcunet, example]
```

**Eval-only experiment:**
```yaml
eval_only: true
checkpoint: results/dcunet_rps_dregon/best_model.ckpt
model:
  type: dcunet
  base_config: configs/7a_DCUNet_RPS_DREGON.yaml
dataset:
  name: DREGON-LM
eval:
  metrics: [estoi, si_sdr, pesq]
wandb:
  tags: [eval, dcunet, rps, dregon]
```

### Job Lifecycle

```
DEFINED → SUBMITTED → TRAINING → EVAL → DONE
                ↓          ↓
              QUEUED    FAILED
```

- Jobs queue automatically when no GPU is available (`NoCapacityError`)
- `drain_queue()` picks up queued jobs when capacity frees
- Failed jobs can be resumed from best checkpoint with `postdoc job resume`
- Error classification: OOM, NaN, DataLoading, CUDA, Unknown

### Core Interfaces

| Interface | Location | Purpose |
|-----------|----------|---------|
| `JobTracker` | `interfaces/tracker.py` | SQLite-backed job state + metrics persistence |
| `Scheduler` | `interfaces/scheduler.py` | GPU allocation, subprocess spawn, lifecycle |
| `StorageBackend` | `interfaces/storage.py` | Artifact storage (configs, logs, metrics) |

## RPS Conditioning (Paper 2)

### What is RPS?

**Rotor Per-Second** — rotational speed of each drone motor (4 rotors). Drone noise harmonics depend directly on rotor speed; providing RPS as a conditioning signal helps models better separate speech from noise.

### Fusion Strategies

Configured via `dcunet_rps_fusion` or the DCCRN architecture:

| Strategy | Key | Description |
|----------|-----|-------------|
| Bottleneck | `bottleneck` | RPS → RotorEncoder → 64-dim → project to bottleneck dim → add at bottleneck |
| GRU | `gru` | RPS → RotorEncoder → 64-dim → concat with flattened features before GRU |
| Hierarchical | `hierarchical` | RPS features injected at multiple encoder levels |

### Auxiliary RPS Prediction Head

FPN-style multi-scale head (`RPSPredictionHead` in `dcunet.py`):
- Processes all encoder levels: pool frequency → concat real/imag → project to 64-dim
- Bottom-up merge with upsampling
- Final: upsample to STFT frames → 2-layer conv → 4 rotor speed predictions
- Multi-task loss: `total_loss = main_loss + lambda_rps * rps_mse_loss`
- Enabled via `predict_rps: true` in config

### RotorEncoder

Shared across DCUNet and DCCRN:
- 2-layer 1D convolution (4 → 32 → 64 channels)
- Resamples RPS from motor sampling rate to match STFT frame count

## Generating Comparison Plots and Tables

```bash
# Sync results first
./sync_results.sh

# Compare specific models (metrics plots + summary table)
python generate_comparison.py --models Edge-BS-RoFormer DCUNet --output_dir presentations/fig1

# Include audio waveform/spectrogram comparisons
python generate_comparison.py --models Edge-BS-RoFormer DCUNet --samples 00000 00001 --output_dir presentations/audio

# Per-SNR comparison
python plot_per_snr.py
```

**Available model names**: `Edge-BS-RoFormer`, `DCUNet`, `DPTNet`, `HTDemucs`, `Diffusion-Buffer-BBED`

## Agent Skills (Cursor)

Located in `.cursor/skills/`:

| Skill | Description |
|-------|-------------|
| `generate-model-comparisons` | Generate publication-ready plots/tables from eval results |
| `generate-slidev-presentation` | Create Slidev presentations with mermaid diagrams + result figures |
| `examine-presentation-slides` | Start Slidev and visually inspect slides via browser MCP |

## Tests

Tests are in `tests/` and target the postdoc system:

```bash
pytest tests/
```

Key test files: `test_cli.py`, `test_config.py`, `test_context.py`, `test_experiment.py`, `test_run_job.py`, `test_local_scheduler.py`, `test_local_storage.py`, `test_tracker.py`, `test_integration.py`.

## Common Tasks

### Adding a New Model

1. Implement in `models/` (see `dccrn.py` for a recent example with RPS support)
2. Register in `utils.py:get_model_from_config()` with a model type key
3. Create config YAML in `configs/`
4. Create experiment YAML in `experiments/` for postdoc submission
5. Or add training command to `replicate_paper.sh`

### Adding RPS Support to a Model

1. Add `use_rps` flag to config
2. Import `RotorEncoder` from `models.dcunet`
3. Encode RPS: `rps_features = rotor_encoder(rps, target_length=n_stft_frames)`
4. Choose fusion strategy (bottleneck add / GRU concat / hierarchical)
5. Optionally add `RPSPredictionHead` for auxiliary prediction
6. Dataset automatically loads `rps.npy` when `load_rps: true` in config

### Running a New Experiment via Postdoc

1. Create experiment YAML in `experiments/`
2. Submit: `postdoc job submit experiments/my_experiment.yaml`
3. Monitor: `postdoc job status <job_id>` or `postdoc job logs <job_id> --tail`
4. On failure: `postdoc job resume <job_id>` (resumes from best checkpoint)
5. Results: `postdoc results show <job_id>`

### Debugging Training

- Training loop details: `docs/debug-training-loop.md`
- W&B dashboard: project `harmonic-noise-suppression`, entity `flyingleafe`
- Logs in `results/<model_name>/` or via `postdoc job logs <job_id>`
- Postdoc classifies failures: OOM / NaN / DataLoading / CUDA / Unknown

## Notes for AI Agents

- **Always sync results before analysis**: Run `./sync_results.sh` first
- **Config files**: YAML format loaded via `ml_collections.ConfigDict` (or `OmegaConf` for htdemucs)
- **Model type keys**: Must match exactly in `utils.py:get_model_from_config()`
- **RPS data**: `rps.npy` shape is `(4, n_motor_samples)` — 4 rotors, resampled to STFT frames by `RotorEncoder`
- **Datasets**: Created locally, not committed (gitignored). DN-LM via `create_dataset.py`, DREGON-LM via `create_dregon_librimix.py`
- **Results**: gitignored. Training results local in `results/`, evaluation results synced from `vast-server`
- **Slides**: gitignored. Slidev presentations in `slides/<date>/`
- **W&B**: Credentials in `.env` (gitignored). Training auto-logs to W&B when key is set
- **Postdoc cloud backend**: Not implemented — local only for now
- **Design docs**: See `docs/superpowers/specs/` for postdoc platform architecture and job layer design

## References

- **Paper 1**: "Edge-Deployed Band-Split RoPE Transformer for Ultra-Low SNR UAV Speech Enhancement" (Liu et al., Drones 2025)
- **Paper 2 direction**: RPS-conditioned speech enhancement (inspired by Gulli et al., EURASIP 2025)
- **Training framework**: Based on [Music-Source-Separation-Training](https://github.com/ZFTurbo/Music-Source-Separation-Training) by ZFTurbo
- **Datasets**: [LibriSpeech](https://www.openslr.org/12/), [DroneAudioDataset](https://github.com/saraalemadi/DroneAudioDataset), [DREGON](https://huggingface.co/datasets/) (via HuggingFace `datasets`)
