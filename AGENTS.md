# AGENTS.md - Edge-BS-RoFormer-DroneNoise-LibriMix

This document provides guidance for AI agents working with this repository.

## Repository Overview

This repository contains the official implementation and the DroneNoise-LibriMix (DN-LM) dataset for the paper "Edge-BS-RoFormer: Edge-Deployed Band-Split RoPE Transformer for Ultra-Low SNR UAV Speech Enhancement".

### Key Components

- **Models**: Edge-BS-RoFormer (proposed), DCUNet, DPTNet, HTDemucs, Diffusion Buffer (BBED)
- **Dataset**: DN-LM (DroneNoise-LibriMix) - synthesized from LibriSpeech and DroneAudioDataset
- **Training**: PyTorch-based training pipeline with configurable models
- **Evaluation**: Metrics including SI-SDR, SDR, PESQ, STOI

## Important Workflows

### Result Analysis Workflow

**⚠️ CRITICAL: Before analyzing results, always sync evaluation results from the remote server.**

Evaluation results are stored on `vast-server` and must be synced locally before analysis:

```bash
./sync_results.sh
```

This script:
- Syncs results from `vast-server:Edge-BS-RoFormer-DroneNoise-LibriMix/results/evaluation` 
- To local `results/evaluation`
- Uses rsync with progress display
- Verifies SSH connectivity before syncing

**Always run this script before:**
- Opening `analyze_results.ipynb`
- Running any result analysis scripts
- Generating plots or reports from evaluation data
- Using `generate_comparison.py` for presentation figures

### Training Workflow

Use `replicate_paper.sh` for complete replication:

```bash
./replicate_paper.sh all        # Full replication
./replicate_paper.sh train      # Train all models
./replicate_paper.sh eval       # Evaluate models
```

Individual model training:

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

### Evaluation Workflow

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

## Project Structure

```
Edge-BS-RoFormer-DroneNoise-LibriMix/
├── configs/                    # Model configuration files (YAML)
├── models/                     # Model implementations
│   ├── edge_bs_rof/           # Edge-BS-RoFormer (proposed)
│   ├── dcunet.py              # DCUNet baseline
│   ├── dptnet/                # DPTNet baseline
│   ├── demucs4ht.py           # HTDemucs baseline
│   └── diffusion_buffer.py    # Diffusion Buffer (BBED)
├── datasets/                   # DN-LM dataset (created locally)
├── results/                    # Training and evaluation results
│   └── evaluation/            # Final evaluation results (synced from vast-server)
├── train.py                   # Training script
├── valid.py                   # Validation during training
├── final_valid.py             # Final evaluation script
├── analyze_results.ipynb      # Result analysis notebook
├── generate_comparison.py     # Generate comparison plots and tables (agentic)
├── sync_results.sh            # Sync results from vast-server
└── replicate_paper.sh         # One-click replication script
```

## Configuration Files

Model configurations are in `configs/`:
- `3_FA_RoPE(64).yaml` - Edge-BS-RoFormer (proposed method)
- `9_Diffusion_Buffer_BBED.yaml` - Diffusion Buffer (BBED)
- `5_Baseline_dcunet.yaml` - DCUNet baseline
- `7_Baseline_dptnet.yaml` - DPTNet baseline
- `8_Baseline_htdemucs.yaml` - HTDemucs baseline

## Common Tasks

### Adding a New Model

1. Implement model in `models/`
2. Create config file in `configs/`
3. Add training command to `replicate_paper.sh`
4. Update evaluation scripts if needed

### Analyzing Results

1. **First**: Run `./sync_results.sh` to get latest results
2. Open `analyze_results.ipynb`
3. Ensure notebook points to `results/evaluation/`

### Generating Comparison Plots and Tables

For presentation preparation, use `generate_comparison.py` to create comparison plots and tables:

```bash
# Compare specific models
python generate_comparison.py --models Edge-BS-RoFormer DCUNet --output_dir presentations/fig1

# Compare all models
python generate_comparison.py --models all --output_dir results/comparison
```

**Agent Skill**: See `.cursor/skills/generate-model-comparisons/SKILL.md` for detailed usage patterns and examples. This skill enables flexible model subset selection for different presentation contexts (e.g., "make one plot with this set of models compared and another with this one").

### Generating Slidev Presentations

For creating complete presentations, use the Slidev presentation generation skill:

```bash
# The agent will guide you through creating slides based on your description
# It automatically uses generate_comparison.py for results slides
# And creates mermaid diagrams for approach/explanation slides
```

**Agent Skill**: See `.cursor/skills/generate-slidev-presentation/SKILL.md` for detailed instructions. This skill:
- Generates Slidev presentations from user slide descriptions
- Creates approach/explanation slides with mermaid diagrams
- Generates results slides using `generate_comparison.py`
- Handles audio sample slides with spectrograms

### Debugging Training

- Check `docs/debug-training-loop.md` for training loop details
- Use `valid.py` for validation during training
- Check logs in `results/<model_name>/`

## Notes for AI Agents

- **Always sync results before analysis**: Use `sync_results.sh` first
- **Config files**: YAML format, model-specific parameters
- **Dataset**: Must be created locally using `create_dataset.py` or `replicate_paper.sh dataset`
- **Results location**: Local training results in `results/`, evaluation results synced from `vast-server`
- **SSH access**: Requires `vast-server` to be accessible via SSH
- **Python environment**: Uses Python 3.12+, requires CUDA-capable GPU for training

## References

- Paper: "Edge-Deployed Band-Split RoPE Transformer for Ultra-Low SNR UAV Speech Enhancement"
- Dataset sources:
  - LibriSpeech: https://www.openslr.org/12/
  - DroneAudioDataset: https://github.com/saraalemadi/DroneAudioDataset
