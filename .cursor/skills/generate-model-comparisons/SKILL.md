---
name: generate-model-comparisons
description: Generates model comparison plots (SI-SDR, STOI, PESQ), summary tables, and audio comparison plots (waveforms/spectrograms) from evaluation results for presentation preparation. Use when the user needs comparison visualizations, wants to compare specific model subsets, selects samples for audio analysis, or prepares presentation figures.
---

# Generate Model Comparisons

Generates publication-ready comparison plots and summary tables from evaluation results, enabling flexible model subset selection for different presentation contexts.

## Quick Start

Before generating comparisons, ensure evaluation results are synced:

```bash
./sync_results.sh
```

Then generate comparisons:

```bash
# Compare specific models (metrics only)
python generate_comparison.py --models Edge-BS-RoFormer DCUNet --output_dir presentations/fig1

# Compare all models
python generate_comparison.py --models all --output_dir results/comparison

# Generate audio comparison plots for specific samples
python generate_comparison.py --models Edge-BS-RoFormer DCUNet --samples 00000 00001 --output_dir presentations/audio_comparison
```

## Usage Pattern

When the user requests comparison plots or tables:

1. **Sync results first** (if not already done):
   ```bash
   ./sync_results.sh
   ```

2. **Generate comparison** with selected models and output location:
   ```bash
   python generate_comparison.py --models <model1> <model2> ... --output_dir <output_path>
   ```

3. **For multiple comparisons** (e.g., different model subsets for different slides):
   - Generate each comparison with a unique output directory
   - Use descriptive `--plot_name` and `--table_name` if needed

## Available Models

- `Edge-BS-RoFormer` - Proposed method
- `DCUNet` - Baseline
- `DPTNet` - Baseline
- `HTDemucs` - Baseline
- `Diffusion-Buffer-BBED` - Diffusion-based method

## Output Files

Each run generates:
- **Comparison plot**: 3-panel figure (SI-SDR, STOI, PESQ) as PNG (300 DPI)
- **Summary table**: CSV with mean/std/min/max statistics per model
- **Audio comparison plots** (if `--samples` specified): For each sample, generates a 2-row plot:
  - Top row: Waveform comparison across all models
  - Bottom row: Spectrogram comparison across all models
  - Saved as `sample_XXXXX_comparison.png` (300 DPI)

## Common Scenarios

### Scenario 1: Single Comparison Set
User: "Create a comparison plot with Edge-BS-RoFormer, DCUNet, and HTDemucs"

```bash
python generate_comparison.py \
    --models Edge-BS-RoFormer DCUNet HTDemucs \
    --output_dir presentations/comparison_baselines
```

### Scenario 2: Multiple Comparison Sets
User: "Make one plot comparing Edge-BS-RoFormer vs baselines, and another comparing all models"

```bash
# First comparison: proposed vs baselines
python generate_comparison.py \
    --models Edge-BS-RoFormer DCUNet DPTNet HTDemucs \
    --output_dir presentations/proposed_vs_baselines \
    --plot_name proposed_vs_baselines.png

# Second comparison: all models
python generate_comparison.py \
    --models all \
    --output_dir presentations/all_models \
    --plot_name all_models_comparison.png
```

### Scenario 3: Custom Output Location
User: "Generate comparison for my presentation in the slides folder"

```bash
python generate_comparison.py \
    --models Edge-BS-RoFormer DCUNet \
    --output_dir slides/figures \
    --plot_name model_comparison.png
```

### Scenario 4: Audio Comparison for Selected Samples
User: "Create waveform and spectrogram comparison plots for samples 00000, 00001, and 00002"

```bash
python generate_comparison.py \
    --models Edge-BS-RoFormer DCUNet HTDemucs \
    --samples 00000 00001 00002 \
    --output_dir presentations/audio_comparison
```

This generates:
- `sample_00000_comparison.png` - Waveform and spectrogram comparison for sample 00000
- `sample_00001_comparison.png` - Waveform and spectrogram comparison for sample 00001
- `sample_00002_comparison.png` - Waveform and spectrogram comparison for sample 00002
- Plus the standard metric comparison plot and summary table

### Scenario 5: Combined Metrics and Audio Comparison
User: "Generate both metric plots and audio comparisons for these samples"

```bash
python generate_comparison.py \
    --models Edge-BS-RoFormer DCUNet \
    --samples 00000 00001 \
    --output_dir presentations/full_comparison
```

This generates both metric plots/tables AND audio comparison plots in the same output directory.

## Script Parameters

| Parameter | Required | Description |
|-----------|----------|-------------|
| `--models` | Yes | Model names (space-separated) or "all" |
| `--output_dir` | Yes | Output directory for plots and tables |
| `--eval_dir` | No | Custom evaluation directory (default: `results/evaluation/evaluation`) |
| `--plot_name` | No | Plot filename (default: `model_comparison.png`) |
| `--table_name` | No | Table filename (default: `performance_summary.csv`) |
| `--samples` | No | Sample IDs for audio comparison (e.g., `00000 00001`). If not provided, only metric plots are generated |
| `--audio_dir` | No | Audio files directory (default: `results/evaluation`) |

## Notes

- **Always sync results first**: Run `./sync_results.sh` before generating comparisons
- **Model names are case-sensitive**: Use exact names as listed above
- **Output directory is created automatically**: Parent directories are created if needed
- **Plots are publication-ready**: 300 DPI PNG format with proper styling
- **Tables include statistics**: Mean, std, min, max for SI-SDR, PESQ, STOI
- **Sample IDs format**: Use 5-digit zero-padded format (e.g., `00000`, `00001`, `01234`)
- **Audio files location**: Audio files are expected in `results/evaluation/<model_dir>/sample_XXXXX_vocals.wav`
- **Audio comparison plots**: Each sample generates a separate comparison plot with waveforms (top) and spectrograms (bottom) for all selected models

## Error Handling

If evaluation directory is missing:
```
Error: Evaluation directory not found: results/evaluation/evaluation
Hint: Run './sync_results.sh' first to sync results from vast-server
```

If invalid model names are provided:
```
Error: Invalid model names: ['InvalidModel']
Available models: ['Edge-BS-RoFormer', 'DCUNet', 'DPTNet', 'HTDemucs', 'Diffusion-Buffer-BBED']
```
