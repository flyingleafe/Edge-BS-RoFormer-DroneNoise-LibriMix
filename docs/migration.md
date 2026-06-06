# Migration: Legacy RPS scripts → Unified API

**Date:** 2026-06-05

Old scripts were archived to `legacy/`. Below is the old→new mapping.

## Evaluation scripts

| Old script (LOC) | New equivalent |
|---|---|
| `eval_rps_val.py` (174) | `evaluate-rps --input-set datasets/DREGON-LM/valid -m simple_conv@.../best.pt --alignment shape_stretch` |
| `evaluate_rps_predictor.py` (353) | `evaluate-rps --input-set datasets/DREGON-LM/valid -m A@... -m B@... --tex table.tex` |
| `evaluate_rps_predictor_samples.py` (302) | eval + `make-plot --type=rps_prediction.sample_comparison` |
| `evaluate_rps_long_samples.py` (207) | `[f for f in samples if f["audio"].duration > 8.0]` + `evaluate()` |
| `compute_rps_per_snr.py` (278) | `result.per_snr()` (built into `EvalResult`; reads `frame.tags["input_snr"]`) |
| `analyze_rps_high_snr.py` (353) | `load_dregon_highsnr()` (TODO) + `evaluate()` |
| `analyze_rps_full_sequence.py` (419) | `load_dregon_freeflight()` (TODO) + `evaluate()` + `PLOT_TYPES["rps_prediction.full_sequence"]()` |
| `extract_long_rps_samples.py` (101) | One-liner: `[f for f in samples if f["audio"].duration > 8.0]` |
| `extract_specific_rps_samples.py` (70) | One-liner: `[f for f in samples if f.tags.get("id") in wanted]` |
| `generate_rps_samples.py` (82) | In-memory `EvalResult` (no .npz hop needed) |

## Plotting scripts

| Old script (LOC) | New equivalent |
|---|---|
| `plot_rps_samples.py` (156) | `make-plot --type=rps_prediction.sample_comparison --sample <path>` |
| `plot_rps_comparison_long.py` (203) | `make-plot --type=rps_prediction.sample_comparison --sample <path> -m A@... -m B@...` |
| `plot_rps_comparison_with_spectrogram.py` (230) | `make-plot --type=rps_prediction.sample_comparison` (spectrogram is default) |
| `generate_rps_comparison_plots.py` (336) | `make-plot --type=rps_prediction.summary_metrics --results metrics.json` |
| `generate_rps_comparison_table.py` (149) | `evaluate-rps ... --tex table.tex` |
| `generate_rps_slides.py` (110) | `make-plot --type=rps_prediction.summary_metrics` |
| `plot_rps_training.py` (72) | `make-plot --type=rps_prediction.training_curves --log .../training_log.csv` |

**Total eliminated:** ~3,600 LOC across 17 scripts → 5 thin wrappers + 2 CLI commands

## Python API (for notebooks)

```python
from tasks.rps_prediction import load_predictor, load_input_set, evaluate
from utils.plots.rps_prediction import PLOT_TYPES

# Load
pred = load_predictor("simple_conv@results/rps_exp/best.pt")  # or "cepstral"
samples = list(load_input_set("datasets/DREGON-LM/valid"))

# Evaluate
result = evaluate(pred, samples, alignment="stft_timestamps")
print(result.aggregate)           # {mse, rmse, mae_frame, mae_clip, r2_mean, ...}
print(result.per_sample[0])       # per-sample metrics
print(result.per_snr())           # stratified by SNR bin
result.to_json("metrics.json")

# Plot
fig = PLOT_TYPES["rps_prediction.sample_comparison"](sample=samples[0])
fig = PLOT_TYPES["rps_prediction.summary_metrics"](results=[result])
fig.savefig("plot.pdf")
```

## CLI

```bash
# Evaluate
evaluate-rps --input-set datasets/DREGON-LM/valid \
  -m simple_conv@results/rps_exp/best.pt \
  -m dccrn_enc_rps@results/.../best.pt \
  --alignment stft_timestamps -o results/metrics.json

# Plot
make-plot --type=rps_prediction.sample_comparison \
  --sample datasets/DREGON-LM/valid/sample_00299

make-plot --type=rps_prediction.summary_metrics --results metrics.json
make-plot --type=rps_prediction.per_snr --results metrics.json
make-plot --type=rps_prediction.training_curves --log results/rps_exp/training_log.csv
make-plot --type=rps_prediction.full_sequence  (requires audio+rps arrays via API)
```

## New artifacts generated

- `results/rps_predictor_comparison/val_inference/metrics_new.json` — full DREGON-LM valid eval
- `results/rps_predictor_comparison/val_inference/per_snr_table_new.tex` — LaTeX per-SNR table
- `results/rps_predictor_comparison/val_inference/full_sequence_new.pdf` — full-sequence plot
- `results/rps_predictor_comparison/val_inference/sample_comparison_new.pdf` — sample comparison

Backed up to `/tmp/rps-refactor-backup/` for comparison by image-capable model.

## Architecture

```
src/tasks/                     # Task-separated evaluation
  checkpoints.py               # §0 — task-agnostic model loader
  rps_prediction.py            # §A — RPS eval, metrics, aggregation
  cli.py                       # evaluate-rps CLI

src/utils/plots/               # Task-separated plotting
  __init__.py                  # Shared plot registry + make-plot CLI
  rps_prediction/
    sample_comparison.py        # Spectrogram + GT + per-model rows
    summary_metrics.py          # Bar charts across models
    per_snr.py                  # Metric vs SNR-bin lines
    training_curves.py          # Train/val loss + R² curves
    full_sequence.py            # 3-panel full-take figure

src/utils/data/                # Extended with tags + resample/interpolate
tests/
  tasks/test_rps_regression.py # Golden-artifact regression
  utils/data/                  # 73 Hypothesis property tests
```

## Known limitations

1. `load_dregon_freeflight()` and `load_dregon_highsnr()` input-set loaders not yet implemented (marked TODO)
2. Full-sequence normalization needs matching to original `analyze_rps_full_sequence.py` logic
3. `make-plot` CLI has a typer arg-parsing edge case (works via Python API)
4. Pre-existing Hypothesis failure in `test_many_cuts_rejoin` (edge-case time frames)
