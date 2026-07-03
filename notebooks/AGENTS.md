# notebooks/ — Jupyter Notebooks

Interactive analysis notebooks for dataset inspection, result analysis, and model visualization.

## Why this directory exists

Jupyter notebooks provide interactive exploration that scripts and CLI tools can't. Used for one-off analyses, visualizations, and debugging sessions that produce figures or insights.

## Notebooks

| Notebook | Purpose |
|----------|---------|
| `analyze_results.ipynb` | Paper 1 result analysis and comparison |
| `rps_experiment_results.ipynb` | Paper 2 RPS experiment analysis |
| `inspect_dregon_librimix.ipynb` | DREGON-LM dataset inspection |
| `explore_data.ipynb` | General data exploration |
| `visualize_models.ipynb` | Model architecture visualization |
| `inference_comparison.ipynb` | Side-by-side inference comparisons |
| `rps_evaluation_interactive.ipynb` | Interactive RPS evaluation |
| `salience_baselines_dregon_v4.ipynb` | `multif0_salience` + `basic_pitch` final validation on DREGON-LM-V4 + per-sample spectrogram / **salience map** / RPS-vs-GT viz (uses the `"salience"` renderer + `plots.rps_prediction.salience_comparison`) |

## Gotchas

- Notebooks may reference `results/` data that needs syncing first (`./sync_results.sh`)
- `.ipynb_checkpoints/` is gitignored
- For publication figures, prefer `generate_comparison.py` and `plot_per_snr.py` (root scripts) — see `generate-model-comparisons` skill