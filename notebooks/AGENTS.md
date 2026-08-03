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
| `generator_lab.ipynb` | **The one place to drive every noise generator.** Pick one or more variants (learned `deep/*`, JASA GP, CONA auralization, real), drive them from a real recording slice or a synthetic RPS trajectory, move the per-drone embedding / jitter linewidth / wind channel live, and compare spectrograms + a spectrum-under-slider + audio. Logic lives in `generator_lab.py` so it is reusable outside the notebook. Supersedes `drone_embedding_explorer`, `noise_gen_real_vs_generated`, `noise_four_way_comparison` and `jasa_gp_interactive`. |
| `salience_baselines_dregon_v4.ipynb` | `multif0_salience` + `basic_pitch` final validation on DREGON-LM-V4 + per-sample spectrogram / **salience map** / RPS-vs-GT viz (uses the `"salience"` renderer + `plots.rps_prediction.salience_comparison`) |

## Gotchas

- **Several older generator notebooks no longer import.** `drone_embedding_explorer`,
  `noise_gen_real_vs_generated` and friends still reach for `data_processing.dregon` /
  `data_processing.michaels`, which the data-layer refactor moved into
  `data_processing.sources`. Use `generator_lab.ipynb` instead; it is the consolidation.
- Notebooks may reference `results/` data that needs syncing first (`omnirun pull <job>`; legacy rsync fallback `./scripts/sync_results.sh`)
- `.ipynb_checkpoints/` is gitignored
- For publication figures, prefer `eval.py` + `src/plots` comparison plots (absorbs the former `generate_comparison.py`/`plot_per_snr.py`) — see `generate-model-comparisons` skill