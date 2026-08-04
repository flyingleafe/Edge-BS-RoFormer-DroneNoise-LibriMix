# notebooks/ — Jupyter Notebooks

Interactive analysis notebooks for dataset inspection, result analysis, and model visualization.

## Why this directory exists

Jupyter notebooks give interactive exploration that scripts and CLI tools cannot. Used for one-off analyses, visualizations, and debugging sessions that produce figures or insights.

## Notebooks

| Notebook | Purpose |
|----------|---------|
| `rps_tracking.ipynb` | **The focused RPS-tracking notebook** (Phase 6 of `docs/refactor-2026-08-plan.md`): pull a recording span (`plots.explore.pick`), look at it (`plots.dwym`), run a zoo predictor (`zoo.load` → `FrameModel`), run the blind DSP ladder (`tracking.pipeline` of stages), overlay both against telemetry, read `meta["tracking"]`. Thin cells only — the logic lives in the library. Needs R2 credentials in `.env`. |
| `noise_generation.ipynb` | **The scripted noise-generator notebook**: list noise-gen checkpoints (`zoo.checkpoints(task="noise_generation")`), load one (`zoo.load`), synthesize from a synthetic RPS trajectory (`data_processing.rps_synthesis.generate_intermittent`) and from real telemetry, compare real vs generated with `plots.dwym` (audio players included). For live sliders (embedding walk, jitter, wind) use `generator_lab.ipynb` — the two coexist on purpose. |
| `speech_enhancement.ipynb` | **Planned, deliberately unwritten.** The user builds it by hand from `docs/notebook-primitives-tutorial.md` and reports every friction point (the Phase 6 ergonomics probe). Do NOT write this notebook for them. Until it exists, `generalization_explorer` and `se_baselines_explorer` cover SE exploration. |
| `generator_lab.ipynb` | **The one place to drive every noise generator.** Pick one or more variants (learned `deep/*`, JASA GP, CONA auralization, real), drive them from a real recording slice or a synthetic RPS trajectory, move the per-drone embedding / jitter linewidth / wind channel live, and compare spectrograms + a spectrum-under-slider + audio. Logic lives in `generator_lab.py` so it is reusable outside the notebook. Supersedes the deleted `drone_embedding_explorer`, `noise_gen_real_vs_generated`, `noise_four_way_comparison` and `jasa_gp_interactive` notebooks. |
| `generalization_explorer.ipynb` | SE generalization probe: DCUNet vs Edge-BS-RoFormer vs MP-SENet on SEEN → UNSEEN-recording → UNSEEN-drone noise. Logic lives in `generalization_lib.py`. |
| `se_baselines_explorer.ipynb` | Interactive per-clip exploration of the F1 SE baselines on the SE valid sets (metrics, spectrograms, audio). Logic lives in `se_baselines_explorer.py`. |
| `geom_calibration.ipynb` | Mic-array geometry calibration for the DREGON and Michael's arrays (the 180° mic-frame fix, the horizontal-ring fix). Logic lives in `geom_calibration.py` — `tests/test_geom_calibration.py` imports that module, so keep it importable. |
| `stage0_rotor_rtf.ipynb` | Stage-0 free-field rotor-to-mic RTF validation. Helpers live in `stage0_rtf_utils.py`. |
| `michael_data_analysis.ipynb` | Exploration of Michael's drone recordings (FLY124/FLY125 audio and telemetry). |
| `salience_baselines_dregon_v4.ipynb` | `multif0_salience` + `basic_pitch` final validation on DREGON-LM-V4 + per-sample spectrogram / **salience map** / RPS-vs-GT viz (uses the `"salience"` renderer + `plots.rps_prediction.salience_comparison`) |
| `visualize_models.ipynb` | Model architecture visualization (legacy models rebuilt from the inline `conf/model/*.yaml` configs via `models.registry.build_legacy_inline`) |

## Helper modules without a notebook

- `four_way_lib.py` — GP loading/rendering + CONA fetch helpers. Its parent notebook (`noise_four_way_comparison.ipynb`) is deleted, but `generator_lab.py` imports `load_gp`, `render_gp`, and the CONA helpers from it, so the module stays.

## Gotchas

- **2026-08 cleanup**: the stale/superseded notebooks were deleted; git history keeps them recoverable, and focused replacement notebooks arrive in a later refactor phase.
- Notebooks may reference `results/` data that needs syncing first (`omnirun pull <job>`)
- `.ipynb_checkpoints/` is gitignored
- For publication figures, prefer `eval.py` + `src/plots` comparison plots (absorbs the former `generate_comparison.py`/`plot_per_snr.py`) — see `generate-model-comparisons` skill
