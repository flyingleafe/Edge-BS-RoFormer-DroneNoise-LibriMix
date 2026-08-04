# tests/ — Framework Test Suite

Pytest suite for the training framework: datasets, losses, metrics, models,
tasks, the training loop, and utilities. (The old `postdoc` CLI tests were
removed when that subsystem was deleted.)

## Layout

| Path | Covers |
|---|---|
| `data_processing/` | DREGON loader, Frame datasets, online mixing, generated-noise pool, RPS synthesis |
| `losses/` | spectral, masked, PIT, regularizers, salience, composite losses |
| `metrics/` | separation, RPS, salience, perf metrics + `MetricSuite` |
| `models/` | RPS predictor, salience-RPS, positional harmonic noise generator |
| `tasks/` | task spec/contract, checkpoints, CLI, RPS prediction/regression, noise generation |
| `training/` | training loop, collate, validate, val-logging, R2 artifact upload (`conftest.py` + `_fixtures.py` here) |
| `plots/` | plot registry, RPS plots, `plots.dwym` dispatch + `plots.coerce` coercion (moved from `tests/utils/plots/`) |
| `utils/` | path helpers |
| `test_basic_pitch.py`, `test_online_mixing.py`, `test_plot_timeframe.py`, `test_dataloader_benchmark.py` | standalone module tests; `basic_pitch_ref.npz` is fixture data |

## Running

**Never run an unbounded `pytest`** — some tests build large tensors and a wide
`-k` selection has OOM-frozen small machines. Run a bounded subdirectory or a
single file, e.g.:

```bash
pytest tests/losses -q
pytest tests/data_processing/test_generated_noise.py -q
```

Run inside the nix devshell (`nix develop`) so torch and the editable install
resolve.

## If adding a feature

Add its test under the matching subdirectory (mirror `src/`'s layout). New
model → `tests/models/`; new loss → `tests/losses/`; new task behaviour →
`tests/tasks/`. Keep fixtures local to the subdirectory's `conftest.py` where
possible.
