# src/ — Source Packages

Top-level source package directory.

## Why this directory exists

Separates installable Python packages from root-level scripts. Packages here are registered as entry points in `pyproject.toml`.

## Contents

| Package | Purpose | Details |
|---------|---------|---------|
| `models/` | Model implementations + spectral front-ends | See `models/AGENTS.md` |
| `tasks/` | Task interface definitions | See `tasks/AGENTS.md` |
| `utils/` | `utils` package — legacy ZFTurbo helpers only (`src/utils/data` time-series algebra was deleted; replaced by the PyPI `tdseries` package) | See `utils/AGENTS.md` |
| `data_processing/` | Dataset creation + RPS processing; `tdseries`/Frame-producing adapters and torch Datasets | See `data_processing/AGENTS.md` |
| `plots/` | All plotting (moved from `src/utils/plots`) — comparison plots, spectrogram/salience/RPS renderers | See `plots/AGENTS.md` |
| `losses/` | Consolidated loss implementations (spectral, PIT, masked, regularizers, salience, composite) | See `losses/AGENTS.md` |
| `metrics/` | Consolidated metric implementations (separation, RPS, perf) + `MetricSuite` | See `metrics/AGENTS.md` |
| `training/` | Training loop, checkpointing, config, R2 artifact upload, LoRA config seam | See `training/AGENTS.md` |
| `localization/` | Rotor position estimation from mic-array audio (near-field SRP-PHAT) | See `localization/AGENTS.md` |