# src/ — Source Packages

Top-level source package directory.

## Why this directory exists

Separates installable Python packages from root-level scripts. Packages here are registered as entry points in `pyproject.toml`.

## Contents

| Package | Purpose | Details |
|---------|---------|---------|
| `models/` | Model implementations + spectral front-ends | See `models/AGENTS.md` |
| `tasks/` | Task interface definitions | See `tasks/AGENTS.md` |
| `utils/` | `utils` package (legacy helpers + `utils.data` time-series algebra) | See `utils/AGENTS.md` |
| `localization/` | Rotor position estimation from mic-array audio (near-field SRP-PHAT) | See `localization/AGENTS.md` |
| `postdoc/` | Experiment orchestration CLI | See `postdoc/AGENTS.md` |