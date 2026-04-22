# src/ — Source Packages

Top-level source package directory. Currently contains only `postdoc/`.

## Why this directory exists

Separates installable Python packages from root-level scripts. Packages here are registered as entry points in `pyproject.toml`.

## Contents

| Package | Purpose | Details |
|---------|---------|---------|
| `postdoc/` | Experiment orchestration CLI | See `postdoc/AGENTS.md` |