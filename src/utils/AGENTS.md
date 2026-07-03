# src/utils/ — `utils` package

This directory holds the `utils` Python package. It is editable-installed via
`pyproject.toml` (`[tool.hatch.build.targets.wheel] packages = [..., "src/utils"]`).

`__init__.py` contains the legacy "ZFTurbo"-style helpers (model factory,
config loader, training utilities) that used to live in `utils.py` at the
project root. Callers — `src/models/registry.py::build_legacy_model`,
`src/postdoc/infer.py`, notebooks — keep working unchanged: `from utils import
load_config, get_model_from_config, ...`.

## Why a package, not a module

This used to be a sub-package (`utils.data`, the in-repo timeseries algebra)
before it was replaced by the PyPI `tdseries` package and deleted (see
docs/refactor-unified-framework.md). Converting `utils.py` → `utils/__init__.py`
preserved every legacy import while opening the namespace for future sub-packages.

## Files

| Path | Purpose | Details |
|------|---------|---------|
| `dataloader_benchmark.py` | Reusable `benchmark_dataloader(...)` helper for finite and infinite PyTorch loaders; reports batch/example/effective-audio-clip throughput. | Use before optimizing online-mixing dataloaders. |

## Adding new sub-packages

Create a new subdirectory with its own `__init__.py` and `AGENTS.md`. Do not
add unrelated helpers to `__init__.py` — they belong in a sub-module so the
top-level namespace stays comprehensible.
