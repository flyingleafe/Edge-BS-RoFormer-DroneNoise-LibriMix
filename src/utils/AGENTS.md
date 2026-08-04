# src/utils/ — `utils` package

This directory holds the `utils` Python package. It is editable-installed via
`pyproject.toml` (`[tool.hatch.build.targets.wheel] packages = [..., "src/utils"]`).

`__init__.py` contains the legacy "ZFTurbo"-style helpers (config loader,
audio I/O, `demix`/TTA inference, LoRA and weight-loading utilities) that used
to live in `utils.py` at the project root. The model factory
(`get_model_from_config`/`build_model_from_config`) moved to
`src/models/registry.py::LEGACY_MODEL_BUILDERS`, so `utils` is now a **leaf
package** — it imports no other project package.

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
