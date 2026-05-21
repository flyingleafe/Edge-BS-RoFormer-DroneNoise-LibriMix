# src/utils/ — `utils` package

This directory holds the `utils` Python package. It is editable-installed via
`pyproject.toml` (`[tool.hatch.build.targets.wheel] packages = [..., "src/utils"]`).

`__init__.py` contains the legacy "ZFTurbo"-style helpers (model factory,
config loader, training utilities) that used to live in `utils.py` at the
project root. All existing callers — `train.py`, `valid.py`, `final_valid.py`,
`src/postdoc/infer.py`, notebooks — keep working unchanged: `from utils import
load_config, get_model_from_config, ...`.

## Why a package, not a module

We needed a sub-package (`utils.data`) for a new abstraction, which is not
possible if `utils` is a flat module. Converting `utils.py` → `utils/__init__.py`
preserved every legacy import while opening the namespace for sub-packages.

## Subdirectories

| Subdir | Purpose | Details |
|--------|---------|---------|
| `data/` | Aligned time-series containers (audio + RPS + segments + ...) | See `data/AGENTS.md` |

## Adding new sub-packages

Create a new subdirectory with its own `__init__.py` and `AGENTS.md`. Do not
add unrelated helpers to `__init__.py` — they belong in a sub-module so the
top-level namespace stays comprehensible.
