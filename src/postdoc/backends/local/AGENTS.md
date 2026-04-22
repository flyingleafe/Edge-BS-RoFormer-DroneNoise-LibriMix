# src/postdoc/backends/local/ — Local Backend

Implements local GPU scheduling and disk-based storage for postdoc.

## Why this directory exists

Local execution backend — the only currently working backend. Manages GPU allocation, subprocess spawning, and artifact storage on the local machine.

## Files

| File | Purpose |
|------|---------|
| `scheduler.py` | `LocalScheduler` — tracks GPU availability, spawns training processes, handles job lifecycle |
| `storage.py` | `LocalStorage` — stores configs, logs, and metrics on disk under `results/` |
| `__init__.py` | Exports for backend package |

## Key Behavior

- GPU allocation via device IDs (0, 1, ... up to `postdoc.yaml:local.gpus - 1`)
- Job logs written to `results/<job_id>/`
- Queue mechanism when all GPUs are occupied