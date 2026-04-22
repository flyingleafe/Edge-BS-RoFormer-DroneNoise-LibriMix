# src/postdoc/backends/ — Backend Implementations

Concrete implementations of postdoc's storage, scheduling, and tracking interfaces.

## Why this directory exists

Pluggable backends for experiment orchestration. Currently only local backend is implemented; cloud is a stub for future work.

## local/ — Local Backend

| File | Purpose |
|------|---------|
| `scheduler.py` | `LocalScheduler` — manages GPU allocation, spawns training subprocesses, handles job lifecycle |
| `storage.py` | `LocalStorage` — disk-based artifact storage (configs, logs, metrics under `results/`) |

## cloud/ — Cloud Backend (Stub)

Returns `NotImplementedError` for all methods. Placeholder for future cloud (AWS/GCP) integration.

## Gotchas

- `LocalScheduler` allocates GPUs by device ID — configured via `postdoc.yaml:local.gpus`
- Storage paths are relative to `postdoc.yaml:local.results_dir`
- Cloud backend will need `cloud/` subdirectory with matching interface implementations