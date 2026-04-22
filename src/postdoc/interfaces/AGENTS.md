# src/postdoc/interfaces/ — Abstract Base Classes

Defines the contract for postdoc's pluggable backends. Stable interfaces that backends implement.

## Why this directory exists

Separates the orchestration logic from backend implementation. Enables future cloud/AWS backends without changing core logic.

## Interfaces

| Interface | File | Purpose |
|-----------|------|---------|
| `Scheduler` | `scheduler.py` | GPU allocation, subprocess spawning, job lifecycle |
| `StorageBackend` | `storage.py` | Artifact storage (configs, logs, metrics) |
| `JobTracker` | `tracker.py` | Job state persistence, metrics, SQLite-backed |

Each interface is an ABC with abstract methods that backends must implement. See `__init__.py` for imports.