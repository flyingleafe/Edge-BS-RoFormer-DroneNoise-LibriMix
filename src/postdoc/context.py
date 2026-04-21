from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from postdoc.config import PostdocConfig, load_config
from postdoc.interfaces.storage import StorageBackend
from postdoc.interfaces.scheduler import Scheduler
from postdoc.interfaces.tracker import JobTracker


@dataclass
class PostdocContext:
    storage: StorageBackend
    scheduler: Scheduler
    tracker: JobTracker
    config: PostdocConfig


def _default_db_path(config: PostdocConfig) -> Path:
    return Path(config.local.results_dir) / ".postdoc.db"


def create_context(
    config_path: Path | None = None,
    backend: str | None = None,
    db_path: Path | None = None,
) -> PostdocContext:
    if config_path is None:
        config_path = Path("postdoc.yaml")

    config = load_config(config_path)
    if backend:
        config.backend = backend

    if db_path is None:
        db_path = _default_db_path(config)

    tracker = JobTracker(db_path)

    if config.backend == "local":
        from postdoc.backends.local.storage import LocalStorage
        from postdoc.backends.local.scheduler import LocalScheduler

        storage = LocalStorage(config.local.results_dir)
        scheduler = LocalScheduler(
            num_gpus=config.local.gpus,
            tracker=tracker,
            results_dir=Path(config.local.results_dir),
        )
    elif config.backend == "cloud":
        raise NotImplementedError(
            "Cloud backend is not yet implemented. Use backend='local'."
        )
    else:
        raise ValueError(f"Unknown backend: {config.backend}")

    return PostdocContext(
        storage=storage,
        scheduler=scheduler,
        tracker=tracker,
        config=config,
    )
