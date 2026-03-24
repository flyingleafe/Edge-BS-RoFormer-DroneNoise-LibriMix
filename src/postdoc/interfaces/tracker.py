from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path


class JobState(Enum):
    DEFINED = "defined"
    SUBMITTED = "submitted"
    QUEUED = "queued"
    TRAINING = "training"
    EVAL = "eval"
    COMPLETING = "completing"
    DONE = "done"
    FAILED = "failed"


@dataclass
class JobRecord:
    job_id: str
    experiment_name: str
    state: JobState
    git_branch: str
    git_commit: str
    process_handle: str | None = None
    gpu_ids: list[int] | None = None
    submitted_at: datetime | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    failed_at: datetime | None = None
    error_category: str | None = None
    error_message: str | None = None
    metrics: dict | None = None
    metrics_incomplete: bool = False
    wandb_run_id: str | None = None
    config_snapshot: dict = field(default_factory=dict)


_SCHEMA = """\
CREATE TABLE IF NOT EXISTS jobs (
    job_id TEXT PRIMARY KEY,
    experiment_name TEXT NOT NULL,
    state TEXT NOT NULL,
    git_branch TEXT NOT NULL,
    git_commit TEXT NOT NULL,
    process_handle TEXT,
    gpu_ids TEXT,
    submitted_at TEXT,
    started_at TEXT,
    completed_at TEXT,
    failed_at TEXT,
    error_category TEXT,
    error_message TEXT,
    metrics TEXT,
    metrics_incomplete INTEGER DEFAULT 0,
    wandb_run_id TEXT,
    config_snapshot TEXT NOT NULL
);
"""


class JobTracker:
    def __init__(self, db_path: Path):
        self._db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute(_SCHEMA)
        self._conn.commit()

    def close(self):
        self._conn.close()

    def create_job(self, experiment_name: str, config: dict, git_branch: str, git_commit: str) -> str:
        job_id = uuid.uuid4().hex[:12]
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            "INSERT INTO jobs (job_id, experiment_name, state, git_branch, git_commit, "
            "submitted_at, config_snapshot) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (job_id, experiment_name, JobState.DEFINED.value, git_branch, git_commit,
             now, json.dumps(config)),
        )
        self._conn.commit()
        return job_id

    def update_state(self, job_id: str, state: JobState, **fields) -> None:
        sets = ["state = ?"]
        vals: list = [state.value]

        timestamp_map = {
            JobState.TRAINING: "started_at",
            JobState.DONE: "completed_at",
            JobState.FAILED: "failed_at",
        }
        ts_col = timestamp_map.get(state)
        if ts_col:
            sets.append(f"{ts_col} = ?")
            vals.append(datetime.now(timezone.utc).isoformat())

        # When resuming from FAILED, clear error fields
        if state in (JobState.SUBMITTED, JobState.QUEUED, JobState.TRAINING):
            sets.extend(["error_category = ?", "error_message = ?", "failed_at = ?"])
            vals.extend([None, None, None])

        for k, v in fields.items():
            if k == "gpu_ids":
                sets.append("gpu_ids = ?")
                vals.append(json.dumps(v))
            elif k in ("error_category", "error_message", "process_handle", "wandb_run_id"):
                sets.append(f"{k} = ?")
                vals.append(v)
            else:
                raise ValueError(f"Unknown field: {k}")

        vals.append(job_id)
        self._conn.execute(f"UPDATE jobs SET {', '.join(sets)} WHERE job_id = ?", vals)
        self._conn.commit()

    def get_job(self, job_id: str) -> JobRecord:
        row = self._conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        if row is None:
            raise KeyError(f"Job not found: {job_id}")
        return self._row_to_record(row)

    def list_jobs(self, state: JobState | None = None, limit: int = 50) -> list[JobRecord]:
        if state:
            rows = self._conn.execute(
                "SELECT * FROM jobs WHERE state = ? ORDER BY submitted_at DESC LIMIT ?",
                (state.value, limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM jobs ORDER BY submitted_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def get_queued_jobs(self) -> list[JobRecord]:
        rows = self._conn.execute(
            "SELECT * FROM jobs WHERE state = ? ORDER BY submitted_at ASC",
            (JobState.QUEUED.value,),
        ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def set_metrics(self, job_id: str, metrics: dict, incomplete: bool = False) -> None:
        self._conn.execute(
            "UPDATE jobs SET metrics = ?, metrics_incomplete = ? WHERE job_id = ?",
            (json.dumps(metrics), int(incomplete), job_id),
        )
        self._conn.commit()

    def get_running_jobs_on_branch(self, branch: str) -> list[JobRecord]:
        active_states = [s.value for s in (JobState.SUBMITTED, JobState.QUEUED,
                                            JobState.TRAINING, JobState.EVAL)]
        placeholders = ",".join("?" * len(active_states))
        rows = self._conn.execute(
            f"SELECT * FROM jobs WHERE git_branch = ? AND state IN ({placeholders})",
            (branch, *active_states),
        ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def _row_to_record(self, row: sqlite3.Row) -> JobRecord:
        return JobRecord(
            job_id=row["job_id"],
            experiment_name=row["experiment_name"],
            state=JobState(row["state"]),
            git_branch=row["git_branch"],
            git_commit=row["git_commit"],
            process_handle=row["process_handle"],
            gpu_ids=json.loads(row["gpu_ids"]) if row["gpu_ids"] else None,
            submitted_at=datetime.fromisoformat(row["submitted_at"]) if row["submitted_at"] else None,
            started_at=datetime.fromisoformat(row["started_at"]) if row["started_at"] else None,
            completed_at=datetime.fromisoformat(row["completed_at"]) if row["completed_at"] else None,
            failed_at=datetime.fromisoformat(row["failed_at"]) if row["failed_at"] else None,
            error_category=row["error_category"],
            error_message=row["error_message"],
            metrics=json.loads(row["metrics"]) if row["metrics"] else None,
            metrics_incomplete=bool(row["metrics_incomplete"]),
            wandb_run_id=row["wandb_run_id"],
            config_snapshot=json.loads(row["config_snapshot"]),
        )
