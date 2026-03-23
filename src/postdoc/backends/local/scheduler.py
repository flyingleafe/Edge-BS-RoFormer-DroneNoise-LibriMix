from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from postdoc.interfaces.scheduler import Scheduler, SchedulerSubmitResult, NoCapacityError

if TYPE_CHECKING:
    from postdoc.interfaces.tracker import JobTracker


class LocalScheduler(Scheduler):

    def __init__(self, num_gpus: int, tracker: "JobTracker", results_dir: Path):
        self._num_gpus = num_gpus
        self._tracker = tracker
        self._results_dir = results_dir
        self._gpu_alloc: dict[int, str] = {}
        self._rebuild_gpu_alloc()

    def _rebuild_gpu_alloc(self) -> None:
        from postdoc.interfaces.tracker import JobState
        for state in (JobState.SUBMITTED, JobState.TRAINING, JobState.EVAL):
            for job in self._tracker.list_jobs(state=state):
                if job.gpu_ids and job.process_handle and self.is_alive(job.process_handle):
                    for gpu_id in job.gpu_ids:
                        self._gpu_alloc[gpu_id] = job.job_id

    def _find_free_gpu(self) -> int | None:
        for gpu_id in range(self._num_gpus):
            if gpu_id not in self._gpu_alloc:
                return gpu_id
        return None

    def _allocate_gpu(self, job_id: str, gpu_id: int) -> None:
        self._gpu_alloc[gpu_id] = job_id

    def _release_gpu(self, job_id: str) -> None:
        self._gpu_alloc = {k: v for k, v in self._gpu_alloc.items() if v != job_id}

    def available_capacity(self) -> int:
        return self._num_gpus - len(self._gpu_alloc)

    def submit(self, job_id: str, resolved_config: Path, experiment: dict) -> SchedulerSubmitResult:
        gpu_id = self._find_free_gpu()
        if gpu_id is None:
            raise NoCapacityError("No free GPUs available")

        self._allocate_gpu(job_id, gpu_id)

        job_results = self._results_dir / job_id
        job_results.mkdir(parents=True, exist_ok=True)
        log_path = job_results / "runner.log"

        manifest = self._results_dir / job_id / "manifest.json"

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        cmd = [
            sys.executable, "-m", "postdoc.run_job",
            "--manifest", str(manifest),
        ]

        with open(log_path, "w") as log_file:
            proc = subprocess.Popen(
                cmd,
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )

        return SchedulerSubmitResult(
            process_handle=str(proc.pid),
            gpu_ids=[gpu_id],
        )

    def cancel(self, job_id: str, process_handle: str) -> None:
        pid = int(process_handle)
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        self._release_gpu(job_id)

    def is_alive(self, process_handle: str) -> bool:
        try:
            os.kill(int(process_handle), 0)
            return True
        except PermissionError:
            return True
        except (ProcessLookupError, ValueError):
            return False

    def drain_queue(self, tracker: "JobTracker") -> None:
        from postdoc.interfaces.tracker import JobState
        queued = tracker.get_queued_jobs()
        for job in queued:
            if self.available_capacity() <= 0:
                break
            try:
                config_path = self._results_dir / job.job_id / "config.yaml"
                if not config_path.exists():
                    continue
                result = self.submit(job.job_id, config_path, job.config_snapshot)
                tracker.update_state(
                    job.job_id, JobState.SUBMITTED,
                    process_handle=result.process_handle,
                    gpu_ids=result.gpu_ids,
                )
            except NoCapacityError:
                break
