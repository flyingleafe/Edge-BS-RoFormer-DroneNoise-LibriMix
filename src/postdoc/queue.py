"""postdoc queue runner.

Two entry points:

  postdoc queue-start   — starts the tmux daemon on vast-server
  postdoc queue-stop    — stops it
  postdoc queue-status — is it running?

And `run_one(job_json_path)` called by the runner to execute a single job.

The runner loop (postdoc-runner) lives in a tmux session called
`postdoc-queue` on vast-server. It:

1. Maintains a GPU allocation table in memory.
2. Reads job descriptors from /root/.postdoc/queue.fifo (blocking, 60s timeout).
3. When free GPUs >= job.gpus: runs it immediately (nohup postdoc-job).
4. When not enough: re-queues the job (writes back to FIFO).
5. Polls running jobs every 10s: marks done/failed based on pid existence,
   frees their GPU allocations.
"""

from __future__ import annotations

import atexit
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

# ------------------------------------------------------------------ #
# constants (mirror direct.py)
# ------------------------------------------------------------------ #

POSTDOC_DIR = "/root/.postdoc"
POSTDOC_QUEUE_FIFO = f"{POSTDOC_DIR}/queue.fifo"
POSTDOC_JOBS_DIR = f"{POSTDOC_DIR}/jobs"
POSTDOC_REPO_DIR = os.environ.get("POSTDOC_REPO_DIR", "/root/harmonic-noise-suppression")
GPU_FREE_THRESHOLD_MIB = 500
POLL_INTERVAL = 10
FIFO_TIMEOUT = 60

# ------------------------------------------------------------------ #
# GPU allocation table
# ------------------------------------------------------------------ #

# {gpu_index: job_id_or_None}
_gpu_table: dict[int, int | None] = {}


def _nvidia_smi_free() -> list[int]:
    """Parse nvidia-smi output, return indices of free GPUs (<500 MiB used)."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
            text=True,
        )
    except Exception:
        return []
    free = []
    for line in out.strip().splitlines():
        idx_str, used_str = re.split(r"\s*,\s*", line.strip())
        if int(used_str) < GPU_FREE_THRESHOLD_MIB:
            free.append(int(idx_str))
    return free


def _alloc_gpus(gpus_needed: int, job_id: int) -> list[int] | None:
    """Try to allocate gpus_needed GPUs to job_id.

    Returns the list of allocated indices, or None if not enough free GPUs.
    Updates _gpu_table in place.
    """
    free = _nvidia_smi_free()
    # Exclude any still tracked as allocated
    candidates = [i for i in free if _gpu_table.get(i) is None]
    if len(candidates) < gpus_needed:
        return None
    allocated = candidates[:gpus_needed]
    for idx in allocated:
        _gpu_table[idx] = job_id
    return allocated


def _free_gpus_for_job(job_id: int) -> None:
    """Remove all GPU allocations for job_id."""
    for idx, jid in list(_gpu_table.items()):
        if jid == job_id:
            _gpu_table[idx] = None


def _update_job_status(job_dir: str, status: str) -> None:
    """Write status field into job.json."""
    subprocess.run(
        [
            "python3",
            "-c",
            f"import json, pathlib; "
            f"d=json.loads(pathlib.Path('{job_dir}/job.json').read_text()); "
            f"d['status']='{status}'; "
            f"json.dump(d, pathlib.Path('{job_dir}/job.json').open('w'))",
        ],
        check=False,
    )


# ------------------------------------------------------------------ #
# job execution
# ------------------------------------------------------------------ #


def _job_script(
    job_id: int, name: str, sha: str, cmd: str, gpu_mask: list[int], log_path: str
) -> str:
    gpu_env = " ".join(map(str, gpu_mask))
    return f"""\
set -eo pipefail
export CUDA_VISIBLE_DEVICES="{gpu_env}"
export POSTDOC_JOB_ID="{job_id}"
export POSTDOC_JOB_NAME="{name}"
export POSTDOC_GIT_SHA="{sha}"

REPO_DIR="{POSTDOC_REPO_DIR}"
LOG_FILE="{log_path}"

cd "$REPO_DIR"

echo "[postdoc-job] fetching code at $POSTDOC_GIT_SHA"
git fetch origin 2>&1 | tail -3 || true
git fetch origin "+refs/postdoc/*:refs/postdoc/*" 2>/dev/null || true
git reset --hard "$POSTDOC_GIT_SHA"
git submodule update --init --recursive 2>/dev/null || true

echo "[postdoc-job] syncing venv"
uv sync --no-dev 2>&1 | tail -5 || true

if ls *.dvc datasets/*.dvc >/dev/null 2>&1; then
  echo "[postdoc-job] pulling datasets"
  uv run dvc pull 2>&1 | tail -20 || echo "WARNING: dvc pull failed"
fi

echo "[postdoc-job] activating venv"
# shellcheck disable=SC1091
export PATH="/root/.local/bin:$REPO_DIR/.venv/bin:$PATH"

echo "[postdoc-job] running: {cmd}"
echo "[postdoc-job] started at $(date -Iseconds)"
{cmd} >> "$LOG_FILE" 2>&1
EXIT=$?

echo "[postdoc-job] done at $(date -Iseconds) exit=$EXIT"
exit $EXIT
"""


def run_one(job_json: dict[str, Any]) -> None:
    """Execute a single job.

    Called by the runner loop. Takes ownership of the job.json fields:
    id, name, sha, cmd, gpus, log_path.
    """
    job_id = job_json["id"]
    name = job_json["name"]
    sha = job_json["sha"]
    cmd = job_json["cmd"]
    gpus = job_json["gpus"]
    log_path = job_json["log_path"]

    job_dir = f"{POSTDOC_JOBS_DIR}/{name}__{job_id}"

    # Allocate GPUs
    gpu_mask = _alloc_gpus(gpus, job_id)
    if gpu_mask is None:
        raise RuntimeError("not enough free GPUs")

    # Write job.json (status=running) + run.sh
    job_script = _job_script(job_id, name, sha, cmd, gpu_mask, log_path)

    setup_lines = [
        f"mkdir -p {job_dir}",
        f"chmod +x {job_dir}",
        f"cat > {job_dir}/run.sh << 'SHEOF'\n{job_script}\nSHEOF",
        "chmod +x run.sh",
        f"cat > {job_dir}/job.json << 'JOBEOF'",
        json.dumps(
            {
                "id": job_id,
                "name": name,
                "sha": sha,
                "cmd": cmd,
                "gpus": gpus,
                "started_at": job_json["started_at"],
                "status": "running",
                "pid": None,
                "gpu_mask": gpu_mask,
                "log_path": log_path,
            }
        ),
        "JOBEOF",
    ]
    subprocess.run(["bash", "-c", " && ".join(setup_lines)], check=True)

    # nohup it
    pid_path = f"{job_dir}/pid.txt"
    launch = (
        f"cd {job_dir} && "
        f"nohup bash run.sh >> {log_path} 2>&1 </dev/null & "
        f"echo $! > {pid_path} && "
        f'python3 -c "'
        f"import json, pathlib; "
        f"d=json.loads(pathlib.Path('{job_dir}/job.json').read_text()); "
        f"d['pid']=int(pathlib.Path('{pid_path}').read_text().strip()); "
        f"json.dump(d, pathlib.Path('{job_dir}/job.json').open('w'))\""
    )
    subprocess.run(["bash", "-c", launch], check=True)

    _gpu_table.update({idx: job_id for idx in gpu_mask})


# ------------------------------------------------------------------ #
# re-queue logic
# ------------------------------------------------------------------ #


def _reenqueue(job_json: dict[str, Any]) -> None:
    """Write a job back to the queue FIFO (for when GPUs aren't free)."""
    # Remove gpu_mask / started_at — runner will fill those in when it runs
    clean = {k: v for k, v in job_json.items() if k not in ("gpu_mask", "pid", "status")}
    with open(POSTDOC_QUEUE_FIFO, "w") as fifo:
        fifo.write(json.dumps(clean) + "\n")
        fifo.flush()


# ------------------------------------------------------------------ #
# runner loop
# ------------------------------------------------------------------ #


def _poll_and_gc() -> None:
    """Check running jobs, mark done/failed if their pids are gone."""
    import glob

    for job_path in glob.glob(f"{POSTDOC_JOBS_DIR}/*/job.json"):
        try:
            d = json.loads(Path(job_path).read_text())
        except Exception:
            continue
        if d.get("status") != "running":
            continue
        pid = d.get("pid")
        if pid is None:
            continue
        # Check if process is still alive
        try:
            os.kill(pid, 0)
        except OSError:
            # Process dead — determine exit status from log
            log_path = d.get("log_path", "")
            exit_code = 1
            if log_path and Path(log_path).exists():
                last_line = Path(log_path).read_text().splitlines()[-1:][0]
                m = re.search(r"exit=(\d+)", last_line)
                if m:
                    exit_code = int(m.group(1))
            status = "failed" if exit_code != 0 else "done"
            job_dir = str(Path(job_path).parent)
            _update_job_status(job_dir, status)
            _free_gpus_for_job(d["id"])


def _read_fifo_nonblocking() -> dict[str, Any] | None:
    """Read one JSON object from the queue FIFO (non-blocking)."""
    import select

    try:
        rf, _, _ = select.select([open(POSTDOC_QUEUE_FIFO)], [], [], 0.1)
    except Exception:
        return None
    if not rf:
        return None
    fifo = rf[0]
    line = fifo.readline()
    if not line:
        return None
    try:
        return json.loads(line.strip())
    except json.JSONDecodeError:
        return None


def _read_fifo_blocking() -> dict[str, Any] | None:
    """Read one JSON object from the queue FIFO (blocks up to FIFO_TIMEOUT s)."""
    # Use select with timeout, then read one line
    import select

    try:
        fifo_fd = open(POSTDOC_QUEUE_FIFO)
        atexit.register(lambda: fifo_fd.close())
    except Exception:
        return None

    try:
        rf, _, _ = select.select([fifo_fd], [], [], FIFO_TIMEOUT)
    except Exception:
        return None
    if not rf:
        return None
    line = fifo_fd.readline()
    if not line:
        return None
    try:
        return json.loads(line.strip())
    except json.JSONDecodeError:
        return None


def _runner_loop() -> None:
    """Main loop for the runner daemon."""
    print("[postdoc-runner] started", flush=True)
    print(f"[postdoc-runner] watching {POSTDOC_QUEUE_FIFO}", flush=True)

    while True:
        _poll_and_gc()

        job = _read_fifo_blocking()
        if job is None:
            continue

        job_id = job["id"]
        gpus = job["gpus"]

        gpu_mask = _alloc_gpus(gpus, job_id)
        if gpu_mask is None:
            # Not enough free GPUs — re-queue and sleep
            job["started_at"] = job.get("started_at", "")
            _reenqueue(job)
            print(
                f"[postdoc-runner] job {job_id} ({job['name']}) "
                f"re-queued (need {gpus} GPUs, none free)",
                flush=True,
            )
            time.sleep(POLL_INTERVAL)
            continue

        job["started_at"] = job.get("started_at", "")
        try:
            run_one(job)
            print(
                f"[postdoc-runner] started job {job_id} ({job['name']}) on GPUs {gpu_mask}",
                flush=True,
            )
        except Exception as e:
            print(f"[postdoc-runner] failed to start job {job_id}: {e}", flush=True)
            # Re-queue
            _reenqueue(job)
            time.sleep(POLL_INTERVAL)


# ------------------------------------------------------------------ #
# CLI entry points (postdoc-runner script)
# ------------------------------------------------------------------ #


def main() -> None:
    """postdoc-runner entry point: start the queue daemon."""
    if len(sys.argv) > 1 and sys.argv[1] == "run_one":
        # Used by the runner to exec a job
        raise NotImplementedError("run_one is called directly, not via CLI")

    # Start the runner loop in the foreground.
    # Wrapped by tmux in the actual daemon.
    _runner_loop()
