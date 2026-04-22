"""Direct SSH backend for postdoc.

Probes vast-server for free GPUs, submits jobs over plain SSH, manages
job state via the filesystem on the server.

Server state layout (on vast-server at /root/.postdoc/):
    jobs/               ← one dir per job: <name>__<id>/
      <name>__<id>/
        job.json        ← {id, name, sha, cmd, gpus, started_at, status,
                           pid, gpu_mask, log_path}
        log.txt         ← stdout+stderr

    queue.fifo          ← named pipe; submit writes job json here
"""
from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ------------------------------------------------------------------ #
# types
# ------------------------------------------------------------------ #


@dataclass
class GPUInfo:
    index: int
    memory_used_mib: int
    memory_total_mib: int
    utilization: int  # percent


@dataclass
class JobInfo:
    id: int
    name: str
    sha: str
    cmd: str
    gpus: int
    started_at: str
    status: str  # queued | running | done | failed | cancelled
    pid: int | None
    gpu_mask: list[int] | None
    log_path: str


# ------------------------------------------------------------------ #
# server path helpers
# ------------------------------------------------------------------ #

DEFAULT_SERVER_USER = "root"
DEFAULT_SERVER_HOST = "vast-server"
DEFAULT_POSTDOC_DIR = "/root/.postdoc"
DEFAULT_REPO_DIR = "/root/harmonic-noise-suppression"


def _ssh_base(user: str = DEFAULT_SERVER_USER,
              host: str = DEFAULT_SERVER_HOST) -> list[str]:
    return ["ssh", "-o", "BatchMode=yes", f"{user}@{host}"]


# ------------------------------------------------------------------ #
# GPU probing
# ------------------------------------------------------------------ #


def probe_gpus(*,
               user: str = DEFAULT_SERVER_USER,
               host: str = DEFAULT_SERVER_HOST) -> list[GPUInfo]:
    """Return GPU info for all GPUs on the server.

    Calls `nvidia-smi --query-gpu=...` over SSH and parses output.
    Raises on failure.
    """
    cmd = _ssh_base(user, host) + [
        "nvidia-smi --query-gpu=index,memory.used,memory.total,"
        "utilization.gpu --format=csv,noheader,nounits"
    ]
    out = subprocess.check_output(cmd, text=True)
    gpus = []
    for line in out.strip().splitlines():
        idx, used, total, util = re.split(r'\s*,\s*', line.strip())
        gpus.append(GPUInfo(
            index=int(idx),
            memory_used_mib=int(used),
            memory_total_mib=int(total),
            utilization=int(util),
        ))
    return gpus


def free_gpus(*,
              user: str = DEFAULT_SERVER_USER,
              host: str = DEFAULT_SERVER_HOST,
              threshold_mib: int = 500) -> list[int]:
    """GPU indices with < threshold_mib memory used."""
    return [g.index for g in probe_gpus(user=user, host=host)
            if g.memory_used_mib < threshold_mib]


# ------------------------------------------------------------------ #
# job filesystem helpers
# ------------------------------------------------------------------ #


def _ensure_postdoc_dir(user: str = DEFAULT_SERVER_USER,
                        host: str = DEFAULT_SERVER_HOST) -> None:
    """Create /root/.postdoc/ and jobs/ on the server if they don't exist."""
    subprocess.run(
        _ssh_base(user, host) +
        ["bash -c '"
         "mkdir -p /root/.postdoc/jobs && "
         "[ -p /root/.postdoc/queue.fifo ] || mkfifo /root/.postdoc/queue.fifo"
         "'"],
        check=True,
    )


def list_jobs(user: str = DEFAULT_SERVER_USER,
              host: str = DEFAULT_SERVER_HOST) -> list[JobInfo]:
    """List all jobs by reading /root/.postdoc/jobs/*/job.json."""
    script = (
        "import glob, json, os, sys; "
        "jobs = []; "
        "for d in sorted(glob.glob('/root/.postdoc/jobs/*/')): "
        "  jf = os.path.join(d, 'job.json'); "
        "  if not os.path.isfile(jf): continue; "
        "  try: "
        "    d2 = json.load(open(jf)); "
        "    bn = os.path.basename(d); "
        "    parts = bn.rsplit('__', 1); "
        "    d2['name'] = d2.get('name', parts[0] if len(parts) == 2 else bn); "
        "    jobs.append(d2); "
        "  except: pass; "
        "print(json.dumps(jobs))"
    )
    cmd = _ssh_base(user, host) + [f"python3 -c {script!r}"]
    try:
        out = subprocess.check_output(cmd, text=True)
    except subprocess.CalledProcessError:
        return []
    try:
        data = json.loads(out.strip())
    except json.JSONDecodeError:
        return []
    jobs = []
    for d in data:
        jobs.append(JobInfo(
            id=d["id"],
            name=d.get("name", ""),
            sha=d.get("sha", ""),
            cmd=d.get("cmd", ""),
            gpus=d.get("gpus", 0),
            started_at=d.get("started_at", ""),
            status=d.get("status", ""),
            pid=d.get("pid"),
            gpu_mask=d.get("gpu_mask"),
            log_path=d.get("log_path", ""),
        ))
    return jobs


# ------------------------------------------------------------------ #
# submit
# ------------------------------------------------------------------ #

def _render_job_script(job_id: int, name: str, sha: str, cmd: str,
                       gpu_mask: list[int], log_path: str,
                       repo_dir: str = DEFAULT_REPO_DIR) -> str:
    """Return the bash script that runs one job inside the job dir."""
    gpu_env = " ".join(map(str, gpu_mask))
    return f"""\
set -eo pipefail
export CUDA_VISIBLE_DEVICES="{gpu_env}"
export POSTDOC_JOB_ID="{job_id}"
export POSTDOC_JOB_NAME="{name}"

REPO_DIR="{repo_dir}"
LOG_FILE="{log_path}"

cd "$REPO_DIR"

echo "[postdoc-job] fetching code at $POSTDOC_GIT_SHA"
git fetch origin 2>&1 | tail -3 || true
git fetch origin "+refs/postdoc/*:refs/postdoc/*" 2>/dev/null || true
git reset --hard "$POSTDOC_GIT_SHA"
git submodule update --init --recursive 2>/dev/null || true

# uv sync: fast no-op when lockfile unchanged.
echo "[postdoc-job] syncing venv"
uv sync --no-dev 2>&1 | tail -5 || true

# dvc pull
if ls *.dvc datasets/*.dvc >/dev/null 2>&1; then
  echo "[postdoc-job] pulling datasets"
  uv run dvc pull 2>&1 | tail -20 || echo "WARNING: dvc pull failed"
fi

echo "[postdoc-job] activating venv"
# shellcheck disable=SC1091
source .venv/bin/activate

echo "[postdoc-job] running: {cmd}"
echo "[postdoc-job] started at $(date -Iseconds)"
{cmd} >> "$LOG_FILE" 2>&1
EXIT=$?

echo "[postdoc-job] done at $(date -Iseconds) exit=$EXIT"
exit $EXIT
"""


def _next_job_id(user: str, host: str, postdoc_dir: str) -> int:
    script = (
        "import glob, os; "
        "dirs = glob.glob('/root/.postdoc/jobs/*/'); "
        "ids = [int(os.path.basename(d).split('__')[1]) for d in dirs] "
        "if dirs else [0]; print(max(ids) + 1)"
    )
    cmd = _ssh_base(user, host) + [f"python3 -c {script!r}"]
    try:
        return int(subprocess.check_output(cmd, text=True).strip())
    except subprocess.CalledProcessError:
        return 1


def submit_direct(
    name: str,
    sha: str,
    cmd: str,
    gpus: int,
    *,
    user: str = DEFAULT_SERVER_USER,
    host: str = DEFAULT_SERVER_HOST,
    repo_dir: str = DEFAULT_REPO_DIR,
    postdoc_dir: str = DEFAULT_POSTDOC_DIR,
) -> tuple[int, str]:
    """Submit a job via direct SSH.

    1. Ensure /root/.postdoc/{jobs,queue.fifo} exist on server.
    2. Pick the next job ID.
    3. Decide: run now (free GPUs) or queue.
    4. If running: write job dir + script, nohup it, update job.json.
       If queued: append JSON to queue.fifo.
    5. Return (job_id, status).

    Raises on SSH failure.
    """
    _ensure_postdoc_dir(user, host)
    job_id = _next_job_id(user, host, postdoc_dir)

    job_dir = f"{postdoc_dir}/jobs/{name}__{job_id}"
    log_path = f"{job_dir}/log.txt"
    job_json_path = f"{job_dir}/job.json"

    # Probe free GPUs
    available = free_gpus(user=user, host=host)

    if len(available) >= gpus:
        # Run now — allocate GPUs and launch
        gpu_mask = available[:gpus]
        gpu_env = " ".join(map(str, gpu_mask))

        job_script = _render_job_script(job_id, name, sha, cmd, gpu_mask,
                                        log_path, repo_dir)

        started_at = datetime.now(timezone.utc).isoformat()
        job_json = {
            "id": job_id,
            "name": name,
            "sha": sha,
            "cmd": cmd,
            "gpus": gpus,
            "started_at": started_at,
            "status": "running",
            "pid": None,
            "gpu_mask": gpu_mask,
            "log_path": log_path,
        }

        # Write job dir + job.json via a server-side bash script
        setup_lines = [
            f"mkdir -p {job_dir}",
            f"python3 -c {json.dumps(json.dumps(job_json))!r} > {job_json_path}",
            f"cat > {job_dir}/run.sh << 'SHEOF'\n{job_script}\nSHEOF",
            f"chmod +x {job_dir}/run.sh",
        ]
        subprocess.run(
            _ssh_base(user, host) + ["bash -c '" + " && ".join(setup_lines) + "'"],
            check=True,
        )

        # Launch nohup, capture PID
        launch_script = (
            f"cd {job_dir} && "
            f"nohup bash run.sh >> {log_path} 2>&1 & "
            f"echo $! > {job_dir}/pid.txt && "
            f"echo PID:$(cat {job_dir}/pid.txt)"
        )
        out = subprocess.check_output(
            _ssh_base(user, host) + [f"bash -c {launch_script!r}"],
            text=True,
        )

        # Extract PID and update job.json
        pid_match = re.search(r'PID:(\d+)', out)
        if pid_match:
            pid = int(pid_match.group(1))
            updater = (
                f"python3 -c "
                f"{json.dumps(json.dumps(pid))!r} "
                f"> /dev/null && "
                f"import json, pathlib; "
                f"d=json.load(open('{job_json_path}')); "
                f"d['pid']={pid}; "
                f"json.dump(d, open('{job_json_path}','w'))"
            )
            subprocess.run(
                _ssh_base(user, host) + [updater],
                check=True,
            )
        return job_id, "running"

    else:
        # Queue — write to FIFO
        job_desc = {
            "id": job_id,
            "name": name,
            "sha": sha,
            "cmd": cmd,
            "gpus": gpus,
            "log_path": log_path,
            "started_at": datetime.now(timezone.utc).isoformat(),
        }
        fifo_payload = json.dumps(job_desc)
        subprocess.run(
            _ssh_base(user, host) +
            [f"python3 -c {json.dumps(fifo_payload)!r} >> {postdoc_dir}/queue.fifo"],
            check=True,
        )
        return job_id, "queued"


# ------------------------------------------------------------------ #
# logs
# ------------------------------------------------------------------ #


def read_logs(name_and_id: str,
              *,
              user: str = DEFAULT_SERVER_USER,
              host: str = DEFAULT_SERVER_HOST,
              follow: bool = False,
              lines: int = 50) -> str:
    """Tail (or cat) the log file for a job.

    name_and_id: "<sanitized_name>__<job_id>", e.g. "dccrn__42"
    """
    job_dir = f"/root/.postdoc/jobs/{name_and_id}"
    if follow:
        cmd = _ssh_base(user, host) + [f"tail -F {job_dir}/log.txt"]
    else:
        cmd = _ssh_base(user, host) + [f"tail -{lines} {job_dir}/log.txt"]
    return subprocess.check_output(cmd, text=True)


# ------------------------------------------------------------------ #
# cancel
# ------------------------------------------------------------------ #


def cancel_job(name_and_id: str,
               user: str = DEFAULT_SERVER_USER,
               host: str = DEFAULT_SERVER_HOST) -> bool:
    """Kill the job's process and mark it cancelled.

    Returns True if the job was cancelled, False if it wasn't running.
    """
    job_dir = f"/root/.postdoc/jobs/{name_and_id}"
    job_json_path = f"{job_dir}/job.json"

    # Get PID
    pid_script = f"import json; print(json.load(open('{job_json_path}'))['pid'])"
    pid_cmd = _ssh_base(user, host) + [f"python3 -c {pid_script!r}"]
    try:
        pid_out = subprocess.check_output(pid_cmd, text=True).strip()
        pid = int(pid_out) if pid_out and pid_out != "None" else None
    except subprocess.CalledProcessError:
        pid = None

    killed = False
    if pid is not None:
        subprocess.run(_ssh_base(user, host) + [f"kill {pid} 2>/dev/null; echo ok"],
                       check=False)
        killed = True

    # Mark cancelled
    cancel_script = (
        f"import json; "
        f"d=json.load(open('{job_json_path}')); "
        f"d['status']='cancelled'; "
        f"json.dump(d, open('{job_json_path}','w'))"
    )
    subprocess.run(
        _ssh_base(user, host) + [f"python3 -c {cancel_script!r}"],
        check=False,
    )
    return killed
