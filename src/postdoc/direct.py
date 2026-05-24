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
import textwrap
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
    return ["ssh", "-o", "BatchMode=yes", "-o", "ServerAliveInterval=60",
            "-o", "ServerAliveCountMax=3", f"{user}@{host}"]


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
    # Use a heredoc to avoid Python one-liner limitations with for-loops.
    script = textwrap.dedent("""
        import glob, json, os
        jobs = []
        for d in sorted(glob.glob('/root/.postdoc/jobs/*/')):
            jf = os.path.join(d, 'job.json')
            if not os.path.isfile(jf):
                continue
            try:
                d2 = json.load(open(jf))
                bn = os.path.basename(d.rstrip('/'))
                parts = bn.rsplit('__', 1)
                d2['name'] = d2.get('name', parts[0] if len(parts) == 2 else bn)
                jobs.append(d2)
            except Exception:
                pass
        print(json.dumps(jobs))
    """).strip()
    cmd = _ssh_base(user, host) + [f"python3 << 'PYEOF'\n{script}\nPYEOF"]
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
                       repo_dir: str = DEFAULT_REPO_DIR,
                       no_sync: bool = False) -> str:
    """Return the bash script that runs one job inside the job dir.

    no_sync=True skips `git fetch / reset --hard` and `dvc pull`.
    Use when datasets are already on the server and you don't want
    in-flight DVC state clobbered (e.g. during an active `dvc push`).
    """
    gpu_env = " ".join(map(str, gpu_mask))

    if no_sync:
        sync_block = "echo \"[postdoc-job] --no-sync: skipping git fetch/reset and dvc pull\""
    else:
        sync_block = f"""\
echo "[postdoc-job] fetching code at $POSTDOC_GIT_SHA"
git fetch origin 2>&1 | tail -3 || true
git fetch origin "+refs/postdoc/*:refs/postdoc/*" 2>/dev/null || true
git reset --hard "$POSTDOC_GIT_SHA"
git submodule update --init --recursive 2>/dev/null || true

# dvc pull
if ls *.dvc datasets/*.dvc >/dev/null 2>&1; then
  echo "[postdoc-job] pulling datasets"
  uv run dvc pull 2>&1 | tail -20 || echo "WARNING: dvc pull failed"
fi"""

    return f"""\
set -eo pipefail
export CUDA_VISIBLE_DEVICES="{gpu_env}"
export POSTDOC_JOB_ID="{job_id}"
export POSTDOC_JOB_NAME="{name}"
export POSTDOC_GIT_SHA="{sha}"

REPO_DIR="{repo_dir}"
LOG_FILE="{log_path}"

cd "$REPO_DIR"

{sync_block}

# Load .env (AWS_*, WANDB_API_KEY, R2_ACCOUNT_ID, AWS_DEFAULT_REGION) into the
# shell so subprocess invocations of `dvc`, `aws`, etc. see them. Python code
# also calls `load_dotenv()` itself, but `uv run dvc pull` below does not.
if [ -f .env ]; then
  set -a
  . ./.env
  set +a
fi

# uv sync: fast no-op when lockfile unchanged.
echo "[postdoc-job] syncing venv"
export PATH="/root/.local/bin:$REPO_DIR/.venv/bin:$PATH"
uv sync --no-dev 2>&1 | tail -5 || true

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
        "ids = [int(os.path.basename(d.rstrip('/')).split('__')[1]) for d in dirs] "
        "if dirs else [0]; print(max(ids) + 1)"
    )
    cmd = _ssh_base(user, host) + [f"python3 -c {script!r}"]
    try:
        return int(subprocess.check_output(cmd, text=True).strip())
    except subprocess.CalledProcessError:
        return 1


def _write_and_launch(job_id: int, name: str, sha: str, cmd: str,
                      gpus: int, gpu_mask: list[int], log_path: str,
                      job_dir: str, job_json_path: str,
                      job_script: str,
                      user: str, host: str) -> tuple[int, str]:
    """Write job dir + run.sh + job.json on server, then nohup it."""

    # Use a single Python heredoc on the server to avoid all quote issues
    setup_py = _SetupPy(
        job_dir=job_dir,
        job_json={
            "id": job_id,
            "name": name,
            "sha": sha,
            "cmd": cmd,
            "gpus": gpus,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "status": "running",
            "pid": None,
            "gpu_mask": gpu_mask,
            "log_path": log_path,
        },
        run_script=job_script,
    ).render()

    subprocess.run(
        _ssh_base(user, host) + [f"python3 - << 'PYEOF'\n{setup_py}\nPYEOF"],
        check=True,
    )

    # Launch nohup, capture PID
    launch_script = (
        f"cd {job_dir} && "
        f"nohup bash run.sh >> {log_path} 2>&1 </dev/null & "
        f"echo $! > {job_dir}/pid.txt && "
        f"echo PID:$(cat {job_dir}/pid.txt)"
    )
    out = subprocess.check_output(
        _ssh_base(user, host) + ["-n", f"bash -c {launch_script!r}"],
        text=True,
    )

    # Extract PID and update job.json
    pid_match = re.search(r'PID:(\d+)', out)
    if pid_match:
        pid = int(pid_match.group(1))
        pid_py = (
            f"import json, pathlib; "
            f"d=json.loads(pathlib.Path({job_json_path!r}).read_text()); "
            f"d['pid']={pid}; "
            f"json.dump(d, pathlib.Path({job_json_path!r}).open('w'))"
        )
        subprocess.run(_ssh_base(user, host) + [f"python3 -c {pid_py!r}"], check=True)
    return job_id, "running"


class _SetupPy:
    """Render a server-side Python setup script for one job (avoids quote hell)."""

    def __init__(self, job_dir: str, job_json: dict, run_script: str):
        self._job_dir = job_dir
        self._job_json = job_json
        self._run_script = run_script

    def render(self) -> str:
        return "\n".join([
            "import json, os",
            f"os.makedirs({self._job_dir!r}, exist_ok=True)",
            f"with open({self._job_json_path!r}, 'w') as f:",
            f"    json.dump({self._job_json!r}, f)",
            f"with open({self._run_sh_path!r}, 'w') as f:",
            f"    f.write({self._run_script!r})",
            f"os.chmod({self._run_sh_path!r}, 0o755)",
        ])

    @property
    def _job_json_path(self) -> str:
        return f"{self._job_dir}/job.json"

    @property
    def _run_sh_path(self) -> str:
        return f"{self._job_dir}/run.sh"


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
    no_sync: bool = False,
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
        gpu_mask = available[:gpus]
        job_script = _render_job_script(job_id, name, sha, cmd, gpu_mask,
                                        log_path, repo_dir, no_sync=no_sync)
        return _write_and_launch(
            job_id, name, sha, cmd, gpus, gpu_mask, log_path,
            job_dir, job_json_path, job_script, user, host,
        )
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
