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

import base64
import json
import re
import subprocess
import textwrap
from dataclasses import dataclass
from datetime import UTC, datetime

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


def _ssh_base(user: str = DEFAULT_SERVER_USER, host: str = DEFAULT_SERVER_HOST) -> list[str]:
    return [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ServerAliveInterval=60",
        "-o",
        "ServerAliveCountMax=3",
        f"{user}@{host}",
    ]


# ------------------------------------------------------------------ #
# GPU probing
# ------------------------------------------------------------------ #


def probe_gpus(
    *, user: str = DEFAULT_SERVER_USER, host: str = DEFAULT_SERVER_HOST
) -> list[GPUInfo]:
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
        idx, used, total, util = re.split(r"\s*,\s*", line.strip())
        gpus.append(
            GPUInfo(
                index=int(idx),
                memory_used_mib=int(used),
                memory_total_mib=int(total),
                utilization=int(util),
            )
        )
    return gpus


def free_gpus(
    *, user: str = DEFAULT_SERVER_USER, host: str = DEFAULT_SERVER_HOST, threshold_mib: int = 500
) -> list[int]:
    """GPU indices with < threshold_mib memory used."""
    return [g.index for g in probe_gpus(user=user, host=host) if g.memory_used_mib < threshold_mib]


# ------------------------------------------------------------------ #
# job filesystem helpers
# ------------------------------------------------------------------ #


def _ensure_postdoc_dir(user: str = DEFAULT_SERVER_USER, host: str = DEFAULT_SERVER_HOST) -> None:
    """Create /root/.postdoc/ and jobs/ on the server if they don't exist."""
    subprocess.run(
        _ssh_base(user, host)
        + [
            "bash -c '"
            "mkdir -p /root/.postdoc/jobs && "
            "[ -p /root/.postdoc/queue.fifo ] || mkfifo /root/.postdoc/queue.fifo"
            "'"
        ],
        check=True,
    )


def list_jobs(user: str = DEFAULT_SERVER_USER, host: str = DEFAULT_SERVER_HOST) -> list[JobInfo]:
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
        jobs.append(
            JobInfo(
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
            )
        )
    return jobs


# ------------------------------------------------------------------ #
# submit
# ------------------------------------------------------------------ #


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

    # Create placeholder dir immediately so _next_job_id never reuses this ID
    # for a concurrent submit. The daemon will overwrite with real contents.
    subprocess.run(
        _ssh_base(user, host) + [f"mkdir -p {job_dir}"],
        check=True,
    )

    # Always queue — let the daemon handle launching locally.
    # Direct SSH launch is flaky: nohup + backgrounding over SSH often
    # keeps the connection open, causing postdoc submit to hang.
    job_desc = {
        "id": job_id,
        "name": name,
        "sha": sha,
        "cmd": cmd,
        "gpus": gpus,
        "log_path": log_path,
        "started_at": datetime.now(UTC).isoformat(),
    }
    fifo_payload = json.dumps(job_desc)
    payload_b64 = base64.b64encode(fifo_payload.encode()).decode()
    subprocess.run(
        _ssh_base(user, host)
        + [
            f"python3 -c 'import base64; print(base64.b64decode(\"{payload_b64}\").decode())' "
            f">> {postdoc_dir}/queue.fifo"
        ],
        check=True,
    )
    return job_id, "queued"


# ------------------------------------------------------------------ #
# logs
# ------------------------------------------------------------------ #


def read_logs(
    name_and_id: str,
    *,
    user: str = DEFAULT_SERVER_USER,
    host: str = DEFAULT_SERVER_HOST,
    follow: bool = False,
    lines: int = 50,
) -> str:
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


def cancel_job(
    name_and_id: str, user: str = DEFAULT_SERVER_USER, host: str = DEFAULT_SERVER_HOST
) -> bool:
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
        subprocess.run(_ssh_base(user, host) + [f"kill {pid} 2>/dev/null; echo ok"], check=False)
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
