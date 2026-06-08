"""Cloud backend for postdoc via SkyPilot managed jobs.

Uses `sky jobs launch` — SkyPilot's native managed-jobs API. Each job
gets a fresh container (no shared state), appropriate for ephemeral cloud
nodes. This is the complement to the direct SSH backend.

Prerequisites on the local machine:
    - sky CLI installed and authenticated (`sky check`)
    - Cloud credentials configured (AWS / GCP / Azure / Lambda)

Usage:
    postdoc submit --cloud <cmd...>   # force cloud routing
    postdoc submit --cloud-only <cmd>  # skip direct even if GPUs available

The cloud backend ignores vast-server GPU availability entirely. It lets
SkyPilot handle resource allocation and provisioning.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

# ------------------------------------------------------------------ #
# constants
# ------------------------------------------------------------------ #

# SkyPilot cloud to use when auto-selecting. Set POSTDOC_CLOUD to override.
DEFAULT_CLOUD = "kubernetes"  # fallback to auto-selection
REPO_DIR = "/root/harmonic-noise-suppression"


# ------------------------------------------------------------------ #
# types
# ------------------------------------------------------------------ #


@dataclass
class CloudJobInfo:
    id: int
    name: str
    sha: str
    cmd: str
    gpus: int
    status: str
    cloud: str
    region: str | None


# ------------------------------------------------------------------ #
# helpers
# ------------------------------------------------------------------ #


def _sky_available() -> bool:
    return shutil.which("sky") is not None


def _run_sky(
    args: list[str], *, check: bool = True, capture: bool = False
) -> subprocess.CompletedProcess:
    if not _sky_available():
        raise RuntimeError("sky CLI not found. Run `sky check` first.")
    cmd = ["sky", *args]
    if capture:
        return subprocess.run(
            cmd, check=check, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
    return subprocess.run(cmd, check=check)


def _project_root() -> Path:
    p = Path.cwd().resolve()
    for parent in [p, *p.parents]:
        if (parent / ".git").exists():
            return parent
    return p


# ------------------------------------------------------------------ #
# GPU request string
# ------------------------------------------------------------------ #


def _accelerator_str(gpus: int, gpu_type: str | None = None) -> str:
    """SkyPilot accelerator spec for task YAML."""
    if gpus == 0:
        return "null"
    if gpu_type:
        return f"{gpu_type}:{gpus}"
    return f":{gpus}"  # any GPU type, N GPUs


# ------------------------------------------------------------------ #
# managed job YAML (inline, no separate file needed)
# ------------------------------------------------------------------ #


_CLOUD_TASK = """\
{accelerators}
{num_nodes}
{cloud}
{region}
{resources}
name: "{name}"
envs:
  POSTDOC_GIT_SHA: "{sha}"
  POSTDOC_GIT_URL: "{url}"
  POSTDOC_REPO_DIR: "{repo_dir}"
setup: |
  set -eo pipefail
  echo "[cloud job] bootstrap starting"
  if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
  fi
  export PATH="$HOME/.local/bin:$PATH"
  git config --global --add safe.directory "$POSTDOC_REPO_DIR"
  cd "$POSTDOC_REPO_DIR"
  git fetch origin "+refs/postdoc/*:refs/postdoc/*" 2>/dev/null || true
  git reset --hard "$POSTDOC_GIT_SHA"
  git submodule update --init --recursive 2>/dev/null || true
  uv sync --no-dev 2>&1 | tail -5 || true
  if ls *.dvc datasets/*.dvc >/dev/null 2>&1; then
    uv run dvc pull 2>&1 | tail -20 || echo "WARNING: dvc pull failed"
  fi
  echo "[cloud job] bootstrap done"
run: |
  set -eo pipefail
  export PATH="$HOME/.local/bin:$PATH"
  export CUDA_VISIBLE_DEVICES=0
  cd "$POSTDOC_REPO_DIR"
  # shellcheck disable=SC1091
  source .venv/bin/activate
  echo "[cloud job] running: {cmd_label}"
  {cmd}
"""


# ------------------------------------------------------------------ #
# submit
# ------------------------------------------------------------------ #


def submit_cloud(
    name: str,
    sha: str,
    url: str,
    cmd: str,
    gpus: int,
    *,
    gpu_type: str | None = None,
    cloud: str | None = None,
    region: str | None = None,
    repo_dir: str = REPO_DIR,
    envs: dict[str, str] | None = None,
    dry_run: bool = False,
) -> tuple[int, str]:
    """Submit a job via SkyPilot managed jobs.

    Returns (job_id, "submitted").
    Raises on failure.
    """
    acc = _accelerator_str(gpus, gpu_type)
    cloud_line = f"cloud: {cloud}" if cloud else ""
    region_line = f"region: {region}" if region else ""

    task_yaml = _CLOUD_TASK.format(
        accelerators=f"accelerators: {acc}" if gpus > 0 else "# no GPU",
        num_nodes="num_nodes: 1",
        cloud=cloud_line,
        region=region_line,
        resources="# resources section handled by accelerators above",
        name="",
        sha=sha,
        url=url,
        repo_dir=repo_dir,
        cmd_label=cmd.replace("\n", " ")[:120],
        cmd=cmd,
    )

    with tempfile.NamedTemporaryFile("w", suffix=".sky.yaml", delete=False) as tf:
        tf.write(task_yaml)
        task_path = Path(tf.name)

    try:
        r = _run_sky(
            ["jobs", "launch", "-y", "--detach-run", "-n", name, str(task_path)],
            capture=True,
        )
    finally:
        task_path.unlink(missing_ok=True)

    # Parse job ID from output: "Job ID: <id>"
    match = re.search(r"Job ID:\s*(\d+)", r.stdout or "")
    if not match:
        raise RuntimeError(f"Could not parse job ID from sky output:\n{r.stdout}")
    job_id = int(match.group(1))
    return job_id, "submitted"


# ------------------------------------------------------------------ #
# list / status / cancel / logs (delegate to sky jobs)
# ------------------------------------------------------------------ #


def list_jobs_cloud() -> list[CloudJobInfo]:
    """List all managed jobs via `sky jobs queue`."""
    r = _run_sky(["jobs", "queue", "--all"], capture=True, check=False)
    # Output format: header + one line per job: "<id>  <name>  <status>  ..."
    jobs = []
    for line in (r.stdout or "").splitlines()[2:]:  # skip header lines
        if not line.strip():
            continue
        parts = re.split(r"\s{2,}", line.strip())
        if len(parts) < 3:
            continue
        try:
            jobs.append(
                CloudJobInfo(
                    id=int(parts[0]),
                    name=parts[1],
                    status=parts[2],
                    sha="",
                    cmd="",
                    gpus=0,
                    cloud=cloud,  # pyright: ignore[reportUndefinedVariable]
                    region=region,  # pyright: ignore[reportUndefinedVariable]
                )
            )
        except ValueError:
            continue
    return jobs


def cancel_job_cloud(job_id: int) -> None:
    _run_sky(["jobs", "cancel", str(job_id), "-y"])


def logs_job_cloud(job_id: int, *, follow: bool = False) -> str:
    args = ["logs", str(job_id)]
    if not follow:
        args.append("--no-follow")
    r = _run_sky(["jobs"] + args, capture=True, check=False)
    return r.stdout or r.stderr or ""
