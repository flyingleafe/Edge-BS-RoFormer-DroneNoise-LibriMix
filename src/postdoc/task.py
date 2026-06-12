"""SkyPilot task YAML generators.

Two task shapes:

1. **Bootstrap** (`build_bootstrap_task`) — used by `postdoc cluster-up`.
   Launches a long-lived cluster on the SSH node pool. The pod mounts the
   repo directory from the host (`hostPath`) so that all subsequent jobs
   share one clone, one `.venv`, one `datasets/`, one `results/` etc.

2. **Exec** (`build_exec_task`) — used by `postdoc submit`. A short script
   that `git reset --hard`s the shared repo to the pushed SHA, `uv sync`s,
   and runs the user's command. Submitted with `sky exec`, so no cluster
   spin-up, no rsync, no re-download.

Schema refs:
  YAML: https://docs.skypilot.co/en/latest/reference/yaml-spec.html
  SSH pod_config: https://docs.skypilot.co/en/stable/reservations/existing-machines.html
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

DEFAULT_POOL = os.environ.get("POSTDOC_SSH_POOL", "vast-server")
DEFAULT_CLUSTER_GPUS = int(os.environ.get("POSTDOC_CLUSTER_GPUS", "2"))
DEFAULT_JOB_GPUS = int(os.environ.get("POSTDOC_DEFAULT_GPUS", "1"))
# GPU type that SkyPilot sees on the pool. Run `sky gpus list --infra ssh`
# to discover it. `:N` (any type) is invalid in task YAML — SkyPilot mangles
# it into the synthetic instance-name and crashes; the type must be concrete.
DEFAULT_GPU_TYPE = os.environ.get("POSTDOC_GPU_TYPE", "RTX4070-TI")
# Path on the host that is mounted into the pod at the same path. Must exist
# on the host. Reuses the existing vast-server clone so datasets/ + results/
# + .venv are shared across all jobs on the cluster.
DEFAULT_REPO_DIR = os.environ.get("POSTDOC_REPO_DIR", "/root/harmonic-noise-suppression")


def _accelerator_spec(gpus: int, gpu_type: str) -> str | None:
    """Return the SkyPilot accelerators string, or None for no GPU."""
    if not gpus:
        return None
    return f"{gpu_type}:{gpus}"


BOOTSTRAP_SETUP = """\
set -eo pipefail
echo "[postdoc cluster] bootstrap starting at $(date -Iseconds)"
# Install uv into the pod once per cluster lifecycle.
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"
# The repo is bind-mounted from the host where it's owned by root; the pod
# process is also root but git's "dubious ownership" safeguard trips anyway
# across the bind-mount. Whitelist the path once for the pod's root user.
git config --global --add safe.directory "$POSTDOC_REPO_DIR"
cd "$POSTDOC_REPO_DIR"
# Fast-forward the mounted checkout (no-op if nothing to fetch).
git fetch --all --prune --tags 2>&1 | tail -3 || true
# Pre-sync the venv so the first submit doesn't pay that cost.
# --no-dev skips the dev group (jupyter/nnviz/torchlens/etc.) — those
# need system-level build deps (libgraphviz-dev etc.) not present in
# the SkyPilot pod and not needed for training.
uv sync --no-dev
echo "[postdoc cluster] bootstrap done"
"""

BOOTSTRAP_RUN = """\
echo "[postdoc cluster] up; pod ready for sky exec"
sleep 10
"""


EXEC_SCRIPT = """\
set -eo pipefail
export PATH="$HOME/.local/bin:$PATH"
cd "$POSTDOC_REPO_DIR"

echo "[postdoc job] swapping code to $POSTDOC_GIT_SHA"
git fetch origin 2>&1 | tail -3
# Also fetch the refs/postdoc/* namespace for detached-HEAD submits.
git fetch origin "+refs/postdoc/*:refs/postdoc/*" 2>/dev/null || true
git reset --hard "$POSTDOC_GIT_SHA"
git submodule update --init --recursive 2>/dev/null || true
echo "[postdoc job] repo at $(git rev-parse --short HEAD)"

# uv sync: fast no-op when lockfile unchanged.
uv sync --no-dev 2>&1 | tail -5 || true

# DVC pull: fetch datasets referenced by .dvc pointers that aren't cached.
if ls *.dvc datasets/*.dvc >/dev/null 2>&1; then
  uv run dvc pull 2>&1 | tail -20 || echo "WARNING: dvc pull failed; job may lack data"
fi

# shellcheck disable=SC1091
source .venv/bin/activate
echo "[postdoc job] running: {command_label}"
{command}
"""


# Host path on vast-server holding the credentials the pod needs for
# `git fetch` from origin (and ssh hostkeys). Mounted read-only at /root/.ssh
# inside the pod so both host-root and pod-root use the same keys.
HOST_SSH_DIR = os.environ.get("POSTDOC_HOST_SSH_DIR", "/root/.ssh")


def _pod_config_with_hostpath(host_path: str, mount_path: str) -> dict[str, Any]:
    """Inject hostPath volumes (project + ~/.ssh) and runAsUser=0 into the pod spec."""
    return {
        "ssh": {
            "pod_config": {
                "spec": {
                    "securityContext": {"runAsUser": 0, "fsGroup": 0},
                    "containers": [
                        {
                            "volumeMounts": [
                                {"mountPath": mount_path, "name": "project"},
                                {"mountPath": "/root/.ssh", "name": "ssh-creds", "readOnly": True},
                            ],
                        }
                    ],
                    "volumes": [
                        {
                            "name": "project",
                            "hostPath": {"path": host_path, "type": "Directory"},
                        },
                        {
                            "name": "ssh-creds",
                            "hostPath": {"path": HOST_SSH_DIR, "type": "Directory"},
                        },
                    ],
                }
            }
        }
    }


def build_bootstrap_task(
    *,
    pool: str = DEFAULT_POOL,
    gpus: int = DEFAULT_CLUSTER_GPUS,
    gpu_type: str = DEFAULT_GPU_TYPE,
    repo_dir: str = DEFAULT_REPO_DIR,
) -> dict[str, Any]:
    """Task for `sky launch -c postdoc` — brings up the shared cluster."""
    task: dict[str, Any] = {
        "name": "postdoc-cluster",
        "resources": {"infra": f"ssh/{pool}"},
        "envs": {"POSTDOC_REPO_DIR": repo_dir},
        "setup": BOOTSTRAP_SETUP,
        "run": BOOTSTRAP_RUN,
        "config": _pod_config_with_hostpath(repo_dir, repo_dir),
    }
    acc = _accelerator_spec(gpus, gpu_type)
    if acc:
        task["resources"]["accelerators"] = acc
    return task


def build_exec_task(
    command: str,
    *,
    git_sha: str,
    git_url: str,
    name: str | None = None,
    gpus: int = DEFAULT_JOB_GPUS,
    gpu_type: str = DEFAULT_GPU_TYPE,
    repo_dir: str = DEFAULT_REPO_DIR,
    envs: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Task for `sky exec postdoc` — runs one job on the shared cluster.

    Note: `sky exec` ignores `resources.infra` / `setup` / `workdir` /
    `config` (those are set by the cluster). Only `name`, `envs`, `run`, and
    accelerator requirements are meaningful.
    """
    base_envs = {
        "POSTDOC_GIT_SHA": git_sha,
        "POSTDOC_GIT_URL": git_url,
        "POSTDOC_REPO_DIR": repo_dir,
    }
    if envs:
        base_envs.update(envs)

    label = command.replace("\n", " ")[:120]
    run_script = EXEC_SCRIPT.format(command=command, command_label=label)

    task: dict[str, Any] = {}
    if name:
        task["name"] = name
    task["resources"] = {}
    acc = _accelerator_spec(gpus, gpu_type)
    if acc:
        task["resources"]["accelerators"] = acc
    task["envs"] = base_envs
    task["run"] = run_script
    return task


class _LiteralDumper(yaml.SafeDumper):
    """yaml.SafeDumper that emits multi-line strings as ``|`` block scalars."""


def _str_representer(dumper: yaml.Dumper, data: str):
    style = "|" if "\n" in data else None
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style=style)


_LiteralDumper.add_representer(str, _str_representer)  # type: ignore[arg-type]


def dump_task_yaml(task: dict[str, Any], path: Path) -> Path:
    path.write_text(yaml.dump(task, Dumper=_LiteralDumper, sort_keys=False))
    return path


def task_to_yaml(task: dict[str, Any]) -> str:
    """Readable YAML of a task dict (multiline strings as block scalars)."""
    return yaml.dump(task, Dumper=_LiteralDumper, sort_keys=False)
