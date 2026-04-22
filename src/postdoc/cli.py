"""postdoc — thin wrapper over SkyPilot on an SSH node pool.

Architecture: one persistent cluster (`sky launch -c postdoc`) whose pod mounts
the repo directory from the host via hostPath. Jobs are submitted with
`sky exec` — no cluster spin-up, no re-download, all jobs share the cluster's
`.venv` / `datasets/` / `results/` on disk.

Commands
--------
    postdoc pool-up          sky ssh up           (k3s bootstrap, one-time)
    postdoc cluster-up       sky launch -c postdoc <bootstrap.yaml>
    postdoc cluster-down     sky down postdoc
    postdoc cluster-status   sky status postdoc
    postdoc submit <cmd>     preflight + sky exec -d postdoc <exec.yaml>
    postdoc list             sky queue postdoc
    postdoc status <id>      sky queue postdoc (filtered)
    postdoc logs <id>        sky logs postdoc <id>
    postdoc cancel <id>      sky cancel postdoc <id>
    postdoc ssh              ssh <pool>
    postdoc dashboard        sky dashboard
    postdoc check            sky check ssh + sky status
    postdoc infer <run-dir>  local inference tool (unchanged)
"""
from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import typer
import yaml

from postdoc import git_state, infer as infer_mod
from postdoc.task import (
    DEFAULT_CLUSTER_GPUS,
    DEFAULT_JOB_GPUS,
    DEFAULT_POOL,
    DEFAULT_REPO_DIR,
    build_bootstrap_task,
    build_exec_task,
    dump_task_yaml,
    task_to_yaml,
)


CLUSTER_NAME = os.environ.get("POSTDOC_CLUSTER", "postdoc")

app = typer.Typer(
    name="postdoc",
    help="Submit shell commands as sky-exec jobs on a persistent SSH-pool cluster.",
    no_args_is_help=True,
    context_settings={"help_option_names": ["-h", "--help"]},
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _check_sky_installed() -> None:
    if shutil.which("sky") is None:
        typer.echo(
            "ERROR: `sky` CLI not found. `uv sync` (ensures skypilot[kubernetes]) "
            "and re-enter the dev shell.",
            err=True,
        )
        raise typer.Exit(127)


def _run_sky(args: list[str], *, check: bool = True, capture: bool = False):
    _check_sky_installed()
    cmd = ["sky", *args]
    if capture:
        return subprocess.run(cmd, check=check, text=True,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return subprocess.run(cmd, check=check)


def _project_root() -> Path:
    p = Path.cwd().resolve()
    for parent in [p, *p.parents]:
        if (parent / ".git").exists():
            return parent
    return p


def _auto_name(command: str) -> str:
    first_word = shlex.split(command)[0] if command.strip() else "job"
    stem = Path(first_word).stem[:20] or "job"
    ts = datetime.now().strftime("%m%d-%H%M%S")
    return f"{stem}-{ts}"


def _cluster_exists() -> bool:
    """True if a cluster named CLUSTER_NAME is listed in `sky status`."""
    r = _run_sky(["status", CLUSTER_NAME, "--no-show-services"],
                 check=False, capture=True)
    return r.returncode == 0 and CLUSTER_NAME in (r.stdout or "")


# ---------------------------------------------------------------------------
# cluster lifecycle
# ---------------------------------------------------------------------------

@app.command("cluster-up")
def cmd_cluster_up(
    pool: str = typer.Option(DEFAULT_POOL, "--pool"),
    gpus: int = typer.Option(DEFAULT_CLUSTER_GPUS, "--gpus",
                             help="GPUs the cluster pod reserves (max concurrent jobs)"),
    repo_dir: str = typer.Option(DEFAULT_REPO_DIR, "--repo-dir",
                                 help="Host path mounted into the pod"),
    dry_run: bool = typer.Option(False, "--dry-run"),
):
    """Launch (or re-launch) the persistent `postdoc` cluster on the SSH pool."""
    task = build_bootstrap_task(pool=pool, gpus=gpus, repo_dir=repo_dir)
    if dry_run:
        typer.echo(task_to_yaml(task))
        return
    with tempfile.NamedTemporaryFile("w", suffix=".sky.yaml", delete=False) as tf:
        dump_task_yaml(task, Path(tf.name))
        task_path = Path(tf.name)
    try:
        # --detach-run: return after job submission; don't tail logs (which
        # would trigger a spurious FileNotFoundError when the log file isn't
        # yet created by the remote runtime).
        _run_sky(["launch", "-y", "--detach-run", "-c", CLUSTER_NAME, str(task_path)])
    finally:
        task_path.unlink(missing_ok=True)


@app.command("cluster-down")
def cmd_cluster_down(
    yes: bool = typer.Option(False, "-y", "--yes"),
):
    """Tear down the `postdoc` cluster (host data is preserved — hostPath stays)."""
    args = ["down", CLUSTER_NAME]
    if yes:
        args.append("-y")
    _run_sky(args)


@app.command("cluster-status")
def cmd_cluster_status():
    _run_sky(["status", CLUSTER_NAME], check=False)


# ---------------------------------------------------------------------------
# submit
# ---------------------------------------------------------------------------

@app.command("submit", context_settings={"allow_extra_args": True,
                                         "ignore_unknown_options": True})
def cmd_submit(
    ctx: typer.Context,
    name: str | None = typer.Option(None, "--name", "-n"),
    gpus: int = typer.Option(DEFAULT_JOB_GPUS, "--gpus", "-g"),
    remote: str = typer.Option("origin", "--remote"),
    allow_dirty: bool = typer.Option(False, "--dirty"),
    skip_push: bool = typer.Option(False, "--skip-push"),
    env: list[str] = typer.Option(None, "--env", "-e"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    auto_up: bool = typer.Option(True, "--auto-up/--no-auto-up",
                                 help="Start the cluster if it's not running"),
):
    """Submit a shell command as a job on the persistent cluster.

    Preflight: verify clean tree, push HEAD to origin. The job on the cluster
    runs `git reset --hard <SHA>` + `uv sync` before executing the command.
    """
    command_tokens = ctx.args
    if not command_tokens:
        typer.echo("ERROR: no command given. Usage: postdoc submit <command...>", err=True)
        raise typer.Exit(2)
    command = shlex.join(command_tokens)

    env_dict: dict[str, str] = {}
    for kv in env or []:
        if "=" not in kv:
            typer.echo(f"ERROR: --env expects KEY=VALUE, got {kv!r}", err=True)
            raise typer.Exit(2)
        k, v = kv.split("=", 1)
        env_dict[k] = v

    # Git preflight (always — even for dry-run — so the user sees push errors early).
    try:
        snap = git_state.snapshot(
            cwd=_project_root(),
            allow_dirty=allow_dirty,
            skip_push=skip_push,
            remote=remote,
        )
    except git_state.GitError as e:
        typer.echo(f"ERROR: {e}", err=True)
        raise typer.Exit(3)

    typer.echo(
        f"[postdoc] branch={snap['branch']}  sha={snap['sha'][:12]}  "
        f"push={snap['refspec']}  dirty={snap['dirty']}"
    )
    if allow_dirty and snap["dirty"] == "True":
        typer.echo("WARNING: --dirty was set; uncommitted changes are NOT on the remote.",
                   err=True)

    task = build_exec_task(
        command=command,
        git_sha=snap["sha"],
        git_url=snap["url"],
        name=name or _auto_name(command),
        gpus=gpus,
        envs=env_dict or None,
    )
    task_name = task.get("name", "job")

    if dry_run:
        typer.echo(task_to_yaml(task))
        return

    # Auto-up the cluster on first submit.
    if auto_up and not _cluster_exists():
        typer.echo(f"[postdoc] cluster '{CLUSTER_NAME}' not running — bringing it up...")
        _cluster_up_inline()

    with tempfile.NamedTemporaryFile("w", suffix=".sky.yaml", delete=False) as tf:
        dump_task_yaml(task, Path(tf.name))
        task_path = Path(tf.name)
    try:
        _run_sky(["exec", CLUSTER_NAME, "-d", "-n", task_name, str(task_path)])
    finally:
        task_path.unlink(missing_ok=True)


def _cluster_up_inline() -> None:
    """Auto-up used by submit when the cluster is down."""
    task = build_bootstrap_task()
    with tempfile.NamedTemporaryFile("w", suffix=".sky.yaml", delete=False) as tf:
        dump_task_yaml(task, Path(tf.name))
        task_path = Path(tf.name)
    try:
        _run_sky(["launch", "-y", "--detach-run", "-c", CLUSTER_NAME, str(task_path)])
    finally:
        task_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# queue / logs / cancel
# ---------------------------------------------------------------------------

@app.command("list")
def cmd_list(
    all_: bool = typer.Option(False, "--all", "-a",
                              help="Include finished jobs"),
):
    args = ["queue", CLUSTER_NAME]
    if all_:
        args.append("--all")
    _run_sky(args, check=False)


@app.command("status")
def cmd_status(job_id: int = typer.Argument(...)):
    """Show a single job (filters the queue locally)."""
    r = _run_sky(["queue", CLUSTER_NAME, "--all"], check=False, capture=True)
    lines = (r.stdout or "").splitlines()
    head = [l for l in lines[:3] if l.strip()]
    match = [l for l in lines if l.split()[:1] == [str(job_id)]]
    out = "\n".join(head + match) if match else f"No such job: {job_id}"
    typer.echo(out)


@app.command("logs")
def cmd_logs(
    job_id: int = typer.Argument(...),
    follow: bool = typer.Option(True, "--follow/--no-follow", "-f"),
):
    args = ["logs", CLUSTER_NAME, str(job_id)]
    if not follow:
        args.append("--no-follow")
    _run_sky(args, check=False)


@app.command("cancel")
def cmd_cancel(
    job_ids: list[int] = typer.Argument(None),
    all_: bool = typer.Option(False, "--all"),
    yes: bool = typer.Option(False, "-y"),
):
    args = ["cancel", CLUSTER_NAME]
    if yes:
        args.append("-y")
    if all_:
        args.append("--all")
    else:
        if not job_ids:
            typer.echo("ERROR: pass job IDs or --all", err=True)
            raise typer.Exit(2)
        args += [str(i) for i in job_ids]
    _run_sky(args)


# ---------------------------------------------------------------------------
# ssh / dashboard / check / pool-up
# ---------------------------------------------------------------------------

@app.command("ssh")
def cmd_ssh(pool: str = typer.Option(DEFAULT_POOL, "--pool")):
    """Interactive shell on the GPU host (bypasses k8s; direct ssh)."""
    subprocess.run(["ssh", "-t", pool])


@app.command("dashboard")
def cmd_dashboard():
    _run_sky(["dashboard"])


@app.command("check")
def cmd_check():
    _run_sky(["check", "ssh"], check=False)
    _run_sky(["status"], check=False)


@app.command("pool-up")
def cmd_pool_up():
    """Bootstrap the SSH node pool (k3s install on the remote host). One-time."""
    _run_sky(["ssh", "up"])


@app.command("pool-down")
def cmd_pool_down():
    """Tear down the SSH node pool (removes k3s)."""
    _run_sky(["ssh", "down"])


# ---------------------------------------------------------------------------
# infer (local, unchanged)
# ---------------------------------------------------------------------------

@app.command("infer", help="Run inference with a trained model on audio files.")
def cmd_infer(
    run: str = typer.Argument(...),
    checkpoint: str = typer.Option("best", "--checkpoint", "-c"),
    input: str = typer.Option(..., "--input", "-i"),
    output: str = typer.Option(..., "--output", "-o"),
    device: int = typer.Option(0, "--device", "-d"),
    rps_file: str | None = typer.Option(None, "--rps-file"),
):
    infer_mod.infer_cmd(
        run, checkpoint=checkpoint, input=input,
        output=output, device=device, rps_file=rps_file,
    )


def main() -> None:
    app()


if __name__ == "__main__":
    main()
