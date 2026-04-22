"""postdoc — thin wrapper over SkyPilot managed jobs on an SSH node pool.

Jobs are shell commands. This CLI:
  1. Generates a minimal SkyPilot task YAML (with a `setup:` that does `dvc pull` etc.)
  2. Shells out to `sky jobs launch|queue|logs|cancel`.

One-time setup: see docs/skypilot/README.md.

Env overrides
-------------
  POSTDOC_SSH_POOL       ssh node pool name       (default: vast-server)
  POSTDOC_DEFAULT_GPUS   GPUs per job             (default: 1)
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
from postdoc.task import DEFAULT_GPUS, DEFAULT_POOL, DEFAULT_REPO_DIR, build_task, dump_task_yaml


app = typer.Typer(
    name="postdoc",
    help="Submit shell commands as managed SkyPilot jobs.",
    no_args_is_help=True,
    context_settings={"help_option_names": ["-h", "--help"]},
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _check_sky_installed() -> None:
    if shutil.which("sky") is None:
        typer.echo(
            "ERROR: `sky` CLI not found. Install with `pip install 'skypilot>=0.11'` "
            "and follow docs/skypilot/README.md to register the SSH node pool.",
            err=True,
        )
        raise typer.Exit(127)


def _run_sky(args: list[str], *, check: bool = True, capture: bool = False) -> subprocess.CompletedProcess:
    """subprocess.run(["sky", *args]) with nice errors."""
    _check_sky_installed()
    cmd = ["sky", *args]
    if capture:
        return subprocess.run(cmd, check=check, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return subprocess.run(cmd, check=check)


def _project_root() -> Path:
    """Walk up for a .git dir; fall back to CWD."""
    p = Path.cwd().resolve()
    for parent in [p, *p.parents]:
        if (parent / ".git").exists():
            return parent
    return p


def _auto_name(command: str) -> str:
    """Derive a short human-readable name from a command."""
    first_word = shlex.split(command)[0] if command.strip() else "job"
    stem = Path(first_word).stem[:20] or "job"
    ts = datetime.now().strftime("%m%d-%H%M%S")
    return f"{stem}-{ts}"


# ---------------------------------------------------------------------------
# submit
# ---------------------------------------------------------------------------

@app.command("submit", context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def cmd_submit(
    ctx: typer.Context,
    name: str | None = typer.Option(None, "--name", "-n", help="Job name (default: auto from command)"),
    gpus: int = typer.Option(DEFAULT_GPUS, "--gpus", "-g", help="GPUs per job"),
    pool: str = typer.Option(DEFAULT_POOL, "--pool", help="SSH node pool name"),
    remote: str = typer.Option("origin", "--remote", help="Git remote to push to / pull from"),
    repo_dir: str = typer.Option(DEFAULT_REPO_DIR, "--repo-dir", help="Path on the server where the repo is checked out"),
    allow_dirty: bool = typer.Option(False, "--dirty", help="Allow submission with a dirty tree (uncommitted changes will NOT be shipped)"),
    skip_push: bool = typer.Option(False, "--skip-push", help="Skip `git push` (assume HEAD is already on the remote)"),
    file: Path | None = typer.Option(None, "--file", "-f", help="Use an explicit SkyPilot task YAML (bypasses git wrapper)"),
    env: list[str] = typer.Option(None, "--env", "-e", help="ENV=val, repeatable"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print task YAML and exit without launching"),
    detach: bool = typer.Option(True, "--detach/--attach", help="Return immediately after enqueue (default) or stream logs"),
):
    """Submit a shell command as a managed SkyPilot job.

    Preflight: verify clean tree, push HEAD to the remote. The remote runs
    `git reset --hard <SHA>` + `uv sync` before executing the command.

    Examples
    --------
        postdoc submit python train.py --model_type dccrn --config configs/dccrn.yaml
        postdoc submit -n dccrn-dregon --gpus 1 -- python train.py ...
        postdoc submit --dirty python quick_test.py
        postdoc submit -f my_task.yaml
    """
    if file is not None:
        if ctx.args:
            typer.echo("ERROR: do not pass a command together with --file.", err=True)
            raise typer.Exit(2)
        # Bypass git mode entirely: user owns the task YAML and its setup/run.
        sky_args = ["jobs", "launch", "-y"]
        if name:
            sky_args += ["-n", name]
        if detach:
            sky_args.append("--detach-run")
        sky_args.append(str(file))
        _run_sky(sky_args)
        return

    command_tokens = ctx.args
    if not command_tokens:
        typer.echo("ERROR: no command given. Usage: postdoc submit <command...>", err=True)
        raise typer.Exit(2)
    command = shlex.join(command_tokens)

    env_dict = {}
    for kv in env or []:
        if "=" not in kv:
            typer.echo(f"ERROR: --env expects KEY=VALUE, got {kv!r}", err=True)
            raise typer.Exit(2)
        k, v = kv.split("=", 1)
        env_dict[k] = v

    # -- Git preflight --
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
        typer.echo(
            "WARNING: --dirty was set; uncommitted changes are NOT on the remote.",
            err=True,
        )

    task = build_task(
        command=command,
        git_sha=snap["sha"],
        git_url=snap["url"],
        name=name or _auto_name(command),
        gpus=gpus,
        pool=pool,
        repo_dir=repo_dir,
        envs=env_dict or None,
    )
    task_name = task["name"]

    if dry_run:
        typer.echo(yaml.safe_dump(task, sort_keys=False))
        return

    with tempfile.NamedTemporaryFile("w", suffix=".sky.yaml", delete=False) as tf:
        dump_task_yaml(task, Path(tf.name))
        task_path = Path(tf.name)

    try:
        sky_args = ["jobs", "launch", "-y", "-n", task_name]
        if detach:
            sky_args.append("--detach-run")
        sky_args.append(str(task_path))
        _run_sky(sky_args)
    finally:
        task_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# list / status / logs / cancel
# ---------------------------------------------------------------------------

@app.command("list")
def cmd_list(
    all_: bool = typer.Option(False, "--all", "-a", help="Include finished jobs"),
    refresh: bool = typer.Option(True, "--refresh/--no-refresh", help="Fetch fresh status"),
):
    """List managed jobs (running + queued, or all with --all)."""
    args = ["jobs", "queue"]
    if all_:
        args.append("--all")
    if refresh:
        args.append("--refresh")
    _run_sky(args)


@app.command("status")
def cmd_status(job_id: int = typer.Argument(..., help="Managed-job ID")):
    """Show status for a single managed job."""
    _run_sky(["jobs", "queue", "--refresh", "--job-ids", str(job_id)])


@app.command("logs")
def cmd_logs(
    job_id: int = typer.Argument(..., help="Managed-job ID"),
    controller: bool = typer.Option(False, "--controller", help="Show controller logs (scheduling, recovery)"),
    follow: bool = typer.Option(True, "--follow/--no-follow", "-f", help="Stream logs"),
):
    """Stream logs for a managed job."""
    args = ["jobs", "logs", str(job_id)]
    if controller:
        args.append("--controller")
    if not follow:
        args.append("--no-follow")
    _run_sky(args)


@app.command("cancel")
def cmd_cancel(
    job_ids: list[int] = typer.Argument(None, help="Managed-job IDs"),
    all_: bool = typer.Option(False, "--all", help="Cancel all your jobs"),
    yes: bool = typer.Option(False, "-y", help="Skip confirmation"),
):
    """Cancel managed job(s)."""
    args = ["jobs", "cancel"]
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
# ssh / dashboard / check
# ---------------------------------------------------------------------------

@app.command("ssh")
def cmd_ssh(
    pool: str = typer.Option(DEFAULT_POOL, "--pool", help="SSH node pool name (used as host alias)"),
):
    """Open an interactive shell on the GPU host."""
    subprocess.run(["ssh", "-t", pool])


@app.command("dashboard")
def cmd_dashboard():
    """Open the SkyPilot dashboard (jobs + clusters + infra)."""
    _run_sky(["dashboard"])


@app.command("check")
def cmd_check():
    """Show SkyPilot infra status (including SSH node pools)."""
    _run_sky(["check", "ssh"], check=False)
    _run_sky(["status"], check=False)


@app.command("pool-up")
def cmd_pool_up():
    """Bootstrap the configured SSH node pool (runs `sky ssh up`)."""
    _run_sky(["ssh", "up"])


# ---------------------------------------------------------------------------
# infer (unchanged local tool)
# ---------------------------------------------------------------------------

@app.command("infer", help="Run inference with a trained model on audio files.")
def cmd_infer(
    run: str = typer.Argument(..., help="Run ID or checkpoint dir"),
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
