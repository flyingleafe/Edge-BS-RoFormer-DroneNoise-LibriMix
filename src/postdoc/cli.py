"""postdoc CLI.

Two backends:
  - direct:  plain SSH to vast-server; jobs share the host's repo + venv
  - cloud:   SkyPilot managed jobs (sky jobs launch); fresh container per job

Routing:
  postdoc submit <cmd>      tries direct, falls back to cloud
  postdoc submit --cloud    forces cloud only
  postdoc submit --direct   forces direct only (error if unavailable)

Daemon (tmux, on vast-server):
  postdoc queue-start       start the queue watcher in a tmux session
  postdoc queue-stop        kill the tmux session
  postdoc queue-status      is it running?

Job management:
  postdoc list              list all jobs (direct + cloud)
  postdoc logs <id>         tail logs (follows by default)
  postdoc cancel <id>       kill the job process and mark cancelled
  postdoc status <id>       one-line summary for a job

Utilities:
  postdoc ssh               interactive SSH to vast-server
  postdoc infer             local inference (unchanged)
"""
from __future__ import annotations

import os
import shlex
import subprocess
import sys
import webbrowser
from datetime import datetime
from pathlib import Path

import typer
import yaml

from postdoc import git_state, infer as infer_mod
from postdoc import direct
from postdoc import cloud as cloud_mod

# ------------------------------------------------------------------ #
# constants
# ------------------------------------------------------------------ #

DEFAULT_POSTDOC_HOST = os.environ.get("POSTDOC_HOST", "vast-server")
DEFAULT_POSTDOC_USER = os.environ.get("POSTDOC_USER", "root")
CLUSTER_NAME = os.environ.get("POSTDOC_CLUSTER", "postdoc")
DEFAULT_JOB_GPUS = int(os.environ.get("POSTDOC_DEFAULT_GPUS", "1"))

app = typer.Typer(
    name="postdoc",
    help="Submit shell commands as jobs on vast-server (direct SSH) or cloud "
         "(SkyPilot managed jobs).",
    no_args_is_help=True,
    context_settings={"help_option_names": ["-h", "--help"]},
)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

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


def _run_sky(args: list[str], *, check: bool = True,
             capture: bool = False) -> subprocess.CompletedProcess:
    cmd = ["sky", *args]
    if capture:
        return subprocess.run(cmd, check=check, text=True,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return subprocess.run(cmd, check=check)


def _format_jobs(jobs: list) -> str:
    lines = [f"{'ID':<6} {'Name':<30} {'SHA':<12} {'GPUs':<5} "
             f"{'Status':<12} {'GPU Mask':<12}"]
    lines.append("-" * 80)
    for j in jobs:
        lines.append(
            f"{j.id:<6} {j.name:<30} {j.sha[:12]:<12} {j.gpus:<5} "
            f"{j.status:<12} {str(j.gpu_mask or ''):<12}"
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# queue daemon
# --------------------------------------------------------------------------- #

@app.command("queue-start")
def cmd_queue_start(
    host: str = typer.Option(DEFAULT_POSTDOC_HOST, "--host"),
    user: str = typer.Option(DEFAULT_POSTDOC_USER, "--user"),
):
    """Start the postdoc queue watcher in a tmux session on vast-server.

    The watcher:
    - Reads job descriptors from /root/.postdoc/queue.fifo
    - Allocates GPUs to jobs as they free up
    - Runs each job via nohup postdoc-job
    """
    # Check if already running
    r = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", f"{user}@{host}",
         "tmux has-session -t postdoc-queue 2>/dev/null && echo running || echo stopped"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    if "running" in r.stdout:
        typer.echo("[postdoc] queue already running in tmux postdoc-queue")
        return

    # Start the runner in a detached tmux session
    typer.echo("[postdoc] starting queue watcher in tmux postdoc-queue...")
    subprocess.run(
        ["ssh", "-o", "BatchMode=yes", f"{user}@{host}",
         "tmux new-session -d -s postdoc-queue "
         "'source ~/.bashrc 2>/dev/null; "
         "cd ~/harmonic-noise-suppression && "
         "/root/harmonic-noise-suppression/.venv/bin/postdoc-runner'"],
        check=True,
    )

    # Verify it started
    r2 = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", f"{user}@{host}",
         "tmux has-session -t postdoc-queue 2>/dev/null && echo ok || echo fail"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    if "ok" in r2.stdout:
        typer.echo("[postdoc] queue started")
    else:
        typer.echo("[postdoc] WARNING: could not verify queue started. "
                    "Check manually with `postdoc queue-status`.")


@app.command("queue-stop")
def cmd_queue_stop(
    host: str = typer.Option(DEFAULT_POSTDOC_HOST, "--host"),
    user: str = typer.Option(DEFAULT_POSTDOC_USER, "--user"),
):
    """Kill the postdoc queue watcher tmux session."""
    subprocess.run(
        ["ssh", "-o", "BatchMode=yes", f"{user}@{host}",
         "tmux kill-session -t postdoc-queue 2>/dev/null; echo done"],
        check=False,
    )
    typer.echo("[postdoc] queue stopped")


@app.command("queue-status")
def cmd_queue_status(
    host: str = typer.Option(DEFAULT_POSTDOC_HOST, "--host"),
    user: str = typer.Option(DEFAULT_POSTDOC_USER, "--user"),
):
    """Show whether the queue watcher is running."""
    r = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", f"{user}@{host}",
         "tmux has-session -t postdoc-queue 2>/dev/null && echo running || echo stopped"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    status = "running" if "running" in r.stdout else "stopped"
    typer.echo(f"[postdoc] queue: {status}")


# --------------------------------------------------------------------------- #
# submit
# --------------------------------------------------------------------------- #

@app.command("submit", context_settings={"allow_extra_args": True,
                                        "ignore_unknown_options": True})
def cmd_submit(
    ctx: typer.Context,
    name: str | None = typer.Option(None, "--name", "-n"),
    gpus: int = typer.Option(DEFAULT_JOB_GPUS, "--gpus", "-g"),
    remote: str = typer.Option("origin", "--remote"),
    allow_dirty: bool = typer.Option(False, "--dirty"),
    skip_push: bool = typer.Option(False, "--skip-push"),
    env: list[str] | None = typer.Option(None, "--env", "-e"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    direct: bool = typer.Option(False, "--direct",
                                help="Force direct SSH to vast-server"),
    cloud_backend: bool = typer.Option(False, "--cloud",
                               help="Force SkyPilot cloud backend"),
    host: str = typer.Option(DEFAULT_POSTDOC_HOST, "--host"),
    user: str = typer.Option(DEFAULT_POSTDOC_USER, "--user"),
):
    """Submit a shell command as a job.

    Routes to direct SSH (vast-server) or SkyPilot cloud based on GPU
    availability and flags. Git preflight (push HEAD to origin) always runs.

    Direct backend:
      Probes vast-server for free GPUs. If enough are free, launches
      immediately via SSH. Otherwise, appends the job to the queue FIFO
      on vast-server; the postdoc queue daemon picks it up when GPUs free.

    Cloud backend:
      Uses `sky jobs launch` — fresh container per job, no shared state.
    """
    command_tokens = ctx.args
    if not command_tokens:
        typer.echo("ERROR: no command given. Usage: postdoc submit <command...>",
                   err=True)
        raise typer.Exit(2)

    if cloud_backend and direct:
        typer.echo("ERROR: --direct and --cloud are mutually exclusive", err=True)
        raise typer.Exit(2)

    command = shlex.join(command_tokens)

    # Parse env
    env_dict: dict[str, str] = {}
    for kv in env or []:
        if "=" not in kv:
            typer.echo(f"ERROR: --env expects KEY=VALUE, got {kv!r}", err=True)
            raise typer.Exit(2)
        k, v = kv.split("=", 1)
        env_dict[k] = v

    # Git preflight
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
        typer.echo("WARNING: --dirty set; uncommitted changes are NOT on the remote.",
                   err=True)

    # Determine backend before dry-run (backend is used in dry-run output)
    if cloud_backend:
        backend = "cloud"
    elif direct:
        backend = "direct"
    else:
        try:
            available = direct.free_gpus(user=user, host=host)
            backend = "direct" if len(available) >= gpus else "cloud"
        except Exception:
            backend = "cloud"

    if dry_run:
        typer.echo(f"[postdoc] dry-run  backend={backend}  "
                    f"sha={snap['sha'][:12]}  cmd={command[:80]}")
        return

    job_name = name or _auto_name(command)
    job_id: int | None = None
    status: str = ""

    if backend == "direct":
        try:
            job_id, status = direct.submit_direct(
                name=job_name,
                sha=snap["sha"],
                cmd=command,
                gpus=gpus,
                user=user,
                host=host,
            )
        except Exception as e:
            typer.echo(f"[postdoc] direct submit failed: {e}", err=True)
            typer.echo("[postdoc] falling back to cloud...", err=True)
            backend = "cloud"

    if backend == "cloud":
        try:
            job_id, status = cloud_mod.submit_cloud(
                name=job_name,
                sha=snap["sha"],
                url=snap["url"],
                cmd=command,
                gpus=gpus,
                envs=env_dict or None,
                dry_run=dry_run,
            )
        except Exception as e:
            typer.echo(f"[postdoc] cloud submit failed: {e}", err=True)
            raise typer.Exit(4)

    # Print result
    suffix = ""
    if job_id is not None:
        suffix = f"  job={job_name}__{job_id}"
    typer.echo(f"[postdoc] {backend}  status={status}{suffix}")


# --------------------------------------------------------------------------- #
# list / status / logs / cancel
# --------------------------------------------------------------------------- #

@app.command("list")
def cmd_list(
    all_: bool = typer.Option(False, "--all", "-a",
                              help="Include finished jobs"),
    host: str = typer.Option(DEFAULT_POSTDOC_HOST, "--host"),
    user: str = typer.Option(DEFAULT_POSTDOC_USER, "--user"),
):
    """List all jobs (direct backend from vast-server)."""
    jobs = direct.list_jobs(user=user, host=host)
    if not jobs:
        typer.echo("No jobs found.")
        return
    if all_:
        visible = jobs
    else:
        visible = [j for j in jobs if j.status in ("running", "queued")]
    if not visible:
        typer.echo("No active jobs.")
        return
    typer.echo(_format_jobs(visible))


@app.command("status")
def cmd_status(
    name_and_id: str = typer.Argument(...,
                                      help="<name>__<id>, e.g. dccrn__42"),
    host: str = typer.Option(DEFAULT_POSTDOC_HOST, "--host"),
    user: str = typer.Option(DEFAULT_POSTDOC_USER, "--user"),
):
    """Show one-line status for a specific job."""
    job_dir = f"/root/.postdoc/jobs/{name_and_id}"
    r = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", f"{user}@{host}",
         f"cat {job_dir}/job.json 2>/dev/null || echo null"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    import json
    try:
        d = json.loads(r.stdout)
    except Exception:
        typer.echo(f"Job not found: {name_and_id}", err=True)
        raise typer.Exit(1)
    typer.echo(
        f"  name: {d['name']}\n"
        f"  id:   {d['id']}\n"
        f"  sha:  {d['sha'][:12]}\n"
        f"  cmd:  {d['cmd'][:80]}\n"
        f"  gpus: {d['gpus']}\n"
        f"  status: {d['status']}\n"
        f"  pid:  {d.get('pid', 'N/A')}\n"
        f"  gpu_mask: {d.get('gpu_mask', 'N/A')}"
    )


@app.command("logs")
def cmd_logs(
    name_and_id: str = typer.Argument(...,
                                      help="<name>__<id>, e.g. dccrn__42"),
    follow: bool = typer.Option(True, "--follow/--no-follow", "-f"),
    lines: int = typer.Option(50, "--lines", "-n"),
    host: str = typer.Option(DEFAULT_POSTDOC_HOST, "--host"),
    user: str = typer.Option(DEFAULT_POSTDOC_USER, "--user"),
):
    """Tail (or cat) the log file for a job."""
    job_dir = f"/root/.postdoc/jobs/{name_and_id}"
    if follow:
        cmd = ["ssh", "-o", "BatchMode=yes", f"{user}@{host}",
               f"tail -F {job_dir}/log.txt"]
        subprocess.run(cmd)
    else:
        r = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", f"{user}@{host}",
             f"tail -{lines} {job_dir}/log.txt"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        typer.echo(r.stdout or "")


@app.command("cancel")
def cmd_cancel(
    name_and_id: str = typer.Argument(...,
                                      help="<name>__<id>, e.g. dccrn__42"),
    host: str = typer.Option(DEFAULT_POSTDOC_HOST, "--host"),
    user: str = typer.Option(DEFAULT_POSTDOC_USER, "--user"),
):
    """Kill the job process and mark it cancelled."""
    ok = direct.cancel_job(name_and_id, user=user, host=host)
    if ok:
        typer.echo(f"[postdoc] cancelled {name_and_id}")
    else:
        typer.echo(f"[postdoc] could not cancel {name_and_id} (not running?)")


# --------------------------------------------------------------------------- #
# utility
# --------------------------------------------------------------------------- #

@app.command("ssh")
def cmd_ssh(
    host: str = typer.Option(DEFAULT_POSTDOC_HOST, "--host"),
    user: str = typer.Option(DEFAULT_POSTDOC_USER, "--user"),
):
    """Interactive SSH to vast-server."""
    subprocess.run(["ssh", "-t", f"{user}@{host}"])


@app.command("check")
def cmd_check(
    host: str = typer.Option(DEFAULT_POSTDOC_HOST, "--host"),
    user: str = typer.Option(DEFAULT_POSTDOC_USER, "--user"),
):
    """Check GPU availability on vast-server."""
    typer.echo(f"[postdoc] probing GPUs on {user}@{host}...")
    try:
        gpus = direct.probe_gpus(user=user, host=host)
        free = [g for g in gpus if g.memory_used_mib < 500]
        typer.echo(f"[postdoc] {len(gpus)} GPUs total, {len(free)} free:")
        for g in gpus:
            flag = " [FREE]" if g.memory_used_mib < 500 else ""
            typer.echo(
                f"  GPU {g.index}: {g.memory_used_mib}/{g.memory_total_mib} MiB  "
                f"util={g.utilization}%{flag}"
            )
    except Exception as e:
        typer.echo(f"[postdoc] ERROR: {e}", err=True)
        raise typer.Exit(1)


@app.command("probe")
def cmd_probe(
    host: str = typer.Option(DEFAULT_POSTDOC_HOST, "--host"),
    user: str = typer.Option(DEFAULT_POSTDOC_USER, "--user"),
):
    """Alias for `postdoc check`."""
    cmd_check(host=host, user=user)


# --------------------------------------------------------------------------- #
# infer (local, unchanged)
# --------------------------------------------------------------------------- #

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


# --------------------------------------------------------------------------- #
# backwards-compat stubs for old SkyPilot commands (removed / relocated)
# --------------------------------------------------------------------------- #

@app.command("pool-up", hidden=True)
def cmd_pool_up():
    typer.echo("[postdoc] pool-up is no longer needed. "
               "The direct SSH backend uses plain SSH — no k3s required.", err=True)
    raise typer.Exit(1)


@app.command("pool-down", hidden=True)
def cmd_pool_down():
    typer.echo("[postdoc] pool-down is no longer needed.", err=True)


@app.command("cluster-up", hidden=True)
def cmd_cluster_up():
    typer.echo("[postdoc] cluster-up is replaced by `postdoc queue-start`. "
               "Direct SSH jobs don't need a persistent cluster.", err=True)
    typer.echo("[postdoc] Run `postdoc queue-start` to start the queue daemon.", err=True)
    raise typer.Exit(1)


@app.command("cluster-down", hidden=True)
def cmd_cluster_down():
    typer.echo("[postdoc] cluster-down: teardown is no longer needed for direct SSH.",
               err=True)


@app.command("cluster-status", hidden=True)
def cmd_cluster_status():
    cmd_check()


@app.command("dashboard", hidden=True)
def cmd_dashboard():
    typer.echo("[postdoc] dashboard: Ray dashboard is gone (Ray cluster removed). "
               "Use `postdoc ssh` to log in to vast-server directly.", err=True)
    raise typer.Exit(1)


@app.command("queue", hidden=True)
def cmd_queue():
    typer.echo("[postdoc] queue command removed. Use:",
               err=True)
    typer.echo("  postdoc queue-start   # start the daemon", err=True)
    typer.echo("  postdoc queue-status  # check status", err=True)
    typer.echo("  postdoc queue-stop    # stop the daemon", err=True)
    raise typer.Exit(1)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
