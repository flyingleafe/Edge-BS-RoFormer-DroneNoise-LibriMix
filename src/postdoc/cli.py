from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path, PurePosixPath

import typer

from postdoc.context import PostdocContext, create_context
from postdoc.experiment import load_experiment, resolve_config
from postdoc.interfaces.scheduler import NoCapacityError
from postdoc.interfaces.tracker import JobState

app = typer.Typer(name="postdoc", help="Experiment platform for ML research")
job_app = typer.Typer(help="Job management")
results_app = typer.Typer(help="Results management")
app.add_typer(job_app, name="job")
app.add_typer(results_app, name="results")

_backend_option = typer.Option(None, "--backend", help="Override backend (local/cloud)")


def _get_ctx(backend: str | None = None) -> PostdocContext:
    return create_context(config_path=Path("postdoc.yaml"), backend=backend)


def _get_git_info() -> tuple[str, str]:
    branch = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True,
    ).stdout.strip()
    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True, text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        capture_output=True, text=True,
    ).stdout.strip()
    if status:
        typer.echo(
            f"WARNING: Uncommitted changes detected. Job pinned to commit {commit} "
            f"but working tree is dirty. Consider committing first."
        )
    return branch, commit


@job_app.command("submit")
def job_submit(
    experiment_paths: list[Path] = typer.Argument(..., help="Experiment YAML file(s)"),
    backend: str | None = _backend_option,
):
    ctx = _get_ctx(backend)
    for exp_path in experiment_paths:
        exp = load_experiment(exp_path)
        exp_name = exp_path.stem
        git_branch, git_commit = _get_git_info()
        results_dir = Path(ctx.config.local.results_dir)
        job_id = ctx.tracker.create_job(exp_name, exp, git_branch, git_commit)
        job_dir = results_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        resolved = job_dir / "config.yaml"
        resolve_config(exp, resolved)

        # Resolve checkpoint for eval-only jobs
        start_checkpoint = exp.get("checkpoint")
        if start_checkpoint:
            start_checkpoint = str(Path(start_checkpoint).resolve())

        # Resolve dataset paths from the resolved config
        import yaml as _yaml
        with open(resolved) as _f:
            _resolved_cfg = _yaml.safe_load(_f)
        training_cfg = _resolved_cfg.get("training", {})
        data_paths = training_cfg.get("data_path", [])
        valid_paths = training_cfg.get("valid_path", data_paths)

        if not data_paths:
            typer.echo(f"ERROR: No training.data_path in resolved config for {exp_path}")
            raise typer.Exit(1)

        manifest = {
            "job_id": job_id,
            "postdoc_config": str(Path("postdoc.yaml").resolve()),
            "experiment": exp,
            "resolved_config": str(resolved),
            "data_path": data_paths,
            "valid_path": valid_paths,
            "device_ids": [0],
            "start_checkpoint": start_checkpoint,
        }
        manifest_path = job_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2))

        ctx.storage.put_json(job_id, PurePosixPath("experiment.yaml"), exp)

        try:
            result = ctx.scheduler.submit(job_id, resolved, exp)
            ctx.tracker.update_state(
                job_id, JobState.SUBMITTED,
                process_handle=result.process_handle,
                gpu_ids=result.gpu_ids,
            )
            typer.echo(f"Submitted job {job_id} ({exp_name}) on GPU {result.gpu_ids}")
        except NoCapacityError:
            ctx.tracker.update_state(job_id, JobState.QUEUED)
            typer.echo(f"Queued job {job_id} ({exp_name}) — no free GPUs")


@job_app.command("list")
def job_list(
    state: str | None = typer.Option(None, "--state", help="Filter by state"),
    backend: str | None = _backend_option,
):
    ctx = _get_ctx(backend)
    filter_state = JobState(state) if state else None
    jobs = ctx.tracker.list_jobs(state=filter_state)
    if not jobs:
        typer.echo("No jobs found.")
        return
    typer.echo(f"{'ID':<14} {'Name':<25} {'State':<12} {'Metrics'}")
    typer.echo("-" * 70)
    for j in jobs:
        metrics_str = ""
        if j.metrics:
            parts = [f"{k}={v:.3f}" for k, v in j.metrics.items()]
            metrics_str = ", ".join(parts)
        typer.echo(f"{j.job_id:<14} {j.experiment_name:<25} {j.state.value:<12} {metrics_str}")


@job_app.command("status")
def job_status(
    job_id: str = typer.Argument(..., help="Job ID"),
    backend: str | None = _backend_option,
):
    ctx = _get_ctx(backend)
    try:
        job = ctx.tracker.get_job(job_id)
    except KeyError:
        typer.echo(f"Job not found: {job_id}")
        raise typer.Exit(1)
    typer.echo(f"Job:        {job.job_id}")
    typer.echo(f"Experiment: {job.experiment_name}")
    typer.echo(f"State:      {job.state.value}")
    typer.echo(f"Branch:     {job.git_branch}")
    typer.echo(f"Commit:     {job.git_commit}")
    if job.process_handle:
        alive = ctx.scheduler.is_alive(job.process_handle)
        typer.echo(f"PID:        {job.process_handle} ({'alive' if alive else 'dead'})")
    if job.gpu_ids:
        typer.echo(f"GPU(s):     {job.gpu_ids}")
    if job.error_category:
        typer.echo(f"Error:      [{job.error_category}] {job.error_message}")
    if job.metrics:
        typer.echo("Metrics:")
        for k, v in job.metrics.items():
            typer.echo(f"  {k}: {v:.4f}")


@job_app.command("logs")
def job_logs(
    job_id: str = typer.Argument(..., help="Job ID"),
    tail: bool = typer.Option(False, "--tail", help="Show only last 50 lines"),
    backend: str | None = _backend_option,
):
    ctx = _get_ctx(backend)
    for phase in ["training", "eval"]:
        for stream in ["stderr", "stdout"]:
            log_path = PurePosixPath(f"{phase}/logs/{stream}.txt")
            if ctx.storage.exists(job_id, log_path):
                data = ctx.storage.get(job_id, log_path).decode(errors="replace")
                if data.strip():
                    typer.echo(f"--- {phase}/{stream} ---")
                    if tail:
                        lines = data.strip().split("\n")
                        typer.echo("\n".join(lines[-50:]))
                    else:
                        typer.echo(data)


@job_app.command("cancel")
def job_cancel(
    job_id: str = typer.Argument(..., help="Job ID"),
    backend: str | None = _backend_option,
):
    ctx = _get_ctx(backend)
    try:
        job = ctx.tracker.get_job(job_id)
    except KeyError:
        typer.echo(f"Job not found: {job_id}")
        raise typer.Exit(1)
    if job.process_handle:
        ctx.scheduler.cancel(job_id, job.process_handle)
    ctx.tracker.update_state(job_id, JobState.FAILED,
                             error_category="Cancelled", error_message="Cancelled by user")
    typer.echo(f"Cancelled job {job_id}")


@results_app.command("show")
def results_show(
    job_id: str = typer.Argument(..., help="Job ID"),
    backend: str | None = _backend_option,
):
    ctx = _get_ctx(backend)
    try:
        job = ctx.tracker.get_job(job_id)
    except KeyError:
        typer.echo(f"Job not found: {job_id}")
        raise typer.Exit(1)
    if not job.metrics:
        typer.echo(f"No metrics for job {job_id} (state: {job.state.value})")
        return
    typer.echo(f"Results for {job_id} ({job.experiment_name}):")
    for k, v in sorted(job.metrics.items()):
        typer.echo(f"  {k}: {v:.4f}")


@results_app.command("compare")
def results_compare(
    job_ids: list[str] = typer.Argument(..., help="Job IDs to compare"),
    backend: str | None = _backend_option,
):
    ctx = _get_ctx(backend)
    jobs = []
    for jid in job_ids:
        try:
            jobs.append(ctx.tracker.get_job(jid))
        except KeyError:
            typer.echo(f"Job not found: {jid}")
            raise typer.Exit(1)
    all_keys = set()
    for j in jobs:
        if j.metrics:
            all_keys.update(j.metrics.keys())
    all_keys = sorted(all_keys)
    if not all_keys:
        typer.echo("No metrics to compare.")
        return
    header = f"{'Metric':<12}" + "".join(f"{j.job_id:<14}" for j in jobs)
    typer.echo(header)
    typer.echo("-" * len(header))
    for key in all_keys:
        row = f"{key:<12}"
        for j in jobs:
            val = j.metrics.get(key) if j.metrics else None
            row += f"{val:<14.4f}" if val is not None else f"{'N/A':<14}"
        typer.echo(row)
