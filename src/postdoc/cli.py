from __future__ import annotations

import glob as _glob
import json
import re
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
    results_dir = Path(ctx.config.local.results_dir)
    for phase in ["training", "eval"]:
        for stream in ["stderr", "stdout"]:
            log_path = PurePosixPath(f"{phase}/logs/{stream}.txt")
            # Try reading directly from disk first (live logs), then storage
            disk_path = results_dir / job_id / phase / "logs" / f"{stream}.txt"
            if disk_path.exists():
                data = disk_path.read_text(errors="replace")
            elif ctx.storage.exists(job_id, log_path):
                data = ctx.storage.get(job_id, log_path).decode(errors="replace")
            else:
                continue
            if data.strip():
                typer.echo(f"--- {phase}/{stream} ---")
                if tail:
                    lines = data.strip().split("\n")
                    typer.echo("\n".join(lines[-50:]))
                else:
                    typer.echo(data)


@job_app.command("resume")
def job_resume(
    job_id: str = typer.Argument(..., help="Job ID to resume"),
    override: list[str] | None = typer.Option(None, "--set", help="Config overrides as key=value (dot notation)"),
    backend: str | None = _backend_option,
):
    """Resume a failed or cancelled job from its latest checkpoint."""
    import yaml as _yaml
    from postdoc.run_job import find_best_checkpoint

    ctx = _get_ctx(backend)
    try:
        job = ctx.tracker.get_job(job_id)
    except KeyError:
        typer.echo(f"Job not found: {job_id}")
        raise typer.Exit(1)

    if job.state not in (JobState.FAILED,):
        typer.echo(f"Cannot resume job in state '{job.state.value}'. Only failed jobs can be resumed.")
        raise typer.Exit(1)

    results_dir = Path(ctx.config.local.results_dir)
    train_dir = results_dir / job_id / "training"
    checkpoint = find_best_checkpoint(train_dir)
    if checkpoint is None:
        typer.echo(f"No checkpoint found in {train_dir}")
        raise typer.Exit(1)

    # Apply config overrides if any
    resolved = results_dir / job_id / "config.yaml"
    if override:
        from postdoc.experiment import _set_nested
        with open(resolved) as f:
            cfg = _yaml.safe_load(f)
        for item in override:
            key, _, value = item.partition("=")
            # Try to parse as number/bool
            try:
                value = int(value)
            except ValueError:
                try:
                    value = float(value)
                except ValueError:
                    if value.lower() in ("true", "false"):
                        value = value.lower() == "true"
            _set_nested(cfg, key.strip().split("."), value)
        with open(resolved, "w") as f:
            _yaml.dump(cfg, f, default_flow_style=False)
        typer.echo(f"Updated config: {', '.join(override)}")

    # Retrieve wandb_run_id from the job record (if previously stored)
    wandb_run_id = job.wandb_run_id

    # Update manifest for resume
    manifest_path = results_dir / job_id / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["start_checkpoint"] = str(checkpoint)
    manifest["is_resume"] = True
    if wandb_run_id:
        manifest["wandb_run_id"] = wandb_run_id
    manifest_path.write_text(json.dumps(manifest, indent=2))

    # Resubmit
    try:
        result = ctx.scheduler.submit(job_id, resolved, manifest["experiment"])
        ctx.tracker.update_state(
            job_id, JobState.SUBMITTED,
            process_handle=result.process_handle,
            gpu_ids=result.gpu_ids,
        )
        typer.echo(f"Resumed job {job_id} from {checkpoint.name} on GPU {result.gpu_ids}")
    except NoCapacityError:
        ctx.tracker.update_state(job_id, JobState.QUEUED)
        typer.echo(f"Queued job {job_id} for resume — no free GPUs")


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


# ---------------------------------------------------------------------------
# infer command
# ---------------------------------------------------------------------------

@app.command(
    "infer",
    help=(
        "Run inference with a trained model on audio files.\n"
        "\n"
        "  Usage:\n"
        "    postdoc infer [experiment id | model name] --checkpoint [epoch|filename|best|latest]\n"
        "              --input [directory | file | glob] --output [dir | file]\n"
        "\n"
        "  Examples:\n"
        "    postdoc infer a1b2c3d4 --checkpoint best --input audio.wav --output out/\n"
        "    postdoc infer dcunet --checkpoint 50 --input '*.wav' --output results/\n"
        "    postdoc infer edge_bs_rof --checkpoint latest --input dir/ --output out.wav"
    ),
)
def infer(
    experiment: str = typer.Argument(
        ...,
        help="Job ID or model name (fuzzy matched against job names in DB)",
    ),
    checkpoint: str = typer.Option(
        "best",
        "--checkpoint", "-c",
        help="Checkpoint: epoch number, filename, 'best', or 'latest'",
    ),
    input: str = typer.Option(
        ...,
        "--input", "-i",
        help="Input: directory, single file path, or glob pattern (quote it)",
    ),
    output: str = typer.Option(
        ...,
        "--output", "-o",
        help="Output directory or file (use with single input)",
    ),
    device: int = typer.Option(0, "--device", "-d", help="GPU device ID"),
    backend: str | None = _backend_option,
    rps_file: str | None = typer.Option(
        None,
        "--rps-file",
        help="Path to .npy file with RPS data (shape: [4, samples]), overrides audio input name",
    ),
):
    """
    Run inference with a trained model on audio files.

    Resolves EXPERIMENT by exact job ID first, then fuzzy matches against job names.
    Checkpoint: epoch number, filename, 'best' (default), or 'latest'.
    Input: directory, single file path, or glob pattern (quote it).
    Output: directory for multiple inputs, or output file for single input.
    Audio -> .wav, RPS predictions -> .npy.
    Output filename: {basename}_out.{ext}
    """
    import yaml as _yaml
    import numpy as np
    import torch
    import soundfile as sf
    from utils import (
        get_model_from_config,
        demix,
        load_start_checkpoint,
        read_audio_transposed,
    )
    from argparse import Namespace
    from pathlib import Path as _Path

    ctx = _get_ctx(backend)
    results_dir = _Path(ctx.config.local.results_dir)

    # ------------------------------------------------------------------
    # 1. Resolve experiment -> job record + resolved config + checkpoint
    # ------------------------------------------------------------------
    job = None
    # Try exact job ID match
    try:
        job = ctx.tracker.get_job(experiment)
    except KeyError:
        pass

    if job is None:
        # Fuzzy match: search job names containing the query (case-insensitive)
        query = experiment.lower()
        candidates = ctx.tracker.list_jobs(limit=200)
        matches = [
            j for j in candidates
            if query in j.experiment_name.lower() or query in j.job_id.lower()
        ]
        if not matches:
            typer.echo(f"No job found matching: {experiment}")
            raise typer.Exit(1)
        if len(matches) > 1:
            typer.echo(f"Multiple matches for '{experiment}':")
            for m in matches:
                typer.echo(f"  {m.job_id}  {m.experiment_name}  [{m.state.value}]")
            typer.echo("Use job ID for a unique match.")
            raise typer.Exit(1)
        job = matches[0]

    job_id = job.job_id
    typer.echo(f"Using job: {job_id} ({job.experiment_name})")

    # Load resolved config
    resolved_config_path = results_dir / job_id / "config.yaml"
    if not resolved_config_path.exists():
        typer.echo(f"Resolved config not found: {resolved_config_path}")
        raise typer.Exit(1)
    with open(resolved_config_path) as f:
        config = _yaml.safe_load(f)

    model_type = config.get("model", {}).get("model") or config.get("model", {}).get("type")
    if not model_type:
        # Fallback: look in experiment config_snapshot
        model_type = job.config_snapshot.get("model", {}).get("type")
    if not model_type:
        typer.echo("Could not determine model type from config.")
        raise typer.Exit(1)

    # ------------------------------------------------------------------
    # 2. Resolve checkpoint
    # ------------------------------------------------------------------
    train_dir = results_dir / job_id / "training"
    if not train_dir.exists():
        typer.echo(f"Training directory not found: {train_dir}")
        raise typer.Exit(1)

    ckpt_files = list(train_dir.glob("*.ckpt"))
    if not ckpt_files:
        typer.echo(f"No checkpoints found in {train_dir}")
        raise typer.Exit(1)

    # Parse checkpoint selector
    ckpt_path: _Path | None = None
    ckpt_lc = checkpoint.lower()
    if ckpt_lc == "best":
        candidates_ = [c for c in ckpt_files if "best" in c.name.lower()]
        if candidates_:
            ckpt_path = candidates_[0]
        else:
            typer.echo("No 'best' checkpoint found, using latest.")
            ckpt_lc = "latest"
    if ckpt_lc == "latest":
        ckpt_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        ckpt_path = ckpt_files[0]
    else:
        # Try as epoch number
        epoch_match = re.match(r"^(\d+)$", checkpoint)
        if epoch_match:
            epoch_num = int(epoch_match.group(1))
            candidates_ = [
                c for c in ckpt_files
                if re.search(rf"epoch[_-]?{epoch_num}[^\d]", c.name, re.IGNORECASE)
                or re.search(rf"e[_-]?{epoch_num}[^\d]", c.name, re.IGNORECASE)
            ]
            if candidates_:
                ckpt_path = candidates_[0]
            else:
                # Try matching just the number anywhere in name
                for c in ckpt_files:
                    if str(epoch_num) in c.stem:
                        ckpt_path = c
                        break
        else:
            # Treat as exact filename (or partial)
            for c in ckpt_files:
                if checkpoint in c.name:
                    ckpt_path = c
                    break

    if ckpt_path is None:
        typer.echo(f"Checkpoint not found: {checkpoint}")
        typer.echo("Available: " + ", ".join(c.name for c in ckpt_files))
        raise typer.Exit(1)

    typer.echo(f"Using checkpoint: {ckpt_path.name}")

    # ------------------------------------------------------------------
    # 3. Resolve input files
    # ------------------------------------------------------------------
    input_path = _Path(input)
    if input_path.is_dir():
        audio_files = sorted(_glob.glob(str(input_path / "*.wav"))) + sorted(_glob.glob(str(input_path / "*.flac")))
        if not audio_files:
            typer.echo(f"No .wav or .flac files found in {input_path}")
            raise typer.Exit(1)
    elif _glob.has_magic(input):
        audio_files = sorted(_glob.glob(input))
        if not audio_files:
            typer.echo(f"No files match glob pattern: {input}")
            raise typer.Exit(1)
    elif input_path.is_file():
        audio_files = [str(input_path)]
    else:
        typer.echo(f"Input not found: {input}")
        raise typer.Exit(1)

    typer.echo(f"Processing {len(audio_files)} file(s)...")

    # ------------------------------------------------------------------
    # 4. Resolve output
    # ------------------------------------------------------------------
    output_path = _Path(output)
    if len(audio_files) > 1 and output_path.is_file():
        typer.echo("Multiple input files but output is a file. Using output as directory.")
        output_path = output_path.parent
    if output_path.is_file() and len(audio_files) > 1:
        typer.echo("Multiple input files but output is a file. Will write to parent directory.")
        output_path = output_path.parent
    output_path.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 5. Load model
    # ------------------------------------------------------------------
    base_config = results_dir / job_id / "config.yaml"
    device_obj = torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")
    typer.echo(f"Device: {device_obj}")

    model, cfg = get_model_from_config(model_type, str(base_config))
    model = model.to(device_obj)

    # Build fake args for load_start_checkpoint
    fake_args = Namespace(
        start_check_point=str(ckpt_path),
        model_type=model_type,
        lora_checkpoint=None,
    )
    load_start_checkpoint(fake_args, model, type_="infer")
    model.eval()

    use_rps = config.get("use_rps", False)
    predict_rps = config.get("predict_rps", False)
    typer.echo(f"Model: {model_type}  |  RPS conditioning: {use_rps}  |  RPS prediction head: {predict_rps}")

    # ------------------------------------------------------------------
    # 6. Run inference
    # ------------------------------------------------------------------
    from ml_collections import ConfigDict
    cfg_obj = cfg if isinstance(cfg, ConfigDict) else ConfigDict(cfg)

    for audio_path_str in audio_files:
        audio_path = _Path(audio_path_str)
        stem = audio_path.stem

        # Determine RPS path
        rps_path: _Path | None = None
        if rps_file:
            rps_path = _Path(rps_file)
        elif use_rps or predict_rps:
            # Try same stem with .npy
            candidate = audio_path.with_suffix(".npy")
            if candidate.exists():
                rps_path = candidate
            else:
                typer.echo(f"  [WARN] RPS file not found for {stem}, skipping RPS.")

        # Load RPS if needed
        rps_data = None
        if rps_path and rps_path.exists():
            rps_data = np.load(str(rps_path))
            typer.echo(f"  Loaded RPS: {rps_path.name} shape={rps_data.shape}")

        # Load audio
        mix, sr = read_audio_transposed(str(audio_path))
        mix_tensor = torch.from_numpy(mix).float()

        typer.echo(f"  Processing: {audio_path.name}  ({mix.shape[-1]/sr:.2f}s)")

        with torch.no_grad(), torch.cuda.amp.autocast(enabled=getattr(cfg_obj.training, "use_amp", True)):
            result = demix(
                cfg_obj,
                model,
                mix_tensor,
                device_obj,
                model_type=model_type,
                rps=rps_data,
            )

        # result: dict[str, np.ndarray] or np.ndarray (demucs single instrument)
        if isinstance(result, dict):
            for instr, audio in result.items():
                out_wav = output_path / f"{stem}_out_{instr}.wav"
                sf.write(str(out_wav), audio.T, sr)
                typer.echo(f"    -> {out_wav.name}")
        else:
            out_wav = output_path / f"{stem}_out.wav"
            sf.write(str(out_wav), result.T, sr)
            typer.echo(f"    -> {out_wav.name}")

        # Save RPS data if provided (for models with RPS conditioning)
        if rps_data is not None:
            out_rps = output_path / f"{stem}_out_rps.npy"
            np.save(str(out_rps), rps_data)
            typer.echo(f"    -> {out_rps.name}")

    typer.echo(f"\nDone. Outputs in: {output_path}")
