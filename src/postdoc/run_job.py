from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path, PurePosixPath

from postdoc.experiment import build_train_args, build_eval_args
from postdoc.interfaces.storage import StorageBackend
from postdoc.interfaces.tracker import JobTracker, JobState


_METRIC_AVG_PATTERN = re.compile(r"Metric avg\s+([\w_]+)\s*:\s*([\d.eE+-]+)")
_INSTR_PATTERN = re.compile(r"Instr\s+\w+\s+([\w_]+):\s*([\d.eE+-]+)")


def extract_metrics(stdout: str) -> tuple[dict, bool]:
    metrics = {}
    for match in _METRIC_AVG_PATTERN.finditer(stdout):
        name = match.group(1).strip()
        metrics[name] = float(match.group(2))
    if not metrics:
        for match in _INSTR_PATTERN.finditer(stdout):
            name = match.group(1).strip()
            metrics[name] = float(match.group(2))
    incomplete = len(metrics) == 0
    return metrics, incomplete


def find_best_checkpoint(checkpoint_dir: Path) -> Path | None:
    if not checkpoint_dir.exists():
        return None
    ckpts = list(checkpoint_dir.glob("*.ckpt"))
    if not ckpts:
        return None
    best = [c for c in ckpts if "best" in c.name.lower()]
    if best:
        return best[0]
    ckpts.sort()
    return ckpts[-1]


def classify_error(stderr: str) -> str:
    stderr_lower = stderr.lower()
    if "cuda out of memory" in stderr_lower or "oom" in stderr_lower:
        return "OOM"
    if "nan" in stderr_lower and ("loss" in stderr_lower or "grad" in stderr_lower):
        return "NaN"
    if "filenotfounderror" in stderr_lower or "no such file" in stderr_lower:
        return "DataLoading"
    if "cuda" in stderr_lower or "cudnn" in stderr_lower:
        return "CUDA"
    return "Unknown"


def run_job(
    tracker: JobTracker,
    storage: StorageBackend,
    job_id: str,
    experiment: dict,
    resolved_config: Path,
    data_path: list[Path],
    valid_path: list[Path],
    device_ids: list[int],
    start_checkpoint: Path | None = None,
) -> None:
    results_root = Path(storage.job_root_path(job_id))
    eval_only = experiment.get("eval_only", False)
    eval_results = results_root / "eval"
    eval_results.mkdir(parents=True, exist_ok=True)

    if eval_only:
        if start_checkpoint is None:
            tracker.update_state(
                job_id, JobState.FAILED,
                error_category="NoCheckpoint",
                error_message="eval_only=true but no checkpoint provided",
            )
            return
        checkpoint = start_checkpoint
    else:
        # Training
        train_results = results_root / "training"
        train_results.mkdir(parents=True, exist_ok=True)

        tracker.update_state(job_id, JobState.TRAINING)
        train_args = build_train_args(
            experiment, resolved_config,
            results_path=train_results,
            data_path=data_path,
            valid_path=valid_path,
            device_ids=device_ids,
            start_checkpoint=start_checkpoint,
        )
        train_stdout = Path(storage.job_root_path(job_id)) / "training" / "logs" / "stdout.txt"
        train_stderr = Path(storage.job_root_path(job_id)) / "training" / "logs" / "stderr.txt"
        train_stdout.parent.mkdir(parents=True, exist_ok=True)
        with open(train_stdout, "w") as out_f, open(train_stderr, "w") as err_f:
            train_result = subprocess.run(
                [sys.executable, "train.py", *train_args],
                stdout=out_f, stderr=err_f,
            )

        if train_result.returncode != 0:
            stderr_text = train_stderr.read_text()
            tracker.update_state(
                job_id, JobState.FAILED,
                error_category=classify_error(stderr_text),
                error_message=stderr_text[-2000:] if stderr_text else "",
            )
            return

        checkpoint = find_best_checkpoint(train_results / "checkpoints")
        if checkpoint is None:
            tracker.update_state(
                job_id, JobState.FAILED,
                error_category="NoCheckpoint",
                error_message="No checkpoint found after training",
            )
            return

    # Eval (shared path for both modes)
    tracker.update_state(job_id, JobState.EVAL)
    eval_metrics = experiment.get("eval", {}).get("metrics")
    eval_args = build_eval_args(
        experiment, resolved_config,
        checkpoint_path=checkpoint,
        valid_path=valid_path,
        store_dir=eval_results / "samples",
        device_ids=device_ids,
        metrics=eval_metrics,
    )
    eval_stdout = eval_results / "logs" / "stdout.txt"
    eval_stderr = eval_results / "logs" / "stderr.txt"
    eval_stdout.parent.mkdir(parents=True, exist_ok=True)
    with open(eval_stdout, "w") as out_f, open(eval_stderr, "w") as err_f:
        eval_result = subprocess.run(
            [sys.executable, "final_valid.py", *eval_args],
            stdout=out_f, stderr=err_f,
        )

    if eval_result.returncode != 0:
        stderr_text = eval_stderr.read_text()
        tracker.update_state(
            job_id, JobState.FAILED,
            error_category=classify_error(stderr_text),
            error_message=stderr_text[-2000:] if stderr_text else "",
        )
        return

    # Metrics
    metrics, incomplete = extract_metrics(eval_stdout.read_text())
    tracker.set_metrics(job_id, metrics, incomplete)
    storage.put_json(job_id, PurePosixPath("eval/metrics.json"), metrics)

    # Done
    meta = {
        "job_id": job_id,
        "experiment_name": experiment.get("model", {}).get("type", "unknown"),
        "metrics_incomplete": incomplete,
    }
    storage.put_json(job_id, PurePosixPath("meta.json"), meta)
    tracker.update_state(job_id, JobState.DONE)


if __name__ == "__main__":
    import argparse
    import json as _json
    from postdoc.context import create_context

    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    cli_args = parser.parse_args()

    manifest = _json.loads(Path(cli_args.manifest).read_text())
    ctx = create_context(config_path=Path(manifest["postdoc_config"]))

    start_ckpt = manifest.get("start_checkpoint")

    try:
        run_job(
            tracker=ctx.tracker,
            storage=ctx.storage,
            job_id=manifest["job_id"],
            experiment=manifest["experiment"],
            resolved_config=Path(manifest["resolved_config"]),
            data_path=[Path(p) for p in manifest["data_path"]],
            valid_path=[Path(p) for p in manifest["valid_path"]],
            device_ids=manifest["device_ids"],
            start_checkpoint=Path(start_ckpt) if start_ckpt else None,
        )
    finally:
        ctx.scheduler._release_gpu(manifest["job_id"])
        ctx.scheduler.drain_queue(ctx.tracker)
        ctx.tracker.close()
