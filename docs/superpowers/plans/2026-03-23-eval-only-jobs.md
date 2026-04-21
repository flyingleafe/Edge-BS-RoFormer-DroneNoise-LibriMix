# Eval-Only Jobs with ESTOI, SI-SDR, PESQ Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Evaluate DCUNet baseline and DCUNet+RPS checkpoints on DREGON-LM valid split using ESTOI, SI-SDR, and PESQ metrics via the postdoc job system.

**Architecture:** Add eval-only mode to `run_job.py` (skip training, use provided checkpoint), add ESTOI metric to `final_valid.py` by extending existing `calculate_stoi` with an `extended` parameter (leveraging `pystoi`), pass `--metrics` through `build_eval_args`, and create two experiment YAMLs with a new `eval_only` flag.

**Tech Stack:** Python, pystoi (already installed v0.4.1), pesq, postdoc job system, uv

---

### Task 1: Add ESTOI metric to `final_valid.py`

**Files:**
- Modify: `final_valid.py:77-118` (refactor `calculate_stoi` to support `extended` param)
- Modify: `final_valid.py:810-812` (add `estoi` to `--metrics` choices)
- Modify: `final_valid.py:296-301` (init `estoi` in `all_metrics` when requested)
- Modify: `final_valid.py:400-407` (compute and collect `estoi`)
- Modify: `final_valid.py:413-418` (add `estoi` to results row)

- [ ] **Step 1: Refactor `calculate_stoi` to accept `extended` parameter**

Replace the existing `calculate_stoi` function (lines 77-118) to support both STOI and ESTOI:

```python
def calculate_stoi(ref, est, orig_sr, extended=False):
    """
    Calculate Short-Time Objective Intelligibility (STOI or ESTOI) metric.

    Parameters:
    ----------
    ref : numpy.ndarray
        Reference audio signal
    est : numpy.ndarray
        Estimated audio signal
    orig_sr : int
        Original sample rate
    extended : bool
        If True, compute ESTOI instead of STOI

    Returns:
    -------
    float
        STOI/ESTOI score, range [0, 1], higher is better
    """
    # If 2D array, convert to mono
    if ref.ndim == 2:
        if ref.shape[0] > 1:
            ref = librosa.to_mono(ref)
        else:
            ref = ref.squeeze(0)
    if est.ndim == 2:
        if est.shape[0] > 1:
            est = librosa.to_mono(est)
        else:
            est = est.squeeze(0)

    # STOI requires 10000Hz sample rate, need to resample
    target_sr = 10000
    ref = librosa.resample(ref, orig_sr=orig_sr, target_sr=target_sr)
    est = librosa.resample(est, orig_sr=orig_sr, target_sr=target_sr)

    try:
        score = stoi(ref, est, target_sr, extended=extended)
    except Exception as e:
        metric_name = "ESTOI" if extended else "STOI"
        print(f"[DEBUG] {metric_name} calculation failed: {e}")
        score = np.nan
    return score
```

- [ ] **Step 2: Add `estoi` to CLI choices**

In `parse_args`, update the `--metrics` choices list (line ~811) to include `'estoi'`:

```python
choices=['sdr', 'l1_freq', 'si_sdr', 'neg_log_wmse', 'aura_stft', 'aura_mrstft', 'bleedless',
         'fullness', 'pesq', 'stoi', 'estoi']
```

- [ ] **Step 3: Wire ESTOI into the evaluation loop**

In the evaluation loop, add ESTOI collection alongside STOI:

1. In `all_metrics` init (~line 299-301), also init `estoi` if requested:
```python
if 'estoi' not in all_metrics and 'estoi' in args.metrics:
    all_metrics['estoi'] = {instr: [] for instr in config.training.instruments}
```

2. In the per-track metric computation (~line 404-407), add after the existing STOI block:
```python
if 'estoi' in args.metrics:
    estoi_score = calculate_stoi(track, estimates, sr, extended=True)
    track_metrics['estoi'] = estoi_score
    all_metrics['estoi'][instr].append(estoi_score)
```

3. In the results row dict (~line 413-418), add:
```python
'estoi': track_metrics.get('estoi', None),
```

- [ ] **Step 4: Verify ESTOI works**

Run: `uv run python -c "from pystoi import stoi; import numpy as np; x=np.random.randn(16000); print(stoi(x, x+0.01*np.random.randn(16000), 16000, extended=True))"`

Expected: A float close to 1.0 (identical signals with tiny noise).

- [ ] **Step 5: Commit**

```bash
git add final_valid.py
git commit -m "feat: add ESTOI metric to final_valid.py evaluation"
```

---

### Task 2: Pass `--metrics` through `build_eval_args`

**Files:**
- Modify: `src/postdoc/experiment.py:53-68` (add `metrics` parameter to `build_eval_args`)

- [ ] **Step 1: Update `build_eval_args` signature**

In `src/postdoc/experiment.py`, add `metrics` parameter to `build_eval_args`:

```python
def build_eval_args(
    experiment: dict,
    resolved_config: Path,
    checkpoint_path: Path,
    valid_path: list[Path],
    store_dir: Path,
    device_ids: list[int],
    metrics: list[str] | None = None,
) -> list[str]:
    args = [
        "--model_type", experiment["model"]["type"],
        "--config_path", str(resolved_config),
        "--start_check_point", str(checkpoint_path),
        "--valid_path", *[str(p) for p in valid_path],
        "--store_dir", str(store_dir),
        "--device_ids", *[str(d) for d in device_ids],
    ]
    if metrics:
        args.extend(["--metrics", *metrics])
    return args
```

- [ ] **Step 2: Commit**

```bash
git add src/postdoc/experiment.py
git commit -m "feat: add metrics parameter to build_eval_args"
```

---

### Task 3: Add eval-only mode to `run_job.py` and `cli.py`

**Files:**
- Modify: `src/postdoc/run_job.py:56-143` (restructure to support eval-only)
- Modify: `src/postdoc/run_job.py:146-172` (pass `start_checkpoint` in `__main__`)
- Modify: `src/postdoc/cli.py:49-103` (resolve checkpoint path and add to manifest)

- [ ] **Step 1: Rewrite `run_job` function body to support eval-only**

Replace the `run_job` function body in `src/postdoc/run_job.py` (lines 56-143):

```python
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
        train_result = subprocess.run(
            [sys.executable, "train.py", *train_args],
            capture_output=True, text=True,
        )
        storage.put(job_id, PurePosixPath("training/logs/stdout.txt"), train_result.stdout.encode())
        storage.put(job_id, PurePosixPath("training/logs/stderr.txt"), train_result.stderr.encode())

        if train_result.returncode != 0:
            tracker.update_state(
                job_id, JobState.FAILED,
                error_category=classify_error(train_result.stderr),
                error_message=train_result.stderr[-2000:] if train_result.stderr else "",
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
    eval_result = subprocess.run(
        [sys.executable, "final_valid.py", *eval_args],
        capture_output=True, text=True,
    )
    storage.put(job_id, PurePosixPath("eval/logs/stdout.txt"), eval_result.stdout.encode())
    storage.put(job_id, PurePosixPath("eval/logs/stderr.txt"), eval_result.stderr.encode())

    if eval_result.returncode != 0:
        tracker.update_state(
            job_id, JobState.FAILED,
            error_category=classify_error(eval_result.stderr),
            error_message=eval_result.stderr[-2000:] if eval_result.stderr else "",
        )
        return

    # Metrics
    metrics, incomplete = extract_metrics(eval_result.stdout)
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
```

- [ ] **Step 2: Update `__main__` block in `run_job.py` to pass `start_checkpoint`**

In `src/postdoc/run_job.py`, update the `__main__` block (lines 146-172) to read and pass `start_checkpoint` from the manifest:

```python
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
```

- [ ] **Step 3: Update `job_submit` in `cli.py` to resolve checkpoint and add to manifest**

In `src/postdoc/cli.py`, in the `job_submit` function, after `resolve_config(exp, resolved)` (line 64), add checkpoint resolution:

```python
        # Resolve checkpoint for eval-only jobs
        start_checkpoint = exp.get("checkpoint")
        if start_checkpoint:
            start_checkpoint = str(Path(start_checkpoint).resolve())
```

Then in the manifest dict (lines 79-87), add the checkpoint:

```python
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
```

- [ ] **Step 4: Commit**

```bash
git add src/postdoc/run_job.py src/postdoc/cli.py
git commit -m "feat: add eval-only mode to job pipeline"
```

---

### Task 4: Update tests for new functionality

**Files:**
- Modify: `tests/test_run_job.py` (add tests for eval-only, metrics passthrough, and ESTOI extraction)

- [ ] **Step 1: Add test for ESTOI metric extraction**

```python
def test_extract_metrics_with_estoi():
    stdout = "Metric avg estoi       : 0.7890\nMetric avg si_sdr      : 6.5\n"
    metrics, incomplete = extract_metrics(stdout)
    assert metrics["estoi"] == pytest.approx(0.789)
    assert not incomplete
```

- [ ] **Step 2: Add test for `build_eval_args` with metrics**

```python
def test_build_eval_args_with_metrics():
    from postdoc.experiment import build_eval_args
    args = build_eval_args(
        experiment={"model": {"type": "dcunet"}},
        resolved_config=Path("/tmp/config.yaml"),
        checkpoint_path=Path("/tmp/best.ckpt"),
        valid_path=[Path("/tmp/valid")],
        store_dir=Path("/tmp/eval"),
        device_ids=[0],
        metrics=["estoi", "si_sdr", "pesq"],
    )
    assert "--metrics" in args
    idx = args.index("--metrics")
    assert args[idx + 1:idx + 4] == ["estoi", "si_sdr", "pesq"]


def test_build_eval_args_without_metrics():
    from postdoc.experiment import build_eval_args
    args = build_eval_args(
        experiment={"model": {"type": "dcunet"}},
        resolved_config=Path("/tmp/config.yaml"),
        checkpoint_path=Path("/tmp/best.ckpt"),
        valid_path=[Path("/tmp/valid")],
        store_dir=Path("/tmp/eval"),
        device_ids=[0],
    )
    assert "--metrics" not in args
```

- [ ] **Step 3: Add test for eval-only job success**

```python
@patch("postdoc.run_job.subprocess.run")
def test_run_job_eval_only(mock_run, tracker, storage, job_setup, tmp_results_dir):
    job_id, exp, resolved = job_setup
    exp["eval_only"] = True
    exp["eval"] = {"metrics": ["estoi", "si_sdr", "pesq"]}

    # Create a fake checkpoint
    ckpt = tmp_results_dir / "fake_best.ckpt"
    ckpt.write_bytes(b"fake")

    eval_result = MagicMock()
    eval_result.returncode = 0
    eval_result.stdout = "Metric avg si_sdr      : 7.5\nMetric avg pesq        : 2.8\nMetric avg estoi       : 0.85\n"
    eval_result.stderr = ""

    mock_run.return_value = eval_result  # Only one subprocess call (no training)

    run_job(
        tracker=tracker,
        storage=storage,
        job_id=job_id,
        experiment=exp,
        resolved_config=resolved,
        data_path=[Path("/fake/data")],
        valid_path=[Path("/fake/valid")],
        device_ids=[0],
        start_checkpoint=ckpt,
    )

    job = tracker.get_job(job_id)
    assert job.state == JobState.DONE
    assert job.metrics["si_sdr"] == pytest.approx(7.5)
    assert job.metrics["pesq"] == pytest.approx(2.8)
    assert job.metrics["estoi"] == pytest.approx(0.85)
    assert mock_run.call_count == 1  # Only eval, no training
```

- [ ] **Step 4: Add test for eval-only without checkpoint fails**

```python
@patch("postdoc.run_job.subprocess.run")
def test_run_job_eval_only_no_checkpoint_fails(mock_run, tracker, storage, job_setup, tmp_results_dir):
    job_id, exp, resolved = job_setup
    exp["eval_only"] = True

    run_job(
        tracker=tracker,
        storage=storage,
        job_id=job_id,
        experiment=exp,
        resolved_config=resolved,
        data_path=[Path("/fake/data")],
        valid_path=[Path("/fake/valid")],
        device_ids=[0],
    )

    job = tracker.get_job(job_id)
    assert job.state == JobState.FAILED
    assert "NoCheckpoint" in (job.error_category or "")
    mock_run.assert_not_called()
```

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/test_run_job.py -v`
Expected: All tests pass.

- [ ] **Step 6: Commit**

```bash
git add tests/test_run_job.py
git commit -m "test: add tests for eval-only mode, ESTOI extraction, and build_eval_args"
```

---

### Task 5: Create experiment YAMLs and submit eval jobs

**Files:**
- Create: `experiments/eval_dcunet_baseline_dregon.yaml`
- Create: `experiments/eval_dcunet_rps_dregon.yaml`

- [ ] **Step 1: Create DCUNet baseline eval experiment**

```yaml
# Eval-only: DCUNet baseline on DREGON-LM valid split
eval_only: true
checkpoint: results/dcunet_baseline_dregon/model_dcunet_ep_119_sdr_1.2846.ckpt

model:
  type: dcunet
  base_config: configs/7b_DCUNet_baseline_DREGON.yaml

dataset:
  name: DREGON-LM

eval:
  metrics: [estoi, si_sdr, pesq]

wandb:
  tags: [eval, dcunet, baseline, dregon]
```

- [ ] **Step 2: Create DCUNet+RPS eval experiment**

```yaml
# Eval-only: DCUNet+RPS on DREGON-LM valid split
eval_only: true
checkpoint: results/dcunet_rps_dregon/model_dcunet_ep_124_sdr_1.2997.ckpt

model:
  type: dcunet
  base_config: configs/7a_DCUNet_RPS_DREGON.yaml

dataset:
  name: DREGON-LM

eval:
  metrics: [estoi, si_sdr, pesq]

wandb:
  tags: [eval, dcunet, rps, dregon]
```

- [ ] **Step 3: Commit experiment files**

```bash
git add experiments/eval_dcunet_baseline_dregon.yaml experiments/eval_dcunet_rps_dregon.yaml
git commit -m "feat: add eval-only experiments for DCUNet vs DCUNet+RPS on DREGON"
```

- [ ] **Step 4: Submit both eval jobs**

```bash
uv run postdoc job submit experiments/eval_dcunet_baseline_dregon.yaml experiments/eval_dcunet_rps_dregon.yaml
```

- [ ] **Step 5: Monitor and compare results**

```bash
uv run postdoc job list
# Once done:
uv run postdoc results compare <job_id_1> <job_id_2>
```
