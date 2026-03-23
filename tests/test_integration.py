"""End-to-end integration test — no GPU required."""
import pytest
from pathlib import Path, PurePosixPath
from unittest.mock import patch, MagicMock

from postdoc.context import create_context
from postdoc.experiment import load_experiment, resolve_config
from postdoc.interfaces.tracker import JobState
from postdoc.run_job import run_job


@pytest.fixture
def integration_ctx(sample_postdoc_yaml):
    ctx = create_context(config_path=sample_postdoc_yaml)
    yield ctx
    ctx.tracker.close()


@patch("postdoc.run_job.subprocess.run")
def test_full_pipeline(mock_run, integration_ctx, sample_experiment_yaml):
    ctx = integration_ctx

    train_result = MagicMock(returncode=0, stdout="training done", stderr="")
    eval_result = MagicMock(
        returncode=0,
        stdout="Instr vocals sdr: 10.5 (Std: 1.0)\nInstr vocals si_sdr: 9.2 (Std: 0.8)\nMetric avg sdr         : 10.5000\nMetric avg si_sdr      : 9.2000\nMetric avg pesq        : 2.8000\nMetric avg stoi        : 0.9100\n",
        stderr="",
    )
    mock_run.side_effect = [train_result, eval_result]

    exp = load_experiment(sample_experiment_yaml)
    job_id = ctx.tracker.create_job("integration-test", exp, "exp/integration", "abc123")

    results_dir = Path(ctx.config.local.results_dir)
    job_dir = results_dir / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    resolved = job_dir / "config.yaml"
    resolve_config(exp, resolved)

    ckpt_dir = job_dir / "training" / "checkpoints"
    ckpt_dir.mkdir(parents=True)
    (ckpt_dir / "model_0001.ckpt").write_bytes(b"fake checkpoint")

    run_job(
        tracker=ctx.tracker,
        storage=ctx.storage,
        job_id=job_id,
        experiment=exp,
        resolved_config=resolved,
        data_path=[Path("/fake/data")],
        valid_path=[Path("/fake/valid")],
        device_ids=[0],
    )

    job = ctx.tracker.get_job(job_id)
    assert job.state == JobState.DONE
    assert job.metrics["sdr"] == pytest.approx(10.5)
    assert job.metrics["si_sdr"] == pytest.approx(9.2)
    assert job.metrics["pesq"] == pytest.approx(2.8)
    assert job.metrics["stoi"] == pytest.approx(0.91)

    assert ctx.storage.exists(job_id, PurePosixPath("training/logs/stdout.txt"))
    assert ctx.storage.exists(job_id, PurePosixPath("eval/logs/stdout.txt"))
    assert ctx.storage.exists(job_id, PurePosixPath("eval/metrics.json"))
    assert ctx.storage.exists(job_id, PurePosixPath("meta.json"))

    metrics_stored = ctx.storage.get_json(job_id, PurePosixPath("eval/metrics.json"))
    assert metrics_stored["sdr"] == pytest.approx(10.5)

    all_jobs = ctx.tracker.list_jobs()
    assert len(all_jobs) == 1
    assert all_jobs[0].job_id == job_id


@patch("postdoc.run_job.subprocess.run")
def test_full_pipeline_training_failure(mock_run, integration_ctx, sample_experiment_yaml):
    ctx = integration_ctx

    train_result = MagicMock(
        returncode=1, stdout="",
        stderr="RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB",
    )
    mock_run.return_value = train_result

    exp = load_experiment(sample_experiment_yaml)
    job_id = ctx.tracker.create_job("fail-test", exp, "exp/fail", "def456")

    results_dir = Path(ctx.config.local.results_dir)
    job_dir = results_dir / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    resolved = job_dir / "config.yaml"
    resolve_config(exp, resolved)

    run_job(
        tracker=ctx.tracker, storage=ctx.storage,
        job_id=job_id, experiment=exp, resolved_config=resolved,
        data_path=[Path("/fake/data")], valid_path=[Path("/fake/valid")], device_ids=[0],
    )

    job = ctx.tracker.get_job(job_id)
    assert job.state == JobState.FAILED
    assert job.error_category == "OOM"
    assert "CUDA out of memory" in job.error_message
    assert not ctx.storage.exists(job_id, PurePosixPath("eval/logs/stdout.txt"))
    assert ctx.storage.exists(job_id, PurePosixPath("training/logs/stderr.txt"))


def test_queuing_when_no_capacity(integration_ctx, sample_experiment_yaml):
    ctx = integration_ctx
    exp = load_experiment(sample_experiment_yaml)

    for i in range(ctx.config.local.gpus):
        jid = ctx.tracker.create_job(f"busy-{i}", exp, f"exp/busy-{i}", "aaa")
        ctx.scheduler._allocate_gpu(jid, i)

    from postdoc.interfaces.scheduler import NoCapacityError
    job_id = ctx.tracker.create_job("queued-job", exp, "exp/queued", "bbb")
    results_dir = Path(ctx.config.local.results_dir)
    job_dir = results_dir / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    resolved = job_dir / "config.yaml"
    resolve_config(exp, resolved)

    with pytest.raises(NoCapacityError):
        ctx.scheduler.submit(job_id, resolved, exp)
