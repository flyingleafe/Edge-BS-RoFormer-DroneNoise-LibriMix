import json
import subprocess
import pytest
from pathlib import Path, PurePosixPath
from unittest.mock import patch, MagicMock
from postdoc.run_job import run_job, extract_metrics, find_best_checkpoint
from postdoc.interfaces.tracker import JobTracker, JobState
from postdoc.backends.local.storage import LocalStorage


@pytest.fixture
def tracker(tmp_path):
    t = JobTracker(tmp_path / "test.db")
    yield t
    t.close()


@pytest.fixture
def storage(tmp_results_dir):
    return LocalStorage(tmp_results_dir)


@pytest.fixture
def job_setup(tracker, storage, tmp_results_dir, sample_experiment_yaml, tmp_path):
    from postdoc.experiment import load_experiment, resolve_config
    exp = load_experiment(sample_experiment_yaml)
    job_id = tracker.create_job("test-exp", exp, "exp/test", "abc123")
    job_dir = tmp_results_dir / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    resolved = job_dir / "config.yaml"
    resolve_config(exp, resolved)
    return job_id, exp, resolved


def test_extract_metrics_from_stdout():
    stdout = """
Instr vocals sdr: 8.2340 (Std: 1.2345)
Instr vocals si_sdr: 7.8910 (Std: 1.1234)
Instr noise sdr: 6.0000 (Std: 0.5000)
Instr noise si_sdr: 5.5000 (Std: 0.4000)
Metric avg sdr         : 7.1170
Metric avg si_sdr      : 6.6955
Metric avg pesq        : 2.3450
Metric avg stoi        : 0.8560
"""
    metrics, incomplete = extract_metrics(stdout)
    assert metrics["sdr"] == pytest.approx(7.117)
    assert metrics["si_sdr"] == pytest.approx(6.6955)
    assert metrics["pesq"] == pytest.approx(2.345)
    assert metrics["stoi"] == pytest.approx(0.856)
    assert not incomplete


def test_extract_metrics_single_instrument():
    stdout = "Instr vocals sdr: 8.234 (Std: 1.0)\nInstr vocals si_sdr: 7.5 (Std: 0.8)\n"
    metrics, incomplete = extract_metrics(stdout)
    assert metrics["sdr"] == pytest.approx(8.234)
    assert metrics["si_sdr"] == pytest.approx(7.5)
    assert not incomplete


def test_extract_metrics_empty():
    metrics, incomplete = extract_metrics("")
    assert metrics == {}
    assert incomplete


def test_find_best_checkpoint(tmp_path):
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    (ckpt_dir / "model_0001.ckpt").write_bytes(b"a")
    (ckpt_dir / "model_0005.ckpt").write_bytes(b"b")
    (ckpt_dir / "model_0003.ckpt").write_bytes(b"c")
    best = find_best_checkpoint(ckpt_dir)
    assert best.name == "model_0005.ckpt"


def test_find_best_checkpoint_prefers_best_prefix(tmp_path):
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    (ckpt_dir / "model_0005.ckpt").write_bytes(b"a")
    (ckpt_dir / "best_model.ckpt").write_bytes(b"b")
    best = find_best_checkpoint(ckpt_dir)
    assert "best" in best.name


@patch("postdoc.run_job.subprocess.run")
def test_run_job_success(mock_run, tracker, storage, job_setup, tmp_results_dir):
    job_id, exp, resolved = job_setup

    train_result = MagicMock()
    train_result.returncode = 0
    train_result.stdout = ""
    train_result.stderr = ""

    eval_result = MagicMock()
    eval_result.returncode = 0
    eval_result.stdout = "Metric avg sdr         : 8.0000\nMetric avg si_sdr      : 7.5000\n"
    eval_result.stderr = ""

    mock_run.side_effect = [train_result, eval_result]

    ckpt_dir = tmp_results_dir / job_id / "training" / "checkpoints"
    ckpt_dir.mkdir(parents=True)
    (ckpt_dir / "model_0001.ckpt").write_bytes(b"fake")

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
    assert job.state == JobState.DONE
    assert job.metrics is not None
    assert job.metrics["sdr"] == pytest.approx(8.0)


@patch("postdoc.run_job.subprocess.run")
def test_run_job_training_failure(mock_run, tracker, storage, job_setup, tmp_results_dir):
    job_id, exp, resolved = job_setup

    train_result = MagicMock()
    train_result.returncode = 1
    train_result.stdout = ""
    train_result.stderr = "RuntimeError: CUDA out of memory"

    mock_run.return_value = train_result

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
    assert "OOM" in (job.error_category or "")


def test_extract_metrics_with_estoi():
    stdout = "Metric avg estoi       : 0.7890\nMetric avg si_sdr      : 6.5\n"
    metrics, incomplete = extract_metrics(stdout)
    assert metrics["estoi"] == pytest.approx(0.789)
    assert not incomplete


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
