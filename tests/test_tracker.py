import pytest
from postdoc.interfaces.tracker import JobTracker, JobState, JobRecord


@pytest.fixture
def tracker(tmp_path):
    t = JobTracker(tmp_path / "test.db")
    yield t
    t.close()


def test_create_and_get_job(tracker):
    job_id = tracker.create_job("exp-1", {"model": "dcunet"}, "exp/exp-1", "abc123")
    job = tracker.get_job(job_id)
    assert job.experiment_name == "exp-1"
    assert job.state == JobState.DEFINED
    assert job.git_branch == "exp/exp-1"
    assert job.git_commit == "abc123"
    assert job.config_snapshot == {"model": "dcunet"}
    assert job.submitted_at is not None


def test_get_nonexistent_job(tracker):
    with pytest.raises(KeyError):
        tracker.get_job("nonexistent")


def test_update_state(tracker):
    job_id = tracker.create_job("exp-1", {}, "b", "c")
    tracker.update_state(job_id, JobState.TRAINING, process_handle="12345", gpu_ids=[0])
    job = tracker.get_job(job_id)
    assert job.state == JobState.TRAINING
    assert job.process_handle == "12345"
    assert job.gpu_ids == [0]
    assert job.started_at is not None


def test_update_state_failed(tracker):
    job_id = tracker.create_job("exp-1", {}, "b", "c")
    tracker.update_state(job_id, JobState.FAILED,
                         error_category="OOM", error_message="CUDA out of memory")
    job = tracker.get_job(job_id)
    assert job.state == JobState.FAILED
    assert job.error_category == "OOM"
    assert job.failed_at is not None


def test_list_jobs(tracker):
    tracker.create_job("exp-1", {}, "b1", "c1")
    tracker.create_job("exp-2", {}, "b2", "c2")
    jobs = tracker.list_jobs()
    assert len(jobs) == 2


def test_list_jobs_by_state(tracker):
    j1 = tracker.create_job("exp-1", {}, "b1", "c1")
    j2 = tracker.create_job("exp-2", {}, "b2", "c2")
    tracker.update_state(j1, JobState.TRAINING, process_handle="1", gpu_ids=[0])
    jobs = tracker.list_jobs(state=JobState.TRAINING)
    assert len(jobs) == 1
    assert jobs[0].job_id == j1


def test_get_queued_jobs_ordered(tracker):
    j1 = tracker.create_job("exp-1", {}, "b1", "c1")
    tracker.update_state(j1, JobState.QUEUED)
    j2 = tracker.create_job("exp-2", {}, "b2", "c2")
    tracker.update_state(j2, JobState.QUEUED)
    queued = tracker.get_queued_jobs()
    assert len(queued) == 2
    assert queued[0].job_id == j1


def test_set_metrics(tracker):
    job_id = tracker.create_job("exp-1", {}, "b", "c")
    metrics = {"si_sdr": 12.5, "pesq": 2.3, "stoi": 0.85}
    tracker.set_metrics(job_id, metrics)
    job = tracker.get_job(job_id)
    assert job.metrics == metrics
    assert not job.metrics_incomplete


def test_set_metrics_incomplete(tracker):
    job_id = tracker.create_job("exp-1", {}, "b", "c")
    tracker.set_metrics(job_id, {}, incomplete=True)
    job = tracker.get_job(job_id)
    assert job.metrics_incomplete


def test_running_jobs_on_branch(tracker):
    j1 = tracker.create_job("exp-1", {}, "exp/test", "c1")
    tracker.update_state(j1, JobState.TRAINING, process_handle="1", gpu_ids=[0])
    j2 = tracker.create_job("exp-2", {}, "exp/other", "c2")
    tracker.update_state(j2, JobState.TRAINING, process_handle="2", gpu_ids=[1])
    running = tracker.get_running_jobs_on_branch("exp/test")
    assert len(running) == 1
    assert running[0].job_id == j1
