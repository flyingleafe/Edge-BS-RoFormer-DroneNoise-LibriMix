import os
import signal
import subprocess
import time
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from postdoc.backends.local.scheduler import LocalScheduler
from postdoc.interfaces.scheduler import NoCapacityError
from postdoc.interfaces.tracker import JobTracker, JobState


@pytest.fixture
def tracker(tmp_path):
    t = JobTracker(tmp_path / "test.db")
    yield t
    t.close()


@pytest.fixture
def scheduler(tracker, tmp_results_dir):
    return LocalScheduler(num_gpus=2, tracker=tracker, results_dir=tmp_results_dir)


def test_available_capacity_initial(scheduler):
    assert scheduler.available_capacity() == 2


def test_available_capacity_after_allocation(scheduler, tracker):
    job_id = tracker.create_job("exp", {}, "b", "c")
    scheduler._allocate_gpu(job_id, 0)
    assert scheduler.available_capacity() == 1


def test_allocate_gpu_returns_free_gpu(scheduler, tracker):
    job_id = tracker.create_job("exp", {}, "b", "c")
    gpu = scheduler._find_free_gpu()
    assert gpu in [0, 1]


def test_no_capacity_when_full(scheduler, tracker):
    j1 = tracker.create_job("exp1", {}, "b1", "c1")
    j2 = tracker.create_job("exp2", {}, "b2", "c2")
    scheduler._allocate_gpu(j1, 0)
    scheduler._allocate_gpu(j2, 1)
    assert scheduler.available_capacity() == 0
    assert scheduler._find_free_gpu() is None


def test_release_gpu(scheduler, tracker):
    job_id = tracker.create_job("exp", {}, "b", "c")
    scheduler._allocate_gpu(job_id, 0)
    assert scheduler.available_capacity() == 1
    scheduler._release_gpu(job_id)
    assert scheduler.available_capacity() == 2


def test_is_alive_with_running_process(scheduler):
    proc = subprocess.Popen(["sleep", "60"])
    assert scheduler.is_alive(str(proc.pid))
    proc.terminate()
    proc.wait()


def test_is_alive_with_dead_process(scheduler):
    assert not scheduler.is_alive("99999999")
