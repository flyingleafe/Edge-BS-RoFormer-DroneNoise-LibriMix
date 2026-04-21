import json
import pytest
from pathlib import PurePosixPath
from postdoc.backends.local.storage import LocalStorage


@pytest.fixture
def storage(tmp_results_dir):
    return LocalStorage(tmp_results_dir)


def test_put_and_get(storage):
    storage.put("job-1", PurePosixPath("training/logs/stdout.txt"), b"hello world")
    data = storage.get("job-1", PurePosixPath("training/logs/stdout.txt"))
    assert data == b"hello world"


def test_get_nonexistent(storage):
    with pytest.raises(FileNotFoundError):
        storage.get("job-1", PurePosixPath("nope.txt"))


def test_exists(storage):
    assert not storage.exists("job-1", PurePosixPath("foo.txt"))
    storage.put("job-1", PurePosixPath("foo.txt"), b"data")
    assert storage.exists("job-1", PurePosixPath("foo.txt"))


def test_list_artifacts(storage):
    storage.put("job-1", PurePosixPath("training/log1.txt"), b"a")
    storage.put("job-1", PurePosixPath("training/log2.txt"), b"b")
    storage.put("job-1", PurePosixPath("eval/metrics.json"), b"c")

    all_files = storage.list("job-1")
    assert len(all_files) == 3

    training_files = storage.list("job-1", prefix=PurePosixPath("training"))
    assert len(training_files) == 2


def test_put_json_and_get_json(storage):
    data = {"si_sdr": 12.5, "pesq": 2.3}
    storage.put_json("job-1", PurePosixPath("eval/metrics.json"), data)
    result = storage.get_json("job-1", PurePosixPath("eval/metrics.json"))
    assert result == data


def test_get_to_file(storage, tmp_path):
    storage.put("job-1", PurePosixPath("model.ckpt"), b"model data")
    dest = tmp_path / "downloaded.ckpt"
    storage.get_to_file("job-1", PurePosixPath("model.ckpt"), dest)
    assert dest.read_bytes() == b"model data"


def test_job_root_path(storage, tmp_results_dir):
    path = storage.job_root_path("job-1")
    assert path == str(tmp_results_dir / "job-1")
