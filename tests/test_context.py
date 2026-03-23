import pytest
from postdoc.context import create_context
from postdoc.backends.local.storage import LocalStorage
from postdoc.backends.local.scheduler import LocalScheduler
from postdoc.interfaces.tracker import JobTracker


def test_create_context_local(sample_postdoc_yaml):
    ctx = create_context(config_path=sample_postdoc_yaml)
    assert isinstance(ctx.storage, LocalStorage)
    assert isinstance(ctx.scheduler, LocalScheduler)
    assert isinstance(ctx.tracker, JobTracker)
    ctx.tracker.close()


def test_create_context_cloud_raises(sample_postdoc_yaml, monkeypatch):
    monkeypatch.setenv("POSTDOC_BACKEND", "cloud")
    with pytest.raises(NotImplementedError):
        create_context(config_path=sample_postdoc_yaml)


def test_create_context_explicit_backend(sample_postdoc_yaml):
    ctx = create_context(config_path=sample_postdoc_yaml, backend="local")
    assert ctx.config.backend == "local"
    ctx.tracker.close()
