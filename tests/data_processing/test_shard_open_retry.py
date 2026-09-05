"""The shard-open retry guard installed at import of ``data_processing.streams``.

dload 0.3.0 opens a shard with no retry, so one dropped TLS connection inside a
DataLoader worker kills a whole training job (vast.ai: urllib3 ``SSLError``).
The wrapper retries network errors and lets everything else through.
"""

from __future__ import annotations

import ssl

import dload
import pytest
from urllib3.exceptions import ProtocolError

from data_processing import streams


class _Flaky:
    """A stand-in ``open_shard`` that fails ``n_failures`` times, then works."""

    def __init__(self, n_failures: int, exc: BaseException) -> None:
        self.n_failures = n_failures
        self.exc = exc
        self.calls = 0

    def __call__(self, repo, shard):  # noqa: ANN001 - a bound-method stand-in
        self.calls += 1
        if self.calls <= self.n_failures:
            raise self.exc
        return f"shard:{shard}"


@pytest.fixture
def install(monkeypatch):
    """Install the retry wrapper over a flaky stand-in, with no real sleeping."""
    slept: list[float] = []
    monkeypatch.setattr(streams.time, "sleep", slept.append)

    def _install(flaky: _Flaky):
        # monkeypatch records the real wrapper here and restores it on teardown.
        monkeypatch.setattr(dload.Repository, "open_shard", flaky, raising=False)
        streams.install_shard_open_retry()
        return dload.Repository.open_shard, slept

    return _install


def test_retries_twice_then_succeeds(install):
    flaky = _Flaky(2, ssl.SSLError("handshake failed"))
    open_shard, slept = install(flaky)

    assert open_shard(object(), "s0") == "shard:s0"
    assert flaky.calls == 3
    assert slept == [1.0, 2.0]  # exponential backoff from 1 s


def test_urllib3_errors_are_retried_too(install):
    flaky = _Flaky(1, ProtocolError("connection broken"))
    open_shard, slept = install(flaky)

    assert open_shard(object(), "s1") == "shard:s1"
    assert (flaky.calls, slept) == (2, [1.0])


def test_gives_up_after_five_attempts(install):
    flaky = _Flaky(99, OSError("connection reset"))
    open_shard, slept = install(flaky)

    with pytest.raises(OSError, match="connection reset"):
        open_shard(object(), "s2")
    assert flaky.calls == streams.SHARD_OPEN_ATTEMPTS == 5
    assert slept == [1.0, 2.0, 4.0, 8.0]


def test_other_exceptions_are_not_retried(install):
    flaky = _Flaky(1, ValueError("corrupt manifest"))
    open_shard, slept = install(flaky)

    with pytest.raises(ValueError, match="corrupt manifest"):
        open_shard(object(), "s3")
    assert (flaky.calls, slept) == (1, [])


def test_install_is_idempotent(install):
    flaky = _Flaky(0, OSError("unused"))
    install(flaky)
    first = dload.Repository.open_shard
    streams.install_shard_open_retry()
    assert dload.Repository.open_shard is first
