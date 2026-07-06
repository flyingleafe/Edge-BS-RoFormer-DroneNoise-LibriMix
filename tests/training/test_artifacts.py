"""Unit tests for ``training.artifacts.ArtifactStore``.

Uses a small in-memory fake filesystem (dependency-injected via
``ArtifactStore(fs=...)``) instead of monkeypatching ``s3fs`` internals — see
module docstring of ``training/artifacts.py``. The one test that talks to
real Cloudflare R2 lives in ``test_artifacts_r2_integration.py``.
"""

from __future__ import annotations

import io
import json

import numpy as np
import soundfile as sf

from training.artifacts import R2_ENV_VARS, ArtifactStore, ValSample, _load_r2_env


class _FakeFile:
    def __init__(self, fs: FakeFS, path: str, mode: str) -> None:
        self._fs = fs
        self._path = path
        self._mode = mode
        self._buf = io.BytesIO()

    def __enter__(self) -> _FakeFile:
        return self

    def __exit__(self, *exc: object) -> bool:
        if "w" in self._mode:
            self._fs.files[self._path] = self._buf.getvalue()
        return False

    def write(self, data: bytes | str) -> int:
        if isinstance(data, str):
            data = data.encode("utf-8")
        return self._buf.write(data)


class FakeFS:
    """Minimal in-memory stand-in for ``s3fs.S3FileSystem``."""

    def __init__(self, *, fail_paths: set[str] | None = None) -> None:
        self.files: dict[str, bytes] = {}
        self.dirs: set[str] = set()
        self.fail_paths = fail_paths or set()

    def makedirs(self, path: str, exist_ok: bool = True) -> None:
        self.dirs.add(path)

    def open(self, path: str, mode: str = "rb") -> _FakeFile:
        if path in self.fail_paths:
            raise RuntimeError(f"synthetic failure opening {path}")
        return _FakeFile(self, path, mode)


# ─── upload_checkpoint ──────────────────────────────────────────────────────


def test_upload_checkpoint_writes_expected_key(tmp_path):
    ckpt = tmp_path / "best.ckpt"
    ckpt.write_bytes(b"fake-checkpoint-bytes")
    fs = FakeFS()
    store = ArtifactStore(experiment_name="exp1", fs=fs, enabled=True)

    uri = store.upload_checkpoint(ckpt)

    expected_key = "ml-data/artifacts/exp1/checkpoints/best.ckpt"
    assert uri == f"r2://{expected_key}"
    assert fs.files[expected_key] == b"fake-checkpoint-bytes"
    assert "ml-data/artifacts/exp1/checkpoints" in fs.dirs


def test_upload_checkpoint_custom_bucket_and_prefix(tmp_path):
    ckpt = tmp_path / "ep3_mse_0.1234.ckpt"
    ckpt.write_bytes(b"x")
    fs = FakeFS()
    store = ArtifactStore(
        experiment_name="exp2", bucket="other-bucket", prefix="/runs/", fs=fs, enabled=True
    )

    uri = store.upload_checkpoint(ckpt)

    assert uri == "r2://other-bucket/runs/exp2/checkpoints/ep3_mse_0.1234.ckpt"


def test_upload_checkpoint_disabled_is_noop(tmp_path):
    ckpt = tmp_path / "best.ckpt"
    ckpt.write_bytes(b"x")
    fs = FakeFS()
    store = ArtifactStore(experiment_name="exp1", fs=fs, enabled=False)

    assert store.upload_checkpoint(ckpt) is None
    assert fs.files == {}


def test_upload_checkpoint_exception_does_not_propagate(tmp_path):
    ckpt = tmp_path / "best.ckpt"
    ckpt.write_bytes(b"x")
    fs = FakeFS(fail_paths={"ml-data/artifacts/exp1/checkpoints/best.ckpt"})
    store = ArtifactStore(experiment_name="exp1", fs=fs, enabled=True)

    # Must not raise.
    result = store.upload_checkpoint(ckpt)
    assert result is None


# ─── upload_val_samples ─────────────────────────────────────────────────────


def test_upload_val_samples_writes_audio_figures_and_manifest():
    fs = FakeFS()
    store = ArtifactStore(experiment_name="exp1", fs=fs, enabled=True)

    wav = (np.linspace(-1.0, 1.0, 1600, dtype=np.float32), 16000)
    samples = [
        ValSample(
            sample_id="sample_000",
            input_snr=-12.5,
            audio={"mixture": wav, "target": wav, "output": wav},
            figures={"rps_overlay": b"\x89PNGfakebytes"},
            metrics={"mse": 0.05, "r2": 0.8},
        ),
        ValSample(sample_id="sample_001", input_snr=None, audio={"mixture": wav}),
    ]

    manifest_uri = store.upload_val_samples(7, samples)

    root = "ml-data/artifacts/exp1/val_samples/epoch_7"
    assert manifest_uri == f"r2://{root}/manifest.json"

    # Audio + figure keys present with the expected naming convention.
    assert f"{root}/sample_000__mixture.wav" in fs.files
    assert f"{root}/sample_000__target.wav" in fs.files
    assert f"{root}/sample_000__output.wav" in fs.files
    assert f"{root}/sample_000__rps_overlay.png" in fs.files
    assert fs.files[f"{root}/sample_000__rps_overlay.png"] == b"\x89PNGfakebytes"
    assert f"{root}/sample_001__mixture.wav" in fs.files

    # The wav bytes are a real, readable WAV file at the right sample rate.
    data, sr = sf.read(io.BytesIO(fs.files[f"{root}/sample_000__mixture.wav"]))
    assert sr == 16000
    assert len(data) == 1600

    manifest = json.loads(fs.files[f"{root}/manifest.json"])
    assert manifest["epoch"] == 7
    ids = {row["sample_id"]: row for row in manifest["samples"]}
    assert ids["sample_000"]["input_snr"] == -12.5
    assert ids["sample_000"]["metrics"] == {"mse": 0.05, "r2": 0.8}
    assert ids["sample_000"]["keys"]["mixture"] == f"r2://{root}/sample_000__mixture.wav"
    assert ids["sample_000"]["keys"]["rps_overlay"] == f"r2://{root}/sample_000__rps_overlay.png"
    assert ids["sample_001"]["input_snr"] is None


def test_upload_val_samples_empty_list_returns_none():
    fs = FakeFS()
    store = ArtifactStore(experiment_name="exp1", fs=fs, enabled=True)
    assert store.upload_val_samples(0, []) is None
    assert fs.files == {}


def test_upload_val_samples_disabled_is_noop():
    fs = FakeFS()
    store = ArtifactStore(experiment_name="exp1", fs=fs, enabled=False)
    wav = (np.zeros(100, dtype=np.float32), 16000)
    samples = [ValSample(sample_id="s0", audio={"mixture": wav})]

    assert store.upload_val_samples(0, samples) is None
    assert fs.files == {}


def test_upload_val_samples_exception_does_not_propagate():
    fs = FakeFS(fail_paths={"ml-data/artifacts/exp1/val_samples/epoch_0/s0__mixture.wav"})
    store = ArtifactStore(experiment_name="exp1", fs=fs, enabled=True)
    wav = (np.zeros(100, dtype=np.float32), 16000)
    samples = [ValSample(sample_id="s0", audio={"mixture": wav})]

    result = store.upload_val_samples(0, samples)
    assert result is None


# ─── missing-env / no-fs no-op path ─────────────────────────────────────────


def test_load_r2_env_returns_none_when_vars_missing(monkeypatch):
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: None)
    for var in R2_ENV_VARS:
        monkeypatch.delenv(var, raising=False)

    assert _load_r2_env() is None


def test_load_r2_env_returns_values_when_present(monkeypatch):
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: None)
    monkeypatch.setenv("R2_ACCOUNT_ID", "acct")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "key")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret")

    env = _load_r2_env()
    assert env == {
        "R2_ACCOUNT_ID": "acct",
        "AWS_ACCESS_KEY_ID": "key",
        "AWS_SECRET_ACCESS_KEY": "secret",
    }


def test_store_without_fs_or_env_is_noop(tmp_path, monkeypatch):
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: None)
    for var in R2_ENV_VARS:
        monkeypatch.delenv(var, raising=False)

    ckpt = tmp_path / "best.ckpt"
    ckpt.write_bytes(b"x")
    store = ArtifactStore(experiment_name="exp1", enabled=True)  # no fs injected

    assert store.active is False
    assert store.upload_checkpoint(ckpt) is None
