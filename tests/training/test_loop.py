"""Loop smoke test: tiny synthetic map-style dataset + 2-layer model on the
rps_prediction task, 2 epochs CPU, wandb mocked out entirely.

``wandb.init`` reassigns ``wandb.log``/``wandb.run`` internally when a real
run starts, which silently defeats a plain ``monkeypatch.setattr(wandb,
"log", ...)`` done *before* ``run_training`` calls ``wandb.init`` — so this
mocks the ``wandb`` name binding inside ``training.loop`` itself, replacing
it with a tiny recording stub for the duration of each test.
"""

from __future__ import annotations

import math

import training.loop as loop_module
from tests.training.conftest import make_tiny_config
from tests.training.test_artifacts import FakeS3Client
from training.artifacts import ArtifactStore
from training.loop import run_training


class _FakeRun:
    id = "fake-run-id"

    def __init__(self) -> None:
        self.summary: dict = {}


class _FakeWandb:
    def __init__(self) -> None:
        self.logged: list[dict] = []
        self.last_run: _FakeRun | None = None

    def init(self, *args, **kwargs):
        run = _FakeRun()
        self.last_run = run
        return run

    def log(self, data, *args, **kwargs):
        self.logged.append(dict(data))

    def finish(self, *args, **kwargs):
        pass


def test_run_training_writes_checkpoints_and_produces_finite_loss(tmp_path, monkeypatch):
    fake_wandb = _FakeWandb()
    monkeypatch.setattr(loop_module, "wandb", fake_wandb)

    cfg = make_tiny_config(
        results_root=str(tmp_path),
        experiment_name="tiny_loop",
        epochs=2,
        n_train=6,
        n_valid=4,
        batch_size=2,
    )
    result = run_training(cfg)

    run_dir = tmp_path / "tiny_loop"
    assert (run_dir / "best.ckpt").is_file()

    train_losses = [row["train/loss"] for row in fake_wandb.logged if "train/loss" in row]
    assert len(train_losses) == 2  # one per epoch
    assert all(math.isfinite(v) for v in train_losses)

    # The training loss is also evaluated on validation data and logged as
    # val/loss (additive; early stopping still watches `monitor`).
    val_losses = [row["val/loss"] for row in fake_wandb.logged if "val/loss" in row]
    assert len(val_losses) == 2  # one per epoch
    assert all(math.isfinite(v) for v in val_losses)

    assert "best_mse" in result
    assert math.isfinite(result["best_mse"])
    assert math.isfinite(result["final_epoch"])


def test_run_training_refuses_to_overwrite_a_nonempty_run_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(loop_module, "wandb", _FakeWandb())

    cfg = make_tiny_config(results_root=str(tmp_path), experiment_name="tiny_loop2", epochs=1)
    run_training(cfg)

    try:
        run_training(cfg)
    except FileExistsError:
        pass
    else:
        raise AssertionError("expected FileExistsError on a second run without resume=true")


def test_run_training_uploads_best_checkpoint_to_injected_artifact_store(tmp_path, monkeypatch):
    fake_wandb = _FakeWandb()
    monkeypatch.setattr(loop_module, "wandb", fake_wandb)

    client = FakeS3Client()
    experiment_name = "tiny_loop_artifacts"
    store = ArtifactStore(experiment_name=experiment_name, client=client, enabled=True)

    cfg = make_tiny_config(
        results_root=str(tmp_path),
        experiment_name=experiment_name,
        epochs=2,
        n_train=6,
        n_valid=4,
        batch_size=2,
        artifacts_enabled=True,
        num_val_samples=0,  # this test is scoped to checkpoint upload, not sample logging
    )
    run_training(cfg, artifact_store=store)

    ckpt_key = f"ml-data/artifacts/{experiment_name}/checkpoints/best.ckpt"
    assert ckpt_key in client.objects, (
        f"expected checkpoint at {ckpt_key}, got keys: {list(client.objects)}"
    )

    assert fake_wandb.last_run is not None
    recorded_uri = fake_wandb.last_run.summary.get("r2/best_checkpoint")
    assert recorded_uri == f"r2://{ckpt_key}"


def test_run_training_val_sample_logging_and_upload_when_enabled(tmp_path, monkeypatch):
    fake_wandb = _FakeWandb()
    monkeypatch.setattr(loop_module, "wandb", fake_wandb)

    client = FakeS3Client()
    experiment_name = "tiny_loop_val_samples"
    store = ArtifactStore(experiment_name=experiment_name, client=client, enabled=True)

    cfg = make_tiny_config(
        results_root=str(tmp_path),
        experiment_name=experiment_name,
        epochs=1,
        n_train=6,
        n_valid=4,
        batch_size=2,
        artifacts_enabled=True,
        num_val_samples=2,
    )
    run_training(cfg, artifact_store=store)

    # wandb.Audio/Image payloads logged alongside the usual train/val scalars.
    audio_rows = [row for row in fake_wandb.logged if any(k.startswith("val/") for k in row)]
    assert audio_rows, "expected at least one wandb.log call carrying val/ sample keys"

    # And the same samples reached the (fake) R2 client as a manifest.
    manifest_keys = [k for k in client.objects if k.endswith("manifest.json")]
    assert manifest_keys, f"expected a val_samples manifest.json, got keys: {list(client.objects)}"
