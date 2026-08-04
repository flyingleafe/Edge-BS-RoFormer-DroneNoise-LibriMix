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

import pytest
import torch

import training.loop as loop_module
from tests.training.conftest import make_tiny_config
from tests.training.test_artifacts import FakeS3Client
from training.artifacts import ArtifactStore
from training.loop import run_training

pytestmark = pytest.mark.slow


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


def test_resume_continues_from_the_saved_epoch_instead_of_restarting(tmp_path, monkeypatch):
    """A relaunch with resume=true must pick up where the last one stopped.

    This is what makes preemptible/short-time-limit queues usable: before, an
    interrupted run restarted from scratch (`resume` only unlocked the run dir),
    so a 1 h wall-clock limit capped total training at 1 h no matter how many
    times it was relaunched.
    """
    fake_wandb = _FakeWandb()
    monkeypatch.setattr(loop_module, "wandb", fake_wandb)

    def cfg_for(epochs: int):
        return make_tiny_config(
            results_root=str(tmp_path),
            experiment_name="tiny_resume",
            epochs=epochs,
            n_train=6,
            n_valid=4,
        )

    run_training(cfg_for(2))
    run_dir = tmp_path / "tiny_resume"
    assert (run_dir / "train_state.pt").is_file()

    first_epochs = [row["epoch"] for row in fake_wandb.logged if "epoch" in row]
    assert first_epochs == [0, 1]

    fake_wandb.logged.clear()
    cfg = cfg_for(4)
    cfg.resume = True
    run_training(cfg)

    # Epochs 2 and 3 only — not 0..3 again.
    assert [row["epoch"] for row in fake_wandb.logged if "epoch" in row] == [2, 3]


def test_resume_on_a_fresh_run_dir_starts_from_zero(tmp_path, monkeypatch):
    """resume=true is the safe default to submit with, so it must be a no-op
    when there is nothing to resume from."""
    monkeypatch.setattr(loop_module, "wandb", (fake_wandb := _FakeWandb()))

    cfg = make_tiny_config(
        results_root=str(tmp_path), experiment_name="tiny_fresh", epochs=2, n_train=6, n_valid=4
    )
    cfg.resume = True
    run_training(cfg)

    assert [row["epoch"] for row in fake_wandb.logged if "epoch" in row] == [0, 1]


def test_resume_after_early_stop_exits_without_training(tmp_path, monkeypatch):
    """A chain of short segments must stop by itself once the run has converged.

    Early stopping is checked at the END of an epoch, so a resumed already-done
    run would otherwise train one full epoch before re-discovering it was done —
    once per queued segment, indefinitely.
    """
    monkeypatch.setattr(loop_module, "wandb", (fake_wandb := _FakeWandb()))

    def cfg_for(epochs: int):
        cfg = make_tiny_config(
            results_root=str(tmp_path),
            experiment_name="tiny_done",
            epochs=epochs,
            n_train=6,
            n_valid=4,
        )
        cfg.patience = 1  # so the tiny run early-stops within its epoch budget
        return cfg

    run_training(cfg_for(3))
    # The tiny model improves every epoch, so drive it to the converged state
    # directly rather than contriving a plateau.
    state_path = tmp_path / "tiny_done" / "train_state.pt"
    state = torch.load(state_path, weights_only=False)
    state["no_improve"] = 1
    torch.save(state, state_path)

    fake_wandb.logged.clear()
    cfg = cfg_for(10)
    cfg.resume = True
    result = run_training(cfg)

    assert [row for row in fake_wandb.logged if "epoch" in row] == []  # no epoch ran
    assert math.isfinite(result["best_mse"])
