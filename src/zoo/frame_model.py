"""``FrameModel`` — the (model, codec, task) triple as one Frame → Frame
callable — plus :func:`load`, which builds one from an experiment name.

``load`` follows the recipe proven in ``scripts/rps_predictor_vk_eval.py``:
Hydra-compose ``experiment=<name>`` against ``<repo>/conf``, instantiate the
model via :func:`training.config.instantiate_model`, resolve ``r2://``
checkpoint URIs through :func:`utils.checkpoints.resolve_checkpoint_uri`,
and pair the model with the codec from
:func:`training.config.build_task_and_codec`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import tdseries as td
import torch

from data_processing.collate import batch_size as _batch_size
from data_processing.collate import frame_collate, slice_sample
from tasks.codecs import Codec
from tasks.task import Task
from utils.checkpoints import resolve_checkpoint_uri
from zoo.cache import REPO_ROOT

__all__ = ["FrameModel", "load"]


def _is_batched(frame: td.Frame) -> bool:
    try:
        _batch_size(frame)
    except ValueError:
        return False
    return True


class FrameModel:
    """A trained model wrapped with its codec: ``td.Frame`` in, ``td.Frame`` out.

    ``__call__`` accepts either a single **unbatched** sample Frame (it is
    collated to a batch of one — the same ``frame_collate`` eval.py uses —
    run through the codec triple under ``torch.no_grad()``, and sliced back
    to an unbatched Frame) or an **already-batched** Frame (detected by a
    ``"batch"`` dim; passed through the codec directly, batched output
    returned). Outputs always come back on CPU.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        codec: Codec,
        task: Task,
        *,
        experiment: str | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        self.model = model
        self.codec = codec
        self.task = task
        self.experiment = experiment
        self.device = torch.device(device)

    @property
    def task_name(self) -> str:
        return self.task.name

    def __call__(self, frame: td.Frame) -> td.Frame:
        batched = _is_batched(frame)
        batch = frame if batched else frame_collate([frame])
        batch = batch.map_data(lambda t: torch.as_tensor(t).to(self.device))
        with torch.no_grad():
            inputs = self.codec.to_inputs(batch)
            outputs = self.codec.call_model(self.model, inputs)
            pred = self.codec.to_frame(outputs, batch)
        pred = pred.map_data(lambda t: t.detach().cpu())
        return pred if batched else slice_sample(pred, 0)

    def __repr__(self) -> str:
        exp = f", experiment={self.experiment!r}" if self.experiment else ""
        return (
            f"FrameModel(task={self.task.name!r}{exp}, "
            f"model={type(self.model).__name__}, device={self.device})"
        )


def _checkpoint_ref(name: str, ckpt: str, cfg: Any) -> str:
    """Turn ``ckpt`` into something ``resolve_checkpoint_uri`` accepts.

    Accepted forms: an ``r2://`` URI or existing local path (used as-is), or
    a bare checkpoint name like ``"best"``/``"ep12_mse_3.1200.ckpt"`` — tried
    first under ``<results_root>/<name>/``, else mapped onto the artifact
    store's ``<prefix>/<name>/checkpoints/<file>`` convention (bucket/prefix
    from the composed ``cfg.artifacts``, the same source training uses).
    """
    ckpt = str(ckpt)
    if ckpt.startswith("r2://"):
        return ckpt
    if Path(ckpt).exists():
        return ckpt
    filename = ckpt if "." in Path(ckpt).name else f"{ckpt}.ckpt"
    local = Path(cfg.results_root) / name / filename
    if local.is_file():
        return str(local)
    prefix = str(cfg.artifacts.prefix).strip("/")
    return f"r2://{cfg.artifacts.bucket}/{prefix}/{name}/checkpoints/{filename}"


def load(name: str, ckpt: str = "best", device: str | torch.device = "cpu") -> FrameModel:
    """Load experiment ``name``'s model as a ready-to-call :class:`FrameModel`.

    ``ckpt`` selects the checkpoint (see :func:`_checkpoint_ref`); the default
    ``"best"`` picks ``best.ckpt`` — locally when present, else from R2.
    """
    from hydra import compose, initialize_config_dir

    from training.config import build_task_and_codec, instantiate_model, register_configs

    register_configs()
    with initialize_config_dir(config_dir=str(REPO_ROOT / "conf"), version_base=None):
        cfg = compose(config_name="config", overrides=[f"experiment={name}"])

    task, codec = build_task_and_codec(cfg.model)
    model = instantiate_model(cfg.model)

    ref = _checkpoint_ref(name, ckpt, cfg)
    local = resolve_checkpoint_uri(ref, REPO_ROOT / ".cache" / "r2_checkpoints")
    state = torch.load(local, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.to(torch.device(device)).eval()
    return FrameModel(model, codec, task, experiment=name, device=device)
