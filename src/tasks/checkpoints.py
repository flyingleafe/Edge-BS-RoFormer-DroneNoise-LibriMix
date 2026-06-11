"""Task-agnostic model loading — §0 of the task-separated architecture.

Usage::

    from tasks.checkpoints import load_model

    model = load_model("simple_conv@results/rps_exp/best.pt")
    # -> nn.Module, on CPU, in eval mode, ready for any task's adapter.

Current format: ``Type@ckpt`` — ``Type`` is a ``MODEL_REGISTRY`` key,
``ckpt`` is a path to a bare ``state_dict``.

Future (backward-compatible): extended checkpoints that embed class +
constructor kwargs, so ``load_model(ckpt)`` needs no ``@Type`` prefix.
"""

from __future__ import annotations

import os
from typing import Any

import torch
import torch.nn as nn

# ── Model registries ──────────────────────────────────────────────────────


# RPS prediction models (from train_rps_predictor.py).
def _make_rps_registry() -> dict[str, Any]:
    """Return {name: callable(**kw) -> nn.Module} for RPS models."""
    from train_rps_predictor import MODEL_REGISTRY as RPS_REGISTRY

    return dict(RPS_REGISTRY)


# Suppression/encoder models (from utils.get_model_from_config).
# These are config-keyed, not name-keyed; expose a thin wrapper.
def _make_suppression_registry() -> dict[str, Any]:
    """Return {name: callable(**kw) -> nn.Module} for suppression models."""
    from models import (
        MODEL_TYPES as SUPP_MODEL_TYPES,  # pyright: ignore[reportAttributeAccessIssue]
    )

    registry: dict[str, Any] = {}
    for name, cls in SUPP_MODEL_TYPES.items():
        registry[name] = cls
    return registry


# ── Core API ──────────────────────────────────────────────────────────────


def load_model(spec: str, device: str = "cpu") -> nn.Module:
    """Load a model from a spec string ``Type@ckpt``.

    Parameters
    ----------
    spec : str
        Format ``Type@/path/to/checkpoint.pt``.  ``Type`` is resolved
        against the known model registries:
        * RPS predictors  — ``MODEL_REGISTRY`` keys from
          ``train_rps_predictor`` (e.g. ``simple_conv``, ``dccrn_enc_rps``).
        * Suppression models — ``utils.get_model_from_config`` model-type
          keys (e.g. ``dcunet``, ``dccrn``).
    device : str
        Where to place the model (default ``"cpu"``).

    Returns
    -------
    nn.Module
        Loaded, moved to ``device``, in ``eval()`` mode.

    Raises
    ------
    ValueError
        If ``Type`` is unknown or the spec format is malformed.
    FileNotFoundError
        If the checkpoint path does not exist.
    """
    if "@" not in spec:
        raise ValueError(f"Invalid spec {spec!r}: expected 'Type@/path/to/ckpt.pt' (no '@' found)")

    model_type, _, ckpt_path = spec.partition("@")
    model_type = model_type.strip()
    ckpt_path = ckpt_path.strip()

    if not model_type:
        raise ValueError(f"Invalid spec {spec!r}: missing model type before '@'")
    if not ckpt_path:
        raise ValueError(f"Invalid spec {spec!r}: missing checkpoint path after '@'")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path!r}")

    # Resolve the model class.
    rps_reg = _make_rps_registry()
    if model_type in rps_reg:
        factory = rps_reg[model_type]
        model = factory(n_fft=2048, hop_length=512, num_rotors=4)
    else:
        try:
            from utils import get_model_from_config  # noqa: F401

            # For suppression models we need a fake config; this is a
            # transitional path — when extended checkpoints land, the
            # config is embedded in the checkpoint.
            raise ValueError(
                f"Suppression-model loading via Type@ckpt is not yet "
                f"supported.  RPS types: {sorted(rps_reg)}"
            )
        except ImportError:
            pass
        raise ValueError(f"Unknown model type {model_type!r}.  Known RPS types: {sorted(rps_reg)}")

    # Load state dict.
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    model.load_state_dict(ckpt)
    model.to(device)
    model.eval()
    return model
