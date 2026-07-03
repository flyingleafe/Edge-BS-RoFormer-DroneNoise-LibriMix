"""LoRA fine-tuning seam.

docs/refactor-unified-framework.md § "Future expansions (design headroom)":
"LoRA fine-tuning keeps a config seam (``lora.*``) in the unified trainer" —
decided but **deliberately not ported yet**. This module is that seam:
:func:`maybe_apply_lora` is the one hook the model-build path
(``training.loop.run_training``) calls right after
``training.config.instantiate_model``; today it is only ever a no-op or a
loud, actionable error.

The old implementation (``loralib``-based) lived in the pre-refactor root
``train.py``: see ``git show d94ce9f:train.py`` for ``bind_lora_to_model``
(wraps target `nn.Linear`/`nn.Conv*` modules in ``loralib`` adapters per
``config.training.lora_config``), ``lora.mark_only_lora_as_trainable(model)``
(freezes everything except the injected LoRA params), and
``lora.lora_state_dict(model)`` (used at checkpoint-save time to persist only
the adapter weights). Porting it onto the Frame/Task-typed model registry
here means re-deriving ``bind_lora_to_model``'s module-name matching against
``LoraConfig.target_modules`` and wiring ``LoraConfig.checkpoint`` (initial
adapter weights) through ``training.config.instantiate_model`` — out of
scope for this change; the config shape exists so a future PR can land the
port without touching the training loop's call site.
"""

from __future__ import annotations

from typing import Any

import torch

__all__ = ["maybe_apply_lora"]


def maybe_apply_lora(model: torch.nn.Module, lora_cfg: Any) -> torch.nn.Module:
    """Apply LoRA adapters to ``model`` per ``lora_cfg`` (``training.config.LoraConfig``).

    Returns ``model`` unchanged when ``lora_cfg.enabled`` is falsy (the
    default) — the common case, and the only one implemented today. When
    ``lora_cfg.enabled`` is true, raises :class:`NotImplementedError` with a
    pointer to the old ``loralib``-based implementation (see module
    docstring) rather than silently doing nothing.
    """
    if not getattr(lora_cfg, "enabled", False):
        return model
    raise NotImplementedError(
        "LoRA fine-tuning (lora.enabled=true) is not ported to the unified "
        "trainer yet — the config seam (training.config.LoraConfig) exists, "
        "the adapter-injection logic does not. See the old loralib-based "
        "implementation via `git show d94ce9f:train.py` "
        "(bind_lora_to_model / lora.mark_only_lora_as_trainable / "
        "lora.lora_state_dict) and training.lora's module docstring."
    )
