"""Weighted sum-of-losses combinator — replaces ``train.py::choice_loss``.

``choice_loss`` picked exactly one of 8 hardcoded combinations (MultiSTFT +
MSE + L1, MultiSTFT + MSE, ..., masked-loss-only) based on CLI flags.
``CompositeLoss`` generalises this to an arbitrary weighted sum of any Loss
components declared in a config — the same MultiSTFT/MSE/L1/masked pieces are
now just named entries with a weight, instead of a fixed combinatorial
menu.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import NamedTuple

import tdseries as td
import torch
from torch import nn

from losses._common import Loss
from tasks.spec import FrameSpec, merge_specs


class LossTerm(NamedTuple):
    weight: float
    loss: Loss


class CompositeLoss(nn.Module):
    """Weighted sum of named Loss components.

    ``requires_pred``/``requires_target`` are the union (``merge_specs``) of
    every component's requirements, so spec validation sees the composite's
    true data needs. After a forward pass, ``last_breakdown`` holds each
    component's *weighted* contribution (for logging).

    Components that are themselves ``nn.Module`` (e.g. spectral losses with
    STFT window buffers) are registered as submodules so ``.to(device)`` /
    ``.parameters()`` reach them; plain callables (no torch state) are held
    directly.
    """

    def __init__(self, components: Mapping[str, LossTerm | tuple[float, Loss]]) -> None:
        super().__init__()
        terms = {
            name: (c if isinstance(c, LossTerm) else LossTerm(*c)) for name, c in components.items()
        }
        if not terms:
            raise ValueError("CompositeLoss requires at least one component")
        self._order = list(terms.keys())
        self._weights = {name: term.weight for name, term in terms.items()}
        modules = {
            name: term.loss for name, term in terms.items() if isinstance(term.loss, nn.Module)
        }
        self._plain = {name: term.loss for name, term in terms.items() if name not in modules}
        self._loss_modules = nn.ModuleDict(modules)

        self.requires_pred: FrameSpec = merge_specs(
            *(term.loss.requires_pred for term in terms.values())
        )
        self.requires_target: FrameSpec = merge_specs(
            *(term.loss.requires_target for term in terms.values())
        )
        self.last_breakdown: dict[str, torch.Tensor] = {}

    def forward(self, pred: td.Frame, target: td.Frame) -> torch.Tensor:
        total: torch.Tensor | None = None
        breakdown: dict[str, torch.Tensor] = {}
        for name in self._order:
            fn = self._loss_modules[name] if name in self._loss_modules else self._plain[name]
            value = fn(pred, target) * self._weights[name]
            breakdown[name] = value
            total = value if total is None else total + value
        self.last_breakdown = breakdown
        assert total is not None
        return total


__all__ = ["LossTerm", "CompositeLoss"]
