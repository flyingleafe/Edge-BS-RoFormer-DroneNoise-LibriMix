"""Tests for losses.composite (CompositeLoss, the choice_loss replacement)."""

import numpy as np
import tdseries as td
import torch

from losses.composite import CompositeLoss, LossTerm
from losses.masked import MaskedLoss
from losses.regularizers import SmoothnessPenalty
from tasks.spec import FrameSpec, SeriesSpec

SR = 16000


class _ConstantLoss:
    """Tiny stub Loss for testing the combinator in isolation."""

    def __init__(self, value: float, entry: str = "enhanced"):
        self.value = value
        self.requires_pred = FrameSpec({entry: SeriesSpec(dims=("time",))})
        self.requires_target = FrameSpec({})

    def __call__(self, pred, target):
        del pred, target
        return torch.tensor(self.value)


def _mono_frame(entry: str, x: np.ndarray) -> td.Frame:
    return td.Frame({entry: td.uniform(x.astype(np.float32), SR, dims=("time",))})


def test_composite_loss_weighted_sum():
    components = {
        "a": (2.0, _ConstantLoss(3.0)),
        "b": (0.5, _ConstantLoss(10.0)),
    }
    combo = CompositeLoss(components)
    pred = _mono_frame("enhanced", np.zeros(10))
    target = _mono_frame("target", np.zeros(10))
    total = combo(pred, target)
    assert torch.isclose(total, torch.tensor(2.0 * 3.0 + 0.5 * 10.0))
    assert set(combo.last_breakdown) == {"a", "b"}
    assert torch.isclose(combo.last_breakdown["a"], torch.tensor(6.0))
    assert torch.isclose(combo.last_breakdown["b"], torch.tensor(5.0))


def test_composite_loss_requires_are_merged():
    components = {
        "a": LossTerm(1.0, _ConstantLoss(1.0, entry="enhanced")),
        "b": LossTerm(1.0, _ConstantLoss(1.0, entry="rps_pred")),
    }
    combo = CompositeLoss(components)
    assert "enhanced" in combo.requires_pred.entries
    assert "rps_pred" in combo.requires_pred.entries


def test_composite_loss_rejects_empty():
    try:
        CompositeLoss({})
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for empty components")


def test_composite_loss_mixes_module_and_plain_components():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(4000).astype(np.float32)
    ramp = np.linspace(0.0, 1.0, 40).astype(np.float32)
    rps = np.stack([ramp, ramp])

    pred = td.Frame(
        {
            "enhanced": td.uniform(x, SR, dims=("time",)),
            "rps_pred": td.uniform(rps, 100, dims=("rotor", "time")),
        }
    )
    target = td.Frame({"target": td.uniform(x, SR, dims=("time",))})

    combo = CompositeLoss(
        {
            "masked": (1.0, MaskedLoss(q=0.9)),  # plain object, no torch state
            "smooth": (0.1, SmoothnessPenalty(entry="rps_pred")),  # plain object too
        }
    )
    total = combo(pred, target)
    # masked loss is 0 (identical signal), smoothness is 0 (linear ramp) -> total 0.
    assert torch.isclose(total, torch.tensor(0.0), atol=1e-6)
