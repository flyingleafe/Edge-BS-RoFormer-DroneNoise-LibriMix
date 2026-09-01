"""Validation metrics for the per-rotor salience layers.

Two numbers, and the split between them is deliberate.

``bce`` mirrors :class:`losses.LayerPITSalienceBCELoss` exactly, so
``optim.monitor: bce`` selects checkpoints on the training objective itself
rather than on a second, misaligned scalar.

``rps_mae`` is the same object in rev/s: each layer's peak, refined by the
three-point log-parabolic fit, PIT-aligned to the target. It reads the layers
the way `models.salience_crf` proves is exact — the log of a Gaussian is a
parabola, so three consecutive bins locate the vertex with no residue — and it
is NOT the deployed decoder, which is a CRF best path per layer
(`models.harmonic_ports.layer_readout`). The difference is temporal robustness
only: on a correct layer the two agree, and the CRF additionally survives a
frame whose own peak is wrong. The CRF is not used here because it costs about
15 s per clip on CPU, and metrics run per sample on CPU.
"""

from __future__ import annotations

from itertools import permutations

import numpy as np
import tdseries as td
import torch
import torch.nn.functional as F

from framespec import FrameSpec, SeriesSpec
from losses.salience_layers import layer_pit_bce
from metrics._common import get_array
from models.multif0.utils import linear_freq_grid
from models.salience_crf import gaussian_layer_target

__all__ = ["LayerPITSalienceBCEMetric", "LayerPeakRPSMetric", "peak_readout"]


def peak_readout(scores: torch.Tensor, grid: np.ndarray) -> torch.Tensor:
    """``(B, R, G, T)`` LOG-domain layers -> ``(B, R, T)`` rev/s.

    Argmax along the rate axis, then the three-point log-parabolic vertex. On a
    Gaussian layer this is exact; it is the readout without the CRF's temporal
    model.
    """
    g = torch.as_tensor(np.asarray(grid), dtype=torch.float64, device=scores.device)
    step = float(g[1] - g[0])
    n_g = g.numel()
    s = scores.double()
    i0 = s.argmax(dim=2).clamp(1, n_g - 2)  # (B, R, T)
    a = s.gather(2, (i0 - 1).unsqueeze(2)).squeeze(2)
    c0 = s.gather(2, i0.unsqueeze(2)).squeeze(2)
    c = s.gather(2, (i0 + 1).unsqueeze(2)).squeeze(2)
    den = a - 2 * c0 + c
    delta = torch.where(den.abs() < 1e-300, torch.zeros_like(den), 0.5 * (a - c) / den)
    return g[i0] + delta.clamp(-1, 1) * step


def _pit_mae(pred: torch.Tensor, tgt: torch.Tensor) -> float:
    """Best-assignment mean absolute error over ``(R, T)`` pairs, rev/s."""
    r = pred.shape[0]
    cost = (pred[:, None] - tgt[None, :]).abs().mean(dim=-1)  # (R_pred, R_tgt)
    best = min(float(sum(cost[i, p[i]] for i in range(r))) for p in permutations(range(r)))
    return best / r


class _LayerGridMetric:
    """Shared grid/spec plumbing for the two layer metrics."""

    name = "bce"

    def __init__(
        self,
        *,
        out_fmin: float = 0.0,
        out_fmax: float = 150.0,
        out_bins: int = 300,
        n_layers: int = 4,
        sigma_bins: float = 1.0,
        focus: float = 0.0,
        rate: tuple[int, int] | None = None,
        pred_key: str = "salience",
        target_key: str = "rps",
    ) -> None:
        self._freqs = linear_freq_grid(out_fmin, out_fmax, out_bins)
        self.n_layers = int(n_layers)
        self.sigma_bins = float(sigma_bins)
        self.focus = float(focus)
        self.pred_key = pred_key
        self.target_key = target_key
        self.requires_pred = FrameSpec(
            {pred_key: SeriesSpec(dims=("batch", "freq", "time"), time="grid", rate=rate)}
        )
        self.requires_target = FrameSpec(
            {target_key: SeriesSpec(dims=("batch", "rotor", "time"), time="grid", rate=None)}
        )

    def _unpack(self, pred: td.Frame, target: td.Frame) -> tuple[torch.Tensor, torch.Tensor]:
        """One sample -> ``(1, R, G, T)`` layers and ``(1, R, T)`` rev/s on that grid."""
        logits = torch.as_tensor(get_array(pred, self.pred_key)).unsqueeze(0)  # (1, R*G, T)
        rps = torch.as_tensor(get_array(target, self.target_key)).float().unsqueeze(0)
        b, fg, n_t = logits.shape
        r, n_g = self.n_layers, len(self._freqs)
        if fg != r * n_g:
            raise ValueError(f"model emits {fg} bins; {r} layers x {n_g} rate bins configured")
        layers = logits.reshape(b, r, n_g, n_t)
        rps_grid = F.interpolate(rps, size=n_t, mode="linear", align_corners=False)
        return layers, rps_grid


class LayerPITSalienceBCEMetric(_LayerGridMetric):
    """Per-sample twin of :class:`losses.LayerPITSalienceBCELoss`."""

    name = "bce"

    def __call__(self, pred: td.Frame, target: td.Frame) -> float:
        layers, rps_grid = self._unpack(pred, target)
        tgt = gaussian_layer_target(rps_grid, self._freqs, sigma_bins=self.sigma_bins)
        return float(layer_pit_bce(layers, tgt, focus=self.focus).item())


class LayerPeakRPSMetric(_LayerGridMetric):
    """PIT mean absolute error in rev/s, read off the layers by peak + parabola."""

    name = "rps_mae"

    def __call__(self, pred: td.Frame, target: td.Frame) -> float:
        layers, rps_grid = self._unpack(pred, target)
        speeds = peak_readout(F.logsigmoid(layers), self._freqs)  # (1, R, T)
        return _pit_mae(speeds[0], rps_grid[0].double())
