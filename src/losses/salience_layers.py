"""Permutation-invariant BCE over PER-ROTOR salience layers.

WHY A SECOND SALIENCE LOSS. :class:`losses.SalienceRPSBCELoss` trains one shared
map with a triangular kernel, and `models.salience_crf` measured what that pair
costs: encoding real training telemetry and decoding it again returns the
trajectory 8.24 rev/s away on average, with 39-45% of frames more than half a bin
off. The cause is representational, not statistical — DREGON's rotor pairs sit
0.13-0.86 rev/s apart, INSIDE one 0.5 rev/s bin, so a shared map has fewer peaks
than it has rotors. One layer per rotor removes the merge entirely, and a
Gaussian kernel makes the sub-bin readout exact, because the log of a Gaussian is
a parabola globally and three consecutive bins locate its vertex.

Per-rotor layers cost one thing: the layers have no canonical order, so the loss
must be permutation invariant. This is ordinary PIT — all ``R!`` assignments per
CLIP (24 at four rotors), the best one kept — the same object
:mod:`losses.pit` applies to direct RPS regression, moved onto the map.

``pos_weight`` IS 1 AND MUST STAY 1. With a soft target ``p``, weighted BCE is
minimized at ``sigmoid(z) = w p / (w p + 1 - p)``, not at ``p``. At the
``auto`` weight this grid would otherwise take (about 119 for a single Gaussian
layer) that saturates every bin within 2.2 sigma of the peak to ``sigmoid(z) ~
1``, which FLATTENS the parabola the decoder fits — the imbalance correction and
the exact readout are directly opposed. Unweighted BCE against a soft target is
minimized at ``sigmoid(z) = p`` exactly, so ``log sigmoid(z) = -d^2 / 2 sigma^2``
and the readout stays exact.

THE IMBALANCE IS HANDLED BY ``focus`` INSTEAD, and the difference is where the
weight is applied. ``pos_weight`` reweights ONE SIDE of a bin's cross-entropy
against the other, which moves that bin's minimum. ``focus`` multiplies a bin's
WHOLE cross-entropy by ``1 + focus * target``, and a positive constant does not
move a minimum: every bin still bottoms out at ``sigmoid(z) = target``, so the
readout stays exact while the peak gets the emphasis. A Gaussian layer holds
about ``sqrt(2 pi) sigma`` bins of mass in ``G``, so ``focus = G / (sqrt(2 pi)
sigma)`` splits the total weight about evenly between the peak and the rest.
"""

from __future__ import annotations

from itertools import permutations

import tdseries as td
import torch
import torch.nn.functional as F

from framespec import FrameSpec, SeriesSpec
from losses._common import get_tensor
from models.multif0.utils import linear_freq_grid
from models.salience_crf import gaussian_layer_target

__all__ = ["layer_pit_bce", "LayerPITSalienceBCELoss"]


def layer_pit_bce(
    logits: torch.Tensor,
    target: torch.Tensor,
    pos_weight: float | None = None,
    focus: float = 0.0,
) -> torch.Tensor:
    """Permutation-invariant BCE between two stacks of layers.

    Args:
        logits: ``(B, R, G, T)`` raw per-rotor salience logits.
        target: ``(B, R, G, T)`` soft per-rotor target in ``[0, 1]``.
        pos_weight: positive-class weight; ``None`` (the default) means 1, and
            see the module docstring for why anything else breaks the readout.
        focus: PER-BIN emphasis, weight ``1 + focus * target``. This is the
            class-imbalance handle that ``pos_weight`` cannot be here. It scales
            each bin's WHOLE cross-entropy by a positive constant, and a
            positive constant does not move a minimum — every bin's optimum
            stays ``sigmoid(z) = target``, so the readout stays exact. A
            Gaussian layer carries about ``sqrt(2 pi) sigma`` bins of mass out
            of ``G``, so ``focus ~ G / (sqrt(2 pi) sigma)`` puts about equal
            total weight on the peak and on everything else.

    Returns:
        Scalar: the mean over the batch of the best assignment's mean BCE.
    """
    b, r, _g, _t = logits.shape
    if target.shape != logits.shape:
        raise ValueError(
            f"shape mismatch: logits {tuple(logits.shape)} target {tuple(target.shape)}"
        )
    pw = None
    if pos_weight is not None:
        pw = torch.as_tensor(pos_weight, dtype=logits.dtype, device=logits.device)

    # cost[b, i, j] = mean BCE of predicted layer i against target layer j.
    # Built one TARGET layer at a time: the full outer product would hold
    # B*R*R*G*T floats at once, and validation clips are eight times longer
    # than training clips.
    cols = []
    for j in range(r):
        tgt = target[:, j : j + 1].expand(-1, r, -1, -1).to(logits.dtype)
        bce = F.binary_cross_entropy_with_logits(logits, tgt, pos_weight=pw, reduction="none")
        if focus:
            bce = bce * (1.0 + float(focus) * tgt)
        cols.append(bce.mean(dim=(2, 3)))  # (B, R)
    cost = torch.stack(cols, dim=2)  # (B, R, R)

    idx = torch.arange(r, device=logits.device)
    scores = torch.stack(
        [
            cost[:, idx, torch.as_tensor(p, device=logits.device)].sum(dim=1)
            for p in permutations(range(r))
        ],
        dim=1,
    )  # (B, R!)
    return (scores.min(dim=1).values / r).mean()


class LayerPITSalienceBCELoss:
    """Frame adapter: derives the per-rotor Gaussian target from ``target["rps"]``.

    Like :class:`losses.SalienceRPSBCELoss` it owns the grid, because no dataset
    emits a ``salience`` entry. It differs in three ways, all of them the reason
    it exists:

    1. The target is :func:`models.salience_crf.gaussian_layer_target` — one
       GAUSSIAN layer per rotor, sigma in bins — not one triangular-kernel map
       shared by every rotor.
    2. A stopped rotor is a peak at bin 0, not a dark column. The old target
       drops any rotor under 0.1 rev/s, which encodes "stopped" as absence of
       evidence and is what forces the decoder to carry a threshold.
    3. The loss is permutation invariant over the layers.

    The prediction arrives in the codec's ``(batch, freq, time)`` wire format
    with the rotor layers stacked along the frequency axis, so ``n_layers *
    out_bins`` must equal the model's output width.
    """

    def __init__(
        self,
        *,
        out_fmin: float = 0.0,
        out_fmax: float = 150.0,
        out_bins: int = 300,
        n_layers: int = 4,
        sigma_bins: float = 1.0,
        pos_weight: float | None = None,
        focus: float = 0.0,
        rate: tuple[int, int] | None = None,
        pred_key: str = "salience",
        target_key: str = "rps",
    ) -> None:
        self._freqs = linear_freq_grid(out_fmin, out_fmax, out_bins)
        self.n_layers = int(n_layers)
        self.sigma_bins = float(sigma_bins)
        self.pos_weight = pos_weight
        self.focus = float(focus)
        self.pred_key = pred_key
        self.target_key = target_key
        self.requires_pred = FrameSpec(
            {pred_key: SeriesSpec(dims=("batch", "freq", "time"), time="grid", rate=rate)}
        )
        self.requires_target = FrameSpec(
            {target_key: SeriesSpec(dims=("batch", "rotor", "time"), time="grid", rate=None)}
        )

    def layers_and_target(
        self, logits: torch.Tensor, rps: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``(B, R*G, T)``, ``(B, R, T_any)`` -> two ``(B, R, G, T)`` stacks."""
        b, fg, n_grid_t = logits.shape
        r = self.n_layers
        if fg != r * len(self._freqs):
            raise ValueError(
                f"model emits {fg} output bins; {r} layers x {len(self._freqs)} "
                "rate bins were configured"
            )
        layers = logits.reshape(b, r, len(self._freqs), n_grid_t)
        rps_grid = F.interpolate(rps.float(), size=n_grid_t, mode="linear", align_corners=False)
        tgt = gaussian_layer_target(rps_grid, self._freqs, sigma_bins=self.sigma_bins)
        return layers, tgt

    def __call__(self, pred: td.Frame, target: td.Frame) -> torch.Tensor:
        logits = get_tensor(pred, self.pred_key)
        rps = get_tensor(target, self.target_key).float()
        layers, tgt = self.layers_and_target(logits, rps)
        return layer_pit_bce(layers, tgt, pos_weight=self.pos_weight, focus=self.focus)
