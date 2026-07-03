"""Quantile-masked MSE loss (outlier/per-pixel loss clipping).

Ported from ``train.py::masked_loss``. The mechanism: for each element along
a leading "group" axis (in the original multi-stem separation trainer, the
instrument-stem axis), compute a per-group quantile threshold over the
remaining (non-batch) loss values and zero out the loss contributions above
it. This suppresses domination of the gradient by a few high-loss outliers
(pathological stems/pixels) rather than emphasising hard examples — despite
the training-config name ``coarse_loss_clip``, this is *robust clipping*, not
hard-example mining.
"""

from __future__ import annotations

import tdseries as td
import torch

from losses._common import AUDIO_RATE, audio_series_spec, get_tensor
from tasks.spec import FrameSpec

# ─── Pure tensor function ────────────────────────────────────────────────────


def masked_loss(y_: torch.Tensor, y: torch.Tensor, q: float, coarse: bool = True) -> torch.Tensor:
    """Quantile-masked MSE.

    Args:
        y_: predictions, shape ``(G, B, ...)`` — ``G`` is the axis the
            per-``B`` quantile is computed over (instrument stems in the
            original multi-stem trainer), ``B`` is the batch axis, ``...`` is
            any number of trailing (e.g. channel, time) axes.
        y: targets, same shape as ``y_``.
        q: quantile threshold in [0, 1]; elements with loss at or above the
            per-batch-row quantile are masked out.
        coarse: if True, first collapse the two trailing axes (assumes
            exactly two, e.g. channel+time) to a per-``(B, G)`` scalar before
            quantile masking (cheaper, coarser granularity). If False,
            quantile masking runs at full per-element granularity.

    Returns:
        Scalar loss (mean over the surviving masked elements).
    """
    loss = torch.nn.MSELoss(reduction="none")(y_, y).transpose(0, 1)
    if coarse:
        loss = torch.mean(loss, dim=(-1, -2))
    loss = loss.reshape(loss.shape[0], -1)
    ell = loss.detach()
    quantile = torch.quantile(ell, q, interpolation="linear", dim=1, keepdim=True)
    mask = quantile > ell
    return (loss * mask).mean()


# ─── Frame adapter ────────────────────────────────────────────────────────────


class MaskedLoss:
    """Frame adapter around :func:`masked_loss` for a single audio target.

    ``masked_loss``'s ``G`` axis (instrument stems in the original multi-stem
    trainer) has no equivalent in this project's single-target
    ``speech_enhancement`` task, so this adapter normalizes the entry to
    ``(G=1, B, rest...)``: per-sample mono ``(T,)`` becomes ``(1, 1, T)``,
    per-sample multichannel / batched mono ``(X, T)`` becomes ``(1, X, T)``
    (the quantile is then per channel / per sample over time), and batched
    multichannel ``(B, C, T)`` becomes ``(1, B, C, T)``. ``coarse`` is forced
    to False: with ``G = 1`` the coarse path would reduce each batch row to a
    single value whose self-quantile masks it out (strict ``>``), yielding a
    constant zero loss.
    """

    def __init__(
        self,
        *,
        q: float = 0.9,
        n_channels: int | None = None,
        sr: tuple[int, int] = AUDIO_RATE,
        pred_key: str = "enhanced",
        target_key: str = "target",
    ) -> None:
        self.q = q
        self.pred_key = pred_key
        self.target_key = target_key
        spec = audio_series_spec(n_channels, sr)
        self.requires_pred = FrameSpec({pred_key: spec})
        self.requires_target = FrameSpec({target_key: spec})

    def __call__(self, pred: td.Frame, target: td.Frame) -> torch.Tensor:
        est = get_tensor(pred, self.pred_key)
        tgt = get_tensor(target, self.target_key)
        if est.dim() == 1:  # per-sample mono (T,) -> keep time in the rest axes
            est = est.unsqueeze(0)
            tgt = tgt.unsqueeze(0)
        return masked_loss(est.unsqueeze(0), tgt.unsqueeze(0), q=self.q, coarse=False)


__all__ = ["masked_loss", "MaskedLoss"]
