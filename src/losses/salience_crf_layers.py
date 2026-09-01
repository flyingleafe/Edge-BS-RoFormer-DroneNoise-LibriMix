"""Structured CRF loss over per-rotor salience layers — training through the decoder.

WHY THIS EXISTS. The ports fit per-frame BCE and deploy a Viterbi path
(`models.harmonic_ports.layer_readout.LayerCRFReadout`). Those are different
objects, and the campaign has already paid for the gap once: the trained head's
advantage over the untrained one did not survive the switch from an argmax
decoder to a Viterbi decoder (`close` 0.459 -> 0.669). BCE never sees the
transition model, so nothing in training rewards a trajectory the decoder can
actually follow; it rewards each frame separately and hopes.

The CRF negative log-likelihood is the same decoder's own objective:

    loss  =  log Z  -  score(gold path),      score(p) = sum_t S(p_t, t) - sum_t pen(dp)

with `S` and `pen` the emissions and hinge the decoder uses. Minimizing it
maximizes the probability of the TRUE trajectory under exactly the model
`comb_crf.viterbi` maximizes over, so selection, training and deployment become
one object. `models.comb_crf` supplies both halves and its test locks `viterbi`
to the classical `tracking.comb_seed._viterbi_ridge` index-for-index, so the
transition structure cannot drift between the loss and the decoder.

NOT CTC. CTC marginalizes an unknown alignment between a short label sequence
and many frames. Here the labels are dense — there is a true rate in every
frame — so that alignment does not exist and CTC's blank/collapse machinery
would add a degree of freedom the task does not have.

EMISSIONS ARE `logsigmoid`, NOT THE RAW LOGIT, for the same reason the readout
is: the decoder scores `log sigmoid(z)`, so that is what the loss must score.

COST. `log_partition` does not depend on the gold path, so the R x R assignment
matrix needs only R forward passes plus R^2 cheap gathers — not R^2 forward
passes. The forward algorithm is a python loop over T that stores a
`(B, 2*span+1, G)` tensor per step for backward, so memory is linear in
`T * span * G` and the transition band is the knob that matters. See
`max_step_rev_s` below.
"""

from __future__ import annotations

from itertools import permutations

import numpy as np
import tdseries as td
import torch
import torch.nn.functional as F

from framespec import FrameSpec, SeriesSpec
from losses._common import get_tensor
from models import comb_crf
from models.multif0.utils import linear_freq_grid
from models.salience_crf import band_for_rev_s

__all__ = ["gold_indices", "layer_pit_crf_nll", "LayerPITCRFLoss"]


def gold_indices(rps: torch.Tensor, grid: np.ndarray) -> torch.Tensor:
    """``(B, R, T)`` rev/s -> ``(B, R, T)`` nearest grid indices.

    A stopped rotor lands on index 0, which is a value the CRF can score. It is
    NOT dropped: the shared-map target's ``active = rps_grid > 0.1`` is what
    encoded "stopped" as an absence and forced a decode threshold.
    """
    g0, step = float(grid[0]), float(grid[1] - grid[0])
    idx = torch.round((rps - g0) / step)
    return idx.clamp(0, len(grid) - 1).long()


def layer_pit_crf_nll(
    scores: torch.Tensor, gold: torch.Tensor, span: int, pen: torch.Tensor
) -> torch.Tensor:
    """Permutation-invariant CRF NLL. ``(B, R, G, T)``, ``(B, R, T)`` -> scalar.

    Returns the mean over the batch of the best assignment's NLL PER ROTOR PER
    FRAME, so the number does not move when the clip length or the rotor count
    does.
    """
    b, r, _g, t = scores.shape
    if gold.shape != (b, r, t):
        raise ValueError(f"gold {tuple(gold.shape)} does not match scores {(b, r, t)}")
    pen = pen.to(device=scores.device, dtype=scores.dtype)

    # log Z is path-independent: R forward passes serve all R^2 pairs.
    logz = [comb_crf.log_partition(scores[:, i], span, pen) for i in range(r)]
    cost = torch.stack(
        [
            torch.stack(
                [
                    logz[i] - comb_crf.path_score(scores[:, i], span, pen, gold[:, j])
                    for j in range(r)
                ],
                dim=1,
            )
            for i in range(r)
        ],
        dim=1,
    )  # (B, R_pred, R_true)

    best: torch.Tensor | None = None
    for perm in permutations(range(r)):
        tot = cost[:, 0, perm[0]]
        for i in range(1, r):
            tot = tot + cost[:, i, perm[i]]
        best = tot if best is None else torch.minimum(best, tot)
    assert best is not None
    return (best / float(r * t)).mean()


class LayerPITCRFLoss:
    """Frame adapter: per-rotor salience layers scored by the decoder's own NLL.

    Drop-in replacement for :class:`losses.LayerPITSalienceBCELoss`. It reads the
    same ``(batch, freq, time)`` wire format with the rotor layers stacked along
    the frequency axis, and it owns the same grid, because no dataset emits a
    ``salience`` entry.

    ``max_step_rev_s`` is the per-frame change the transition band makes FREE,
    and the band is truncated at four times it. It MUST match the decoder's
    (`layer_readout.MAX_STEP_REV_S`), or training and deployment part company
    again — which is the whole reason this loss exists. It is also the cost
    knob: span grows linearly in it, and the forward algorithm's memory grows
    with span.
    """

    def __init__(
        self,
        *,
        out_fmin: float = 0.0,
        out_fmax: float = 150.0,
        out_bins: int = 300,
        n_layers: int = 4,
        max_step_rev_s: float = 25.0,
        stiff: float = 40.0,
        rate: tuple[int, int] | None = None,
        pred_key: str = "salience",
        target_key: str = "rps",
    ) -> None:
        self._freqs = linear_freq_grid(out_fmin, out_fmax, out_bins)
        self.n_layers = int(n_layers)
        self.max_step_rev_s = float(max_step_rev_s)
        self.stiff = float(stiff)
        self.pred_key = pred_key
        self.target_key = target_key
        grid_step = float(np.median(np.diff(np.asarray(self._freqs, dtype=np.float64))))
        self._span, self._pen = band_for_rev_s(self.max_step_rev_s, grid_step, self.stiff)
        self.requires_pred = FrameSpec(
            {pred_key: SeriesSpec(dims=("batch", "freq", "time"), time="grid", rate=rate)}
        )
        self.requires_target = FrameSpec(
            {target_key: SeriesSpec(dims=("batch", "rotor", "time"), time="grid", rate=None)}
        )

    @property
    def span(self) -> int:
        return int(self._span)

    def scores_and_gold(
        self, logits: torch.Tensor, rps: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``(B, R*G, T)``, ``(B, R, T_any)`` -> emissions ``(B, R, G, T)`` + gold."""
        b, fg, n_grid_t = logits.shape
        r = self.n_layers
        if fg != r * len(self._freqs):
            raise ValueError(
                f"model emits {fg} output bins; {r} layers x {len(self._freqs)} "
                "rate bins were configured"
            )
        layers = logits.reshape(b, r, len(self._freqs), n_grid_t)
        rps_grid = F.interpolate(rps.float(), size=n_grid_t, mode="linear", align_corners=False)
        return F.logsigmoid(layers), gold_indices(rps_grid, self._freqs)

    def __call__(self, pred: td.Frame, target: td.Frame) -> torch.Tensor:
        logits = get_tensor(pred, self.pred_key)
        rps = get_tensor(target, self.target_key).float()
        # FLOAT32, AUTOCAST OFF. The forward algorithm accumulates a logsumexp
        # over ~2*span+1 candidates once per frame for the whole clip, so a
        # 250-frame clip is a 250-deep chain. float16 has neither the range for
        # the running total nor the precision for the differences that decide
        # the path, and its 65504 ceiling also breaks the -inf stand-in that
        # blocks out-of-grid transitions.
        with torch.autocast(device_type=logits.device.type, enabled=False):
            scores, gold = self.scores_and_gold(logits.float(), rps)
            return layer_pit_crf_nll(scores, gold, self._span, self._pen)
