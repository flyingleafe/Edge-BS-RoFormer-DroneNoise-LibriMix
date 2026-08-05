"""Validation-time BCE-on-salience metric — a ``MetricSuite``-compatible
wrapper around the same computation as ``losses.salience.SalienceRPSBCELoss``.

``training.loop.run_training``'s ``optim.monitor`` scheduling/early-stop/
checkpoint decision reads ``val_metrics[monitor]`` (from the *validation*
``MetricSuite``) unless ``monitor == "loss"``, in which case it uses the
*training* loss instead (``training/loop.py::run_training``). A salience
model's BCE loss has no comparable RPS-space metric (no metric that runs
``predict_rps()`` exists yet — see REPLICATION.md § C7/C8), so this metric
exists purely to let ``optim.monitor: bce`` track the BCE objective on the
*validation* split, matching best practice for early stopping / LR
scheduling (rather than falling back to the train-loss ``monitor: loss``
path). See docs/refactor-unified-framework.md § "Pre-run validation".
"""

from __future__ import annotations

from typing import Literal

import tdseries as td
import torch
import torch.nn.functional as F

from framespec import FrameSpec, SeriesSpec
from losses.salience import auto_pos_weight, salience_bce_loss
from metrics._common import get_array
from models.multif0.utils import cqt_freq_grid, linear_freq_grid, salience_target_from_resampled_rps

__all__ = ["SalienceBCEMetric"]


class SalienceBCEMetric:
    """Per-sample BCE-on-RPS-derived-salience metric.

    Same grid-parameter contract as :class:`losses.salience.SalienceRPSBCELoss`
    (log-spaced CQT grid via ``fmin``/``n_octaves``/``over_sample``/``n_bins``/
    ``bins_per_octave``, or a decoupled linear grid via ``out_fmin``/
    ``out_fmax``/``out_bins``) — pass the *same* params used to build the
    matching loss so the monitored metric and the training objective agree.

    ``__call__`` operates on one *unbatched* ``(freq/rotor, time)`` sample
    pair, per :class:`~metrics.suite.MetricSuite`'s per-sample evaluation
    contract — but ``requires_pred``/``requires_target`` still declare the
    *batched* ``(batch, freq/rotor, time)`` dims, matching
    ``metrics.rps.RPSMetric``'s convention: ``training.validate.validate_config``
    checks every metric's spec against the batched task/dataset spec
    (``merge_specs(task.output_spec, batch_spec)``), never the per-sample
    shape actually seen at evaluation time.
    """

    def __init__(
        self,
        *,
        fmin: float = 32.7,
        n_octaves: int = 6,
        over_sample: int = 5,
        n_bins: int | None = None,
        bins_per_octave: int | None = None,
        out_fmin: float | None = None,
        out_fmax: float | None = None,
        out_bins: int | None = None,
        blur_bins: int = 0,
        pos_weight: float | Literal["auto"] | None = None,
        num_rotors: int = 4,
        rate: tuple[int, int] | None = None,
        pred_key: str = "salience",
        target_key: str = "rps",
    ) -> None:
        if out_fmin is not None or out_fmax is not None or out_bins is not None:
            if out_fmin is None or out_fmax is None or out_bins is None:
                raise ValueError(
                    "out_fmin/out_fmax/out_bins must all be set together "
                    "(decoupled linear output grid)"
                )
            self._freqs = linear_freq_grid(out_fmin, out_fmax, out_bins)
        else:
            self._freqs = cqt_freq_grid(
                fmin=fmin,
                n_octaves=n_octaves,
                over_sample=over_sample,
                n_bins=n_bins,
                bins_per_octave=bins_per_octave,
            )
        self.blur_bins = blur_bins
        self.pos_weight: float | None = (
            auto_pos_weight(len(self._freqs), num_rotors, blur_bins)
            if pos_weight == "auto"
            else pos_weight
        )
        self.pred_key = pred_key
        self.target_key = target_key
        self.name = "bce"
        self.requires_pred = FrameSpec(
            {pred_key: SeriesSpec(dims=("batch", "freq", "time"), time="grid", rate=rate)}
        )
        self.requires_target = FrameSpec(
            {target_key: SeriesSpec(dims=("batch", "rotor", "time"), time="grid", rate=None)}
        )

    def __call__(self, pred: td.Frame, target: td.Frame) -> float:
        logits = torch.as_tensor(get_array(pred, self.pred_key)).unsqueeze(0)  # (1, n_bins, T)
        rps = torch.as_tensor(get_array(target, self.target_key)).float().unsqueeze(0)  # (1, R, T)
        n_grid = logits.shape[-1]
        rps_grid = F.interpolate(rps, size=n_grid, mode="linear", align_corners=False)
        sal_target = salience_target_from_resampled_rps(
            rps_grid, self._freqs, blur_bins=self.blur_bins
        )
        loss = salience_bce_loss(logits, sal_target, pos_weight=self.pos_weight)
        return float(loss.item())
