"""PIT-aware RPS-prediction metrics — the ONE implementation.

Ported from the inline metric computation in ``train_rps_predictor.py``
(``evaluate()``: ``mse``/``std_mse``/``mae_frame``/``mae_clip``/``r2``/``r2_median``)
and generalised to reuse the Hungarian-assignment alignment already used for
*plotting* in ``src/losses/pit.py`` (``align_rps_to_gt``), per
docs/refactor-unified-framework.md § "Metrics": "PIT alignment reuses
align_rps_to_gt". This kills two independent PIT-search implementations
(the brute-force 24-permutation loop in ``train_rps_predictor.py`` and
whatever a given evaluation script did) in favour of one shared, per-sample
Hungarian match — the same one that keeps overlay plots and metrics
consistent.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import tdseries as td

from framespec import FrameSpec
from losses.pit import align_rps_to_gt
from metrics._common import Metric, get_array, rps_series_spec

# ─── Pure numpy functions ────────────────────────────────────────────────────


def _align(pred: np.ndarray, target: np.ndarray, pit: bool) -> np.ndarray:
    """Reorder ``target``'s rotor rows to best match ``pred`` (see
    :func:`losses.pit.align_rps_to_gt`), or return ``target``
    unchanged when ``pit=False``.

    Note the direction: ``align_rps_to_gt(pred, gt)`` permutes *pred* to
    match *gt*; since MSE/MAE/R² are symmetric under a joint permutation of
    both arrays, aligning either one to the other gives the same score, and
    aligning the (fixed-order) target to the (permutation-invariant)
    prediction is what ``train_rps_predictor.py::evaluate`` did.
    """
    if not pit:
        return target
    if not np.isfinite(pred).all():
        # A non-finite prediction would crash the Hungarian assignment inside
        # align_rps_to_gt and kill the whole run mid-validation. Any alignment
        # is as good as another for a NaN/inf prediction — the metric comes
        # out nan either way; skip the assignment and let it.
        return target
    return align_rps_to_gt(target, pred)


def rps_mse(pred: np.ndarray, target: np.ndarray, pit: bool = True) -> float:
    """Mean squared error over all rotors/frames, optionally PIT-aligned.

    Args:
        pred: (R, T) predicted RPS.
        target: (R, T) ground-truth RPS.
        pit: if True, permute ``target``'s rotors to best match ``pred``
            (:func:`losses.pit.align_rps_to_gt`) before scoring.
    """
    target = _align(pred, target, pit)
    return float(np.mean((pred - target) ** 2))


def rps_rmse(pred: np.ndarray, target: np.ndarray, pit: bool = True) -> float:
    """Root mean squared error — see :func:`rps_mse`."""
    return float(np.sqrt(rps_mse(pred, target, pit=pit)))


def rps_mae_frame(pred: np.ndarray, target: np.ndarray, pit: bool = True) -> float:
    """Per-frame mean absolute error over all rotors/frames — see :func:`rps_mse`."""
    target = _align(pred, target, pit)
    return float(np.mean(np.abs(pred - target)))


def rps_mae_clip(pred: np.ndarray, target: np.ndarray, pit: bool = True) -> float:
    """Per-clip mean absolute error: rotors are time-averaged before comparing."""
    target = _align(pred, target, pit)
    clip_pred = pred.mean(axis=-1)
    clip_target = target.mean(axis=-1)
    return float(np.mean(np.abs(clip_pred - clip_target)))


def rps_r2(pred: np.ndarray, target: np.ndarray, pit: bool = True) -> float:
    """R² using the *sample's own mean* as baseline (macro, within-sample).

    Measures within-sample temporal tracking quality without inflating the
    metric with between-sample RPS variance (ported verbatim from
    ``train_rps_predictor.py::evaluate``'s per-sample R² definition — note
    that function then averaged this across a *batch* of samples; here it is
    computed for one sample, and :class:`~metrics.suite.MetricSuite`
    aggregation (mean/median across samples) reproduces that behaviour).
    """
    target = _align(pred, target, pit)
    ss_res = float(np.sum((pred - target) ** 2))
    ss_tot = float(np.sum((target - target.mean()) ** 2))
    if ss_tot <= 1e-6:
        return float("nan")
    return 1.0 - ss_res / ss_tot


_STATS: dict[str, Callable[[np.ndarray, np.ndarray, bool], float]] = {
    "mse": rps_mse,
    "rmse": rps_rmse,
    "mae_frame": rps_mae_frame,
    "mae_clip": rps_mae_clip,
    "r2": rps_r2,
}


# ─── Frame adapter ────────────────────────────────────────────────────────────


class RPSMetric:
    """Frame adapter over one of the :data:`_STATS` PIT-aware RPS statistics.

    Compares ``pred[pred_key]`` (default ``"rps_pred"``) against
    ``target[target_key]`` (default ``"rps"``); both per-sample, i.e. shape
    ``(rotor, time)`` (no batch axis — see :class:`~metrics.suite.MetricSuite`,
    which evaluates one sample Frame pair at a time).
    """

    def __init__(
        self,
        stat: str,
        *,
        rate: tuple[int, int] | None = None,
        pit: bool = True,
        pred_key: str = "rps_pred",
        target_key: str = "rps",
    ) -> None:
        if stat not in _STATS:
            raise ValueError(f"Unknown RPS stat {stat!r}; choose one of {sorted(_STATS)}")
        self.stat = stat
        self._fn = _STATS[stat]
        self.pit = pit
        self.pred_key = pred_key
        self.target_key = target_key
        self.name = stat
        spec = rps_series_spec(rate)
        self.requires_pred = FrameSpec({pred_key: spec})
        self.requires_target = FrameSpec({target_key: spec})

    def __call__(self, pred: td.Frame, target: td.Frame) -> float:
        est = get_array(pred, self.pred_key)
        tgt = get_array(target, self.target_key)
        return self._fn(est, tgt, self.pit)


def rps_metric_suite(*, rate: tuple[int, int] | None = None, pit: bool = True) -> dict[str, Metric]:
    """Convenience: one :class:`RPSMetric` per stat, keyed by name — the
    default set for :class:`~metrics.suite.MetricSuite` RPS-prediction eval."""
    return {stat: RPSMetric(stat, rate=rate, pit=pit) for stat in _STATS}


__all__ = [
    "rps_mse",
    "rps_rmse",
    "rps_mae_frame",
    "rps_mae_clip",
    "rps_r2",
    "RPSMetric",
    "rps_metric_suite",
]
