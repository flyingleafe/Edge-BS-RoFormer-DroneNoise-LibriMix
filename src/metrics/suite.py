"""``MetricSuite``: a named collection of metrics evaluated per sample.

Generalises the ad-hoc per-metric accumulation loops in ``final_valid.py``
(``all_metrics[metric][instr].append(...)`` + ``compute_metric_avg``) and
``train_rps_predictor.py::evaluate`` into one reusable, metric-agnostic
runner: evaluate every metric on every ``(pred, target)`` sample pair,
collect per-sample rows, then aggregate (mean/median), optionally grouped by
a metadata key (e.g. ``input_snr`` — see ``compute_per_snr_summary`` in
``final_valid.py``, which this replaces).
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import tdseries as td

from data_processing.frames import get_meta
from metrics._common import Metric

Aggregation = Literal["mean", "median"]

_AGG_FN = {"mean": np.nanmean, "median": np.nanmedian}


@dataclass
class SuiteResult:
    """Per-sample rows from a :meth:`MetricSuite.evaluate` run, plus aggregation."""

    rows: list[dict[str, Any]] = field(default_factory=list)
    metric_names: list[str] = field(default_factory=list)
    group_by: str | None = None

    def aggregate(self, how: Aggregation = "mean") -> dict[str, float]:
        """Aggregate every metric across all rows (ignoring NaNs)."""
        fn = _AGG_FN[how]
        out: dict[str, float] = {}
        for name in self.metric_names:
            vals = [r[name] for r in self.rows if name in r]
            out[name] = float(fn(vals)) if vals else float("nan")
        return out

    def grouped(self, how: Aggregation = "mean") -> dict[Any, dict[str, float]]:
        """Aggregate every metric within each ``group_by`` value.

        Raises ``ValueError`` if the suite was evaluated without ``group_by``.
        """
        if self.group_by is None:
            raise ValueError("MetricSuite was not evaluated with group_by set")
        fn = _AGG_FN[how]
        groups: dict[Any, list[dict[str, Any]]] = defaultdict(list)
        for row in self.rows:
            groups[row.get(self.group_by)].append(row)

        out: dict[Any, dict[str, float]] = {}
        for key, group_rows in groups.items():
            out[key] = {}
            for name in self.metric_names:
                vals = [r[name] for r in group_rows if name in r]
                out[key][name] = float(fn(vals)) if vals else float("nan")
        return out


class MetricSuite:
    """A named collection of :class:`~metrics._common.Metric` instances.

    Each metric is evaluated independently on every sample; a metric that
    raises is not silently swallowed (unlike the try/except-per-backend
    pattern inside individual metric functions like ``pesq``/``stoi`` — those
    already return NaN on failure, which is what should propagate here).
    """

    def __init__(self, metrics: Mapping[str, Metric]) -> None:
        if not metrics:
            raise ValueError("MetricSuite requires at least one metric")
        self.metrics = dict(metrics)

    def evaluate_one(self, pred: td.Frame, target: td.Frame) -> dict[str, float]:
        """Run every metric on a single ``(pred, target)`` sample pair."""
        return {name: metric(pred, target) for name, metric in self.metrics.items()}

    def evaluate(
        self,
        samples: Iterable[tuple[td.Frame, td.Frame]],
        *,
        group_by: str | None = None,
    ) -> SuiteResult:
        """Run every metric over an iterable of sample pairs.

        Args:
            samples: iterable of ``(pred, target)`` Frame pairs (per-sample,
                no batch axis).
            group_by: optional metadata key (looked up via
                ``data_processing.frames.get_meta`` on ``target``, falling
                back to ``pred``) to attach to each row for
                :meth:`SuiteResult.grouped` — e.g. ``"input_snr"``.
        """
        rows: list[dict[str, Any]] = []
        for pred, target in samples:
            row = self.evaluate_one(pred, target)
            if group_by is not None:
                row[group_by] = get_meta(target, group_by, get_meta(pred, group_by))
            rows.append(row)
        return SuiteResult(rows=rows, metric_names=list(self.metrics), group_by=group_by)


__all__ = ["MetricSuite", "SuiteResult", "Aggregation"]
