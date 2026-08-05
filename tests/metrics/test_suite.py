"""Tests for metrics.suite (MetricSuite: per-sample evaluation + aggregation)."""

import numpy as np
import tdseries as td

from data_processing.frames import with_meta
from framespec import FrameSpec
from metrics.suite import MetricSuite


class _ConstantMetric:
    def __init__(self, value: float):
        self.value = value
        self.requires_pred = FrameSpec({})
        self.requires_target = FrameSpec({})

    def __call__(self, pred, target):
        del pred, target
        return self.value


class _EchoMetaMetric:
    """Returns the numeric value stored in target['meta']['x']."""

    def __init__(self):
        self.requires_pred = FrameSpec({})
        self.requires_target = FrameSpec({})

    def __call__(self, pred, target):
        del pred
        return float(target["meta"]["x"])


def _blank_frame() -> td.Frame:
    return td.Frame({"x": td.uniform(np.zeros(4, dtype=np.float32), 16000, dims=("time",))})


def test_metric_suite_evaluate_one():
    suite = MetricSuite({"a": _ConstantMetric(1.0), "b": _ConstantMetric(2.0)})
    row = suite.evaluate_one(_blank_frame(), _blank_frame())
    assert row == {"a": 1.0, "b": 2.0}


def test_metric_suite_aggregate_mean_and_median():
    values = [1.0, 2.0, 3.0, 100.0]
    suite = MetricSuite({"m": _EchoMetaMetric()})
    samples = [(_blank_frame(), with_meta(_blank_frame(), x=v)) for v in values]
    result = suite.evaluate(samples)

    agg_mean = result.aggregate("mean")
    agg_median = result.aggregate("median")
    assert abs(agg_mean["m"] - np.mean(values)) < 1e-9
    assert abs(agg_median["m"] - np.median(values)) < 1e-9


def test_metric_suite_aggregate_ignores_nan():
    suite = MetricSuite({"m": _EchoMetaMetric()})
    samples = [
        (_blank_frame(), with_meta(_blank_frame(), x=1.0)),
        (_blank_frame(), with_meta(_blank_frame(), x=3.0)),
    ]
    result = suite.evaluate(samples)
    # Manually poison one row with NaN, as a real metric failure might.
    result.rows[0]["m"] = float("nan")
    agg = result.aggregate("mean")
    assert agg["m"] == 3.0


def test_metric_suite_group_by():
    suite = MetricSuite({"m": _EchoMetaMetric()})
    samples = []
    for snr, val in [(-10, 1.0), (-10, 3.0), (0, 5.0), (0, 7.0)]:
        target = with_meta(_blank_frame(), x=val, input_snr=snr)
        samples.append((_blank_frame(), target))

    result = suite.evaluate(samples, group_by="input_snr")
    grouped = result.grouped("mean")
    assert grouped[-10]["m"] == 2.0
    assert grouped[0]["m"] == 6.0


def test_metric_suite_grouped_raises_without_group_by():
    suite = MetricSuite({"m": _ConstantMetric(1.0)})
    result = suite.evaluate([(_blank_frame(), _blank_frame())])
    try:
        result.grouped()
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError when group_by was not set")


def test_metric_suite_rejects_empty():
    try:
        MetricSuite({})
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for empty metrics dict")
