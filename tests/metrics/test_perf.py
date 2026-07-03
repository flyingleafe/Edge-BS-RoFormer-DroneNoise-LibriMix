"""Tests for metrics.perf (RTF, FLOPs, GPU-mem measurement utilities)."""

import time

import numpy as np
import tdseries as td
import torch
from torch import nn

from data_processing.frames import with_meta
from metrics.perf import RTFMetric, compute_rtf, measure_flops, measure_inference, peak_gpu_mem_mb


def test_compute_rtf_basic():
    assert compute_rtf(elapsed_s=1.0, audio_duration_s=2.0) == 0.5
    assert compute_rtf(elapsed_s=2.0, audio_duration_s=1.0) == 2.0


def test_measure_inference_reports_elapsed_time():
    with measure_inference("cpu") as stats:
        time.sleep(0.01)
    assert stats["elapsed_s"] >= 0.01
    assert stats["peak_mem_mb"] == 0.0  # not CUDA


def test_peak_gpu_mem_mb_zero_on_cpu():
    assert peak_gpu_mem_mb("cpu") == 0.0


def test_measure_flops_linear_layer():
    model = nn.Linear(8, 4)
    x = torch.randn(1, 8)
    flops = measure_flops(model, (x,))
    assert flops >= 0.0


def test_measure_flops_returns_zero_on_failure():
    class Unprofilable(nn.Module):
        def forward(self, x):
            raise RuntimeError("boom")

    flops = measure_flops(Unprofilable(), (torch.randn(1),))
    assert flops == 0.0


def _meta_frame(entry: str, **meta: float) -> td.Frame:
    frame = td.Frame({entry: td.uniform(np.zeros(4, dtype=np.float32), 16000, dims=("time",))})
    return with_meta(frame, **meta)


def test_rtf_metric_frame_adapter():
    pred = _meta_frame("enhanced", elapsed_s=0.5)
    target = _meta_frame("target", audio_duration_s=1.0)
    metric = RTFMetric()
    assert metric(pred, target) == 0.5


def test_rtf_metric_raises_when_metadata_missing():
    pred = td.Frame({"enhanced": td.uniform(np.zeros(4, dtype=np.float32), 16000, dims=("time",))})
    target = td.Frame({"target": td.uniform(np.zeros(4, dtype=np.float32), 16000, dims=("time",))})
    metric = RTFMetric()
    try:
        metric(pred, target)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError when meta is absent")
