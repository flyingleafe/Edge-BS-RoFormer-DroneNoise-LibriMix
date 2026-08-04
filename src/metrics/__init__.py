"""Consolidated metrics (docs/refactor-unified-framework.md § "Metrics").

Every public metric is a small class declaring ``requires_pred`` /
``requires_target`` (``framespec.FrameSpec``) and
``__call__(pred: td.Frame, target: td.Frame) -> float`` — see
:class:`metrics._common.Metric`. Pure numpy-level implementations (no Frame
dependency) are also exported for direct use/testing.
"""

from __future__ import annotations

from metrics._common import Metric
from metrics.perf import RTFMetric, compute_rtf, measure_flops, measure_inference, peak_gpu_mem_mb
from metrics.rps import (
    RPSMetric,
    rps_mae_clip,
    rps_mae_frame,
    rps_metric_suite,
    rps_mse,
    rps_r2,
    rps_rmse,
)
from metrics.salience import SalienceBCEMetric
from metrics.separation import (
    AuraMRSTFTMetric,
    AuraSTFTMetric,
    BleedlessMetric,
    ESTOIMetric,
    FullnessMetric,
    L1FreqMetric,
    NegLogWMSEMetric,
    PESQMetric,
    SDRMetric,
    SISDRMetric,
    STOIMetric,
    aura_mrstft,
    aura_stft,
    bleed_full,
    bleedless,
    estoi,
    fullness,
    l1_freq,
    neg_log_wmse,
    pesq,
    sdr,
    si_sdr,
    stoi,
)
from metrics.suite import MetricSuite, SuiteResult

__all__ = [
    "Metric",
    "MetricSuite",
    "SuiteResult",
    # separation
    "sdr",
    "si_sdr",
    "l1_freq",
    "neg_log_wmse",
    "aura_stft",
    "aura_mrstft",
    "bleed_full",
    "bleedless",
    "fullness",
    "pesq",
    "stoi",
    "estoi",
    "SDRMetric",
    "SISDRMetric",
    "L1FreqMetric",
    "NegLogWMSEMetric",
    "AuraSTFTMetric",
    "AuraMRSTFTMetric",
    "BleedlessMetric",
    "FullnessMetric",
    "PESQMetric",
    "STOIMetric",
    "ESTOIMetric",
    # rps
    "rps_mse",
    "rps_rmse",
    "rps_mae_frame",
    "rps_mae_clip",
    "rps_r2",
    "RPSMetric",
    "rps_metric_suite",
    # salience
    "SalienceBCEMetric",
    # perf
    "compute_rtf",
    "measure_inference",
    "peak_gpu_mem_mb",
    "measure_flops",
    "RTFMetric",
]
