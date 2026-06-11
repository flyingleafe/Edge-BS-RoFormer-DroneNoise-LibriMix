"""
Multi-F0 Estimation in Vocal Ensembles using Convolutional Neural Networks
===========================================================================

PyTorch reimplementation of:

    H. Cuesta, B. McFee, E. Gómez,
    "Multiple F0 Estimation in Vocal Ensembles using Convolutional Neural Networks",
    ISMIR 2020.

Based on the original Keras/TensorFlow implementation:
    https://github.com/helenacuesta/multif0-estimation-polyvocals

Provides:
    - HCQT (Harmonic Constant-Q Transform) computation
    - Multi-F0 CNN models (Early/Shallow, Early/Deep, Late/Deep)
    - Post-processing (peak picking + thresholding → multi-F0 output)
"""

from .hcqt import (
    HCQT,
    compute_hcqt,
    compute_hcqt_mag_phase,
    freq_grid,
    hcqt_params,
    time_grid,
)
from .model import (
    EarlyDeep,
    EarlyShallow,
    LateDeep,
    LateDeepNoPhase,
    MultiF0Estimator,
)
from .rps_predictor import MultiF0RPSPredictor

__all__ = [
    # HCQT
    "HCQT",
    "hcqt_params",
    "compute_hcqt",
    "compute_hcqt_mag_phase",
    "freq_grid",
    "time_grid",
    # Models
    "MultiF0Estimator",
    "EarlyShallow",
    "EarlyDeep",
    "LateDeep",
    "LateDeepNoPhase",
    # RPS adaptation
    "MultiF0RPSPredictor",
]
