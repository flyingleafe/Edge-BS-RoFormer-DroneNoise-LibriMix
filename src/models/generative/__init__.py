"""
Generative models for drone propeller noise from rotor speed (RPS) telemetry.

Ported from the legacy `drone_audition` project. Originally designed for 44.1 kHz
audio; here the default sample rate is 16 kHz to match this repo's pipelines
(DREGON-LM, etc.). Sample rate is a constructor argument on every module — no
implicit dependency on global settings.

Modules
-------
- `dsp`                : oscillator banks, frequency filtering, harmonic synthesis
- `math_utils`         : math helpers (hz<->midi, exp_sigmoid, overlap_and_add, ...)
- `harmonic_noise_gen` : `PropellerNoiseGen`, `DroneNoiseGen` — sinusoidal harmonic
                         synthesis from RPS (the "sinusoidal modelling" half);
                         `PolynomialRegression` / `PolyWithExpLog` gain regressors
- `filtered_noise`     : `FilteredNoiseSynth` + `RPSFilterNet` — filtered-noise
                         residual generator (the "frequency filter" half), plus
                         `DroneNoisePlusFilterGen` which combines both.
- `harmonic_transform` : VP-transform analysis/synthesis (`VP_transform`,
                         `lstsq_VP_transform`, `HarmonicTransformModule`, ...)
- `harmonic_gen_new`   : NN noise modellers (`HarmonicNoiseGenNew`,
                         `JointAmplitudePredictor`, `ConstantAmplitudePredictor`, ...)
- `nn`                 : shared building blocks (`CausalConv1d`, `ResNet`, ...)
- `losses`             : `MultiScaleSTFT` (DDSP-style multi-scale spectral loss)
                         + `smoothness_penalty` (2nd-difference control-curve regulariser)
"""

from .filtered_noise import DroneNoisePlusFilterGen, FilteredNoiseSynth, RPSFilterNet
from .harmonic_gen_new import (
    ConstantAmplitudePredictor,
    DirectionalOutputHead,
    HarmonicNoiseGenNew,
    JointAmplitudePredictor,
    LearnableTimeShift,
    SpeedsPostprocessingWrapper,
)
from .harmonic_noise_gen import (
    DroneNoiseGen,
    PolynomialRegression,
    PolyWithExpLog,
    PropellerNoiseGen,
)
from .harmonic_transform import (
    HarmonicTransformModule,
    VP_transform,
    harmonic_VP_transform,
    inverse_VP_transform,
    lstsq_VP_transform,
)
from .losses import MultiScaleSTFT, smoothness_penalty
from .positional_harmonic_gen import (
    PositionalHarmonicNoiseGen,
    fractional_delay,
    propagate,
)

__all__ = [
    "PropellerNoiseGen",
    "DroneNoiseGen",
    "PolynomialRegression",
    "PolyWithExpLog",
    "FilteredNoiseSynth",
    "RPSFilterNet",
    "DroneNoisePlusFilterGen",
    "HarmonicTransformModule",
    "VP_transform",
    "lstsq_VP_transform",
    "inverse_VP_transform",
    "harmonic_VP_transform",
    "HarmonicNoiseGenNew",
    "JointAmplitudePredictor",
    "ConstantAmplitudePredictor",
    "DirectionalOutputHead",
    "LearnableTimeShift",
    "SpeedsPostprocessingWrapper",
    "MultiScaleSTFT",
    "smoothness_penalty",
    "PositionalHarmonicNoiseGen",
    "fractional_delay",
    "propagate",
]
