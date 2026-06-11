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
                         synthesis from RPS (the "sinusoidal modelling" half)
- `filtered_noise`     : `FilteredNoiseSynth` + `RPSFilterNet` — filtered-noise
                         residual generator (the "frequency filter" half), plus
                         `DroneNoisePlusFilterGen` which combines both.
- `harmonic_transform` : `HarmonicTransformModule` (VP-transform for analysis/synthesis)
- `losses`             : `MultiScaleSTFT` (DDSP-style multi-scale spectral loss)
"""

from .filtered_noise import DroneNoisePlusFilterGen, FilteredNoiseSynth, RPSFilterNet
from .harmonic_noise_gen import DroneNoiseGen, PropellerNoiseGen
from .losses import MultiScaleSTFT

__all__ = [
    "PropellerNoiseGen",
    "DroneNoiseGen",
    "FilteredNoiseSynth",
    "RPSFilterNet",
    "DroneNoisePlusFilterGen",
    "MultiScaleSTFT",
]
