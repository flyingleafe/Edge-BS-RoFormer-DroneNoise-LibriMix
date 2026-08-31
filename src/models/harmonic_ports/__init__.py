"""Multi-pitch architectures ported to rotor-rate salience on a LINEAR STFT.

Every model in this package is one paper architecture with a single organ
replaced: the log-frequency harmonic SHIFT becomes an explicit GATHER at
``k * r``. `docs/harmonic-ports-design.md` carries the measurements behind that
substitution; do not re-derive them here.
"""

from models.harmonic_ports.harmof0_rps import HarmoF0RPS
from models.harmonic_ports.hft_rps import HFTRPS
from models.harmonic_ports.hppnet_rps import HPPNetRPS

__all__ = ["HarmoF0RPS", "HFTRPS", "HPPNetRPS"]
