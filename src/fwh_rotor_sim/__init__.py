"""
FWH Rotor Acoustic Simulator

A PyTorch-based acoustic field simulator for rotors using the
Ffowcs-Williams Hawkings equation with Farassat 1A formulation.
"""

from .bemt import BEMTAerodynamics, ThinAirfoilPolar
from .fwh import Farassat1ASolver
from .geometry import Blade, Rotor
from .solver import FWHRotorSolver

__all__ = [
    "Blade",
    "Rotor",
    "BEMTAerodynamics",
    "ThinAirfoilPolar",
    "Farassat1ASolver",
    "FWHRotorSolver",
]
