"""Multi-pitch architectures ported to rotor-rate salience on a LINEAR STFT.

Every ``*_rps`` model in this package is one paper architecture with a single
organ replaced: the log-frequency harmonic SHIFT becomes an explicit GATHER at
``k * r``. `docs/harmonic-ports-design.md` carries the measurements behind that
substitution; do not re-derive them here.

The ``*_orig`` models are the CONTROLS for that substitution — the same two
papers with nothing replaced, on their own 352-bin log grid, wired into the same
`salience_rps` task. Read the pair, not either one alone.

THE MODELS LOAD ON FIRST USE, and the reason is an import cycle. Each model
here imports ``models.salience_rps`` for its base class, and
``models.salience_rps`` now imports ``LayerCRFReadout`` back out of this
package, because the two multi-pitch salience baselines read per-rotor layers
with the same mixin. ``layer_readout`` itself depends on nothing in this
package, so exporting it eagerly and the models lazily removes the cycle
without moving a file or changing an import path.
"""

from importlib import import_module
from typing import Any

from models.harmonic_ports.layer_readout import LayerCRFReadout, split_maps

__all__ = [
    "HarmoF0Orig",
    "HarmoF0RPS",
    "HFTRPS",
    "HPPNetOrig",
    "HPPNetRPS",
    "LayerCRFReadout",
    "split_maps",
]

_MODEL_MODULES = {
    "HarmoF0Orig": "models.harmonic_ports.harmof0_orig",
    "HarmoF0RPS": "models.harmonic_ports.harmof0_rps",
    "HFTRPS": "models.harmonic_ports.hft_rps",
    "HPPNetOrig": "models.harmonic_ports.hppnet_orig",
    "HPPNetRPS": "models.harmonic_ports.hppnet_rps",
}


def __getattr__(name: str) -> Any:
    """PEP 562 hook: ``from models.harmonic_ports import HarmoF0RPS`` still works."""
    if name in _MODEL_MODULES:
        value = getattr(import_module(_MODEL_MODULES[name]), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
