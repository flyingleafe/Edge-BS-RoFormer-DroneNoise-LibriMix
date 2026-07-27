"""Vendored flat-KLA (kla-loglinear@11e5a39, private repo) + the FKLA RPS model.

See ``torch_op.py``/``layer.py`` headers for the vendoring provenance and
``model.py`` for the cross-implementation ablation rationale.
"""

from models.fkla.layer import FlatKLABlock, FlatKLALayer
from models.fkla.model import FKLARPSModel, TemporalFKLAHead
from models.fkla.torch_op import flat_kla

__all__ = [
    "FKLARPSModel",
    "FlatKLABlock",
    "FlatKLALayer",
    "TemporalFKLAHead",
    "flat_kla",
]
