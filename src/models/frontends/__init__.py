"""
Pluggable spectral front-ends.

Each front-end transforms raw audio into a (B, C, F, T) feature tensor.
Models receive the tensor and no longer own the transform — swapping the
front-end measures how the representation alone affects performance.

Contract
--------
.. code:: python

    class SpectralFrontEnd(nn.Module):
        out_channels: int                     # C dimension of forward() output
        def forward(self, audio: Tensor) -> Tensor:
            # (B, N) → (B, C, F, T)
        def num_frames(self, n_samples: int) -> int:
            # time-grid length for a given input length
"""

import torch.nn as nn

# ── Base class ───────────────────────────────────────────────────────────────


class SpectralFrontEnd(nn.Module):
    """Abstract base for time-frequency feature extractors.

    Subclasses set ``out_channels`` and implement ``forward`` and
    ``num_frames``.  The forward is expected to be deterministic (no
    dropout / learned parameters in the frontend itself).
    """

    out_channels: int

    def num_frames(self, n_samples: int) -> int:
        """Number of output time frames for an audio segment of length *n_samples*."""
        raise NotImplementedError


# ── Registry ─────────────────────────────────────────────────────────────────

FRONTEND_REGISTRY: dict[str, type] = {}


def register_frontend(cls: type) -> type:
    """Decorator: register a SpectralFrontEnd subclass under ``cls.key``."""
    if not issubclass(cls, SpectralFrontEnd):
        raise TypeError(f"{cls.__name__} must be a SpectralFrontEnd subclass")
    key = getattr(cls, "key", None)
    if key is None:
        key = cls.__name__
    FRONTEND_REGISTRY[key] = cls
    return cls


def build_frontend(name: str, **kwargs) -> SpectralFrontEnd:
    """Build a front-end by registered key.

    Parameters
    ----------
    name : str
        Key in ``FRONTEND_REGISTRY`` (``"stft_mag"``, ``"hcqt"``, …).
    **kwargs
        Passed to the constructor.
    """
    _ensure_imported()
    if name not in FRONTEND_REGISTRY:
        raise ValueError(f"Unknown frontend {name!r}.  Available: {sorted(FRONTEND_REGISTRY)}")
    cls = FRONTEND_REGISTRY[name]
    return cls(**kwargs)


_IMPORTED = False


def _ensure_imported():
    """Lazy-import front-end modules so the registry is populated."""
    global _IMPORTED
    if _IMPORTED:
        return
    from . import comb, hcqt, stft  # noqa: F401

    _IMPORTED = True
