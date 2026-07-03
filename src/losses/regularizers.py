"""Temporal/spectral smoothness regularisers — the ONE implementation.

Ported from ``src/models/generative/losses.py`` (``smoothness_penalty``,
``_second_difference``), which is the general form of the inline copy in
``train_rps_predictor.py`` (second-order finite difference over the time axis
of a predicted RPS trajectory). The two are numerically identical when called
as ``smoothness_penalty(rps_pred, dims=(-1,))`` — the only behavioural
difference is that this implementation additionally guards axes shorter than
3 samples (contributing 0 instead of raising), which the inline
``train_rps_predictor.py`` copy did not need because RPS sequences are always
long enough.
"""

from __future__ import annotations

import tdseries as td
import torch

from losses._common import get_tensor, rps_series_spec
from tasks.spec import FrameSpec, SeriesSpec, TimeKind

# ─── Pure tensor functions ───────────────────────────────────────────────────


def _second_difference(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Discrete 2nd difference ``x[i+1] - 2 x[i] + x[i-1]`` along ``dim``."""
    n = x.size(dim)
    a = x.narrow(dim, 0, n - 2)
    b = x.narrow(dim, 1, n - 2)
    c = x.narrow(dim, 2, n - 2)
    return c - 2.0 * b + a


def smoothness_penalty(x: torch.Tensor, dims: tuple[int, ...]) -> torch.Tensor:
    """Mean squared 2nd-difference of ``x`` summed over ``dims``.

    Penalises curvature of a control curve so the network prefers
    slowly-varying trajectories. Used on predicted RPS tracks (time only),
    and on generative models' harmonic amplitudes (time) and diffuse-noise
    filter shape (time *and* frequency).

    Args:
        x: any tensor; the penalty is applied independently along each axis
            in ``dims`` and summed. Axes with fewer than 3 elements
            contribute 0 (no 2nd difference is defined).
        dims: axes to smooth over (e.g. ``(-1,)`` for time only, ``(-2, -1)``
            for frequency *and* time).

    Returns:
        Scalar tensor (mean over all elements, so it is invariant to tensor
        size and can be weighted directly against another loss term).
    """
    total = x.new_zeros(())
    for dim in dims:
        if x.size(dim) >= 3:
            total = total + _second_difference(x, dim).pow(2).mean()
    return total


# ─── Frame adapter ────────────────────────────────────────────────────────────


class SmoothnessPenalty:
    """Frame adapter around :func:`smoothness_penalty`.

    Applies the penalty to ``pred[entry]`` (default ``"rps_pred"``, time
    axis only — ``dims=(-1,)``, matching the ``train_rps_predictor.py``
    inline usage). ``target`` is unused (this is a pure regulariser on the
    prediction) but kept in the signature for Loss-protocol conformance.

    ``series_dims``/``series_time`` let this same adapter target a
    differently-shaped pred entry — e.g. E3's noise-generation smoothness
    regularisers act on
    :class:`~models.generative.PositionalHarmonicNoiseGen`'s internal
    control curves (``harm_amps`` ``(batch, rotor, O, H, t)``, ``noise_amps``
    ``(batch, rotor, F, t)``, exposed via
    ``tasks.codecs.NoiseGenerationCodec(return_dict=True)``), not an
    ``(batch, rotor, time)`` RPS trajectory. Default (``None``) keeps the
    original RPS-shaped spec (``rps_series_spec(rate)``) for backward
    compatibility with the RPS-prediction smoothness usage.
    """

    def __init__(
        self,
        *,
        entry: str = "rps_pred",
        dims: tuple[int, ...] = (-1,),
        rate: tuple[int, int] | None = None,
        series_dims: tuple[str | None, ...] | None = None,
        series_time: TimeKind | None = "grid",
    ) -> None:
        self.entry = entry
        self.dims = dims
        if series_dims is None:
            spec = rps_series_spec(rate)
        else:
            spec = SeriesSpec(dims=series_dims, time=series_time, rate=rate)
        self.requires_pred = FrameSpec({entry: spec})
        self.requires_target = FrameSpec({})

    def __call__(self, pred: td.Frame, target: td.Frame) -> torch.Tensor:
        del target
        x = get_tensor(pred, self.entry)
        return smoothness_penalty(x, self.dims)


__all__ = ["smoothness_penalty", "SmoothnessPenalty"]
