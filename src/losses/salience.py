"""BCE-on-salience loss with positive-class weighting.

Ported from the inline training step for salience-map RPS baselines
(``multif0_salience`` / ``basic_pitch_salience``) in ``train_rps_predictor.py``:
``F.binary_cross_entropy_with_logits(logits, sal_target, pos_weight=...)`` plus
the "auto" ``pos_weight`` heuristic (roughly #negative-bins / #positive-bins
per frame).
"""

from __future__ import annotations

from typing import Literal

import tdseries as td
import torch
import torch.nn.functional as F

from framespec import FrameSpec, SeriesSpec
from losses._common import get_tensor
from models.multif0.utils import cqt_freq_grid, linear_freq_grid, salience_target_from_resampled_rps

# ─── Pure tensor functions ───────────────────────────────────────────────────


def auto_pos_weight(n_bins: int, num_rotors: int, blur_bins: int) -> float:
    """Heuristic positive-class weight for salience BCE.

    ``active`` is the number of positive bins per frame (each of ``num_rotors``
    active harmonics/bins gets ``2 * blur_bins + 1`` bins after target
    blurring); the weight is roughly (#negative bins) / (#positive bins).
    """
    active = num_rotors * (2 * blur_bins + 1)
    return (n_bins - active) / max(active, 1)


def salience_bce_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    pos_weight: float | torch.Tensor | None = None,
) -> torch.Tensor:
    """Binary cross-entropy on per-bin salience logits.

    Args:
        logits: (B, n_bins, T) raw model output (pre-sigmoid).
        target: (B, n_bins, T) target salience (binary or blurred/soft).
        pos_weight: scalar or per-bin positive-class weight, or None.
    """
    pw = None
    if pos_weight is not None:
        pw = torch.as_tensor(pos_weight, dtype=logits.dtype, device=logits.device)
    return F.binary_cross_entropy_with_logits(logits, target, pos_weight=pw)


# ─── Frame adapter ────────────────────────────────────────────────────────────


class SalienceBCELoss:
    """Frame adapter around :func:`salience_bce_loss`.

    Compares ``pred[pred_key]`` (default ``"salience"``, raw logits) against
    ``target[target_key]`` (default ``"salience"``, the RPS-derived target
    built by the salience-RPS task codec).
    """

    def __init__(
        self,
        *,
        pos_weight: float | None = None,
        rate: tuple[int, int] | None = None,
        pred_key: str = "salience",
        target_key: str = "salience",
    ) -> None:
        self.pos_weight = pos_weight
        self.pred_key = pred_key
        self.target_key = target_key
        spec = SeriesSpec(dims=("batch", "freq", "time"), time="grid", rate=rate)
        self.requires_pred = FrameSpec({pred_key: spec})
        self.requires_target = FrameSpec({target_key: spec})

    def __call__(self, pred: td.Frame, target: td.Frame) -> torch.Tensor:
        logits = get_tensor(pred, self.pred_key)
        tgt = get_tensor(target, self.target_key)
        return salience_bce_loss(logits, tgt, pos_weight=self.pos_weight)


class SalienceRPSBCELoss:
    """Frame adapter deriving its own BCE target from ``target["rps"]``.

    Historically ``train_rps_predictor.py`` built the salience target inline
    each step via ``model.salience_target_from_frame_rps(...)``, so every
    dataloader only had to provide the common ``(mixture, rps)`` RPS-
    prediction batch — no dataset ever emits a ``"salience"`` entry (see
    REPLICATION.md § C7/C8). This loss reproduces that: it owns the
    salience-grid parameters a model would otherwise supply
    (``fmin``/``n_octaves``/``over_sample``/``n_bins``/``bins_per_octave``
    for a log-spaced CQT-style grid — mirrors
    ``models.salience_rps.SalienceRPSPredictor.grid_params()`` — or
    ``out_fmin``/``out_fmax``/``out_bins`` for a decoupled fine *linear*
    output grid, mirroring a super-resolution head's ``out_freqs``) and
    derives the target from ``target["rps"]`` on every call, through the
    exact same :func:`models.multif0.utils.salience_target_from_resampled_rps`
    the model method itself calls — one implementation, never duplicated.

    The target's time grid is read off ``pred[pred_key]``'s own frame count
    (not recomputed from a front-end object this loss doesn't have), so it
    stays correct regardless of the model's specific front-end/hop.
    """

    def __init__(
        self,
        *,
        fmin: float = 32.7,
        n_octaves: int = 6,
        over_sample: int = 5,
        n_bins: int | None = None,
        bins_per_octave: int | None = None,
        out_fmin: float | None = None,
        out_fmax: float | None = None,
        out_bins: int | None = None,
        blur_bins: int = 0,
        pos_weight: float | Literal["auto"] | None = None,
        num_rotors: int = 4,
        rate: tuple[int, int] | None = None,
        pred_key: str = "salience",
        target_key: str = "rps",
    ) -> None:
        if out_fmin is not None or out_fmax is not None or out_bins is not None:
            if out_fmin is None or out_fmax is None or out_bins is None:
                raise ValueError(
                    "out_fmin/out_fmax/out_bins must all be set together "
                    "(decoupled linear output grid)"
                )
            self._freqs = linear_freq_grid(out_fmin, out_fmax, out_bins)
        else:
            self._freqs = cqt_freq_grid(
                fmin=fmin,
                n_octaves=n_octaves,
                over_sample=over_sample,
                n_bins=n_bins,
                bins_per_octave=bins_per_octave,
            )
        self.blur_bins = blur_bins
        self.pos_weight: float | None = (
            auto_pos_weight(len(self._freqs), num_rotors, blur_bins)
            if pos_weight == "auto"
            else pos_weight
        )
        self.pred_key = pred_key
        self.target_key = target_key
        self.requires_pred = FrameSpec(
            {pred_key: SeriesSpec(dims=("batch", "freq", "time"), time="grid", rate=rate)}
        )
        self.requires_target = FrameSpec(
            {target_key: SeriesSpec(dims=("batch", "rotor", "time"), time="grid", rate=None)}
        )

    def __call__(self, pred: td.Frame, target: td.Frame) -> torch.Tensor:
        logits = get_tensor(pred, self.pred_key)  # (B, n_bins, T_grid)
        rps = get_tensor(target, self.target_key).float()  # (B, rotor, T_stft)
        n_grid = logits.shape[-1]
        rps_grid = F.interpolate(rps, size=n_grid, mode="linear", align_corners=False)
        sal_target = salience_target_from_resampled_rps(
            rps_grid, self._freqs, blur_bins=self.blur_bins
        )
        return salience_bce_loss(logits, sal_target, pos_weight=self.pos_weight)


__all__ = [
    "auto_pos_weight",
    "salience_bce_loss",
    "SalienceBCELoss",
    "SalienceRPSBCELoss",
]
