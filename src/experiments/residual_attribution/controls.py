"""Null controls: does the *rotor* hypothesis earn its explanatory power?

A four-source free-field model has enough freedom to explain a good deal of any
coherent field. So "the fit explains 80 % of the off-diagonal energy" is not by
itself evidence that the four fitted powers belong to the four rotors. Each
control replaces the true rotor positions with a different hypothesis of the
same dimension and refits; only the excess of the true geometry over the
controls is attributable to rotor identity.

Controls
--------
``true``
    The published rotor positions.
``rot45``
    The rotor square rotated 45 degrees about the array axis. Same distance
    from centre, same count, different positions.
``rot90``
    Rotated 90 degrees. **Degenerate by construction** on DREGON: the rotors
    sit at ``(+-0.1715, +-0.1715)`` about their centroid, an exact square, so a
    90-degree rotation maps the rotor *set* onto itself and merely permutes the
    design columns. Its score is identical to ``true`` by algebra, not by
    evidence. It is kept only as a self-check that the pipeline is
    permutation-invariant; never read it as a control.
``mirror_z``
    Rotors reflected to below the array. Same lateral pattern, opposite
    elevation.
``random``
    ``n_draw`` draws of four positions on the sphere of the mean rotor radius.
    The distribution of their scores is the null band.
``centroid``
    A single source at the rotor centroid (rank-one model, three fewer
    parameters). Tells whether four sources are needed at all.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .design import IndexPlan
from .fit import fit_offdiag
from .steering import steering

__all__ = ["control_geometries", "score_control", "run_controls", "displacement_curve"]


def _rotate_z(pos: np.ndarray, deg: float, centre: np.ndarray) -> np.ndarray:
    a = np.deg2rad(deg)
    rot = np.array([[np.cos(a), -np.sin(a), 0.0], [np.sin(a), np.cos(a), 0.0], [0.0, 0.0, 1.0]])
    return (pos - centre) @ rot.T + centre


def control_geometries(
    rotor_pos: np.ndarray,
    *,
    n_draw: int = 16,
    seed: int = 0,
) -> dict[str, list[np.ndarray]]:
    """Name -> list of candidate rotor-position arrays."""
    rot = np.asarray(rotor_pos, dtype=np.float64)
    centre = rot.mean(axis=0)
    centre_xy = np.array([centre[0], centre[1], 0.0])
    radius = float(np.linalg.norm(rot - centre, axis=1).mean())
    rng = np.random.default_rng(seed)

    randoms = []
    for _ in range(n_draw):
        v = rng.standard_normal((len(rot), 3))
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        randoms.append(centre + radius * v)

    return {
        "true": [rot],
        "rot45": [_rotate_z(rot, 45.0, centre_xy)],
        "rot90": [_rotate_z(rot, 90.0, centre_xy)],
        "mirror_z": [rot * np.array([1.0, 1.0, -1.0])],
        "random": randoms,
        "centroid": [centre[None, :]],
    }


def score_control(
    R: np.ndarray,
    freqs: np.ndarray,
    mic_pos: np.ndarray,
    rotor_pos: np.ndarray,
    plan: IndexPlan,
    band_mask: np.ndarray,
) -> float:
    """Energy-weighted fraction of weighted off-diagonal energy explained, over
    the bins in ``band_mask``."""
    g = steering(mic_pos, rotor_pos, freqs)
    att = fit_offdiag(R, g, plan)
    v = att.off_explained[band_mask]
    return float(np.nanmean(v))


def run_controls(
    R: np.ndarray,
    freqs: np.ndarray,
    mic_pos: np.ndarray,
    rotor_pos: np.ndarray,
    plan: IndexPlan,
    bands: Sequence[tuple[float, float]],
    *,
    n_draw: int = 16,
    seed: int = 0,
) -> dict[str, dict[str, float]]:
    """``{band_label: {control_name: mean explained fraction}}``."""
    geoms = control_geometries(rotor_pos, n_draw=n_draw, seed=seed)
    out: dict[str, dict[str, float]] = {}
    for lo, hi in bands:
        m = (freqs >= lo) & (freqs < hi)
        label = f"{lo:g}-{hi:g}"
        row: dict[str, float] = {}
        for name, cands in geoms.items():
            scores = [score_control(R, freqs, mic_pos, c, plan, m) for c in cands]
            row[name] = float(np.mean(scores))
            if len(scores) > 1:
                row[name + "_max"] = float(np.max(scores))
                row[name + "_std"] = float(np.std(scores))
        out[label] = row
    return out


def displacement_curve(
    R: np.ndarray,
    freqs: np.ndarray,
    mic_pos: np.ndarray,
    rotor_pos: np.ndarray,
    plan: IndexPlan,
    band: tuple[float, float],
    *,
    offsets_m: np.ndarray | None = None,
    n_draw: int = 8,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """How far can the rotors be moved before the fit notices?

    Perturbs every rotor by an isotropic random vector of fixed length and
    refits. The offset at which the explained fraction starts to fall is the
    array's effective spatial resolution for this residual: if the curve is
    flat out to a rotor spacing, per-rotor attribution is not identifiable in
    practice however well-conditioned the design looks on paper.

    Returns ``(offsets (K,), mean explained (K,))``.
    """
    rng = np.random.default_rng(seed)
    offs = np.array([0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3]) if offsets_m is None else offsets_m
    m = (freqs >= band[0]) & (freqs < band[1])
    out = np.zeros(len(offs))
    for i, off in enumerate(offs):
        if off == 0.0:
            out[i] = score_control(R, freqs, mic_pos, rotor_pos, plan, m)
            continue
        vals = []
        for _ in range(n_draw):
            v = rng.standard_normal(np.asarray(rotor_pos).shape)
            v /= np.linalg.norm(v, axis=1, keepdims=True)
            vals.append(score_control(R, freqs, mic_pos, rotor_pos + off * v, plan, m))
        out[i] = float(np.mean(vals))
    return offs, out
