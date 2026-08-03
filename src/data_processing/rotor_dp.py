"""Exact single-rotor Viterbi lattice tracker with a claim-masked residual emission.

The joint 4-rotor beam search is approximate in a ``len(grid)^4`` product
space, so a bad window never says which half failed — the objective or the
search.  For ONE rotor the state space is a scalar speed grid (~1000 points),
so exact dynamic programming is tractable and the search error is ZERO by
construction: any failure on a window is the objective's.  That is the
isolation ``scripts/sr_dp_probe.py`` runs on, and it is the same
objective-vs-search split ``jb_probe --mode cost`` measures, with the search
half removed rather than merely compared against.

Three pieces:

- :func:`viterbi_path` — exact banded Viterbi over a per-frame-normalised
  ``(D, T)`` score surface.  The transition cost is Huber in the innovation
  ``|dg| / s_rps`` (quadratic within :attr:`LatticeCfg.huber_knee` innovation
  scales, linear beyond) — the same asymmetry :class:`OUPrior` gives the
  common mode, and for the same reason: a takeoff ramp holds a large per-frame
  innovation for seconds, and a quadratic cost charges it enough to flatten
  the track.  The hard band :attr:`LatticeCfg.max_step_rps` bounds the
  relaxation to ``O(D * B)`` per frame, which is what makes exactness cheap.
- :func:`residual_scores` — the single-comb emission with the spectrogram
  bins already claimed by FIXED rotor trajectories excluded from the pooling.
  This is the sequential form of the joint tracker's union emission: rotor 2
  is scored only on what rotor 1 does not explain, so it cannot re-claim the
  loudest comb — the collapse the union was built to prevent, enforced here
  by masking instead of joint search.
- :func:`greedy_peel` — blind sequential extraction: track, claim, repeat.
  Greedy in the extraction ORDER only; each individual track is exact.

Emission machinery (tables, pooling, per-frame normalisation) is reused from
:mod:`data_processing.joint_beam_tracker` unchanged, so scores here are on
the same scale as the joint tracker's and ``lambda_e`` keeps its meaning.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from data_processing.joint_beam_tracker import (
    _INVALID_BIN,
    CombTables,
    EmissionCfg,
    comb_scores_from_tables,
    normalise_scores,
)


@dataclass(frozen=True)
class LatticeCfg:
    """Transition model of the 1-D lattice (analog of ``BeamCfg``'s prior half).

    The scales are per FRAME (32 ms hop), like ``OUPrior``'s innovation
    scales, not per second.
    """

    #: Transition innovation scale, rev/s per frame.
    s_rps: float = 0.4
    #: Huber knee, in innovation scales.  Quadratic within, linear beyond —
    #: the linear tail is what keeps a sustained ramp (takeoff) affordable.
    huber_knee: float = 1.5
    #: Hard band half-width of the transition, rev/s per frame.  Bounds the
    #: per-frame relaxation and forbids teleporting between combs.
    max_step_rps: float = 5.0
    #: Emission weight (same convention as ``BeamCfg.lambda_e``).
    lambda_e: float = 3.0
    #: Claim-mask half-width, in rounded spectrogram bins.  A claimed tooth
    #: suppresses candidate teeth within this bin distance, not only its own
    #: bin.  The value is physical: the Hann analysis window's mainlobe is
    #: +-2 bins, so a comb displaced 0.2-0.4 rev/s from a claimed one still
    #: reads the claimed teeth's FLANK energy — its high-k teeth land on the
    #: adjacent bin, inside the mainlobe.  With exact-bin masking (0) the
    #: first probe run measured the consequence: every oracle-masked DP
    #: reproduced the raw unmasked trajectory bit-for-bit and the greedy peel
    #: parked all four tracks within 0.4 rev/s of one comb.  At +-2 bins a
    #: flank impostor is suppressed at EVERY k while a genuine 0.9 rev/s twin
    #: keeps its k >= 9 teeth (n_fft 4096: k * 0.9 Hz clears 2 bins from
    #: k = 9), which is exactly the separation the masking must make.
    mask_halfwidth_bins: int = 2
    #: Minimum surviving teeth for a residual score to count; below it the
    #: speed is floored like a zero-survivor one.  The second probe run
    #: measured why a dilated mask alone is not enough: on FLY124 an impostor
    #: 0.7 rev/s from a claimed comb keeps only its k = 14-16 teeth (3 of 16)
    #: and still pools POSITIVE off the claimed teeth's jitter-broadened
    #: skirts — real tooth linewidth grows with k and exceeds the window
    #: mainlobe, so a handful of barely-cleared teeth is flank energy, not
    #: evidence.  A genuine twin must instead clear the mask on enough
    #: independent teeth to pool over (a 0.9 rev/s twin keeps 6+ at
    #: ``k_max = 16``, 15+ at 30).
    min_surv_teeth: int = 4


def viterbi_path(
    surface: torch.Tensor, grid: torch.Tensor, cfg: LatticeCfg
) -> tuple[torch.Tensor, float]:
    """Exact banded Viterbi over a ``(D, T)`` per-frame-normalised surface.

    Path cost::

        sum_t -lambda_e * surface[d_t, t]
        + sum_{t>=1} huber(|g[d_t] - g[d_{t-1}]| / s_rps)

    with ``huber(z) = 0.5 z^2`` for ``z <= knee`` and
    ``knee * (z - 0.5 knee)`` beyond; steps larger than ``max_step_rps`` are
    forbidden (infinite cost).  The grid is uniform, so the transition cost
    depends only on the index offset: it is precomputed once for
    ``delta in [-B, B]`` and the per-frame relaxation is one pad + unfold +
    min over the band — no python loop over states, no device round-trips.

    Returns ``(path_idx, total_cost)``: ``(T,)`` long indices into ``grid``
    (on ``surface``'s device) and the exact minimum of the cost above.
    """
    d_n, t_n = surface.shape
    step = float(grid[1] - grid[0])
    band = max(1, min(int(round(cfg.max_step_rps / step)), d_n - 1))
    inf = float("inf")
    deltas = torch.arange(-band, band + 1, device=surface.device, dtype=surface.dtype) * step
    z = deltas.abs() / cfg.s_rps
    tc = torch.where(z <= cfg.huber_knee, 0.5 * z**2, cfg.huber_knee * (z - 0.5 * cfg.huber_knee))
    tc = torch.where(deltas.abs() <= cfg.max_step_rps + 1e-9, tc, torch.full_like(tc, inf))

    emit = -cfg.lambda_e * surface
    cost = emit[:, 0].clone()
    back = torch.empty((t_n, d_n), device=surface.device, dtype=torch.int32)
    idx = torch.arange(d_n, device=surface.device, dtype=torch.int32)
    for t in range(1, t_n):
        padded = F.pad(cost, (band, band), value=inf)
        # window j of state d is the previous state d - band + j
        win = padded.unfold(0, 2 * band + 1, 1)  # (D, 2B+1)
        best, arg = (win + tc[None, :]).min(dim=1)
        back[t] = idx + (arg.to(torch.int32) - band)
        cost = best + emit[:, t]

    end = int(torch.argmin(cost))
    total = float(cost[end])
    path = torch.empty(t_n, device=surface.device, dtype=torch.long)
    path[t_n - 1] = end
    for t in range(t_n - 1, 0, -1):
        path[t - 1] = back[t, path[t]].long()
    return path, total


def residual_scores(
    tab: CombTables,
    emis: EmissionCfg,
    claimed_idx: torch.Tensor,
    mask_halfwidth_bins: int = 2,
    min_surv_teeth: int = 1,
) -> torch.Tensor:
    """Raw ``(D, T)`` single-comb scores with claimed teeth excluded from the pool.

    ``claimed_idx`` is ``(R, T)`` long — grid indices of ``R`` already-extracted
    rotor trajectories (``R = 0`` returns :func:`comb_scores_from_tables`
    unchanged).  Per frame, the claimed bin set is the union of the claimed
    speeds' valid on-tooth bins, DILATED by ``mask_halfwidth_bins`` on each
    side; every grid speed is then pooled over its OWN valid teeth whose bin
    is NOT in that dilated set.  Bin identity is ``tab.bid_on``, the identity
    the joint tracker's union deduplicates on; the dilation is what the union
    lacked — see :attr:`LatticeCfg.mask_halfwidth_bins` for why exact-bin
    exclusion lets a claimed comb survive as its own flanks
    (``mask_halfwidth_bins = 0`` reproduces the exact-bin behaviour).

    A speed with fewer than ``min_surv_teeth`` surviving teeth at a frame gets
    that frame's minimum scored value minus 1 — the DP avoids it, but the
    surface stays finite so ``normalise_scores``'s per-frame statistics are
    not poisoned by infinities.  See :attr:`LatticeCfg.min_surv_teeth` for why
    a small survivor set is flank energy rather than evidence.

    Implementation is a loop over frames (T ~ 500-600): the claimed set is
    per-frame, and a fully vectorised form needs a ``(D, K, T, C)`` comparison
    tensor that dwarfs the tables themselves.
    """
    if claimed_idx.numel() == 0:
        return comb_scores_from_tables(tab, emis)
    if emis.pool not in ("quantile", "mean"):
        raise ValueError(f"unsupported pool {emis.pool!r} for residual scoring")

    contrast = tab.v_on - tab.v_half  # (D, K, T) per-tooth contrast
    valid = tab.w > 0  # (D, K)
    d_n, k_n, t_n = contrast.shape
    flat_bids = tab.bid_on.reshape(-1)
    big = torch.finfo(contrast.dtype).max
    out = torch.empty((d_n, t_n), device=contrast.device, dtype=contrast.dtype)
    dil = torch.arange(
        -mask_halfwidth_bins, mask_halfwidth_bins + 1, device=contrast.device, dtype=torch.long
    )
    for t in range(t_n):
        cis = claimed_idx[:, t]
        cb = tab.bid_on[cis]  # (R, K)
        cb = cb[(tab.w[cis] > 0) & (cb != _INVALID_BIN)]
        cb = (cb[:, None] + dil[None, :]).reshape(-1)
        surv = valid & ~torch.isin(flat_bids, cb).reshape(d_n, k_n)
        n_surv = surv.sum(dim=1)
        d_t = contrast[:, :, t]
        if emis.pool == "quantile":
            # same semantics as `comb_scores_from_tables`: non-pooled teeth are
            # pushed to +inf and the quantile position counts survivors only
            dv = torch.where(surv, d_t, torch.full_like(d_t, big))
            srt, _ = dv.sort(dim=1)
            pos = ((n_surv - 1).clamp_min(0).to(d_t.dtype) * emis.pool_q).round().long()
            sc = srt.gather(1, pos[:, None])[:, 0]
        else:  # "mean": weighted mean over survivors, weights renormalised
            w_s = tab.w * surv.to(tab.w.dtype)
            sc = (w_s * d_t).sum(dim=1) / w_s.sum(dim=1).clamp_min(1e-12)
        has = n_surv >= min_surv_teeth
        if bool(has.all()):
            out[:, t] = sc
        else:
            floor = sc[has].min() - 1.0 if bool(has.any()) else sc.new_tensor(-1.0)
            out[:, t] = torch.where(has, sc, floor)
    return out


def track_masked(
    tab: CombTables,
    emis: EmissionCfg,
    lat: LatticeCfg,
    claimed_idx: torch.Tensor | None = None,
    grid: torch.Tensor | None = None,
) -> dict[str, Any]:
    """One exact single-rotor track on the residual surface.

    ``tab`` is prebuilt (:func:`comb_tables`) because the tables are the
    expensive part (~``(1081, 16, 600)`` floats at the probe's grid) and the
    caller reuses them across rotors.  ``grid`` defaults to ``emis.grid()``
    and must be the grid the tables were built on.

    Returns ``path_idx`` (``(T,)`` long, on the tables' device), ``speeds``
    (``(T,)`` float ndarray in rev/s), ``total_cost``, and
    ``support_raw_mean`` — the mean of the RAW residual score along the path.
    The last one is the honest support number: ``normalise_scores`` divides by
    ``peak - median`` per frame, so an empty frame's best coincidence still
    scores ~1.0 and a normalised mean cannot say "there was no comb here".
    """
    device = tab.v_on.device
    t_n = tab.v_on.shape[2]
    if grid is None:
        grid = torch.as_tensor(emis.grid(), device=device, dtype=tab.v_on.dtype)
    if grid.shape[0] != tab.v_on.shape[0]:
        raise ValueError(
            f"grid has {grid.shape[0]} points but tables were built on {tab.v_on.shape[0]}"
        )
    if claimed_idx is None:
        claimed_idx = torch.zeros((0, t_n), device=device, dtype=torch.long)
    raw = residual_scores(tab, emis, claimed_idx, lat.mask_halfwidth_bins, lat.min_surv_teeth)
    path, total = viterbi_path(normalise_scores(raw, emis), grid, lat)
    support = float(raw[path, torch.arange(t_n, device=device)].mean())
    return {
        "path_idx": path,
        "speeds": grid[path].detach().cpu().numpy().astype(np.float64),
        "total_cost": total,
        "support_raw_mean": support,
    }


def greedy_peel(
    tab: CombTables,
    emis: EmissionCfg,
    lat: LatticeCfg,
    n_rotors: int = 4,
    grid: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Blind sequential extraction of ``n_rotors`` tracks.

    Track on the unclaimed surface, claim the result, repeat — each pass exact,
    only the extraction order greedy.  The residual masking is what stops pass
    ``r+1`` from re-finding pass ``r``'s comb: its teeth are excluded, so a
    duplicate scores the per-frame floor.

    Returns ``speeds`` (``(n_rotors, T)`` ndarray, grid rev/s), ``path_idx``
    (``(n_rotors, T)`` long), and per-pass ``supports`` / ``costs`` in
    extraction order — ``supports`` should DECREASE along the peel; a late
    pass whose raw support is near zero extracted a rotor that is not there.
    """
    device = tab.v_on.device
    t_n = tab.v_on.shape[2]
    claimed = torch.zeros((0, t_n), device=device, dtype=torch.long)
    speeds: list[np.ndarray] = []
    supports: list[float] = []
    costs: list[float] = []
    for _ in range(n_rotors):
        res = track_masked(tab, emis, lat, claimed, grid)
        speeds.append(res["speeds"])
        supports.append(res["support_raw_mean"])
        costs.append(res["total_cost"])
        claimed = torch.cat([claimed, res["path_idx"][None]], dim=0)
    return {
        "speeds": np.stack(speeds),
        "path_idx": claimed,
        "supports": supports,
        "costs": costs,
    }
