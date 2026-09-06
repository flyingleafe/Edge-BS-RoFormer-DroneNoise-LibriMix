"""The v2 emission of the slot-comb CRF: gap reads, a cross-order net, learned widths.

WHAT THIS IS. `PartialEmission` (`models.comb_slots`) replaced the mean over
harmonic orders by a learned reliability over (order, channel). This module adds
the three EMISSION groups of `docs/slot-comb-v2-design.md` -- sections 3.4, 3.5
and 3.7 -- as `PartialEmissionV2`, a subclass. The chain groups (3.1 the OFF
state, 3.2 the grid below 30 rev/s, 3.3 the learned transition, 3.6 the pairwise
rate prior) are not here. Select it with `SlotCombNet(emission="v2")`.

THE INVARIANT. Every group starts at the setting it replaces, so
`emission="v2"` at initialization reproduces `emission="partial"` with the same
`parts` to better than 1e-6 on the score tensor, and training can only leave
that corner by lowering the CRF loss. `tests/models/test_comb_slots_v2_emission.py`
locks this. The four `PARTIAL_PARTS` keep their meaning and their defaults, so
`emission="v2"` with no v2 part named IS the partial emission.

THE FOUR NEW PARTS.

* `gap` (3.4). A second gather at the half-integer orders `k + 1/2` reads the
  spectrum BETWEEN the teeth. It serves two purposes. As a MULTIPLE
  discriminator it charges a hypothesis whose gaps are full, which is what a
  hypothesis at twice the true rate has and what no test on the teeth can see
  (the empty-tooth hinge is its mirror image and rejects the SUBHARMONIC). As a
  comb-conditioned FLOOR it gives a floor estimate that the running median
  cannot give below about 20 rev/s, where the teeth are 5 bins apart or closer
  and the median measures the comb itself.
* `cross_order` (3.5). A 1-D convolutional network ALONG THE ORDER AXIS, applied
  independently at every (rate, frame). It is the capacity lever: the weighted
  mean cannot express the SHAPE of the harmonic vector, which is what tells a
  two-blade rotor from a coaxial pair and what moves with the rate.
* `read_width_learned` (3.7). The read of one harmonic becomes a Gaussian of
  learned width, in place of the single interpolated bin. The single bin is
  right for a delta-thin line and wrong for a Lorentzian one whose width grows
  with the order.
* `claim_width_learned` (3.7). The 1.5-bin Gaussian claim of `CombMaskBank`
  becomes learnable, and the bank is rebuilt each forward.

THE GRADIENT AT THE CORNER, AND WHY TWO KNOBS START "TOO FAR OFF". A family that
contains its corner in the INTERIOR pays for it: the derivative of an almost-
dead knob is almost dead too. `softplus(mu)` at `mu = -16` is 1.1e-7, and the
Gaussian read at `sigma = 0.15` bins differs from the single bin by 2.2e-10, so
both gradients are real but far below what 1500 steps can move. That is the
same effect the C1 campaign measured on the octave charge, which stayed at 7e-4
from a `softplus(-8)` start and only ever moved in the arm that started it ON
(A7). So the two values are constructor arguments, `gap_mu_init` and
`read_sigma_init`, and `warm_start(net.emit, gap_mu=-2.0, read_sigma=0.7)` sets
them on a built model, because `SlotCombNet.__init__` has a fixed signature and
forwards no keyword to its emission. Leave them at the default for an ablation
that must reproduce the corner. The design's own value for `mu` is -8, which
costs 2.3e-4 on the score and would not meet the 1e-6 invariant; -16 is this
module's default for that reason.
"""

from __future__ import annotations

import math
from typing import NamedTuple, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from models.comb_salience import CombGather, CombScoreHead
from models.comb_slots import _SOFTPLUS_ONE, PARTIAL_PARTS, PartialEmission

__all__ = [
    "PARTIAL_V2_PARTS",
    "V2_PARTS",
    "LearnedCombMaskBank",
    "OffsetCombGather",
    "PartialEmissionV2",
    "attach_v2_emission",
    "warm_start",
]

#: The emission groups this module adds. Each is switchable on its own, next to
#: the four `PARTIAL_PARTS`, so an ablation stays a command-line argument.
V2_PARTS = ("gap", "cross_order", "read_width_learned", "claim_width_learned")

#: Every part `emission="v2"` accepts.
PARTIAL_V2_PARTS = PARTIAL_PARTS + V2_PARTS

#: The distance, in bins, that stands for "this candidate has no harmonic here".
#: A larger sentinel is not better: the claim is `exp(-0.5 (d / w)^2)`, whose
#: gradient carries the factor `d^2 / w^3`, so an infinite distance would make a
#: zero claim with a non-finite gradient.
_NO_HARMONIC = 1.0e4

#: `softplus(0)`. The cross-order weight is divided by it, so an untrained
#: network multiplies every order's weight by EXACTLY one. A constant factor
#: cancels between the numerator and the denominator in exact arithmetic, but
#: 0.693 is not a power of two, and the rounding it leaves was measured at
#: 2.4e-6 on the score -- twice the invariant this module must meet.
_SOFTPLUS_ZERO = float(math.log(2.0))


def _inv_softplus(y: float) -> float:
    """The raw value whose softplus is ``y``."""
    return float(math.log(math.expm1(float(y))))


def _norm_log_floor(gf: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """``(B, K, G, T)`` floor -> its log, standardized per batch item.

    The same five lines `PartialEmission._features` uses, because the cross-order
    network takes the SAME normalized log floor and must not be able to read
    level: a gain change moves the mean and the variance together and cannot
    move this. They are repeated here rather than factored out of
    `comb_slots.py`, which this module must leave untouched.
    """
    lf = gf.clamp_min(1e-12).log()
    n = valid.sum() * gf.shape[-1]
    mu = (lf * valid).sum(dim=(1, 2, 3), keepdim=True) / n.clamp_min(1.0)
    var = (((lf - mu) * valid) ** 2).sum(dim=(1, 2, 3), keepdim=True) / n.clamp_min(1.0)
    return ((lf - mu) / var.clamp_min(1e-12).sqrt()) * valid


class OffsetCombGather(CombGather):
    """`CombGather` read at the orders ``k + offset`` instead of ``k``.

    ``offset = 0.5`` reads the midpoint between two teeth, which is the one place
    the emission never looked. Everything else -- the grid, the float64
    positions, the `f_max` mask and the forward pass -- is the parent's, so the
    gap read and the tooth read cannot drift apart.
    """

    def __init__(
        self,
        *,
        grid: torch.Tensor,
        k_max: int,
        sr: int,
        n_fft: int,
        f_max: float,
        offset: float = 0.5,
    ):
        super().__init__(grid=grid, k_max=k_max, sr=sr, n_fft=n_fft, f_max=f_max)
        self.offset = float(offset)
        ks = torch.arange(1, self.k_max + 1, dtype=torch.float64) + self.offset
        g = cast(torch.Tensor, self.grid)
        fk = ks[:, None] * g[None, :]
        df = float(sr) / float(n_fft)
        pos = fk / df
        lo = torch.floor(pos)
        # Assignment, not `register_buffer`: the names exist already and the
        # parent's precision discipline (float64 positions, the fraction taken
        # as `(f - lo*df)/df`) is what is being kept.
        self.bin_lo = lo.long()
        self.bin_frac = (fk - lo * df) / df
        self.valid = (fk < self.f_max).to(torch.float64)
        self.count = (fk < self.f_max).to(torch.float64).sum(0).clamp_min(1.0)


class LearnedCombMaskBank(nn.Module):
    """`CombMaskBank` with a LEARNED claim width, rebuilt each forward.

    THE BANK IS ONE EXPONENTIAL. The fixed bank takes the pointwise maximum over
    harmonics of `exp(-0.5 ((f - f_k) / w)^2)`. Both the exponential and the
    division by a positive width are monotonic, so that maximum equals

        bank(g, f) = exp(-0.5 (dmin(g, f) / w)^2),
        dmin(g, f) = min over the valid k of |f - k r_g|, in bins,

    and `dmin` does not contain the width. So the chunked pass over the
    harmonics runs ONCE, at construction, and each forward is one elementwise
    exponential over ``(G, F)``. That is what makes a learnable width affordable:
    the 250 to 750 harmonics of a full-band claim never touch the autograd graph.
    """

    dmin: torch.Tensor

    def __init__(
        self,
        grid: torch.Tensor,
        n_fft: int,
        sr: int,
        k_max: int,
        f_max: float,
        width_bins: float = 1.5,
        n_freq: int | None = None,
    ):
        super().__init__()
        n_freq = n_freq if n_freq is not None else n_fft // 2 + 1
        df = float(sr) / float(n_fft)
        fbin = torch.arange(n_freq, dtype=torch.float32)
        dmin = torch.full((len(grid), n_freq), _NO_HARMONIC, dtype=torch.float32)
        # Chunked for the same reason the fixed bank is chunked: a full-band
        # claim is 700 x 250 x 2049 cells at once, which killed an evaluation
        # pool. This loop holds no graph, so 16 harmonics at a time is enough.
        for k0 in range(1, k_max + 1, 16):
            ks = torch.arange(k0, min(k0 + 16, k_max + 1), dtype=torch.float32)
            fk = ks[None, :] * grid.to(torch.float32)[:, None]  # (G, kc)
            d = (fbin[None, None, :] - (fk / df)[:, :, None]).abs()
            d = torch.where((fk < float(f_max))[:, :, None], d, torch.full_like(d, _NO_HARMONIC))
            dmin = torch.minimum(dmin, d.amin(dim=1))
        self.register_buffer("dmin", dmin, persistent=False)
        self.width_raw = nn.Parameter(torch.tensor(_inv_softplus(width_bins)))

    def width(self) -> torch.Tensor:
        """The claim width in bins. The clamp keeps a zero width off the graph."""
        return F.softplus(self.width_raw).clamp_min(1e-3)

    def bank(self, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        w = self.width().to(dtype)
        return torch.exp(-0.5 * (self.dmin.to(dtype) / w) ** 2)

    def forward(self, p: torch.Tensor) -> torch.Tensor:
        """``(B, G, T)`` rate distribution -> ``(B, F, T)`` claim in ``[0, 1]``."""
        return torch.einsum("bgt,gf->bft", p, self.bank(p.dtype)).clamp(0.0, 1.0)


class _Reads(NamedTuple):
    """What one channel's front end produces, before any weighting.

    ``z`` is the tooth reading against ``floor``, the floor the emission scores
    with. ``z_gap`` is the reading between the teeth, against ``gf_med``, the
    RUNNING-MEDIAN floor alone -- a gap read against a floor built from the gaps
    would be about `log 2` everywhere and would discriminate nothing.
    """

    z: torch.Tensor
    z_gap: torch.Tensor | None
    floor: torch.Tensor
    gf_med: torch.Tensor


class PartialEmissionV2(PartialEmission):
    """`PartialEmission` plus the gap read, the cross-order net and learned widths.

    HOW THE GROUPS COMPOSE. The partial emission scores one channel as
    `sum_k g_k phi(z_k) / sum_k g_k`, with `g_k = softplus(a_k) sigmoid(b_c)
    (1 + MLP)` and the invalid orders masked out. The cross-order network enters
    that quotient MULTIPLICATIVELY on the weight and ADDITIVELY on the evidence:

        g_k  <- g_k * softplus(l_k) / softplus(0),   phi(z_k) <- phi(z_k) + e_k

    At initialization the network's last layer is zero, so `l_k = 0`, the factor
    is exactly one and `e_k = 0`. The score is then the partial emission's, and
    every part of it still acts. The weight is bounded below by zero because it
    is a softplus and not a `1 + MLP`: the campaign's arms A3 and A6 went
    non-finite when a `1 + MLP` weight sum reached zero.

    The gap penalty is subtracted from the finished score, next to the empty-
    tooth charge, and like it is read on the MEAN channel only: it is a statement
    about the comb, and eight noisy copies of it would only add variance.

    Args:
        gather: the tooth gather the parent `SlotCombNet` will pass to
            `forward`. The half-order gather is built from it, so the two reads
            share a grid by construction.
        gap_mu_init: the raw gain of the multiple discriminator. See the module
            docstring: -16 keeps the 1e-6 corner, -8 is the design's value and
            -2 is a value that trains.
        read_sigma_init: the Gaussian read width in bins at initialization.
            0.15 puts 2.2e-10 of the kernel off the centre bin, which is the
            single interpolated bin to 1e-6 for any neighbour-to-centre power
            ratio below 4500.
        read_trunc: the Gaussian read kernel is cut at this many widths.
        cross_hidden, cross_kernel: the cross-order network's width and its
            kernel along the order axis.
    """

    def __init__(
        self,
        k_max: int = 40,
        n_mic: int = 8,
        parts: tuple[str, ...] = PARTIAL_V2_PARTS,
        hidden: int = 8,
        n_knots: int = 8,
        z_max: float = 12.0,
        floor_widths: tuple[int, ...] = (15, 31, 61),
        gather: CombGather | None = None,
        sr: int = 16000,
        n_fft: int = 4096,
        gap_mu_init: float = -16.0,
        read_sigma_init: float = 0.15,
        read_trunc: float = 4.0,
        cross_hidden: int = 16,
        cross_kernel: int = 5,
    ):
        bad = set(parts) - set(PARTIAL_V2_PARTS)
        if bad:
            raise ValueError(f"unknown emission parts {sorted(bad)}")
        super().__init__(
            k_max,
            n_mic=n_mic,
            parts=tuple(p for p in parts if p in PARTIAL_PARTS),
            hidden=hidden,
            n_knots=n_knots,
            z_max=z_max,
            floor_widths=floor_widths,
        )
        self.parts = tuple(parts)
        self.read_trunc = float(read_trunc)

        # ── 3.4 the gap gather ───────────────────────────────────────────────
        self.mu = nn.Parameter(torch.tensor(float(gap_mu_init)))
        self.w_gap = nn.Parameter(torch.full((int(k_max),), _SOFTPLUS_ONE))
        # The warp of the gap reading is the same family as the tooth warp, and
        # a SEPARATE instance: a full gap and a full tooth are opposite evidence
        # and must not share a shape. Its per-order weight is frozen, because
        # `w_gap` is the per-order weight here.
        self.gap_warp = CombScoreHead(k_max, "learned", n_knots=n_knots, z_max=z_max)
        self.gap_warp.w.requires_grad_(False)
        # One per order, initialized at 1: at 1 the mixture is the running median
        # alone. It is unconstrained on purpose -- the mixture is geometric, so
        # any real value keeps the floor positive.
        self.alpha = nn.Parameter(torch.ones(int(k_max)))

        # ── 3.5 the cross-order network ──────────────────────────────────────
        pad = int(cross_kernel) // 2
        self.cross = nn.Sequential(
            nn.Conv1d(6, int(cross_hidden), int(cross_kernel), padding=pad),
            nn.GELU(),
            nn.Conv1d(int(cross_hidden), int(cross_hidden), int(cross_kernel), padding=pad),
            nn.GELU(),
            nn.Conv1d(int(cross_hidden), 2, 1),
        )
        last = cast(nn.Conv1d, self.cross[4])
        nn.init.zeros_(last.weight)
        nn.init.zeros_(cast(torch.Tensor, last.bias))

        # ── 3.7 the learned read width ───────────────────────────────────────
        self.s0 = nn.Parameter(torch.tensor(_inv_softplus(read_sigma_init)))
        self.s1 = nn.Parameter(torch.tensor(0.0))

        self.gap_gather: OffsetCombGather | None = None
        if self.needs_gap and gather is not None:
            self.gap_gather = OffsetCombGather(
                grid=cast(torch.Tensor, gather.grid),
                k_max=gather.k_max,
                sr=int(sr),
                n_fft=int(n_fft),
                f_max=gather.f_max,
                offset=0.5,
            )

    # ── What is on ───────────────────────────────────────────────────────────

    @property
    def needs_gap(self) -> bool:
        """The gap read feeds the discriminator, the floor AND the cross net."""
        return ("gap" in self.parts) or ("cross_order" in self.parts)

    # ── The read ─────────────────────────────────────────────────────────────

    def sigma(self, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """The Gaussian read width per order, in bins: ``(K,)``.

        The clamp is the numerical guard. At `sigma = 0` the kernel exponent is
        `0 / 0` at the centre bin, which is a NaN and not a delta.
        """
        k = torch.arange(1, self.k_max + 1, device=self.s0.device, dtype=dtype)
        return F.softplus(self.s0.to(dtype) + self.s1.to(dtype) * k).clamp_min(1e-3)

    def read_kernel(self, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """One normalized Gaussian per order: ``(K, 1, 2R + 1)``.

        The radius follows the widest order, so a width that grows is not
        truncated. The kernel is NORMALIZED, which is what makes the wide read a
        variance reduction rather than a gain: a flat floor reads the same at
        every width, and only a line wider than a bin gains from it.
        """
        sig = self.sigma(dtype)
        r = max(1, int(math.ceil(self.read_trunc * float(sig.detach().max()))))
        j = torch.arange(-r, r + 1, device=sig.device, dtype=dtype)
        w = torch.exp(-0.5 * (j[None, :] / sig[:, None]) ** 2)
        return (w / w.sum(dim=-1, keepdim=True).clamp_min(1e-12)).unsqueeze(1)

    def _read(
        self, x_c: torch.Tensor, gather: CombGather
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """``(B, F, T)`` power -> the tooth reading and the gap reading.

        With `read_width_learned` the spectrogram is smoothed K times, once per
        order, and order k is then gathered from ITS OWN copy. The gather itself
        is unchanged, so at a delta kernel this is the single interpolated bin
        the classical scan reads. The gap at order `k + 1/2` comes from copy k
        as well: a tooth and the gap that follows it are one line width apart
        and must be read at one width, or their ratio would move with it.
        """
        gap = self._gap_gather(gather)
        if "read_width_learned" not in self.parts:
            return gather(x_c), (gap(x_c) if gap is not None else None)
        b, n_f, n_t = x_c.shape
        ker = self.read_kernel(x_c.dtype)
        flat = x_c.permute(0, 2, 1).reshape(b * n_t, 1, n_f)
        smooth = F.conv1d(flat, ker, padding=ker.shape[-1] // 2)  # (B*T, K, F)
        h = _gather_stack(smooth, gather, b, n_t)
        return h, (_gather_stack(smooth, gap, b, n_t) if gap is not None else None)

    def _gap_gather(self, gather: CombGather) -> OffsetCombGather | None:
        if not self.needs_gap:
            return None
        if self.gap_gather is None:
            raise RuntimeError("the v2 emission needs the gather it was built with")
        if gather.k_max != self.gap_gather.k_max or gather.n_grid != self.gap_gather.n_grid:
            raise ValueError("the v2 emission reads the gather it was built with")
        return self.gap_gather

    def reads(self, x_c: torch.Tensor, xf_c: torch.Tensor, gather: CombGather) -> _Reads:
        """One channel's readings and the floor they are read against.

        THE COMB-CONDITIONED FLOOR is a geometric mixture, written as

            floor_k = median_k * (P_gap_k / median_k) ^ (1 - alpha_k)

        which is the design's `alpha log median + (1 - alpha) log P_gap` and is
        EXACTLY the median at `alpha = 1`, where `exp(log(x))` is not. The
        median stays detached, as it is in the partial emission. Orders whose
        gap falls outside the band keep the median, because a missing gap read
        is a zero and its log is not a floor.
        """
        h, h_gap = self._read(x_c, gather)
        gf_med = gather(xf_c).clamp_min(1e-12)
        z_gap = torch.log1p(h_gap / gf_med) if h_gap is not None else None
        floor = gf_med
        if "gap" in self.parts and h_gap is not None:
            gap = cast(OffsetCombGather, self.gap_gather)
            ok = cast(torch.Tensor, gap.valid).to(h.dtype)[None, :, :, None]
            mix = (1.0 - self.alpha.to(h.dtype))[None, :, None, None] * ok
            floor = gf_med * torch.exp(mix * (h_gap.clamp_min(1e-12) / gf_med).log())
        return _Reads(z=torch.log1p(h / floor), z_gap=z_gap, floor=floor, gf_med=gf_med)

    # ── The emission ─────────────────────────────────────────────────────────

    def _cross_order(
        self,
        ev: torch.Tensor,
        z_gap: torch.Tensor,
        lfn: torch.Tensor,
        gather: CombGather,
        is_mean: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The network over the order axis: ``(B, K, G, T)`` -> a logit and an evidence.

        The six inputs are the warped tooth reading, the gap reading, the
        normalized log floor, the order fraction, the log rate and the channel
        kind. NO RAW POWER ENTERS: every level in the list is a ratio or a
        standardized log, so the family cannot learn to read gain. The rate
        enters as ONE feature, which is what keeps the read proportional to the
        hypothesis -- one network serves every rate, exactly as one gather does.
        """
        b, n_k, n_g, n_t = ev.shape
        lr = cast(torch.Tensor, gather.grid).to(ev.dtype).clamp_min(1e-6).log()
        lr = (lr - lr.mean()) / lr.std().clamp_min(1e-6)
        kf = torch.arange(1, n_k + 1, device=ev.device, dtype=ev.dtype) / n_k
        # (B, G, T, K) each, so the stack below is contiguous in the order axis
        # and the reshape into the convolution costs no second copy.
        cols = [
            ev.permute(0, 2, 3, 1),
            z_gap.permute(0, 2, 3, 1),
            lfn.permute(0, 2, 3, 1),
            kf[None, None, None, :].expand(b, n_g, n_t, n_k),
            lr[None, :, None, None].expand(b, n_g, n_t, n_k),
            torch.full((b, n_g, n_t, n_k), float(is_mean), dtype=ev.dtype, device=ev.device),
        ]
        feats = torch.stack(cols, dim=3).reshape(b * n_g * n_t, 6, n_k)
        out = self.cross(feats).view(b, n_g, n_t, 2, n_k)
        return out[:, :, :, 0].permute(0, 3, 1, 2), out[:, :, :, 1].permute(0, 3, 1, 2)

    def _gap_penalty(self, z_gap: torch.Tensor) -> torch.Tensor:
        """The multiple discriminator: ``(B, K, G, T)`` -> ``(B, G, T)``.

        A hypothesis at twice the true rate has every tooth present, so no test
        on the teeth rejects it. Its GAPS are the odd harmonics of the truth and
        are full, and this term charges for them.
        """
        gap = cast(OffsetCombGather, self.gap_gather)
        ok = cast(torch.Tensor, gap.valid).to(z_gap.dtype)[None, :, :, None]
        w = F.softplus(self.w_gap).to(z_gap.dtype)[None, :, None, None] * ok
        num = (w * self.gap_warp._warp(z_gap)).sum(dim=1)
        den = w.sum(dim=1).clamp_min(1e-12).expand_as(num)
        return F.softplus(self.mu).to(z_gap.dtype) * num / den

    def _channel(  # type: ignore[override]
        self, x_c: torch.Tensor, xf_c: torch.Tensor, gather: CombGather, c: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """One channel's weighted evidence: numerator, weight, hinge, gap charge."""
        rd = self.reads(x_c, xf_c, gather)
        z = rd.z
        valid = cast(torch.Tensor, gather.valid).to(z.dtype)[None, :, :, None]
        is_mean = 1.0 if c == 0 else 0.0
        w = F.softplus(self.a).to(z.dtype)[None, :, None, None] * torch.sigmoid(self.b[c])
        if "reliability" in self.parts:
            w = w * (1.0 + self.mlp(self._features(z, rd.floor, valid, is_mean)).squeeze(-1))
        ev = self.warp._warp(z)
        if "cross_order" in self.parts:
            assert rd.z_gap is not None
            lfn = _norm_log_floor(rd.floor, valid)
            logit, extra = self._cross_order(ev, rd.z_gap, lfn, gather, is_mean)
            w = w * F.softplus(logit).div(_SOFTPLUS_ZERO)
            ev = ev + extra
        g = w * valid
        num = (g * ev).sum(dim=1)
        den = g.sum(dim=1).expand_as(num)
        zero = torch.zeros_like(num)
        empty = ((self.tau - z).clamp_min(0.0) * valid).sum(dim=1) if c == 0 else zero
        charge = (
            self._gap_penalty(cast(torch.Tensor, rd.z_gap))
            if c == 0 and "gap" in self.parts
            else zero
        )
        return num, den, empty, charge

    def forward(  # type: ignore[override]
        self, x: torch.Tensor, xfloor: torch.Tensor, gather: CombGather, use_ckpt: bool = False
    ) -> torch.Tensor:
        """``(B, C, F, T)`` power and floor -> salience ``(B, G, T)``.

        The parent's channel loop, with the gap charge carried alongside the
        empty-tooth hinge. Channels stay one at a time under their own
        checkpoint: the cross-order network's activations live on the
        ``(B, C, K, G, T)`` grid, 145 MB per crop per layer at K=40, G=900 and
        T=63, so holding every channel at once would multiply the peak by the
        channel count.
        """
        num = den = empty = charge = None
        for c in range(x.shape[1]):
            args = (x[:, c], xfloor[:, c], gather, c)
            if use_ckpt:
                n_c, d_c, e_c, g_c = cast(
                    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
                    checkpoint(self._channel, *args, use_reentrant=False),
                )
            else:
                n_c, d_c, e_c, g_c = self._channel(*args)
            num = n_c if num is None else num + n_c
            den = d_c if den is None else den + d_c
            empty = e_c if empty is None else empty
            charge = g_c if charge is None else charge
        assert num is not None and den is not None and empty is not None and charge is not None
        s = num / den.clamp_min(1e-12)
        if "empty_tooth" in self.parts:
            cnt = cast(torch.Tensor, gather.count).to(s.dtype)[None, :, None]
            s = s - F.softplus(self.lam_raw) * empty / cnt
        if "gap" in self.parts:
            s = s - charge
        return s


def _gather_stack(smooth: torch.Tensor, gather: CombGather, b: int, n_t: int) -> torch.Tensor:
    """Read order k of ``(B*T, K, F)`` from copy k: ``-> (B, K, G, T)``.

    The arithmetic is `CombGather.forward`'s, order by order: the same integer
    bin, the same float64 fraction and the same `f_max` mask. Only the source
    changes, from one spectrogram to K of them.
    """
    n_f = smooth.shape[-1]
    lo = cast(torch.Tensor, gather.bin_lo).clamp(0, n_f - 2)
    frac = cast(torch.Tensor, gather.bin_frac).to(smooth.dtype)
    out = []
    for k in range(gather.k_max):
        a = smooth[:, k, lo[k]]
        c = smooth[:, k, lo[k] + 1]
        out.append(a + (c - a) * frac[k][None, :])
    h = torch.stack(out, dim=1).view(b, n_t, gather.k_max, -1).permute(0, 2, 3, 1)
    return h * cast(torch.Tensor, gather.valid).to(smooth.dtype)[None, :, :, None]


def warm_start(
    emit: PartialEmissionV2, gap_mu: float | None = None, read_sigma: float | None = None
) -> PartialEmissionV2:
    """Move the two gradient-starved knobs off the corner, in place.

    `SlotCombNet.__init__` has a fixed signature and does not forward keywords to
    its emission, so this is how a run asks for the values that TRAIN rather
    than the values that reproduce the corner:

        warm_start(net.emit, gap_mu=-2.0, read_sigma=0.7)

    It is the same decision the campaign's arm A7 made for the octave charge,
    and the same one that arm read back as learned (lambda 0.69 -> 1.02) while
    every arm that started at -8 read back 7e-4.
    """
    with torch.no_grad():
        if gap_mu is not None:
            emit.mu.fill_(float(gap_mu))
        if read_sigma is not None:
            emit.s0.fill_(_inv_softplus(read_sigma))
            emit.s1.zero_()
    return emit


def attach_v2_emission(
    net,
    *,
    k_max: int,
    n_mic: int,
    parts: tuple[str, ...],
    floor_widths: tuple[int, ...],
    f_max: float,
    notch_width: float,
    sr: int,
    n_fft: int,
    **kw,
) -> PartialEmissionV2:
    """Give ``net`` the v2 emission, and its learned claim bank if that part is on.

    This is the whole hook `SlotCombNet` needs. It runs after the fixed bank is
    built, so `claim_width_learned` pays one extra pass over the harmonics at
    construction and none afterwards.
    """
    emit = PartialEmissionV2(
        int(k_max),
        n_mic=int(n_mic),
        parts=tuple(parts),
        floor_widths=tuple(floor_widths),
        gather=net.gather,
        sr=int(sr),
        n_fft=int(n_fft),
        **kw,
    )
    net.emit = emit
    if "claim_width_learned" in parts:
        net.masks = LearnedCombMaskBank(
            cast(torch.Tensor, net.gather.grid),
            int(n_fft),
            int(sr),
            int(net.mask_k_max),
            float(f_max),
            float(notch_width),
        )
    return emit
