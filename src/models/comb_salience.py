"""Comb-salience models: hypothesis scoring with an explicit harmonic gather.

WHY THIS FAMILY EXISTS. The regression models in `rps_predictor` are bounded by
two things that no amount of width or depth can move, both measured.

1. THE HEAD IS A REGRESSOR AND THE LOSS IS SQUARED ERROR, so the optimal output
   is the conditional MEAN. Where the model cannot resolve the individual
   rotors it shrinks every estimate toward their average, which is exactly the
   fixed "fan" the campaign measured: on cruise clips whose rotors span 20
   rev/s, `comb_floor_deep` places the centre almost perfectly (75.38 against a
   true 75.00) and returns a spread of 10.5. On the comb-floor validation set a
   predictor that is GIVEN the per-frame mean and has no spread at all scores
   2.214 rev/s against that model's 2.155 — the model's whole score is
   reproducible from the centre alone.

2. THE FEATURES CANNOT BE GATHERED AT A HYPOTHESIS. A comb at rate `r` is a
   DILATION of a comb at rate `r'`, not a translation of it. Convolutions share
   weights across translations, so on a linear-frequency spectrogram a
   convolutional stack has no weight sharing across the one symmetry the
   problem actually has, and must learn a separate detector per rate. The
   classical scan sidesteps this by INDEXING the spectrum at `k r` — reading
   the bins a hypothesis predicts — which has no convolutional equivalent.

THE FAMILY. Score a grid of rate hypotheses instead of regressing a number.
`CombGather` reads the spectrum at every harmonic of every candidate rate,
which is the classical operation made differentiable; `CombScoreHead` turns
those readings into a score per (rate, frame). Decoding takes modes, not means.

THE CORNER CASE. With `CombScoreHead` in its `classical` setting — a mean over
harmonics of `log1p(power / floor)` — the salience IS the Whittle comb score of
`tracking.comb_seed`, bit-comparable on the same periodogram. Training only
relaxes that fixed head into a learned one, so the classical method is a point
in the family's parameter space rather than a baseline beside it.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["CombGather", "CombScoreHead", "local_floor_torch"]


def local_floor_torch(pw: torch.Tensor, width_bins: int) -> torch.Tensor:
    """Running-median floor along the frequency axis of ``(B, F, T)``.

    The same local line test the classical scan uses: an absolute threshold
    cannot say whether a bin holds a line when several dense combs are present,
    because most bins then clear it. A median is not differentiable in a useful
    way, so training uses this as a detached normalizer — it sets the scale, it
    is not a parameter.
    """
    w = max(3, int(width_bins) | 1)
    pad = w // 2
    x = F.pad(pw.transpose(1, 2), (pad, pad), mode="replicate")
    med = x.unfold(-1, w, 1).median(dim=-1).values
    return med.transpose(1, 2).clamp_min(1e-12)


class CombGather(nn.Module):
    """Read a spectrogram at every harmonic of every candidate rate.

    ``(B, F, T)`` power spectrogram -> ``(B, K, G, T)``, where entry
    ``(k, g)`` is the spectrum interpolated at frequency ``(k+1) * rate[g]``.
    This is the operation a convolution cannot express: the offsets it gathers
    are PROPORTIONAL to the hypothesis, so one set of weights applies to every
    rate. Harmonics above ``f_max`` are masked out and excluded from the count,
    exactly as the classical scan excludes them.

    Args:
        r_lo, r_hi, n_grid: the rate grid, in rev/s. Pass ``grid`` instead to
            supply the candidate rates verbatim.
        k_max: harmonics per hypothesis.
        sr, n_fft: define the frequency of bin ``i`` as ``i * sr / n_fft``.
        f_max: harmonics at or above this frequency are dropped.
    """

    def __init__(
        self, r_lo: float = 30.0, r_hi: float = 100.0, n_grid: int = 512,
        k_max: int = 40, sr: int = 16000, n_fft: int = 4096, f_max: float = 7500.0,
        grid=None,
    ):
        super().__init__()
        self.k_max, self.f_max = int(k_max), float(f_max)
        # Built in float64 and kept there. The corner-case claim is that this
        # module REPRODUCES the classical scan, and at float32 the interpolation
        # positions differ in the seventh digit, which showed up as a 8e-6
        # relative mismatch. `forward` casts to the input's dtype, so training
        # still runs in float32.
        if grid is None:
            grid = torch.linspace(float(r_lo), float(r_hi), int(n_grid), dtype=torch.float64)
        else:
            grid = torch.as_tensor(grid, dtype=torch.float64)
        self.n_grid = int(grid.numel())
        ks = torch.arange(1, self.k_max + 1, dtype=torch.float64)
        fk = ks[:, None] * grid[None, :]                      # (K, G) Hz
        df = float(sr) / float(n_fft)
        # Split the interpolation into an integer bin and a fraction ONCE, and
        # compute the fraction as `(f - lo*df) / df` rather than `f/df - lo`.
        # They are equal in exact arithmetic; in float64 the second loses
        # precision at large bin indices (up to 15360 here), which showed up as
        # a 1.2e-11 disagreement with the classical scan.
        pos = fk / df
        lo = torch.floor(pos)
        frac = (fk - lo * df) / df
        self.register_buffer("grid", grid, persistent=False)
        self.register_buffer("bin_lo", lo.long(), persistent=False)     # (K, G)
        self.register_buffer("bin_frac", frac, persistent=False)        # (K, G)
        self.register_buffer("valid", (fk < self.f_max).to(torch.float64), persistent=False)
        self.register_buffer("count", (fk < self.f_max).to(torch.float64).sum(0).clamp_min(1.0),
                             persistent=False)

    def forward(self, pw: torch.Tensor) -> torch.Tensor:
        b, n_f, n_t = pw.shape
        lo = self.bin_lo.reshape(-1).clamp(0, n_f - 2)
        frac = self.bin_frac.reshape(-1).to(pw.dtype)[None, :, None]    # (1, K*G, 1)
        a = pw[:, lo, :]
        c = pw[:, (lo + 1).clamp(max=n_f - 1), :]
        h = a + (c - a) * frac                                          # (B, K*G, T)
        return h.view(b, self.k_max, self.n_grid, n_t) * self.valid.to(pw.dtype)[None, :, :, None]


class CombScoreHead(nn.Module):
    """Turn harmonic readings ``(B, K, G, T)`` into a score ``(B, G, T)``.

    ``mode="classical"`` computes ``mean_k log1p(h / floor)`` — the Whittle comb
    score, with no free parameters. That setting is the corner case the family
    is built around, and it is what the learned modes are initialized to.

    ``mode="learned"`` keeps the same shape but lets the network decide how a
    harmonic's elevation maps to evidence and how harmonics combine: a shared
    piecewise-linear warp of each reading and a learned weight per harmonic
    order. Both are initialized to zero effect, so the head starts AT the
    classical score and can only depart from it if that helps.
    """

    def __init__(self, k_max: int = 40, mode: str = "classical", n_knots: int = 8,
                 z_max: float = 12.0):
        super().__init__()
        if mode not in ("classical", "learned"):
            raise ValueError(f"unknown mode {mode!r}")
        self.mode, self.k_max = mode, int(k_max)
        if mode == "learned":
            # A learned PIECEWISE-LINEAR warp of the per-harmonic evidence, not
            # an MLP. The readings live on a (batch, harmonic, rate, frame)
            # grid, so a hidden channel dimension would cost hundreds of
            # millions of activations; a warp with `n_knots` hinge terms costs
            # `n_knots` multiply-adds per reading and no extra memory.
            #
            #     phi(z) = z + sum_j a_j * relu(z - c_j)
            #
            # with `a_j` initialized to zero, so the head STARTS as the exact
            # classical score and can only leave it by learning.
            self.register_buffer("knots", torch.linspace(0.0, z_max, n_knots))
            self.slope = nn.Parameter(torch.zeros(n_knots))
            self.w = nn.Parameter(torch.zeros(self.k_max))

    def _warp(self, z: torch.Tensor) -> torch.Tensor:
        k = self.knots.to(z.dtype)
        a = self.slope.to(z.dtype)
        return z + torch.einsum("j,...j->...", a, F.relu(z.unsqueeze(-1) - k))

    def forward(self, h: torch.Tensor, floor: torch.Tensor, count: torch.Tensor) -> torch.Tensor:
        # floor: (B, F, T) -> gathered alongside h by the caller, already (B,K,G,T)
        z = torch.log1p(h / floor)
        cnt = count.to(z.dtype)[None, :, None]
        if self.mode == "classical":
            return z.sum(dim=1) / cnt
        w = (1.0 + self.w).to(z.dtype)[None, :, None, None]
        return (self._warp(z) * w).sum(dim=1) / cnt
