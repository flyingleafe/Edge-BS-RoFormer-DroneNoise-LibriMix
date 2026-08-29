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


class CombSalienceNet(nn.Module):
    """Audio -> a salience map over (candidate rate, frame).

    The output is NOT R rotor rates. It is a score for "some rotor turns at
    this rate in this frame", which removes the assignment problem the
    regression family has: there is nothing to permute, so no PIT loss and no
    pressure to average two rotors into one output. Decoding takes MODES.

    Set ``head_mode="classical"`` for the untrained corner case — the network is
    then the classical Whittle scan evaluated on the model's own STFT, with no
    learned parameters at all.
    """

    def __init__(
        self, sr: int = 16000, n_fft: int = 4096, hop_length: int = 512,
        r_lo: float = 30.0, r_hi: float = 100.0, n_grid: int = 700,
        k_max: int = 32, f_max: float = 7500.0, head_mode: str = "learned",
        floor_hz: float = 120.0,
    ):
        super().__init__()
        self.n_fft, self.hop_length, self.sr = int(n_fft), int(hop_length), int(sr)
        self.floor_bins = max(3, int(round(floor_hz / (sr / n_fft))) | 1)
        self.gather = CombGather(r_lo, r_hi, n_grid, k_max, sr, n_fft, f_max)
        self.head = CombScoreHead(k_max, head_mode)
        self.register_buffer("window", torch.hann_window(int(n_fft)), persistent=False)

    @property
    def grid(self) -> torch.Tensor:
        return self.gather.grid

    def spectrum(self, audio: torch.Tensor) -> torch.Tensor:
        """Audio -> power spectrogram ``(B, F, T)``."""
        if audio.dim() == 3:
            audio = audio.squeeze(1)
        spec = torch.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length,
                          window=self.window.to(audio.dtype), center=True,
                          return_complex=True)
        return spec.real.pow(2) + spec.imag.pow(2)

    def score(self, pw: torch.Tensor, floor: torch.Tensor | None = None) -> torch.Tensor:
        """Power spectrogram -> salience ``(B, G, T)``.

        Split out from `forward` so a peel loop can rescore a spectrum it has
        modified. The floor is computed from the ORIGINAL spectrum when one is
        supplied, so notching a comb out does not drag the noise reference down
        with it and inflate whatever remains.
        """
        if floor is None:
            floor = local_floor_torch(pw, self.floor_bins)
        # The floor sets the scale and is not a parameter: detached, exactly as
        # the classical scan's median floor is data and not a fit.
        floor = floor.detach()
        h = self.gather(pw)
        fh = self.gather(floor).clamp_min(1e-12)
        return self.head(h, fh, self.gather.count)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        return self.score(self.spectrum(audio))

    def notch(self, pw: torch.Tensor, rate: torch.Tensor, width_bins: float = 1.5,
              depth: float = 1.0) -> torch.Tensor:
        """Suppress the comb at ``rate`` ``(B, T)`` in ``pw`` ``(B, F, T)``.

        A soft Gaussian notch at every harmonic, down to the local floor. This
        is the explain-away step: without it the strongest comb wins every
        round, and the decoder has to guess from salience magnitude alone
        whether a second peak is a real rotor — which was measured not to work
        at any threshold.
        """
        b, n_f, n_t = pw.shape
        df = self.sr / self.n_fft
        fbin = torch.arange(n_f, device=pw.device, dtype=pw.dtype)[None, :, None]
        floor = local_floor_torch(pw, self.floor_bins).detach()
        mask = torch.ones_like(pw)
        ks = torch.arange(1, self.gather.k_max + 1, device=pw.device, dtype=pw.dtype)
        for k in ks:
            centre = (k * rate / df).unsqueeze(1)                     # (B, 1, T)
            g = torch.exp(-0.5 * ((fbin - centre) / width_bins) ** 2)
            mask = mask * (1.0 - depth * g)
        return pw * mask + floor * (1.0 - mask)


def _gather_at(pw: torch.Tensor, rate: torch.Tensor, k_max: int, sr: int,
               n_fft: int, f_max: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Read ``pw`` ``(B, F, T)`` at harmonics of a PER-FRAME rate ``(B, T)``.

    The grid gather serves a fixed set of candidates; this serves one rate that
    differs per frame, which is what the octave test needs after a pick.
    """
    b, n_f, n_t = pw.shape
    df = sr / n_fft
    ks = torch.arange(1, k_max + 1, device=pw.device, dtype=pw.dtype)
    fk = ks[None, :, None] * rate[:, None, :]                        # (B, K, T)
    pos = (fk / df).clamp(0, n_f - 2)
    lo = pos.floor().long()
    frac = pos - lo.to(pos.dtype)
    a = torch.gather(pw, 1, lo)
    c = torch.gather(pw, 1, (lo + 1).clamp(max=n_f - 1))
    return a + (c - a) * frac, (fk < f_max)


def octave_fix(
    pw: torch.Tensor, floor: torch.Tensor, rate: torch.Tensor, k_max: int, sr: int,
    n_fft: int, f_max: float, ratio: float = 0.6, r_hi: float = 100.0,
    r_lo: float = 30.0, max_mult: int = 2, mode: str = 'scored',
) -> torch.Tensor:
    """Raise a pick off a subharmonic, by the odd-to-even evidence ratio.

    The same test `tracking.comb_seed` uses, and it is needed for the same
    reason: a candidate at half the true rate covers every true line and fills
    the gaps with whatever is there, so the score alone does not always reject
    it. Without this the net was measured at 25.8 rev/s on a 40 rev/s centre,
    where the MULTIPLE at 80 sits inside the search grid, against 8.9 for the
    classical pipeline which has the test.

    If a candidate is really a half-rate its odd harmonics fall between the true
    lines and score low against its even ones; at the true rate both sets are
    its own lines and the ratio is near one. A ratio needs no absolute
    threshold, which is what makes it usable.
    """
    def _lev(r: torch.Tensor):
        h, ok = _gather_at(pw, r, k_max, sr, n_fft, f_max)
        fh, _ = _gather_at(floor, r, k_max, sr, n_fft, f_max)
        return torch.log1p(h / fh.clamp_min(1e-12)) * ok, ok

    def oe(r: torch.Tensor) -> torch.Tensor:
        lev, _ = _lev(r)
        return lev[:, 0::2].mean(dim=1) / lev[:, 1::2].mean(dim=1).clamp_min(1e-9)

    def sc(r: torch.Tensor) -> torch.Tensor:
        """The comb score itself, at one rate per frame."""
        lev, ok = _lev(r)
        return lev.sum(dim=1) / ok.to(lev.dtype).sum(dim=1).clamp_min(1.0)

    cur = rate
    # DOWN first. A MULTIPLE of the true rate cannot be rejected by asking
    # whether its own harmonics are present — they are a subset of the true
    # comb's lines, so they always are. It is caught by asking about HALF of
    # it: at the truth, `r/2` is a subharmonic and its odd harmonics fall in
    # the gaps, so the ratio is low; if `r` is a multiple, `r/2` is a real comb
    # and the ratio is near one. This direction matters only when the multiple
    # lands inside the search grid, which is why a 40 rev/s centre needs it
    # (multiples at 65-94 are in a 30-100 grid) and a 75 rev/s centre does not
    # (multiples at 139-161 are outside it).
    # `mode` picks how a demotion is gated, and the two settings TRADE — no
    # single one wins, so the caller chooses:
    #   "scored" also requires the half to score better. A true fundamental is
    #     then never demoted, and the 75 rev/s cells keep their accuracy
    #     (typical 0.209). It does NOT rescue a 40 rev/s centre (28.0),
    #     because by the third peel the notch has already removed the
    #     neighbours' low harmonics, so on that spectrum the half honestly no
    #     longer scores better.
    #   "ratio" demotes on the odd/even evidence alone. That rescues the
    #     40 rev/s centre (28.0 -> 6.9, beating the classical 8.9) and fires
    #     spuriously at 75, costing those cells (0.209 -> 0.740).
    for _ in range(max_mult):
        half = cur * 0.5
        # Demote only if the HALF actually scores better. That is the same
        # mean-of-logs score the salience uses, and it is exactly the quantity
        # that ranks a subharmonic BELOW the truth — so a real fundamental is
        # never demoted, while a multiple is. Using a second ratio threshold
        # here instead was measured to fire spuriously at a 75 rev/s centre and
        # cost the cells that already worked (0.209 -> 0.740).
        take = (oe(half) >= ratio) & (half >= r_lo)
        if mode == 'scored':
            take = take & (sc(half) > sc(cur))
        if not bool(take.any()):
            break
        cur = torch.where(take, half, cur)
    # UP, for a pick that really is a subharmonic.
    for _ in range(max_mult):
        need = (oe(cur) < ratio) & (2.0 * cur <= r_hi)
        if not bool(need.any()):
            break
        cur = torch.where(need, 2.0 * cur, cur)
    return cur


def decode_peel(
    model: CombSalienceNet, audio: torch.Tensor, n_rot: int = 4,
    width_bins: float = 1.5, refine: bool = True, octave: bool = True,
    octave_mode: str = "scored",
) -> torch.Tensor:
    """Peel ``n_rot`` combs out of one clip: rates ``(B, n_rot, T)``.

    Each round scores the current spectrum, takes the strongest rate per frame,
    and notches that comb out before the next round. This is the classical
    peel, and it is what supplies model order: a rotor sharing a rate with
    another leaves a peak that survives its own notch only if the evidence
    really is there for two.
    """
    pw = model.spectrum(audio)
    floor = local_floor_torch(pw, model.floor_bins).detach()
    grid = model.grid.to(pw.dtype)
    step = float(grid[1] - grid[0])
    picks = []
    for _ in range(n_rot):
        sal = model.score(pw, floor)
        idx = sal.argmax(dim=1)                                       # (B, T)
        if refine:
            g = sal.shape[1]
            i0 = idx.clamp(1, g - 2)
            a = torch.gather(sal, 1, (i0 - 1).unsqueeze(1)).squeeze(1)
            c0 = torch.gather(sal, 1, i0.unsqueeze(1)).squeeze(1)
            c = torch.gather(sal, 1, (i0 + 1).unsqueeze(1)).squeeze(1)
            den = a - 2 * c0 + c
            d = torch.where(den.abs() < 1e-12, torch.zeros_like(den), 0.5 * (a - c) / den)
            r = grid[i0] + d.clamp(-1, 1) * step
        else:
            r = grid[idx]
        if octave:
            r = octave_fix(pw, floor, r, model.gather.k_max, model.sr,
                           model.n_fft, model.gather.f_max,
                           r_hi=float(model.grid[-1]), r_lo=float(model.grid[0]),
                           mode=octave_mode)
        picks.append(r)
        pw = model.notch(pw, r, width_bins=width_bins)
    return torch.stack(picks, dim=1).sort(dim=1).values


def decode_topr(
    sal: torch.Tensor, grid: torch.Tensor, n_rot: int = 4,
    min_sep: float = 0.6, refine: bool = True,
) -> torch.Tensor:
    """Salience ``(B, G, T)`` -> rates ``(B, n_rot, T)``, sorted ascending.

    Takes the ``n_rot`` strongest local maxima per frame, keeping picks at least
    ``min_sep`` rev/s apart, and refines each to sub-grid precision by a
    parabolic fit — the grid is 0.1 rev/s and the target is finer than that, so
    without refinement the discretization would floor the result.
    """
    b, g, t = sal.shape
    step = float(grid[1] - grid[0])
    sep = max(1, int(round(min_sep / step)))
    out = sal.new_zeros(b, n_rot, t)
    work = sal.clone()
    for r in range(n_rot):
        idx = work.argmax(dim=1)                                     # (B, T)
        if refine:
            i0 = idx.clamp(1, g - 2)
            a = torch.gather(sal, 1, (i0 - 1).unsqueeze(1)).squeeze(1)
            c0 = torch.gather(sal, 1, i0.unsqueeze(1)).squeeze(1)
            c = torch.gather(sal, 1, (i0 + 1).unsqueeze(1)).squeeze(1)
            den = a - 2 * c0 + c
            d = torch.where(den.abs() < 1e-12, torch.zeros_like(den), 0.5 * (a - c) / den)
            out[:, r] = grid.to(sal.dtype)[i0] + d.clamp(-1, 1) * step
        else:
            out[:, r] = grid.to(sal.dtype)[idx]
        lo = (idx - sep).clamp(0, g - 1)
        ar = torch.arange(g, device=sal.device)[None, :, None]
        work = work.masked_fill((ar >= lo.unsqueeze(1)) & (ar <= (idx + sep).clamp(0, g - 1).unsqueeze(1)),
                                float("-inf"))
    return out.sort(dim=1).values


def decode_peaks(
    sal: torch.Tensor, grid: torch.Tensor, n_rot: int = 4,
    rel: float = 0.9, guard: int = 1, refine: bool = True,
) -> torch.Tensor:
    """Salience ``(B, G, T)`` -> rates ``(B, n_rot, T)``, sorted ascending.

    Decides HOW MANY distinct rotors the salience supports instead of forcing
    ``n_rot`` separated picks. A fixed rate-space separation cannot work, and
    the sweep that showed it is the same trade-off the classical peel has: at
    0.02 rev/s the decoder is right when rotors coincide (1.15) and wrong when
    they spread (7.49 at spread 20); at 0.6 it is right when they spread (1.80)
    and wrong when they coincide (17.4). No constant serves both.

    So take LOCAL MAXIMA, strongest first, and accept the next one only if it
    stands at ``rel`` of the strongest peak's salience — otherwise repeat the
    strongest. Coincident rotors then produce one peak that every rotor is
    assigned to, and separated rotors produce one peak each, with no rate scale
    hard-coded anywhere.
    """
    b, g, t = sal.shape
    step = float(grid[1] - grid[0])
    left = F.pad(sal[:, :-1], (0, 0, 1, 0), value=float("-inf"))
    right = F.pad(sal[:, 1:], (0, 0, 0, 1), value=float("-inf"))
    is_pk = (sal >= left) & (sal > right)
    masked = torch.where(is_pk, sal, torch.full_like(sal, float("-inf")))

    out = sal.new_zeros(b, n_rot, t)
    work = masked.clone()
    first_val = None
    for r in range(n_rot):
        val, idx = work.max(dim=1)                                   # (B, T)
        if first_val is None:
            first_val, first_idx = val, idx
        else:
            weak = val < rel * first_val
            idx = torch.where(weak, first_idx, idx)
            val = torch.where(weak, first_val, val)
        if refine:
            i0 = idx.clamp(1, g - 2)
            a = torch.gather(sal, 1, (i0 - 1).unsqueeze(1)).squeeze(1)
            c0 = torch.gather(sal, 1, i0.unsqueeze(1)).squeeze(1)
            c = torch.gather(sal, 1, (i0 + 1).unsqueeze(1)).squeeze(1)
            den = a - 2 * c0 + c
            d = torch.where(den.abs() < 1e-12, torch.zeros_like(den), 0.5 * (a - c) / den)
            out[:, r] = grid.to(sal.dtype)[i0] + d.clamp(-1, 1) * step
        else:
            out[:, r] = grid.to(sal.dtype)[idx]
        ar = torch.arange(g, device=sal.device)[None, :, None]
        lo, hi = (idx - guard).unsqueeze(1), (idx + guard).unsqueeze(1)
        work = work.masked_fill((ar >= lo) & (ar <= hi), float("-inf"))
    return out.sort(dim=1).values


def decode_peel_viterbi(
    model: CombSalienceNet, audio: torch.Tensor, n_rot: int = 4,
    width_bins: float = 1.5, octave: bool = True, octave_mode: str = "scored",
    slew: float = 12.0, stiff: float = 40.0, hop_s: float | None = None,
) -> torch.Tensor:
    """Peel, but take each rotor as a SMOOTH PATH rather than a per-frame argmax.

    `decode_peel` chooses independently in every frame, which throws away the
    one thing a rotor trajectory certainly has: continuity. On widely separated
    rotors that costs real accuracy against the classical pipeline (0.691
    against 0.374), and the classical pipeline's only structural advantage there
    is exactly this — it runs a Viterbi over the score surface.

    Reuses `tracking.comb_seed._viterbi_ridge`, so the temporal model is
    literally the classical one: a hinge cost that is free up to the airframe's
    physical slew and steep past it.
    """
    from tracking.comb_seed import _viterbi_ridge  # local: keeps the purity rule

    pw = model.spectrum(audio)
    floor = local_floor_torch(pw, model.floor_bins).detach()
    grid_np = model.grid.detach().cpu().numpy()
    dt = hop_s if hop_s is not None else model.hop_length / model.sr
    picks = []
    for _ in range(n_rot):
        sal = model.score(pw, floor)                                  # (B, G, T)
        rows = []
        for b in range(sal.shape[0]):
            idx = _viterbi_ridge(sal[b].detach().cpu().numpy().T, grid_np, slew, dt, stiff)
            rows.append(torch.as_tensor(grid_np[idx], dtype=pw.dtype, device=pw.device))
        r = torch.stack(rows)                                         # (B, T)
        if octave:
            r = octave_fix(pw, floor, r, model.gather.k_max, model.sr, model.n_fft,
                           model.gather.f_max, r_hi=float(model.grid[-1]),
                           r_lo=float(model.grid[0]), mode=octave_mode)
        picks.append(r)
        pw = model.notch(pw, r, width_bins=width_bins)
    return torch.stack(picks, dim=1).sort(dim=1).values
