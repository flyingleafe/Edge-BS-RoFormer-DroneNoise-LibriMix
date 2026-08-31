"""Parallel slot allocation over rate hypotheses — the peel, done jointly.

WHY. `comb_salience.decode_peel_viterbi` removes one comb at a time and freezes
each pick. That greedy commitment is the diagnosed cause of the campaign's one
open failure: at a 40 rev/s centre the peel takes 41.07 correctly and then
52.35, 76.57 and 90.39, which are MULTIPLES of rotors it has not found yet, and
by the third round the notch has destroyed the evidence needed to notice. A
joint allocation has the information the greedy sweep threw away — a slot
claiming 76.57 competes for bins that a slot at 38.0 also wants, and 38.0
additionally explains its own odd harmonics, which 76.57 cannot.

THE MECHANISM. R slots each hold a distribution `p_i(g, t)` over the rate grid.
Slot i's soft comb is `d_i(f, t) = sum_g p_i(g, t) M[g, f]`, where `M` is a bank
of Gaussian comb templates — so a slot's only free variable is its RATE, never a
free per-bin mask. A free mask would make each slot a general-purpose separator
able to explain anything, which throws away the dilation structure that is the
reason this family beats convolutions at all.

Bins are ALLOCATED, not copied. With total claim `c = sum_j d_j`, slot i scores

    Y_i = Y * (1 - (c - d_i) / max(c, 1))  +  floor * ((c - d_i) / max(c, 1))

which removes what OTHER slots claim and nothing else. Two cases fix the design:

* Four rotors at distinct rates: at slot i's own lines `c = d_i = 1` so it sees
  the full spectrum; at another slot's lines it sees the floor. That is
  explain-away, computed simultaneously instead of greedily.
* Four rotors at ONE rate (the `identical` cell, where a collapsed answer is the
  CORRECT answer): `c = 4`, every slot sees `0.25 Y` at the lines. The score
  drops but the RANKING does not move, so the configuration is stable. A plain
  mutual notch would have all four slots erase each other, which is why the
  share normalization is not cosmetic.

THE CORNER CASE IS PRESERVED. `n_iter=0` runs the sequential peel with hard
one-hot slots and is the deployed `decode_peel_viterbi` — the mask bank at a
one-hot `p` reproduces `CombSalienceNet.notch` exactly, since with one claimer
`(c - d_i)/max(c,1)` collapses to that slot's own comb. Joint iterations then
refine an initialization that is the current method, so this family contains it
in the same way the family contains the classical scan.

THE SOFT ASSIGNMENT IS A CRF POSTERIOR, not a per-frame softmax: `p_i` comes
from forward-backward over the same chain the Viterbi decoder maximizes, so a
frame whose own evidence is ambiguous inherits its neighbours' certainty. That
is also what makes selection and deployment the same object.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from models import comb_crf
from models.comb_salience import CombGather, CombScoreHead, local_floor_torch

__all__ = ["CombMaskBank", "SlotCombNet"]


class CombMaskBank(nn.Module):
    """``(G, F)`` soft comb templates: how much of bin ``f`` a rotor at ``r_g`` owns.

    Each template is the pointwise MAXIMUM of Gaussian bumps at ``k * r_g``, not
    their sum: a claim is a fraction of a bin and must stay in ``[0, 1]``, and
    where two harmonics of one comb fall in the same bin a sum would exceed it.
    """

    def __init__(self, grid: torch.Tensor, n_fft: int, sr: int, k_max: int,
                 f_max: float, width_bins: float = 1.5, n_freq: int | None = None):
        super().__init__()
        n_freq = n_freq if n_freq is not None else n_fft // 2 + 1
        df = float(sr) / float(n_fft)
        fbin = torch.arange(n_freq, dtype=torch.float32)
        bank = torch.zeros((len(grid), n_freq), dtype=torch.float32)
        # Accumulated in harmonic CHUNKS. Materializing (G, K, F) at once is
        # 2.9 GB for a full-band claim (700 x 250 x 2049 float64) and killed a
        # six-worker evaluation pool outright.
        for k0 in range(1, k_max + 1, 16):
            ks = torch.arange(k0, min(k0 + 16, k_max + 1), dtype=torch.float32)
            fk = ks[None, :] * grid.to(torch.float32)[:, None]        # (G, kc)
            d = (fbin[None, None, :] - (fk / df)[:, :, None]) / float(width_bins)
            bump = torch.exp(-0.5 * d * d) * (fk < float(f_max))[:, :, None]
            bank = torch.maximum(bank, bump.amax(dim=1))
        self.register_buffer("bank", bank, persistent=False)

    def forward(self, p: torch.Tensor) -> torch.Tensor:
        """``(B, G, T)`` rate distribution -> ``(B, F, T)`` claim in ``[0, 1]``."""
        return torch.einsum("bgt,gf->bft", p, self.bank.to(p.dtype)).clamp(0.0, 1.0)


class SlotCombNet(nn.Module):
    """Audio -> ``R`` rotor trajectories, by joint allocation over a rate grid.

    Args:
        n_rot: number of slots.
        n_iter: joint refinement sweeps AFTER the sequential initialization.
            ``0`` reproduces the deployed peel-plus-Viterbi decoder.
        head_mode: ``"classical"`` (no parameters, the Whittle score),
            ``"learned"`` or ``"learned_cond"``.
        slew, stiff: the hinge transition, in the classical units.
    """

    def __init__(
        self, sr: int = 16000, n_fft: int = 4096, hop_length: int = 512,
        r_lo: float = 30.0, r_hi: float = 100.0, n_grid: int = 700,
        k_max: int = 32, f_max: float = 7500.0, head_mode: str = "classical",
        floor_hz: float = 120.0, n_rot: int = 4, n_iter: int = 2,
        notch_width: float = 1.5, slew: float = 12.0, stiff: float = 40.0,
        use_checkpoint: bool = True, mask_k_max: int | None = None,
        union_mode: str = "noisyor",
    ):
        super().__init__()
        self.sr, self.n_fft, self.hop_length = int(sr), int(n_fft), int(hop_length)
        self.n_rot, self.n_iter = int(n_rot), int(n_iter)
        self.union_mode = str(union_mode)
        self.use_checkpoint = bool(use_checkpoint)
        self.floor_bins = max(3, int(round(floor_hz / (sr / n_fft))) | 1)
        self.gather = CombGather(r_lo, r_hi, n_grid, k_max, sr, n_fft, f_max)
        self.head = CombScoreHead(k_max, head_mode)
        # THE CLAIM SPANS THE WHOLE BAND, THE SCORE DOES NOT. Scoring stops at
        # `k_max` harmonics, but a rotor radiates lines all the way up, so
        # explaining one away must remove all of them. With the claim truncated
        # to 32 harmonics a rotor at 34.6 rev/s was only notched to 1107 Hz, and
        # a slot at 68.5 then fed on the SAME rotor's surviving even harmonics
        # above that — which is the measured `typical-idle` failure.
        self.mask_k_max = int(mask_k_max if mask_k_max is not None
                              else np.ceil(f_max / max(r_lo, 1e-6)))
        self.masks = CombMaskBank(self.gather.grid, n_fft, sr, self.mask_k_max,
                                  f_max, notch_width)
        self.register_buffer("window", torch.hann_window(int(n_fft)), persistent=False)
        step = float(self.gather.grid[1] - self.gather.grid[0])
        self.step_free = max(slew * (hop_length / sr) / step, 1e-9)
        span, pen = comb_crf.band_penalty(self.step_free, stiff)
        self.span = span
        self.register_buffer("pen", pen, persistent=False)

    @property
    def grid(self) -> torch.Tensor:
        return self.gather.grid

    def spectrum(self, audio: torch.Tensor) -> torch.Tensor:
        if audio.dim() == 3:
            audio = audio.squeeze(1)
        spec = torch.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length,
                          window=self.window.to(audio.dtype), center=True,
                          return_complex=True)
        return spec.real.pow(2) + spec.imag.pow(2)

    def _score(self, pw: torch.Tensor, gfloor: torch.Tensor) -> torch.Tensor:
        """One slot's salience from an already-residualized spectrum."""
        h = self.gather(pw)
        return self.head(h, gfloor, self.gather.count, self.gather.grid)

    def _score_ckpt(self, pw, gfloor):
        if self.use_checkpoint and self.training and torch.is_grad_enabled():
            return checkpoint(self._score, pw, gfloor, use_reentrant=False)
        return self._score(pw, gfloor)

    def _residual(self, pw, floor, claims, skip: int | None):
        """Remove every slot's claim except ``skip``'s. ``claims`` is ``(B, R, F, T)``."""
        if claims is None or claims.shape[1] == 0:
            return pw
        c = claims.sum(dim=1)
        mine = claims[:, skip] if skip is not None else torch.zeros_like(c)
        other = ((c - mine) / c.clamp_min(1.0)).clamp(0.0, 1.0)
        return pw * (1.0 - other) + floor * other

    def forward(self, audio: torch.Tensor, hard_init: bool = True):
        """Returns ``(scores, p)``: ``(B, R, G, T)`` salience and rate posteriors."""
        pw = self.spectrum(audio)
        floor = local_floor_torch(pw, self.floor_bins).detach()
        gfloor = self.gather(floor).clamp_min(1e-12)
        b, _, n_t = pw.shape
        n_g = self.gather.n_grid

        # ── Sequential initialization: this IS the deployed peel ──────────────
        claims = pw.new_zeros((b, 0, pw.shape[1], n_t))
        scores, posts = [], []
        for i in range(self.n_rot):
            res = self._residual(pw, floor, claims if claims.shape[1] else None, None)
            s = self._score_ckpt(res, gfloor)
            scores.append(s)
            if hard_init:
                path = comb_crf.viterbi(s.detach(), self.span, self.pen)
                p = torch.zeros_like(s).scatter_(1, path.unsqueeze(1), 1.0)
            else:
                p = comb_crf.posterior_marginals(s, self.span, self.pen)
            posts.append(p)
            claims = torch.cat([claims, self.masks(p).unsqueeze(1)], dim=1)

        # ── Joint sweeps: every slot now sees every OTHER slot's claim ────────
        for _ in range(self.n_iter):
            scores = []
            for i in range(self.n_rot):
                res = self._residual(pw, floor, claims, i)
                scores.append(self._score_ckpt(res, gfloor))
            posts = [comb_crf.posterior_marginals(s, self.span, self.pen) for s in scores]
            claims = torch.stack([self.masks(p) for p in posts], dim=1)

        return torch.stack(scores, dim=1), torch.stack(posts, dim=1)

    # ── Decoding ─────────────────────────────────────────────────────────────

    def _rescore(self, pw, floor, gfloor, claims):
        return [self._score(self._residual(pw, floor, claims, i), gfloor)
                for i in range(self.n_rot)]

    def _claims_from(self, paths, like):
        """One-hot claims from a list of ``(B, T)`` index paths."""
        cl = []
        for path in paths:
            p = torch.zeros_like(like).scatter_(1, path.unsqueeze(1), 1.0)
            cl.append(self.masks(p))
        return torch.stack(cl, dim=1)

    def _solve(self, scores):
        """Viterbi every slot and return the paths with their total path score.

        The total is the quantity every discrete move below is judged by. It is a
        UNION objective in effect, not a per-line mean: because bins are shared
        (`c` in `_residual`), two slots parked on one rotor each see half its
        power and the pair scores less than one slot there plus one on an
        uncovered rotor. The campaign measured all three candidate objectives and
        found the union one is the only one that survives both incentives —
        `docs/experiments/synthetic-solvability-limits.md`, "The joint score".
        """
        paths = [comb_crf.viterbi(s, self.span, self.pen) for s in scores]
        tot = sum(comb_crf.path_score(s, self.span, self.pen, p)
                  for s, p in zip(scores, paths))
        return paths, tot

    def union_evidence(self, pw, floor, claims) -> torch.Tensor:
        """Whittle evidence over the UNION of covered bins: ``(B,)``.

        ``g(u) = u - 1 - log u`` is the generalized likelihood ratio of "a line
        of the observed strength lives here" against "only the floor does", for
        the exponential bin-power law. It is ZERO at ``u = 1``, so covering an
        empty bin is worth nothing, and it saturates nowhere, so covering a loud
        line is worth a lot. Bins are counted ONCE — `claims` is clamped before
        summing — which is what makes a duplicate slot worthless and an
        uncovered rotor valuable.

        This is the objective the campaign identified as the only one of three
        that survives both incentives (`synthetic-solvability-limits.md`, "The
        joint score"). The per-slot path score does not: it ranked a solution
        holding two MULTIPLES above the truth on every `typical-idle` clip
        (1619 against 1508, 1665 against 1590, 1752 against 1609), which made
        that cell an objective wall rather than a search wall.
        """
        # How the slots' claims combine into one coverage map. The three rules
        # trade, and the trade is measured rather than argued -- see
        # `docs/experiments/comb-slot-crf.md`:
        #   "max"     idempotent, so a duplicate slot adds exactly nothing, but
        #             two ADJACENT rotors that genuinely share a bin get credit
        #             for only the louder of them.
        #   "sum"     lets partial claims add, and lets a duplicate inflate the
        #             Gaussian tails (measured 1.335e8 -> 2.116e8 for one copy).
        #   "noisyor" 1 - prod(1 - d): saturating, so duplicates of a claimed
        #             line add nothing, while partial claims still combine.
        if self.union_mode == "max":
            c = claims.amax(dim=1)
        elif self.union_mode == "noisyor":
            c = 1.0 - (1.0 - claims.clamp(0.0, 1.0)).prod(dim=1)
        else:
            c = claims.sum(dim=1).clamp(0.0, 1.0)
        u = (pw / floor).clamp_min(1e-9)
        return (c * (u - 1.0 - u.log()).clamp_min(0.0)).sum(dim=(1, 2))

    def _odd_even(self, pw, floor, r):
        """Ratio of odd-harmonic to even-harmonic evidence at rate ``r`` ``(B, T)``.

        Near one at a true fundamental, far below it at a subharmonic whose odd
        harmonics fall in the gaps. Ported from `comb_salience.octave_fix`.
        """
        from models.comb_salience import _gather_at
        h, ok = _gather_at(pw, r, self.gather.k_max, self.sr, self.n_fft,
                           self.gather.f_max)
        fh, _ = _gather_at(floor, r, self.gather.k_max, self.sr, self.n_fft,
                           self.gather.f_max)
        lev = torch.log1p(h / fh.clamp_min(1e-12)) * ok
        return (lev[:, 0::2].mean(dim=1) / lev[:, 1::2].mean(dim=1).clamp_min(1e-9)).mean(dim=1)

    def _octave_moves(self, pw, floor, gfloor, scores, paths, rounds: int = 2,
                      ratio: float = 0.6):
        """Halve or double one slot, judged where each direction's information is.

        The two directions need DIFFERENT discriminators, and conflating them is
        why `comb_salience` ended with two gates that traded and no blind rule
        for choosing between them:

        * HALVING is rejected by the slot's own odd-to-even evidence ratio. A
          subharmonic's odd harmonics land in the gaps between the true lines,
          so the ratio collapses. This test needs no absolute threshold and no
          knowledge of the other rotors.
        * A MULTIPLE cannot be rejected that way at all — its harmonics are a
          subset of the true comb's and are therefore always present. It is
          rejected by COVERAGE: moving the slot down onto the fundamental adds
          that rotor's odd harmonics to the union, and `union_evidence` rises.

        Because the two gates read different quantities they do not trade, and
        the choice `octave_mode` used to expose is gone.
        """
        grid = self.grid
        lo, hi, n_g = float(grid[0]), float(grid[-1]), len(grid)
        step = float(grid[1] - grid[0])
        claims = self._claims_from(paths, scores[0])
        best = self.union_evidence(pw, floor, claims)
        for _ in range(rounds):
            improved = False
            for i in range(self.n_rot):
                for factor in (0.5, 2.0):
                    r = grid[paths[i]] * factor
                    if bool(((r < lo) | (r > hi)).any()):
                        continue
                    if factor < 1.0 and bool((self._odd_even(pw, floor, r) < ratio).all()):
                        continue                       # a real subharmonic: refuse
                    prop = ((r - lo) / step).round().long().clamp(0, n_g - 1)
                    trial = claims.clone()
                    trial[:, i] = self._claims_from([prop], scores[0])[:, 0]
                    if not bool((self.union_evidence(pw, floor, trial) > best + 1e-6).all()):
                        continue
                    # Accepted: let every other slot re-solve against the new claim.
                    t_scores = self._rescore(pw, floor, gfloor, trial)
                    paths, _ = self._solve(t_scores)
                    scores = t_scores
                    claims = self._claims_from(paths, scores[0])
                    best = self.union_evidence(pw, floor, claims)
                    improved = True
            if not improved:
                break
        return scores, paths

    def decode(self, audio: torch.Tensor, subgrid: bool = True,
               octave: bool = True) -> torch.Tensor:
        """``(B, R, T)`` rates in rev/s, sorted ascending per frame."""
        pw = self.spectrum(audio)
        floor = local_floor_torch(pw, self.floor_bins).detach()
        gfloor = self.gather(floor).clamp_min(1e-12)
        scores, _ = self.forward(audio)
        scores = [scores[:, i] for i in range(scores.shape[1])]
        paths, _ = self._solve(scores)
        if octave:
            scores, paths = self._octave_moves(pw, floor, gfloor, scores, paths)
        grid = self.grid.to(scores[0].dtype)
        out = []
        for s, path in zip(scores, paths):
            r = grid[path]
            if subgrid:
                r = r + self._parabolic(s, path)
            out.append(r)
        return torch.stack(out, dim=1).sort(dim=1).values

    def _parabolic(self, s: torch.Tensor, path: torch.Tensor) -> torch.Tensor:
        """Sub-grid offset of the path, in rev/s, by a three-point parabolic fit.

        The grid step is 0.1 rev/s and the target is finer, so without this the
        discretization alone would floor the error near 0.029 rev/s RMS.
        """
        g = s.shape[1]
        step = float(self.grid[1] - self.grid[0])
        i0 = path.clamp(1, g - 2)
        a = s.gather(1, (i0 - 1).unsqueeze(1)).squeeze(1)
        c0 = s.gather(1, i0.unsqueeze(1)).squeeze(1)
        c = s.gather(1, (i0 + 1).unsqueeze(1)).squeeze(1)
        den = a - 2 * c0 + c
        d = torch.where(den.abs() < 1e-12, torch.zeros_like(den), 0.5 * (a - c) / den)
        return d.clamp(-1, 1) * step

    # ── Loss ─────────────────────────────────────────────────────────────────

    def loss(self, audio: torch.Tensor, rps: torch.Tensor) -> torch.Tensor:
        """CRF negative log-likelihood of the true trajectories, slot-matched.

        `rps` is ``(B, R, T)`` in rev/s. Slots are matched to rotors by the
        assignment that minimizes total NLL — a permutation, resolved once, with
        no squared error anywhere, so nothing pushes a slot toward the mean of
        two rotors the way PIT-MSE does.
        """
        scores, _ = self.forward(audio)
        grid = self.grid.to(scores.dtype)
        gold = (rps.unsqueeze(-1) - grid).abs().argmin(dim=-1)        # (B, R, T)
        b, r = scores.shape[0], scores.shape[1]
        cost = scores.new_zeros((b, r, r))
        for i in range(r):
            for j in range(r):
                cost[:, i, j] = comb_crf.crf_nll(scores[:, i], self.span, self.pen,
                                                 gold[:, j])
        return _min_assignment(cost).mean()


def _min_assignment(cost: torch.Tensor) -> torch.Tensor:
    """Minimum-cost slot-to-rotor assignment per batch item: ``(B, R, R) -> (B,)``.

    Brute force over permutations. R is 4 here, so 24 of them — a Hungarian
    solver would be the same answer with a dependency and a device transfer.
    """
    import itertools
    r = cost.shape[1]
    ar = torch.arange(r, device=cost.device)
    perms = torch.tensor(list(itertools.permutations(range(r))), device=cost.device)
    tot = torch.stack([cost[:, ar, p].sum(dim=1) for p in perms], dim=1)
    return tot.min(dim=1).values
