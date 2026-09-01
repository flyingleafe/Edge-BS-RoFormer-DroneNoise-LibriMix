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
import torch.nn.functional as F
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
        union_mode: str = "noisyor", read_width: int = 0,
        k_refine: int | None = None, refine_band: float = 3.0,
        multichannel: bool = True,
    ):
        super().__init__()
        self.sr, self.n_fft, self.hop_length = int(sr), int(n_fft), int(hop_length)
        self.n_rot, self.n_iter = int(n_rot), int(n_iter)
        self.union_mode = str(union_mode)
        self.multichannel = bool(multichannel)
        # HOW MANY BINS ONE HARMONIC IS READ OVER. The gather reads a single
        # interpolated bin, which is right for a delta-thin line and wrong for a
        # Lorentzian one: the stochastic family's linewidth is
        # `gamma0 + slope * k` Hz, so harmonic 32 can be 26 Hz wide against 3.9 Hz
        # bins and one bin holds a small part of the line and a lot of floor.
        # Measured score margin of the truth over the best decoy, 3 clips:
        #   static     +-0 bins 1.634   +-2 bins 0.531   (thin lines: 0 is right)
        #   coherent   +-0 bins 0.089   +-2 bins 0.106
        #   Rayleigh   +-0 bins 0.017   +-2 bins 0.066   (3.9x)
        # The optimum is family-dependent, which is what makes this a front-end
        # parameter worth LEARNING rather than a constant worth tuning.
        self.read_width = int(read_width)
        # TWO HARMONIC COUNTS, BECAUSE THEY WANT OPPOSITE THINGS. Measured on the
        # real beat-VK windows: a candidate only pays for its empty gaps if its
        # harmonic list spans the whole band, so a SHORT list lets the half-rate
        # win (FLY124 cruise, truth beaten 6/7 windows at k_max=32, winning 6/7
        # at 200). But a LONG list drags in the decohered high harmonics of a
        # real rotor, and precision falls with it (DREGON nosource w2, 0.94 at
        # k_max=32 against 1.92 at 200). One number cannot serve both, so the
        # octave is settled with the long list and the rate is then refined with
        # the short one, inside a band around the settled path -- the seed-then-
        # refine split the classical pipeline already uses for the same reason.
        self.k_refine = int(k_refine) if k_refine else 0
        self.refine_band = float(refine_band)
        self.use_checkpoint = bool(use_checkpoint)
        self.floor_bins = max(3, int(round(floor_hz / (sr / n_fft))) | 1)
        self.gather = CombGather(r_lo, r_hi, n_grid, k_max, sr, n_fft, f_max)
        self.head = CombScoreHead(k_max, head_mode)
        if self.k_refine:
            self.gather_lo = CombGather(grid=self.gather.grid, k_max=self.k_refine,
                                        sr=sr, n_fft=n_fft, f_max=f_max)
            self.head_lo = CombScoreHead(self.k_refine, "classical")
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
        """``(B, T)`` or ``(C, T)`` -> ``(B, F, T)`` power, averaging over channels.

        A multi-microphone input is averaged in POWER, never in waveform. The mics
        sit metres apart, so summing their waveforms comb-filters exactly the lines
        being read; averaging ``|STFT|^2`` leaves the mean spectrum alone and
        divides the per-bin variance by the channel count. On real recordings that
        variance is what the score margin is made of -- the truth outscores the
        best decoy by about 0.03 nats there, against 1.65 on the synthetic static
        comb, so variance reduction is the lever with the most room in it.
        """
        if audio.dim() == 3:
            audio = audio.squeeze(1)
        multi = audio.dim() == 2 and audio.shape[0] > 1 and self.multichannel
        spec = torch.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length,
                          window=self.window.to(audio.dtype), center=True,
                          return_complex=True)
        pw = spec.real.pow(2) + spec.imag.pow(2)
        if multi:
            pw = pw.mean(dim=0, keepdim=True)
        if self.read_width > 0:
            k = 2 * self.read_width + 1
            pw = F.avg_pool1d(pw.transpose(1, 2), k, 1, self.read_width,
                              count_include_pad=False).transpose(1, 2) * k
        return pw

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

    def _relocate_moves(self, pw, floor, gfloor, scores, paths, rounds: int = 2):
        """Re-solve one slot against the others, and keep it iff coverage improves.

        This is coordinate ascent on `union_evidence`, and it exists because the
        measured stochastic failure is a DUPLICATE: on the `wide` cell two slots
        settled near 78 rev/s while the rotor at 71.6 went uncovered. No octave
        move can express that — the offending slot is not at a multiple of
        anything, it is simply in the wrong place — and the joint sweeps do not
        repair it either, because they use soft posteriors and accept whatever
        the mean field converges to instead of asking whether coverage went up.

        The move is worth making because the diagnostic says the information is
        present: on every stochastic clip tested, the union evidence at the TRUTH
        exceeds the union evidence of the decoded solution, which makes these
        cells a search wall rather than an evidence wall.
        """
        claims = self._claims_from(paths, scores[0])
        best = self.union_evidence(pw, floor, claims)
        for _ in range(rounds):
            improved = False
            for i in range(self.n_rot):
                others = claims.clone()
                others[:, i] = 0.0
                s_i = self._score(self._residual(pw, floor, others, None), gfloor)
                cand = comb_crf.viterbi(s_i, self.span, self.pen)
                trial = claims.clone()
                trial[:, i] = self._claims_from([cand], scores[0])[:, 0]
                j = self.union_evidence(pw, floor, trial)
                if bool((j > best + 1e-6).all()):
                    claims, best = trial, j
                    paths = list(paths); paths[i] = cand
                    scores = list(scores); scores[i] = s_i
                    improved = True
            if not improved:
                break
        return scores, paths

    def contrast(self, scores) -> torch.Tensor:
        """Per-frame ``max - median`` over the candidate grid: ``(B, T)``.

        The "is any rotor turning at all" statistic, and it has to be a CONTRAST
        rather than a level. The salience is a mean of ``log1p(power/floor)``, so
        its absolute value tracks how peaky the spectrum happens to be; measured
        on 8 s clips it reads 1.02 on white noise and 1.03 at the 5th percentile
        of real DREGON cruise, which no threshold can separate. The gap between
        the best candidate and the median candidate does not have that problem —
        it asks whether ONE rate explains the spectrum better than the rest do,
        which is what "a rotor is turning" means.

        Calibrated on 8 s clips, per-frame, mean (p05, p95):

            no rotors, white noise      0.240  (0.188, 0.305)
            no rotors, pink-ish         0.244  (0.194, 0.306)
            no rotors, near-silence     0.240  (0.190, 0.305)
            static comb, typical cell   2.782  (2.341, 3.235)
            stochastic comb, coherent   0.423  (0.292, 0.571)
            real DREGON cruise          0.416 to 0.549
            real FLY124 WARM-UP         0.294  (0.220, 0.376)

        The three no-rotor cases agree to three digits across a 10^6 range of
        input level, which is the floor-independence doing its job. The honest
        limit is the last row: a slowly turning rotor produces almost no contrast,
        so the zero and low regimes OVERLAP for any comb-based method, and a
        threshold that catches silence also silences part of the warm-up. That is
        a property of the signal, not of this statistic — a rotor at 3 rev/s puts
        its harmonics 3 Hz apart, below one 3.9 Hz analysis bin.
        """
        stack = scores if torch.is_tensor(scores) else torch.stack(scores, dim=1)
        if stack.dim() == 4:                      # (B, R, G, T) -> best slot
            stack = stack.amax(dim=1)
        return stack.max(dim=1).values - stack.median(dim=1).values

    def decode(self, audio: torch.Tensor, subgrid: bool = True,
               octave: bool = True, relocate: bool = True,
               zero_contrast: float = 0.0) -> torch.Tensor:
        """``(B, R, T)`` rates in rev/s, sorted ascending per frame.

        ``zero_contrast`` > 0 emits 0.0 for every slot in frames whose contrast
        falls below it — the project-wide "silence means zero rotor speed"
        convention that `salience_to_rps_segmented` already applies on the
        salience side. 0.30 is the 95th percentile of the no-rotor distribution
        above. The default is 0.0 (OFF), so every number measured before this
        existed still reproduces.
        """
        pw = self.spectrum(audio)
        floor = local_floor_torch(pw, self.floor_bins).detach()
        gfloor = self.gather(floor).clamp_min(1e-12)
        scores, _ = self.forward(audio)
        scores = [scores[:, i] for i in range(scores.shape[1])]
        paths, _ = self._solve(scores)
        for _ in range(2):
            if relocate:
                scores, paths = self._relocate_moves(pw, floor, gfloor, scores, paths)
            if octave:
                scores, paths = self._octave_moves(pw, floor, gfloor, scores, paths)
            if not relocate:
                break
        if self.k_refine:
            scores, paths = self._refine_band(pw, floor, scores, paths)
        grid = self.grid.to(scores[0].dtype)
        out = []
        for s, path in zip(scores, paths):
            r = grid[path]
            if subgrid:
                r = r + self._parabolic(s, path)
            out.append(r)
        rates = torch.stack(out, dim=1).sort(dim=1).values
        if zero_contrast > 0.0:
            quiet = self.contrast(scores) < float(zero_contrast)   # (B, T)
            rates = rates.masked_fill(quiet.unsqueeze(1), 0.0)
        return rates

    def _refine_band(self, pw, floor, scores, paths):
        """Re-solve each slot with the SHORT harmonic list, near its settled path.

        The band is what makes this safe: the short list is the more precise
        scorer and the more octave-prone one, so it is only ever allowed to move
        a rotor by `refine_band` rev/s, never to another octave.
        """
        gfloor_lo = self.gather_lo(floor).clamp_min(1e-12)
        claims = self._claims_from(paths, scores[0])
        grid = self.grid.to(pw.dtype)
        n_band = max(1, int(round(self.refine_band / float(grid[1] - grid[0]))))
        ar = torch.arange(len(grid), device=pw.device)[None, :, None]
        out_s, out_p = [], []
        for i, path in enumerate(paths):
            res = self._residual(pw, floor, claims, i)
            h = self.gather_lo(res)
            s_lo = self.head_lo(h, gfloor_lo, self.gather_lo.count, grid)
            keep = (ar - path.unsqueeze(1)).abs() <= n_band
            s_lo = torch.where(keep, s_lo, torch.full_like(s_lo, -1e30))
            out_s.append(s_lo)
            out_p.append(comb_crf.viterbi(s_lo, self.span, self.pen))
        return out_s, out_p

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
