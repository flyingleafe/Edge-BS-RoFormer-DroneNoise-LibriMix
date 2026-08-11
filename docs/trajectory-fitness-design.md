# A trajectory-fitness measure fit for optimization — literature synthesis and design

**Status:** design record, 2026-08-10. No code yet. Source: three parallel literature
sweeps (order tracking / IAS; pitch and differentiable DSP; continuation and GNC),
each run against the project bibliography plus the web. Full per-paper detail lives
in the sweep transcripts; this document keeps the load-bearing facts and the design
they force.

**The demand this answers** (user, 2026-08-10): the ridge statistic is too specific
and gradient-dead. Candidate measures must be compared on (1) global optimum at the
true RPS trajectories and (2) monotone degradation with distance from truth. A
measure with these properties, if differentiable, turns tracking into standard
non-convex optimization.

---

## 1. The verdict of the literature, in five facts

### Fact 1 — the basin law is closed-form and appears three independent times

A fit measure that sums coherent evidence over harmonics `k = 1..K` on a window of
length `T` seconds has a basin (main lobe) in `f0` of width

```
Δf0 ≈ 1 / (K · T)          [Hz]
```

Three literatures derive the same law without citing each other:

| Family | Statement | Source |
|---|---|---|
| Harmonic NLS main lobe | grid must be `5·N·L` points → lobe `f_s/(N·L)` | Nielsen et al., Signal Processing 135 (2017); EUSIPCO 2016; code `jkjaer/fastF0Nls` |
| Vold–Kalman passband | speed error tolerated: `B_3dB/(2k)` | Tůma bandwidth relations; B&K Tech. Review 1 (1999) |
| PLL lock-in range | ≈ loop bandwidth / k | Gardner; Kuznetsov et al. (lock-in range) |

At `K = 80`, `T = 1 s`: **0.0125 Hz**. The 40–100 Hz search interval then contains
~4,800 local maxima per comb. DREGON's measured label error (0.35–0.85 % of ~60 Hz)
sits **16–40 basin-widths** from truth. This one number explains the high-k
decoherence, why `pi_kalman` is inert outside capture range, and why any gradient
method needs a seed or a continuation schedule.

The efficiency side of the same coin: `var(f0) ∝ σ²/(N³·Σ_k k²A_k²)` — the high
harmonics carry the precision. The narrow basin is the price of the precision;
they are the same phenomenon.

### Fact 2 — sub-multiple aliases cannot be removed by any smoothing (the nesting argument)

The comb at `f0/m` with `mK` harmonics **contains the true comb as a subset**. It is
a nested model: its residual is the true residual plus whatever noise the empty
slots collect. Under any energy-sum measure the sub-multiple mode is essentially as
tall as the true mode, **independent of K**, so no continuation schedule and no
basin shaping ever kills it. (Derives Klapuri's "half or twice" observation and
YIN's 8.6 %-low / 0.18 %-high error asymmetry from first principles.)

Consequences for the two required properties:

- **Global optimum at truth** is achievable only with an explicit counter-term:
  a model-order (Occam) penalty (g-prior Bayes factor — Nielsen/Christensen/Jensen,
  TASLP 2013), or a two-sided likelihood that charges for predicted lines landing
  on no energy (Duan/Pardo/Zhang, TASLP 2010: +7.3 % median accuracy and balanced
  octave errors; same idea as Maher–Beauchamp two-way mismatch, JASA 1994).
- **Monotone degradation** is achievable *within the smooth envelope between lobes*
  and *within a lobe* — not across the sub-multiple lattice. At the aliases the
  measure needs a penalty or a discrete move, not a slope.

Discrete escapes that compose with continuous optimization:

- **Multiplier moves** `u ∈ {1/3, 1/2, 2/3, 3/2, 2, 3}` — rescale trajectory,
  re-fit envelopes, keep the best (Walmsley/Godsill/Rayner, WASPAA 1999). Six
  evaluations per rotor per restart.
- **Coprime harmonic subsets** (robust Chinese remainder theorem, from radar
  staggered-PRF: Xia & Wang line, IEEE T-SP 2007–2010). A subset like {7, 11, 13}
  cannot share a common divisor with an alias; error bound gcd/4 is proved.
  Integer arithmetic, so not differentiable — use as proposal generator and
  consistency test. Never applied to rotor combs — citable transplant.

Refinements from the Bayesian-harmonic deep dive (Walmsley/Godsill/Rayner
EUSIPCO 1998 + WASPAA 1999; Davy & Godsill TR.431; Nielsen TSP 2014; Shi 2019):

- The degeneracy is **exact**, not approximate: the f0/2 basis contains the f0
  basis, so the fit term `R²` is *identical* at f0/2 (Walmsley 1998, verbatim).
  The Occam factor is the only tie-breaker — per added partial, `(1+g)^{-(I+1)}`.
- **Warning: the adaptive Occam factor weakens at low SNR.** In the Bayesian
  treatments `g` is sampled and shrinks with SNR, weakening the octave penalty
  exactly where we need it. Our penalty must be fixed/calibrated (on synthetic
  truth), not evidence-adapted.
- **Discrete moves must correct the order jointly**: a bare f0 → f0/2 proposal
  is almost always rejected; the working move halves f0 AND doubles the partial
  count, preserving the spectral structure (TR.431 divide/multiply moves;
  Walmsley multiplier set `{1/3,1/2,2/3,3/2,2,3}` with `H ← ceil(H/u)`).
- **Temporal priors do not rescue a bad landscape**: Shi 2019's tracked-vs-
  per-frame table shows the tracker fixes isolated octave jumps but moves the
  aggregate error rate by ≤0.02 at −5…0 dB. The error floor is the per-window
  landscape — invest in the measure, not the smoother.
- **Twin collinearity is diagnosed but unbounded in all prior art**: when a
  harmonic of one source nears a harmonic of another, `GᵀG` goes
  ill-conditioned and amplitude estimates blow up (EUSIPCO 1998 fn. 2); no
  CRLB or resolution limit for two close harmonic sources exists anywhere in
  the strand. The VK smoothness term is precisely a regularizer of this ridge
  (the VK literature's "orders coincide → ill conditioning → widen bandwidth"
  is the same phenomenon) — characterizing the twin-resolution limit of the
  regularized objective is another citable first.
- Transferable priors: AR-prewhitening of the residual (colored broadband noise
  is modeled, not ignored); low-pass per-partial amplitude prior
  `k_m = 1/(1+(Tm)^ν)`; and Cemgil 2006's alternative octave defence — a
  *learned per-partial amplitude covariance* (timbre prior) breaks aliases where
  the landscape cannot. For rotors: a measured/learned comb-envelope prior
  (which the generator work already provides) is an alias tie-breaker
  independent of the order penalty.

### Fact 3 — the differentiable whole-trajectory objective already exists, unused

The Vold–Kalman cost is quadratic in the envelopes for a fixed trajectory.
Substituting the closed-form envelope solution back gives the **profiled residual**

```
F(φ) = min_a ‖y − Σ_m Re[a_m c_m(φ)]‖² + Σ_m ρ_m²‖Δ² a_m‖²
     = yᴴy − yᴴ C(φ) [C(φ)ᴴC(φ) + SᵀRᵀRS]⁻¹ C(φ)ᴴ y
```

with `c_m(t) = exp(j2πk_m ∫ r_{i(m)})`. Properties:

- Smooth in the trajectory; gradient available in closed form (envelope theorem;
  `∂C/∂φ = jC·∂Θ/∂φ`, diagonal) or by autograd through the solve.
- The regularization weight `ρ` maps analytically to a bandwidth (Tůma), so the
  capture range is a *knob with units*, not a hyperparameter.
- The off-diagonal coupling blocks `conj(c_m)c_n` make near-equal rotors compete
  for the same energy — **the only published mechanism** that breaks twin-rotor
  ambiguity; a per-rotor comb score has no equivalent.
- Without the smoothness term it reduces to the exact harmonic-subspace projector
  `yᴴP(φ)y` — which is simultaneously the ML estimator, and (after g-prior /
  Jeffreys marginalization) the Bayesian MAP: the amplitude-marginalized posterior
  depends on the trajectory only through this statistic (Nielsen et al., TASLP
  2013, eq. 58). So the "assume candidates true, solve, score the residual"
  proposal is the principled object — the failure of our components 1–3 was the
  *normalization* (bounded shares that saturate), not the construction.

**Nobody optimizes it over the trajectory.** The VK literature takes the speed as
tachometer-given. IAVKF (Jiang/Chen/Wang, IEEE T-II 2024) states outright that
optimizing the cost over the IF is "not feasible". Forty years of sinusoidal
modeling (McAulay–Quatieri → HMM/LP partial tracking) stops at discrete peak
assignment. The continuous step over a whole trajectory is a named, verifiable gap.

Caveat from the same sweep: use the **exact projector**, not the FFT harmonic-sum
approximation — Nielsen et al. warn the approximation degrades precisely for small,
closely-spaced frequencies (the twin-rotor case).

### Fact 4 — spectral losses are refuted as the outer objective; the fixes are known

- Turian & Henry (NeurIPS-W 2020): gradient-sign ranking accuracy (GRA) of a plain
  spectrogram distance in f0 is 0.617; mel is 0.511 — a coin flip — on a *single
  clean sinusoid*. Multi-scale spectral reaches 0.771 analytic / 0.978 coarse but
  keeps fine-grained local minima.
- Torres/Peeters/Richard (ICASSP 2024): gradient descent on default MSS lands
  **−2.3 octaves** off on average (RPA 20 %); their spectral-optimal-transport
  loss (1-D Wasserstein along frequency) fixes it to RPA 75 % / median 99.7 % —
  the only measure found whose value is proportional to the frequency error at
  arbitrary distance. Needs a support cutoff and a vertical guard term; per-comb
  masking needed for multi-source.
- Schwär & Müller (IEEE SPL 2023): most of MSS's pathology is *configuration*.
  Flat-top window (low sidelobes → no leakage ripple), `log(1+γ|·|)` instead of
  `log(ε+|·|)`, squared L2, non-power-of-two window sizes → **30× lower**
  frequency error in the same pipeline. Basin width = mainlobe overlap: window
  length is a basin knob.
- Hayes/Saitis/Fazekas (ICASSP 2023): time-domain MSE with a damped
  complex-exponential surrogate reaches the CRLB from a single random init;
  DFT-magnitude MSE floors at −83 dB² — the magnitude spectrum discards phase.
  The damping `|z| < 1` widens lines and is discovered by the optimizer: **soft
  annealing**. Documented failure mode: several model components collapse onto
  one target component — the twin-rotor risk, named and reproducible.
- DDSP line (Engel et al. 2020 + review 2023): consensus that f0 is "not directly
  solvable by straightforward gradient descent over an audio loss"; the two
  published escapes are parameter-regression pretraining on synthetic audio with
  exact labels (= our generator bootstrap) and the complex surrogate.

### Fact 5 — the continuation axes and what theory promises

- **K (harmonic count / taper) is the principled axis**: basin `1/(K·T)` is known
  in closed form at every level — something no GNC paper can offer. Soft version:
  per-harmonic damping/taper (Hayes' `ρ^k`); hard version: truncation ladder
  (K: 4→8→…→80). VK bandwidth `ρ` and window `T` are the other two axes.
- **GNC robustness weights are the orthogonal axis** (Yang et al., RA-L 2020, via
  Black–Rangarajan duality): closed-form per-harmonic weights `w_k`, annealed
  geometrically (μ-step ≈ 1.4), handle masked/decohered harmonics. This does NOT
  widen the f0 basin — robustness continuation, not resolution continuation. Our
  WP18 measurement (optimal weight `1/v_k ~ k^-2`) is the measured version of the
  ML harmonic weighting (Peeters et al., MSSP 2022) and sets the *starting*
  weights; phase-noise coherence time falls like 1/k, so fully coherent
  integration over k=1..80 is unreachable — high k must enter with measured
  weights or block-incoherent averaging.
- **Theory**: global convergence of continuation is proved only for "σ-nice"
  functions (Hazan/Levy/Shalev-Shwartz, ICML 2016) — unverifiable here. The
  practical acceptance test: sweep the anneal parameter, check the argmin path is
  continuous and the coarse level has a single basin containing truth. Coarse
  levels favor sub-multiples (nesting argument), so the coarse stage must carry
  the penalty term and/or an external seed (large-displacement-flow lesson:
  Brox & Malik 2011 — structures lost at coarse scale never return; inject
  discrete candidates instead of coarsening further).
- Trajectory-level globalization: exact DP on a rotor-speed grid is our existing
  blind Viterbi; a differentiable relaxation exists (log-sum-exp Bellman —
  Mensch & Blondel 2018; temperature = trajectory-level basin knob) if a smooth
  trajectory-level score is ever needed end-to-end.

### Fact 6 — the trajectory level has its own laws (DP pitch tracking + track-before-detect)

From the DP-pitch and radar/sonar track-before-detect sweep:

- **The SNR wall.** In DP over trajectories, the best noise-only path grows as
  `sqrt(2·ln M)` per frame (M = transition branching factor), independent of the
  number of frames (Bampton/Ma/Do, arXiv:2512.11170; empirically Tonissen &
  Evans 1996: below the wall, "detection performance does not improve regardless
  of how many frames"). Per-frame merit SNR must exceed the wall: 3.4 dB at
  M = 3 (±1 bin), 7.4 dB at M = 16. Comb summation over K harmonics buys
  ~`10·log10 K` dB (~16 dB at K = 40) — this is the arithmetic that makes
  −20 dB broadband tracking feasible, and matches passive-sonar practice
  (4 simultaneous lines Viterbi-tracked at −20…−24 dB, Luo & Shen 2019).
- **The bin-matching rule** (continuous-gravitational-wave searches, Suvorova/
  Sun/Melatos line): choose the frame length so the per-frame coherent bin width
  `1/(2·T_frame)` equals the maximum per-frame wander — then the transition
  neighborhood is ±1 bin, which maximizes per-frame SNR and minimizes the wall
  at the same time. The most actionable single design constraint found.
- **Transition term: convex, log-domain.** Use `λ·(Δ log f)²` (Kaldi pitch
  tracker — convexity of the transition cost gives a near-linear exact search).
  Do NOT copy RAPT's octave branch (a deliberate second basin at the octave —
  its `double_cost` structure makes an exact octave jump *cheaper* than a 1.7×
  jump). PEFAC's capped quadratic is the robust variant if genuine RPS steps
  must stay cheap.
- **Expect a plateau, not a spike.** Detection is strictly easier than
  localization: the DP uncertainty radius grows O(k) even when detection is
  solid (Tonissen & Evans; Bampton et al.). Gradient refinement of a whole
  trajectory wanders along a near-flat plateau of near-optimal trajectories
  unless the smoothness prior pins it.
- **Calibrate the null with Gumbel, not Gaussian.** The DP-accumulated value
  under noise is extreme-value distributed (Tonissen & Evans via EVT). The
  detection threshold on `F_total` over a trajectory search must use it.
- **pYIN's lesson**: carrying multiple calibrated candidates into the trajectory
  stage beats improving the per-frame decision (pYIN vs YIN+smoothing: the whole
  gain). Its hard failure: Viterbi can never recover a pitch absent from the
  candidate set (live-recording recall 0.750 vs 0.99 elsewhere) — keep candidate
  generation generous, prune at the trajectory level.
- **Architecture precedent** (SLASH, Interspeech 2025): discrete global search
  to get inside the basin, then continuous gradient refinement of the whole
  trajectory against a differentiable score — exactly the seed → anneal → F_VK
  plan of §2.

---

## 2. The design that follows

One layered measure family, each layer answering one failure mode:

```
F_total(φ) = F_VK(φ; ρ, w)  +  λ_order · Ω(φ)          [core + alias penalty]
   scheduled over  (K_eff taper, ρ bandwidth, GNC μ)     [continuation]
   globalized by   seed (Viterbi/neural) + multiplier & permutation
                   moves + coprime-subset consistency     [discrete layer]
```

- **Core** `F_VK`: profiled coupled-VK residual, torch autograd through
  `vk_envelopes` (the solver exists; R1 made the stack torch-only — the gradient
  is nearly free). Per-harmonic noise weights start at the measured `1/v_k`.
- **Alias penalty** `Ω`: model-order/Occam term or Duan-style non-peak charge —
  mandatory (Fact 2). This is the component the ridge could not be and the
  shares never had.
- **Continuation**: anneal K_eff from ~5 (basin ≈ 0.2 rev/s at T = 1 s — wide
  enough to swallow the DREGON label error) to 80; co-anneal ρ (bandwidth) and
  optionally T. Acceptance: argmin-path continuity.
- **Discrete layer**: existing blind Viterbi seed; Walmsley multiplier proposals;
  rotor-permutation swaps; coprime-subset consistency check as a cheap veto.
- **Coarse alternative** (if the seed is weak): SOT term on masked support, or a
  comb-weighted spectral first moment (Turian's log-centroid scores GRA 1.000 at
  both scales) — globally oriented, locally noisy; hand off to F_VK inside the
  basin.

Role separation with the existing instruments:

- `fitness.ridge` stays what it is: a calibrated **lock detector / verdict
  instrument** (0 dB null, controls, holdouts). It validates a final trajectory;
  it is not an objective.
- Components 1–3 (residual shares) are superseded by `F_VK` — same algebra,
  correct normalization and weighting, plus the penalty term.

## 3. The benchmark (measures as landscapes)

Ground truth: synthetic combs (controlled SNR, phase noise per WP18 law),
generated-noise pool (exact labels), FLY124 cruise (real, proven 2.7 dB lock).
Candidates measured: F_VK (±penalty, ± weights), ridge, broadband share,
harmonic sum, SOT, MSS (fixed config per Schwär–Müller) as the published baseline.

Per measure, four figures of merit:

1. **G** — global-optimum margin over the structured alias set: sub-multiples,
   half-integer combs, twin swaps, constant scale/offset, time shift.
2. **M** — directional basin profiles `F(truth + a·d)` for canonical directions
   `d`, amplitudes 0.01–5 rev/s; Spearman between distance and score for random
   smooth perturbations.
3. **GRA** — gradient-sign ranking accuracy at step sizes ~0.3–300 cents
   (the Turian/Schwär standard; cheap; one number per step size).
4. **Continuation validity** — argmin-path continuity and coarse-level basin
   count under the (K_eff, ρ) schedule.

Predictions to falsify: ridge = sharp G-margin, dead M/GRA beyond 0.1 rev/s;
shares = the opposite; F_VK without penalty fails G at f0/2; F_VK with penalty +
schedule passes all four.

## 4. Paper narrative (contribution 1, restated)

1. Negative result, measured: share-type residual fitness saturates (6c/6d record)
   and spectral losses are landscape-broken (literature, Fact 4).
2. The measure: profiled coupled-VK residual with measured noise weights and an
   order penalty — MAP-equivalent, differentiable, twin-aware; landscape
   characterized by the 1/(K·T) law and the benchmark of §3.
3. The consequence (contribution 2): tracking as continuation-scheduled
   non-convex optimization of that measure — the continuous step the VK and
   sinusoidal-modeling literatures never took (IAVKF quote as the foil).
4. The validation instrument: the ridge detector with its nulls/controls scores
   the *outcome* (lock, in dB above floor), independent of the objective.

Two short citable side results the sweeps identified as unstated in the
literature: (a) the joint law — basin shrinks as 1/K while the sub-multiple score
stays a constant fraction under energy sums; (b) coprime harmonic subsets for
rotor-comb alias rejection (robust-CRT transplant from radar).

## 5. Key references (verified in sweeps; none added to the bibliography DB yet)

Core: Vold & Leuridan SAE 931288 (1993); Feldbauer & Höldrich DAFx-00; Tůma
(bandwidth); Herlufsen et al. B&K Review 1/1999; Jiang et al. IAVKF, IEEE T-II
2024. NLS/Bayes: Nielsen/Christensen/Jensen TASLP 2013; Nielsen et al. Signal
Processing 2017 + EUSIPCO 2016; Christensen & Jakobsson (2009); Elvander &
Jakobsson TSP 2020 (OMT surrogate, sub-octave statement). Aliases: Duan et al.
TASLP 2010; Maher & Beauchamp JASA 1994; Walmsley et al. WASPAA 1999; Klapuri
TSAP 2003; Xia/Wang robust CRT line. Landscapes: Turian & Henry 2020; Schwär &
Müller SPL 2023; Hayes et al. ICASSP 2023; Torres et al. ICASSP 2024 (SOT);
Hayes et al. review, Frontiers 2024. Continuation: Blake & Zisserman 1987; Black
& Rangarajan IJCV 1996; Yang et al. RA-L 2020; Hazan et al. ICML 2016; Mobahi &
Fisher AAAI 2015; Brox & Malik TPAMI 2011; Mensch & Blondel ICML 2018. IAS:
Leclère/André/Antoni MOPA MSSP 2016 (degenerates for integer combs — non-
commensurate orders are its alias immunity); Peeters et al. MSSP 2019 review;
Peeters et al. MSSP 2022 (ML harmonic weighting); Hua & Hao ARC 2026
(curvature-adaptive prior, keeps octave alternatives alive).
