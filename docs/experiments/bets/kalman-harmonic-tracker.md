# Bet — RPS-driven Kalman harmonic tracker ("complex-KLA" filter)

Status: **killed at Phase-0 gate (K2)** (2026-07-10; H1 passed via the joint
per-order variant, H2 refuted — see Running notes for the verdict and the
salvage lessons for the KLA-layer ambition).
**Checkpoint write-up:** [`../kalman-harmonic-tracker-phase0.md`](../kalman-harmonic-tracker-phase0.md)
(motivation/results/conclusion + the ranked mitigation list for any revival).
Owner: DM · Time budget: **Phase 0 = 3 working days**, hard-capped; Phase 1
only if Phase 0 passes its gate.

## One-line idea

Replace the framed `lstsq_VP_transform` with a *causal, recursive,
per-harmonic complex Kalman filter* whose rotation is driven by measured RPS
and whose process noise explicitly models RPS/phase error — i.e. a
tachometer-synchronized adaptive comb filter with statistically optimal,
time-varying per-harmonic bandwidth, later made learnable KLA-style.

## Background / provenance

The KLA paper ([Kalman Linear Attention, arXiv:2602.10743](https://arxiv.org/abs/2602.10743))
shows a diagonal linear-Gaussian Kalman filter with token-dependent
parameters is a valid, scan-parallelizable sequence-mixer layer, with the
posterior precision acting as an uncertainty-driven gate. Nothing in its
information-form algebra requires a real transition: with a complex
`a_t = exp((−γ + i·2π·f_h(t))·Δt)` the precision/Möbius track depends only on
`|a_t|²` (rotation drops out of the uncertainty recursion; the 2×2 scan
matrices stay real) while the mean track gets a complex forget gate
`f_t = a_t/(|a_t|² + p_t·λ)` — rotation passes through the gate untouched.
Complex-diagonal recurrences of exactly this `exp(−γ+iθ)` form are standard
and trainable at scale ([LRU, arXiv:2303.06349](https://arxiv.org/abs/2303.06349);
[S4D, arXiv:2206.11893](https://arxiv.org/abs/2206.11893)). The scan algebra
(incl. the composition lemmas) is machine-verified in Lean for the scalar
case in the sibling project `~/Projects/kla-loglinear/lean/`.

Classical anchors (why this mechanism is sound, and our baselines-in-spirit):

- **Kalman with known time-varying basis ≡ recursive least squares** —
  the filter *is* `lstsq_VP_transform` computed recursively with
  uncertainty-weighted exponential forgetting
  ([Sayed & Kailath, 1994, doi:10.1109/79.295229](https://doi.org/10.1109/79.295229)).
  With frequencies supplied by RPS, variable projection collapses to linear
  LS with a time-varying basis — the Kalman setting exactly.
- **Adaptive comb filtering for harmonic retrieval**
  ([Nehorai & Porat, 1986, doi:10.1109/TASSP.1986.1164952](https://doi.org/10.1109/TASSP.1986.1164952)).
- **Tacho-referenced engine-order cancellation** (synchronized-reference
  FxLMS in automotive ANC; survey: [Kuo & Morgan, 1999,
  doi:10.1109/5.763310](https://doi.org/10.1109/5.763310)) — the industrial
  proof that "plug the rotation sensor into the filter" works.
- **PLL ≈ Kalman phase tracker**: the filter's steady-state gain is the
  optimal loop bandwidth given phase-drift rate vs measurement noise.
- Learned-oscillator precedent: [DDSP, arXiv:2001.04643](https://arxiv.org/abs/2001.04643)
  (synthesis side); ours is the analysis/tracking side.

## Hypothesis

**H1 (sanity):** With oracle RPS, a per-harmonic complex Kalman tracker
(diagonal information filter, heterodyne/demodulated form) removes rotor
harmonics from DREGON-LM-style mixtures at least as well as the framed
`lstsq_VP` projection at matched harmonic count — despite being causal,
streaming, O(R·H) per sample, and lstsq-free.

**H2 (the actual bet):** Under realistic RPS *error* (slow drift / jitter —
the regime where accumulated phase error kills coherent methods), the
tracker's enhancement quality degrades gracefully (process noise `p` absorbs
phase drift; bandwidth widens automatically), while the framed lstsq fit
degrades sharply with window length × drift. Quantitatively: at ~1 % slow
RPS drift the tracker retains ≥ 70 % of its oracle-RPS SI-SDR gain while
lstsq loses > 50 % of its.

If H2 holds, this directly de-risks **Bet 1 (pseudo-RPS)**: the tracker is
by construction the robust consumer of noisy pseudo-RPS.

## Model (Phase-0, no learning)

Per rotor r and harmonic h: demodulate the mic signal by the RPS-integrated
phase (`y = 2·v·e^{−iφ_h(t)}`, φ from cumsum of `2π·h·B·RPS/sr` in float64),
then run a scalar complex information-form Kalman step per sample:

```
λ⁻ = λ/(ā² + p·λ);  η⁻ = (ā/(ā²+p·λ))·η      # predict (ā = e^{−γ/sr})
λ  = λ⁻ + q;        η  = η⁻ + q·y_t           # update (q = measurement precision)
```

Demodulated form ⇔ rotating-latent form via `z = c·e^{iφ}` (change of
variables); demodulated is numerically nicer. Noise estimate = Σ Re(ĉ·e^{iφ})
using the *predicted* (prior) mean → strictly causal subtraction, no
self-leakage of the current sample. Physical parameterization: γ [1/s] =
amplitude coherence rate; `p ∝ h²` (phase error scales with harmonic index);
`q` from broadband noise floor. Speech = measurement noise of this model;
the innovation is the enhanced signal.

Known approximation: **diagonal filter = projection, joint lstsq = joint
solve**. Four rotors at near-identical RPS give nearly-coincident combs that
independent trackers fight over (same failure class as `neighboring_freqs`
hack). Phase-0 measures this; the cheap fix (Phase 2) is a 4×4 joint update
per harmonic order across rotors.

## MVP (Phase 0) — `src/experiments/kalman_harmonic/`

Synthetic-first (rotor noise via our own `oscillator_bank`, so ground truth
is exact), then one DREGON-LM chunk:

1. Mixture = harmonic rotor noise (4 rotors, wandering RPS) + speech(-proxy)
   at −10 dB.
2. Enhance by harmonic subtraction with (a) Kalman tracker, (b)
   `harmonic_lstsq_VP_transform` + inverse. Metric: SI-SDR vs clean speech
   (`metrics.separation.si_sdr`), plus unprocessed floor.
3. **Robustness sweep**: corrupt RPS with slow multiplicative drift of
   relative magnitude σ ∈ {0, 0.2, 0.5, 1, 2, 5} %; plot SI-SDR vs σ for
   both methods. H2 = tracker curve visibly flatter.
4. Sweep `p` at fixed σ to show the bandwidth/robustness knob does what the
   theory says (optimum p tracks σ²).

Runner: `python -m experiments.kalman_harmonic.phase0` (synthetic default;
`--speech <wav>` optional; DREGON hook stubbed for day 2).

## Kill criterion

- **K1**: oracle-RPS tracker SI-SDR < lstsq − 1 dB on synthetic *after* one
  round of γ/p/q tuning → the diagonal approximation is too lossy → try the
  per-order 4×4 joint update once; if still failing, **kill**.
- **K2**: H2 fails (tracker no flatter than lstsq under drift) → the core
  robustness claim is false → **kill** (the causal/streaming advantage alone
  does not justify a bet slot).

## Phases beyond MVP (only on green)

- **Phase 1**: make `(q_t, p_t, k_t)` outputs of a small encoder
  (KLA-style token-dependent gates; "don't update while speech dominates"),
  train end-to-end vs P2 architecture on DREGON-LM. Complex-KLA layer =
  drop-in mixer.
- **Phase 2**: 4×4 per-order joint update; pseudo-RPS input (Bet-1 synergy);
  MIMII fan/pump regime for C4 generality.

## Constraint check

- C1/C4: mechanism is "RPS-instrumented periodic interference" generic —
  parameters in physical units (γ [1/s], p from sensor spec, RPM-agnostic).
- C7: this card; budget above; mid-point review applies.
- Compute: Phase-0 is CPU/laptop-scale.

## Running notes

- 2026-07-10 — card drafted; Phase-0 scaffolded in
  `src/experiments/kalman_harmonic/` (filter + synthetic runner). Not run yet.
- 2026-07-10 — **Phase 0 executed; bet killed at K2.** Full data:
  `results/kalman_harmonic_phase0{,_joint}/`. Synthetic scene per MVP (4
  rotors ~90 RPS, 25 harmonics, ±30 % AM at 0.5–1.5 Hz, speech proxy at
  −10 dB, 4 s; SI-SDR scored past a 0.5 s warm-up and the lstsq frame tail,
  same region for all methods).

  **Scaffold fixes before science**: (a) `inverse_VP_transform` returns
  per-rotor waveforms → sum over rotors; (b) `si_sdr` needs 2D input; (c) the
  drafted flat prior (`init_prec=1e-4`) made all R·H channels claim the whole
  signal at t=0 — a 1/t over-subtraction transient that alone produced −30 dB
  SI-SDR. Principled fix: λ₀ = K/(2·var), i.e. prior matches mixture power
  spread over channels. (d) γ=10/s exceeded the tracking bandwidth → the
  steady-state amplitude estimate is biased low by k/(1−ā(1−k)) ≈ 0.1 — γ
  must sit well below k·sr; tuned optimum γ ≈ 0.03–0.1/s.

  **H1 / K1 — PASS, but only with the per-order R×R joint update.** Oracle
  RPS, after the one budgeted tuning round (γ, p grid): diagonal tracker
  +2.07 dB vs lstsq_VP +5.60 dB (unproc −9.59) — the diagonal approximation
  costs 3.5 dB, exactly the predicted "coincident combs fight" failure. K1's
  prescribed escape (per-order joint update, `kalman_harmonic_track_joint`:
  scalar sample observed through the R-rotor steering vector, rank-1
  Sherman–Morrison update; the diagonal filter is this minus off-diagonal
  covariance) recovers +4.62 dB — within 1 dB of lstsq, causal, streaming,
  lstsq-free. Notably its optimum p is ~15× wider than the diagonal's: the
  joint solve removes the speech-double-counting penalty of bandwidth.

  **H2 / K2 — FAIL (the core robustness claim is false).** Under slow
  multiplicative RPS drift (OU, τ=0.5 s), both filters *collapse faster than
  lstsq*, not slower. Joint tracker, best (p_base, p_h2) per σ vs lstsq
  (recorded sweep, `results/kalman_harmonic_phase0_joint/`):
  σ=0.2 %: −2.83 vs **+2.02** (retains 48 % of oracle gain vs lstsq's 76 %);
  σ=1 %: −7.22 vs −6.79 (17 % vs 18 % — both dead). H2 required ≥70 %
  retention at 1 %. (For completeness: at σ≥2 % the matched tracker does edge
  past lstsq — −7.39/−7.75 vs −7.30/−8.02 at 2 %/5 % — i.e. its curve is
  eventually flatter, but only after both methods have collapsed to ~2 dB of
  gain at −10 dB SNR; this does not rescue H2's claim where enhancement is
  meaningful.) The p-knob *does* behave as theorized (optimum p_h2 grows
  with σ, h²-scaling beats uniform widening by ~3 dB at σ=0.2 %) — but it
  cannot rescue the mechanism, because **isotropic process noise is the wrong
  model for a systematic rotation**: a frequency error of Δf makes the target
  phasor rotate coherently at 2π·h·Δf; a first-order loop must widen its
  bandwidth past that rate to follow, and at σ=0.2 % that is already ~0.4·h
  Hz — by h≈25 the required bandwidth admits more speech than the harmonic it
  removes. Framed lstsq re-fits amplitudes from scratch every 128 ms window
  (effective bandwidth ~8 Hz, *jointly* over all 100 components), which under
  drift is simply a better bias-variance point than any recursive
  exponential-forgetting tracker with fixed rotation.

  **Salvage — what this means for the KLA-attention-layer ambition** (the
  reason this bet existed): the measured failure is structural, not a tuning
  artifact, and it localizes precisely:
  1. *Diagonal linear attention over near-coincident oscillators is
     insufficient* — the joint R×R (matrix-valued state per order) update is
     worth 2.5 dB at oracle. A KLA-style layer for this problem needs
     block-diagonal, not scalar-diagonal, state; rank-1 information updates
     keep it scan-friendly (matrix Riccati/Möbius composition).
  2. *Uncertainty gates (p_t, q_t) cannot absorb RPS error* — the layer must
     be able to correct the **rotation itself** (token-dependent complex
     transition a_t with a learned Δphase, i.e. a learned PLL discriminator /
     second-order loop). This is representable in the KLA parameterization
     (a_t is already token-dependent) but was explicitly out of Phase-0's
     no-learning scope — and without it the mechanism loses to a framed
     re-fit at 0.2 % RPS error, far below real tacho/pseudo-RPS accuracy.
  3. *Strict causality was the premise that failed*, and it is also the
     harshest deployment constraint; a fixed-lag smoother (~100 ms lookahead,
     = chunked bidirectional scan in layer form) would reclaim much of the
     lstsq advantage while staying streaming. Untested here (budget).

  Verdict: the Phase-0 *filter* is dead as a bet (K2 explicit: the
  causal/streaming advantage alone does not justify a slot). Any revival must
  lead with learned rotation correction (point 2) — which is a different bet,
  with H2 replaced by "a small encoder can regress Δphase from the
  innovation", and should cite this card's numbers as its baseline.
