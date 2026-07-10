# Bet — RPS-driven Kalman harmonic tracker ("complex-KLA" filter)

Status: **drafted** (2026-07-10, not yet started; Phase-0 code scaffolded)
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
