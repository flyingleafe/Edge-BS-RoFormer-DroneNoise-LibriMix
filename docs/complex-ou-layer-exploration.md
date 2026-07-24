# Complex-OU filtering layers with input-dependent rotation — exploration

Design-space exploration (2026-07-24) prompted by: `src/experiments/kalman_harmonic`
(the killed K2 tracker), KLA (arXiv 2602.10743), and the hunch that a *complex*
OU latent — decay AND rotation, rotation input-dependent, speed errors
time-correlated — stacked with proper dynamics, is the right layer family for
audio→RPS. Literature: bibliography tag `complex-ou-layer` (27 papers; digest
in the session log of 2026-07-24).

## The idea, formalized

Take KLA's probabilistic sequence layer (information-form Kalman filter,
Möbius-scan parallel, real diagonal OU prior) and make the latent **complex**:

    z_t | z_{t-1} ~ CN( ā_t z_{t-1}, p̄_t ),   ā_t = e^{(−γ + i ω_t) Δt}

with the rotation rate ω_t **input-dependent** (for harmonic k of rotor r:
ω = 2πk·f̂0_r(t)), and f̂0's *error* itself a slow real OU process estimated by
the next layer up. Three structural facts make this attractive:

1. **Rotation is free for scan parallelism.** KLA's parallel structure lives
   in the precision recursion, which sees only |ā|² = e^{−2γΔt}; the unit-
   modulus rotation multiplies the mean/information path, and products of
   unit complex scalars are associative. An input-dependent-rotation KLA
   keeps the entire Möbius-scan machinery AND its beyond-linear expressivity.
2. **K2 was one layer of this, run open-loop.** `kalman_harmonic/filter.py`
   is literally the complex-OU filter with oracle rotation (demodulated
   coordinates; its docstring already cites KLA). Its kill (drift collapses
   it; diagonal channels fight over twin combs) names the two missing
   pieces: closed-loop rotation (the hierarchy) and structured cross-channel
   coupling — both of which the classical VK has and this design imports.
3. **Rotation-carrying input-dependent transitions are exactly what buys
   state-tracking expressivity** (independent line of evidence): negative
   eigenvalues → parity (Grazzi+ ICLR'25); Householder products →
   permutation groups (DeltaNet/DeltaProduct); PD-SSM's permutation ×
   complex diagonal → FSAs; diagonal LTI provably stuck in TC⁰ (Merrill+
   ICML'24; Ran-Milo+ NeurIPS'24 for the real-vs-complex separation).

## Prior-art map (what exists, what doesn't)

| ingredient | exists as | missing there |
|---|---|---|
| complex-OU latent layer | LRU / S5 / (D-)LinOSS — LTI, static phases | input dependence, filtering semantics |
| input-dependent complex rotation | **Mamba-3** (damped rotation, trapezoidal disc.) | no belief state, no noise gating, heuristic |
| scan-parallel Bayesian filtering layer | **KLA** | real diagonal only — no rotation |
| complex AR(1) as spectrogram prior | Turner–Sahani 2014 (probabilistic TF analysis), PAD's slow-over-fast hierarchy | shallow, fixed, generative-only |
| state-dependent complex eigenvalues | Lusch/Kutz/Brunton Koopman AE (aux net predicts λ, ω) | forecasting, not filtering; no stack |
| hierarchical frequency-error estimation | classical dual/adaptive Kalman (Wan & Nelson), PLL loop filters, adaptive comb (Nehorai–Porat 1986 = two-level cascade) | **no deep-learning instance at all** |
| the whole combination | — | **unclaimed**, incl. the "learned coupled VK" framing (order tracking never met SSMs) |

## Candidate architecture (the learned VK, causal and trainable)

Two coupled layer types, stacked/unrolled:

- **Layer A — envelope bank** (fast complex OU): per-(rotor, harmonic)
  filtering channels in demodulated coordinates, rotation set by the current
  f̂0 trajectory from below/previous iteration; per-channel learned γ, process
  noise with the physical h² scaling (K2's robustness knob), KLA-style
  precision gating. Cross-channel coupling for near-coincident frequencies
  (twins) — start diagonal + a structured pairwise correction, NOT a free
  joint update (the K2 kill).
- **Layer B — frequency layer** (slow real OU on δf0): driven by Layer A's
  per-channel phase residuals (posterior-mean rotations), Fisher-fused with
  w_k ∝ k²·SNR_k exactly as VK's update; belief-state precision gives a
  principled per-rotor confidence output. This is dual estimation as a
  *layer*, with time-scale separation (γ_B ≪ γ_A) as the stability
  discipline.
- **Stack N (A,B) pairs** = unrolled capture→refine; anneal effective
  bandwidth over depth like VK's k_max schedule. Readout: f̂0 posterior mean
  (+ variance). Optional analysis-by-synthesis auxiliary loss via the
  Turner–Sahani generative reading (the model *is* a generative spectrogram
  prior, so reconstruction is free supervision).
- Training: scan-parallel (Särkkä-style associative filtering + KLA Möbius
  scans; implementation substrate exists in ~/Projects/kla-loglinear).
  Inference: O(1)/sample streaming, causal — the deployment profile VK
  lacks (non-causal, rtf ~1).

## Risks (literature-warned) and mitigations

1. **Trainable-phase instability / low precision**: LRU log-magnitude +
   ring-near-unit-circle parameterization; Gated KalmaNet's conditioning
   gates; run fp32 (bfloat16 KF recursions are documented unstable).
2. **Capture range / cycle slipping** (EKF-PLL theory: convergence only
   near truth; harmonic/subharmonic capture at low SNR; washout ⇒ zero
   gradient when rotation is wrong): keep VK-style annealing over the
   stack; init Layer B from a coarse scan (our band-capped comb scan is
   the natural free initializer); expect multimodal loss — curriculum from
   high SNR / synthetic.
3. **Bilinear hierarchy pathologies** (dual-EKF bias/divergence;
   ∂a/∂ω phase-amplitude gradient mixing — the reason D-LinOSS decouples
   damping from frequency): strict time-scale separation, decoupled
   (γ, ω) parameterization, freeze-alternate training schedule if joint
   training misbehaves.

## De-risk plan (cheap, staged)

- **P0 (synthetic, CPU)**: 2-layer (A,B) stack, single rotor + drifting f0,
  known SNR sweep. Gate: tracks blind through drift where K2 (open-loop)
  collapsed; graceful degradation to the EKF-PLL capture boundary.
- **P1 (real, protocol)**: 4 rotors + twin coupling + N=2..3 unrolls,
  trained on the E12 stream, evaluated on the vk_valid_comparison protocol.
  Gate: beat the neural floor 2.481; the interesting threshold is ~1.0
  (FLY124 blind-VK bar) since this architecture can in principle do what
  VK does causally.
- Relation to other threads: independent of the front-end ledger (this
  replaces the trunk, not the front-end); complementary to VK-distillation
  (a data program feeds any architecture; this one has the right inductive
  bias to need less data).
