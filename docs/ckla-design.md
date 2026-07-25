# CKLA — Complex Kalman Linear Attention for RPS prediction

Design of record for the complex-OU KLA architecture bet (goal: beat the
neural floor 2.481 on the vk_valid_comparison protocol, or produce a
quantified definitive negative). Follow-up to
`docs/complex-ou-layer-exploration.md` (prior-art map, risks) — this doc
pins the math, the parameterization, the model wiring, and the experiment
ladder. Substrate semantics: KLA (arXiv 2602.10743) as implemented in
`~/Projects/kla-loglinear/src/fkla/{reference,layer}.py`; we vendor the
*flat* recursion only (no Fenwick tree — our sequences are ~126–250 frames,
where a sequential fp32 scan is cheap and exact).

## 1. The scan op

State grid G = (N, D): N state slots × D value channels. Per (n, d) cell the
latent is a **complex** information pair (η ∈ ℂ, λ ∈ ℝ≥0). Per-step inputs
(all broadcastable to G):

    ā_t   ∈ ℂ   discretised complex-OU transition  ā = e^{−γ + i ω_t}, γ > 0
    p̄_t   ∈ ℝ≥0 discretised process noise
    φ_t   ∈ ℝ≥0 evidence precision   k_t² · λv_t
    κ_t   ∈ ℂ   evidence information k_t · λv_t · v_t   (v may be complex)

Flat recursion (information form; flat prior η = λ = 0), the *only* change
vs KLA being ā complex in the η numerator and |ā|² replacing ā² everywhere
in the real precision algebra:

    den_t = |ā_t|² + p̄_t · λ_{t−1}
    λ_t   = λ_{t−1} / den_t + φ_t                      (real — identical to KLA)
    η_t   = ā_t · η_{t−1} / den_t + κ_t                (complex — rotation here)

Why this is exactly right and not an approximation: for a complex-Gaussian
latent with transition z_t = ā z_{t−1} + ε, ε ~ CN(0, p̄), the predicted
covariance is |ā|²Σ + p̄ — rotation is unitary and drops out of second
moments. The KLA precision recursion is therefore untouched (and keeps its
proven scan algebra); the rotation acts only on the information vector,
where it multiplies as a unit-modulus scalar. Input-dependent ω_t breaks
none of this because λ never sees ω.

Readout (KLA precision readout, dead-zone at λ ≤ ε):

    μ_t[n,d] = η_t[n,d] / max(λ_t[n,d], ε)             (complex)
    y_t[d]   = Σ_n q_t[n] · μ_t[n,d]                   (complex)
    out_t    = W_o [Re y_t ; Im y_t]                   (2D → D)

Implementation: η stored as an explicit (re, im) pair of real fp32 tensors
(no torch complex dtype — avoids complex-autograd slow paths and keeps the
kernel trivially portable). Sequential loop over T with (B, N, D) tensors;
at T ≈ 126–250 loop overhead is negligible. **fp32 mandatory** below any
autocast (same discipline and cast points as `FenwickKLALayer.forward`).
Optional later optimization: two-pass form — λ path first (real Möbius
scan), then η as a first-order linear recurrence with known complex
coefficients c_t = ā_t/den_t (associative; parallelizable) — only if the
sequential loop ever shows up in profiles.

## 2. The layer

`ComplexKLALayer(d_model, n_state=16)` — scaffolding copied from
`FenwickKLALayer` (causal conv1d(k=4) + SiLU, QK L2-norm, λv softplus
+ 1e−4, gated residual, RMSNorm, out_proj), minus everything Fenwick
(levels, fold, buckets). New pieces:

- **Decay** γ[n, d]: softplus-inverse-stored, discretised ā_mag = e^{−γ·Δ}
  with learnable per-slot Δ as in KLA (a log-spaced S4D init, p init 0.01,
  Δ init log-uniform [0.001, 0.1]).
- **Rotation** ω_t[n] = ω0[n] + W_ω h_t, applied per slot (broadcast over
  d). ω0 initialized linearly spaced over [0, π] LRU-style (ring init);
  **W_ω zero-initialized** so training starts LTI (a complex-OU LRU with
  uncertainty) and input dependence grows only where gradients ask for it.
  ω unbounded (rotation is periodic; wrapping is free), but the projection
  output is scaled by a learnable per-slot gate s[n] init 0.1 to keep early
  rotation excursions small.
- **Value path**: v real (D channels) feeding κ_re = k·λv·v, κ_im = 0.
  The *state* is still complex — rotation mixes the two quadratures, so the
  layer can represent phase-accumulating structure even from real evidence.
  (A complex-v variant reading [re, im] channel pairs is a P1 ablation, not
  v1.)
- Output: `out_proj(cat[Re y, Im y])` with `nn.Linear(2*d_model, d_model)`.

`CKLABlock` = pre-norm residual block (mixer + MLP), identical shape to
`FenwickKLABlock`.

## 3. Model wiring

`SimpleConvV2CKLA` (model registry key `simple_conv_v2_ckla`): identical
trunk to `SimpleConvV2Transformer` — front-end → 6× ResidualConvBlock2d
encoder → FrequencyAttentionPool → (B, 128, T) — with the temporal head
replaced by `TemporalCKLAHead`: input proj 128 → d_model, `n_layers`
CKLABlocks over time, head conv → (B, 4, T). Defaults d_model = 128,
n_layers = 2, n_state = 16 (parameter budget ≈ the 2-layer transformer head
it replaces; report exact counts).

Front-end: `stft_mag_if` (G2b — the only front-end that beat baseline;
warped-axis fronts are refuted, do not revisit). Constructor takes
`frontend=None` → builds stft_mag_if by default for this model.

Stacking IS the capture→refine story: block 1 sees magnitude+IF evidence
and produces features whose rotation projection W_ω can encode a coarse
frequency belief; block 2's rotation is conditioned on block 1's output
(input-dependence across depth = closed-loop rotation, the piece K2
lacked). We do not hard-wire the (A,B) split in v1; if diagnostics show
rotation unused (see §6), the physically-tied variant (ω[n] = h_n · ω_base,
harmonic-number tying) is the first escalation.

## 4. P0 — synthetic drift gate (cheap, decisive direction check)

Task: blind single-target f0 tracking through drift, the scenario that
collapsed open-loop K2. Data generated on the fly (no dataset): 4-rotor
harmonic noise via `synth_rps`/`synth_rotor_noise`
(`src/experiments/kalman_harmonic/phase0.py`), white + speech-proxy
interference at SNR ∈ {+10, 0, −10} dB, drift = the same OU `corrupt_rps`
profile family. Model input: audio → stft_mag_if frames; target: per-rotor
RPS at frame rate; loss: PIT-MSE (existing task loss).

Arms (matched trunk, matched ~params): `TemporalCKLAHead` vs
`TemporalTransformerHead` vs `CausalGRUHead`. Small budget: d_model 64,
~200k steps of batches of fresh synthetic 4 s clips, one free-GPU job per
arm (kaggle/colab), ~2–3 h each.

Gate (from the exploration doc): CKLA (a) trains stably in fp32, (b)
matches or beats the transformer head on synthetic PIT-MAE, (c) degrades
gracefully with SNR rather than cliff-collapsing, (d) diagnostics show the
rotation path is *used* (‖W_ω h‖ significantly above init scale, ablating
rotation → measurable loss increase). Failing (a/b) at matched budget on
*synthetic* data — where data volume is unlimited and the task is exactly
the model's inductive bias — is a strong early negative that P1 cannot
plausibly reverse; it triggers the diagnosis ladder (§6), not P1.

## 5. P1 — real protocol

Train `simple_conv_v2_ckla` with the E12 two-stage schedule (50k plain
warmup + weak augs — the load-bearing schedule per G5) on the same
online-mix v4-michaels stream as every G-series arm; eval
`scripts/rps_predictor_vk_eval.py`, pools dregon_cruise / fly124_cruise.
Bars: neural floor **2.481** (g2_if), VK blind bars 0.68–0.74 / 1.027.
"Significantly better than previously trained best" = beats 2.481 on
DREGON cruise without regressing FLY124 above the g2_if 2.33, by a margin
larger than the seed noise of the G-series arms (~0.1–0.15).

Ablation ladder if the base arm underperforms (each is one job, ordered by
information/cost):
1. rotation ablated (s[n] frozen 0) — isolates whether complex path helps
   at all vs a real-KLA head;
2. n_layers 3 / d_model 192 — capacity;
3. harmonic-tied ω (physical structure);
4. complex-v from [mag, IF] pairs;
5. front-end swap stft_mag_if → stft_mag (interaction check).

## 6. Quantified-failure diagnostics (the definitive-negative kit)

If every arm loses, the negative must name the mechanism with numbers:

- **Rotation usage**: distribution of ‖ω_t − ω0‖ per layer; fraction of
  slots whose s[n] grew vs shrank; loss delta from rotation ablation at
  eval. If rotation is unused → the claim "input-dependent rotation helps"
  is refuted for this task/trunk; report the ablation delta.
- **Precision dynamics**: λ_t trajectories (does the model exploit
  uncertainty, or does λ saturate → plain linear attention?); p̄ learned
  values vs init.
- **Capture behavior**: on synthetic eval clips with controlled drift,
  per-clip lock/no-lock classification (error < 2 rev/s sustained) vs
  drift rate and SNR → an empirical capture boundary to compare against
  K2's collapse point and the EKF-PLL prediction. If the learned layer has
  the same boundary as K2, the closed-loop hypothesis is refuted
  quantitatively.
- **Overfitting position**: best-epoch val curve vs the G-series pattern
  (best at ep 8–18, val doubles by ~ep 40). If CKLA shows the *same*
  data-pool-bound overfitting, the negative is "architecture class cannot
  fix a data problem" — quantified by train/val gap at matched epochs.

## 7. Constraints

- Jobs: free GPUs (kaggle P100 / colab T4) + uni-cpu only; nothing
  CPU-heavy on the laptop. omnirun from a clean pushed HEAD.
- fp32 for all scan math regardless of autocast.
- No dependency on the kla-loglinear checkout at runtime — the op is
  self-contained in this repo (cluster jobs can't see ~/Projects).
