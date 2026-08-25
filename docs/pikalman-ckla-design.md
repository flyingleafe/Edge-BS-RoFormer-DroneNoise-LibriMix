# HG-CKLA: the harmonic-gather CKLA cell (CKLA as a true pi_kalman pass)

Status: DESIGN, 2026-08-25. No implementation yet. Companion to
`docs/ckla-design.md` (the original CKLA layer) and
`src/tracking/phase_increment_tracker.py` (the classical algorithm this
mirrors). Part of the neural-RPS program (seed -> annealed refinement ->
heads).

## 1. Motivation

The CKLA campaign built a complex-OU Kalman linear-attention layer and used
it as a drop-in replacement for the Transformer temporal head: conv trunk
over log-magnitude(+IF) features -> frequency-attention pool -> CKLA
sequence mixer over pooled 128-dim frame vectors. The intent was "a layer
that does what one pi_kalman pass does". The built thing cannot do that,
for one structural reason: **pi_kalman's measurements are conditioned on
its own state** — it reads the spectrogram *at the harmonic positions its
current estimate predicts* — while the CKLA head receives measurements that
are state-independent, because the frequency pool collapses the spectral
axis before the recurrence ever sees it. After the pool there is no "at
harmonic k of rotor r" left to read. A sequence mixer over pooled features
can filter; it cannot implement an extended-Kalman measurement update.

This document specifies the smallest architectural change that makes the
original intent true: move the measurement inside the recurrence, as a
differentiable gather at state-predicted harmonic positions.

## 2. What pi_kalman actually computes

From `phase_increment_tracker.py`, per outer iteration and rotor (numbered,
because the neural cell must mirror each step):

1. **Demodulate at the state.** For each usable harmonic k, demodulate the
   audio by the predicted phase $k\hat\phi(t)$, lowpass to $\pm$band,
   decimate to the envelope grid: envelopes $z_k(t)$, all channels.
2. **Phase-increment measurement.**
   $d\psi_k(t) = \arg\!\big(z_k(t)\,\overline{z_k(t-1)}\big)
   \approx 2\pi\,dt\,k\,dr(t) + db_k + v$ — informative about the rate
   error $dr$ long after absolute phase decoheres.
3. **Per-measurement variance.** SNR term (high-SNR von Mises:
   $\sigma_v^2/|z_k|^2$, noise floor from an off-comb demodulation) plus
   the diffusion term $q_k\,dt$; $q_k$ estimated FROM THE DATA as the
   robust excess of residual variance over the SNR prediction.
4. **Fusion.** All surviving increments feed a scalar random-walk Kalman
   smoother (RTS) over frames; $\hat r \mathrel{+}= dr$; re-demodulate on
   the next outer iteration.
5. **Annealing.** Iteration j admits harmonics up to `k_caps[j]`
   (coarse-to-fine: while the residual is large, high orders alias and
   leave the demod band).
6. **Twin gating.** Per frame, measurements of rotor i's harmonic k are
   gated out where any other rotor's harmonic comes within band+guard;
   tight pairs get the joint two-phasor treatment.

The empirical anchors this construction earned: the WP18 weight law
($1/v_k \propto k^{2.0}$ on DREGON, $k^{1.5}$ on the MD2 rig), one-pass
convergence, and the capture-range limitation (refinement is inert when the
seed is off by more than the demod band).

## 3. Gap analysis of the tried CKLA

What the existing layer already supplies, mapped onto the algorithm:

| pi_kalman ingredient | CKLA analog | Status |
|---|---|---|
| Kalman gain (SNR-dependent) | input-dependent per-slot gain ($\lambda$) | present |
| phase advance of the state | input-dependent complex rotation | present |
| random-walk smoother over frames | complex-OU scan (bidirectional variant = RTS) | present |
| measurement AT the state (step 1) | — | **missing** |
| per-harmonic complex measurements (step 2) | — (pooled real features) | **missing** |
| k-structured variance + annealing (3, 5) | — | **missing** |
| twin gating (6) | — | **missing** |

The first missing row is the fatal one; the others follow from it.

## 4. The HG-CKLA cell

One cell = one neural pi_kalman iteration. L stacked cells = L outer
iterations, each re-gathering at the updated state.

**State** (per rotor r, carried by the recurrence): frequency $f_r(t)$ in
rev/s, a confidence scalar, and the CKLA complex slots (the learned
analog of the per-harmonic drift memory $b_k$).

**Measurement operator (the new piece).** From the complex STFT
$X(f, t)$ on the standard 2048/512 grid:

1. Positions: $p_{r,k}(t) = k \cdot f_r(t) / \Delta f$ for $k = 1..K$
   (rotation-rate harmonics, K about 25-40).
2. Soft gather: $\hat X_{r,k}(t) = \sum_f w(f - p_{r,k}) X(f, t)$ with a
   narrow Gaussian window $w$ — the analog of the demod band, and the
   term that keeps gradients local (a point interpolation would see only
   two bins; the window is the capture band).
3. Innovation phasor:
   $u_{r,k}(t) = \hat X_{r,k}(t)\,\overline{\hat X_{r,k}(t-1)}\;
   e^{-i\,2\pi\,k\,f_r(t)\,H/f_s}$ — its angle is the per-harmonic
   frequency-error signal $2\pi\,dt\,k\,df_r$; feed the cell the UNIT
   phasors $u/|u|$ together with $\log(1+|\hat X_{r,k}|)$, never an
   explicit $\arg$ (wrap-free, and no atan2 gradient blowup at small
   magnitude).
4. The linear-physics path: $\widehat{df_r}(t) =
   \sum_k w_k \, \mathrm{Im}(u_{r,k}/|u_{r,k}|) / (2\pi\,k\,dt)$ with
   $w_k$ initialized to the WP18 law $\propto k^2$ (normalized) and
   learned; a small MLP on the full phasor/magnitude stack adds a
   correction term. The CKLA gain modulates how much of
   $\widehat{df_r}$ enters the state — the Kalman gain, now fed by
   magnitude (SNR) inputs as in step 3 of the algorithm.

**Annealing.** Cell j applies a harmonic mask up to `k_caps[j]`
(e.g. 10 / 25 / 40 for a 3-cell stack) — fixed, not learned, exactly the
classical schedule.

**Twin gate.** A differentiable soft mask
$\sigma\big((|k_i f_i - k_j f_j| - \text{band})/\tau\big)$ down-weights
collided gathers, computed from the states themselves. The joint
two-phasor pair mode is explicitly out of scope for v1.

**Seeding and modes.** Two operating modes, sharing all weights:

- *Refiner*: the frequency state initializes from a conditioning track
  (telemetry, a coarse model, the blind seed) — the
  `SimpleConvV2CKLACond` contract, trained on corrupted-GT pairs from
  `data_processing/rps_corruption.py` with non-PIT MSE.
- *End-to-end*: an existing trunk (the HB winner) proposes per-rotor
  seeds; the HG-CKLA stack refines. PIT applies to the seed assignment
  once; the refinement is order-preserving.

**Voicing for free.** The gathered magnitudes are an off-state signal: a
gate $\sigma(g(\{|\hat X_{r,k}|\}))$ multiplies the output speed, which
derives the HB campaign's voicing gate from measurements instead of
bolting it onto the head.

## 5. Training plan

1. **Stage A (refiner, cheap, decisive):** corrupted-GT -> truth on the R2
   stream, non-PIT MSE, the existing conditional-refiner harness. This
   stage alone answers the core question (Section 6, gate G1/G2).
2. **Stage B (end-to-end):** seed head + HG-CKLA stack, R2 regime,
   PIT-MSE at the seed, plain MSE after assignment.
3. Optional bootstrap loop per the neural-RPS program: annotate unlabeled
   corpora with the refiner, retrain, repeat.

## 6. Gates (falsifiable, in order)

- **G1 (synthetic):** on synthetic combs with drift and known corruption,
  the stage-A refiner must either exceed pi_kalman's capture range or
  match its precision inside it. Failing both kills the design.
- **G2 (protocol):** `scripts/rps_eval.py --protocol beatvk --pred
  model:<refiner> ` vs `--pred telem --refine pi_kalman`: the neural
  refiner must reach parity on cruise windows (the classical pass is the
  ceiling here by construction) while running >10x faster on GPU.
- **G3 (frozen split):** end-to-end mode vs the HB winner on
  `DREGON-LM-V4-michaels-valid-full`, per-regime. The interesting cell is
  cruise MSE: this is the only architecture with a mechanistic reason to
  beat the 11-13 flight-MSE plateau.

## 7. Risks and answered objections

- **Capture range.** Gathers outside the comb produce noise-phasor inputs
  and uninformative gradients. Mitigations: the Gaussian gather window IS
  a capture band; the anneal schedule keeps early cells low-k (wide
  effective band); the seed head bounds the initial error.
- **The kalman-harmonic-tracker precedent.** That bet died at gate K2
  (drift robustness) because it phase-locked an EKF to absolute phase.
  HG-CKLA mirrors the *increment* likelihood with diffusion-as-noise —
  the exact construction that survived where the EKF died. The precedent
  argues for this design, not against it.
- **Compute.** Gathers are $O(R\,K\,T)$ with tiny constants; the CKLA
  scan already has the Triton kernel. The cell is cheaper than the conv
  trunk in front of it.
- **Twins.** v1 accepts the soft gate's precision loss on tight pairs
  (the classical joint mode stays ahead there); a pair-joint neural head
  is future work.
- **Complex STFT input.** The cell needs $X$ itself, so the front-end
  must expose the complex spectrogram alongside the trunk features — an
  additional front-end output, not a new front-end.

## 8. Reuse map

| Piece | Source |
|---|---|
| complex-OU scan + input-dependent gains | `src/models/ckla.py` (+ Triton kernel) |
| corrupted-track training harness | `SimpleConvV2CKLACond` + `rps_corruption.py` |
| conditioning codec | `tasks.codecs` cond path |
| weight-law init | WP18 ($1/v_k \propto k^2$) |
| anneal schedule | `pi_kalman` `k_caps` |
| eval | `scripts/rps_eval.py` protocols; the unified frozen-split probe |

New code: the soft gather operator, the innovation-phasor construction,
the HG-CKLA cell wiring, one front-end flag to expose complex STFT.

## 9. v1 implementation notes (2026-08-25, `src/models/hg_ckla.py`)

Built as specified, 221k params, 12 tests. Three recorded deviations: the
physics path uses a guarded atan2 on the unit phasor (the linearized
imaginary part under-reads 55% at 2 rad, so the design's Im(u/|u|) form
cannot pass the 10% gate); the voicing gate is deferred; the model computes
its own STFT. Two engineering findings that belong to the design:

- **The gather window must be phase-aligned.** `torch.stft` refers bin
  phase to the frame start, so a real Gaussian window sums the main lobe
  with alternating signs (24 dB of scalloping). The complex weight
  `gauss(d) * exp(i pi d)` gathers every tooth at full amplitude.
- **One gather serves both frames of an innovation pair** (at the position
  the state predicts for frame t). The window's own phase divides out
  exactly, so a moving state adds no bias — measured drifting-cond bias
  1e-5 rev/s.

Innovation-physics recovery error is below 0.01% for df in [0.2, 1.0]
rev/s. Stage-A training: `hb_hgckla_ref` on the R2 stream.
