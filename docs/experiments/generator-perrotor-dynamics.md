# The per-rotor-embedding paradox: a strict superset that trains worse

**Date**: 2026-08-12 · **Branch**: `tracking-opt` · **Status**: RESOLVED —
mechanism measured, remedy demonstrated. Follow-up of
`generator-refined-labels.md` §verdict item 3.

## The paradox

`gen_r2_*_perrotor` = `gen_r1_*` + per-rotor conditioning shifts
`z_r = z_drone + δz_r`, zero-initialized (64 of 245 905 parameters). The r1
model is the exact δz=0 point of the r2 space, yet the published r2
checkpoints have far weaker combs (k10-24 peak-to-floor +0.83 vs +4.22 dB,
refined pair). A superset that loses points at the optimizer, the objective,
or the selection — not at the architecture.

## Evidence chain (probes in `results/perrotor_probe/*.json`)

1. **α-sweep** (scale the trained δz by α ∈ [−0.25, 1.25], re-measure per-k
   PTF): every band moves ≤ 0.07 dB. The final deltas are render-inert; the
   comb deficit lives in the SHARED weights. Anatomy: |δz_r| up to 2.2·|z|,
   aligned with ±z (per-rotor gain directions), pair-structured.
2. **best vs last checkpoints**: the plain arms erode too (r1 refined k10-24:
   4.22 best → 2.52 last). Sharp combs are a TRANSIENT of training.
3. **Epoch-resolved replicas** (`gen_r1_refined_ep` 14 epochs,
   `gen_r2_refined_perrotor_ep` 30 epochs, `checkpoint_every=1`): both
   architectures form sharp combs at epoch 0 — r2's epoch 0 BEATS r1's
   (k25-49: 2.51 vs 1.57 dB) — then high-k combs erode while train loss falls
   monotonically.
4. **The monitor is the villain**: val/mrstft correlates with high-k comb at
   +0.19 (r1 run) and **−0.54** (r2 run). Best-by-mrstft picked ep5 (r1) and
   ep21 (r2, deeply eroded). The published original runs drew ep0 (r1 — the
   lucky draw) and ep7 (r2). Seed repeat (`seed=43`): the plain arm's best
   drops 4.22 → 2.38 dB at k10-24 — the published r1-vs-r2 gap is within the
   lottery's own spread.
5. **δz gradient anatomy at the comb peak** (r1 best embedded in r2 space):
   per-batch δz gradients are noise-dominated (cosine to the mean direction
   0.02-0.20); the mean direction is ±z with a rotor-pair structure
   (differential gain, 4× the common mode, 10× below single-batch noise).
   Delta-only descent (200 Adam steps): |δz| drifts to ~0.05, loss and combs
   unchanged — deltas alone are harmless. Their real effect is churn: Adam
   gives noisy coordinates full-size steps, the conditioning random-walks,
   the shared net chases it, the val-mrstft series gets noisier, patience
   resets, the run doubles in length, and the "best" epoch lands deeper in
   the erosion.
6. **Why MSSTFT erodes combs at all**: real high-k lines decohere within one
   2048-sample window (shaft-speed wander σ≈0.6 rev/s, τ≈16 ms → phase wander
   ≈0.24·k rad/window: ~12 rad at k=50; measured real PTF ≈ null above k≈25
   even on refined tracks). Per-window band magnitudes fluctuate around a low
   median; log-L1 fits the median, so a steady rendered line at high k gets a
   persistent downward amplitude gradient — measured directly:
   d(loss)/d(log-amp) > 0 (push down) at k25-80 and < 0 (push up) at k1-24,
   with jitter on or off. Energy migrates to the stochastic branch (coherent
   share 0.506 → 0.451 best→last). The global coherent gain is loss-optimal
   at γ≈1 (flat), so erosion is a reallocation, not a volume change.

## Verdict

The δz dimensions neither impede comb formation nor damage rendering. The
paradox is (erosion under a comb-diluted objective) × (checkpoint selection
by a comb-blind, in the r2 run comb-ANTI-correlated, monitor) × (delta-churn
lengthening runs and shifting the draw). With comb-aware selection the
superset wins, as theory demands: r2_ep's best-by-comb epoch (ep17: k10-24
7.16, k25-49 2.62, k50-80 0.91 dB) beats every r1 epoch.

## Remedies (in force order)

1. **Select checkpoints on the comb readout** (per-k PTF from
   `scripts/eval_gen_comb_real.py` logic, or a composite with mrstft). This
   repairs every arm and is demonstrated above.
2. Snapshot early epochs densely — combs peak within the first ~3 epochs.
3. Optional: weight-decay or early-freeze δz to cut churn (secondary; the
   deltas are otherwise harmless and may yet earn per-rotor timbre).
4. Deeper fix: a loss that defends lines (comb-weighted term / coherence-aware
   objective) — same lesson as the wind-channel post-mortem.

Caveats: one recording; eval chunks overlap training audio
(fitting-capacity comparison); one seed per published arm (variance
quantified); a production selection criterion should also penalize
overshoot (r2_ep's ep17 overshoots k1-9 by +10 dB) — |Δlog-mag| along truth
tracks is the safer composite.

Jobs: `gen-r1-refined-ep-2b3fba`, `gen-r2-refined-perrotor--11ed72`,
`gen-r1-refined-s43-90467a`, `gen-r2-refined-perrotor--f307b4`. Explainer
artifact source: session scratchpad `perrotor_explainer.html` (published
separately).
