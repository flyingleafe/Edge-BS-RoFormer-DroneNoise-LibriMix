# Work inventory since last report

- generated: 2026-07-27T13:33:07+01:00
- boundary artifact: writing/reports/2026-07-24_vk-parity-status
- boundary commit: 64343fa 2026-07-24 VK-parity report: G2 protocol results (HCQT refuted, IF marginal best)
- HEAD: b1d6d28 2026-07-27 g2_if_freqscale: matched transformer control for the freq-scale regime

> Numbers in docs below may predate later fixes — always cross-check the
> newest report/doc before quoting. Sync results before analysis (Rule 5).

## Commits (newest first)

```
b1d6d28 g2_if_freqscale: matched transformer control for the freq-scale regime
09bb432 CKLA: rotation-off control in the live-gain regime (the goal-deciding attribution cell)
f03e83c CKLA levers confirmed on full-envelope val (pnoise 44.8, freqscale 63.0 vs base 85.2): eval registry + combined pnfs arm
999f3d2 CKLA: 4s rotation attribution null (eval-time ablation on 4s-trained ckpt)
d0d6404 CKLA: p0_4s result (context null on static-comb) + queue-stall postmortem
b1e8db3 CKLA mechanistic levers: p_init knob (gain-collapse fix) + freq-scale-only aug arm
796c9b1 CKLA activation analysis: scan/layer state taps + 6-question mechanistic diagnostics
d83f269 CKLA P1 vk_eval: dregon 2.87 (floor 2.481), fly124 1.39 (new best, was 2.33); 4s + norot arms
3d965e8 CKLA: norot control result — rotation contributes nothing at 1s context
c83f1b4 vk_eval: register ckla_p1_if best/last checkpoints
98cbdf9 CKLA P0b results (transformer never locks; rotation ablation null) + 4s context pair
9da9f80 CKLA rotation=False control: exact real-KLA head variants + ckla_p0_norot arm
beb3035 CKLA batch doc: fair E8 rescore (85.4 vs CKLA 21.7 on clean valid), partial capture table
34902d6 CKLA batch doc: P0 result (7x faster on-distribution convergence), P0b/P1 job handles
9d7c49d P0b: capture/lock boundary diagnostic for RPS predictors
a9ca8c0 CKLA campaign: design doc, batch doc, P0/P1 experiment configs
cec249e CKLA: complex Kalman linear attention layer, model, registry + tests
0e583b3 Complex-OU layer exploration: learned coupled VK as stacked filtering layers
28601b3 G8a/G8a2 conclusions: warped-axis front-ends refuted as a class; C3 survives on linear axis
c9a1acb G8a2: dense band collapse on the pyramid front-end (channel-sparsity fix)
1f0baed G8a: multi-resolution STFT pyramid front-end with per-band IF (C1)
48c532a G6 conclusions: strong augmentation refuted on protocol; aug lever class closed
3d655d6 VK-parity eval: register G6 strong-aug checkpoints
4b29ce9 G8 design: multi-resolution pyramid + harmonic fusion + Fisher coarse-to-fine decode
0f01cb4 G7: ramped strong-augmentation schedule (plain -> mild p=0.3 -> full), submission-ready
ab2786c G5 conclusions: aug-from-0 refuted on both arms (transformer crashed) — warmup is load-bearing
15a4dcb G6: strong noise-augmentation family (6 transforms) + transformer/IF arms
aa38f9a G5: augmentation-from-sample-0 arms (warmup stage overlapped overfit onset)
538bb29 G4b refuted; criterion 2.3 ledger — cheap levers exhausted, parity not achieved
0f5be57 G4b: CoordConv row-f0 channel on the comb front-end (position readout fix)
8683e98 G4a result: refuted at val (position-readout diagnosis); G4b coord-channel next
3ba9a20 G4: comb matched-filter front-end — the VK whitened scan as a trainable input
```

## Experiment configs (conf/experiment/)

```
  A	conf/experiment/ckla_p0_4s.md
  A	conf/experiment/ckla_p0_4s.yaml
  A	conf/experiment/ckla_p0_4s_norot.md
  A	conf/experiment/ckla_p0_4s_norot.yaml
  A	conf/experiment/ckla_p0_norot.md
  A	conf/experiment/ckla_p0_norot.yaml
  A	conf/experiment/ckla_p0_staticcomb.md
  A	conf/experiment/ckla_p0_staticcomb.yaml
  A	conf/experiment/ckla_p1_4s.md
  A	conf/experiment/ckla_p1_4s.yaml
  A	conf/experiment/ckla_p1_freqscale.md
  A	conf/experiment/ckla_p1_freqscale.yaml
  A	conf/experiment/ckla_p1_if.md
  A	conf/experiment/ckla_p1_if.yaml
  A	conf/experiment/ckla_p1_norot.md
  A	conf/experiment/ckla_p1_norot.yaml
  A	conf/experiment/ckla_p1_pnfs.md
  A	conf/experiment/ckla_p1_pnfs.yaml
  A	conf/experiment/ckla_p1_pnoise.md
  A	conf/experiment/ckla_p1_pnoise.yaml
  A	conf/experiment/ckla_p1_pnoise_norot.md
  A	conf/experiment/ckla_p1_pnoise_norot.yaml
  A	conf/experiment/g2_if_freqscale.md
  A	conf/experiment/g2_if_freqscale.yaml
  A	conf/experiment/g4_comb_transformer.md
  A	conf/experiment/g4_comb_transformer.yaml
  A	conf/experiment/g4b_comb_coord_transformer.md
  A	conf/experiment/g4b_comb_coord_transformer.yaml
  A	conf/experiment/g5_augfrom0_if.md
  A	conf/experiment/g5_augfrom0_if.yaml
  A	conf/experiment/g5_augfrom0_transformer.md
  A	conf/experiment/g5_augfrom0_transformer.yaml
  A	conf/experiment/g6_strongaug_if.md
  A	conf/experiment/g6_strongaug_if.yaml
  A	conf/experiment/g6_strongaug_transformer.md
  A	conf/experiment/g6_strongaug_transformer.yaml
  A	conf/experiment/g7_ramp_if.md
  A	conf/experiment/g7_ramp_if.yaml
  A	conf/experiment/g7_ramp_transformer.md
  A	conf/experiment/g7_ramp_transformer.yaml
  A	conf/experiment/g8a2_pyramid_dense_transformer.md
  A	conf/experiment/g8a2_pyramid_dense_transformer.yaml
  A	conf/experiment/g8a_pyramid_transformer.md
  A	conf/experiment/g8a_pyramid_transformer.yaml
```

## Docs (docs/) — excerpts for added files

### ADDED: docs/ckla-design.md
```
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
```
### ADDED: docs/complex-ou-layer-exploration.md
```
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
```
### ADDED: docs/experiments/ckla-activation-analysis.md
```
# CKLA activation analysis — what the trained head actually computes

**Status:** done — 2026-07-26. Analysis note (no training); companion to the
CKLA batch doc [`ckla.md`](./ckla.md) and the design-§6 diagnostics kit in
`docs/ckla-design.md`. Tooling: `scripts/ckla_activation_analysis.py` +
the `return_state`/`capture_state` instrumentation added to
`src/models/ckla.py`. Raw numbers/figures: `results/ckla_activation_analysis/`
(gitignored; regenerate with the command below).

```
python scripts/ckla_activation_analysis.py \
    --data datasets/DREGON-LM-V4-michaels-full/valid   # or dload: URI
```

Protocol: 12 seeded clips (8 dregon_cruise + 4 fly124_cruise) from the
vk_valid_comparison table, mic ch0, 8 s, CPU fp32 no_grad. Models:
`ckla_p1_if` best (the P1 model), `ckla_p0_staticcomb` best (synthetic
reference for A1–A3), `g2_if_transformer` best (comparator for A5/A6).
Intact 12-clip PIT-MAE on this subset: ckla_p1 2.712, g2_if 2.779,
ckla_p0 5.064 (off-distribution, as expected).

## Mechanistic summary

The trained CKLA head is **a bank of fixed very-long-horizon evidence
integrators with a regime-level (not frame-level) input gate, plus a small
but real rotation contribution on DREGON**. Uncertainty gating in the
Kalman sense is essentially unused: within cruise clips the evidence
precision λ_v is near-constant (CV 0.03–0.17) and only weakly tracks
acoustics; its one strong behaviour is a step change at flight-regime
transitions (idle→cruise). State precisions λ never converge within an 8 s
```
### ADDED: docs/experiments/ckla.md
```
# CKLA — complex Kalman linear attention for RPS prediction

Batch doc for the CKLA architecture campaign (design:
`docs/ckla-design.md`; exploration + prior art:
`docs/complex-ou-layer-exploration.md`). Goal: either beat the neural
floor **2.481** (g2_if, dregon_cruise PIT-MAE on the vk_valid_comparison
protocol) by more than seed noise without regressing FLY124, or produce a
quantified definitive negative naming the mechanism (design §6 kit).

## Ladder

| stage | experiment | question | gate |
|---|---|---|---|
| P0 | `ckla_p0_staticcomb` | can the CKLA head track combs at all, at matched budget vs the E8 transformer arm? | train-dist PIT-MSE ≤ E8 transformer at common epoch; stable fp32 training; rotation path used (§6 diagnostics) |
| P0b | capture-boundary eval | where does lock break vs drift rate × SNR? | boundary at or beyond K2's collapse point |
| P1 | `ckla_p1_*` (E12 schedule, v4-michaels stream) | does it beat 2.481 on real cruise? | > seed-noise margin (~0.15), FLY124 ≤ 2.33 |
| P1a–e | ablation ladder (design §5) | which ingredient carries/fails? | — |

## P0 protocol

Identical to `e8_staticcomb_s1_transformer` in every field except
`model` → `simple_conv_v2_ckla_mag` (stft_mag front-end — isolates the
head; the E8 arms all ran stft_mag). Comparison numbers from the E8/E9
batch (post valid-cleanup, [[sim2real-rps-transfer-findings]]): the
on-distribution comparison is the *train* PIT-MSE trajectory at common
epochs (wandb), the transfer read is the fixed real valid.

## Results

### P0 — `ckla_p0_staticcomb` (kaggle `python-9d450c`, wandb `jcrr4tqe`, 2026-07-25)
```
- MODIFIED: docs/experiments/g1-vk-parity.md
### ADDED: docs/g8-hierarchical-frontend-design.md
```
# G8 — Hierarchical front-end: multi-resolution pyramid + harmonic fusion + Fisher decode

Design doc for the next VK-parity front-end phase (criterion 2.3). Grounded
in the 2026-07-24 literature sweep (bibliography tag `g7-frontend`) and the
G1–G6 evidence chain (`docs/experiments/g1-vk-parity.md`).

## Why (the evidence so far)

The single-window STFT (n_fft 2048 @16 kHz: 7.8 Hz bins, 32 ms hop-frames)
is caught in the classic resolution conundrum, and it lands differently per
harmonic:

- At the **fundamental** (30–120 Hz), 7.8 Hz bins are catastrophically
  coarse (a whole rotor-speed range spans ~12 bins), but the *signal* there
  moves slowly in Hz (1 rev/s error = 1 Hz at k=1) — what's needed is fine
  FREQUENCY resolution; time can be coarse.
- At **high harmonics** (k≈15–25, 1–2 kHz), 7.8 Hz bins are fine in rev/s
  terms (0.5 rev/s error = ~10 Hz at k=20), but the comb moves *fast* in Hz
  (the same RPS wiggle is k× amplified) — what's needed is fine TIME
  resolution and phase stability; frequency bins can be coarse because IF
  provides sub-bin readings.

One window cannot serve both. Constant-Q allocates resolution exactly
backwards (fine at the fundamental, coarse at high k — G2a refuted, 3.32
protocol / 195.5 val). The IF channel (G2b) is the only arm that beat the
baseline (2.481 vs 2.62) — phase evidence works; the comb-ridge front-end
(G4a/b) died at val. Severe overfitting on ~2 drones of real data (val
doubles within 20 epochs of best, all arms) means **parameter-light
structural priors beat learned modules** — which is also the literature's
converged position (LEAF filters barely move from init; free waveform
```

## Writing artifacts created/updated in the window


## Code changes (summary)

```
 tests/models/test_g8_pyramid_frontend.py   | 165 ++++++
 tests/test_noise_augmentations.py          | 289 ++++++++++
 18 files changed, 3787 insertions(+), 7 deletions(-)
      7 src/models
      3 src/data_processing
      1 src/tasks
      1 scripts/rps_predictor_vk_eval.py
      1 scripts/ckla_capture_boundary.py
      1 scripts/ckla_activation_analysis.py
```

## Untracked candidates (not yet committed)

```
  (none)
```

## Prep notes found (read these fully — often a ready-made narrative seed)

