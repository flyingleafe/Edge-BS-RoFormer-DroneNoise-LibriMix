# Unified baseline evaluation on the frozen validation split

Status: IN PROGRESS 2026-08-24.

## Motivation

Every baseline family in this project was scored on a different set: the
May-2026 classical baselines on 10 DREGON-LM test samples, the June salience
baselines on `DREGON-LM-V4/valid`, the OT multi-pitch baseline and the
neural models on `DREGON-LM-V4-michaels-valid-full`. The numbers are not
comparable. This campaign scores every family once, on one split, with one
protocol.

## Protocol (frozen)

- Split: `dload:DREGON-LM-V4-michaels-valid-full` — 37 clips x 8 channels.
- Grid: per-frame predictions on the 2048/512 STFT grid at 16 kHz.
- Matching: per-frame Hungarian PIT on |pred - target|.
- Regimes from the target: zero (max rotor < 1 rev/s), flight (mean >= 45),
  low (the rest). Report MAE and MSE per regime and overall.
- This is the exact protocol of `results/m3cur_regime_probe/regime_probe.py`.

## Rows

| Family | Method(s) | Source of numbers | Status |
|---|---|---|---|
| Classical (May 2026) | PYIN, cepstral, HPS, matched filter, NMF | `src/experiments/classical_rps/valid_eval.py` (restored from `00753c4`; report `writing/reports/2026-05-29_classical-baselines/`) — local run 2026-08-24 | done |
| OT multi-pitch | Björkman & Elvander 2026 (adapted config) | `results/otmp_protocol/` (run 2026-08-23) — re-aggregated under this table | done, merge |
| Salience (June ckpts) | multif0_salience, basic_pitch_salience | old V4-trained checkpoints, if loadable from the zoo | optional |
| Salience (retrained) | `hb_sal_multif0`, `hb_sal_bp` | retrained on the hb online regime (same data + augmentations as the HB grid) | pending training |
| Neural direct (current) | scv2 / transformer-IF / uni_gru128 (real-only fs_v2, m3cur s2, m3abl comb s2) | `results/m3cur_regime_probe/regime_probe.json` | done, merge |
| Neural direct (HB grid) | 10 hb_* runs | regime probe after the campaign lands | pending training |
| Blind two-stage tracker | vit2dsp ladder (guarded), refusal -> 0 | `scripts/blind_valid_row.py` — annotation running on uni-cpu (blind-valid-row-cc0ecd) | running |

## Notes

- The classical methods and the OT baseline are training-free; the salience
  retraining is the only arm whose training data changed since its June
  numbers (fixed V4 mixtures then, hb online stream now), so its old and new
  rows must both appear.
- Blade count for the classical methods is 2, harmonics 15, grid constants
  identical to the project defaults — no adaptation was needed.

## Regime-matched reruns (R2-R5 at fixed architecture)

The regime taxonomy (R1 architecture search, R2 real-only honest, R3 gen+comb
curriculum, R4 comb-only curriculum, R5 mixed one-stage) is only readable when
one thing changes per row. The as-run R3/R4/R5 rows do not satisfy that: their
real component is the plain fs_v2 pool, while R2 adds a zero-labeled silence
arm and an SNR reference floor. A gap between R2 and R3 today mixes two
causes — the synthetic ingredient and the real recipe.

The reruns make the regimes nested. Every cell keeps ONE real component, the
R2 honest pool, and differs only in the synthetic ingredient and its schedule.
Three design decisions:

- Fixed architecture. The comparison runs at the ORIGINAL trio model configs
  (`simple_conv_v2`, `simple_conv_v2_transformer_if`,
  `simple_conv_v2_uni_gru128`) — plain registry models with no voicing gate
  and no front-end override. The HB grid changes the model and the regime at
  the same time, so it cannot serve as the R2 row; two ungated controls fill
  the gap next to the existing `hb_scv2_mag_nogate`.
- Reused warm starts. Stage 1 of R3 and R4 is unchanged (the synthetic task
  did not move), so the stage-2 reruns warm-start from the EXISTING
  `m3cur_<arch>_s1` and `m3abl_comb_<arch>_s1` checkpoints. Only the fine-tune
  stream changes. This halves the compute: 11 runs, not 17.
- Preserved ratios. The mixed pool keeps real : generated : comb = 2 : 1 : 1
  and silence : real = 0.4 : 2.0, thus the shares become real 45.5%, silence
  9.1%, generated 22.7%, comb 22.7%.

Two new policies:

- `conf/online_mix/hb_m3s2_dload.yaml` — the R2 pool with the 50k warm-up
  stage removed (the m3cur stage-2 argument: a converged model must not get
  50k samples on which the RPS prior wins again). Used by R3 and R4.
- `conf/online_mix/hb_m3mixed_dload.yaml` — the m3abl_mixed pool re-based on
  R2. Used by R5. Its generated source starts a CUDA producer, thus this
  stream does not run on a CPU-only box.

The eleven experiments:

| Regime | scv2 | transformer-IF | uni_gru128 |
|---|---|---|---|
| R2 real-only honest, ungated | `hb_scv2_mag_nogate` (exists) | `r2hb_tr_nogate` | `r2hb_gru_nogate` |
| R3 gen+comb curriculum, stage 2 | `r3hb_scv2` | `r3hb_tr` | `r3hb_gru` |
| R4 comb-only curriculum, stage 2 | `r4hb_scv2` | `r4hb_tr` | `r4hb_gru` |
| R5 mixed one-stage | `r5hb_scv2` | `r5hb_tr` | `r5hb_gru` |

All eleven validate on the frozen split under the protocol above, so their
rows drop straight into the leaderboard.

## Results

### Classical five (2026-08-24, full 1480-unit run, `results/classical_valid_eval/`)

Per-frame Hungarian PIT MAE / MSE in rev/s on valid-full:

| method | zero | low | flight | all |
|---|---|---|---|---|
| PYIN | 90.8 / 9341 | 46.7 / 3159 | 34.1 / 2031 | 43.0 / 3110 |
| cepstral | 95.5 / 9859 | 67.5 / 5539 | 19.6 / 839 | 35.7 / 2617 |
| HPS | 64.1 / 4660 | 27.6 / 1196 | 20.9 / 623 | 27.3 / 1212 |
| matched filter | 87.2 / 8686 | 66.8 / 5988 | 30.4 / 1531 | 42.5 / 3039 |
| NMF | 83.8 / 7631 | 59.5 / 4411 | **8.1 / 188** | 24.7 / 1701 |

Reading: (1) the zero regime is a structural failure for all five — the
search grids clamp to [50, 150] rev/s, so a stopped rotor scores near the
clamp floor; a silence decision is outside these methods' vocabulary.
(2) NMF at cruise (8.1 MAE) is the best training-free number we have —
better than the OT multi-pitch baseline's 16.3 on the same frames. The
May-2026 ranking (NMF best classical) holds under the unified protocol.
(3) Every learned model (2.4-3.6 flight MAE) stays 2-4x below NMF at
cruise and 10x+ below everything classical on zeros.

### OT multi-pitch (merged from `results/otmp_protocol/`, 2026-08-23)

Flight 16.3 / 544, low 28.3 / 1526, zero 68.8 / 5438, all 24.5 / 1292
(MAE / MSE). Same split, same PIT protocol (8-channel units).

### HB grid — first rows (scv2 trunk, gated, 2026-08-24 probe)

Per-frame Hungarian PIT MAE / MSE, `results/hb_regime_probe/probe.json`;
best val/mse from W&B history in parentheses:

| run | zero | low | flight | clean-off rate |
|---|---|---|---|---|
| hb_scv2_mag (39.7) | 3.68 / 62.2 | 10.68 / 167.4 | 2.57 / 13.4 | 0.76 |
| hb_scv2_if (34.9) | 4.50 / 121.2 | 4.99 / 83.1 | 2.49 / 12.8 | 0.84 |
| hb_scv2_ssq (47.1) | 6.04 / 183.6 | 6.05 / 104.0 | 2.52 / 14.2 | 0.79 |
| hb_gru_mag (60.7) | 10.62 / 306.6 | 3.99 / 58.1 | 2.66 / 20.0 | 0.65 |
| hb_gru_if | 7.20 / 197.0 | 6.11 / 123.6 | 2.52 / 13.3 | 0.75 |
| hb_gru_ssq (39.8) | 4.57 / 102.8 | 6.14 / 121.1 | 2.67 / 15.5 | 0.83 |
| hb_tr_mag (31.7) | 3.70 / 80.2 | 4.24 / 60.8 | 2.79 / 19.8 | 0.84 |
| hb_tr_if (33.6) | 3.55 / 86.3 | 4.51 / 71.8 | 3.08 / 18.4 | 0.84 |
| hb_tr_ssq (36.7) | 4.42 / 118.1 | 5.44 / 89.7 | 2.55 / 14.5 | 0.81 |

Reading (updated as cells land): the honest regime alone closes most
of the zero gap (real-only 11.83 -> 3.68-6.04 MAE) at almost no cruise cost
(flight MSE 12.8-14.2 vs real-only 11.6). The mag front-end regresses on
ramps (10.68 vs real-only 4.83); the IF front-end holds them (4.99) and has
the best aggregate (34.9) — between real-only (52.5) and the comb curriculum
(21.1). The comb curriculum still leads on aggregate, consistent with the
coverage story: the synthetic stage supplies unlimited ramp/transition
trajectories that 34 s of real ramps cannot match.

The gated 3x3 grid is complete (aggregates): scv2 39.7/34.9/47.1,
transformer 31.7/33.6/36.7, gru 60.7/50.5/39.8 (mag/if/ssq). The
front-end winner is architecture-dependent: IF for scv2, plain magnitude
for the transformer, synchrosqueezed for the causal GRU — where ssq fixes
most of the zero deficit (10.62 -> 4.57 MAE, drift mass 0.10, the lowest
of the trunk): sharp reassigned evidence substitutes for the missing
future context. Overall HB winner: hb_tr_mag (31.7).

Architecture notes: hb_tr_mag (31.7 aggregate, best HB cell so far) beats
its real-only control (42.3) in every regime with no ramp regression.
hb_gru_mag shows the opposite trade: ramps improve (6.51 -> 3.99) and
zeros get worse (7.01 -> 10.62, drift mass 0.35) — the causal head has no
future context to confirm silence, so the gate hesitates.

### Blind two-stage tracker (vit2dsp, 2026-08-25, `results/blind_valid_row/`)

Annotated on the four parent recordings (20 s windows, 8-channel seed +
spatial joint Viterbi), stitched, scored on the 37 clips. Two refusal
conventions, both recorded:

| convention | zero | low | flight | all | note |
|---|---|---|---|---|---|
| ungated | 79.36 | 39.10 | **2.27** (RMSE 4.99) | 17.01 | finds combs in silence |
| gated g1+g5, refusal -> 0 | **0.01** | 29.82 | 48.35 | 39.72 | 8/20 windows accepted |

Compute: 9.87 CPU-s per second of audio (about 10x realtime on CPU).
Reading: ungated, the tracker beats every neural cell on flight MAE
(2.27 vs 2.49-3.08) while losing on RMSE (4.99 vs ~3.6 — occasional bad
windows) and failing silence completely. The gates give a perfect silence
decision and reject over half the cruise windows — they were calibrated
for pseudo-label precision, and recall was never their target. The paper
row needs both conventions or a recalibrated gate; flagged for the author.

### Salience retrained on R2 (landing)

| run | zero | low | flight |
|---|---|---|---|
| hb_sal_bp (Basic Pitch) | 34.04 / 2071 | 13.32 / 874 | 31.69 / 19172 |
| hb_sal_multif0 (standard grid) | 52.77 / 4037 | 21.02 / 912 | 4.01 / 77.3 |
| hb_sal_multif0_nsr (widened 20-130) | 48.21 / 3668 | 16.13 / 602 | 4.68 / 128.4 |

Basic Pitch stays broken on the honest regime — the June verdict is
architectural, and retraining does not rescue it. (Probed through the
fixed zero-decode path.) The widened narrow-SR grid improves ramps
(21.0 -> 16.1) and leaves silence broken (48.2 vs the 2.25 GT round-trip
floor): the salience family lights bins on content-rich silence — a model
limitation, not the grid clamping. Best salience cruise (4.0) stays ~1.7x
behind the neural cells (2.4).

### Regime-rerun cells (landing)

| run | zero | low | flight | best val/mse |
|---|---|---|---|---|
| r2hb_gru_nogate | 6.02 / 173.7 | 8.17 / 164.4 | 3.06 / 26.8 | 61.9 |
| r3hb_gru | 4.89 / 118.9 | 5.47 / 118.2 | 2.79 / 17.8 | 41.8 |
| r3hb_scv2 | 3.86 / 53.8 | 4.27 / 47.6 | 2.49 / 13.1 | 22.6 |
| r5hb_scv2 | 16.08 / 653.5 | 9.14 / 181.3 | 5.20 / 68.6 | 147.6 |
| r4hb_gru | 6.19 / 123.5 | 4.86 / 74.7 | 2.85 / 17.6 | 37.6 |
| r5hb_tr | 5.46 / 132.3 | 5.37 / 85.7 | 4.25 / 47.7 | 59.2 |
| r5hb_gru | 10.50 / 333.6 | 8.18 / 166.4 | 3.81 / 32.5 | 85.8 |
| hb_scv2_mag_nogate | 3.36 / 59.3 | 4.18 / 52.2 | 2.35 / 11.1 | **22.1** |
| r2hb_tr_nogate | 5.52 / 143.2 | 5.14 / 94.0 | 2.65 / 16.0 | 41.8 |

RESOLUTION (2026-08-25): r3hb_scv2 (gen+comb curriculum ON TOP of the
honest base, ungated) lands at 22.6 against the plain honest base's 22.1
— the curriculum adds nothing for the headline trunk, in every regime.
The synthetic-data claim resolves to its strong form for scv2 (coverage,
not realism; honest real data supplies the coverage), with the per-trunk
nuance that the causal GRU still gains from the curriculum (41.8 vs its
61.9 control). r5hb_scv2 (mixed one-stage on the R2 base) lands at 147.6
— worse than the old-regime mixed (93.6): the staging-necessity claim
strengthens; honest real data does not detoxify mixed-in synthetic. r5hb_tr (59.2)
softens the transformer's mixed penalty vs the old regime (103.8) but
stays well behind its nogate control (41.8) — direction unchanged.
r5hb_gru (85.8 vs 179.7 old, control 61.9) completes the R5 trio: mixed
one-stage training loses to its real-only control on ALL THREE trunks
under the honest base — the staging-necessity claim is unconditional.
r4hb_gru (comb-only curriculum on R2) reaches 37.6 — the best causal-GRU
cell of the campaign (vs r3hb_gru 41.8, gated hb_gru_ssq 39.8, nogate
control 61.9): for the weakest trunk the cheap analytic comb beats the
neural generator as pre-training, mirroring the old-regime finding.

HEADLINE (2026-08-25): hb_scv2_mag_nogate reaches 22.1 — the best neural
aggregate of the campaign, real data only, essentially level with the old
comb curriculum (21.1), and best-in-class in every regime (flight MSE 11.1
beats even the old real-only cruise). Attribution for scv2: honest DATA
does nearly everything (52.5 -> 22.1); the gate HURTS this trunk (gated
39.7). Combined with the GRU result (gate helps: 61.9 -> 39.8), the
voicing gate is architecture-dependent — useful for causal heads that
cannot see future context, harmful for the bidirectional trunk. The paper
claim "honest silence closes most of the stopped-rotor failure" is now
supported at full strength for the best model.

The full three-trunk attribution matrix (aggregates, old real-only ->
R2 ungated -> R2 gated best): scv2 52.5 -> 22.1 -> 39.7 (gate -17);
transformer-IF 42.3 -> 41.8 -> 33.6 (gate +8); causal GRU 59.2 -> 61.9 ->
39.8 (gate +22). The honest-data windfall is trunk-specific; the gate
converts it for the attention and causal heads and destroys it for the
BiGRU trunk. multif0 on the standard grid: cruise-decent (4.01) and
silence-blind (52.8) — the widened narrow-SR arm is the fix under test.

r3hb_gru (gen+comb curriculum on the R2 base, ungated) lands at 41.8 —
well below its nogate R2 control (61.9) and below the old-regime m3cur
unigru (51.4): for the causal trunk the curriculum still pays on top of
honest data, and the nested comparison is now clean.

First cell: the R2 ungated GRU control lands at 61.9 aggregate — level
with its fs_v2 real-only control (59.2) and behind every GATED hb_gru
cell (39.8-60.7). For the causal trunk the gate is doing real work on
this regime; the honest data alone does not move the aggregate.

### Edge-BS-RoFormer and CKLA on the R2 regime (2026-08-25)

| run | zero | low | flight | best val/mse |
|---|---|---|---|---|
| hb_ckla | 5.95 / 112.0 | 4.25 / 67.3 | 3.51 / 23.9 | 40.6 |
| hb_ebsrof | 34.49 / 2299.4 | 12.26 / 583.0 | 3.21 / 37.8 | 396.2 |

hb_ebsrof (TODO 9 diagnostics): the R2 regime unblocks learning — the
July R1 run was flat near 1150 for its full budget, while this run
descends steadily to 396 over 47 epochs. The trained model is
regime-split: flight MAE 3.21 is competitive with the conv trunks
(2.35-2.85), but the zero regime fails outright (MAE 34.5, near the
salience models' silence blindness). The band-split attention trunk
tracks combs in flight and does not learn the silence-to-zero mapping,
so the aggregate (396) stays an order behind the winners. Conclusion:
the earlier "ebsrof cannot learn RPS" verdict is refuted — the R1 data
was the blocker — but the architecture brings no advantage over scv2 at
~30x the parameter count. The low-lr reserve arm is unnecessary.

hb_ckla (TODO 10): the phase-only CKLA head on the R2 regime becomes a
coherent model across all three regimes (zero 5.95, flight 3.51,
aggregate 40.6) — a large step from the July CKLA campaign, which never
had honest zeros to learn from. It still trails the scv2 winner (22.1)
in every cell, consistent with the design gap recorded in
`docs/pikalman-ckla-design.md`: the pooled-feature CKLA cannot do a
state-conditioned measurement, which is what `hb_hgckla_ref` tests.

### The regime grid completes (2026-08-25, final cells)

| run | zero | low | flight | agg MSE | all-MAE |
|---|---|---|---|---|---|
| r4hb_scv2 | 2.87 / 20.5 | 3.48 / 34.3 | 2.49 / 14.0 | **17.6** | **2.68** |
| r3hb_tr | 5.14 / 127.9 | 5.26 / 96.6 | 2.71 / 15.2 | 40.5 | 3.36 |
| r4hb_tr | 5.99 / 115.1 | 5.19 / 84.4 | 3.09 / 22.5 | 42.6 | 3.74 |

REVISED HEADLINE (2026-08-25): r4hb_scv2 — the comb-only curriculum on
top of the R2 base — is the new best neural cell of the campaign:
aggregate 17.6 against the R2 control's 22.1, best zero cell of any
regression model (2.87 MAE), best all-frame MAE (2.68 against 2.72).
The full per-trunk regime matrix (aggregate MSE, controls first):

| trunk | R2 control | R3 gen+comb | R4 comb-only | R5 mixed |
|---|---|---|---|---|
| scv2 (BiGRU) | 22.1 | 22.6 | **17.6** | 147.6 |
| Transformer | 41.8 | 40.5 | 42.6 | 59.2 |
| causal GRU | 61.9 | 41.8 | **37.6** | 85.8 |

The synthetic-data verdict, final form: the neural GENERATOR is nowhere
necessary — R3 never beats R4 where curricula help, and never beats the
R2 control on the best trunk. The analytic COMB curriculum improves two
of three trunks (scv2 −20%, causal GRU −39% against controls) and is
neutral for the Transformer. Mixing loses everywhere. So: coverage, not
realism — and the cheapest possible synthetic source (the closed-form
comb with exact labels) delivers all of the coverage value that
survives the honest real regime.

### HG-CKLA stage-A refiner (2026-08-25, TODO 11)

Training: flat — best val/mse 2.78 at epoch 1, no improvement over 21
epochs (identity baseline of the corrupted conditioning on the same
valid stream: 3.87). The physics path refines immediately; the learned
components do not progress. Head-to-head on identical channel-0
corrupted inits over all 37 valid clips (full envelope):

| method | all MSE / MAE | zero MSE | low MSE | flight MSE |
|---|---|---|---|---|
| identity (corrupted cond) | 3.56 / 1.29 | 0.12 | 3.27 | 4.21 |
| one pi_kalman pass | 3.44 / 1.23 | 0.17 | 3.52 | 3.98 |
| HG-CKLA v1 | 3.03 / 1.13 | 0.11 | 3.52 | 3.45 |

The regime split localizes the effect: all refinement happens in
flight, where the neural cell removes 18% of the corruption MSE against
the classical pass's 5% — one pi_kalman pass barely moves under this
corruption level (outside its capture band), and slightly degrades the
zero and ramp regions it should leave alone, which the neural cell does
not.

The v1 cell modestly beats one classical pass under heavy corruption
(OU sigma up to 1.5 rev/s — partly outside pi_kalman's capture band)
pooled over the full envelope. The design-doc G1 gate (synthetic
capture range) and the cruise-precision regime of G2 (telemetry-grade
inits, where pi_kalman reaches 0.03-0.2 rev/s) remain unrun; nothing
here shows the cell can match the classical pass where the classical
pass is strong. Verdict: the harmonic-gather measurement works (the
epoch-1 result), the learning on top of it does not yet, and the
refiner is not a leaderboard row.
