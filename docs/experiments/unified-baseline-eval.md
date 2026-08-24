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

Reading (preliminary, scv2 cells only): the honest regime alone closes most
of the zero gap (real-only 11.83 -> 3.68-6.04 MAE) at almost no cruise cost
(flight MSE 12.8-14.2 vs real-only 11.6). The mag front-end regresses on
ramps (10.68 vs real-only 4.83); the IF front-end holds them (4.99) and has
the best aggregate (34.9) — between real-only (52.5) and the comb curriculum
(21.1). The comb curriculum still leads on aggregate, consistent with the
coverage story: the synthetic stage supplies unlimited ramp/transition
trajectories that 34 s of real ramps cannot match.

### Remaining rows

The other 7 HB runs, the nogate control, salience retraining
(`hb_sal_multif0`, `hb_sal_bp`), and the blind-tracker row are queued;
their probes land here as they finish.
