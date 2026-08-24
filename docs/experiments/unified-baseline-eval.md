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
| Classical (May 2026) | PYIN, cepstral, HPS, matched filter, NMF | `src/experiments/classical_rps/valid_eval.py` (restored from `00753c4`; report `writing/reports/2026-05-29_classical-baselines/`) — uni-cpu gridrun | pending run |
| OT multi-pitch | Björkman & Elvander 2026 (adapted config) | `results/otmp_protocol/` (run 2026-08-23) — re-aggregated under this table | done, merge |
| Salience (June ckpts) | multif0_salience, basic_pitch_salience | old V4-trained checkpoints, if loadable from the zoo | optional |
| Salience (retrained) | `hb_sal_multif0`, `hb_sal_bp` | retrained on the hb online regime (same data + augmentations as the HB grid) | pending training |
| Neural direct (current) | scv2 / transformer-IF / uni_gru128 (real-only fs_v2, m3cur s2, m3abl comb s2) | `results/m3cur_regime_probe/regime_probe.json` | done, merge |
| Neural direct (HB grid) | 10 hb_* runs | regime probe after the campaign lands | pending training |
| Blind two-stage tracker | Viterbi seed + peel/pi_kalman | blind annotation of the 37 clips (CPU cluster, `scripts/vk_pseudolabel.py` machinery) | planned, not scheduled |

## Notes

- The classical methods and the OT baseline are training-free; the salience
  retraining is the only arm whose training data changed since its June
  numbers (fixed V4 mixtures then, hb online stream now), so its old and new
  rows must both appear.
- Blade count for the classical methods is 2, harmonics 15, grid constants
  identical to the project defaults — no adaptation was needed.

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

### Remaining rows

Salience retraining (`hb_sal_multif0`, `hb_sal_bp`) and the HB grid are
queued on the cluster; their regime probes land here when training ends.
