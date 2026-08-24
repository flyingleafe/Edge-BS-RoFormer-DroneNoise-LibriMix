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

Pending.
