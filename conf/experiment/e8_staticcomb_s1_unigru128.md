---
experiment: e8_staticcomb_s1_unigru128
training_config: conf/experiment/e8_staticcomb_s1_unigru128.yaml
batch: docs/experiments/e8-static-comb.md
---

# `e8_staticcomb_s1_unigru128`

## Motivation

E8 sim-to-real probe with the analytic **static-comb** noise model (simple_conv_v2_uni_gru128
head — the C10 real-data winner (PIT MSE 7.33)). Amplitudes are static and RPS-independent and >=30% of each
rotor's harmonics clear the broadband floor, so the ONLY cue for RPS is the
comb's frequency spacing — forcing harmonic tracking. Train on static-comb only,
validate on the fixed **real** DREGON+Michael's split. Contrast with the E7
neural-generated arm (real val PIT MSE ~222, R2 -10.5).

## Setup

Data `rps_static_comb_only` (analytic static-comb train stream, fixed real
valid), model `simple_conv_v2_uni_gru128`, loss `pit_mse`, metrics `rps`. Patience 8, batch 16,
`samples_per_validation=5000`. Cloud: override train policy to
`rps_static_comb_only_dload.yaml`, valid to `dload:DREGON-LM-V4-michaels-valid`.
See the [E8 batch doc](../../docs/experiments/e8-static-comb.md).

Train: `python train.py experiment=e8_staticcomb_s1_unigru128`.

## Conclusion

Ran 2026-07-11. Best val PIT-MSE 222.6 (R² −10.5) — no gain over E7's 222.3. On the **contaminated** valid; never rescored. See the conclusion of [e8-static-comb.md](../../docs/experiments/e8-static-comb.md).

*(Backfilled 2026-08-20.)*
