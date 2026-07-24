---
experiment: g3_gp_aug_transformer
training_config: conf/experiment/g3_gp_aug_transformer.yaml
batch: docs/experiments/g3-gp-curriculum.md
---

# `g3_gp_aug_transformer`

## Motivation

Campaign criterion 3.4: does adding GP-generated rotor noise to the
RPS-predictor curriculum improve real-data performance? Previous synthetic
families both disappoint: the neural generator fails sim2real outright (E7:
real val PIT MSE ~222, R^2 ~ -10 — the predictor reverse-engineers its
amplitude dynamics), and the analytic static comb helps the transformer only
(E8: 225 -> 189 on the synthetic-only probe). The per-drone egonoise GP is a
third point on the realism axis: a physics-anchored comb (Fourier coefficients
regressed over mic-xyz x rps on the CONA-auralized ego-noise sweep) whose
coefficients are frozen at the chunk-mean rps — so, like the static comb, the
only within-chunk RPS cue is the comb's instantaneous frequency, but the timbre
is the fitted drone's. Unlike E7/E8 (synthetic-only probes), this arm tests the
*augmentation* configuration directly: real E12 full-flight stream + ~1/3 GP
chunks with exact synthetic RPS labels.

## Setup

Data `g3_gp_aug` — the E12 real full-flight policy with real shares unchanged
plus two `kind: gp` sources (DREGON + Matrice-100 rigs, weight 0.5 each vs 2.0
real => ~1/3 GP chunks; `data_processing/gp_noise.py`, checkpoints
`r2://ml-data/artifacts/gp_egonoise/{dregon,matrice100}/best.pt`). Model
`simple_conv_v2_transformer`, loss `pit_mse`, metrics `rps`, epochs 200,
patience 20, batch 16. Valid `dload:DREGON-LM-V4-michaels-valid-full` (FLY124 —
no leakage), identical to the E12 baseline.

Train: `python train.py experiment=g3_gp_aug_transformer`.

## Success metric

Protocol eval `scripts/rps_predictor_vk_eval.py` vs the E12 transformer
baseline: DREGON-cruise per-clip PIT MAE 3.186 (raw) / 2.62 (phase-A smoothed).

## Conclusion

_Pending run._
