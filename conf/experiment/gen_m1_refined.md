---
experiment: gen_m1_refined
training_config: conf/experiment/gen_m1_refined.yaml
batch: docs/experiments/generator-refined-labels.md
---

# `gen_m1_refined`

## Motivation

The full-dataset counterpart of `gen_r1_refined`: DREGON + Michael's 8-mic
stream, DREGON labels replaced by the L-BFGS-refined sidecar, per-drone
codebook only. The question: does the label fix carry over when Michael's
shares the training stream? Per-epoch checkpoints are ON so a comb-aware
selection is possible afterward (best-by-mrstft is a lottery — see
`docs/experiments/generator-perrotor-dynamics.md`).

## Conclusion

Pending.
