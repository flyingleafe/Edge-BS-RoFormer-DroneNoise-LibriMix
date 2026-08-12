---
experiment: gen_r1_orig
training_config: conf/experiment/gen_r1_orig.yaml
batch: docs/experiments/generator-label-sensitivity.md
---

# `gen_r1_orig`

## Motivation

The baseline of the real-DREGON label A/B: per-drone codebook generator, DREGON only, ORIGINAL telemetry labels.

Phase 7 measured the effect of label error on a synthetic comb, where the truth
is known. This arm and its partners move the same question onto real DREGON
audio: the labels are the only thing that changes across the arms, so a
difference in `mrstft` is a difference in the labels or in the per-rotor
sub-embeddings, and nothing else. Hyperparameters are copied from
`gen_v1_recal_mm`, the multi-observer magnitude baseline.

Full batch context: [Generator label sensitivity](../../docs/experiments/generator-label-sensitivity.md).

## Conclusion

Pending — the arm is not run yet.
