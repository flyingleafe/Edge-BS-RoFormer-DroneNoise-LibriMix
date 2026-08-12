---
experiment: gen_r1_scaled
training_config: conf/experiment/gen_r1_scaled.yaml
batch: docs/experiments/generator-label-sensitivity.md
---

# `gen_r1_scaled`

## Motivation

The constant-fix control: the same generator on labels multiplied by 0.99458.

Phase 7 found that a constant label bias alone costs 8.6 dB at k = 50-80, so a
scale is not benign. This arm separates "the labels were off by one number"
from "the labels were off in shape": if it reaches `gen_r1_refined`, the
refinement recovers no more than a scale.

Full batch context: [Generator label sensitivity](../../docs/experiments/generator-label-sensitivity.md).

## Conclusion

Pending — the arm is not run yet.
