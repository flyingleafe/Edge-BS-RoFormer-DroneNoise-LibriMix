---
experiment: gen_r2_refined_perrotor
training_config: conf/experiment/gen_r2_refined_perrotor.yaml
batch: docs/experiments/generator-label-sensitivity.md
---

# `gen_r2_refined_perrotor`

## Motivation

The full cell of the label A/B: per-rotor sub-embeddings AND refined labels.

Per-rotor identity is a per-rotor claim, and the original telemetry cannot
resolve the near-equal rotors. Refined trajectories are what can make the four
codes mean four different rotors instead of four copies of one, so this is the
arm the campaign expects to win.

Full batch context: [Generator label sensitivity](../../docs/experiments/generator-label-sensitivity.md).

## Conclusion

Pending — the arm is not run yet.
