---
experiment: gen_r1_refined
training_config: conf/experiment/gen_r1_refined.yaml
batch: docs/experiments/generator-label-sensitivity.md
---

# `gen_r1_refined`

## Motivation

The treatment of the real-DREGON label A/B: the same generator on REFINED rotor-speed labels.

The refined trajectories come from the trajectory-refinement job as one sidecar
per recording; the loader replaces the label VALUES and keeps the timebase, so
the audio, the split and the schedule are identical to `gen_r1_orig`. The
question: does a better rotor-speed trajectory make a better generator on real
audio?

Full batch context: [Generator label sensitivity](../../docs/experiments/generator-label-sensitivity.md).

## Conclusion

Pending — the arm is not run yet.
