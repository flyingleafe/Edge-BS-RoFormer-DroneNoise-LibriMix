---
experiment: gen_v2_recal
training_config: conf/experiment/gen_v2_recal.yaml
batch: docs/experiments/michaels-recalibration-generators.md
---

# `gen_v2_recal`

## Motivation

Noise-generator retrain — per-rotor sub-embedding arch (v2) re-trained on the recalibrated Michael's labels
(michaels-frames pin fdef818432e9, derivations recipe_version 2). Paired with
its stale-label predecessor so the label fix can be isolated per drone.

Full batch context: [Michael's recalibration — generator retrains](../../docs/experiments/michaels-recalibration-generators.md).

## Conclusion

Reported in the batch write-up — see
[Michael's recalibration — generator retrains](../../docs/experiments/michaels-recalibration-generators.md).
