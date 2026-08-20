---
experiment: gen_c1_amp_combined
training_config: conf/experiment/gen_c1_amp_combined.yaml
batch: docs/experiments/amplitude-target-training.md
---

# `gen_c1_amp_combined`

## Motivation

Combined-rig amplitude-target arm 1: one embedding per rig, DREGON + Michael's on the v3 per-rig decompositions, through the amplitude-only propagation head (1/r x learned per-(rig, mic) EQ).

## Conclusion

Blocked: `conf/data/decomp_frames_v3_combined.yaml` names v3 dataset pins that are not materialized yet. On hold with the amplitude-target campaign.
