---
experiment: gen_m3_refined_all_perrotor
training_config: conf/experiment/gen_m3_refined_all_perrotor.yaml
batch: docs/experiments/generator-refined-labels.md
---

# `gen_m3_refined_all_perrotor`

## Motivation

`gen_m2_refined_perrotor` trained with refined labels on the DREGON side only;
the Michael's side used the recalibrated telemetry. The FLY124/FLY125 refined
sidecars now exist (produced from `frames:michaels-frames` itself, same time
reference as the DREGON sidecars) and this run wires them in via the new
`michaels_rps_override_dir` knob — refined labels on BOTH rigs. This is the
candidate production generator for the "train the RPS predictor on generated
noise with the real-data augmentation schedule" experiment.

Checkpoint selection plan: compare the mrstft-selected checkpoint against the
comb-aware selection over ALL epochs, on the validation chunks of both rigs
(the FLY side read along the refined sidecar tracks), visually and by ear,
before any downstream use.

## Conclusion

Pending.
