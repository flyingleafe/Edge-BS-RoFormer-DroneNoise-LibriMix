---
experiment: ckla_p0_4s_norot
training_config: conf/experiment/ckla_p0_4s_norot.yaml
batch: docs/experiments/ckla.md
---

# `ckla_p0_4s_norot`

## Motivation

Rotation-off pair of `ckla_p0_4s`: identical 4 s static-comb stream, model
`simple_conv_v2_ckla_mag_norot` (exact real-KLA head, no complex path).
The `ckla_p0_4s` − `ckla_p0_4s_norot` gap measures the complex rotation's
contribution at a context length where phase accumulation is possible.

## Setup

See `ckla_p0_4s.md`; only the model differs.

Train: `python train.py experiment=ckla_p0_4s_norot`.

## Conclusion

_Pending run._
