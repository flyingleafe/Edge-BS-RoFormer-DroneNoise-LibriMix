---
experiment: ckla_phaseonly_8s_ft
training_config: conf/experiment/ckla_phaseonly_8s_ft.yaml
batch: docs/experiments/beat-vk.md
---

# `ckla_phaseonly_8s_ft`

## Motivation

The training-window ladder is the strongest neural lever found on the fixed
raw protocol (1 s -> 4 s: 3.22/1.28 -> 2.55/1.08 pooled PIT-MAE), but the
from-scratch 8 s arm (`ckla_phaseonly_8s`) failed to train: best val at
epoch ~1 (best_mse 44.5), monotonic degradation afterwards, final val
diverged (R² catastrophically negative). Protocol score of its best ckpt:
3.102/1.154 — worse than 4 s. With batch_size 16 frames = **2 chunks** per
step, lr 1e-3 is far too hot for 8 s sequences; the failure is
optimization, not architecture.

Attempt 2 removes the optimization problem: warm-start from
`ckla_phaseonly_4s` best.ckpt (the current best neural row, 2.546/1.077)
and fine-tune at lr 2e-4 on the 8 s online-mix stream. If longer context
keeps paying (anchor precision on DREGON steady windows is the remaining
gap vs blind_fullrange 1.807), this is the cheapest way to collect it.

## Setup

Identical to `ckla_phaseonly_8s` (data
`e12_fullflight_freqscale_v2_8s_dload`, model
`simple_conv_v2_ckla_phaseonly`, PIT-MSE, grad_clip 1.0) except:
`checkpoint:` warm-start from the 4 s best, `lr: 2e-4`.

## Conclusion

TBD.
