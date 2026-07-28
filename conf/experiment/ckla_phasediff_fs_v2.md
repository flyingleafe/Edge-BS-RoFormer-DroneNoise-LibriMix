---
experiment: ckla_phasediff_fs_v2
training_config: conf/experiment/ckla_phasediff_fs_v2.yaml
batch: docs/experiments/ckla.md
---

# `ckla_phasediff_fs_v2`

## Motivation

Under fs_v2 the rotation-on CKLA lost to plain KLA (norot 30.4 MSE vs CKLA
41.4). Mechanistic candidate for why: if the complex state phasor tracks a
rotor, the informative quantity is its *angular velocity* — the angle first
differential arg(y_t·conj(y_{t−1}))·frame_rate/2π is exactly the tracked
instantaneous frequency — but the complex-mean readout [Re y, Im y] discards
it and instead passes the ω-oscillation through to the mix layer as feature
noise. This arm augments the readout with that differential:
[Re y, Im y, arg d, |y|] → Linear(4·d_model, d_model)
(`ckla.py::phase_diff_features`), giving the rotating state a direct path to
expose what it tracks. Everything upstream of the mix layer is unchanged.

## Setup

Clone of `ckla_pnoise_fs_v2` (p_init 1.0, rotation on, uniform freq-scale v2
policy p=1.0, α∈[0.7, 1.3]) with `simple_conv_v2_ckla_phasediff`
(readout="phase_diff"). Compare against `ckla_pnoise_fs_v2` (complex-mean
readout) and `ckla_norot_fs_v2` (plain KLA): if the readout was the
bottleneck, this arm should close the 41.4→30.4 gap. Scale-response probe +
cruise pools as in the fs_v2 batch.

Train: `python train.py experiment=ckla_phasediff_fs_v2`.

## Conclusion

_Pending run._
