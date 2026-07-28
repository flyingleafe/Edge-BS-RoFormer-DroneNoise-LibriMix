---
experiment: ckla_phaseonly_fs_v2
training_config: conf/experiment/ckla_phaseonly_fs_v2.yaml
batch: docs/experiments/ckla.md
---

# `ckla_phaseonly_fs_v2`

## Motivation

Aggressive companion to `ckla_phasediff_fs_v2`. Same mechanistic hypothesis:
rotation-on lost to plain KLA under fs_v2 (norot 30.4 MSE vs CKLA 41.4)
because the complex-mean readout discards the state phasor's angular
velocity — the tracked instantaneous frequency
arg(y_t·conj(y_{t−1}))·frame_rate/2π — while passing the raw ω-oscillation
through as feature noise. Where phase_diff *adds* the differential next to
[Re y, Im y], this arm removes the raw quadratures entirely: features
[|y|, arg d] → Linear(2·d_model, d_model) — a readout invariant to the
state's absolute rotation phase. If the oscillating quadratures are pure
noise, this arm should match or beat phase_diff at the same mix-layer width
as the baseline.

## Setup

Clone of `ckla_pnoise_fs_v2` (p_init 1.0, rotation on, uniform freq-scale v2
policy p=1.0, α∈[0.7, 1.3]) with `simple_conv_v2_ckla_phaseonly`
(readout="phase_only"). Compare against `ckla_pnoise_fs_v2`,
`ckla_phasediff_fs_v2`, and `ckla_norot_fs_v2`. Scale-response probe +
cruise pools as in the fs_v2 batch.

Train: `python train.py experiment=ckla_phaseonly_fs_v2`.

## Conclusion

_Pending run._
