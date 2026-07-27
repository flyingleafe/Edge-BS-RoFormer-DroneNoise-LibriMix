---
experiment: ckla_norot_fs_v2
training_config: conf/experiment/ckla_norot_fs_v2.yaml
batch: docs/experiments/ckla.md
---

# `ckla_norot_fs_v2`

## Motivation

Rotation ablation of the uniform-freq-scale winner. `ckla_pnoise_fs_v2`
follows 99% of a 10% frequency shift where the matched transformer manages
71% — but is that the *rotating* complex state, or just the Kalman-scan
recurrence? This arm is identical to `ckla_pnoise_fs_v2` except rotation is
disabled (ω ≡ 0): the state never turns — plain-KLA (real OU) dynamics with
everything else held fixed. Companion cross-implementation check: the fkla
(flat KLA, kla-loglinear repo) arm.

## Setup

`ckla_pnoise_fs_v2` with `simple_conv_v2_ckla_pnoise_norot` (same p_init 1.0
gain fix). Uniform freq-scale policy p=1.0, α∈[0.7, 1.3]. Success metric:
A6 multi-alpha scale-response + probe PIT-MAE vs the rotating arm and the
transformer.

## Conclusion

(pending)
