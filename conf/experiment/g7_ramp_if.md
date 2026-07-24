---
experiment: g7_ramp_if
training_config: conf/experiment/g7_ramp_if.yaml
batch: docs/experiments/g1-vk-parity.md
---

# `g7_ramp_if`

## Motivation

G5 showed cold-start augmentation destabilizes optimization (the plain
transformer diverged and crashed; IF degraded 63.7 -> 117.9): the plain
warmup is load-bearing. G6 introduces the strong augmentation family as a
step function at 50k samples — this arm (simple_conv_v2_transformer_if) replaces the step with a
staircase: plain warmup (<50k) -> mild strong-augs (50k-100k, probability
0.3, reduced ranges: alpha 0.9-1.11, recolor 4 dB, RT60 <=0.4 s, <=2 teeth,
1 mask, floor <=-8 dB) -> full G6 strength (>=100k). Submit after the G6
readout if its curves show an instability signature at the 50k boundary (or
to A/B the schedule regardless). Success gate: best val/mse below the G6
counterpart with a stable post-50k curve; then the vk_valid_comparison
protocol eval vs the VK bars (DREGON 0.68-0.74, FLY124 1.03).

## Conclusion

(pending)
