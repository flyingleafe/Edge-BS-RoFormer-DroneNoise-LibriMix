# CKLA — complex Kalman linear attention for RPS prediction

Batch doc for the CKLA architecture campaign (design:
`docs/ckla-design.md`; exploration + prior art:
`docs/complex-ou-layer-exploration.md`). Goal: either beat the neural
floor **2.481** (g2_if, dregon_cruise PIT-MAE on the vk_valid_comparison
protocol) by more than seed noise without regressing FLY124, or produce a
quantified definitive negative naming the mechanism (design §6 kit).

## Ladder

| stage | experiment | question | gate |
|---|---|---|---|
| P0 | `ckla_p0_staticcomb` | can the CKLA head track combs at all, at matched budget vs the E8 transformer arm? | train-dist PIT-MSE ≤ E8 transformer at common epoch; stable fp32 training; rotation path used (§6 diagnostics) |
| P0b | capture-boundary eval | where does lock break vs drift rate × SNR? | boundary at or beyond K2's collapse point |
| P1 | `ckla_p1_*` (E12 schedule, v4-michaels stream) | does it beat 2.481 on real cruise? | > seed-noise margin (~0.15), FLY124 ≤ 2.33 |
| P1a–e | ablation ladder (design §5) | which ingredient carries/fails? | — |

## P0 protocol

Identical to `e8_staticcomb_s1_transformer` in every field except
`model` → `simple_conv_v2_ckla_mag` (stft_mag front-end — isolates the
head; the E8 arms all ran stft_mag). Comparison numbers from the E8/E9
batch (post valid-cleanup, [[sim2real-rps-transfer-findings]]): the
on-distribution comparison is the *train* PIT-MSE trajectory at common
epochs (wandb), the transfer read is the fixed real valid.

## Results

_Pending._

## Conclusion

_Pending._
