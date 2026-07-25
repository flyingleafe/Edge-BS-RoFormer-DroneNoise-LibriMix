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

### P0 — `ckla_p0_staticcomb` (kaggle `python-9d450c`, wandb `jcrr4tqe`, 2026-07-25)

Trained stably (fp32 scan under amp, lr 1e-3, no divergence); early-stopped
ep 15, ~2 min/epoch on P100.

**On-distribution (train PIT-MSE, same static-comb stream — the clean
comparison):** CKLA reaches train ≈ 3.4 by **epoch 3**; the E8 transformer
(`2sabeq2g`) needs **~23 epochs** to reach the same level (ep 3–7 window:
CKLA 3.3–4.8 vs transformer 8.4–12.7; E8 uni_gru128 never got below ~9).
≈7× faster epoch-convergence to an equal-or-lower floor → gates (a)+(b)
of design §4 passed decisively.

**Real-valid transfer (caveat — valid sets differ):** CKLA best val/mse
**21.7** (rmse 4.48, R² at last ep −0.52) on the CLEAN
`DREGON-LM-V4-michaels-valid` (pin b6ece43d) vs E8 transformer's recorded
188.7 on the *contaminated* pre-cleanup valid — NOT directly comparable.
For scale: the E9 hard-combined transformer (50% neural gen + 50%
static-comb + augs) scored ~20.7 on the clean valid; CKLA matches that
from static-comb-only training with no augmentation. Fair rescore of the
E8 transformer ckpts on the clean valid: uni-cpu job `bash-4fe1ac`
(_pending_).

### P0b — capture boundary + rotation ablation

uni-cpu job `python-20c95f`: `scripts/ckla_capture_boundary.py`, CKLA-P0
best vs E8-transformer best, drift (aggressiveness 0.25–4) × SNR (+10…−20),
16 clips/cell, `--ablate-rotation`. _Pending._

### P1 — `ckla_p1_if` (kaggle `python-3fd926`)

Launched on gates (a)+(b) without waiting for P0b (diagnostics cannot
reverse the on-distribution result). _Pending._

## Conclusion

_Pending._
