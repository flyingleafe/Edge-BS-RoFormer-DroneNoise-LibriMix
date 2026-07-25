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
from static-comb-only training with no augmentation.

**Fair rescore (uni-cpu `bash-653c8d`, eval.py, identical clean valid):**
E8 transformer best.ckpt → MSE **85.4**, rmse 9.01, mae_clip 7.90,
R² −1.37. CKLA P0 best (train-time val) → MSE **21.7**, rmse 4.48,
mae_clip 4.54 — **~4× lower MSE / 2× lower RMSE** at identical training
data and protocol. (E8 has no last.ckpt — predates the feature. Symmetric
eval.py scoring of the CKLA ckpt: `bash-94cdbf`.) Gate (c) supported by
the partial capture table (job `python-20c95f` OOM'd after the CKLA
block): graceful degradation — MAE 1.3–1.8 rev/s locked at aggressiveness
≤ 1, 3.3 at 2, capture lost only at 4; interference axis monotone and
mild. Rerun with `--mem 16`: `python-91cd13`.

### P0b — capture boundary + rotation ablation (uni-cpu `python-91cd13`, DONE)

`scripts/ckla_capture_boundary.py`, drift (aggressiveness 0.25–4) × SNR
(+10…−20 dB speech-rel), 16×4 s clips/cell, outputs in
`results/ckla_capture_boundary/` (pull via `omnirun pull python-91cd13`).

1. **CKLA locks, the transformer never does.** CKLA: lock fraction
   0.38–0.69 / MAE 1.3–1.8 rev/s at aggressiveness ≤ 1, degrading
   gracefully (3.3 at agg 2) to capture loss at agg 4 (15.6). E8
   transformer: lock fraction **0.00 in every cell**, MAE 3.3–5.2 — never
   a sustained lock even at the easiest cell. Gate (c) passed.
2. **Rotation ablation is NULL.** Zeroing s/ω0/W_ω on the trained model:
   ΔMAE within ±0.03 everywhere. Gate (d) FAILED — the trained model does
   not use the complex path.
3. **Parameter forensics** (best.ckpt): ω0 at ring init, s ≈ init 0.1,
   W_ω grew only to ~0.03 from zero-init, OU decay/noise params ≈ init.
   The model trained its projections, not its dynamics.

**Symmetric-harness confirmation** (`bash-94cdbf`, eval.py both models,
identical clean valid): CKLA best MSE **21.74** / rmse 4.48 / mae_clip
2.76 / **R² +0.50** vs E8 transformer best 85.4 / 9.01 / 7.90 / −1.37.

**Interpretation:** the P0 win belongs (so far) to the KLA
uncertainty-gated recurrence, not the complex extension — plausibly
because 1 s clips (T≈32) give phase accumulation nothing to do.

### `ckla_p0_norot` — train-time rotation-off control (kaggle `python-3c4ae9`, wandb `08k0ct9x`, DONE)

Best val/mse **21.51** (ep 4) vs rotation-on **21.70** (ep 7); train
convergence identical (~3.4 by ep 3–4 both arms). **At 1 s context the
complex rotation contributes exactly nothing** — proven from both
directions (eval-time ablation null + train-time control identical). The
1 s P0 win is entirely the real-KLA uncertainty-gated recurrence. The
complex hypothesis now rests on the 4 s pair `ckla_p0_4s` /
`ckla_p0_4s_norot` (kaggle `python-72ff01` / `python-761e7f`).

### P1 — `ckla_p1_if` (kaggle `python-3fd926`)

Launched on gates (a)+(b) without waiting for P0b (diagnostics cannot
reverse the on-distribution result). _Pending._

## Conclusion

_Pending._
