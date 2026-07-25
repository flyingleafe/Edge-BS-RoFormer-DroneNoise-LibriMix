---
experiment: ckla_p0_norot
training_config: conf/experiment/ckla_p0_norot.yaml
batch: docs/experiments/ckla.md
---

# `ckla_p0_norot`

## Motivation

Rotation-off control of the CKLA campaign (`docs/ckla-design.md` §5 ladder
item 1). P0 showed the CKLA head converging ~7× faster than the E8
transformer on static-comb and transferring at 21.7 vs 85.4 clean-valid
MSE — but P0b's eval-time rotation ablation measured a NULL delta (±0.03
MAE), and the trained rotation parameters sat at init. This arm trains the
same head from scratch with the complex path removed entirely (exact
real-KLA recursion). Equal scores ⇒ the win belongs to the uncertainty-gated
KLA recurrence and the complex extension is refuted at this protocol (1 s
clips); a gap ⇒ rotation matters through training dynamics even though the
final function ignores it.

## Setup

Exact clone of `ckla_p0_staticcomb` with `model` →
`simple_conv_v2_ckla_mag_norot`. Cloud: same dload twins overrides.

Train: `python train.py experiment=ckla_p0_norot`.

## Conclusion

_Pending run._
