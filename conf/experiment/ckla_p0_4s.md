---
experiment: ckla_p0_4s
training_config: conf/experiment/ckla_p0_4s.yaml
batch: docs/experiments/ckla.md
---

# `ckla_p0_4s`

## Motivation

4-second-context arm of the CKLA campaign. At the E8 default 1 s clips
(T≈32 frames) the complex rotation path measured null (P0b ablation delta
±0.03 MAE; rotation params sat at init) — but 1 s leaves phase accumulation
no room to pay off. This arm re-runs the P0 protocol with 4 s clips
(T≈126); paired with `ckla_p0_4s_norot`, the rotation-on/off gap at 4 s is
the fair test of the complex extension.

## Setup

`ckla_p0_staticcomb` with the 4 s train policy
(`conf/online_mix/rps_static_comb_only_4s_dload.yaml`, baked into the
experiment config) and batch 8 (4× the samples per clip). Valid unchanged
(fixed real 1 s protocol — duration-agnostic model). Cloud: valid override
to `dload:DREGON-LM-V4-michaels-valid` as usual.

Train: `python train.py experiment=ckla_p0_4s`.

## Conclusion

_Pending run._
