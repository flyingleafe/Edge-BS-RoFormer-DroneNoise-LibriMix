---
experiment: jhtr_c1_s2
training_config: conf/experiment/jhtr_c1_s2.yaml
parent: r4hb_scv2
---

# `jhtr_c1_s2`

## Motivation

Replace the model with audio-only JHTR and replace the stage-1 checkpoint with JHTR’s own compatible `jhtr_c1_s1` best checkpoint.

## Setup

Hydra inherits [`r4hb_scv2.yaml`](r4hb_scv2.yaml), then applies only the declared JHTR overrides. This is the current checked-in C1 recipe, not a reproduction claim for historical as-run numbers. Inherit `r4hb_scv2` verbatim, including `hb_m3s2_dload.yaml`, batch 128, 40,000 samples/validation, 200 epochs, patience 20, PIT-MSE and `mse` selection. The checkpoint URI is `r2://ml-data/artifacts/jhtr_c1_s1/checkpoints/best.ckpt`; stage 1 must be completed and published first. Count pretraining exposure separately.

Optimizer, scheduler, AMP, augmentation, data pins, policies, split seeds and exposure ceilings remain inherited; architecture cost earns no extra training updates. Launch only after composed-recipe parity, numerical reader/model gates and full fixed/paired conditioning parity pass. Existing entry point: `python train.py experiment=jhtr_c1_s2`.

## Decision and evidence

Audio-only adoption requires at least 10% lower matched all-frame PIT-MAE, a positive paired improvement interval, and no named regime degrading beyond max(0.1 rev/s, 5% of reference MAE). Later cells are gated on the conditional mechanism decision; defining this config is not an instruction to run a rejected architecture.

No result is claimed by this configuration. Select checkpoints only with the inherited monitor and stopping rule; oracle/capture/ablation diagnostics may reject a scientific claim but never select a different checkpoint. Supplied-guess and audio-only results are different task/input comparisons, not a pure architecture comparison. No HG-CKLA training run is introduced.
