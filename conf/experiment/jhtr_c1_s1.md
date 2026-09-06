---
experiment: jhtr_c1_s1
training_config: conf/experiment/jhtr_c1_s1.yaml
parent: m3abl_comb_scv2_s1
---

# `jhtr_c1_s1`

## Motivation

Replace the model with audio-only JHTR in the established C1 static-comb stage 1; this is not SALV2 S1.

## Setup

Hydra inherits [`m3abl_comb_scv2_s1.yaml`](m3abl_comb_scv2_s1.yaml), then applies only the declared JHTR overrides. Inherit the complete `m3abl_comb_scv2_s1` recipe and its validation set, PIT-MSE, batch 128, 40,000 samples/validation, 200 epochs, patience 20 and `mse` selection. Its own compatible best checkpoint is the only stage-1 source for `jhtr_c1_s2`.

Optimizer, scheduler, AMP, augmentation, data pins, policies, split seeds and exposure ceilings remain inherited; architecture cost earns no extra training updates. Launch only after composed-recipe parity, numerical reader/model gates and full fixed/paired conditioning parity pass. Existing entry point: `python train.py experiment=jhtr_c1_s1`.

## Decision and evidence

Audio-only adoption requires at least 10% lower matched all-frame PIT-MAE, a positive paired improvement interval, and no named regime degrading beyond max(0.1 rev/s, 5% of reference MAE). Later cells are gated on the conditional mechanism decision; defining this config is not an instruction to run a rejected architecture.

No result is claimed by this configuration. Select checkpoints only with the inherited monitor and stopping rule; oracle/capture/ablation diagnostics may reject a scientific claim but never select a different checkpoint. Supplied-guess and audio-only results are different task/input comparisons, not a pure architecture comparison. No HG-CKLA training run is introduced.
