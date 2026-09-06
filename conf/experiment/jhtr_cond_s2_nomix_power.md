---
experiment: jhtr_cond_s2_nomix_power
training_config: conf/experiment/jhtr_cond_s2_nomix_power.yaml
parent: jhtr_cond_s2_nomix
---

# `jhtr_cond_s2_nomix_power`

## Motivation

Zero the complex lag-product input channels while retaining envelope power, global memory, and all six blocks.

## Setup

Hydra inherits [`jhtr_cond_s2_nomix.yaml`](jhtr_cond_s2_nomix.yaml), then applies only the declared JHTR overrides. All data, conditioning, loss, model width/block count, seeds and optimization settings come from the full conditional parent. Only the named mechanism flag and experiment identifier change.

Optimizer, scheduler, AMP, augmentation, data pins, policies, split seeds and exposure ceilings remain inherited; architecture cost earns no extra training updates. Launch only after composed-recipe parity, numerical reader/model gates and full fixed/paired conditioning parity pass. Existing entry point: `python train.py experiment=jhtr_cond_s2_nomix_power`.

## Decision and evidence

Require correction over identity and oracle preservation on the unchanged selected-checkpoint benchmark; an identity update does not pass. Full-vs-power and full-vs-frozen mechanism claims require at least 10% MAE improvement and a positive paired recording/flight-group interval. Independent-slot claims require collision evidence.

No result is claimed by this configuration. Select checkpoints only with the inherited monitor and stopping rule; oracle/capture/ablation diagnostics may reject a scientific claim but never select a different checkpoint. Supplied-guess and audio-only results are different task/input comparisons, not a pure architecture comparison. No HG-CKLA training run is introduced.
