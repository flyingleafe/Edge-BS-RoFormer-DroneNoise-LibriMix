---
experiment: jhtr_real_r3
training_config: conf/experiment/jhtr_real_r3.yaml
parent: real_r3_scv2
---

# `jhtr_real_r3`

## Motivation

Replace the model with audio-only JHTR in this established real-only rung. No synthetic warm start or conditional telemetry input.

## Setup

Hydra inherits [`real_r3_scv2.yaml`](real_r3_scv2.yaml), then applies only the declared JHTR overrides. Inherit the exact parent stream, data pins, microphone flattening, PIT-MSE, batch 128, 40,000 samples/validation, 200 epochs maximum and patience 20. R1–R3 select on `mae_frame`; R4 selects on `mse`. Preserve those differences. R1’s mono-example exposure is not normalized to the eight-microphone rungs.

Optimizer, scheduler, AMP, augmentation, data pins, policies, split seeds and exposure ceilings remain inherited; architecture cost earns no extra training updates. Launch only after composed-recipe parity, numerical reader/model gates and full fixed/paired conditioning parity pass. Existing entry point: `python train.py experiment=jhtr_real_r3`.

## Decision and evidence

Audio-only adoption requires at least 10% lower matched all-frame PIT-MAE, a positive paired improvement interval, and no named regime degrading beyond max(0.1 rev/s, 5% of reference MAE). Later cells are gated on the conditional mechanism decision; defining this config is not an instruction to run a rejected architecture.

No result is claimed by this configuration. Select checkpoints only with the inherited monitor and stopping rule; oracle/capture/ablation diagnostics may reject a scientific claim but never select a different checkpoint. Supplied-guess and audio-only results are different task/input comparisons, not a pure architecture comparison. No HG-CKLA training run is introduced.
