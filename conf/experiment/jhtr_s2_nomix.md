---
experiment: jhtr_s2_nomix
training_config: conf/experiment/jhtr_s2_nomix.yaml
parent: salv2_scv2_stoch_nomix
---

# `jhtr_s2_nomix`

## Motivation

Replace SimpleConvV2 with audio-only JHTR, using its own locator and the inherited final-output PIT-MSE. No supplied guesses, locator pretraining or auxiliary objectives.

## Setup

Hydra inherits [`salv2_scv2_stoch_nomix.yaml`](salv2_scv2_stoch_nomix.yaml), then applies only the declared JHTR overrides. Inherit the complete SALV2 recipe: unchanged synthetic policy and seeds, 4 s training / 8 s validation, batch 32, 16,000 samples/validation, 200 epochs maximum, patience 20 and `mae_frame` selection. Mix retains paired speech/no-speech validation; nomix retains its fixed 256-frame set. Start from scratch.

Optimizer, scheduler, AMP, augmentation, data pins, policies, split seeds and exposure ceilings remain inherited; architecture cost earns no extra training updates. Launch only after composed-recipe parity, numerical reader/model gates and full fixed/paired conditioning parity pass. Existing entry point: `python train.py experiment=jhtr_s2_nomix`.

## Decision and evidence

Audio-only adoption requires at least 10% lower matched all-frame PIT-MAE, a positive paired improvement interval, and no named regime degrading beyond max(0.1 rev/s, 5% of reference MAE). Later cells are gated on the conditional mechanism decision; defining this config is not an instruction to run a rejected architecture.

No result is claimed by this configuration. Select checkpoints only with the inherited monitor and stopping rule; oracle/capture/ablation diagnostics may reject a scientific claim but never select a different checkpoint. Supplied-guess and audio-only results are different task/input comparisons, not a pure architecture comparison. No HG-CKLA training run is introduced.
