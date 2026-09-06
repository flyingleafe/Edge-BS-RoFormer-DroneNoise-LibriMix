---
experiment: jhtr_cond_real
training_config: conf/experiment/jhtr_cond_real.yaml
parent: hb_hgckla_ref_v2
---

# `jhtr_cond_real`

## Motivation

Replace the conditional model with JHTR; inherit the real-refiner task and complete training setup without launching an HG-CKLA control.

## Setup

Hydra inherits [`hb_hgckla_ref_v2.yaml`](hb_hgckla_ref_v2.yaml), then applies only the declared JHTR overrides. The inherited stream override is `conf/online_mix/hb_silence_dload.yaml`, including its established warm-up. Keep ordered `mse_cond`, monitor `mse`, batch 128, 40,000 samples/validation, 40 epochs maximum, patience 8, and gradient clip 1. Start from scratch, not an S1/S2 checkpoint.

Optimizer, scheduler, AMP, augmentation, data pins, policies, split seeds and exposure ceilings remain inherited; architecture cost earns no extra training updates. Launch only after composed-recipe parity, numerical reader/model gates and full fixed/paired conditioning parity pass. Existing entry point: `python train.py experiment=jhtr_cond_real`.

## Decision and evidence

Require correction over identity and oracle preservation on the unchanged selected-checkpoint benchmark; an identity update does not pass. Full-vs-power and full-vs-frozen mechanism claims require at least 10% MAE improvement and a positive paired recording/flight-group interval. Independent-slot claims require collision evidence.

No result is claimed by this configuration. Select checkpoints only with the inherited monitor and stopping rule; oracle/capture/ablation diagnostics may reject a scientific claim but never select a different checkpoint. Supplied-guess and audio-only results are different task/input comparisons, not a pure architecture comparison. No HG-CKLA training run is introduced.
