---
experiment: jhtr_cond_s2_nomix
training_config: conf/experiment/jhtr_cond_s2_nomix.yaml
parent: salv2_scv2_stoch_nomix
---

# `jhtr_cond_s2_nomix`

## Motivation

Replace the model with conditional JHTR and enable the existing supplied-guess task seam: `use_cond: true`, ordered `mse_cond`, and unchanged RPSCorruption defaults (training seed 20260729; validation seed 777).

## Setup

Hydra inherits [`salv2_scv2_stoch_nomix.yaml`](salv2_scv2_stoch_nomix.yaml), then applies only the declared JHTR overrides. The data child inherits the exact SALV2 parent and adds only `rps_corruption`. Keep 4 s training / 8 s validation, batch 32, 16,000 samples/validation, maximum 200 epochs, patience 20, and `mae_frame` selection. Nomix retains 256 validation frames; mix retains both paired halves totaling 512. No S1-to-S2 warm start.

Optimizer, scheduler, AMP, augmentation, data pins, policies, split seeds and exposure ceilings remain inherited; architecture cost earns no extra training updates. Launch only after composed-recipe parity, numerical reader/model gates and full fixed/paired conditioning parity pass. Existing entry point: `python train.py experiment=jhtr_cond_s2_nomix`.

## Decision and evidence

Require correction over identity and oracle preservation on the unchanged selected-checkpoint benchmark; an identity update does not pass. Full-vs-power and full-vs-frozen mechanism claims require at least 10% MAE improvement and a positive paired recording/flight-group interval. Independent-slot claims require collision evidence.

No result is claimed by this configuration. Select checkpoints only with the inherited monitor and stopping rule; oracle/capture/ablation diagnostics may reject a scientific claim but never select a different checkpoint. Supplied-guess and audio-only results are different task/input comparisons, not a pure architecture comparison. No HG-CKLA training run is introduced.
