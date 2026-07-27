---
experiment: ebsrof_rps_e12
training_config: conf/experiment/ebsrof_rps_e12.yaml
batch: docs/experiments/ckla.md
---

# `ebsrof_rps_e12`

## Motivation

The Edge-BS-RoFormer paper (Liu et al., Drones 2025) claims its rotary
time/freq positional embeddings help the axial transformers track harmonic
lines. RPS prediction is the direct test — the target IS the harmonic-line
trajectory. This arm runs the Paper-1 SE trunk (F1 rope48 configuration,
phase-aware complex-STFT input) with a band-pool RPS head on the standard
E12 protocol, joining the head comparison (transformer 63.7 / CKLA base
85.2 / CKLA pnoise 44.8 full-envelope val).

## Setup

Standard P1 recipe (data `e12_real_fullflight`, pit_mse, patience 20) with
`model` → `edge_bs_rof_rps` (`src/models/edge_bs_rof/rps.py`, 559k
params). Freqscale twin: `ebsrof_rps_freqscale`.

Train: `python train.py experiment=ebsrof_rps_e12`.

## Conclusion

_Pending run._
