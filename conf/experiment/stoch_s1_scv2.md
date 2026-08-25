---
experiment: stoch_s1_scv2
training_config: conf/experiment/stoch_s1_scv2.yaml
batch: docs/experiments/stochastic-transfer.md
---

# `stoch_s1_scv2`

## Motivation

Synthetic-only training whose target is transfer to the real frozen validation
split, on the bidirectional GRU trunk. The whole noise pool is the
stochastic rotor-noise family (`kind: stochastic`,
`conf/online_mix/stoch_s1_dload.yaml`): a smooth colored floor plus one
Lorentzian line per harmonic of every rotor, with every amplitude drifting
slowly in time as a Gaussian process drawn independently of the rotor-speed
trajectory. A silence arm at weight 0.2 supplies the room tone a stopped-rotor
span carries and the model would otherwise never hear, and
`snr_ref_floor_rms: 0.02` keeps a quiet noise chunk from dragging the whole
sample down.

The comparison row is `m3abl_comb_scv2_s1` — the analytic static comb as the
whole pool on the same trunk, the same optimizer, the same validation split and
the same three augmentation blocks. It reaches 337 validation PIT-MSE, against
17.6 for the best real-trained model, and the diagnosis was a family too narrow
to contain the real thing. This arm holds everything else fixed and widens the
family. No real noise appears anywhere in the stream.

Data `stoch_s1`, model `simple_conv_v2`, loss `pit_mse`, metrics `rps`, batch 128
frames, `samples_per_validation=40000`, validation on the fixed FULL-envelope
real split `dload:DREGON-LM-V4-michaels-valid-full`.
Train: `python train.py experiment=stoch_s1_scv2`.

## Conclusion

PENDING — the run has not finished.
