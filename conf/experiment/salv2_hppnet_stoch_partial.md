---
experiment: salv2_hppnet_stoch_partial
training_config: conf/experiment/salv2_hppnet_stoch_partial.yaml
batch: docs/experiments/salv2-speech-and-objective-grid.md
---

# `salv2_hppnet_stoch_partial`

## Motivation

`salv2_hppnet_stoch_nomix` — HPPNet with a comb-gather front end, trained on the
stochastic rotor-noise family without speech — scores 2.646 rev/s on its own
synthetic validation set and reads **real** drone audio at about HALF the true
speed on cruise: PIT MAE near 35 rev/s, median relative bias -48%.

The suspected mechanism is a **fully observed comb**. `StochasticNoisePool`
sizes the comb so its last line lands at Nyquist, so every synthetic clip
carries the whole harmonic series. A real rotor does not: its harmonics die into
the broadband floor above an order that changes with the clip, the microphone
and the speed, the four rotors' lines mask each other, and the lines are broad.
On a full comb a plain mean over harmonics is a sufficient readout of the speed.
On a short comb it is not — the half rate puts a candidate line on every true
line and spends its remaining candidates on empty bins, which costs nothing, so
the half rate wins.

This cell makes the training family **partially observed** and asks whether the
half-speed reading on real audio goes away, zero-shot.

## Setup

Hydra wiring — data `salv2_stoch_partial_nomix` · model `hppnet_rps_l4` · loss
`salience_layers_r150` · metrics `salience_layers_r150`. Train with
`python train.py experiment=salv2_hppnet_stoch_partial`.

Two changes against the parent, both inside
`conf/online_mix/salv2_stoch_partial.yaml`:

- `n_harmonics_range: [10, 80]` on both stochastic sources. The comb length is
  drawn uniformly per clip instead of filling the band. Drawing it per clip is
  also what keeps the band edge from becoming a speed cue: a comb of fixed
  length stops at `n_harmonics * rps`, an edge frequency proportional to the
  speed being predicted, while a random length makes that edge the product of
  two unknowns.
- `tooth_dropout` at probability 0.5 (`max_teeth: 8`, `max_harmonic: 40`,
  `halfwidth_bins: 2`), in the `noise_augmentations` placement
  `conf/online_mix/g7_ramp_dload.yaml` uses. Lines then go missing from the
  middle of the comb as well as from its end.

The run is a **warm start** from the parent's `best.ckpt` on R2, at 40 epochs and
patience 15, so it measures what the wider family does to the trained port. The
**validation block is the parent's, unchanged** (policy `salv2_stoch.yaml`,
`n: 256`, `base_seed: 881101`, 8 s, unaugmented): `val/rps_mae` therefore stays
comparable with the parent's 2.646 and acts as a regression guard. The result
itself is read off the real benchmark part with `scripts/rps_dump.py --sets
real,stoch,comb`, against the parent's dumps under `results/rps_dump/`.

## Conclusion

Pending — the cell is running.
