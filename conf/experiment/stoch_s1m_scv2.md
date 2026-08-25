---
experiment: stoch_s1m_scv2
training_config: conf/experiment/stoch_s1m_scv2.yaml
batch: docs/experiments/stochastic-transfer.md
---

# `stoch_s1m_scv2`

## Motivation

Synthetic-only training whose target is transfer to the real frozen validation
split, on the convolutional trunk, with no real noise anywhere in the stream.
Policy `conf/online_mix/stoch_s1m_dload.yaml`.

The base is `stoch_s1g_scv2`, the best synthetic-only model this project has:
172.1 validation PIT-MSE against 204.0 for the analytic comb, and the best ramp
cell of any of them (16.20 rev/s). The arms that narrowed the family to make it
easier to fit, H and J, fit far better and transfer worse (299.3 and 285.3), so
arm L keeps every bit of arm G's width and changes two other things.

The rotor speed spans a decade. Arm G draws cruise from 46 to 110 rev/s while
every real cruise clip sits at 80.3, so a prior does much of the work — and the
opening measurement of this campaign says it does, the comb-only predictor
reading 0.836 of the truth at a response slope of 0.94. A model that transfers
across drones cannot carry that prior, because different aircraft turn at
different rates and the only thing that survives is finding a comb wherever it
sits. `rps_scale_range: [0.25, 2.5]`, log-uniform, gives a per-clip hover of 22
to 211 rev/s. The level reference, the linewidth and the harmonic count all
follow the clip's own hover, so the width teaches comb-finding instead of
handing over absolute speed through loudness or sharpness.

The warm-up also idles where a drone idles. Arm G's idle band reaches 0.05 of
hover, which makes the aircraft sit at 5 rev/s — 45 dB below cruise, silence
with a nonzero label — and that is the most likely cause of its one weak cell,
stopped rotors at 20.27. The band returns to [0.28, 0.55] and the low-speed
coverage comes from a longer spin-up.

Data `stoch_s1m`, model `simple_conv_v2`, loss `pit_mse`, metrics `rps`, batch
128 frames, `samples_per_validation=40000`, validation on the fixed
FULL-envelope real split `dload:DREGON-LM-V4-michaels-valid-full`.
Train: `python train.py experiment=stoch_s1m_scv2`.

Arm M is arm L with `level_mode: window` — the level-to-speed map removed. Level
is a real cue, but it is also a regression a model can lean on, and arm G shows
what happens when its low end is anchored wrong: on real frames it fits
`pred = 0.42 * truth + 36.6` and reads a stopped rotor as 27 rev/s, while the
old comb family, which normalizes every window and so has no such map, fits
`1.42 * truth + 0.8` and reads a stopped rotor as 2.9. Arm L repairs the anchor;
arm M removes the map, so silence must be recognized by the absence of a comb.
That is the cue that survives a drone whose hover is somewhere else, which is
what the decade of speed is for. The two are a matched pair.

## Conclusion

PENDING — the run has not finished.
