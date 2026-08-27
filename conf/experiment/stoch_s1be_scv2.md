---
experiment: stoch_s1be_scv2
training_config: conf/experiment/stoch_s1be_scv2.yaml
batch: docs/experiments/stochastic-transfer.md
---

# `stoch_s1be_scv2`

## Motivation

Synthetic-only training whose target is transfer to the real frozen
validation split, on the bidirectional GRU trunk. No real noise appears anywhere
in the stream.

One change from arm ID: both noise engines stop putting a speed-linear cutoff
at the top of the comb.

Every synthetic clip ended its comb at `k_max * rps`. In the stochastic engine
`k_max` was sized from the flight's HOVER speed; in the static comb a
per-timestep gate zeroed each order the instant it crossed Nyquist. Either way a
frame slower than hover carried a sharp spectral edge at a frequency exactly
proportional to the rotor speed the model is asked to predict — the answer,
written into the spectrogram as a visible line.

Measured with the spectrum's own tilt differenced out and restricted to ramp
frames, the step at that cutoff is +1.84 dB in the built stream against +0.50 dB
at the same frequency in real DREGON audio, and 100% of ramp frames carry one. A
model that learns to find the edge learns nothing that transfers, which is
consistent with the ramp cell being where synthetic-only models fail worst.

`band_taper_frac: 0.30` fades the line power to zero across the top of the band
on a raised cosine, the stochastic engine now sizes its comb from the window's
slowest turning frames rather than from hover, and the order caps rise to cover
the band at low RPM. Ramp frames retaining an in-band cutoff fall from 100% to
7.3%.

The comparison row is `stoch_s1id_scv2`, whose phase behaviour this arm
reproduces exactly (see that arm's note).

Data `stoch_s1be`, model `simple_conv_v2`, loss `pit_mse`, metrics `rps`, batch 128
frames, `samples_per_validation=40000`, validation on the fixed FULL-envelope
real split `dload:DREGON-LM-V4-michaels-valid-full`.
Train: `python train.py experiment=stoch_s1be_scv2`.

## Conclusion

PENDING — the run has not finished.
