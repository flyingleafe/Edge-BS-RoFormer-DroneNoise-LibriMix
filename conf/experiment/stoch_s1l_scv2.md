---
experiment: stoch_s1l_scv2
training_config: conf/experiment/stoch_s1l_scv2.yaml
batch: docs/experiments/stochastic-transfer.md
---

# `stoch_s1l_scv2`

## Motivation

Synthetic-only training whose target is per-regime parity with the best
real-trained model on the frozen split, on the convolutional trunk, with no real
noise anywhere in the stream. Noise pool
`conf/online_mix/stoch_s1l_dload.yaml`.

Arm K widens the rotor speed to a decade. Every arm before it drew cruise from a
range about two and a half times wide — arm J 46 to 111 rev/s, arm F 47 to 125,
the old comb family 55 to 91 — which leaves a model room to carry a prior over
where the comb sits, and the campaign's opening diagnostic caught one doing it:
reading 0.836 of the truth on real cruise audio, splitting the difference
between what it measures and what it expects. The only cue that transfers to a
different aircraft is the spacing itself, so the training speed is drawn
log-uniformly over [0.25, 2.5] of hover and a clip's cruise sits anywhere from
about 20 to 200 rev/s.

Two things a decade breaks, both repaired rather than papered over. The level
cue scaled by a fixed 80 rev/s reference would make a 20 rev/s aircraft 34 dB
quieter than an 80 rev/s one, encoding absolute speed — so the reference is now
the clip's own hover, and cruise clips under 70 and over 150 rev/s leave at the
same level to within 10%. And at 80 harmonics a 20 rev/s comb would stop at
1.6 kHz, so the harmonic count is chosen per clip to reach Nyquist.

Everything else is `stoch_s1k_scv2`: the decade of speed, the narrowed widths that produced the
campaign's best cruise cell (2.60 rev/s against the real-trained 2.49), the
level-as-cue treatment, and the warm-up and ramps measured against the split's
own low-regime frames.

Data `stoch_s1l`, model `simple_conv_v2`, loss `pit_mse`, metrics `rps`, batch
128 frames, `samples_per_validation=40000`, validation on the fixed
FULL-envelope real split `dload:DREGON-LM-V4-michaels-valid-full`.
Train: `python train.py experiment=stoch_s1l_scv2`.

Arm L adds the recording floor. Arm G reaches a stopped-rotor cell of 20.27
rev/s while carrying an 18% zero arm, because the level relation in this stream
is one reality does not have: measured against a cruise clip, the frozen split's
stopped-rotor clips sit at 0.175 and its ramp clips at 0.370, while this stream
gave 0.000 and 0.125. A real stopped-rotor clip is not silent — it carries room
tone and the preamp — and driving it to digital zero teaches a model that a
sixth of cruise level means a turning rotor. `floor_static_rel` splits the
broadband floor into the rotors' share, which follows their speed, and the
recording chain's share, which does not.

## Conclusion

PENDING — the run has not finished.
