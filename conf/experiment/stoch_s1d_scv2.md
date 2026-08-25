---
experiment: stoch_s1d_scv2
training_config: conf/experiment/stoch_s1d_scv2.yaml
batch: docs/experiments/stochastic-transfer.md
---

# `stoch_s1d_scv2`

## Motivation

Synthetic-only training whose target is transfer to the real frozen validation
split, on the bidirectional GRU trunk. The whole noise pool is the
stochastic rotor-noise family (`kind: stochastic`,
`conf/online_mix/stoch_s1d_dload.yaml`): a smooth colored floor plus one
Lorentzian line per harmonic of every rotor, with every amplitude drifting
slowly in time as a Gaussian process drawn independently of the rotor-speed
trajectory. A silence arm at weight 0.2 supplies the room tone a stopped-rotor
span carries and the model would otherwise never hear, and
`snr_ref_floor_rms: 0.02` keeps a quiet noise chunk from dragging the whole
sample down.

Arm B adds what arm A's clips cannot have. Its noise chunks are anechoic and
its floors are smooth by construction, while every real recording was made in a
room and every real floor carries narrow features. The augmentation block
therefore draws one of frequency scaling, `spectral_recolor` (a smooth random
equalizer of plus or minus 8 dB) and `random_reverb` (RT60 0.1 to 0.8 s) on
every chunk. The comparison row is `stoch_s1_scv2`, the same family without the
room and the coloration.

Arm C adds the lever the measurement asks for. `m3abl_comb_scv2_s1` reads 0.841
of the truth on real cruise audio, with the tenth and ninetieth percentiles at
0.812 and 0.865 — a systematic scale error, not a lost comb and not a harmonic
confusion. A model that reads frequency with a slope below one and shrinks the
rest toward its training prior produces exactly that. `rps_scale_range: [0.6,
1.5]` multiplies the whole trajectory per window before the audio is rendered
from it, so a cruise window can sit anywhere from 48 to 120 rev/s and no speed
prior is worth carrying. The comparison row is `stoch_s1b_scv2`.

Arm D adds level invariance, which the same measurement asks for even louder.
Every synthetic pool normalizes its chunks to a root-mean-square of 0.1 and a
real validation clip sits at 0.041, so a synthetic-only model has never trained
at the level it is asked to read. Feeding the real clips at the training level
recovers a third of the scale error by itself (0.838 to 0.899 of the truth),
and three octaves below it the prediction collapses to 9.75 rev/s against a
truth of 80.28. The pool now draws its output level log-uniformly over
[0.005, 0.25], and the post-mix gain augmentation fires on every sample over
36 dB. The comparison row is `stoch_s1c_scv2`.

The comparison row is `m3abl_comb_scv2_s1` — the analytic static comb as the
whole pool on the same trunk, the same optimizer, the same validation split and
the same three augmentation blocks. It reaches 337 validation PIT-MSE, against
17.6 for the best real-trained model, and the diagnosis was a family too narrow
to contain the real thing. This arm holds everything else fixed and widens the
family. No real noise appears anywhere in the stream.

Data `stoch_s1d`, model `simple_conv_v2`, loss `pit_mse`, metrics `rps`, batch 128
frames, `samples_per_validation=40000`, validation on the fixed FULL-envelope
real split `dload:DREGON-LM-V4-michaels-valid-full`.
Train: `python train.py experiment=stoch_s1d_scv2`.

## Conclusion

PENDING — the run has not finished.
