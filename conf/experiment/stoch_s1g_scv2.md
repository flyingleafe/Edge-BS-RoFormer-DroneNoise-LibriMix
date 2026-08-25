---
experiment: stoch_s1g_scv2
training_config: conf/experiment/stoch_s1g_scv2.yaml
batch: docs/experiments/stochastic-transfer.md
---

# `stoch_s1g_scv2`

## Motivation

Synthetic-only training whose target is transfer to the real frozen validation
split, on the bidirectional GRU trunk. The whole noise pool is the
stochastic rotor-noise family (`kind: stochastic`,
`conf/online_mix/stoch_s1g_dload.yaml`): a smooth colored floor plus one
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

Arm E adds the per-rotor spread. Four rotors put four combs in one spectrum and
their separation decides how those combs interleave; on the real split's flight
frames that separation averages 13.7 rev/s, while the trajectory model at
aggressiveness 1.0 gives 9.4. Drawing the aggressiveness per window from
[0.8, 2.5] gives 12.7, which brackets the real figure. The comparison row is
`stoch_s1d_scv2`.

Arm F gives the stream the ramps. The frozen split's loss is concentrated
there — 26.3 rev/s against 7.1 at cruise — and two properties of the trajectory
model explain it. The warm-up idles at 0.38 to 0.52 of hover, so the stream
shows a rotor at 30 to 42 rev/s and never between 10 and 30, which is most of
what a real ramp passes through; and the ramps are four times too slow at the
ninetieth percentile. Widening the idle band and shortening the ramps brackets
the real figures on both axes. The comparison row is `stoch_s1e_scv2`.

Arm G puts the level back. Forcing every clip of the split to one level wrecks
the stopped-rotor regime for a real-trained model too — `r4hb_scv2` goes from
2.87 rev/s there to 23.22 at a forced RMS of 0.1 while its cruise cell does not
move — so level is the cue that says whether the rotors are turning, not a
nuisance. Every synthetic pool destroys it by normalizing each window to its own
root-mean-square. `level_mode: flight` treats the level target as the level at
the reference speed and scales by the window's own amplitude, so stopped windows
leave at 0.0004, ramp windows at 0.019 and cruise windows at 0.12. The post-mix
gain narrows from 36 dB back to 12 dB for the same reason. The comparison row is
`stoch_s1f_scv2`.

The comparison row is `m3abl_comb_scv2_s1` — the analytic static comb as the
whole pool on the same trunk, the same optimizer, the same validation split and
the same three augmentation blocks. It reaches 337 validation PIT-MSE, against
17.6 for the best real-trained model, and the diagnosis was a family too narrow
to contain the real thing. This arm holds everything else fixed and widens the
family. No real noise appears anywhere in the stream.

Data `stoch_s1g`, model `simple_conv_v2`, loss `pit_mse`, metrics `rps`, batch 128
frames, `samples_per_validation=40000`, validation on the fixed FULL-envelope
real split `dload:DREGON-LM-V4-michaels-valid-full`.
Train: `python train.py experiment=stoch_s1g_scv2`.

## Conclusion

PENDING — the run has not finished.
