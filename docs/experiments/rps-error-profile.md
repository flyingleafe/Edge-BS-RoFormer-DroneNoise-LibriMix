# RPS error profile: where each model family loses (2026-09-03)

## Question

The monitored number of an RPS predictor is a mean over frames. A mean cannot
tell "imprecise on every clip" from "exact on most clips, lost on a few". The
two call for different fixes. This campaign keeps every per-frame prediction of
27 checkpoints on one common benchmark and reads the distribution.

## Benchmark

Three parts, the same for every model. Each synthetic part is the salv2
without-speech validation set, so the salv2 cells reproduce their W&B numbers
(all no-speech cells agree to three decimals, `r4hb_scv2` real 2.757 against
2.758):

| part | definition | flights | silence clips | frames |
|---|---|---|---|---|
| static comb | `conf/data/salv2_comb_nomix.yaml` valid (seed 880101) | 28 | 4 | 256 |
| stochastic comb | `conf/data/salv2_stoch_nomix.yaml` valid (seed 881101) | 24 | 8 | 256 |
| real | `dload:DREGON-LM-V4-michaels-valid-full` (DREGON room1 + FLY124, whole envelope) | 33 | 4 | 296 |

Two more parts are the speech twins of the synthetic parts (same flights, a
LibriSpeech talker at -30 to 0 dB SNR). A frame is one microphone of an 8 s
clip. Every number below is the PIT per-frame MAE in rev/s, read by each
family's own monitored readout (peak + parabola for the salience ports, the
regressor output as is). "Flights only" removes the silence clips, whose score
is trivially 0 for a model that returns zero on silence.

Reproduce: `scripts/rps_dump.py` (GPU, omnirun jobs `rps-dump-core-2a2e69`
and `rps-dump-speech-9ad6db`) then `scripts/rps_error_profile.py` (CPU). The
dump is `results/rps_dump/`, the profile `results/rps_profile/`.

## Static comb, flights only (224 frames)

| model | training | mean | median | p90 | max | frames > 1 | worst 10% share | octave frames | error in octaves |
|---|---|---|---|---|---|---|---|---|---|
| HPPNet | comb, no speech | 0.53 | 0.37 | 1.20 | 3.1 | 13% | 33% | 0.12% | 9% |
| HPPNet | comb, with speech | 0.55 | 0.35 | 1.28 | 3.8 | 17% | 34% | 0.01% | 0.5% |
| HPPNet | comb, CRF objective | 0.79 | 0.41 | 1.69 | 6.4 | 26% | 40% | 0.41% | 28% |
| HarmoF0 | comb, no speech | 1.36 | 0.70 | 1.82 | 13.5 | 37% | 54% | 1.22% | 51% |
| HarmoF0 | comb, with speech | 0.97 | 0.64 | 2.06 | 8.7 | 34% | 39% | 0.47% | 27% |
| SCV2 regressor | comb, no speech | 1.03 | 1.01 | 1.51 | 2.6 | 50% | 19% | 0 | 0 |
| uni-GRU regressor | mixed real+gen+comb | 1.65 | 1.53 | 2.49 | 4.3 | 86% | 18% | 0 | 0 |
| SCV2 regressor | comb then real (r4hb) | 15.8 | 9.5 | 39 | 82 | 100% | 37% | | |
| HPPNet | comb then real (r4_l4) | 54 | 52 | 80 | 87 | 100% | 15% | | |

An "octave frame" is a rotor-frame whose prediction is 2x the label. Octave
frames occur only where 2x the label is inside the 0-150 rev/s grid (labels
below 75 rev/s); above 75 the count is zero for every model.

## Stochastic comb, flights only (192 frames)

| model | training | mean | median | p90 | max | frames > 1 | worst 10% share | octave frames | error in octaves |
|---|---|---|---|---|---|---|---|---|---|
| HPPNet | stoch, no speech | 3.5 | 2.1 | 6.0 | 35 | 89% | 43% | 4.1% | 27% |
| HPPNet | stoch, with speech | 4.2 | 2.1 | 10.7 | 39 | 88% | 46% | 4.3% | 23% |
| HarmoF0 | stoch, no speech | 11.4 | 6.7 | 24.6 | 45 | 100% | 27% | 2.6% | 6% |
| SCV2 regressor | stoch, no speech | 4.0 | 2.7 | 9.1 | 24 | 88% | 33% | 0.04% | 0 |
| every comb-trained model (3 archs) | comb | 50-51 | 50 | 77 | 92 | 100% | 17% | | |
| uni-GRU regressor | mixed | 29 | 32 | 53 | 64 | 100% | 20% | | |
| SCV2 regressor | comb then real | 37 | 41 | 57 | 66 | 100% | 16% | | |

## Real, flights only (264 frames)

| model | training | mean | median | p90 | max | worst 10% share | ground | ramp | cruise | cruise bias |
|---|---|---|---|---|---|---|---|---|---|---|
| SCV2 regressor (r4hb) | comb then real | 2.88 | 2.63 | 4.4 | 10 | 26% | 1.6 | 5.1 | 2.28 | +1.5% |
| uni-GRU regressor (r4hb) | comb then real | 3.76 | 2.82 | 6.0 | 20 | 35% | 2.4 | 8.4 | 2.50 | +2.1% |
| uni-GRU regressor (m3mixv2) | mixed | 4.20 | 2.75 | 7.4 | 33 | 38% | 1.5 | 9.5 | 2.77 | +1.7% |
| transformer regressor (m3mixv2) | mixed | 4.20 | 2.79 | 10.7 | 21 | 36% | 3.9 | 9.9 | 2.66 | -0.3% |
| SCV2 regressor (m3mixv2) | mixed | 4.16 | 3.04 | 5.4 | 31 | 33% | 17.4 | 8.2 | 3.07 | +3.4% |
| HPPNet (r4_l4) | comb then real | 5.75 | 3.11 | 18.0 | 42 | 44% | 8.5 | 16.5 | 2.86 | +1.1% |
| HarmoF0 (r4_l4) | comb then real | 34 | 32 | 75 | 80 | 22% | | | | |
| every synthetic-only salv2 model | comb or stoch | 19-70 | | | | | | | | |

Ground = all four rotors stopped (4 clips), ramp = label range in the clip
more than 15 rev/s (7 clips: take-offs and landings), cruise = the rest (26
clips). "Cruise bias" is the median of (prediction - label) / label over the
cruise clips. The `hf0_r4_l4` checkpoint was selected on BCE, and its rps_mae
at that epoch is 31; the row is not a HarmoF0 result.

## Findings

1. **HPPNet on the static comb is exact on most clips and skewed by two
   things.** The median flight-frame error is 0.37 rev/s. Maneuvers (label
   slope more than 10 rev/s per second, 12% of the time) carry 2.5x the error
   of steady flight and 26% of the total. Octave jumps (one track reads
   harmonic 2 as the fundamental for a few frames) carry 9%. Both are fast:
   the residual decorrelates in 0.08 s and the per-rotor offset is 0.24 rev/s
   (median). The speech-trained twin has the same mean and almost no octave
   jumps. The CRF-trained twin jumps 3.4x more often.
2. **HarmoF0 fails by octave flicker.** Half of its static-comb error sits in
   1.2% of the rotor-frames, where the prediction is 2x the label. Without
   those frames its MAE is 0.67, close to HPPNet. Its error increases 1.5x
   where two rotors are less than 2 rev/s apart (HPPNet: 1.01x).
3. **The regressor has no outliers and no precision.** SCV2 trained on the
   comb scores 1.0 rev/s on every clip (mean / median 1.02, worst-10% share
   19%). Its error is slow: per-rotor offsets of 0.66 rev/s (median) and a
   residual that decorrelates in 0.46 s, that is a lag through maneuvers. It
   never aliases on the static comb.
4. **On the stochastic comb every model aliases, and the two families alias
   differently.** HPPNet locks onto 2x the speed on whole clips (4% of
   rotor-frames, 27% of its error, always below 75 rev/s) and flickers between
   the two. The regressor slides smoothly to 3/2, 4/3 or 2/3 of the speed
   (its "offset" class, 26% of its error, plus 5/4 aliases 22%). The worst
   clips for both are the four-identical-rotor hovers at 17-19 rev/s and the
   45-65 rev/s clips.
5. **Comb-trained models return zero on the stochastic family.** All three
   architectures trained on the static comb score 50 rev/s on the stochastic
   flights, which is the mean label: the output is a stopped rotor on 100% of
   the frames. The two synthetic families are disjoint to a comb-trained
   model. The real-fine-tuned models do the same on BOTH synthetic families
   (r4hb regressors 16-40, r4_l4 ports 54): real fine-tuning erases the
   synthetic comb.
6. **On real audio the best regressor's error is uniform, the ports' and the
   mixed models' error is in the ramps and on the ground.** r4hb_scv2 has a
   median of 2.6 and a maximum of 10 (worst-10% share 26%). Its cruise error
   is 2.3 rev/s and mostly a per-clip offset (|bias| 2.3 of MAE 2.9,
   residual autocorrelation 0.43 s); 93% of the cruise clips are
   over-estimated, by +1.5% (median). Every good model over-estimates cruise
   speed by 1-2%, which is the size of the known label biases. The mixed
   uni-GRU has the same median (2.75) and loses on ramps (9.5 vs 5.1) and on
   one white-noise-room clip before take-off, where it reads the loudspeaker
   as a 55 rev/s rotor. HPPNet r4_l4 matches the regressors on cruise (2.86)
   and loses on ramps (16.5, 43% of its error at 28% of the time), on ground
   noise (phantom flicker at 2x) and on the FLY124 warm-up at 30-40 rev/s,
   where it returns zero.
7. **There is no consistent time shift between prediction and label on the
   real ramps.** The lag that maximizes correlation is -0.10 s (median) for
   r4hb_scv2 and -0.02 s for the mixed uni-GRU; one clip
   (speech-low_room1 t=8 s) shows +0.35 to +0.80 s.
8. **Silence.** Digital silence: the salience ports and the salv2 regressors
   return 0; the mixed m3mixv2 regressors return 40-49 rev/s (they never saw
   digital silence); r4hb_scv2 returns 12.6. Real ground noise: the r4hb and
   mixed GRU regressors score 1.5-2.4, HPPNet r4_l4 8.5, and the
   synthetic-only HPPNet 36-52 (phantom rotors).
9. **A talker breaks the no-speech HPPNet on two flights** (0.46 to 1.42
   mean, maximum 3 to 59) while the with-speech twin moves 0.48 to 0.55.

## Consequences

- HPPNet's static-comb ceiling is set by maneuvers and octave jumps, not by
  per-frame precision: a temporal decoder with a tight band and a readout that
  cannot jump an octave address 35% of its error; trajectory-steered
  integration (M4) addresses the maneuvers.
- HarmoF0 is HPPNet plus octave flicker; the port is not worth a separate
  campaign until the octave readout is fixed for both.
- The stochastic family is the harder synthetic task and a different one:
  models transfer between the two families in neither direction.
- On real audio the regressor's remaining error is a slow per-clip offset of
  the size of the label bias, and the ramps. The ports lose only in the
  ramps, on the ground, and below 40 rev/s.
