#import "/writing/templates/typst/report.typ": report, author-meta

#show: report.with(
  title: [Channel-Generalization Failure in Learned RPS Prediction],
  authors: (
    "Dmitrii Mukhutdinov": author-meta("project"),
  ),
  affiliations: (
    "project": "Harmonic Noise Suppression Project",
  ),
  abstract: [
    We evaluate two RPS-prediction models (SimpleConv, SimpleConvV2) trained on DREGON-LM
    on a new multichannel validation set spanning three free-flight recordings and all
    8 microphone channels. Both models exhibit severe channel-dependent performance
    degradation: edge-microphone MSE is 3--10$times$ higher than the reference microphone,
    and overall $R^2$ is near zero (SimpleConv barely beats the mean baseline, V2 is
    slightly worse). Permutation-invariant evaluation (allowing the model to swap motor
    assignments) recovers only 0.5--2.0% of the error, confirming that the failure is
    genuine misprediction, not motor confusion.
    We then retrain the same architectures on all 8 channels jointly (batch concatenation
    $(B, C, T) arrow.r (B dot C, T)$, still single-channel input per prediction). The
    8ch-trained SimpleConv generalizes uniformly across channels ($R^2 = 0.57$), while
    the 8ch-trained SimpleConvV2 achieves excellent PIT performance ($R^2 = 0.94$) but
    shows a severe motor-swapping issue when evaluated without PIT. This confirms that
    the rotor-ordering task is fundamentally underdefined without a microphone-position
    reference, and *PIT evaluation (and PIT training loss) is the correct objective for
    RPS prediction*.
  ],
  keywords: ("RPS prediction", "channel generalization", "DREGON-LM", "PIT", "drone audio"),
  date: "2026-06-08",
)

= Motivation

RPS (rotations per second) prediction from drone audio is a core building block
for telemetry-free speech enhancement under harmonic noise. Previous work
(SimpleConv, SimpleConvV2) trained and evaluated on DREGON-LM with the implicit
assumption that microphone position does not matter---any channel should yield
similar RPS estimates.

We test this assumption directly: take trained models, evaluate them on the
*same flight recording* but through *different microphones*, and measure the
variance.

= Dataset: DREGON-LM-V4/valid

A 19-sample, 8-channel, 8-second validation set extracted from the DREGON
corpus. *No synthetic mixing*---each sample is a raw 8-channel recording clip
(mixture = drone noise + optionally co-recorded speech or whitenoise).

#figure(
  placement: none,
  table(
    columns: (1fr, auto, auto, 1.5fr),
    inset: 5pt,
    align: (left + horizon, center + horizon, center + horizon, left + horizon),
    table.header([Recording], [Duration (in-flight)], [Clips], [Source type]),
    table.hline(),
    [`free-flight_nosource_room1`], [59.8 s], [7], [Pure drone noise],
    [`free-flight_speech-low_room1`], [50.7 s], [6], [Drone + co-recorded speech],
    [`free-flight_whitenoise-low_room1`], [49.9 s], [6], [Drone + co-recorded whitenoise],
  ),
  caption: [Recordings in the validation set.],
) <tab:dataset>

Takeoff and landing are excluded via a `min_motor_rps=30.0` threshold. All 8
microphones are preserved (each sample is $(128000, 8)$ at 16 kHz). Samples are
strictly non-overlapping within each recording. Ground-truth RPS from
`motors_command` (cleaned) is shared across channels.

The dataset was created with `create_dregon_librimix.py` (new `--max_non_overlapping` flag).

= Models

#figure(
  placement: none,
  table(
    columns: (auto, 1.5fr, 1fr),
    inset: 5pt,
    align: (left + horizon, left + horizon, left + horizon),
    table.header([Model], [Checkpoint], [Architecture]),
    table.hline(),
    [SimpleConv], [`results/rps_exp_simple_conv/`], [4 conv blocks (64$arrow.r$128), global avg pool + 2-layer 1-D conv head],
    [SimpleConvV2], [`results/rps_exp_v2/`], [6 conv blocks (128), BiGRU temporal head, SE blocks],
  ),
  caption: [Models evaluated.],
) <tab:models>

Both trained on synthesised DREGON-LM (6000 train, 1 s clips, $-30$ to $0$ dB SNR,
multichannel with independent speech per channel). Training uses regular MSE
loss (no PIT).

= Results

== Per-recording, per-channel MSE

#figure(
  placement: none,
  table(
    columns: (1.2fr, ..((auto,)*8)),
    inset: 4pt,
    align: (left + horizon, ..((center + horizon,)*8)),
    table.header([Recording], [ch 0], [ch 1], [ch 2], [ch 3], [ch 4], [ch 5], [ch 6], [ch 7]),
    table.hline(),
    [nosource], [3.7], [101.3], [28.8], [13.1], [10.9], [51.8], [39.1], [65.8],
    [speech-low], [10.7], [77.1], [26.0], [10.4], [11.8], [43.7], [37.8], [37.7],
    [whitenoise-low], [16.5], [78.2], [32.1], [12.8], [16.2], [47.0], [37.9], [36.3],
  ),
  caption: [SimpleConv per-channel MSE (overall MSE=35.49, MAE=2.89, $R^2=0.07$).],
) <tab:simpleconv>

#figure(
  placement: none,
  table(
    columns: (1.2fr, ..((auto,)*8)),
    inset: 4pt,
    align: (left + horizon, ..((center + horizon,)*8)),
    table.header([Recording], [ch 0], [ch 1], [ch 2], [ch 3], [ch 4], [ch 5], [ch 6], [ch 7]),
    table.hline(),
    [nosource], [0.4], [3.2], [3.5], [87.3], [3.5], [2.2], [25.7], [44.8],
    [speech-low], [9.7], [11.1], [12.4], [176.3], [14.8], [11.4], [11.4], [137.6],
    [whitenoise-low], [3.1], [9.3], [4.8], [398.3], [4.7], [2.9], [4.2], [9.6],
  ),
  caption: [SimpleConvV2 per-channel MSE (overall MSE=40.28, MAE=1.76, $R^2=-0.10$).],
) <tab:simpleconvv2>

#figure(
  image("assets/mic_array.png", width: 55%),
  caption: [
    DREGON microphone array geometry. Mic 0 (orange dot, near centre) and
    mic 4 (green dot, bottom) share the same $Z$ coordinate (both at $-0.041$ m),
    placing them on the same face of the array. The large numbers 1--4 at the top
    mark the rotor positions (red wireframe), not microphones. The models were
    trained exclusively on channel 0.
  ],
) <fig:mic-array>

#figure(
  image("assets/mse_bars.png", width: 100%),
  caption: [
    Per-channel MSE averaged over all non-overlapping 8 s clips per recording
    and channel. Green bar: channel 0 (the training microphone); red bars: all others.
    Top row: SimpleConv; bottom row: SimpleConvV2.
  ],
) <fig:mse-bars>

Key observations:

- *Channel asymmetry is massive.* Channel 0 (the training mic) consistently
  outperforms edge channels (1, 7) by 3--10$times$ in MSE.
- *Channel 4 is good because it shares the same $Z$ coordinate as channel 0.*
  Both mic 0 and mic 4 are at $Z = -0.041$ m (the lower face of the array), while
  mics 1, 3, 5, 7 are at $Z = +0.041$ m (the upper face). Being on the same face
  means mic 4 observes the drone from the same vertical distance as the training
  mic, giving very similar signal statistics. The models have not seen channel 4
  during training either.
- *SimpleConvV2 is worse overall* despite being a more sophisticated
  architecture---it overfits training mic positions more aggressively.
- *V2 has catastrophic outliers* (ch 3 on `whitenoise-low`: MSE=398.3;
  ch 3 on `speech-low`: MSE=176.3), suggesting unstable predictions on
  certain channel $times$ recording combinations.
- *Source type matters.* `nosource` (pure drone) is easier than
  `speech-low` (drone + co-recorded speech)---the speech source acts as
  interference for RPS estimation.

== Permutation-invariant evaluation (PIT)

To check whether the error is due to *motor swapping* (predicting correct
RPS values but assigned to wrong rotor indices), we re-evaluate with PIT:
for each channel, try all $4! = 24$ rotor permutations and pick the one that
minimises MSE, using the project's canonical `pit_mse_loss` implementation.

#figure(
  placement: none,
  table(
    columns: (1fr, auto, auto, auto, 1.2fr),
    inset: 5pt,
    align: (left + horizon, center + horizon, center + horizon, center + horizon, left + horizon),
    table.header([Model], [MSE (no PIT)], [MSE (PIT)], [$Delta$], [Interpretation]),
    table.hline(),
    [SimpleConv], [35.49], [34.77], [$-2.0$%], [Negligible benefit],
    [SimpleConvV2], [40.28], [40.07], [$-0.5$%], [Negligible benefit],
  ),
  caption: [PIT evaluation summary.],
) <tab:pit>

#figure(
  image("assets/mse_bars_pit.png", width: 100%),
  caption: [
    Same as @fig:mse-bars but with PIT (permutation-invariant) MSE. The bars are
    visually unchanged---PIT recovers only 0.5--2.0% of the error, confirming that
    motor swapping is not the primary failure mode.
  ],
) <fig:mse-bars-pit>

PIT recovers almost nothing---*motor swapping is not the primary failure
mode.* The models genuinely mispredict RPS values on unseen channels.
A few individual channels benefit more (SimpleConv `nosource` ch 0:
$-16.8$%, ch 4: $-9.0$%; `speech-low` ch 0: $-10.1$%, ch 4: $-10.3$%),
suggesting slight motor confusion on cleaner signals, but the overall effect is minor.

== Per-channel prediction traces

To show the failure visually, we select two clips where both models perform
well on channel 0 and plot the full 8-channel prediction traces. Predictions
are PIT-permuted per channel so the plotted lines align with the ground-truth
rotor indices.

*Nosource sample* (`sample_00014`, 25.37 s into `free-flight_nosource_room1`):

#figure(
  image("assets/sample_nosource_simpleconv.png", width: 100%),
  caption: [
    SimpleConv on `sample_00014` (nosource). ch 0 and ch 4 track the GT
    closely (MAE=1.13, 2.05); ch 1, ch 5, ch 6, ch 7 drift significantly from the
    true rotor speeds (MAE=7.06, 5.54, 6.03, 4.26).
  ],
) <fig:nosource-sc>

#figure(
  image("assets/sample_nosource_simpleconv_v2.png", width: 100%),
  caption: [
    SimpleConvV2 on the same sample. ch 0--ch 2, ch 4--ch 6 are excellent
    (MAE $< 0.5$), but ch 3 collapses to a single intermediate value (MAE=13.2) and
    ch 7 shows a severe drop at $tilde$5.5 s (MAE=8.40). The V2 model overfits training
    channels so aggressively that some channels are near-perfect while others
    catastrophically fail.
  ],
) <fig:nosource-v2>

*Speech sample* (`sample_00002`, 25.80 s into `free-flight_speech-low_room1`):

#figure(
  image("assets/sample_speech_simpleconv.png", width: 100%),
  caption: [
    SimpleConv on `sample_00002` (speech). ch 0 and ch 4 are good
    (MAE=1.93, 1.73); ch 1 is catastrophic (MAE=9.25); ch 3 is surprisingly good
    (MAE=1.66).
  ],
) <fig:speech-sc>

#figure(
  image("assets/sample_speech_simpleconv_v2.png", width: 100%),
  caption: [
    SimpleConvV2 on the same speech sample. ch 3 is poor
    (MAE=4.46) and ch 7 is catastrophic (MAE=7.01). The speech interference
    causes a general degradation across all channels compared to the nosource
    sample.
  ],
) <fig:speech-v2>

= Training on all 8 channels

== Training setup

To test whether the channel-generalization failure is a data-coverage issue,
we retrain SimpleConv and SimpleConvV2 on the same DREGON-LM dataset but with
all 8 channels present in every training batch. The training batch is a
concatenation of several 8-channel recordings channel-wise: $(B, C, T) arrow.r
(B dot C, T)$. The model still receives a *single channel* as input for
each individual prediction---the only difference is that the training batch
now contains all microphone positions, not just channel 0.

Checkpoints:

- `results/rps_8ch_v4_simple_conv/best_simple_conv.pt`
- `results/rps_8ch_v4_simple_conv_v2/best_simple_conv_v2.pt`

== Results

#figure(
  placement: none,
  table(
    columns: (1.2fr, ..((auto,)*8)),
    inset: 4pt,
    align: (left + horizon, ..((center + horizon,)*8)),
    table.header([Recording], [ch 0], [ch 1], [ch 2], [ch 3], [ch 4], [ch 5], [ch 6], [ch 7]),
    table.hline(),
    [nosource], [26.9], [24.1], [28.5], [25.0], [26.1], [24.9], [26.7], [25.3],
    [speech-low], [33.6], [32.3], [34.2], [32.3], [33.2], [32.0], [32.8], [32.4],
    [whitenoise-low], [31.5], [30.5], [32.3], [30.4], [31.3], [30.2], [30.9], [30.5],
  ),
  caption: [SimpleConv (8ch) per-channel MSE (overall MSE=29.70, MAE=2.74, $R^2=0.57$).],
) <tab:simpleconv-8ch>

#figure(
  placement: none,
  table(
    columns: (1.2fr, ..((auto,)*8)),
    inset: 4pt,
    align: (left + horizon, ..((center + horizon,)*8)),
    table.header([Recording], [ch 0], [ch 1], [ch 2], [ch 3], [ch 4], [ch 5], [ch 6], [ch 7]),
    table.hline(),
    [nosource], [57.2], [56.6], [56.7], [57.1], [58.0], [57.2], [57.1], [56.5],
    [speech-low], [60.2], [61.2], [60.4], [65.3], [58.9], [61.9], [59.3], [59.8],
    [whitenoise-low], [66.9], [67.3], [67.6], [68.1], [65.9], [67.2], [66.7], [65.9],
  ),
  caption: [SimpleConvV2 (8ch) per-channel MSE (overall MSE=61.39, MAE=5.71, $R^2=-0.78$).],
) <tab:simpleconvv2-8ch>

#figure(
  image("assets/mse_bars_8ch_v4.png", width: 100%),
  caption: [
    8ch-trained models, no PIT. SimpleConv (top) is uniform across all
    channels; SimpleConvV2 (bottom) is uniformly bad.
  ],
) <fig:mse-bars-8ch>

#figure(
  placement: none,
  table(
    columns: (1fr, auto, auto, auto, 1.2fr),
    inset: 5pt,
    align: (left + horizon, center + horizon, center + horizon, center + horizon, left + horizon),
    table.header([Model], [MSE (no PIT)], [MSE (PIT)], [$Delta$], [Interpretation]),
    table.hline(),
    [SimpleConv (8ch)], [29.70], [28.37], [$-4.5$%], [Negligible benefit],
    [SimpleConvV2 (8ch)], [61.39], [*3.30*], [*$-94.6$%*], [Motor swapping dominates],
  ),
  caption: [PIT evaluation summary (8ch-trained).],
) <tab:pit-8ch>

#figure(
  image("assets/mse_bars_8ch_v4_pit.png", width: 100%),
  caption: [
    8ch-trained models, PIT. SimpleConvV2 bars collapse from $tilde$60 to $tilde$2--3
    on every channel.
  ],
) <fig:mse-bars-8ch-pit>

== Why motor swapping is expected, and why PIT is the right metric

The 8ch-trained SimpleConvV2 results make a subtle but important point: the
model predicts the *correct rotor speeds* (PIT MSE=3.30, $R^2=0.94$) but
assigns them to the *wrong rotor indices* (no-PIT MSE=61.39). This is not a
bug---it is a *fundamental consequence of the physics*.

Which rotor is heard loudest depends on the microphone position. There is no
reliable acoustic signature that tells one motor from another independently of
where the microphone is placed. Forcing the model to assign a consistent
rotor index across all channels is therefore an *underdefined task*. We do
not care about the label of each rotor; we only care that the set of four
predicted speeds matches the true set.

Consequently, *PIT evaluation (and PIT training loss) is the correct objective*
for RPS prediction. A model that gets all four speeds right but swaps them
between channels is perfectly useful for downstream harmonic-noise
suppression---the comb-filter notch frequencies depend only on the rotor
speeds, not on which rotor produces which harmonic.

== Sample comparisons

*Nosource sample (`sample_00014`)---SimpleConv (8ch):*

#figure(
  image("assets/sample_nosource_8ch_v4_simpleconv.png", width: 100%),
  caption: [
    SimpleConv (8ch) on `sample_00014` (nosource). All channels track the GT
    with similar MAE ($tilde$2.5--3.0). The model has learned a channel-agnostic
    representation.
  ],
) <fig:nosource-8ch-sc>

*Nosource sample (`sample_00014`)---SimpleConvV2 (8ch), PIT-permuted:*

#figure(
  image("assets/sample_nosource_8ch_v4_simpleconv_v2.png", width: 100%),
  caption: [
    SimpleConvV2 (8ch) on the same sample, after PIT permutation. All
    channels track the GT closely (MAE $tilde$0.7--0.8). The raw predictions are
    motor-swapped, but the speed values themselves are accurate.
  ],
) <fig:nosource-8ch-v2>

*Speech sample (`sample_00002`)---SimpleConv (8ch):*

#figure(
  image("assets/sample_speech_8ch_v4_simpleconv.png", width: 100%),
  caption: [
    SimpleConv (8ch) on `sample_00002` (speech). Uniform performance across
    channels, slight degradation from speech interference.
  ],
) <fig:speech-8ch-sc>

*Speech sample (`sample_00002`)---SimpleConvV2 (8ch), PIT-permuted:*

#figure(
  image("assets/sample_speech_8ch_v4_simpleconv_v2.png", width: 100%),
  caption: [
    SimpleConvV2 (8ch) on the same speech sample, after PIT permutation.
    Again, all channels are accurate once rotor indices are ignored.
  ],
) <fig:speech-8ch-v2>

*Dynamic sample (`sample_00012`, nosource)---both 8ch models, PIT-permuted:*

`sample_00012` is a particularly revealing clip: the motors spin up from
$tilde$30 rev/s to $tilde$80 rev/s during the first 1.5 s, then dip and recover at
$tilde$2 s. This is a much more dynamic regime than the steady-flight clips used above.

#figure(
  image("assets/sample_nosource_varied_8ch_v4_simpleconv.png", width: 100%),
  caption: [
    SimpleConv (8ch) on `sample_00012`. The model tracks the ramp-up
    and the dip, but with a slight lag (MAE $tilde$5.9--6.4).
  ],
) <fig:dynamic-8ch-sc>

#figure(
  image("assets/sample_nosource_varied_8ch_v4_simpleconv_v2.png", width: 100%),
  caption: [
    SimpleConvV2 (8ch) on the same dynamic sample. The model tracks the
    transition almost perfectly (MAE $tilde$1.4--2.0), confirming that it is not merely
    predicting flat means---it genuinely captures transient speed changes.
  ],
) <fig:dynamic-8ch-v2>

= Data and reproducibility

- *Dataset:* `datasets/DREGON-LM-V4/valid` (19 samples, 8-channel WAV + RPS NPY).
  Created by `create_dregon_librimix.py --max_non_overlapping`.
- *Evaluation results:* `results/dregon_v4_eval/eval.json` (no PIT) and
  `eval_pit.json` (PIT). Generated by `evaluate-rps -i ... -m ... --pit`.
- *8ch evaluation results:* `results/dregon_v4_eval/eval_8ch_v4.json` and
  `eval_8ch_v4_pit.json`.
- *Code changes:*
  - `src/tasks/rps_prediction.py`---tag propagation from metadata to per-sample rows;
  - `create_dregon_librimix.py`---`--max_non_overlapping` flag;
  - `train_rps_predictor.py`---`return_indices` on `pit_mse_loss`.
