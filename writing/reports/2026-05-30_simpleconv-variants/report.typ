#import "/writing/templates/typst/report.typ": report, author-meta

#show: report.with(
  title: [SimpleConv Architecture Variants: A Systematic Evaluation],
  authors: (
    "Dmitrii Mukhutdinov": author-meta("project"),
  ),
  affiliations: (
    "project": "Harmonic Noise Suppression Project",
  ),
  abstract: [
    We conduct a systematic sweep of ten architectural variants of SimpleConv, a lightweight CNN for multi-rotor RPS (rotations per second) estimation from drone audio. All variants are trained under identical conditions on DREGON-LM (6000 train / 600 valid samples, $-30$ to $0$ dB SNR). We report validation-set metrics, full-sequence evaluation on a real $~47$ s free-flight recording, and out-of-distribution tests on clean individual-motor and synchronized four-rotor recordings. The key finding is that adding a bidirectional GRU temporal head yields the largest single improvement ($R^2$ 0.837 $arrow$ 0.945), and a deeper encoder with squeeze-excitation blocks pushes this further to $R^2$ = 0.948 with strong generalisation to real recordings (in-flight MSE 9.9 vs baseline 24.4).
  ],
  keywords: ("RPS prediction", "SimpleConv", "architecture sweep", "drone audio"),
)

= Method

== Model Variants

All models share a convolutional encoder that downsamples frequency while preserving time, followed by a temporal head. The variants differ in:

#table(
  columns: (1fr, 2fr, 2fr, 1fr, 2fr),
  inset: 8pt,
  align: left,
  table.header(
    [*Variant*], [*Encoder*], [*Temporal head*], [*Input*], [*Extra features*],
  ),
  [Baseline], [4 blocks, 64$arrow$128], [Global avg pool + 2-layer 1-D conv], [1-ch log-mag], [---],
  [BiGRU], [4 blocks, 64$arrow$128], [BiGRU (2$times$128)], [1-ch], [---],
  [BiGRU-v2], [6 blocks, 128], [BiGRU (2$times$128)], [1-ch], [SE after each block],
  [v2 (SE+Attn)], [6 blocks, 128], [BiGRU (2$times$128)], [1-ch], [SE + freq-attention pool],
  [TCN], [4 blocks, 64$arrow$128], [Dilated conv (rf=31)], [1-ch], [---],
  [MagPhase], [4 blocks, 64$arrow$128], [BiGRU (2$times$128)], [3-ch (mag+cos+sin)], [---],
  [AttnPool], [4 blocks, 64$arrow$128], [Multi-head attention pool + MLP], [1-ch], [---],
  [Wide], [4 blocks, 128$arrow$256$arrow$512], [Global avg pool + MLP], [1-ch], [Pure width scaling],
  [MultiScale], [4 blocks], [FPN-style fusion + MLP], [1-ch], [Bottom-up skip connections],
  [SE-Next], [6 blocks, 128$arrow$256], [Global avg pool + MLP], [1-ch], [SE + residual, no temporal],
)

All trained with AdamW (lr $1e^{-3}$, wd $1e^{-4}$), batch size 16, mixed precision, gradient clip 5.0, patience 15.

= Results

== Validation-set Leaderboard

#figure(
  image("assets/fig_leaderboard_validation.png"),
  caption: [
    (a) Mean squared error and (b) coefficient of determination on the 600-clip DREGON-LM validation set. Lower MSE and higher $R^2$ are better.
  ],
)

#table(
  columns: (auto, 2fr, auto, auto, auto),
  inset: 6pt,
  align: (center, left, center, center, center),
  table.header(
    [*Rank*], [*Model*], [*MSE $arrow$*], [*$R^2$ $arrow$*], [*Params*],
  ),
  [1], [v2 (SE+Attn)], [2.61], [0.951], [1.50M],
  [2], [BiGRU-v2], [2.67], [0.948], [1.44M],
  [3], [BiGRU], [2.74], [0.945], [0.67M],
  [4], [TCN], [3.09], [0.936], [1.38M],
  [5], [MagPhase], [3.16], [0.917], [0.67M],
  [6], [AttnPool], [4.87], [0.860], [0.56M],
  [7], [Wide], [5.04], [0.847], [3.94M],
  [8], [MultiScale], [5.15], [0.840], [1.36M],
  [9], [Baseline], [5.21], [0.837], [0.54M],
  [10], [SE-Next], [7.30], [0.688], [1.41M],
)

#figure(
  image("assets/fig_pareto_params_r2.png"),
  caption: [
    Parameter count vs. validation $R^2$. The BiGRU family dominates the Pareto frontier; BiGRU (0.67M params) offers 99.4% of v2's performance at 44% of the parameters.
  ],
)

== Full-sequence Evaluation

#figure(
  image("assets/fig_fullsequence_comparison.png"),
  caption: [
    (Top) Mean predicted rotor speed vs. time for five representative variants, overlaid with ground truth. (Bottom) Per-frame MSE (1-s smoothed). BiGRU-v2 shows the tightest tracking and lowest error.
  ],
)

#table(
  columns: (2fr, auto, auto, auto),
  inset: 6pt,
  align: (left, center, center, center),
  table.header(
    [*Model*], [*Global MSE*], [*In-flight MSE $arrow$*], [*Global $R^2$*],
  ),
  strong[BiGRU-v2], strong[73.60], strong[9.90], strong[0.839],
  [v2 (SE+Attn)], [137.48], [11.80], [0.700],
  [BiGRU], [104.10], [15.19], [0.772],
  [Wide], [106.85], [18.87], [0.766],
  [AttnPool], [110.88], [19.99], [0.758],
  [Baseline], [104.14], [24.45], [0.772],
  [SE-Next], [113.67], [23.44], [0.752],
  [MagPhase], [91.71], [25.73], [0.800],
  [TCN], [105.82], [28.14], [0.769],
  [MultiScale], [1940.89], [111.71], [$-0.862$],
)

#figure(
  image("assets/fig_fullsequence_inflight_mse_bar.png"),
  caption: [
    In-flight MSE on the full-sequence recording. BiGRU-v2 halves the baseline error.
  ],
)

== Individual-motor and allMotors Evaluation

All models were trained on four-rotor mixtures with varying speeds. When evaluated on clean single-rotor recordings (constant speed, no speech), all ten variants fail catastrophically (MSE in the thousands), confirming that the network has learned a strong structural prior: it expects four independent rotors and cannot reconcile single-rotor input with its internal model.

#figure(
  image("assets/fig_individual_motor_mse_bar.png"),
  caption: [
    Best-channel MSE on individual-motor recordings. All variants fail by two to three orders of magnitude, as expected.
  ],
)

On allMotors_70 (four synchronized rotors at 70 rev/s), the models behave much better because the input matches their structural expectation:

#table(
  columns: (2fr, auto, auto),
  inset: 6pt,
  align: (left, center, center),
  table.header(
    [*Model*], [*Best MSE $arrow$*], [*Avg MSE $arrow$*],
  ),
  [MultiScale], [16.10], [28.93],
  [Baseline], [22.26], [91.07],
  [Wide], [22.24], [85.26],
  [BiGRU-v2], [22.30], [89.29],
  [SE-Next], [24.12], [94.10],
  [TCN], [28.96], [65.85],
  [AttnPool], [30.06], [113.03],
  [v2 (SE+Attn)], [316.55], [347.13],
  [BiGRU], [392.28], [416.77],
  [MagPhase], [640.30], [676.49],
)

#figure(
  image("assets/fig_single_rotor_allmotors_comparison.png"),
  caption: [
    Predictions on allMotors_70 for five variants. Dotted lines = four output channels; solid = mean; bold = best channel.
  ],
)

#figure(
  image("assets/fig_allmotors_mse_bar.png"),
  caption: [
    MSE on allMotors_70 comparing best channel vs. mean over four channels.
  ],
)

= Discussion

== What Works

1. *BiGRU temporal head is the dominant improvement.* Adding BiGRU alone jumps $R^2$ from 0.837 to 0.945 --- the largest single gain. Every top-5 model has it.
2. *Deeper encoder + SE helps generalisation.* BiGRU-v2 (6 blocks + SE) matches v2 on validation ($R^2$ 0.948 vs 0.951) but wins decisively on the real recording (in-flight MSE 9.9 vs 11.8).
3. *TCN is the best non-recurrent architecture.* Dilated convolutions give $R^2$ = 0.936, competitive but $~0.015$ behind the BiGRU family.

== What Does Not Work

4. *SE-Next is actively harmful.* Without temporal modelling, the SE-heavy 6-block encoder achieves $R^2$ = 0.688 --- worse than baseline.
5. *Width scaling does not help.* The wide model (3.94M params, 7.3$times$ baseline) barely beats baseline on validation.
6. *Phase input adds complexity without gain.* MagPhase_bigru (3-channel input) at $R^2$ = 0.917 is 0.028 behind plain BiGRU.
7. *Multi-scale fusion is unstable.* Oscillating validation loss and broken temporal resolution on full sequences.
8. *Attention pooling is modest.* +0.023 $R^2$ over baseline, far behind BiGRU.

== Practical Recommendation

For downstream use in speech-enhancement pipelines that depend on accurate RPS conditioning, *SimpleConv-BiGRU-v2* offers the best trade-off:
- Strongest real-recording performance (in-flight MSE 9.9, vs baseline 24.4)
- Competitive validation accuracy ($R^2$ 0.948, within 0.003 of the best)
- Moderate size (1.44M params, 2.7$times$ baseline)

If parameter count is critical, *SimpleConv-BiGRU* (0.67M params, $R^2$ 0.945) provides 99.4% of v2's validation performance at 44% of the parameters, with real-recording in-flight MSE of 15.2.
