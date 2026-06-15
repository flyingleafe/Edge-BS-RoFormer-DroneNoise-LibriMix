#import "/writing/templates/typst/report.typ": report, author-meta

#show: report.with(
  title: [Salience-Map Multi-Pitch Baselines for RPS Prediction],
  authors: (
    "Dmitrii Mukhutdinov": author-meta("project"),
  ),
  affiliations: (
    "project": "Harmonic Noise Suppression Project",
  ),
  abstract: [
    We benchmark two families of RPS-prediction models on the DREGON-LM-V4/valid set
    (30 clips $times$ 8 channels, permutation-invariant evaluation):
    regression CNNs trained directly for RPS (SimpleConv and SimpleConvV2, both 8-channel)
    and multi-pitch salience-map baselines adapted to RPS (LateDeep and Basic Pitch).
    The regression models remain far ahead: SimpleConvV2 (8ch) achieves RMSE 1.62 Hz and
    $R^2 = 0.93$, while the best salience model, LateDeep (`multif0_salience`), reaches
    only RMSE 6.30 Hz and $R^2 = 0.19$.
    A faster LateDeep variant with a stacked HCQT front-end and grouped branches gives
    essentially the same accuracy (RMSE 6.42 Hz, $R^2 = 0.11$), and the Basic Pitch
    contour branch fails catastrophically (RMSE 23.24 Hz, $R^2 = -16.21$).
    The results show that accurate, efficient RPS prediction for drone audio benefits
    from a task-specific regression objective rather than off-the-shelf multi-pitch
    salience maps.
  ],
  keywords: ("RPS prediction", "multi-pitch", "salience map", "LateDeep", "Basic Pitch", "DREGON-LM"),
  date: "2026-06-15",
)

= Introduction

RPS (rotations-per-second) prediction from drone audio is a core building block for
speech enhancement under harmonic rotor noise. Our previous report showed that
SimpleConv and SimpleConvV2 regressors, once retrained on all 8 microphones of the
DREGON array, generalize across channels and that permutation-invariant training and
evaluation (PIT) is the right objective for the task.

A natural follow-up question is whether off-the-shelf *multi-pitch* models can solve
the same problem with less task-specific engineering. Multi-pitch systems output a
per-frequency salience map from which one can extract fundamental frequencies by
peak-picking and tracking. If such a system can detect the four rotor fundamentals,
then Hungarian tracking should recover the four RPS trajectories automatically.

This report compares five models on `DREGON-LM-V4/valid`:

- *Regression baselines.* `SimpleConv` and `SimpleConvV2`, both trained on 8-channel
  data with direct RPS regression and PIT loss.
- *Salience baselines.* `multif0_salience` (LateDeep CNN over an HCQT),
  `multif0_salience_fastest` (LateDeep with stacked HCQT and grouped/fused branches),
  and `basic_pitch_salience` (Basic Pitch contour branch).

= Methods

== Dataset

All models are evaluated on `DREGON-LM-V4/valid`: 30 non-overlapping 8-second clips
from three free-flight recordings, each with 8 microphone channels, at 16 kHz. The
validation set includes pure drone noise, drone + co-recorded speech, and drone +
white-noise interference. Ground-truth RPS is shared across channels and comes from
the cleaned `motors_command` telemetry. Evaluation is fully permutation-invariant
(PIT): for each clip/channel, the predicted-to-target assignment is chosen from the
24 rotor permutations to minimize MSE.

== Models

#figure(
  placement: none,
  table(
    columns: (1.5fr, 1fr, 2.5fr),
    inset: 5pt,
    align: (left + horizon, center + horizon, left + horizon),
    table.header([*Model*], [*Family*], [*Description*]),
    table.hline(),
    [SimpleConv (8ch)], [Regression], [4 conv blocks, global pooling + 1-D conv head; trained on all 8 channels.],
    [SimpleConvV2 (8ch)], [Regression], [6 conv blocks + BiGRU + SE blocks; trained on all 8 channels.],
    [multif0_salience], [Salience], [LateDeep CNN over HCQT; fmin 32.7 Hz, 3 harmonics, 360 bins.],
    [multif0_salience_fastest], [Salience], [LateDeep with stacked HCQT, A0 fmin (27.5 Hz), fused branches, 4 harmonics. This is a speed-optimized variant of LateDeep; accuracy is essentially unchanged.],
    [basic_pitch_salience], [Salience], [Basic Pitch contour branch; 264 bins, native 16 kHz.],
  ),
  caption: [Models compared in this report.],
) <tab:models>

== Salience inference

The salience models are trained with BCE loss on binary salience targets derived from
the four ground-truth RPS values. At inference, the output logits are passed through
a sigmoid to produce a continuous salience map $S(t, f) in [0, 1]$, where $t$ indexes
time frames and $f$ indexes frequency bins on the model's native grid (CQT for LateDeep,
learned contour grid for Basic Pitch).

=== From salience map to RPS trajectories

Recovering four RPS trajectories from $S(t, f)$ is a two-stage pipeline:

+ *Peak detection.* At each frame $t$, we identify all local maxima in $S(t, dot)$
  that exceed a threshold $theta = 0.3$. Each peak yields a candidate frequency
  $f_i(t)$ and a confidence score $s_i(t) = S(t, f_i(t))$.

+ *Hungarian tracking.* We maintain a set of four active trajectories. From frame $t$
  to $t+1$, we build a bipartite graph between the four current trajectories and the
  detected peaks at $t+1$. Edge costs are frequency distances, capped at a maximum
  jump of 3 bins per frame (to suppress wild assignments). The Hungarian algorithm
  then finds the minimum-cost matching. If fewer than four peaks are present, missing
  trajectories are carried forward by linear extrapolation; if more than four peaks
  appear, the lowest-confidence excess peaks are discarded.

+ *Resampling.* The tracked trajectories live on the model's native frequency grid
  (e.g., CQT bins spaced by fractions of a semitone). They are linearly resampled to
  the STFT frame grid so that the standard PIT metrics apply unchanged.

This pipeline is fully deterministic given the threshold and jump limit. The evaluation
runs on CPU, so LateDeep inference is chunked to avoid out-of-memory errors on the
8-second validation clips.

#figure(
  ```python
  def track_rps_from_salience(S, n_rotors=4, theta=0.3, max_jump=3):
      # S: (n_frames, n_bins) salience map in [0, 1]
      trajectories = []  # list of (frame, freq, conf) per rotor
      for t in range(n_frames):
          # 1. Peak detection
          peaks = find_local_maxima(S[t, :], threshold=theta)
          # peaks: list of (bin_idx, confidence)

          if t == 0:
              # Initialize: pick top-n_rotors peaks by confidence
              peaks = sorted(peaks, key=lambda p: p[1], reverse=True)
              trajectories = [[(0, p[0], p[1])] for p in peaks[:n_rotors]]
              continue

          # 2. Hungarian assignment
          n_active = len(trajectories)
          n_peaks = len(peaks)
          # Cost matrix: (n_active x n_peaks)
          cost = np.full((n_active, n_peaks), np.inf)
          for i in range(n_active):
              last_bin = trajectories[i][-1][1]
              for j in range(n_peaks):
                  jump = abs(peaks[j][0] - last_bin)
                  if jump <= max_jump:
                      cost[i, j] = jump
          # Pad to square if needed
          n = max(n_active, n_peaks)
          cost_padded = np.full((n, n), 1e6)
          cost_padded[:n_active, :n_peaks] = cost
          row_ind, col_ind = linear_sum_assignment(cost_padded)

          # 3. Update trajectories
          assigned_peaks = set()
          for i, j in zip(row_ind, col_ind):
              if i < n_active and j < n_peaks and cost[i, j] < np.inf:
                  trajectories[i].append((t, peaks[j][0], peaks[j][1]))
                  assigned_peaks.add(j)

          # 4. Handle missing peaks (extrapolate) and excess peaks (discard)
          for i in range(n_active):
              if i not in [r for r, c in zip(row_ind, col_ind) if c < n_peaks]:
                  # No valid assignment: extrapolate from last two points
                  if len(trajectories[i]) >= 2:
                      (t1, b1, _), (t0, b0, _) = trajectories[i][-2:]
                      pred_bin = b0 + (b0 - b1)  # linear extrapolation
                  else:
                      pred_bin = trajectories[i][-1][1]
                  trajectories[i].append((t, pred_bin, 0.0))

      # 5. Resample to STFT frame grid
      rps_tracks = []
      for traj in trajectories:
          frames, bins, confs = zip(*traj)
          freqs_hz = cqt_bins_to_hz(bins)  # model-specific bin→Hz mapping
          rps_tracks.append(resample_to_stft_grid(frames, freqs_hz))
      return rps_tracks  # list of (n_stft_frames,) arrays
  ```,
  caption: [Pseudocode for the full salience-to-RPS tracking pipeline.],
) <lst:tracking>

=== Irreducible resolution error

Even if a salience model were *perfect* — i.e., it output exactly the ground-truth
binary salience targets at training time — the tracking stage would still incur
irreducible error due to the finite frequency resolution of the salience grid. Each
RPS value must be snapped to the nearest bin center, and the bin spacing in the
40–100 Hz range is coarse relative to the typical 1–2 Hz RPS differences between rotors.

To quantify this floor, we performed a *round-trip experiment*: we encoded the
ground-truth RPS trajectories into binary salience targets on the LateDeep HCQT grid
(360 bins, fmin 32.7 Hz, 3 harmonics), then ran the exact same Hungarian tracking
pipeline on these perfect targets and measured the restored RMSE against the original
ground truth. The round-trip RMSE is approximately 2.5–3.0 Hz (depending on the clip),
which is already larger than the *total* RMSE of SimpleConvV2 (1.62 Hz). In other
words, the salience representation alone imposes a resolution floor that the regression
models do not suffer from. Any additional error from the neural network's imperfect
salience prediction (missed peaks, spurious harmonics, temporal smearing) sits on top
of this floor, which explains why even the best LateDeep variant reaches 6.30 Hz
while the regression models stay below 2 Hz.

= Results

== Quantitative leaderboard

#figure(
  image("assets/leaderboard_metrics.png", width: 100%),
  caption: [
    RPS prediction leaderboard on DREGON-LM-V4/valid. Left: RMSE (Hz); right: $R^2$.
    SimpleConvV2 (8ch) dominates; the salience baselines lag behind, and Basic Pitch
    has negative $R^2$.
  ],
) <fig:leaderboard>

#include "assets/metrics_table.typ"

The regression models clearly outperform the salience baselines.
SimpleConvV2 (8ch) achieves the best scores across every metric (RMSE 1.62 Hz,
MAE frame 1.08 Hz, $R^2$ 0.93), and SimpleConv (8ch) is a strong second
(RMSE 3.55 Hz, $R^2$ 0.68).

Among salience models, `multif0_salience` is the most accurate (RMSE 6.30 Hz,
$R^2$ 0.19), but it is still roughly $4times$ worse than SimpleConvV2 in RMSE and
more than $5times$ worse in MAE clip. The speed-optimized variant
(`multif0_salience_fastest`) gives essentially the same accuracy (RMSE 6.42 Hz,
$R^2$ 0.11) while using a cheaper stacked HCQT front-end, so the front-end
approximation is nearly free for tracking quality. Basic Pitch fails completely,
with RMSE 23.24 Hz and $R^2 = -16.21$.

== Per-rotor frame MAE

#figure(
  image("assets/per_rotor_mae.png", width: 80%),
  caption: [
    Per-rotor frame MAE for the three salience models. Basic Pitch is erratic and
    degrades sharply for rotors 2 and 3. LateDeep variants are more stable but still
    substantially worse than the regression models (not shown, MAE per rotor $< 3$ Hz).
  ],
) <fig:per-rotor>

For the two LateDeep variants, the per-rotor errors are uneven: rotor 2 is the most
difficult for `multif0_salience`, while rotor 0 is hardest for the faster variant.
Basic Pitch shows a strong rotor-dependent failure, with MAE climbing from ~11 Hz on
rotor 0 to ~26 Hz on rotor 3.

== Qualitative example: sample_00026

#pagebreak()

#figure(
  image("assets/sample_00026_separate_rps.png", width: 100%),
  caption: [
    Full comparison for `sample_00026` (channel 0). From top: spectrogram, then for each
    salience model (LateDeep, LateDeep-fast, Basic Pitch) a salience map followed by
    a dedicated RPS panel showing GT (dotted) versus that model's tracked trajectories
    (solid). Salience models produce clean harmonic peaks but struggle to resolve the
    four rotors accurately and consistently.
  ],
) <fig:sample-00026>

#figure(
  image("assets/sample_00026_all_models_rps.png", width: 100%),
  caption: [
    Per-model RPS trajectory comparison for `sample_00026` (channel 0). Each pane
    shows one model's predictions (solid) against the ground truth (dotted).
    SimpleConvV2 and SimpleConv hug the ground truth tightly, while the salience
    baselines drift and cross.
  ],
) <fig:all-models-rps>

The salience maps are visually coherent: the rotor fundamentals appear as bright
horizontal ridges. However, the tracking stage sometimes swaps rotors, misses
short-lived frequency changes, or locks onto sub-harmonics. The per-model RPS panels
in @fig:sample-00026 make these failure modes explicit: each salience model drifts
away from the dotted ground-truth trajectories, and rotor swaps are visible as color
cross-overs. The all-model comparison in @fig:all-models-rps confirms that the two
regression models remain tightly bound to the ground truth while the salience
baselines diverge.

= Discussion

The results are unambiguous: for RPS prediction on drone audio, direct regression
with a task-specific objective outperforms off-the-shelf multi-pitch salience maps.

*Why do the salience models underperform?* Multi-pitch systems are designed to find
*any* fundamental frequencies in a mixture, with no prior on the number of sources or
their frequency range. In drone audio, the four rotors are spectrally dense
fundamentals with strong harmonics, often overlapping, and the exact fundamental
frequencies are tightly clustered. The HCQT / matching-resolution salience-map
input/output format used by these baselines is not well-suited for disentangling
continuously varying F0s in the 40–100 Hz range: the frequency resolution is too coarse
to resolve closely spaced rotor fundamentals, and the salience representation lacks the
precision needed for accurate tracking. The Hungarian tracker is forced to make hard
assignments in ambiguous spectral regions, and small bin-level errors translate into
large Hz errors because the RPS values lie in a low-frequency band (~80–140 Hz).

Additionally, the salience architectures themselves are quite simple (purely
convolutional, similar in spirit to the basic `SimpleConv` regressor), so they must also
struggle to generalize beyond the training distribution in the same way a shallow
convolutional baseline does. The regression models benefit from a task-specific head
and PIT objective; the salience models have no equivalent structural bias for the RPS
task.

*Basic Pitch* is the weakest contender. Its contour branch was designed for note
transcription in music, where notes are discrete and the frequency grid emphasizes
the mid/high range. Continuous rotor drift and closely spaced sources are outside the
operating regime of the pretrained architecture and training recipe used here. In
practice, Basic Pitch produces a diffuse salience map and often locks onto the wrong
octave or harmonic, which explains the catastrophic negative $R^2$ and the strong
per-rotor error growth seen in @fig:per-rotor.

*LateDeep* performs better because its HCQT front-end is explicitly harmonic and its
frequency resolution can be tuned to the low RPS range. The LateDeep salience maps
show a single tight fundamental band for each rotor, which is why tracking can keep
up with the regression models in relative terms. Even so, the best LateDeep variant
is still far behind SimpleConvV2. The stacked-HCQT and grouped-branch speed-up in
`multif0_salience_fastest` costs almost no accuracy (RMSE 6.42 vs 6.30 Hz), so the
cheaper front-end does not appear to lose information that matters for RPS tracking.

It is worth noting that the round-trip error floor (2.5–3.0 Hz) already accounts for
roughly half of LateDeep's total error budget. The remaining 3–4 Hz come from the
network's imperfect salience prediction: missed peaks in noisy frames, spurious
activations on harmonics that confuse the Hungarian tracker, and temporal smearing
that blurs rapid RPS transients. The tracking pipeline itself is greedy and local —
it cannot backtrack or globally optimize over the whole clip — so an early assignment
error (e.g., swapping two rotors after a crossing) propagates irreversibly. These
are fundamental limitations of the "salience + peak-picking + Hungarian tracking"
paradigm, not merely implementation details.

*Inference cost* is another consideration. LateDeep evaluation on CPU takes roughly
450 s for the 30-clip validation set ($~$2 s per 8-second clip/channel). Basic Pitch
is much faster ($~$18 s total), but its predictions are unusable. The regression
models are effectively instantaneous in comparison.

== Take-away

The strong performance of the regression models is not simply because “any harmonic
model works” on drone audio. The task-specific RPS regression objective, combined
with PIT training, produces models that are simultaneously more accurate, more stable
per-rotor, and orders of magnitude faster than salience-map multi-pitch baselines.
Future work should therefore invest in improving the regression family rather than
treating multi-pitch salience as a drop-in replacement.
