# Estimating Rotor Speeds from Drone Ego-Noise with Scarce Annotated Data

Draft v0.2 (2026-08-20). Narrative draft: prose and real numbers, figures and
some baseline rows pending. Markers: **[PENDING]** = the experiment is queued
or running; **[TO RUN]** = a planned baseline with no numbers yet;
**[FIGURE]** = a figure to be prepared; **[TODO]** = a detail to verify or
fill in. Strip this block, the markers, and Appendix B before submission.

## Abstract

A drone's rotors imprint a harmonic structure on every audio recording made
on board: a rotor spinning at rate $f$ radiates energy at $f, 2f, 3f, \dots$,
and the positions of these harmonics follow the rotor speed exactly.
Knowledge of the instantaneous motor speeds enables informed ego-noise
suppression and acoustic drone monitoring. We study the task of estimating
the four motor speeds of a quadrotor, per time frame, from on-board audio
alone. Annotated data for this task is scarce: the recordings with motor
speed logs available to us amount to about an hour of audio across two drone
models. Neural regression models trained on this data reach low error on
held-out segments of the training recordings and degrade sharply across
microphones, flight regimes, and drone models. A direct probe explains the
degradation: when we scale the frequency axis of the input by 2%, the
predictions change by 0.03% on average, so the models estimate speeds from
loudness, spectral shape, and the training prior over speed values.
Augmentations that transform the audio and the speed labels together —
time-warping and frequency scaling — raise the models' use of the frequency
cue and improve generalization, at a measurable cost in cruise precision. To
diversify the data further, we build a neural noise generator that is
constrained by construction to produce a harmonic-plus-broadband signal
driven by a given speed trajectory. Adding its output to the training data as
extra recordings makes the models worse, because the synthetic amplitude
patterns offer an easier cue than the real recordings do. We therefore train
on generated data under the same label-transforming augmentation schedule as
on real data **[PENDING: outcome]**. The task has no published direct
baselines, so we compare against methods from two adjacent tasks: multi-pitch
tracking (salience-map models, including a variant modified for the required
frequency resolution) and tacholess order tracking (ridge detection,
iterative adaptive Vold–Kalman filtering, and our own two-stage blind
tracking method, which reaches 0.7–1.0 rev/s on cruise segments against
corrected telemetry and 1.0–1.2 rev/s against measured motor speeds on
recordings held out from all tuning). The comparison makes explicit what
neural models trained on scarce data add over signal processing, and what
they lose.

## 1. Introduction

Most drone applications rely on video. Audio recorded on board could support
voice interaction, acoustic scene monitoring, and self-diagnostics, and it is
held back by one dominant obstacle: the ego-noise of the rotors, which buries
other sources at signal-to-noise ratios between 0 and −30 dB. The ego-noise
has structure. Each rotor produces a comb of harmonics whose spacing equals
the rotor's rotation rate, plus broadband aerodynamic noise. A system that
knows the four instantaneous motor speeds knows where every harmonic sits,
and can use this knowledge for informed suppression of the noise.

Motor speeds are available from the flight controller on some platforms, and
this channel is often impractical: many commercial drones do not expose the
logs, logged speeds require careful time alignment with the audio, and (as we
show in Section 3.3) the logs themselves carry timing and value errors that
matter at the precision the harmonics demand. This motivates the question of
this paper:

> Can we train a model that predicts motor speeds from drone audio alone, and
> does so robustly — across microphones, recordings, flight regimes, and
> drone models?

Answering it leads to a study of what neural models learn when the annotated
data is two drones and about an hour of audio. Our contributions:

1. An evaluation of neural motor-speed regression on two data sources with
   motor speed logs, with a protocol that separates held-out-segment error
   from cross-microphone, cross-regime, and cross-drone generalization
   (Section 5).
2. A probe showing that models trained on this data barely respond to
   frequency scaling of the input, together with an augmentation family
   (time-warping and frequency scaling, both transforming the speed labels
   consistently with the audio) that raises the models' use of the frequency
   cue and improves generalization (Section 6).
3. A structured neural noise generator, constrained by construction to
   harmonic-plus-broadband output driven by a given speed trajectory, with
   two findings we found necessary for its use: the fidelity of its
   harmonics is limited by the accuracy of the motor speed annotations of
   its training data, and aggregate spectral losses select checkpoints with
   degraded harmonic structure, so checkpoint selection needs a per-harmonic
   measurement (Section 7).
4. An evaluation of generated data as training material: added naively, it
   makes models worse; combined with shortcut-removing measures it
   transfers; combined with the label-transforming augmentation schedule
   **[PENDING]** (Section 8).
5. Baselines from two adjacent tasks, including our own two-stage blind
   tracking method — a comb-matched search over the full rate range followed
   by refinement against the acoustic comb — which reaches 0.7–1.0 rev/s on
   cruise segments without any learning and serves as the reference point
   for the neural results (Section 4).

## 2. Related work

**Drone ego-noise.** [Cite: DREGON and its localization line; ego-noise
reduction with microphone arrays; single-channel enhancement studies; our
earlier technical report on drone noise modeling — replace with the citable
version.] Prior work treats motor speed logs as given side information for
suppression. Estimating the speeds from audio has received little direct
attention.

**Multi-pitch tracking.** Estimating several concurrent fundamental
frequencies is a mature task in music information retrieval, with
salience-map models as the standard family [cite: deep salience / multi-f0
CNN models; Basic Pitch]. A quadrotor at cruise is a hard instance of this
task: four fundamentals inside one octave (roughly 55–110 Hz), frequently
within 1 Hz of each other, under broadband noise.

**Order tracking.** Rotating-machinery analysis estimates shaft-speed
profiles from vibration or sound. Tacholess order tracking extracts the
speed from the signal itself, classically by ridge detection on a
time-frequency representation, and more recently by adaptive filtering
methods such as iterative adaptive Vold–Kalman filtering (IAVKF) [cite]. The
drone case adds two complications: four simultaneous, near-equal shafts, and
flight-regime changes (idle, ramps, cruise, landing).

## 3. Task and data

### 3.1 Task definition

Given a mono or multichannel audio clip from a drone-mounted microphone,
predict the four motor speeds, in revolutions per second (rev/s), on a
regular time grid (one estimate per STFT frame). Rotor identity is
unobservable from audio: two rotors at nearly equal speeds can swap labels
freely. All losses and metrics therefore follow the permutation-invariant
training (PIT) convention: the predicted quadruple is matched to the
ground-truth quadruple in the best order before the error is computed. We
report PIT-matched mean squared error (PIT-MSE, rev²/s²) for training-time
comparisons and PIT-matched MAE or RMSE (rev/s) for interpretable
comparisons. One rev/s of error displaces the fortieth harmonic by 40 Hz, so
errors well under 1 rev/s are needed for harmonic-informed suppression.

Throughout the paper, *flight regime* means a phase of flight — ground,
idle/warm-up, take-off ramp, cruise, landing — and we tag segments by the
logged speeds (cruise: all four rotors above 45 rev/s; ground: all rotors
stopped; the remaining powered segments are ramps and warm-up). The *full
flight envelope* means all powered regimes together, from the first motor
spin-up to the last spin-down.

### 3.2 Datasets

Two data sources provide drone audio with motor speed logs.

- **DREGON** [cite] (public): an 8-microphone array under a MikroKopter
  quadrotor, indoor free flight, six flight recordings, roughly half an hour
  of audio counted per channel. All flight recordings log commanded motor
  speeds; a subset of recordings additionally logs measured motor speeds
  from the motor controllers, and these recordings serve as our only
  tachometer-grade reference.
- **MD2** (unpublished; recorded by Michael Clayton at Queen Mary University
  of London, first described in [our technical report — replace with the
  citable version]): two 8-channel outdoor free-flight recordings of a DJI
  Matrice 100 with an on-top circular microphone ring, ~5 minutes of
  8-channel audio (~40 minutes counted per channel), with DJI SDK telemetry
  logs.

For training we mix these recordings with LibriSpeech utterances at SNRs
from −30 to 0 dB, resampled to 16 kHz, generating mixtures on the fly so
every epoch sees fresh combinations. Validation uses fixed held-out time
spans of each recording. Between them, the two sources contain a few dozen
distinct speed trajectories. This is the scarcity that drives the rest of
the paper.

### 3.3 Annotation quality

The logs are imperfect in ways that matter at harmonic precision. Our
measuring tool for all three corrections below is the *harmonic
reconstruction residual*: fit a harmonic model (a Vold–Kalman decomposition,
Section 4.2) to the audio at candidate speed trajectories, and measure the
residual energy; a candidate that places the comb closer to the true
harmonics leaves a smaller residual.

- **Timing.** The MD2 telemetry is late relative to the audio, and the error
  grows linearly with time (a sample-clock dilation). We measured and
  corrected both constants per recording (residual timing error ~3–5 ms RMS
  after correction).
- **Values.** The MD2 logged speeds are ~0.7% low; we apply a measured
  multiplicative correction. For DREGON we measured a bias between 0.35%
  and 0.85%, depending on the estimation method; no independent measurement
  exists that would decide between the candidate corrections, so we leave
  the DREGON labels unchanged and carry the range as an uncertainty floor
  (0.3–0.7 rev/s at cruise speeds) under every DREGON number in this paper.
- **Refinement.** On top of the corrected logs we compute refined
  annotations: per-window speed trajectories obtained by gradient-based
  minimization of the harmonic reconstruction residual, starting from the
  telemetry. The procedure's precision floor, estimated on the recordings
  with measured speeds, is ~0.2 rev/s; the same floor applies wherever the
  procedure reappears (Section 4.2, stage 2).

Section 7 shows that the difference between raw and refined annotations is
decisive for the noise generator. **[FIGURE: spectrogram with telemetry vs
refined harmonic overlays]**

## 4. Baselines from adjacent tasks

The task has no published direct baselines. We assemble baselines from the
two adjacent fields, so that the neural results of Sections 5–8 can be
judged against existing methods. Two evaluation protocols appear in this
paper and their numbers are never comparable across protocols: the *mixture
protocol* (fixed held-out spans of the training recordings, mixed with
speech; Sections 4.1 and 5) and the *window protocol* (16-second windows of
the raw recordings spanning both drones and all flight regimes, per-window
PIT-matched MAE against corrected telemetry; Sections 4.2 and 6). Each table
names its protocol.

### 4.1 Multi-pitch tracking

We adapt two salience-map models to the rotor band and train them with
binary cross-entropy against binary salience targets placed at the
ground-truth fundamental bins: a CNN over a harmonic CQT front-end [cite:
multi-f0 salience CNN] and Basic Pitch [cite]. Per-frame salience peaks are
linked into four trajectories by optimal bipartite (Hungarian) assignment.
Off the shelf, both models fail on this data, and the failure is
informative: on DREGON validation, 55% of frames have two rotors within
1 Hz of each other — comparable to the ~0.9 Hz bin spacing of the standard
salience grids — so the two peaks merge and the tracked trajectories
collapse toward their mean. A modified variant with the input band narrowed
to one octave (55–110 Hz) and a super-resolution output head (0.15 Hz/bin,
trained end-to-end) resolves the near-unison rotors (per-rotor MAE spread
compresses from 1.2–5.5 Hz to 1.8–2.9 Hz).

Mixture protocol, DREGON validation (30 clips × 8 channels), PIT-matched:

| Method | RMSE (rev/s) | R² |
|---|---|---|
| Direct regression (best neural model of Section 5, same data) | 1.62 | 0.93 |
| Salience CNN (standard grid) | 6.30 | 0.19 |
| Salience CNN (narrow band + super-resolution) | 4.03 | 0.57 |
| Basic Pitch (standard grid) | 23.2 | −16.2 |
| Basic Pitch (narrow band + super-resolution) | 11.7 | −3.2 |
| Classical multi-pitch trackers **[TO RUN]** | — | — |

(R² is averaged per clip, so rows imply slightly different target variances;
RMSE is the comparable column.)

After the resolution fix, the best salience model still trails direct
regression by a factor of ~2.5. Salience models designed for discrete
musical notes are a poor fit for four continuous, crossing, near-unison
fundamentals.

### 4.2 Tacholess order tracking

Order tracking methods need no training data, which makes them a natural
reference point for a learning approach that has almost none. All numbers
below use the window protocol.

- **Ridge detection** on a time-frequency representation, the classical
  tacholess method **[TO RUN as a standalone row]**.
- **IAVKF** [cite]: our reimplementation (the method extracts one harmonic
  component at a time — "peeling" — with adaptive bandwidth)
  **[TO RUN: comparison table]**.
- **Our two-stage method.** Stage 1 finds four speed trajectories blindly: a
  comb-match score over the full plausible rate range is tracked through
  time by dynamic programming, with explicit handling of ramps and of the
  near-unison rotor pairs. Stage 2 refines each trajectory by minimizing the
  harmonic reconstruction residual of Section 3.3 over the trajectory: a
  coupled Vold–Kalman decomposition at the candidate speeds is fitted, its
  residual differentiated with respect to the trajectory, and the trajectory
  updated until the reconstruction aligns with the acoustic comb to a
  fraction of the comb spacing. The stage-2 precision floor is the ~0.2
  rev/s of Section 3.3. On DREGON recordings held out from all tuning,
  scored against the measured motor speeds, the full blind method reaches
  0.97–1.22 rev/s.

| Method | DREGON cruise | MD2 cruise |
|---|---|---|
| Two-stage blind (ours) | 0.69 | 1.03 |
| Two-stage, initialized from telemetry (oracle) | 0.85 | 0.78 |
| Best neural rows (Section 6 training setup) | 2.5–2.9 | 2.3–2.4 |

(Window protocol, PIT-MAE, rev/s. The DREGON uncertainty floor of
Section 3.3 — 0.3–0.7 rev/s at cruise — sits under both DREGON columns:
DREGON numbers this small measure agreement with telemetry, and absolute
accuracy is bounded by the floor. The oracle row is better on MD2 and worse
on DREGON because the blind search can escape a local optimum that the
telemetry initialization commits to.)

This comparison is the reference for the rest of the paper: on steady
flight, a signal processing method with no training data is 2–4× more
precise at cruise than the neural models. The value of the neural models
must then come from the other columns of the comparison — coverage of ramps
and idle, single-pass runtime, robustness where the comb assumption
degrades — and Section 9 collects those columns.

## 5. Neural models overfit the scarce data

We train three regression architectures spanning a capacity range: a compact
convolutional network, the same network with a recurrent (GRU) head, and the
same network with a Transformer head **[TODO: parameter counts, STFT
configuration, and output frame rate in a small table]**. All take STFT
features of 1-second audio chunks and emit four speeds per frame, trained
with PIT-MSE on online-generated mixtures. On held-out time spans of the
training recordings, the three architectures reach PIT-MSE between ~7 and
~12 rev²/s² (best 7.3, i.e. ~2.7 rev/s RMSE), with the exact value depending
on the architecture and the mixing setup.

Generalization is much worse, in three nested ways:

- **Across microphones.** Models trained on one microphone of the 8-channel
  arrays degrade on the others; treating channels as extra training samples
  restores parity at no architectural cost.
- **Across flight regimes.** Models trained on cruise-filtered data fail on
  ground and ramp segments in proportion to how far the speeds sit from the
  cruise prior: per-regime PIT-MSE reaches 2450 rev²/s² on ground segments,
  against ~15 at cruise. Keeping the full flight envelope of the same real
  recordings in training, take-off ramp included, cuts the all-regime error
  from 338 to 80 rev²/s² (Transformer) with no new data.
- **Across drones.** A model trained on DREGON alone scores 7.96 rev/s
  PIT-RMSE on an MD2 recording. Adding the other MD2 recording to training
  drops the error on the held-out one to 1.63 rev/s. One recording of a new
  drone suffices, and we observe no zero-shot transfer between drone models.

The capacity ordering confirms overfitting as the mechanism: the Transformer
head, largest of the three, has the worst held-out error before augmentation
(11.76 PIT-MSE, against 9.71 for the smallest under the same setup), and it
gains the most from every diversification below.

## 6. Augmentation, and what the models actually read

**The probe.** We resampled validation audio to scale its frequency axis by
×1.02 — which shifts every harmonic, and hence the apparent motor speeds, by
2% — and measured the mean relative change of the PIT-matched predictions.
The ideal response is 2%. The measured response is 0.03%, across
architectures. The models' predictions are driven by loudness, spectral
shape, and the training prior over speed values; the harmonic positions
contribute two orders of magnitude less than they should. This one number
explains both the good held-out results (the prior is correct on data that
resembles training) and the sharp cross-condition degradation (the prior is
wrong everywhere else). **[FIGURE: predicted speed vs input frequency
scaling, per architecture]**

**Augmentations that transform the labels.** Standard audio augmentations
(gain, polarity, channel drop) keep the speed labels fixed. Two
augmentations transform the audio and the labels together and thereby
manufacture genuinely new audio-and-speed pairs:

- **Time-warping**: resample the noise chunk by a slowly varying rate α(t)
  (|α−1| ≤ 0.12) and warp the speed labels identically. Same spectral
  content, new valid trajectory. Held-out PIT-MSE: Transformer 11.76 → 8.74
  (−26%), convolutional 9.71 → 8.85, GRU unchanged.
- **Frequency scaling**: resample the chunk by a constant α ∈ [0.75, 1.3]
  and multiply the labels by α. This moves the whole comb and the mean of
  the label distribution together, so a model that ignores harmonic
  positions cannot fit the augmented data. (The range is asymmetric so that
  the scaled speeds stay inside the physically plausible band **[TODO:
  verify the stated reason]**.)

Frequency scaling has a consistent, two-sided effect. For the Transformer
setup, full-envelope PIT-MSE improves from 63.7 to 37.6 while DREGON cruise
MAE worsens from 2.48 to 3.23 rev/s (window protocol); the same trade
appears on every architecture we applied it to. The models can be pushed to
read frequency, and the push costs part of the prior-driven precision at
cruise. Augmentation recombines the recordings we have; new drone timbres,
microphone placements, and speed dynamics have to come from somewhere else.
This motivates synthetic data.

## 7. A noise generator constrained to harmonic structure

We build a generator with the harmonic structure hard-wired: for each rotor,
an oscillator bank at multiples $k \cdot f_r(t)$ of the given speed
trajectory, up to order $k = 80$, with learned time-varying amplitudes, plus
a learned broadband component; both are propagated to the microphone
positions of the target array by free-field delay and distance attenuation
with learned per-microphone corrections. A per-drone embedding (optionally
with per-rotor sub-embeddings) conditions the learned parts. By construction
the generator can only place harmonic energy on the commanded comb, and it
produces unlimited multichannel noise with exact speed labels.

We evaluate the generator's harmonic fidelity with a *per-harmonic line
measurement*: on validation audio, the floor-subtracted power of the
generated signal at each harmonic of the reference trajectory, compared with
the same quantity measured on the real recording. Aggregate spectral
distances cannot see this quantity — harmonics are sparse, so a generator
that drops every line above order 50 barely moves a multi-resolution STFT
distance.

Two findings from this instrument:

**Annotation accuracy limits the harmonics.** Trained on data whose speed
labels carry a constant −0.54% bias (one of the measured candidate values of
the DREGON bias, Section 3.3), the generator's harmonics above order ~50
collapse: line power drops by 8.6 dB relative to training on exact labels.
At high orders the biased comb misses the true line positions by more than a
line width, the harmonic bank can no longer explain the observed energy, and
the optimizer reassigns that energy to the broadband component. Training on
the refined annotations of Section 3.3 restores the line *power* at orders
10–49 on real data. Line *sharpness* above order ~25 remains limited by the
training objective itself: the longest window of the multi-resolution STFT
loss is shorter than the beat period of adjacent harmonics there, so the
loss provides no gradient for line width at those orders, whatever the
label quality.

**Aggregate losses select the wrong checkpoints.** Across training epochs,
the aggregate spectral distance keeps improving while the line measurement
peaks early and then erodes; the correlation between the two across epochs
is negative (r = −0.37 in the largest run; the same pattern appeared in all
four training runs where we tracked both). A generator picked by the
standard validation loss has visibly and audibly washed-out harmonics. We
therefore select checkpoints by the line measurement on validation audio of
both drones. **[FIGURE: aggregate loss and line measurement vs epoch;
spectrograms and audio of the two selections]** **[PENDING: the line-vs-loss
checkpoint comparison on both drones]**

## 8. Generated data as training material

Three experiments, in the order we ran them.

**Added naively, generated data hurts.** Adding generator output to the real
training pool as extra noise recordings increased validation PIT-MSE by 27%
**[TODO: recover the absolute pair for this number]**. The synthetic
amplitude patterns follow the speed trajectory by construction, which hands
the model the loudness cue of Section 6 in a purer form than real
recordings do.

**With the shortcut removed, generated-only training transfers.** Two
measures: mix the generator's output in even proportion with an analytic
comb — a synthetic harmonic signal whose per-harmonic amplitudes are fixed
and speed-independent, so no single loudness-to-speed mapping holds across
the pool — and apply the label-fixed augmentations (gain, polarity, channel
drop) from the start of training. Models trained purely on this synthetic
pool, with no real audio, reach 17.8–25.4 PIT-MSE on real free-flight
validation, against ~7.3 for real-data training on its own validation
protocol; a short real-data fine-tune brings them to 11.1–14.1. One honest
note on the history of this result: our first version of this experiment
appeared to fail completely (negative R²), and the failure traced to a
contaminated validation set — ground-idle segments of one recording labeled
as flight. The trap is generic: at low speeds the evaluation data is as
fragile as the models.

**The open experiment.** The transfer result above used only the label-fixed
augmentations. The label-transforming schedule of Section 6 — the one that
forces frequency reading on real data — has not yet been combined with the
strongest generator (refined annotations on both drones, per-rotor
embeddings, line-measurement checkpoint selection). **[PENDING]** Two
outcomes are possible, and the paper reports whichever the experiment
returns:

- *If it helps:* the generator supplies the timbre and trajectory diversity
  the real corpus lacks, the augmentations prevent the shortcut fitting
  that sank the naive attempt, and the combination improves cross-regime or
  cross-drone error over real-data training. The paper's answer: scarce
  annotations can be leveraged into robust models by structured synthesis
  plus label-transforming augmentation.
- *If it does not help:* we report the result with an analysis of where the
  synthetic distribution still fails the models, using the instruments of
  Section 7 (line sharpness statistics, amplitude-trajectory coupling,
  broadband texture). The paper's answer: with the harmonic structure
  guaranteed and the shortcuts removed, the remaining gap between synthetic
  and real drone noise sits in the component the measurements indicate, and
  closing it is the prerequisite for synthetic training data on this task.

## 9. Discussion: what the neural models add

Collecting the comparison started in Section 4.2:

- **Cruise precision** favors signal processing: 0.69–1.03 rev/s blind,
  against 2.3–2.9 rev/s for the best neural rows (window protocol).
- **Coverage** favors the neural models: they emit estimates on ramps,
  idle, and near-silence, where the two-stage method's comb assumptions and
  16-second windows degrade. **[TO RUN: the order-tracking baselines on the
  full-envelope protocol, so this row has numbers on both sides.]**
- **Latency and cost** favor the neural models: one forward pass per frame
  against per-window iterative optimization (seconds to minutes per
  16-second window on CPU). Applications that need speeds online favor the
  neural path.
- **A hybrid** outperformed both of its components on the windows tested:
  neural estimates used as the initialization of the stage-2 refinement
  reached 0.64 rev/s, against 0.87 for the neural estimates alone and
  against the blind method's numbers above. **[PENDING: the hybrid on the
  full window protocol.]**

Limitations: two drone models; tachometer-grade measured speeds exist only
for a subset of DREGON recordings; most numbers come from single training
seeds; all flights are indoor (DREGON) or calm outdoor (MD2); and the
annotation corrections of Section 3.3 are themselves estimated from the
audio, so they share its assumptions — in particular, every DREGON number
carries the 0.3–0.7 rev/s uncertainty floor.

## 10. Conclusion

Can we train a model to predict motor speeds from drone audio, robustly? On
scarce annotated data, the answer so far has three parts. First, plain
training produces models whose held-out error looks good while their
predictions rely on loudness and the speed prior; the frequency-scaling
probe of Section 6 measures this directly and costs nothing, and we suggest
it as a sanity check for models of this kind. Second, label-transforming
augmentations and full-envelope training data repair part of the problem at
a measurable cost in cruise precision, and a training-free two-stage
tracking method remains 2–4× more precise at cruise — a reference we think
learning approaches on this task should report against. Third, structured
synthesis can manufacture the missing diversity only when the annotations
are refined to comb precision and checkpoints are selected by a
per-harmonic measurement; whether it then closes the robustness gap is the
experiment this paper resolves. **[PENDING]**

---

## Appendix A. Planned figure list

1. Task figure: spectrogram of drone noise with the four-rotor comb and the
   telemetry/refined overlays (Section 3).
2. Frequency-scaling probe: prediction response vs input scaling (Section 6).
3. Per-regime error bars: cruise-trained vs full-envelope vs synthetic
   training data (Section 5).
4. Generator line measurement: aggregate loss and line measurement across
   epochs, with spectrograms of the two checkpoint selections (Section 7).
5. Baseline scoreboard: precision vs coverage for all methods (Section 9).

## Appendix B. Experiments backing each claim

Internal bookkeeping for the authors; strip before submission. Each claim
above maps to an experiment log in `docs/experiments/`; the mapping table
lives in `writing/papers/2026-08_wrapup/inventory.md`.
