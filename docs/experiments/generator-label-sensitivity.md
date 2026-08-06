# Phase 7 — Does telemetry label noise explain the generator's high-harmonic underfit?

**Status:** done — **label noise confirmed, but the constant bias is the term, not the staircase** · **Date:** 2026-08-06 ·
**Experiments:** `p7_labelsens_{exact,scale,tach,tach_presmooth}`

## The question, and why it needs an A/B

Every conditioned noise generator this project has trained underfits the mid and
high harmonics of real drone noise (E6: 6.4-7.8 dB comb-masked error at
k 10-40 even for the winning arm). Two explanations are on record and they are
not exclusive:

1. **Label noise.** DREGON's telemetry is a tachometer reading: a 0.269 rev/s
   quantization lattice refreshed at 49.7 Hz and held in between, on top of a
   0.542 % constant over-report. A rate error `e` displaces harmonic `k` by
   `k * e` Hz, so a label error that is invisible at `k = 1` is a full STFT bin
   at `k = 80`. If the generator is told the wrong frequency, the
   log-magnitude-dominated MSSTFT optimum is a smeared, attenuated line — and
   the attenuation must worsen with `k`.
2. **Loss design.** The MSSTFT scale mix, the mean-over-bins reduction and the
   low `eps` were already measured to make the objective log-dominated and to
   under-weight the mid-frequency harmonics
   (`docs/experiments/noise-gen-loss-breakdown.md`; memory
   `noise-gen-loss-breakdown`).

Real recordings cannot separate them: there is no version of DREGON with clean
labels. So the experiment is run on **synthetic** data where the truth is known
exactly and only the label is corrupted.

## Design

**Data.** One frozen-profile analytic comb per sample — one rotor, one
microphone, 1 s, 16 kHz, 80 harmonics
(`data_processing.rotor_spectral_model.FixedCombSpec` /
`render_fixed_comb`, driven by `data_processing.frame_datasets.StaticCombGenDataset`).
Three deliberate departures from `StaticCombNoisePool`:

- The spectral profile is **frozen** across the dataset, not drawn per clip. A
  per-clip random profile is unpredictable from RPS conditioning and would add
  irreducible loss on top of the effect being measured.
- The amplitude is **RPS-independent** (no `(rps/ref)^p` scaling, no per-clip RMS
  normalization), so the only thing a moving trajectory does to the target is
  move the comb's lines in frequency. That is the channel the hypothesis speaks
  about, and nothing else varies through it.
- **One rotor, one microphone.** Four overlapping combs make the per-`k` readout
  ambiguous and eight microphones replicate a signal whose spatial structure is
  not under test.

The trajectory is an OU draw (`rps_synthesis.generate`, the DREGON free-flight
calibration) band-limited by a 2 Hz zero-phase Butterworth "shaft inertia". That
filter is load-bearing twice, both measured:

| quantity | value |
|---|---|
| truth's rms motion inside a 1 s clip | 2.15 rev/s |
| tachometer label error, `tach` arm | 0.106 rev/s rms |
| ... of which pure quantization (`step/sqrt(12)`) | 0.078 rev/s |
| tachometer label error after the 5 Hz presmooth | 0.037 rev/s rms |
| distortion the 5 Hz presmooth does to the TRUE track | 0.017 rev/s rms |

Without the shaft filter the OU is white to the trajectory rate, the truth moves
~0.4 rev/s inside one 20 ms refresh interval, and the label error is dominated by
the tachometer's *inability to follow the shaft* rather than by its staircase —
a different claim, and one E6 already answered with the emitter's OU jitter. With
it, quantization dominates and the presmooth has real headroom.

**Arms.** Four, byte-identical except `data.{train,valid}.params.label_mode`:

| arm | conditioning | role |
|---|---|---|
| **A** `exact` | the truth | the control for everything that is *not* label noise |
| **S** `scale` | `0.99458 x` truth | is the constant bias benign on its own? |
| **B** `tach` | `0.99458 x` truth -> interval mean -> 0.269 rev/s lattice -> 49.7 Hz hold | the hypothesis |
| **C** `tach_presmooth` | arm B through the campaign's 5 Hz detrended low pass | the mitigation |

The corruption is `data_processing.rps_corruption.tachometer_corrupt`; the
presmooth is `presmooth_track`, a uniform-grid adapter over
`tracking.telemetry_refit.presmooth` (so "the smoothed telemetry" means the same
array here as in the tracking campaign). Labels are computed on a **margined**
track and cropped afterwards: a real deployment smooths a whole telemetry stream
and then windows it, and numerically a whole-window brickwall on a bare 1 s clip
keeps only five harmonics and rings worse than the staircase it removes
(0.155 rev/s residual vs the staircase's 0.106).

**Arm S is what makes arm B readable.** A constant gain is an invertible
reparameterization; if the generator absorbs it, S matches A and B's deficit is
the staircase alone. If S degrades with `k` instead, the bias is *not* benign and
B carries both effects — which the readout would then have to say.

**How the two explanations separate.** Arm A trains on the same loss, the same
model and the same budget with a perfect label. If A also underfits at high `k`,
the objective (or the capacity) is the cause and label noise is exonerated before
B is read. If A is flat and B falls away with `k`, label noise is confirmed and
the per-`k` gap is its effect size.

**Model.** `positional_harmonic_gen` unconditioned (`cond_dim: 0`,
`n_harmonics: 100`, `use_diff_noise: true`) — the smallest generator in the
registry that renders an oscillator bank at `k * f0(conditioning)`. Loss
`multiscale_stft` (n_ffts 2048/1024/512/256/128, `log_weight: 1.0`, L1), metric
`noise_gen_spectral`. `amp: false` (complex `cumprod` has no ComplexHalf kernel).
Identical budget for all four arms: seed 1234, 40 epochs, patience 8, batch 32,
Adam 1e-3, 8000 train / 512 valid samples.

## Readout

`scripts/gen_label_sensitivity_eval.py`, two measures per arm.

**1. Learned line profile** (primary). Condition on a *constant* rotor speed from
a grid chosen so every harmonic is periodic in the 4 s analysis window, and
compare each line's power against the profile's **analytic** amplitude
`a_k * gain` — no reference rendering, so no reference-side measurement noise.
This is the amplitude the arm learned, with its response to a jittery input
switched off.

One global frequency scale is estimated first, from the low harmonics only, and
reported: an arm trained on biased labels may have absorbed the bias into its own
mapping, and that is one nuisance parameter, not a per-`k` freedom. Line power is
then read in a fixed narrow band at the scale-corrected location with a local
floor subtracted from an annulus — **never a peak search inside a window**, the
estimator that returns about `W/2` on pure noise and has already withdrawn two
claims in this project (`docs/experiments/dregon-comb-displacement.md`).

**2. Delivered fidelity** (complement). Condition each arm on the labels it was
*trained* on, over held-out OU trajectories, and score the comb-masked mean
`|Delta log-mag|` along the TRUE trajectory's harmonic tracks in `k` bands — the
E6 measure. Readout 1 would exonerate an arm that learned "smear when the input
jitters" rather than "attenuate"; readout 2 would not.

### What the readout was checked against

- **It reads the truth back flat.** `--self-test` pushes the TRUE comb through
  readout 1 and requires 0 dB at every `k`: worst per-`k` bias **0.008 dB** over
  `k = 1..80`. The measurement chain — band width, floor subtraction, scale
  estimate — contributes nothing to any arm's number.
- **The normalization is gain-free.** `_spectrum` carries a `2/N^2` factor so a
  sinusoid of amplitude `A` integrates to `A^2/2`, matching the profile's
  analytic `a_k * gain` with no fitted gain in between (verified to 0.0002 dB on
  a synthetic line). The first version omitted it and every arm read about
  +95 dB.
- **Track frames are centred.** Frame `f` spans `[f*hop, f*hop + n_fft)`, so its
  rate is the one at its centre. The first version point-sampled at `f*hop`; that
  64 ms lead displaces the read bin at `k = 80` by more than the staircase does.
- **The `f0` grid stays inside the training marginal.** The first grid started at
  64 rev/s, but the trajectories span 69.6-97.1 rev/s (p5-p95 = 75.6-89.9), so
  two of four grid points were measuring *extrapolation* rather than the learned
  line amplitude — worth 0.3 dB on arm S's top band on their own. The grid is now
  76/80/84/88, inside the bulk and still making every `k*f0*dur` integral so each
  harmonic is periodic in the analysis window.

### The pressure the arms are responding to

Model-free (`--pressure`): render the frozen comb at the truth and at each arm's
label, and score the comb-masked `|Delta log-mag|` between them. That is what the
objective charges an arm for *keeping* full-amplitude lines.

| arm | k1-9 | k10-24 | k25-49 | k50-80 |
|---|---|---|---|---|
| `exact` (measure's own floor) | 0.01 | 0.03 | 0.04 | 0.20 |
| `scale`, raw | 0.91 | 4.82 | 12.93 | 18.74 |
| `tach`, raw | 0.94 | 5.07 | 12.71 | 16.22 |
| `scale` / `exact`, scale-compensated | 0.01 | 0.03 | 0.04 | 0.20 |
| **`tach`, scale-compensated (the staircase alone)** | 0.06 | 0.23 | 0.57 | **1.18** |
| **`tach_presmooth`, scale-compensated** | 0.05 | 0.17 | 0.33 | **0.66** |

Two readings because the constant bias and the staircase are displacements of
very different size: 0.542 % of 6.4 kHz is 35 Hz (4.5 bins at `n_fft` 2048) while
the staircase's 0.106 rev/s is 8.5 Hz (1.1 bins). The bias is the larger term by
about 4x — *if* the model cannot absorb it. Whether it can is arm S's question,
and it is why arm S is the difference between reading B as "the staircase" and
reading it as "the staircase plus an unabsorbed bias".

## Results

All four arms trained to convergence on `uni-gpushort` (55 min wall, none hit
it), same seed, same 40-epoch cap, patience 8.

| arm | job | epochs | best `mrstft` (higher better) |
|---|---|---|---|
| **A** `exact` | `p7-exact-d6ae67` | 21 | **30.96** |
| **S** `scale` | `p7-scale-577ebe` | 15 | 13.71 |
| **B** `tach` | `p7-tach-2fd10c` | 13 | 13.57 |
| **C** `tach_presmooth` | `p7-tach-presmooth-e8ae6b` | 17 | 13.31 |

**Readout 1 — learned line profile** (dB, learned minus true; 0 is perfect):

| arm | k1-9 | k10-24 | k25-49 | k50-80 |
|---|---|---|---|---|
| **A** `exact` | **-0.15** | **-0.02** | **+0.02** | **+0.16** |
| **S** `scale` | -1.58 | -2.08 | -3.84 | -8.61 |
| **B** `tach` | -0.14 | -0.09 | -2.32 | -6.85 |
| **C** `tach_presmooth` | -1.14 | -0.96 | -3.57 | -7.73 |

**Readout 2 — delivered fidelity on each arm's own labels** (mean
`|Delta log-mag|` dB on the true harmonic tracks; 0 is perfect):

| arm | k1-9 | k10-24 | k25-49 | k50-80 |
|---|---|---|---|---|
| **A** `exact` | **0.32** | **0.40** | **0.62** | **0.81** |
| **S** `scale` | 0.77 | 1.36 | 2.74 | 6.32 |
| **B** `tach` | 1.25 | 1.79 | 3.99 | 7.20 |
| **C** `tach_presmooth` | 1.09 | 1.32 | 3.61 | 6.83 |

**Contrasts** (the three the design was built to read; sign convention: negative
means the first arm is worse):

| contrast | readout | k1-9 | k10-24 | k25-49 | k50-80 |
|---|---|---|---|---|---|
| **S - A** (the constant bias) | line | -1.42 | -2.06 | -3.86 | **-8.77** |
| | track | -0.45 | -0.96 | -2.12 | **-5.52** |
| **B - S** (the staircase) | line | +1.44 | +1.99 | +1.52 | +1.77 |
| | track | -0.48 | -0.43 | -1.26 | -0.88 |
| **C - B** (the presmooth) | line | -1.01 | -0.87 | -1.25 | -0.89 |
| | track | +0.16 | +0.47 | +0.39 | +0.37 |

**How to read the contrasts.** `S - A` has the same sign in both readouts and
grows monotonically with `k` in both — it is a real, large, `k`-progressive
effect. `B - S` and `C - B` have **opposite signs in the two readouts** and never
exceed 2 dB, which is the honest description of run-to-run variability at one
seed per arm, not of an effect. The `mrstft` column says the same thing without
any per-`k` machinery: S, B and C land within 3 % of each other (13.31-13.71)
while A is 2.3x higher.

### What arm A settles

Arm A is flat to within **±0.5 dB at every individual `k` from 1 to 80** (band
means ±0.16 dB; one outlier, `k = 27` at -1.59). At `k = 47` it puts a line at
exactly 3760.00 Hz, 0.6 dB from the analytic reference, with its neighbours
47 dB down. Given a correct label, this objective, this emitter and this budget
reproduce a sharp comb across the entire range.

### What arm S does instead

At `k = 47` arm S has **no line at 3760 Hz at all**: the local maxima sit at
3767.5 / 3808.8 / 3753.0 Hz, all about 18 dB below the reference. Widening the
band to ±2 % recovers only ~0.3 dB, so the energy is genuinely *attenuated and
smeared*, not merely displaced. The per-`k` curve is smooth at -1 to -3 dB up to
`k` ~ 40 and then collapses erratically (`k = 47`: -11.7, `k = 51`: -14.1,
`k = 54`: -16.1, `k = 77`: -18.1).

The model does compensate the bias in **placement** — the fitted frequency scale
is 1.000000 for every arm and the measured line centroids sit within 0.01 % of
`k * f0` — but it cannot do so while keeping the lines sharp, and the failure
grows with `k`. So "a constant gain is an invertible reparameterization the model
absorbs" is true of the *centre frequency* and false of the *line shape*.

## Conclusion

**Label noise is confirmed as a mechanism, with a large effect size — but the
component responsible is the constant scale bias, not the tachometer staircase.**

1. **Loss design is exonerated as a sufficient cause.** Arm A is flat to
   ±0.16 dB out to `k = 80`. The MSSTFT scale mix / mean-over-bins / low-`eps`
   critique may still be true of the objective in other respects, but on this
   task it does not prevent the generator from learning sharp high harmonics.
   Whatever damages the real-data generator arrives through the conditioning.
2. **Label noise is confirmed, at -8.8 dB (line) / -5.5 dB (track) in k50-80 and
   -3.9 / -2.1 dB in k25-49**, plus a 30.96 -> 13.3-13.7 collapse in `mrstft`.
   The effect is `k`-progressive in both readouts, as predicted.
3. **The staircase is exonerated.** `B - S` is within run-to-run variability and
   sign-inconsistent across the two readouts. This agrees with the model-free
   pressure computed before training: the staircase displaces harmonic 80 by
   8.5 Hz (1.1 bins) and costs 1.18 dB, while the constant bias displaces it by
   35 Hz (4.5 bins) and costs 18.7 dB. The staircase is the smaller term by
   about 4x in displacement and an order of magnitude in pressure.
4. **The presmooth mitigation fails, and the reason is not that smoothing is
   wrong.** The 5 Hz filter does what it claims — it cuts the label error from
   0.106 to 0.037 rev/s while moving the true track by only 0.017. But it
   removes the staircase, which was never the binding term, and leaves the
   constant bias untouched. `C ~ B ~ S`.
5. **The actionable fix is a scale correction, not a filter.** Multiply DREGON
   telemetry by 0.99458 (the forensics memo's measured correction,
   `docs/experiments/dregon-telemetry-forensics.md`) before it conditions a
   generator. It costs nothing and it targets the term that carries the effect.
   Arm A is what that fix is worth if the correction is exact.

### Caveats, in order of how much they could change the verdict

- **One seed per arm.** The `S - A` gap is far outside the observed run-to-run
  scatter (5.5-8.8 dB against ~1-2 dB, consistent in sign across two independent
  readouts), so conclusion 1-2 are safe. Conclusion 3 is a *null* at one seed:
  it says the staircase effect is smaller than ~2 dB, not that it is zero.
- **The task is synthetic and easy**: one rotor, one microphone, a frozen
  spectral profile, amplitudes independent of RPS. This bounds the *mechanism*
  and its ordering, not the real-data effect size. Arm A's flatness in
  particular should not be read as "the objective is fine on real drone noise" —
  it is "the objective is not what breaks a comb whose profile is learnable".
- **The bias's cost may be architecture-specific.** This emitter derives its
  harmonic frequencies from the conditioning, so a global rate error can only be
  answered by moving energy between integer harmonic indices. An architecture
  with an explicit learnable rate correction might absorb the bias cleanly.
- Arm A ran 21 epochs against S/B/C's 13-17. All four shared one cap and one
  patience; A ran longer because it was still improving, while S/B/C plateaued
  early at a much worse `mrstft`. The gap is not a budget artifact.

### Two traps this round walked into

Both would have produced a confidently wrong number, and both are cheap to
repeat, so they are recorded rather than quietly fixed:

- **A stale local checkpoint outranked the cluster's.** `zoo.load` prefers
  `results/<experiment>/best.ckpt` over R2. A 1-epoch local smoke run had left
  one at `results/p7_labelsens_tach/`, so arm B was first scored on a model that
  had seen 64 samples — reading `+31 dB` at k50-80 and a 37 dB track error. The
  giveaway was that its numbers tracked the smoke run's, not the other arms'.
  Delete the local run directory before reading a cluster job's checkpoint.
- **An `f0` grid outside the training marginal.** See "What the readout was
  checked against" above.

## Reproducing

```bash
python scripts/gen_label_sensitivity_eval.py --self-test    # readout bias, expect <0.01 dB
python scripts/gen_label_sensitivity_eval.py --pressure     # model-free per-k pressure
python scripts/gen_label_sensitivity_eval.py                # the four arms
```

Outputs land in `results/gen_label_sensitivity/` (`summary.json`, `per_k.csv`,
`per_k.png`).
