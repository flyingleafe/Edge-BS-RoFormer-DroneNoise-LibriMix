# Limits of solving synthetic data

Campaign opened 2026-08-28. The question: how hard is each synthetic family to
solve, and what sets that difficulty. The campaign's premise on opening was
that the analytic static comb is the simplest corner of the stochastic comb
family, reached by setting the family's variances to zero, so that a curriculum
could walk from one to the other by raising those variances.

**That premise is false.** The two families are disjoint in comb contrast, no
exposed parameter closes the gap, and the knobs named "variance" are not the
difficulty axis. What follows is the measurement.

## The observable

Comb contrast is measured as the peak-to-bulk ratio of a spectrum: the 95th
percentile of bin levels minus the median, in dB, inside an octave band. It
assumes nothing about where the lines are, which matters because every
line-position-based estimator is contaminated by the neighbouring rotors' combs.

Two reference points anchor the scale:

- **Filtered white noise reads 5.6 to 6.3 dB.** Bin power is exponential, so
  p95 minus p50 is 10 log10(ln 20 / ln 2) = 6.4 dB with no structure present
  at all. Anything near 6 dB carries no comb.
- **The analytic static comb reads 15 to 25 dB** across every band.

## The stochastic family sits at the noise floor

Realized audio, 8 s windows, cruise frames, by octave band (Hz):

| family | 100 | 250 | 500 | 1k | 2k | 4k |
|---|---|---|---|---|---|---|
| `static_comb` | 25.2 | 14.8 | 17.2 | 16.2 | 20.4 | 18.9 |
| stochastic, full ranges | 7.2 | 7.3 | 8.2 | 7.9 | 7.6 | 7.5 |
| + zero linewidth | 6.7 | 6.4 | 6.7 | 6.8 | 7.2 | 7.2 |
| + floor buried 40 dB | 7.1 | 7.4 | 8.2 | 8.1 | 8.4 | 8.5 |
| + coherent line rendering | 7.9 | 7.9 | 8.9 | 9.1 | 9.7 | 9.7 |
| + bin-integrated Lorentzians | 7.1 | 6.9 | 7.1 | 7.2 | 7.3 | 7.6 |

Every stochastic variant lands within about 2 dB of structureless noise. The
static comb is 8 to 15 dB above the best of them.

## The flatness is in the target spectrum, not the rendering

The model spectrum that the renderer is asked to realize was read directly and
compared against the audio it produced, same parameters:

| | 250 | 500 | 1k | 2k | 4k |
|---|---|---|---|---|---|
| white noise | 5.6 | 6.0 | 6.2 | 6.2 | 6.3 |
| model PSD | 10.6 | 6.4 | 7.5 | 4.5 | 5.0 |
| realized audio | 10.9 | 9.1 | 9.0 | 7.6 | 9.7 |

Above 1 kHz the noiseless target is flatter than white noise. The comb is not
destroyed by the rendering, because it is not in the model spectrum to start
with. The realized audio reads slightly higher than its own target only because
the realization adds its exponential spread on top.

## Why the floor knob does nothing

Band occupancy — the fraction of bins carrying line support — is 56 to 88% for
a four-rotor 80-harmonic comb at 80 rev/s, even with narrow lines. **The comb's
own lines tile the band, so the median bin is a line, not floor.** Contrast
therefore measures the spread of the harmonic amplitude profile, and the
broadband floor never enters it. Burying the floor by 40 dB moves band contrast
by 0.0 dB at every linewidth:

| gamma_slope \ floor | 0 | −6 | −12 | −24 | −40 dB |
|---|---|---|---|---|---|
| 0.00 | 8.3 | 8.5 | 8.5 | 8.5 | 8.5 |
| 0.20 | 7.1 | 7.2 | 7.2 | 7.2 | 7.2 |
| 0.80 | 4.8 | 4.8 | 4.8 | 4.8 | 4.8 |

## The variance knobs are not the difficulty axis

Zeroing `harm_gp_std_db`, `floor_gp_std_db`, `floor_tilt_gp_std`,
`harm_coherence`, `harm_jitter_db`, `rotor_similarity` and `harm_dropout_p`
together moves per-frame line flicker from 3.41 to 3.22 dB and line visibility
from 3.59 to 2.97 dB. Both are inside the measurement's own spread. A curriculum
that raises these knobs from zero to their full ranges is very nearly a no-op.

## `calibrate_floor` holds contrast constant by construction

`calibrate_floor` places the floor `floor_rel_db` under the **peak PSD** of the
median line, and that peak is `profile_db − 10 log10(pi · gamma)`. Narrowing a
line raises its peak, so the solver raises the floor by the same amount. Line
sharpness and floor depth are coupled, which is why zeroing the linewidth alone
made realized contrast slightly **worse** (7.2 against 7.5 dB at 4 kHz).

## A separate defect: point-sampled Lorentzians

`build_psd` evaluates each Lorentzian at the bin centres. A line is 0.5 to 4 Hz
wide against a 7.81 Hz bin, and `k · rps` puts it at an arbitrary sub-bin
position. For one isolated line at n_fft 2048:

| gamma (Hz) | line on a grid point | line between grid points |
|---|---|---|
| 0.05 | 53.4 dB | 14.0 dB |
| 0.50 | 33.4 dB | 13.9 dB |
| 2.00 | 21.4 dB | 13.0 dB |

Below a bin the width stops mattering: 0.05 and 0.5 Hz read the same. The
fraction of a line's power in its peak bin swings from 1.000 to 0.406 with
sub-bin position, which is 3.9 dB of amplitude wobble that tracks the rotor
speed and belongs to no physical process.

The fix is exact and closed-form — the Lorentzian's antiderivative is an
arctangent — and is available as `line_bin_integrate`, default off so every
existing stream and checkpoint is unchanged. **It does not move band contrast**
(8.4 to 8.1 dB), because density and not sharpness is what caps the band. It is
worth having for the spurious wobble it removes, not as a fix for contrast.

## What does move contrast

Two axes, both monotone and both physically meaningful. Band-mean model-PSD
contrast:

| `rolloff_p` | slope 0.0 | slope 0.4 |
|---|---|---|
| 0.4 | 7.7 | 5.5 |
| 1.0 | 8.4 | 5.9 |
| 1.9 | 8.4 | 6.2 |
| 3.0 | 8.8 | 7.0 |
| 5.0 | 10.2 | 8.7 |
| 8.0 | 12.8 | 11.5 |

`rolloff_p` sets how fast the comb decays with harmonic index, and therefore how
many harmonics are strong enough to tile the band. `gamma_slope_hz` sets how
fast a line broadens with its own order. Together they span 5.5 to 12.8 dB.

## Consequences

1. The static comb cannot be reached from the stochastic family. Its minimum
   (about 15 dB) is above the family's maximum (about 12.8 dB).
2. Every stochastic arm in the transfer campaign trained on spectra carrying
   roughly 1 to 2 dB of comb structure over structureless noise. That is a
   candidate explanation for stochastic-only training plateauing at 3.70 PIT-MAE
   against comb-only's 1.29, and for capacity scaling not moving it — the
   information is largely absent from the target, not unread by the model.
3. A curriculum is still constructible, on `rolloff_p` and `gamma_slope_hz`
   rather than on the variances, spanning the family's own 5.5 to 12.8 dB.

## The comb-only runs never converged

Training-loss histories for the three comb-only arms, read from W&B. The
percent-of-descent measure is useless here because the first epochs fall from
about 2650 to about 4, so every arm "reaches 99% of its descent" by epoch 3.
The informative number is the slope over the last quarter of each run:

| run | epochs | final train loss | last-quarter slope |
|---|---|---|---|
| `m3abl_comb_scv2_s1` | 37 | 4.37 | −0.079 per epoch |
| `m3abl_comb_unigru128_s1` | 59 | 3.06 | −0.041 per epoch |
| `m3abl_comb_transformer_s1` | 21 | 8.35 | −0.136 per epoch |

All three were still descending when they stopped. The transformer — the arm on
which the campaign recorded "comb-only stage 1 kills the transformer" — had the
steepest remaining descent and the fewest epochs of the three. That verdict
measured the stopping rule.

This is a lower bound on the ladder's rung length, not a calibration: rung
length must be longer than 59 epochs, and the comb-floor runs (`comb_floor_base`
/ `_wide` / `_deep`, patience 200, epochs 2000, validated on the comb itself)
are what will set it.
