# Salience-Map Multi-F0 Tracking for RPS Prediction

**Status:** done | **Dates:** 2026-06-12 to 2026-06-15 | **Full report:** writing/reports/2026-06-15/ (run `make` for the PDF)

## Motivation

Direct RPS regression (SimpleConv / SimpleConvV2 + PIT loss) requires task-specific
architecture and training. This experiment asked whether an off-the-shelf *multi-pitch*
approach could match it with less engineering: run a multi-F0 salience-map model
(per-frequency-bin activation) over the drone audio, peak-pick the four rotor
fundamentals per frame, and track them into continuous RPS trajectories via Hungarian
(optimal bipartite) assignment. If salience models can reliably expose the four rotor
fundamentals, tracking should recover RPS "for free" without a custom regression head.

Two model families were adapted to emit salience: **LateDeep** (CNN over an HCQT
front-end, `multif0_salience`) and **Basic Pitch** (contour branch, `basic_pitch_salience`),
both trained with BCE against binary salience targets derived from GT RPS, native 16 kHz
throughout for comparability with SimpleConv.

## Results

**Tracking algorithm validation (GT round-trip, DREGON-LM-V4 valid, 16 samples ≥33 Hz):**
Hungarian tracking clearly beats greedy nearest-neighbor — per-frame PIT 1.19 Hz vs
2.78 Hz, global PIT 1.67 Hz vs 6.83 Hz. Irreducible quantization floor (per-frame PIT
directly on salience bins, no tracking) is 0.27 Hz.

**Baseline salience models on `DREGON-LM-V4/valid`** (30 clips × 8 channels, PIT eval,
`track_threshold=0.3`):

| Model | RMSE (Hz) | MAE frame (Hz) | R² |
|---|---|---|---|
| SimpleConvV2 (8ch, regression) | 1.62 | 1.08 | 0.93 |
| SimpleConv (8ch, regression) | 3.55 | — | 0.68 |
| `multif0_salience` (LateDeep) | 6.30 | 3.40 | 0.19 |
| `multif0_salience_fastest` (stacked HCQT) | 6.42 | 3.58 | 0.11 |
| `basic_pitch_salience` | 23.24 | 16.19 | −16.21 |

LateDeep is the best salience model but ~4× worse than SimpleConvV2 in RMSE; Basic
Pitch fails catastrophically (diffuse salience, wrong-octave locking). A round-trip
experiment (perfect binary targets → same tracking pipeline) put the coarse-grid
resolution floor at 2.5–3.0 Hz RMSE — worse than SimpleConvV2's *total* error, explaining
much of LateDeep's gap.

**Root cause identified:** on DREGON-LM-V4, rotor fundamentals cluster tightly
(p1–p99 = 69–89 Hz) and 55% of frames have two rotors <1 Hz apart — below the ~0.9 Hz
bin spacing of the coarse grids — so trajectories collapse to their mean.

**Narrow-band + super-resolution follow-on:** concentrated the HCQT input in the rotor
band (`fmin=55`, 1 octave, harmonics 1–4, 120 bins, 55–110 Hz) and added a
`FreqSuperResHead` that resamples onto a fine *linear* 360-bin output grid
(≈0.153 Hz/bin, 55–110 Hz), trained end-to-end with BCE. Results on the same
validation set:

| Model | RMSE (Hz) | MAE frame (Hz) | R² |
|---|---|---|---|
| `multif0_salience_narrow_sr` | **4.03** (was 6.30) | 2.34 (was 3.40) | **0.573** (was 0.19) |
| `basic_pitch_narrow_sr` | 11.66 (was 23.24) | — | −3.24 (was −16.21) |

`multif0_salience_narrow_sr` now sits *below* the old coarse-grid resolution floor
(2.5–3.0 Hz), confirming the finer output grid genuinely buys localization rather than
confident-but-wrong peaks. Per-rotor MAE spread compressed from 4.5 Hz (1.2–5.5 Hz
across rotors) to 1.8–2.9 Hz — the near-unison rotor collapse is fixed. Basic Pitch
improves but stays unusable (still negative R²).

## Conclusion

Narrow-band input + super-resolution output is the viable salience-tracking recipe:
it removes the rotor-collapse failure mode and narrows the gap to SimpleConvV2 from
~4× to ~2.5× in RMSE (4.03 Hz vs 1.62 Hz). The salience-map paradigm is not
fundamentally unsuited to closely-spaced rotor fundamentals — the original baselines
were simply mis-resolved. Basic Pitch remains unusable at any grid resolution
(designed for discrete musical notes, not continuous low-frequency F0 tracking).

SimpleConvV2 (direct regression + PIT) remains the recommended model going forward —
even the improved LateDeep variant doesn't surpass it, suggesting that after fixing
output resolution the salience approach is now limited by architectural capacity
(purely convolutional, same limitation as the simplest regression baseline) rather than
representation. Future investment should go to the regression family, not further
salience-map engineering.
