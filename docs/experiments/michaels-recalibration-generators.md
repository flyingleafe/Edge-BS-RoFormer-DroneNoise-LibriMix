# Michael's recalibration — noise-generator retrains

**Premise.** Michael's flight-controller telemetry was measured to be wrong and
was corrected on 2026-07-31 (branch `michaels-label-calibration`): the
audio-telemetry lag is a clock **dilation**, not a constant offset (residual RMS
2.9 / 4.5 ms against 12.0 / 16.2 ms for the old constant-lag model), and the
logged speeds read ~0.6 rev/s **low** at cruise (multiplicative correction
x1.00698 for FLY124, x1.00706 for FLY125). Every generator trained before that
date learned to render audio against labels that were wrong in both time and
value.

**Why a generator should care more than a predictor.** The rev/s error is
multiplicative, so it grows with harmonic order: +0.56 rev/s at 80 rev/s
displaces the 40th comb line by ~22 Hz. A generator conditioned on such labels
cannot place its high harmonics correctly and can only reduce the loss by
**broadening** them — which is exactly the mid/high-harmonic washout the E6
linewidth round attributed to RPS jitter. The residual timing error acts the
same way through `d(rps)/dt`, biting hardest in transitions.

**Falsifiable prediction.** If part of the learned per-drone jitter sigma was
absorbing label error rather than physical linewidth, the **Michael's** sigma
should fall after the fix while DREGON's (unaffected by the calibration) stays
put. The pre-fix pair was nearly equal — dregon 0.629, michaels 0.607.

## Arms

| Arm | Architecture | Labels |
|---|---|---|
| `gen_v1_corrected` | E6 per-drone winner, corrected geometry | stale (pin d7425a5cef7b) |
| `gen_v2_perrotor` | v1 + per-rotor sub-embeddings | stale |
| `gen_v1_recal` | identical to `gen_v1_corrected` | **recalibrated** (pin fdef818432e9) |
| `gen_v2_recal` | identical to `gen_v2_perrotor` | **recalibrated** |

`v3_wind` is not re-run: it is the worst arm and its premise (per-microphone
incoherence) is unobservable in the current training path, which is
single-microphone — see the caveat below.

## Protocol

Data, split and schedule are unchanged from the corrected-geometry round;
`conf/data/noise_rps_dregon_michaels_swapped_stream.yaml` pins no explicit
version, so `frames:michaels-frames` picks the corrected labels up from
`dload.lock`. Scoring is `scripts/eval_noise_gen_variants.py
--min-flight-rps 45` (free flight only — the `val_at_start` split holds out
takeoff/idle, and scoring on that idle-dominated remainder inverts rankings).

The eval now reports **per drone** as well as pooled. That split is what makes
the claim testable: DREGON is untouched by the calibration and therefore acts as
the control arm, so an improvement confined to Michael's is attributable to the
labels rather than to run-to-run variance.

The honest comparison is **each model against the labels it was trained with**
(old model on the old valid, new model on the new valid), because the validation
set moved too. Scoring the old checkpoint on the new valid is reported as a
separate diagnostic: it isolates the cost of the label change alone, and is not
the headline number.

## Caveat inherited from the framework

All generator training in the Hydra framework is **single-microphone**:
`NoiseGenFrameDataset.__getitem__` takes channel 0 and row 0 of the geometry,
and the constructor rejects any other `channel_policy`. The historical trainer
rendered all 8 microphones jointly. So these numbers, like v1/v2/v3 before them,
score the propagation path at M = 1. Restoring multi-observer training is a
separate piece of work (`NoiseRPSDataset` must report which channel it drew).

## Conclusion

_Pending runs._
