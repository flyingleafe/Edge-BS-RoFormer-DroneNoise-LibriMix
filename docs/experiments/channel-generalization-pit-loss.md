# Channel-Generalization Failure and the PIT-Loss Fix

**Status:** done | **Dates:** 2026-06-08 | **Full report:** writing/reports/2026-06-08_channel-generalization-failure/ (run `make` for the PDF)

## Motivation

RPS (rotations-per-second) prediction from drone audio is a core building block for telemetry-free speech enhancement under harmonic noise. Prior RPS models (SimpleConv, SimpleConvV2) were trained and evaluated on DREGON-LM channel 0 only, under the implicit assumption that microphone position doesn't matter. This experiment tested that assumption directly by evaluating the same trained models on the same flight recordings through all 8 DREGON microphone channels.

## Results

Both models showed severe channel-dependent degradation: edge-microphone MSE was 3-10x higher than the training microphone (ch 0), overall R² was near zero (SimpleConv R²=0.07, SimpleConvV2 R²=-0.10), and PIT (permutation-invariant) re-evaluation recovered only 0.5-2.0% of the error (SimpleConv: 35.49→34.77 MSE; SimpleConvV2: 40.28→40.07), showing the failure was genuine misprediction, not motor-index swapping. Retraining both architectures on all 8 channels jointly (batch concatenation `(B,C,T)→(B·C,T)`, still single-channel input per prediction, still plain MSE loss) fixed SimpleConv (R²=0.57, uniform across channels) but not SimpleConvV2, which without PIT scored R²=-0.78 (MSE=61.39) — yet with PIT evaluation collapsed to MSE=3.30 (R²=0.94, a 94.6% error reduction). This proved the 8ch-trained SimpleConvV2 predicts the correct rotor speeds but assigns them to inconsistent rotor indices across channels, because which motor is loudest is a function of mic position, not an acoustically recoverable identity — making raw (non-PIT) rotor-index assignment a fundamentally underdefined task.

## Conclusion

PIT evaluation (and PIT training loss) is the correct objective for RPS prediction, since downstream harmonic-noise suppression only needs the set of rotor speeds, not per-rotor identity. Consistent with this, the current framework ships dedicated PIT loss configs (`conf/loss/pit_mse.yaml` and `masked_mse_plus_pit_rps_w{0p1,0p5,2}.yaml`), i.e. PIT-aware loss is available as a standard, first-class option for RPS training.
