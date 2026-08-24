---
experiment: hb_sal_multif0_nsr
training_config: conf/experiment/hb_sal_multif0_nsr.yaml
batch: docs/experiments/unified-baseline-eval.md
---

# `hb_sal_multif0_nsr`

## Motivation

The careful salience-map row of the unified baseline evaluation. `hb_sal_multif0`
retrains the June standard-grid model on the HB regime; this experiment
retrains the June *best* salience recipe — narrow-band HCQT input plus a
super-resolution output grid — on the same regime.

The June narrow+super-resolution arm
(`c7_multif0_salience_narrow_sr`, docs/experiments/salience-map-rps-tracking.md)
took `multif0_salience` from 6.30 to 4.03 RMSE on `DREGON-LM-V4/valid`. The
gain came from resolution: the coarse log grid put ~0.9 Hz between bins, 55% of
the validation frames held two rotors less than 1 Hz apart, and the
trajectories collapsed onto their mean. A 0.153 Hz/bin linear output grid
removes that failure mode.

Its 55-110 Hz spans do not carry over to the HB regime, for two reasons.

- The HB stream (`conf/online_mix/hb_silence_dload.yaml`) freq-scales every
  post-warm-up noise chunk by `alpha ~ U(0.7, 1.3)`, comb and labels together.
  Cruise labels span roughly 45-100 rev/s, thus the augmented cruise band runs
  from 31.5 to 130 rev/s. The June grid clips the top.
- The frozen validation split `dload:DREGON-LM-V4-michaels-valid-full` is
  full-envelope. Of its rotor-frames, 6.2% are zero and a further 4.9% are
  nonzero but below 55 rev/s. A rotor speed below the grid is not dropped, it
  is quantized onto the lowest bin, thus the June grid alone puts a 7.88 rev/s
  floor under the PIT RMSE — worse than the June headline number itself.

Widened grids (the full derivation is a comment block in
`conf/model/multif0_salience_nsr_hb.yaml`):

- Output: linear 20-130 Hz, 720 bins = 0.1530 Hz/bin, against June's 55-110 Hz,
  360 bins = 0.1532 Hz/bin. Same resolution, wider span. The clamp floor drops
  from 7.88 to 2.03 rev/s.
- Input HCQT: `fmin` 20 Hz, 3 octaves, over-sample 10, harmonics `[1,2,3,4]` —
  360 bins spanning 20-158.8 Hz. The super-resolution head clamps outside the
  input span, thus the input must contain the output span. 360 input bins is
  the standard-grid trunk width, thus the LateDeep cost does not move.

Measured ground-truth round trip (real labels of six frozen-valid clips ->
binary target on the grid -> Hungarian decode -> global PIT RMSE). This is the
floor a perfect model hits on each grid:

| Output grid | Round-trip PIT RMSE (rev/s) |
|---|---|
| June narrow-SR, linear 55-110 Hz, 360 bins | 7.24 |
| Standard grid, log 32.7 Hz / 6 octaves, 360 bins | 3.96 |
| This config, linear 20-130 Hz, 720 bins | 2.25 |

Frames below 20 rev/s still clamp onto the lowest bin. No finite grid covers a
ramp that runs down to zero, and the training target is deliberately left
alone; only the decode side was changed (below).

Data, epoch budget, validation split and monitor are `hb_sal_multif0`'s,
unchanged, thus the two rows isolate the grid.

Companion fix, same push: the Hungarian tracker used to carry the last speed
forward across a salience frame with no peak above the threshold, and to emit
NaN before any track started. Both now decode to 0 rev/s for every rotor —
the project-wide silence == zero-rotor-speed convention
(`models/multif0/utils.py`, tests in `tests/models/test_salience_rps.py`). The
HB regime feeds 16.7% zero-labeled chunks, thus this path carries real weight
here, unlike in June.

Train with `python train.py experiment=hb_sal_multif0_nsr`.

## Conclusion

Pending.
