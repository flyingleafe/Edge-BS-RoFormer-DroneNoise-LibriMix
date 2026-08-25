---
experiment: hb_ckla
training_config: conf/experiment/hb_ckla.yaml
batch: docs/experiments/unified-baseline-eval.md
---

# `hb_ckla`

## Motivation

The CKLA campaign (`docs/experiments/ckla.md`, closed 2026-07-28) ended with
the phase-only readout as its best head. The readout drops the raw
quadratures and feeds the mix layer `[|y|, arg(y_t·conj(y_{t−1}))]`, thus it
reads the tracked instantaneous frequency instead of the carrier. On the
fs_v2 family it was the best model on every accuracy axis: full-envelope
26.6, total PIT-MAE 3.10, zero-RPS 3.67, and cruise pools 2.79 (DREGON) /
1.29 (FLY124) — the best FLY124 number of the uniform regime and 2.2x better
than the matched transformer there.

Those numbers come from the fs_v2 regime, thus they do not sit in the same
table as the HB grid. The unified leaderboard needs the head scored in the
R2 honest regime, on the frozen validation split, under the one protocol.
This arm supplies that row. Its control is `hb_fkla` (vendored plain KLA,
no rotation by construction), and its matched HB trunk cell is
`hb_scv2_mag_nogate`.

## Setup

The HB recipe of `hb_scv2_mag` — data `e12_real_fullflight` with the R2
honest pool (`conf/online_mix/hb_silence_dload.yaml`), loss `pit_mse`,
metrics `rps`, 200 epochs, patience 20, batch 128, lr 1e-3, weight decay
1e-4, monitor mse, `samples_per_validation` 40000, validation on
`dload:DREGON-LM-V4-michaels-valid-full`.

Model `simple_conv_v2_ckla_phaseonly` — the exact config of the campaign
best `ckla_phaseonly_fs_v2`: registry name `simple_conv_v2_ckla_phaseonly`
(`readout="phase_only"` inside `src/models/ckla.py`), `p_init: 1.0` (the
live-gain fix; the KLA-paper default 0.01 collapses the Kalman gain), n_fft
2048, hop 512, 4 rotors. The head keeps its native `stft_mag_if` front-end
by construction, thus no front-end override is applied. Budget deviations
from the HB standard: none — batch 128 and
`samples_per_validation` 40000 are what the fs_v2 arm used.

`grad_clip` stays at the 5.0 default, as in `ckla_phaseonly_fs_v2`. The
phase_only readout can explode on `atan2` near a zero phasor: a NaN killed a
seed of `ckla_phaseonly_8s`, which is why that 8 s arm alone clips at 1.0.
The silence arm of the R2 pool makes near-zero phasors more frequent, thus
the risk is higher here than on fs_v2. The training loop drops a non-finite
loss batch rather than poisoning the weights (`src/training/loop.py`). If
the run NaNs or the dropped-batch warning becomes frequent, set
`grad_clip: 1.0` and record the change here.

No voicing gate exists on this head, thus the honest comparison cell is
`hb_scv2_mag_nogate`.

Train: `python train.py experiment=hb_ckla`.

## Conclusion

Pending.
