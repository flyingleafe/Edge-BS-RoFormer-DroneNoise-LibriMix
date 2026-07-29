---
experiment: ckla_refiner
training_config: conf/experiment/ckla_refiner.yaml
batch: docs/experiments/rps-trajectory-refinement.md
---

# `ckla_refiner`

## Motivation

Classical refinement (VK refine, stage D) cannot pull a track with
0.5–2 rev/s error toward the truth on real drone audio, yet the comb
information is demonstrably present at high precision: GT-initialised
VK-refine locks to 0.028 MAE. The information is there — the *pulling
mechanism* is what fails. This arm trains a LEARNED puller: a conditional
refiner that takes (audio, corrupted RPS track) and outputs the corrected
track. Training pairs are (audio, corrupt(GT)) → GT with a synthetic
corruption sampler (smooth OU noise σ ∈ U(0.1, 1.5) with 0.5–2 s correlation
time, constant offsets U(−2.5, 2.5) at p=0.7, and p=0.15 pair-level
swap/twin-capture events — `data_processing/rps_corruption.py`); at inference
the conditioning is a real coarse track (blind-VK Viterbi at ~0.7 err, or a
neural predictor at ~1–2.5 err).

Conditioning also fixes output-rotor identity: output row i corresponds to
conditioning row i (the model predicts a bounded residual,
cond + 3·tanh(·)), so the loss is plain non-PIT MSE against the GT aligned
to the conditioning order (`conf/loss/mse_cond.yaml`).

## Setup

Model `simple_conv_v2_ckla_phaseonly_cond`: the `ckla_phaseonly_fs_v2`
backbone (p_init 1.0, rotation on, phase_only readout) with the conditioning
track MLP-embedded (32 ch) and concatenated to the pooled trunk features
before the temporal head; head predicts a residual bounded by
`max_delta = 3` rev/s via tanh. Data: the fs_v2 stream WITHOUT the AVQ
source (`conf/online_mix/e12_fullflight_freqscale_v2_dload.yaml`) through
the corruption seam (`conf/data/ckla_refiner.yaml`; train corruption seeded
per chunk id × channel, valid corruption FIXED per sample). Training knobs
as `ckla_phaseonly_avq`: batch 128 frames, spv 40000, grad_clip 1.0,
AdamW 1e-3 / wd 1e-4, monitor val PIT-mse.

Train: `python train.py experiment=ckla_refiner`.
Eval on the 37-clip protocol: `python scripts/rps_refiner_eval.py
--ckpt <best.ckpt> --coarse neural:e12_transformer_best` (or
`--coarse npz:<vk_blind_sweep npz>`; `--rounds N` for iterative refinement).

## Conclusion

_Pending run._
