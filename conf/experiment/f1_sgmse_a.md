---
experiment: f1_sgmse_a
training_config: conf/experiment/f1_sgmse_a.yaml
batch: docs/experiments/f1-se-blind-baselines.md
---

# `f1_sgmse_a`

## Motivation

SGMSE+ (Richter et al., IEEE TASLP 2023 — score-based diffusion, OUVE SDE + full
NCSN++ backbone ~65M, trained from scratch at native 16 kHz), Pass A (drone noises only) — the drone-focused floor of the F1
blind speech-enhancement baseline program — the generative-diffusion rung. One
rung of the 5-arch x 2-pass floor comparison, and the only generative (vs.
discriminative/masking) baseline in the set.

Full batch context: [F1 — SE blind baselines](../../docs/experiments/f1-se-blind-baselines.md).

## Setup

Hydra wiring — data `se_baselines_a` (online-mix SE stream + fixed SE-valid-drone) ·
model `f1_sgmse` · loss `masked_mse` · metrics `separation_basic`
(eval with `metrics=separation_full` for PESQ/eSTOI). Train with the bespoke score-matching loop
`python scripts/train_sgmse.py --experiment f1_sgmse_a --wandb` (train.py's discriminative loop cannot host score-based diffusion), evaluate with
`python eval.py experiment=f1_sgmse_a metrics=separation_full`.

Integration mirrors `a2_diffusion_buffer_bbed` exactly (same
`forward(mix, target=None)` DSM-loss / PC-sampler contract, same
`SpeechEnhancementCodec` + `masked_mse` wiring): `forward(mix, target)`
returns the sigma^2-weighted denoising-score-matching scalar loss (unit-tested
directly); through the training loop the codec calls `model(mix)` which runs
the predictor-corrector reverse-SDE sampler and `masked_mse` scores the enhanced
waveform against clean — no training-loop/task/codec changes. Sampling is slow
(~30 min/utterance at N=30, PC with ALD corrector, snr=0.5).

## Conclusion

Reported comparatively in the batch write-up — see
[F1 — SE blind baselines](../../docs/experiments/f1-se-blind-baselines.md).
