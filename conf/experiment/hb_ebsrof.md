---
experiment: hb_ebsrof
training_config: conf/experiment/hb_ebsrof.yaml
batch: docs/experiments/unified-baseline-eval.md
---

# `hb_ebsrof`

## Motivation

The Edge-BS-RoFormer paper (Liu et al., Drones 2025) claims that its rotary
time and frequency embeddings help the axial transformers track harmonic
lines. RPS prediction is the direct test of that claim, because the target
IS the harmonic-line trajectory. The unified leaderboard has no row for this
trunk, thus the R2 honest regime gets one.

The July 2026 attempt failed. `ebsrof_rps_e12` and `ebsrof_rps_freqscale`
never learned: `docs/experiments/ckla.md` § Conclusion records "ebsrof
debugging (failed to learn: val ~1150 flat)", and both `.md` files still say
_Pending run._ The debug arms found three facts: the trunk moved the loss at
lr 3e-4 (6000 to 3500 inside epoch 0), it OOM'd at validation on full-length
clips even at batch 32, and lr 1e-4 limped to val ~905. Nobody found the
mechanism, thus the arm was parked.

This run is instrumented. It is not a plain retry: the post-hoc recipe below
gives a verdict on WHY the trunk is flat, and `hb_ebsrof_lowlr` is the first
diagnostic arm if the main arm is flat again.

## Setup

The HB recipe of `hb_scv2_mag` — data `e12_real_fullflight` with the R2
honest pool (`conf/online_mix/hb_silence_dload.yaml`: the fs_v2 real pool, a
zero-labeled silence arm, an SNR reference floor), loss `pit_mse`, metrics
`rps`, 200 epochs, patience 20, lr 1e-3, weight decay 1e-4, monitor mse,
`samples_per_validation` 40000. Validation is the fixed full-envelope real
split `dload:DREGON-LM-V4-michaels-valid-full`.

Two fields differ from `hb_scv2_mag`, both kept from the July configs:

- Model `edge_bs_rof_rps` (`src/models/edge_bs_rof/rps.py`, 559k parameters,
  F1 `a1_edge_bs_rof_fa_rope48` configuration, internal complex STFT, band
  attention pool plus a linear RPS head, `flash_attn: false`).
- Batch 64 instead of 128. The July comment gives the cause: axial attention
  over full-length validation clips OOM'd a V100-16GB at 128. The debug runs
  saw an OOM at batch 32 too, thus a V100 may still need 32 here. Change the
  batch only after a real OOM, and record the change.

No voicing gate exists on this trunk, thus the matched HB comparison cell is
`hb_scv2_mag_nogate`, not `hb_scv2_mag`.

Train: `python train.py experiment=hb_ebsrof`.

## Diagnostics (post-hoc recipe)

The training loop logs no gradient norms and has no flag to do it
(`src/training/loop.py` clips with `grad_clip` but never records the norm;
`wandb.log` carries `epoch`, `train/loss`, `val/loss`, `lr` and the metrics).
No infrastructure was built for this run. Do the five steps below instead:

1. W&B curve `train/loss` inside epoch 0. A trunk that moves 6000 to 3500 is
   learning, thus a flat `val/loss` after that is a validation or an
   early-stop problem, not a dead trunk.
2. W&B curves `train/loss` against `val/loss` for the first 10 epochs. Both
   flat at ~1150 = no learning. Train falls and val stays flat = the head
   memorizes the pool and the split is out of reach.
3. W&B curve `lr`. `adamw_plateau` decays lr on the monitor; a flat run that
   also decays lr proves that the monitor never improved, thus patience 20
   is the real stop cause.
4. `python scripts/probe_ckpt.py --ckpt zoo:hb_ebsrof --report spectra` — a
   collapsed singular-value spectrum on the band-pool or head matrices shows
   a dead layer, and a spread one shows a live trunk with a bad objective.
5. `python scripts/probe_ckpt.py --ckpt zoo:hb_ebsrof --report params
   --filter head` — a head whose weights stay at initialization value gives
   the constant-output failure, which is what val ~1150 flat looks like.

If steps 1 to 5 say "no learning at any depth", submit `hb_ebsrof_lowlr`.

## Conclusion

Pending.
