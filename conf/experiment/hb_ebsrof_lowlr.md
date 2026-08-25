---
experiment: hb_ebsrof_lowlr
training_config: conf/experiment/hb_ebsrof_lowlr.yaml
batch: docs/experiments/unified-baseline-eval.md
---

# `hb_ebsrof_lowlr`

## Motivation

First diagnostic arm of `hb_ebsrof`. **Submit this only if the main arm is
flat again** — that is, if `hb_ebsrof` reproduces the July failure (val
~1150 flat, `docs/experiments/ckla.md` § Conclusion) and the post-hoc recipe
in `conf/experiment/hb_ebsrof.md` says "no learning at any depth". Do not
run the two arms together; a second flat run at the same lr teaches nothing.

The learning rate is the one lever with July evidence behind it. The debug
runs found that the trunk moved the loss at lr 3e-4 (6000 to 3500 inside
epoch 0) and that lr 1e-4 limped to val ~905, while the family-standard
lr 1e-3 gave the flat curve. The hypothesis is thus a step size too large
for the axial attention stack: the trunk diverges into a constant-output
state in the first epochs, and the plateau scheduler then decays lr around
an already-dead model.

If lr 1e-4 learns and lr 1e-3 does not, the July failure is an optimization
failure, and the trunk earns a place on the leaderboard. If lr 1e-4 is flat
too, the failure is structural (band-pool head, input scaling, or validation
length), the two-arm evidence closes the lr hypothesis, and the arm must be
retired rather than swept further.

## Setup

`hb_ebsrof` with one field changed: `optim.lr` 1e-4 instead of 1e-3. Every
other field is identical — model `edge_bs_rof_rps`, the R2 honest pool
`conf/online_mix/hb_silence_dload.yaml`, batch 64, 200 epochs, patience 20,
weight decay 1e-4, monitor mse, `samples_per_validation` 40000, and the
fixed full-envelope real validation split.

Apply the same five-step post-hoc recipe as `hb_ebsrof` (that doc, §
Diagnostics). Read the two runs side by side in W&B: the `train/loss` curves
of epoch 0 give the fastest verdict.

Train: `python train.py experiment=hb_ebsrof_lowlr`.

## Conclusion

Pending.
