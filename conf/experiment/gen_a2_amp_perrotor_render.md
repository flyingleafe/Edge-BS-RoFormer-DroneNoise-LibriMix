---
experiment: gen_a2_amp_perrotor_render
training_config: conf/experiment/gen_a2_amp_perrotor_render.yaml
batch: docs/experiments/amplitude-target-training.md
---

# `gen_a2_amp_perrotor_render`

## Motivation

Rendering twin of `gen_a2_amp_perrotor` — identical model parameters, ordinary audio codec, so `scripts/eval_gen_comb_real.py` can render an amplitude-trained checkpoint. Never trained; `--checkpoint` always points at a `gen_a2_amp_perrotor` file.

## Conclusion

Not a training experiment — a render-only twin. No conclusion applies.
