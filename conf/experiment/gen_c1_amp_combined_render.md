---
experiment: gen_c1_amp_combined_render
training_config: conf/experiment/gen_c1_amp_combined_render.yaml
batch: docs/experiments/amplitude-target-training.md
---

# `gen_c1_amp_combined_render`

## Motivation

Rendering twin of `gen_c1_amp_combined` — identical model parameters, ordinary audio codec, so `scripts/eval_gen_comb_real.py` can render an amplitude-trained checkpoint. Never trained; `--checkpoint` always points at a `gen_c1_amp_combined` file.

## Conclusion

Not a training experiment — a render-only twin. No conclusion applies.
