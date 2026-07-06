---
experiment: a2_diffusion_buffer_bbed
training_config: conf/experiment/a2_diffusion_buffer_bbed.yaml
batch: docs/experiments/diffusion-buffer-se.md
---

# `a2_diffusion_buffer_bbed`

## Motivation

Adds a diffusion / score-based generative speech-enhancement baseline to the DN-LM comparison set.

Full batch context: [Diffusion-Buffer (BBED) speech enhancement](../../docs/experiments/diffusion-buffer-se.md).

## Setup

Diffusion-Buffer BBED SDE model on DN-LM; historically eval-only in this repo (pretrained checkpoint expected) — see REPLICATION.md § A2 for the K=128-frame/hop=256 chunking mismatch caveat.

Hydra wiring — data `dn_lm` · model `a2_diffusion_buffer_bbed` · loss `masked_mse` · metrics `separation_basic`. Train with `python train.py experiment=a2_diffusion_buffer_bbed`,
evaluate with `python eval.py experiment=a2_diffusion_buffer_bbed`.

## Conclusion

This run's outcome is reported comparatively in the batch write-up — see [Diffusion-Buffer (BBED) speech enhancement](../../docs/experiments/diffusion-buffer-se.md).
