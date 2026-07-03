# Diffusion-Buffer (BBED) speech enhancement

**Status:** config-complete, historically eval-only | **Dates:** 2026-03 | **Reference:** `docs/diffusion-buffer-paper.md` (paper transcription), `docs/diffusion-prompt.md` (implementation notes) | **Commands:** REPLICATION.md § A2

## Motivation

Diffusion / score-based generative models are a strong alternative family for
speech enhancement, particularly at low SNR where discriminative masks
struggle. This batch ports the **Diffusion-Buffer** model with the **BBED** SDE
(from `sp-uhh/Diffusion-Buffer` upstream) into the framework as an additional
DN-LM comparison point alongside the Paper-1 baselines
([paper1-edge-bs-roformer-dn-lm.md](paper1-edge-bs-roformer-dn-lm.md)).

## Experiments

- `a2_diffusion_buffer_bbed` — `DiffusionBufferModel`, BBED SDE, paper-exact
  hyperparameters (n_fft/hop = 510/256 — differs from A1's 2048/512;
  `buffer_size = 20`, SDE `bbed`, c = 0.08, k = 2.6).

## Results

In this repository the model was treated as a **pretrained-checkpoint import**
(the former `replicate_paper.py` listed it under `EVAL_ONLY_MODELS`), not a
from-scratch training target — no training run was defined for it here. A
chunking mismatch caveat applies: the buffer/frame-hop assumptions (K = 128
frames, hop = 256) differ from the rest of the A1 pipeline; see REPLICATION.md
§ A2. No framework-native training/eval numbers were produced.

## Conclusion

Kept as a config-complete, interpretable generative baseline. Training it from
scratch in-framework and resolving the chunking mismatch is deferred — the
diffusion / flow-matching direction is explicitly parked in `GOALS.md`
("~4–6 weeks for a baseline; no infra reuse") pending the mid-point review.
