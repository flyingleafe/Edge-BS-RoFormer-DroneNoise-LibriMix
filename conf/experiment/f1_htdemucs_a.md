---
experiment: f1_htdemucs_a
training_config: conf/experiment/f1_htdemucs_a.yaml
batch: docs/experiments/f1-se-blind-baselines.md
---

# `f1_htdemucs_a`

## Motivation

HTDemucs v4 (hybrid time+spectrogram transformer, ~42M), Pass A (drone noises
only) of the F1 blind speech-enhancement baseline program. Unlike the other
F1 arms, this one is **fine-tuned from the official Meta music-separation
checkpoint**, not trained from scratch — it tests what a large pretrained
separation prior buys on our data.

Full batch context: [F1 — SE blind baselines](../../docs/experiments/f1-se-blind-baselines.md).

## Design decisions (pretrained adaptation)

Implementation: `src/models/htdemucs_ft.py` (wrapper around the in-repo
`models/demucs4ht.py` core, which is an exact architectural copy of
`demucs.htdemucs.HTDemucs`).

1. **Pretrained checkpoint.** Official Meta `htdemucs` snapshot
   `955717e8-8726e21a.th` (sha256 prefix `8726e21a...` matches the release
   name; 44.1 kHz, stereo, 4 stems drums/bass/other/vocals), from
   `https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/`. Staged
   durably at `r2://ml-data/artifacts/htdemucs_pretrained/htdemucs_955717e8-8726e21a.th`
   — cluster jobs fetch it through
   `training.artifacts.resolve_checkpoint_uri` (R2 creds from `.env`), no
   internet needed. The stored constructor kwargs are asserted against the
   hardcoded `OFFICIAL_HTDEMUCS_KWARGS` at load, and the state dict loads
   with `strict=True`.
2. **Sample rate — resample, not retune.** Data is 16 kHz; the checkpoint's
   STFT framing and conv weights assume 44.1 kHz. The wrapper resamples
   16 k -> 44.1 k at the model input and 44.1 k -> 16 k at the output
   (`julius.ResampleFrac`, differentiable), which preserves the pretrained
   feature alignment — the memory-endorsed route when weights are pretrained.
   All losses/metrics are computed at 16 kHz after inverse resampling, so
   scores are directly comparable to the other F1 arms.
3. **Channels — duplicate mono to stereo, average the outputs.** The input
   convs expect 2 channels; duplication keeps both pretrained channel
   filters on-distribution and modifies zero weights, while an input-conv
   adaptation would discard pretrained parameters and cascade into the
   source-grouped output layout.
4. **Head 4 stems -> 2 sources ({speech, noise}).** HTDemucs has no
   per-source embeddings; the only source-dependent tensors are the final
   transposed convolutions. Remapped tensors (out-channel axis is grouped
   in per-source blocks):
   - `decoder.3.conv_tr.weight` `(48, 16, 8, 1)` -> `(48, 8, 8, 1)`: blocks
     of 4 (2 ch x 2 complex-as-channels); speech <- vocals block `[12:16]`,
     noise <- other block `[8:12]`
   - `decoder.3.conv_tr.bias` `(16,)` -> `(8,)`: same blocks
   - `tdecoder.3.conv_tr.weight` `(48, 8, 8)` -> `(48, 4, 8)`: blocks of 2;
     speech <- vocals `[6:8]`, noise <- other `[4:6]`
   - `tdecoder.3.conv_tr.bias` `(8,)` -> `(4,)`: same blocks

   Speech <- VOCALS is the semantically closest stem; noise <- OTHER (the
   broadband instrumental residual). Only the speech output is returned, so
   the noise block receives no gradient (verified in the smoke test) — it
   is inert ballast, kept only to preserve the pretrained output layout.
5. **Fine-tune regime.** fp32 (`amp: false` — complex-as-channels STFT path,
   the known ComplexHalf autocast gap; no internal fp32 boundary exists in
   this model). Single LR `1e-4` (10x below the from-scratch F1 default) for
   trunk and head — the loop's optimizer is one param group, and the
   warm-started head does not need a separate LR. `use_train_segment=False`
   so 1 s train chunks and 2 s valid clips both pass. `resume: true` chains
   wall-limited segments; epochs/patience mirror the other F1 pairs.

Smoke-tested on CPU before submission: strict checkpoint load with the
documented remap only, forward `(2,1,16000) -> (2,16000)`, finite loss,
gradients flow to trunk and speech head, variable-length (2 s) eval passes.

## Setup

Hydra wiring — data `se_baselines_a` (online-mix SE stream + fixed
SE-valid-drone) · model `f1_htdemucs` · loss `si_sdr_mrstft` · metrics
`separation_basic` (eval with `metrics=separation_full` for PESQ/eSTOI).
Train with `python train.py experiment=f1_htdemucs_a`, evaluate with
`python eval.py experiment=f1_htdemucs_a metrics=separation_full`.

## Conclusion

Reported comparatively in the batch write-up — see
[F1 — SE blind baselines](../../docs/experiments/f1-se-blind-baselines.md).
