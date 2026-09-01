---
experiment: hft_r4
training_config: conf/experiment/hft_r4.yaml
batch: docs/experiments/harmonic-multipitch-ports.md
---

# `hft_r4`

## Motivation

ARM C — hFT-Transformer (ported) on the REAL R4 CURRICULUM: the comb-only
curriculum, stage 2, on the R2 honest base.

WHAT R4 IS (docs/experiments/unified-baseline-eval.md § "Regime-matched
reruns"). The regime taxonomy is R1 architecture search, R2 real-only honest,
R3 gen+comb curriculum, R4 comb-only curriculum, R5 mixed one-stage. Every
regime cell keeps ONE real component — the R2 honest pool with the warm-up
stage removed, `conf/online_mix/hb_m3s2_dload.yaml` — and differs only in the
synthetic ingredient and its schedule. R4's synthetic ingredient is the
ANALYTIC STATIC COMB, supplied as a warm start from a stage-1 checkpoint
trained on `conf/online_mix/m3abl_comb_s1_dload.yaml`.

So this row is `r4hb_gru` with the model swapped: same real stream, same
frozen real validation split (`dload:DREGON-LM-V4-michaels-valid-full`, the
protocol of docs/experiments/unified-baseline-eval.md), same renewed patience,
and a warm start from THIS model's own stage 1 — `hft_comb`, which trains on
the same comb policy `m3abl_comb_unigru128_s1` used.

The reference rows this one is read against: `hb_sal_multif0` (the LateDeep
salience family on real data) at flight MAE 4.01, and `r3hb_gru`, the best
regressor, at 2.79.

batch_size 8 with grad_accum_steps 2: the cross-attention map is linear in
the batch, and 8 frames is one chunk's eight microphone rows, so the
accumulation is what makes the step a batch of two scenes rather than one.

Data group is `m3cur_s2` for its real validation split; only the training
stream is overridden, exactly as `r4hb_gru` does it.

Full batch context: [Harmonic multi-pitch architectures ported to the linear STFT](../../docs/experiments/harmonic-multipitch-ports.md).

## Setup

Hydra wiring — data `m3cur_s2` · model `hft_rps` · loss `salience_bce_r150` · metrics `salience_bce_r150`. Train with `python train.py experiment=hft_r4`.

## Conclusion

This arm's outcome is recorded in the batch write-up: [Harmonic multi-pitch architectures ported to the linear STFT](../../docs/experiments/harmonic-multipitch-ports.md).
