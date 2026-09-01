---
experiment: hf0_r4
training_config: conf/experiment/hf0_r4.yaml
batch: docs/experiments/harmonic-multipitch-ports.md
---

# `hf0_r4`

## Motivation

ARM C, faithful R4 — conf/experiment/hf0_real.yaml plus the warm start that
makes it the R4 regime cell rather than a plain real-data row.

R4 in docs/experiments/unified-baseline-eval.md's taxonomy is "comb-only
curriculum, stage 2": pre-train on the analytic static comb alone, then
fine-tune on the honest real pool conf/online_mix/hb_m3s2_dload.yaml. The
regression cells (r4hb_scv2 / r4hb_tr / r4hb_gru) warm-start from
m3abl_comb_<trunk>_s1; those checkpoints are simple_conv_v2-family trunks and
cannot be loaded into a salience model, so the salience stage 1 has to be
built here — and it is exactly arm A (conf/experiment/hf0_comb.yaml), which
trains on the same conf/online_mix/m3abl_comb_s1_dload.yaml policy.

THIS ROW THEREFORE DEPENDS ON ARM A: submit it only after hf0_comb has
written a best.ckpt to the artifact store.

Full batch context: [Harmonic multi-pitch architectures ported to the linear STFT](../../docs/experiments/harmonic-multipitch-ports.md).

## Setup

Hydra wiring — data `m3cur_s2` · model `harmof0_rps` · loss `salience_bce_r150` · metrics `salience_bce_r150`. Train with `python train.py experiment=hf0_r4`.

## Conclusion

This arm's outcome is recorded in the batch write-up: [Harmonic multi-pitch architectures ported to the linear STFT](../../docs/experiments/harmonic-multipitch-ports.md).
