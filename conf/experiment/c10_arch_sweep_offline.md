---
experiment: c10_arch_sweep_offline
training_config: conf/experiment/c10_arch_sweep_offline.yaml
batch: docs/experiments/simpleconv-rps-architecture-search.md
---

# `c10_arch_sweep_offline`

## Motivation

Searches SimpleConv encoder / temporal-head architectures for direct RPS prediction from drone audio, offline and under online mixing.

Full batch context: [SimpleConv RPS architecture search](../../docs/experiments/simpleconv-rps-architecture-search.md).

## Setup

C10 — 26-variant SimpleConv architecture sweep, OFFLINE/fixed-train arm (autoresearch session 20260617-012233). `simple_conv_v2` is the offline winner (PIT MSE 7.89, R² 0.82). Override `model` for the other 25 variants (all have a conf/model/<key>.yaml, same n_fft/hop/num_rotors): python train.py experiment=c10_arch_sweep_offline model=simple_conv_v2_smol_causal_tcn Full 26-key list: simple_conv_v2 (this file's default), simple_conv_v2_smol_causal_tcn, simple_conv_v2_gru96, simple_conv_v2_dwt, simple_conv_v2_smol_tcn, simple_conv_v2_multires, simple_conv_v2_dual_pool, simple_conv_v2_magphase, simple_conv_v2_tcn, simple_conv_v2_smol_bigru, simple_conv_v2_uni_gru96_norm_do03, simple_conv_v2_causal_tcn, smolnet_rps_tcn, simple_conv_v2_local_attn, simple_conv_v2_uni_gru128_norm, simple_conv_tcn, simple_conv_v2_uni_gru128, simple_conv_v2_transformer, smolnet_rps_causal_tcn, simple_conv_v2_uni_gru96_norm_do02, simple_conv_v2_causal_gru, simple_conv_v2_uni_gru64_norm_do03, smolnet_rps_simple_head, simple_conv_v2_uni_gru128_norm_do03, simple_conv_v2_uni_gru, simple_conv_v2_causal_gru96. See conf/experiment/c10_arch_sweep_online.yaml for the online-mix arm and conf/experiment/c10_uni_gru128_online.yaml for the overall winner (the uni-GRU family "fails offline" but wins under online mixing — see REPLICATION.md § C10). Clipped-GRU follow-up (uni_gru* variants only, grad_clip=0.5 instead of 5.0): python train.py experiment=c10_arch_sweep_offline model=simple_conv_v2_uni_gru128 \ grad_clip=0.5 epochs=200 patience=50 Historical command: train_rps_predictor.py --model <key> --data_root datasets/DREGON-LM-V4-michaels --epochs 50 --patience 10 --batch_size 32 --lr 1e-3 --weight_decay 1e-4 --loss pit_mse --epoch-progress. Results on HPC scratch (not synced) — see REPLICATION.md § C10.

Hydra wiring — data `dregon_lm_v4_michaels` · model `simple_conv_v2` · loss `pit_mse` · metrics `rps`. Train with `python train.py experiment=c10_arch_sweep_offline`,
evaluate with `python eval.py experiment=c10_arch_sweep_offline`.

## Conclusion

Offline arm: best baseline `simple_conv_v2` PIT MSE **7.89** (best R^2 `simple_conv_v2_smol_causal_tcn`, 8.38/0.83).
