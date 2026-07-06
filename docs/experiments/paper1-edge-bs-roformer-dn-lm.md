# Paper 1 — Edge-BS-RoFormer on DN-LM (ultra-low-SNR UAV speech enhancement)

**Status:** done (reproduction of published work) | **Dates:** 2026-01 – 2026-03 | **Reference:** Liu et al., "Edge-Deployed Band-Split RoPE Transformer for Ultra-Low SNR UAV Speech Enhancement", *Drones* 2025 | **Commands:** REPLICATION.md § A1

## Motivation

Paper 1 is the project's entry point: speech enhancement under drone (UAV)
noise at ultra-low SNR (0 to −30 dB) on the synthesised **DN-LM**
(DroneNoise-LibriMix) dataset. The proposed model, **Edge-BS-RoFormer**, is a
band-split RoPE transformer sized for edge deployment. This batch reproduces
the paper's ablation ladder and its comparison baselines so that later
RPS-conditioned work (Paper 2) has a validated, framework-native SE baseline to
build on and to benchmark against.

## Experiments

Ablation ladder on the same Edge-BS-RoFormer backbone, each step adding one
component (see REPLICATION.md § A1 for exact hyperparameters and metrics):

- `a1_edge_bs_rof_nothing` — ablation floor: no Flash Attention, no RoPE.
- `a1_edge_bs_rof_fa` — + Flash Attention.
- `a1_edge_bs_rof_fa_rope48` — + Rotary Position Encoding, head dim 48.
- `a1_edge_bs_rof_fa_rope64` — **the headline model**: RoPE head dim 64.

Comparison baselines (published SE architectures, same DN-LM protocol):

- `a1_baseline_dcunet` — DCUNet.
- `a1_baseline_dptnet` — DPTNet.
- `a1_baseline_htdemucs` — HTDemucs.

## Results

The paper's headline claims (reproduced / transcribed in README.md's Results
table and REPLICATION.md § A1): Edge-BS-RoFormer with RoPE(64) leads the
baselines on SI-SDR / PESQ at −15 dB SNR while remaining edge-deployable —
Jetson AGX Xavier real-time factor ≈ 0.325, 8.5 MB weights, < 500 MB runtime
memory. The ablation confirms Flash Attention and RoPE each contribute to the
headline configuration. Per-config numbers live in REPLICATION.md § A1; note
the DN-LM sample-count discrepancy (README 6480 train vs the former
`replicate_paper.py` default 64800) flagged there.

## Conclusion

Edge-BS-RoFormer is the validated, framework-native ultra-low-SNR SE model and
the DCUNet/DPTNet/HTDemucs baselines are the standing comparison set. This
batch is treated as a reproduction of published work rather than a novel
result; its role going forward is to anchor Paper 2's RPS-conditioned
enhancement — see [rps-conditioned-se-dregon.md](rps-conditioned-se-dregon.md).
