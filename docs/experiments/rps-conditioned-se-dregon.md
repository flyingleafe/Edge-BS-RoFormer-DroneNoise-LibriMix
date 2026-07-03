# RPS-conditioned speech enhancement on DREGON-LM (Paper 2, telemetry-given)

**Status:** done (P2 telemetry-given baseline) | **Dates:** 2026-03 – 2026-04 | **Reference:** inspired by Gulli et al., EURASIP 2025 (RPS-DCUN) | **Commands:** REPLICATION.md § B1

## Motivation

Paper 2's premise: harmonic drone noise is *structured* — its comb of tones is
set by the rotor speeds (RPS). If the model is told the rotor speeds, it should
suppress the noise better than a telemetry-blind baseline. This batch
establishes the **telemetry-given upper bound** on **DREGON-LM** (real drone
recordings + LibriSpeech): DCUNet and DCCRN enhancement backbones conditioned
on **oracle RPS** via several fusion strategies, plus the no-RPS baselines they
must beat. It is the "finish the P2 telemetry-given baseline" milestone in
`GOALS.md`'s sequencing, and the reference point for Bets 1 (pseudo-RPS) and 2
(multi-task RPS head).

## Experiments

Baselines (no RPS): `b1_dcunet_baseline_dregon`, `b1_dccrn_baseline_dregon`.

DCUNet + oracle-RPS fusion (encoder side):
- `b1_dcunet_rps_dregon` (bottleneck, 5-layer), `b1_dcunet_rps_dregon_bottleneck`
  (RPS-DCUN6-P, 6-layer, after Gulli et al.), `b1_dcunet_rps_dregon_gru`
  (RPS-DCUN5, GRU fusion), `b1_dcunet_rps_dregon_hierarchical` (RPS-DCUN5-H).
- Dataset-ambiguous early variants (best-effort `dregon_lm_v1`, see REPLICATION.md
  § B1 caveat): `b1_dcunet_rps_bottleneck_5a`, `b1_dcunet_rps_gru_5b`,
  `b1_dcunet_rps_hierarchical_5c`.

DCCRN + oracle-RPS fusion: `b1_dccrn_rps_dregon` (encoder GRU),
`b1_dccrn_lite_rps_dregon` (embedded-deployment DCCRNLite variant).

Auxiliary-RPS-head (multi-task) and RPS-only variants:
- `b1_dcunet_rps_predrps_dregon` (RPS conditioning + aux RPS head, weight 0.5),
  `b1_dccrn_rps_predrps_dregon` (weight 2.0).
- `b1_dcunet_rpsonly_dregon`, `b1_dccrn_rpsonly_dregon` — encoder + RPS-prediction
  head, RPS-only loss (no SE term) — these are RPS-*predictors* built on the SE
  encoders, a bridge to the direct RPS-prediction line
  ([simpleconv-rps-architecture-search.md](simpleconv-rps-architecture-search.md)).
- `b1_legacy_rps_predictor_dregon` — the standalone legacy `rps_predictor`
  model_type, kept for continuity.

## Results

Per-config SE metrics (SI-SDR / SDR / PESQ / STOI) and the historical
checkpoint paths are catalogued in REPLICATION.md § B1. Several checkpoint
paths there are flagged stale/historical, and the 5a/5b/5c variants carry a
documented dataset ambiguity. The decoder-side and disentanglement follow-ups
moved to the refactored batch below.

## Conclusion

This batch is the telemetry-given P2 reference. Whether oracle-RPS conditioning
actually helps, and by how much, is the baseline every downstream bet is
measured against; the concrete kill-criteria comparisons (oracle vs blind vs
pseudo-RPS) live in `GOALS.md`'s portfolio. The decoder-fusion + auxiliary-head
redesign is continued in
[refactored-decoder-rps-fusion.md](refactored-decoder-rps-fusion.md).
