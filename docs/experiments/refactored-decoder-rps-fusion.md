# Refactored DCUNet/DCCRN — decoder-side RPS fusion + auxiliary RPS head

**Status:** done | **Dates:** 2026-04 | **Reference:** `docs/dcunet-refactored.md` (design) | **Commands:** REPLICATION.md § B2

## Motivation

The B1 batch ([rps-conditioned-se-dregon.md](rps-conditioned-se-dregon.md))
fused RPS on the **encoder** side. Two follow-up questions motivated a
refactor: (1) does injecting RPS on the **decoder** side (closer to the mask
output) work better or worse than encoder-side fusion? and (2) can an
**auxiliary RPS-prediction head** (multi-task, telemetry-at-train-only) act as
a disentanglement signal that helps the SE branch — the mechanism behind
`GOALS.md`'s Bet 2? The refactor cleanly separates `Encoder`/`Decoder` modules
(`models.dcunet_refactored`) so decoder fusion and an aux head can be toggled
independently.

## Experiments

Baseline: `b2_dcunet_refactored_baseline` (no RPS; reuses `conf/model/dcunet.yaml`).

Decoder-side RPS fusion (no aux head):
- `b2_dcunet_refactored_decoder_bottleneck`, `b2_dcunet_refactored_decoder_hierarchical`.

Decoder fusion + auxiliary RPS-prediction head (multi-task, `rps_aux_weight = 0.1`):
- `b2_dcunet_refactored_predrps_bottleneck`, `b2_dcunet_refactored_predrps_hierarchical`,
- `b2_dccrn_refactored_predrps_bottleneck`, `b2_dccrn_refactored_predrps_hierarchical`
  (DCCRN variants; the DCCRN aux weight is assumed identical to DCUNet's 0.1 —
  the legacy config comments did not document it separately, flagged in
  REPLICATION.md § B2).

## Results

Per-config SE + auxiliary-RPS metrics are in REPLICATION.md § B2. This batch is
the concrete substrate for the multi-task-head hypothesis (train with
telemetry, evaluate without it); the comparison against the telemetry-blind
baseline and the telemetry-given upper bound is the Bet-2 kill criterion in
`GOALS.md`.

## Conclusion

Decoder-side fusion and the auxiliary RPS head are implemented and runnable as
a first-class part of the framework (`conf/loss/masked_mse_plus_pit_rps_w*.yaml`
supplies the combined SE + weighted-PIT-RPS loss). Whether the aux head beats
the RPS-blind baseline on SE metrics is the open Bet-2 question this batch
exists to answer.
