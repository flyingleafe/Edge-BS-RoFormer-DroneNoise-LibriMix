# Goal: Neural(-Hybrid) RPS Prediction Beats the VK Blind Reference

**Archived 2026-07-29** (paused in favor of the VK-tracking-improvement
program; to be re-entered as the active goal once that program resolves).

## Goal statement (verbatim, set 2026-07-28)

> make CKLA or any other architectural idea beat VK blind reference on
> DREGON + Michael's validation set. attach training regimes, generated
> data, using drone noises without RPS annotation in self-supervised /
> unsupervised manner (dload ls for available data). use subagents for
> simpler tasks, think hard, be creative, and stop at nothing. always use
> omnirun for submitting jobs. you may use vast for quicker training
> results, there is $10 on account.

## Evaluation protocol (frozen 2026-07-29 — supersedes all earlier bars)

Dataset `beatvk-valid-raw@268c766052cb` (4 recordings: 3 DREGON room1
free-flights + FLY124; native audio, RAW measured telemetry, frozen 16 s
window manifest, 15 windows). Single scorer `scripts/beatvk_eval.py`
(per-window PIT-MAE vs raw telemetry; smoothing arms prediction-only).
Success = a neural/hybrid architecture beating the best VK-tracker row of
the scoreboard on BOTH the dregon_cruise and fly124_cruise pools.

Historical (pre-freeze, smoothed-reference) result for the record: on the
blind sweep protocol, pure phase-only CKLA stitched-chmean scored
**0.842 vs blind VK 1.027** on FLY124-cruise — the Michael's bar was
crossed; DREGON (blind 0.688) was not.

## Status at archive time

- Scoreboard (fixed protocol) rows landed so far (dregon_cruise /
  fly124_cruise, arm none): phase-only CKLA 3.225/1.282, scv2_fs_v2
  2.856/2.407, unigru128_fs_v2 4.247/2.267. Pending: KLA, transformer,
  CKLA-mean, e12 rows, VK arms (blind/neural/telem-init seeds), 4 s CKLA.
- Closed negative arms: AVQ pseudo-label retrain (anchor collapse
  unfixed), conditional refiner v1 (no pull from 1 s windows), classical
  post-hoc refinement of any track (stage D / B+C / vk_track / fusion —
  all null on real audio).
- Mechanistic map: DREGON neural error = anchor collapse (even-ladder
  rotor placement) + fluctuation under-tracking; VK blind score = its
  Viterbi ridge stage only (pair-mean ~0.55 + pair-split ~0.55 error);
  DREGON GT carries ±0.6 rev/s raw jitter (now included in the metric).
- Assets: `docs/experiments/beat-vk.md` (campaign design + bets),
  `scripts/{beatvk_eval,beatvk_vk_arms,neural_seeded_vk,vk_pseudolabel,
  rps_refiner_eval,vk_phase_validation}.py`, datasets
  `AVQ-egonoise-vkrps` (pseudo-labels), `beatvk-valid-raw` (protocol),
  SPCUP19 annotation (in flight), memory `beat-vk-campaign`.

## Re-entry criteria

Whatever the VK-improvement program produces (a stronger VK reference or
a proof of its ceiling), this goal resumes against the fixed protocol:
beat the best VK row on both pools with a neural/hybrid architecture,
leveraging the improved tracking (better pseudo-labels, better hybrid
head) or the proven ceiling (as the target to undercut).
