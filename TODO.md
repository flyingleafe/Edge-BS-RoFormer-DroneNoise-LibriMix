# TODO — next ~10 hours (written 2026-08-24 evening)

Working file for the current push. Strike items as they close; move durable
outcomes into `docs/experiments/` and the paper, then delete the item.

## 1. Harvest the running jobs + update the frozen-valid leaderboard

- [ ] Wait for the cluster fleet: 10 HB grid runs (`hb_{scv2,tr,gru}_{mag,if,ssq}`,
      `hb_scv2_mag_nogate`) + 2 salience retrainings (`hb_sal_multif0`,
      `hb_sal_bp`) on uni-gpushort. Best metrics from W&B HISTORY minima,
      never `run.summary`.
- [ ] Also close the FLY103/FLY108 calibration (agent output → constants →
      `derive michaels-test-frames` → `dload pin` → commit). The test set
      stays DORMANT: no training/valid/eval config references it.
- [ ] Run the per-regime probe (zero/low/flight, per-frame Hungarian PIT)
      on every finished checkpoint — same protocol as
      `results/m3cur_regime_probe/regime_probe.py`.
- [ ] Update the leaderboard in `docs/experiments/unified-baseline-eval.md`:
      one table, every row on `dload:DREGON-LM-V4-michaels-valid-full`.
      Rows already present: classical five, OT multi-pitch. Rows to add:
      HB grid (10), salience retrained (2), current neural trio
      (merge from `regime_probe.json`). Optional row: June salience
      checkpoints, if still loadable from the zoo.
- [ ] HB-specific readouts while harvesting: clean-off-call rate on zero
      frames, 10-45 rev/s drift mass, gate saturation statistics, front-end
      ranking consistency across the three architectures.

## 2. Architecture-search provenance → paper section

Goal: a paper section that motivates the winner models (scv2 / transformer-IF
/ uni_gru128) by documenting the search they won.

- [ ] Re-read `docs/experiments/simpleconv-rps-architecture-search.md` +
      report `writing/reports/2026-06-19_rps-arch-sweep-v4-michaels` +
      the C3/C6/C10 config files (`conf/experiment/`, REPLICATION.md).
- [ ] Establish EXACTLY which training data and regime the 26-variant sweep
      used (fixed DREGON-LM-V4-michaels mixtures vs online mixing; which
      augmentations; which validation split and monitor). This defines
      "Regime R1" below.
- [ ] Also collect the later head-to-head evidence that kept the trio:
      CKLA campaign matched-protocol table, G1-G3 front-end arms,
      causal-head sweep notes.
- [ ] Draft the section for `writing/papers/2026-08_wrapup/` (structure:
      search space → protocol/regime → outcome → why these three carry the
      rest of the paper). Mark numbers that need re-verification against
      the docs with \pending{}.

## 3. Data-sources table + training-regimes taxonomy (paper)

- [ ] Define, in one table, the data sources for RPS-tracker training /
      validation / FINAL TEST:
      - Train: DREGON room2 in_flight_noise (5 recs) + FLY125 (+ synthetic
        arms per regime; LibriSpeech train-clean-100 speech).
      - Valid (frozen): DREGON-LM-V4-michaels-valid-full — room1
        free-flight_nosource + speech-low + whitenoise-low + FLY124.
      - TEST (held out, untouched): free-flight_speech-high_room1,
        free-flight_whitenoise-high_room1 (already in DREGON-frames),
        FLY103 + FLY108 (michaels-test-frames once published). No leakage
        note: test shares room1/rig with valid, not with train.
- [ ] Define the training-regime taxonomy used throughout the paper, one
      name each, with the exact policy file per regime:
      1. R1 — architecture-search regime (as established in item 2).
      2. R2 — final real-only regime: full envelope + freq-scale v2 +
         time-warp + gain/polarity + honest silence arm + SNR reference
         floor = `conf/online_mix/hb_silence_dload.yaml` (the regime the
         HB grid trains with now).
      3. R3 — gen+comb curriculum (m3cur: generated+comb stage 1 → real
         stage 2).
      4. R4 — comb-only curriculum (m3abl_comb: comb stage 1 → real
         stage 2).
      5. R5 — mixed one-stage (m3abl_mixed: real 50% / generated 25% /
         comb 25%).
- [ ] Decide naming + notation in the paper, write the two tables into the
      wrap-up draft, and re-point existing sections (§8 ablation, §5
      validation description) at the taxonomy instead of ad-hoc prose.
- [ ] Open question to resolve while writing: whether the R3/R4/R5 rows in
      the final tables should be re-trained on top of R2 (with silence arm)
      for consistency, or reported as-run (on the fs_v2 recipe without the
      silence arm) with a footnote. Decide once HB numbers are in.

## 4. Retrain multi-pitch salience baselines on R2 (be careful with super-resolution, make sure zero RPS is encoded sufficiently), put into comparison with our models

## Standing constraints

- Test set: formed but UNTOUCHED until explicitly opened.
- No heavy compute on the laptop; cluster via omnirun.
- Leaderboard numbers: frozen valid split only, per-regime + aggregate,
  same PIT protocol everywhere.
