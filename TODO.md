# TODO — next ~10 hours (written 2026-08-24 evening)

Working file for the current push. Strike items as they close; move durable
outcomes into `docs/experiments/` and the paper, then delete the item.

## 1. Harvest the running jobs + update the frozen-valid leaderboard

- [ ] Wait for the cluster fleet: 10 HB grid runs (`hb_{scv2,tr,gru}_{mag,if,ssq}`,
      `hb_scv2_mag_nogate`) + 2 salience retrainings (`hb_sal_multif0`,
      `hb_sal_bp`) on uni-gpushort. Best metrics from W&B HISTORY minima,
      never `run.summary`.
- [x] FLY103/FLY108 calibration CLOSED (2026-08-24): fine constants baked
      in (resid lag RMS 2.5/1.1 ms; scales 1.00525/1.00570),
      `michaels-test-frames@353cc523d609` derived + pinned. The test set
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
- [ ] DECIDED (2026-08-24): once HB numbers confirm R2, re-train R3/R4/R5
      with R2 as the real component (~6-9 gpushort runs: m3cur/m3abl
      stage-2 reruns + mixed) so every row differs from R2 only in the
      synthetic ingredient. Trigger: HB harvest looks sane.

## 4. Salience baselines on R2, done carefully (super-resolution + zero encoding)

The two queued runs (`hb_sal_multif0`, `hb_sal_bp`) cover the STANDARD-grid
row. This item is the careful arm:

- [ ] Audit the zero-RPS path end to end: (a) target side — a stopped rotor
      has f0 below the salience grid's fmin, so silence is encoded only as
      an all-dark frame; (b) decode side — what
      `salience_to_rps_segmented` emits for a frame with no peak above
      `track_threshold` (must be 0, not a hold-over or NaN). Fix the decode
      convention if needed; add a unit test for the empty-frame case.
- [ ] Rebuild the narrow+super-resolution variant (the June best: RMSE
      6.30 → 4.03; configs `*_narrow_sr`) on top of the R2 stream:
      `hb_sal_multif0_nsr` (+ Basic Pitch only if the narrow grid applies
      to it). Make sure the narrow grid still covers the full rps range
      AFTER freq-scale augmentation (x0.7-1.3 shifts the comb!) — the June
      narrow grid predates freq-scale; widen if needed.
- [ ] Submit, harvest, add to the leaderboard next to the standard-grid
      rows and the June numbers.

## 5. Central leaderboard table in the paper (all methods, per-regime + total)

- [ ] Missing row first: the blind two-stage tracker (ridge-Viterbi seed +
      peel/pi_kalman) has NEVER run on valid-full. Freeze the run-of-record
      arm from `docs/experiments/beat-vk.md` (the 0.688/1.027 cruise
      configuration), run it over the 37 clips x 8 ch on uni-cpu, score
      under the same PIT protocol. DECIDED (2026-08-24): full-envelope
      scoring, refusal/no-comb -> 0 rev/s — same rows as every other
      method; the zero regime doubles as a refusal-calibration test.
      Estimate CPU cost before submitting.
- [ ] Also add a compute column (per-second-of-audio inference cost, CPU/GPU)
      — the narrative's "beats neural but much more expensive" claim needs
      the number in the same table.
- [ ] Assemble the table: classical five, NMF highlighted, OT, salience
      (June / standard-R2 / narrow-SR-R2), neural trio per regime
      (R2 winner + R3/R4 rows as decided), blind tracker, with zero / low /
      flight / all MAE (and MSE in the appendix version).
- [ ] Port into the paper as the central results table; the doc table in
      `unified-baseline-eval.md` stays the living copy.

## 6. Fix the paper's narrative

Target narrative (author, 2026-08-24 — supersedes the 2026-08-20 outline;
update `draft.md` too, it is the frozen source of record):

- We ask if we can reliably predict mid-flight drone rotor speeds just from onboard audio.
- Since there is no prior art, we take baselines from multi-pitch tracking (classical + neural) and [tacholess] order tracking, slightly adapting them to the task when necessary.
- We try a wide array of small-sized neural regressor models architectures, both causal and non-causal, and select a few winners; we show that they reliably outperform off-the-shelf baselines with a proper training regime
- We also propose augmentations to combat data scarcity; additionally, we show that without augmentations which rescale the f0 of noise, models do not actually learn to track harmonic frequencies, but instead (most likely) learn amplitude cues
- We also design an iterative __classical__ algorithm for blind estimation of rotor speeds, it beats the neural models but is much more computationally expensive during inference.
- We also do experiments with synthetic data: simple combs and convolutional DDSP-based harmonic noise generators; however, training on synthetic data does not transfer.

Sub-tasks:

- [ ] Map each bullet onto the current section structure of
      `writing/papers/2026-08_wrapup/src/index.tex`; list the deltas
      (sections to move/merge/reframe), then execute.
- [ ] "No prior art": phrase precisely (no established method for
      PER-ROTOR speed tracking of a multirotor from onboard audio; adjacent
      art exists — single-source acoustic tachometry, drone detection,
      multi-pitch, order tracking) and verify against the bibliography.
- [ ] "Synthetic does not transfer": DECIDED (2026-08-24) — claim is
      conditional on the HB outcome. If R2 closes the gap: "synthetic
      pre-training helps only by covering regimes the real corpus lacks;
      with an honest real regime the benefit disappears". If not, soften
      to the coverage-vs-realism split. Write after the HB harvest.
- [ ] Amplitude-cues claim: cite our own evidence chain (x1.02 scale-response
      probe, freq-scale regime results) in the section that makes it.

## 7. Citations + figures sweep

- [ ] `grep -n "\\pending\|\\wip" writing/papers/2026-08_wrapup/src/index.tex`
      — resolve every marker: real citations via the bibliography MCP
      (OT paper 2508.02471, Cuesta ISMIR 2020, Bittner ICASSP 2022, VK/order
      tracking, DREGON, LibriSpeech, ...), numbers from the docs.
- [ ] Figure placeholders: regenerate from `eval.py` + `src/plots` (never
      hand-made); check every figure builds from committed results.

## 8. Qualitative output figures per regime

- [ ] Pick 3 representative valid clips: a content-rich zero clip (the
      41-50 Hz rumble clip), a transition clip (stop/start boundary), a
      cruise clip.
- [ ] One figure per clip: GT tracks + predictions overlaid for the
      leaderboard's top methods (best HB neural, blind tracker, NMF, OT,
      best salience) — the May-report per-rotor-panel layout is the
      template, via `plots` renderers (spectrogram top, per-method panels).
- [ ] Wire into the paper build (`make_figures.py` pattern), commit the
      generating script, not hand-tuned images.

## 9. Make slides for supervisor, which should simply present the paper structure

## Standing constraints

- Test set: formed but UNTOUCHED until explicitly opened.
- No heavy compute on the laptop; cluster via omnirun.
- Leaderboard numbers: frozen valid split only, per-regime + aggregate,
  same PIT protocol everywhere.
