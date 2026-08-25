# TODO — next ~10 hours (written 2026-08-24 evening)

Working file for the current push. Strike items as they close; move durable
outcomes into `docs/experiments/` and the paper, then delete the item.

## 1. Harvest the running jobs + update the frozen-valid leaderboard

- [x] DONE (2026-08-25, W&B history minima). Originally: wait for the cluster fleet: 10 HB grid runs (`hb_{scv2,tr,gru}_{mag,if,ssq}`,
      `hb_scv2_mag_nogate`) + 2 salience retrainings (`hb_sal_multif0`,
      `hb_sal_bp`) on uni-gpushort. Best metrics from W&B HISTORY minima,
      never `run.summary`.
- [x] FLY103/FLY108 calibration CLOSED (2026-08-24): fine constants baked
      in (resid lag RMS 2.5/1.1 ms; scales 1.00525/1.00570),
      `michaels-test-frames@353cc523d609` derived + pinned. The test set
      stays DORMANT: no training/valid/eval config references it.
- [x] DONE (2026-08-25) for every finished checkpoint incl. the regime
      reruns, ebsrof, ckla. Originally: run the per-regime probe (zero/low/flight, per-frame Hungarian PIT)
      on every finished checkpoint — same protocol as
      `results/m3cur_regime_probe/regime_probe.py`.
- [x] DONE (2026-08-25), all rows in. Originally: update the leaderboard in `docs/experiments/unified-baseline-eval.md`:
      one table, every row on `dload:DREGON-LM-V4-michaels-valid-full`.
      Rows already present: classical five, OT multi-pitch. Rows to add:
      HB grid (10), salience retrained (2), current neural trio
      (merge from `regime_probe.json`). Optional row: June salience
      checkpoints, if still loadable from the zoo.
- [x] DONE (2026-08-25): off-call/drift columns captured per probe row
      in the doc. Originally: HB-specific readouts: clean-off-call rate on zero
      frames, 10-45 rev/s drift mass, gate saturation statistics, front-end
      ranking consistency across the three architectures.

## 2. Architecture-search provenance → paper section

Goal: a paper section that motivates the winner models (scv2 / transformer-IF
/ uni_gru128) by documenting the search they won.

- [x] Read during the section draft (2026-08-24). Originally: re-read `docs/experiments/simpleconv-rps-architecture-search.md` +
      report `writing/reports/2026-06-19_rps-arch-sweep-v4-michaels` +
      the C3/C6/C10 config files (`conf/experiment/`, REPLICATION.md).
- [x] R1 established + named in sec:splits (2026-08-24). Originally: establish EXACTLY which training data and regime the 26-variant sweep
      used (fixed DREGON-LM-V4-michaels mixtures vs online mixing; which
      augmentations; which validation split and monitor). This defines
      "Regime R1" below.
- [x] Collected: the reshuffle caution + attribution matrix carry it
      (2026-08-25). Originally: collect the later head-to-head evidence that kept the trio:
      CKLA campaign matched-protocol table, G1-G3 front-end arms,
      causal-head sweep notes.
- [x] Section DRAFTED (2026-08-24): \S{}sec:archsearch — search space,
      two arms, three findings, the ranking-reshuffle caution as the
      reason all three architectures carry the paper.

## 3. Data-sources table + training-regimes taxonomy (paper)

- [x] DRAFTED (2026-08-24) as \S{}sec:splits: splits table
      (tab:splits) + the five named regimes R1-R5 in the paper. Remaining:
      re-point \S8/\S5 prose at the taxonomy. Sources for the record:
      - Train: DREGON room2 in_flight_noise (5 recs) + FLY125 (+ synthetic
        arms per regime; LibriSpeech train-clean-100 speech).
      - Valid (frozen): DREGON-LM-V4-michaels-valid-full — room1
        free-flight_nosource + speech-low + whitenoise-low + FLY124.
      - TEST (held out, untouched): free-flight_speech-high_room1,
        free-flight_whitenoise-high_room1 (already in DREGON-frames),
        FLY103 + FLY108 (michaels-test-frames once published). No leakage
        note: test shares room1/rig with valid, not with train.
- [x] DONE — R1-R5 named with policy files, in the paper and the deck
      (2026-08-25). Originally: define the training-regime taxonomy, one
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
- [x] DONE (2026-08-25): tab:splits + the regime list in sec:splits;
      ablation and leaderboard sections use the R1-R5 names. Originally: write the two tables into the
      wrap-up draft, and re-point existing sections (§8 ablation, §5
      validation description) at the taxonomy instead of ad-hoc prose.
- [x] DONE (2026-08-25): all nine rerun cells trained + probed; grid
      complete in the leaderboard doc. Originally: once HB numbers confirm R2, re-train R3/R4/R5
      with R2 as the real component (~6-9 gpushort runs: m3cur/m3abl
      stage-2 reruns + mixed) so every row differs from R2 only in the
      synthetic ingredient. Trigger: HB harvest looks sane.

## 4. Salience baselines on R2, done carefully (super-resolution + zero encoding)

The two queued runs (`hb_sal_multif0`, `hb_sal_bp`) cover the STANDARD-grid
row. This item is the careful arm:

- [x] Zero-path audit DONE (2026-08-24): decode HELD OVER the previous
      speed on dark frames (phantom hover) — fixed to emit exact zeros,
      3 tests added (2 fail pre-fix). Target side: stopped rotors go dark
      correctly, but sub-grid speeds CLAMP onto bin 0 (a 7.88 rev/s floor
      for the June fmin-55 grid on the frozen split).
- [x] Narrow-SR rebuilt as `hb_sal_multif0_nsr` (2026-08-24): widened
      linear 20-130 rev/s output grid, 720 bins at June's 0.153 Hz/bin,
      HCQT input fmin 20 / 3 octaves; GT round-trip floor 7.24 -> 2.25
      rev/s vs the June grid. (The salience grid is in rotor rev/s
      directly — no blade factor.)
- [x] DONE (2026-08-25): hb_sal_multif0_nsr row landed (48.2/16.1/4.7).
      Originally: submit, harvest, add to the leaderboard next to the standard-grid
      rows and the June numbers.

## 5. Central leaderboard table in the paper (all methods, per-regime + total)

- [x] DONE (2026-08-25): blind row on valid-full, both conventions in the
      table (ungated flight 2.27; gated zero 0.01, refusal->0).
      Originally: the blind two-stage tracker (ridge-Viterbi seed +
      peel/pi_kalman) has NEVER run on valid-full. Freeze the run-of-record
      arm from `docs/experiments/beat-vk.md` (the 0.688/1.027 cruise
      configuration), run it over the 37 clips x 8 ch on uni-cpu, score
      under the same PIT protocol. DECIDED (2026-08-24): full-envelope
      scoring, refusal/no-comb -> 0 rev/s — same rows as every other
      method; the zero regime doubles as a refusal-calibration test.
      Estimate CPU cost before submitting.
- [x] Compute column IN (2026-08-25). Originally: add a compute column (per-second-of-audio inference cost, CPU/GPU)
      — the narrative's "beats neural but much more expensive" claim needs
      the number in the same table.
- [x] Central table ASSEMBLED in the paper (2026-08-25): tab:leaderboard
      with 21 rows (training-free, blind both conventions, salience,
      neural per regime), weighted all-frame MAE, compute column, reading
      paragraph. ALL CELLS FILLED (2026-08-25): R3/R4 transformer, R4
      scv2 (17.6 — new campaign best), ebsrof, CKLA rows in; no pending
      markers remain in the table.

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

- [x] Mapped + executed (2026-08-24): abstract and contributions
      rewritten to the six-bullet arc; section order already matched; new
      sec:leaderboard stub before the Discussion holds the central table.
      NOTE: draft.md needs no update — its header says edits happen in
      index.tex from v0.2 on.
- [x] "No prior art" phrased as "no published direct method / no
      published direct baselines" with the adjacent-task suite as the
      response (abstract + contribution 1).
- [x] WRITTEN (2026-08-25), final form: coverage-not-realism with the
      comb-only twist — the comb curriculum on the R2 base BEATS real-only
      for scv2 (17.6 vs 22.1) and the causal GRU; the generator version
      never beats it; mixed degrades all. Abstract, reading paragraph,
      ablation section and conclusion updated. Originally: claim is
      conditional on the HB outcome. If R2 closes the gap: "synthetic
      pre-training helps only by covering regimes the real corpus lacks;
      with an honest real regime the benefit disappears". If not, soften
      to the coverage-vs-realism split. Write after the HB harvest.
- [x] Amplitude-cues claim wired to the probe + freq-scale evidence in
      sec:overfit (2026-08-24 draft pass).

## 7. Citations + figures sweep

- [x] Citations RESOLVED (2026-08-25) except the unpublished MD2
      technical report (2 markers, unresolvable until it exists); the
      stale R3-R5 rerun note removed. Originally: grep pending/wip —
      — resolve every marker: real citations via the bibliography MCP
      (OT paper 2508.02471, Cuesta ISMIR 2020, Bittner ICASSP 2022, VK/order
      tracking, DREGON, LibriSpeech, ...), numbers from the docs.
- [x] DONE (2026-08-25): all four in-section figures real and built from
      committed results — comb overlays (tracking campaign renders),
      freq-scaling probe (regenerated: no-aug ~0.2-0.9% response vs R2
      0.8-1.3% at +-4%), per-regime bars (probe JSONs), generator
      line-vs-loss curves (perrotor bundle; prose re-anchored to the
      recomputed 0.7-1.7 dB selection gaps). Winners size table added.
      Remaining markers: 2 unpublished-tech-report citations, the dormant
      final-test numbers, and the 2 TO-RUN tacholess rows (offered to the
      user, awaiting go/no-go).

## 8. Qualitative output figures per regime

- [x] Clips verified (2026-08-24): 36 (pure zero, 251/0/0), 8
      (transition, 87/59/105), 20 (cruise, 0/0/251).
- [x] Generator built + committed (make_figures.py, zoo:/classical:/npz:
      sources). RE-RENDERED with the final set (2026-08-25): wide-grid
      salience npz + blind-tracker npz + NMF + r4hb_scv2 (best neural),
      big panels, cruise fixed to 60-100 rev/s; figures live in the deck
      and the paper submodule.

## 9. Make slides for supervisor, which should simply present the paper structure

## 9. Edge-BS-RoFormer on R2 + failure diagnostics

- [x] CLOSED (2026-08-25): hb_ebsrof LEARNS under R2 (best 396 vs July's
      flat 1150) — cruise-competitive (flight MAE 3.21), fails zeros
      (34.5); row + diagnosis in the leaderboard doc and paper; low-lr
      reserve arm unnecessary. Originally: `hb_ebsrof` experiment (edge_bs_rof_rps model, R2 policy). The July
      attempt never learned (val ~1150 flat, cause undiagnosed, docs still
      say "Pending run"). Instrument the rerun: gradient norms, output-head
      stats, a 10x-lower-lr arm if flat again. Either it learns under R2
      (new leaderboard row + the RoPE hypothesis finally tested) or the
      diagnosis names the blocker in the paper's architecture section.

## 10. CKLA (+ KLA baseline) on R2 -> paper

- [x] CLOSED (2026-08-25): hb_ckla aggregate 40.6 (zero 5.95 / flight
      3.51), coherent across regimes, trails scv2 everywhere; row in
      leaderboard + paper table; the design-gap lesson added to the
      Discussion. Originally: `hb_ckla` (the campaign's best phase-only variant) on the R2
      policy, same budget as the HB grid. (`hb_fkla` CANCELLED by the user
      2026-08-25 — the plain-KLA cross-implementation baseline is ~6x
      slower per step and its scientific question, rotation-vs-no-rotation,
      was already answered by the campaign's _norot controls.) Probe per-regime, add to the
      leaderboard and the paper's architecture-search section; update the
      narrative where CKLA is mentioned (matched-protocol numbers move
      from the old stream to R2).

## 11. HG-CKLA: implement + train

- [x] Implemented (2026-08-25): src/models/hg_ckla.py, 221k params, 12
      tests, innovation physics <0.01% error; two design findings recorded
      in the design doc §9 (phase-aligned gather, shared-gather pairs).
- [x] CLOSED (2026-08-25): stage-A trained (flat at 2.78 from epoch 1;
      identity 3.87). Head-to-head on identical corrupted inits, 37 valid
      clips: HG-CKLA 3.03 beats one pi_kalman pass 3.44 (all action in
      flight: -18% vs -5% MSE); G1 synthetic gate and cruise-precision G2
      unrun — recorded as future work, not a leaderboard row. Full record
      in unified-baseline-eval.md.

## 12. Slides — start NOW (before all results land)

- [x] Deck BUILT (2026-08-25): writing/slides/2026-08-25_wrapup-progress/
      (14 pages, 2 critic rounds + 1 revise round, render verified).
      Then REBUILT to the user's figures-and-tables structure
      (2026-08-25, 20 pages), output panels re-rendered with the final
      winner set (r4hb_scv2), committed and pushed.

## Compute budget note

- vast.ai: up to $7 approved for A100 rental to parallelize (R5 runs need
  big memory; HG-CKLA/CKLA training benefit). SSH key must be registered
  account-level before first instance.

## Standing constraints

- Test set: formed but UNTOUCHED until explicitly opened.
- No heavy compute on the laptop; cluster via omnirun.
- Leaderboard numbers: frozen valid split only, per-regime + aggregate,
  same PIT protocol everywhere.
