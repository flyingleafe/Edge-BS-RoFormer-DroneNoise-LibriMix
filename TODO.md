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
- [x] Central table ASSEMBLED in the paper (2026-08-25): tab:leaderboard
      with 21 rows (training-free, blind both conventions, salience,
      neural per regime), weighted all-frame MAE, compute column, reading
      paragraph. One pending marker for the still-training cells
      (R3-R5 transformer, R4 scv2, ebsrof, CKLA rows).

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

- [x] Clips verified (2026-08-24): 36 (pure zero, 251/0/0), 8
      (transition, 87/59/105), 20 (cruise, 0/0/251).
- [x] Generator built + committed (make_figures.py, configurable
      method list via zoo:/classical:/npz: sources; renders today with
      hb_scv2_if / real-only / NMF / HPS). Re-render with the final
      winner set + blind npz + OT npz once those rows land — the one
      remaining step of this item.

## 9. Make slides for supervisor, which should simply present the paper structure

## 9. Edge-BS-RoFormer on R2 + failure diagnostics

- [ ] `hb_ebsrof` experiment (edge_bs_rof_rps model, R2 policy). The July
      attempt never learned (val ~1150 flat, cause undiagnosed, docs still
      say "Pending run"). Instrument the rerun: gradient norms, output-head
      stats, a 10x-lower-lr arm if flat again. Either it learns under R2
      (new leaderboard row + the RoPE hypothesis finally tested) or the
      diagnosis names the blocker in the paper's architecture section.

## 10. CKLA (+ KLA baseline) on R2 -> paper

- [ ] `hb_ckla` (the campaign's best phase-only variant) on the R2
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
- [ ] Stage-A refiner training SUBMITTED (hb_hgckla_ref, gpushort); on
      completion run G1 synthetic comparison vs pi_kalman + G2 protocol
      eval (`rps_eval --protocol beatvk --pred model:...`).

## 12. Slides — start NOW (before all results land)

- [x] Deck BUILT (2026-08-25): writing/slides/2026-08-25_wrapup-progress/
      (14 pages, 2 critic rounds + 1 revise round, render verified).
      UNCOMMITTED per the writeup convention — user reviews first.
      Remaining: refresh the in-flight/WIP slides when the rerun numbers
      land (same one-command rebuild as the figures).

## Compute budget note

- vast.ai: up to $7 approved for A100 rental to parallelize (R5 runs need
  big memory; HG-CKLA/CKLA training benefit). SSH key must be registered
  account-level before first instance.

## Standing constraints

- Test set: formed but UNTOUCHED until explicitly opened.
- No heavy compute on the laptop; cluster via omnirun.
- Leaderboard numbers: frozen valid split only, per-regime + aggregate,
  same PIT protocol everywhere.
