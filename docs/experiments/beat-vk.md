# Beat-VK Campaign: Neural(-Hybrid) RPS Prediction Past the Blind VK Reference

**Status:** active — goal set 2026-07-28 · **Prior context:** `docs/experiments/ckla.md`
(readout-family results), `docs/experiments/g1-vk-parity.md` (bars),
`docs/vk-order-tracking-design.md` (blind VK pipeline §7),
`docs/experiments/rps-trajectory-refinement.md` (stages A–D lessons).

## Goal

Beat the blind-VK reference on the DREGON + Michael's validation sets with an
architectural idea (CKLA or otherwise). Allowed levers: training regimes,
generated data, self-/unsupervised use of unannotated drone noise. Bars
(guarded blind VK, 20 s windows): **DREGON cruise pooled 0.688**, **FLY124
cruise 1.027** (err_sm rev/s). Same-clip telemetry-init VK reference (8 s clip
protocol, `ref_mae_vk_rec` in `scripts/rps_predictor_vk_eval.py`): **0.729 /
0.282**.

## Gap analysis (2026-07-28 cruise eval, vk-cruise-cpu-3f32b1)

Best neural (phase-only CKLA, chmean best-arm): dregon_cruise **2.79/2.84**
(two seeds), fly124_cruise **1.29/1.33**. Decomposition:

- FLY124: most clips 0.7–1.3 MAE — several already beat the 1.027 bar; the
  pooled number is dragged by two clips. Gap ≈ 25%.
- DREGON: a **uniform ~2.2–3 floor across all free-flight cruise clips**
  (median 2.26; not outlier-driven) + two bad low-RPS/sweep clips (5.1, 6.0).
  Relative error 2.8% vs VK 0.86% at ~80 rev/s.
- DREGON GT itself carries ±0.6 rev/s fast jitter (refinement audit); VK's
  0.688 is near the audio-information floor. The neural gap is **precision**
  (sub-bin frequency reading, per-rotor phase tracking), not coverage: the
  training set already contains DREGON noise.
- Blind VK's own weakness is dual: cost (~16× realtime, dominated by seeding
  53–102 s + midband 205 s per 16 s window) and FLY124 seeding failures
  (blind 1.027 vs telemetry-init 0.282 on the same recording).

## Bets (priority order)

### R1 — Neural-seeded VK hybrid (highest certainty, cheapest)

The neural predictor replaces blind seeding: per-rotor, time-varying seeds at
err 1.3–2.2 rev/s — strictly better input than the constant blind bases
`CAPTURE_CFG` was designed to capture from. Then `vk_track` capture+refine
(rtf ~0.04–0.35) polishes to VK precision. Expected landing zone ≈
telemetry-init VK (0.73/0.28) at ~real-time total cost, i.e. **beats the blind
bars on both pools while being ~16× faster than blind VK**.

Pilot: `scripts/neural_seeded_vk.py` — 37-clip protocol, arms `refine` /
`caprefine`, seeds from `ckla_phaseonly_best` chmean. Success gate: pooled
dregon_cruise < 1.0 AND fly124_cruise < 0.9 on the 8 s protocol; then a 20 s
window variant for exact bar comparison.

Follow-up if capture is the bottleneck: k-ladder stage-D style refinement
(`rps_refinement.refine_coherent`) between neural and VK stages; learned
(differentiable) refinement head as the fully-neural version of the same idea.

### R2 — VK pseudo-labels on unannotated corpora (pure-neural path)

Run the guarded blind/seeded VK annotator over unannotated real drone noise;
train the neural predictor on GT + confidence-gated pseudo-labels. Attacks
seeding robustness (FLY124-style failures) and domain diversity; transfers
VK's precision into the fast student.

Corpora (dload): `AVQ-egonoise` (705 s, pilot), `SPCUP19-egonoise` (278 clips,
10 rigs — triage first: 5/7 lock per refinement stage results),
`DroneAudioSet` / `drone_audio` / `AeroSonicDB` (later, triage by harmonicity).
Annotator: `scripts/vk_spcup.py` pattern upgraded to `blind_seed(arms={K,R})`
init (per the pipeline map); keep `VKResult.confidence` + `track_comb_confidence`
for gating. uni-cpu via omnirun (16× realtime, parallel across clips).

### R3 — Equivariance self-supervision (no labels needed)

SPICE-style consistency: `pred(freq_scale_α(x)) ≈ α·pred(x)` as an auxiliary
loss on unlabeled real noise batches. Directly reinforces frequency-reading
(the amplitude-anchor failure) on real acoustics. Cheap to add to the online
mixer; combine with R2 corpora.

### R4 — Converged phase-only + tracking-aware selection (in flight)

`phonly_long` (Slurm 21037275, sae) — uncapped run. Both current seeds were
wall-capped mid-improvement. Also: checkpoint selection must use a
tracking-aware metric (envelope MSE snapshots pre-tracking models — CKLA
ledger). Cheap ablation of the readout family at convergence.

## Order of play

1. R1 pilot (37-clip). If gate passes → 20 s window bar-comparison run +
   speed benchmark; the hybrid becomes the deliverable architecture.
2. R2 pilot on AVQ-egonoise in parallel (uni-cpu); SPCUP triage next;
   retrain phase-only with pseudo-label stage; re-eval.
3. R3 folded into the R2 retrain (same new-data plumbing).
4. R4 lands whenever sae dequeues; fold into whichever of R1/R2 wins.

Pure-neural (R2+R3+R4) remains the scientific headline even if R1 crosses the
bar first — R1 quantifies how much of VK's edge is seeding vs precision.

## Results

### Protocol recalibrated and re-scored (2026-07-31)

Michael's telemetry was recalibrated (clock dilation + a +0.698 % rev/s scale
— `docs/experiments/rps-refine-precision.md` §§ WP13/WP14) and the frozen
protocol was republished from `michaels-frames@fdef818432e9`:

**`beatvk-valid-raw@268c766052cb` → `@54849c13ed3a`.**

Driver `scripts/beatvk_rescore.py`; jobs `python-7553a8` (first clean run),
`python-764ca5` (final, adds the seed-swap control). Every `fly124_*` figure
in the 2026-07-29 scoreboard below predates this pin.

**What moved in the protocol itself** (`--prep-only` diff, recorded in
`results/beatvk_rescore_v4/summary.json → prep_diff`):

- **Window boundaries and window count are UNCHANGED on all 15 windows.**
  FLY124 still tiles `[0,16) … [80,96)` — 6 windows, 2 warmup + 4 cruise;
  the three DREGON recordings still 3 + 3 + 3 at unchanged offsets. The
  FLY124 eval span grew 109.413 → 109.485 s, nowhere near a 7th window.
  Regime tags are unchanged. So the protocol measures the same thing.
- **DREGON is bit-identical** — audio *and* labels, on all 9 windows
  (asserted by the driver, which aborts on any DREGON difference). No DREGON
  number in this document changes, and the 0.688 DREGON bar stands.
- FLY124 labels rose **+0.235 / +0.232** rev/s (warmup w00/w01) and
  **+0.521 … +0.564** rev/s (cruise w02–w05), and are re-timed by the
  dilation.
- **The FLY124 window AUDIO also moved**, which is why this needed a
  re-score and not a re-grade. `time_offset` −20.84 → −20.753813 changes how
  much of the WAV `michaels._load_michaels_data_raw` trims off the head, so
  every FLY124 window now cuts audio **86.188 ms earlier** in the recording
  — a pure displacement (cross-correlation r = 0.9997), identical on all six
  windows.

**Re-scored table** (per-window final PIT-MAE vs RAW telemetry,
`--v2-rounds 1` = the WP6/WP12 real-data default). *pre* = the frozen
`@268c7660` build, *post* = `@54849c13`, *fixed-seed* = post audio + post
labels run with the pre-build's blind seeds (the seeding control, below).

| window | regime | baseline pre | **baseline post** | v2/v3 pre | **v2/v3 post** | baseline fixed-seed | v2/v3 fixed-seed |
|---|---|---|---|---|---|---|---|
| nosource w00 | cruise¹ | 3.262 | **3.262** | 3.197 | **3.197** | — | — |
| nosource w01 | cruise | 1.023 | **1.023** | 1.472 | **1.472** | — | — |
| nosource w02 | cruise | 1.019 | **1.019** | 1.320 | **1.320** | — | — |
| speech-low w00 | cruise¹ | 2.862 | **2.862** | 2.952 | **2.952** | — | — |
| speech-low w01 | cruise | 1.049 | **1.049** | 1.333 | **1.333** | — | — |
| speech-low w02 | cruise | 1.006 | **1.006** | 1.404 | **1.404** | — | — |
| whitenoise-low w00 | cruise¹ | 4.063 | **4.063** | 4.032 | **4.032** | — | — |
| whitenoise-low w01 | cruise | 0.993 | **0.993** | 1.531 | **1.531** | — | — |
| whitenoise-low w02 | cruise | 1.151 | **1.151** | 1.561 | **1.561** | — | — |
| FLY124 w00 | warmup | 3.988 | **5.166** | 4.973 | **5.701** | 5.166 | 5.701 |
| FLY124 w01 | warmup | 2.057 | **2.122** | 4.565 | **4.824** | 1.797 | 4.647 |
| FLY124 w02 | cruise | 5.070 | **4.995** | 5.214 | **5.155** | 4.995 | 5.155 |
| FLY124 w03 | cruise | 1.148 | **7.273** | 1.130 | **7.016** | **0.986** | **1.047** |
| FLY124 w04 | cruise | 0.983 | **0.747** | 1.087 | **0.843** | 0.747 | 0.843 |
| FLY124 w05 | cruise | 3.375 | **2.954** | 3.378 | **3.030** | 2.954 | 3.030 |

| pool | baseline pre | **baseline post** | v2/v3 pre | **v2/v3 post** | baseline fixed-seed | v2/v3 fixed-seed |
|---|---|---|---|---|---|---|
| dregon_cruise (9) | 1.825 | **1.825** | 2.089 | **2.089** | — | — |
| fly124_cruise (4) | 2.644 | **3.992** | 2.702 | **4.011** | **2.421** | 2.519 |
| fly124_warmup (2) | 3.022 | **3.644** | 4.769 | **5.262** | 3.482 | 5.174 |
| fly124_all (6) | 2.770 | **3.876** | 3.391 | **4.428** | 2.774 | 3.404 |

¹ tagged `cruise` by the protocol (window-mean rps 59.8–67.6 ≥ 45), but these are each recording's takeoff-ramp window — the `dregon_ramp` lab window is `nosource w00`.

`refine_v2` and `refine_v3` are **identical on every one of the 15 real
windows** — M3 no-ops on all of them, as WP7/WP12 already recorded.

Cross-window-seeded `refine_v3` (`--m3-pool` / `--m3-ref`, OFF by default and
off for every headline number above), FLY124 cruise only:

| window | v3 post | v3 post + cross-window | v3 fixed-seed + cross-window |
|---|---|---|---|
| w02 | 5.155 | 5.155 | 5.155 |
| w03 | 7.016 | 7.016 | 1.047 |
| w04 | 0.843 | 0.843 | 0.843 |
| w05 | 3.030 | 3.030 | **0.972** |
| **pooled (4)** | 4.011 | 4.011 | **2.004** |

**Harness validation.** The *pre* column reproduces every recorded historical
number exactly: baseline `nosource w00` 3.262 and `FLY124 w03` 1.148 (the
`BASELINE_REF` trace references), refine_v2/v3 `FLY124 w03` 1.130 / `w04`
1.087 / `w05` 3.378 (WP12's 1.130 / 1.087 / 3.380), and `nosource w00`
3.197 (WP8's real-pair mean 2.163 = ½(3.197 + 1.130)). The re-score is
therefore the same measurement, on different data.

#### The FLY124 regression is a blind-SEEDING flip, not the labels
*(FIXED — see "Both fixes in" below; w03 now seeds 82.65 and scores 0.978.)*

The 86 ms realignment changes **one** window's blind seed, and it is enough
to swamp the pooled number:

| window | pre-build seed | post-build seed |
|---|---|---|
| w00 | 63.2, 63.3, 72.5, 82.5 | identical |
| w01 | 63.2, 63.3, 72.65, **82.8** | 63.2, 63.3, 72.65, **82.75** |
| w02 | 75.25, 75.35, 93.75, 93.85 | identical |
| **w03** | 74.2, 74.3, **82.7**, 92.35 | **54.45**, 74.2, 74.3, 92.35 |
| w04 | 75.0, 75.1, 82.25, 91.45 | identical |
| w05 | 75.2, 75.3, 91.8, 91.9 | identical |

w03's 82.7 rev/s base is the comb-invisible 4th rotor that WP8 records as
recoverable *only* by arm R's residual re-scan; on the realigned audio the
re-scan takes a spurious 54.45 instead. Running the corrected audio and
corrected labels with the pre-build's seed (the "fixed-seed" column) puts
w03 at **0.986 / 1.047 — its best value ever**, better than the 1.148 /
1.130 it scored pre-recalibration.

The failure also propagates: with w03 mis-tracked, the ~82 rev/s base loses
the second window vote `cross_window_pool` requires, so the WP12 repair that
recovered w05 no longer fires. Restore w03's seed and w05 goes
**3.030 → 0.972** (WP12's old-label recovery was 3.380 → 1.147).

#### Label fix vs estimator work — the attribution

Cleanly separable, on FLY124 cruise pooled (baseline chain):

| effect | Δ |
|---|---|
| pre-recalibration | 2.644 |
| **the label + alignment correction alone** (fixed-seed) | **2.421 (−0.223, −8.4 %)** |
| the w03 blind-seeding flip it happens to trigger | +1.571 |
| as measured today | 3.992 |

So the recalibration itself **improves** FLY124: at cruise every
seed-stable window gets better (w02 5.070 → 4.995, w04 0.983 → 0.747, w05
3.375 → 2.954), consistent with labels that are now ~0.56 rev/s closer to
the truth and 31–62 ms better timed. The pooled regression is entirely the
seeding lottery on one window. Refinement shows the same split: 2.702 →
2.519 (−6.8 %) from the labels, 4.011 as measured.

Honest limit of the attribution: the label change and the 86 ms audio shift
come from the *same two constants*, so this experiment cannot separate them
from each other — only both-together from the seeding flip. Warm-up is the
one place where "both-together" is a loss: w00 goes 3.988 → 5.166 at an
*unchanged* seed, so on that window the corrected protocol genuinely scores
worse and no seed effect explains it. Not chased.

#### Where the blind bar now sits, and whether we beat it

- **Blind VK on FLY124 cruise, fixed raw protocol: 3.992** as measured
  (2.421 with the seeding flip removed). Pre-recalibration it was 2.644.
- **Blind VK on DREGON cruise: 1.825**, unchanged (bit-identical inputs) and
  consistent with the 1.807 recorded for the `blind_fullrange` v2 arm
  without the closing `pi_kalman` pass.
- **The refinement chain does NOT beat the blind bar** on either pool at the
  default settings: DREGON 2.089 vs 1.825, FLY124 cruise 4.011 vs 3.992
  (2.519 vs 2.421 fixed-seed). On DREGON it loses 7 of 9 windows — it wins
  only 2 of 9 windows — two of the three takeoff-ramp windows — and
  degrades every steady one by 0.3–0.5 rev/s. The WP4–WP12 chain was tuned on synthetic batteries plus **two**
  real windows; on the full 15-window protocol that generalization does not
  hold. *(Superseded below: the 0.3–0.5 rev/s is entirely M2, and the
  `--m2-gate move` arm removes it — DREGON 1.819, FLY124 cruise 2.380.)*
- **The one arm that does beat it** is `refine_v3` with cross-window seeding
  on FLY124 cruise: **2.004 vs the 2.421 blind bar** (−17 %), and only when
  the blind seeder finds the 4th rotor. It is off by default, it needs the
  three WP12 assumptions (same recording, cruise, stable sorted-rank
  identity), and it is worth exactly nothing when seeding fails.

**Not re-measured, still stale:** the **1.027** FLY124 bar quoted in this
document's Goal section comes from a *different* protocol — the 20 s
`vk_blind_sweep` r6 cruise clips (`docs/vk-order-tracking-design.md` § 7.5),
not the fixed raw protocol — and was not re-run here. Treat it as
pre-recalibration until `vk_blind_sweep` is re-run on the corrected loader.

### Both fixes in (2026-07-31) — the seed flip repaired, M2 gated

> This supersedes the two paragraphs above about the FLY124 regression and
> about the chain not beating the bar. Driver `scripts/refine_gate_probe.py`;
> jobs **`python-74f6e0`** (diagnostics), **`python-5d65f0`** (first re-score,
> which exposed a second seeder defect), **`python-cf3bbc`** (this table).
> Mechanism: `docs/experiments/rps-refine-precision.md` § WP15.

**Fix 1 — the w03 blind seed was a coin flip, twice over.** arm R's residual
re-scan offered 54.45 and 82.7 **tied to within 0.06 % of score**; the 86 ms
realignment moved both by ~1.5 % and reversed the order. Nothing was on a
knife edge except the *ranking*. Two guards, both missing by omission:
`SeedConfig.r_span_max = 1.45` (an accepted residual comb may not stretch the
seed set's max/min past a quadrotor's rotor band — 54.45 beside 92.35 spans
1.696 against the ~1.25× real rotors sit within) and **mutual dedup of the
accepted residual candidates** at the `dedup_rps = 2.5` the *used* bases
already enforce (with 54.45 gone, 80.90 stopped being filtered as its 3:2
alias, and arm R took two bases 1.75 rev/s apart — one rotor, two slots).
Verified at the seed stage on all 15 windows × both protocol builds: **29 of
30 seed sets bit-identical**, including the old build's w03 reproducing its
historical `[74.2, 74.3, 82.70, 92.35]`. The one change is the new build's
w03, `[54.45, 74.2, 74.3, 92.35]` → `[74.2, 74.3, 82.65, 92.35]`.

**Fix 2 — M2 declines when its own premise fails** (`--m2-gate move`). The
steady-window regression is **entirely M2**: M1 no-ops on 8 of 9 DREGON
windows (its surface-quality gate skips all four rotors), while M2-solo costs
0.29–0.55 rev/s on every steady DREGON window and up to +2.70 on FLY124
warm-up. Mechanism: DREGON cruise is two tight twin pairs, so the siblings'
reconstruction explains only ~⅓ of the RMS and `PK_WIDE`'s 12 Hz first band
lets a single-track solve slide onto the twin's comb — one or two rotors pick
up a 1.2–2.4 rev/s negative bias. Three truth-free per-rotor rules (no
residual → skip; mean move > 0.5 rev/s → reject; landing on a sibling's comb →
reject) reject 54 of 60 proposals, i.e. the chain becomes ≈ its own M1 output.

**Re-scored, `--v2-rounds 1`** (*pre* = the `python-764ca5` columns above):

| pool | pre baseline | pre v2/v3 | **baseline** | **v2/v3** | **v2/v3 gated** | v3 + cross-window | v3 gated + cross-window |
|---|---|---|---|---|---|---|---|
| dregon_cruise (9) | 1.825 | 2.089 | **1.825** | 2.089 | **1.819** | — | — |
| fly124_cruise (4) | 3.992 | 4.011 | **2.418** | 2.510 | **2.380** | 1.995 | **1.864** |
| fly124_warmup (2) | 3.644 | 5.262 | 3.644 | 5.262 | 3.605 | — | — |
| all 15 | 2.646 | 3.025 | **2.226** | 2.624 | **2.207** | — | — |

Per-window, the two that moved: **FLY124 w03 7.273 → 0.978** (baseline;
1.012 refine, 1.038 gated) — better than the 1.148 it scored *before* the
recalibration, which is the label correction showing through — and **w05's
WP12 cross-window repair fires again, 3.030 → 0.972** (0.901 gated), because
w03 once more supplies the second vote for the ~82 base. The recovered blind
baseline reproduces the seed-swap control (2.418 vs 2.421).

**Against the three bars:**

- **DREGON cruise, bar 1.825:** gated chain **1.819**. A tie (−0.3 %), not a
  win — but the 0.264 rev/s regression is gone, and the chain is now ≤ the
  baseline on 8 of 9 windows with a worst case of +0.019.
- **FLY124 cruise, bar 2.421 (fixed-seed reference):** blind baseline
  re-measures at **2.418**, the gated chain at **2.380** (−1.6 %), and the
  cross-window-seeded gated `refine_v3` at **1.864 (−23 %)**.
- **Pooled all-15:** 2.226 → **2.207**.

So: the seed fix is worth **1.574 rev/s** on the FLY124 cruise pool and the
gate turns a −0.26/−0.09 rev/s regression into a tie-or-small-win. The only
arm that *beats* a bar by a real margin is still cross-window-seeded
`refine_v3` on FLY124 cruise, and it still needs the three WP12 assumptions
(same recording, cruise, stable sorted-rank identity) plus a blind seeder that
finds the 4th rotor. On DREGON the honest verdict is unchanged: the
refinement chain does not add precision over blind VK — the gate's achievement
is that it no longer subtracts any.

The gate is **off by default** (`M2_GATE = "off"`), because on the 13-window
synthetic battery — where the sibling reconstruction is clean and M2 does pay
— it costs ~6 % (refine_v2 2.196 → 2.318, refine_v3 1.767 → 1.872). Treat
`--m2-gate move` as a real-data switch, exactly like `--v2-rounds 1`.

### Post-recalibration neural re-score (2026-08-03/04, jobs `bash-3efa14` + `bash-ff5033`)

The neural rows of the 2026-07-29 scoreboard below were scored against the
pre-recalibration pin and were stale. This table re-scores them on
`beatvk-valid-raw@54849c13ed3a` with the SAME scorer and settings
(`scripts/beatvk_eval.py --pred model:<key>`, arm `none`, chmean stitched
inference). **Sanity check passed: every DREGON column reproduces its
2026-07-29 value exactly** (DREGON inputs are bit-identical between pins),
so each delta below is the FLY124 label + 86 ms audio correction only.

| row (registry key) | dregon_cruise | fly124_cruise | fly124 stale | Δ |
|---|---|---|---|---|
| **CKLA phase-only 4 s** (`ckla_phaseonly_4s_best`) | **2.546** | **1.286** | 1.077 | +0.209 |
| CKLA phase-only 8 s (`ckla_phaseonly_8s_best`) | 3.102 | **1.218** | 1.154 | +0.064 |
| CKLA phase-only 1 s (`ckla_phaseonly_best`) | 3.224 | 1.428 | 1.282 | +0.146 |
| CKLA mean fs_v2 (`ckla_pnoise_fs_v2_best`) | 3.066 | 1.511 | 1.403 | +0.108 |
| KLA fs_v2 (`ckla_norot_fs_v2_best`) | 3.129 | 1.813 | 1.517 | +0.296 |
| scv2 fs_v2 (`scv2_fs_v2_best`) | 2.856 | 2.385 | 2.407 | −0.022 |
| uni_gru128 fs_v2 (`unigru128_fs_v2_best`) | 4.247 | 2.255 | 2.267 | −0.012 |
| transformer fs_v2 (`g2_if_freqscale_v2_best`) | 3.385 | 3.473 | 2.923 | +0.550 |

**Hybrid refresh** (job `bash-ff5033`): `beatvk_vk_arms --arms neural_traj`
(seeds `ckla_phaseonly_best`) on FLY124 w03+w04, then
`pi_kalman_protocol --pair-mode joint` — the smoke-window pair of the
2026-07-29 "neural_traj + pi_kalman joint" row. Pooled over w03+w04:
neural track **0.869**, after pi_kalman **0.641** (was 1.016 → 0.859).
The corrected labels improve BOTH sides of that comparison.

Reading:

- CKLA phase-only 4 s stays the best pooled non-oracle row on both pools.
  The 8 s arm is now the best FLY124 pure-neural number (1.218).
- The ranking is unchanged except the fs_v2 transformer, whose real +0.550
  regression drops it below scv2 and uni_gru128 on FLY124.
- The extra `ckla_norot_fs_v2_s2_best` run (3.562 / 1.787) confirms the
  scoreboard "KLA (fs_v2)" row is the `_best` checkpoint (3.129 matches).
- `scripts/rps_predictor_vk_eval.py`'s hardcoded `CLIPS` table (michaels
  GT means/stds, ~L232–246) is NOT used by `beatvk_eval.py` — windows and
  regimes come from the dataset manifest — so no constant fix was needed
  here. That table is still stale for its own 37-clip protocol.
- Raw reports: `results/beatvk_eval/rescore54_<key>/report.json` and
  `results/pi_kalman_protocol/neural_smoke_rescore/report.json` in the two
  jobs' `omnirun pull` outputs.

### Full scoreboard on the fixed raw protocol (2026-07-29) — PRE-RECALIBRATION

> Kept for history. Every `fly124_*` column below was scored against
> `beatvk-valid-raw@268c766052cb`, i.e. the old FLY124 labels *and* the old
> audio alignment; see the re-scored tables above for what these become.
> DREGON columns are unaffected and remain current. **The neural rows
> (including the 2026-07-30 8 s addition and the neural_traj + pi_kalman
> 0.859) are SUPERSEDED by the 2026-08-03/04 re-score above.** The VK rows
> were re-scored 2026-07-31 (also above).

`beatvk-valid-raw@268c766052cb`, 15 windows, per-window PIT-MAE vs RAW
telemetry, pooled (arm `none`). Steady = DREGON w1/w2 ×3 + FLY124 w3–5
(excludes each recording's ramp window + FLY124 warmup).

| row | dregon_cruise | fly124_cruise | steady DREGON | steady FLY124 |
|---|---|---|---|---|
| **CKLA phase-only 4 s** | **2.546** | **1.077** | 2.15 | **0.74** |
| scv2 (fs_v2) | 2.856 | 2.407 | | |
| CKLA mean (fs_v2) | 3.065 | 1.403 | | |
| KLA (fs_v2) | 3.129 | 1.517 | | |
| CKLA phase-only 1 s | 3.224 | 1.282 | | |
| transformer (fs_v2) | 3.384 | 2.923 | | |
| uni_gru128 (fs_v2) | 4.247 | 2.267 | | |
| VK blind (baseline / R / KR seeds) | 6.80–6.82 | 2.77–3.89 | **1.03** | 1.91 |
| VK neural-traj seeded | 2.658 | 1.403 | | |
| VK neural-bases seeded | 7.298 | 1.586 | | |
| VK telemetry-init (oracle) | 0.851 | 0.784 | 0.87 | 0.70 |

**2026-07-29 additions (VK-improvement program)**:

| row | dregon_cruise | fly124_cruise |
|---|---|---|
| VK blind_fullrange v2 (ramp-capable blind) | **1.807** | 2.699 |
| neural_traj + pi_kalman joint (smoke windows) | flat | **0.859** |

blind_fullrange = coarse full-range Viterbi + BPF octave check + rumble-band
energy bridge + DP trust gates (df231a0): every ramp/warmup window 15–36 →
2.9–4.0, steady windows exactly blind_KR. pi_kalman = phase-increment ML
frequency tracker (per-harmonic random-walk model): FLY124 gains −0.14/−0.16;
DREGON blocked by the DISPLACED-COMB property (low harmonics k=2–13 sit
0.3–0.5 rev/s below the mechanical comb in translating flight — 4-probe
verified, estimator exonerated, hover 3–4× weaker; audio-locked refiners
cannot beat ~0.6 vs raw telemetry there).

**2026-07-30 additions**:

| row | dregon_cruise | fly124_cruise |
|---|---|---|
| CKLA phase-only 8 s (from scratch, DIVERGED) | 3.102 | 1.154 |

- The from-scratch 8 s arm failed to train: best val at epoch ~1
  (best_mse 44.5), monotonic degradation after, final val diverged. An
  optimization failure (lr 1e-3 at 2 chunks/step), not an architectural
  negative — retried as `ckla_phaseonly_8s_ft` (warm-start from the 4 s
  best, lr 2e-4).
- Converged 1 s phase-only (`ckla_phonly_long`, 37-clip vk_eval protocol):
  3.369/1.351 — no better than its wall-capped checkpoint. Convergence was
  not the constraint; context length is.
- **S3b/S3c mechanism attribution landed**
  (`results/vk_phase_validation_decomp`): single motors lock 0.72–0.88
  (k=1–2, iter_warp) but **four motors running simultaneously on the
  static bench already collapse lock to 0.02–0.09** (staggered setpoints
  included; iter_warp identical to init — no capture). Hover (S3c) and
  free flight (S4) sit at the same floor. The coherence collapse of
  ceiling layer (1) is therefore **multi-rotor mutual interference**, not
  per-rotor aero noise (trackable alone) and not translation (which only
  adds the displaced-comb bias on top).

Findings:
1. **Blind VK's headline number was a mid-flight-segment artifact**: it
   scores ~1.0 on steady cruise windows (consistent with 0.688-vs-smoothed
   + the raw jitter floor) but fails catastrophically on every ramp window
   (15–23 MAE) and rails at the scan floor on warmup (33–36) — pooled
   full-coverage 6.8. Neural seeding degrades gracefully instead (max 6.9).
2. **CKLA-4s is the best non-oracle row on both pooled cruise metrics**,
   and on steady FLY124 windows it MATCHES the telemetry-init oracle
   (0.74 vs 0.70) while beating blind VK (1.91).
3. The remaining VK edge is steady-DREGON only: blind 1.03 vs neural 2.15
   (oracle 0.87) — the anchor-miss + fluctuation-tracking gap.
4. Longer training windows are the strongest neural lever found (1 s → 4 s:
   3.22/1.28 → 2.55/1.08; anchor collapse half-fixed); 8 s arm training.

## Conclusion (VK-improvement program, 2026-07-29)

Both arms of the program's disjunction were achieved in their respective
domains:

**Much better performance** (fixed raw protocol): blind VK pooled DREGON
6.80 → **1.81** (blind_fullrange: coarse full-range Viterbi + BPF octave
check + rumble-band energy bridge + DP trust gates; all ramp/warmup
catastrophes 15–36 → 2.9–4.0, steady windows bit-identical to blind_KR).
FLY124 refinement: neural tracks 1.016 → **0.859** (pi_kalman joint).

**Definitive ceiling for iterative phase optimization on real free-flight
data**, four-layered: (1) per-harmonic phase coherence at telemetry-truth
is ~absent on free-flight audio (lock ≈0.1 @k1–2, ≈0.03 @k≥5; single
motors reach 0.24-0.67 — the method chain is validated there);
(2) coherent array combination cannot restore it (self-steered upper
bound ≈0.10 @k1–2; delay-and-sum no gain); (3) the strong low harmonics
that DO exist are DISPLACED 0.3–0.5 rev/s below the mechanical comb in
translating flight (4-probe verified, estimator exonerated, hover 3–4×
weaker) — audio-locked refiners are charged this bias by any
telemetry-referenced metric; (4) decoherence budgets measured (k≤5
coherent, τ_k ≈ 0.4–1.7 s at k=8–40 on motors/flight) — these bound
coherent integration for any future method. The phase-increment ML
tracker (pi_kalman) is the best-in-class refiner this ceiling admits:
only method capturing under noise on synthetic, best lock on motors,
FLY124 gains real, DREGON blocked by (3) not by estimation.

Remaining open (nice-to-have, running): S3b/S3c mechanism attribution
(masking vs aero vs translation), CRB tie-in from measured budgets.
