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

### Flagship: blind init + peeled alternation (2026-08-04) — THE DECLARED FLAGSHIP METHOD

> Runner: `scripts/beatvk_flagship.py` (commit `9c2a17f`). Jobs:
> **`flagship-main-9b8ac7`** (main), **`flagship-2xwin-5647df`** (2xwin init
> variant), both uni-cpu, pin `beatvk-valid-raw@54849c13ed3a`. Scorer:
> `beatvk_eval.score_recording` (the frozen scorer) on every row.

**The method** (fully blind — no neural model, no telemetry):

1. **Init** — the `blind_fullrange` v2 chain (blind_KR seeds + BPF octave
   check + coarse full-range frame-Viterbi + energy bridge + DP trust gates
   + vit2dsp ladder, stage guard on).
2. **Peeled alternation**, per application: solve the coherent VK envelopes
   at the current track (bw 1 Hz, k <= 40), give each rotor the audio minus
   the OTHER rotors' comb reconstructions (twin pairs get audio minus the
   non-pair rotors), then one full `pi_kalman` pass (pair_mode joint,
   n_iter 3, band 6 Hz) on the peeled residuals. Iterate to plateau.

**Leaderboard** (pooled window PIT-MAE, rev/s; `ramp` = the three DREGON
w00 takeoff windows, `steady` = DREGON w01/w02 x3):

| row | dregon_cruise | fly124_cruise | fly124_warmup | dregon_ramp | dregon_steady | all 15 |
|---|---|---|---|---|---|---|
| blind_fullrange (init only) | **1.807** | 2.515 | 3.607 | 3.362 | **1.030** | 2.236 |
| + naive pi_kalman x1 | 1.831 | 2.339 | 3.648 | 3.365 | 1.064 | 2.209 |
| + FLAGSHIP peeled x2 | 1.849 | 2.274 | 3.648 | 3.386 | 1.081 | **2.202** |
| + FLAGSHIP peeled x4 | 1.865 | **2.256** | 3.679 | 3.383 | 1.105 | 2.211 |

Full peeled curve (fly124_cruise): 2.515 → 2.345 → 2.274 → 2.263 → 2.256 →
2.258; naive: → 2.339 → 2.319 → 2.293 → 2.281 → 2.285. **Plateau: x2 on the
all-15 pool (2.202), x4 on fly124_cruise** — the alternation plateaus
instead of degrading, and beats naive at every application count on FLY124.

**Sanity gates:**

- The init row reproduces the recorded blind_fullrange numbers **exactly**
  on DREGON: pooled 1.807, per-window bit-level matches (nosource 3.2475 /
  1.0233 / 1.0002, …). FLY124 pooled is 2.515, NOT the recorded 2.699 —
  that number was the pre-recalibration pin with the pre-fix seeder; on the
  corrected pin with the WP15 seed guards, w03 seeds `[74.2, 74.3, 82.65,
  92.35]` and scores 1.168 at init.
- **Peel energy flags** (subtraction must remove energy; flagged, not
  averaged over): every ramp/warmup w00 window fails the gate on most
  applications — `nosource w00` (resid ratio up to 4.96), `speech-low w00`
  (up to 4.61), `whitenoise-low w00` (up to 14.7), `FLY124 w00` (up to
  2.28). On a non-locked ramp track the peel is mis-phased and injects
  energy. All cruise/steady windows pass on every application.

**Per-window movers** (init → peeled x4): FLY124 w03 **1.168 → 0.772**, w04
**0.939 → 0.513** (the twin-pair interference windows — the peel gain), w02
4.990 → 4.741. Everything DREGON is flat-to-slightly-worse (steady 1.030 →
1.105): pi_kalman still cannot add precision over the ladder there
(displaced-comb + raw-jitter floor), and the flagship's honest DREGON value
is its init. Read: **the flagship = blind_fullrange + peeled pi_kalman is
the best fully-blind FLY124-cruise row ever measured (2.256), at an
unchanged DREGON bar (1.807 at init; report init for DREGON).**

#### 2xwin init variant (`blind_fullrange_2xwin`)

Coarse-DP STFT window and hop doubled (4096/1024 — 2x finer in frequency,
2x coarser in time), DP transition penalty gamma halved (keeps the penalty
per rev/s of path total-variation constant relative to the per-second
evidence at the 2x hop). Frame-count smoothing spans double in seconds
(recorded, not adapted).

| row | dregon_cruise | fly124_cruise | fly124_warmup | dregon_ramp | dregon_steady | all 15 |
|---|---|---|---|---|---|---|
| 2xwin init only | 3.523 | 2.475 | **3.140** | 8.142 | 1.213 | 3.192 |
| 2xwin + peeled x4 | 3.614 | 2.262 | 3.334 | 8.239 | 1.301 | 3.216 |

Per-window verdict (only 7 of 15 windows differ from the original arm —
the trust gates reduce the rest to the identical constant init):

- **Wins, FLY124 only**: w00 5.091 → 4.723, w01 2.123 → 1.557 (warmup —
  the finer bins resolve the dense low-rev/s lines), w02 4.990 → 4.830.
- **Losses, every DREGON window where the coarse path engages**: ramps
  nosource w00 3.247 → 4.324, whitenoise w00 4.023 → 4.251, speech w00
  2.816 → **15.850** (catastrophic — the 2x-coarser hop + doubled
  bridge smoothing mistimes the takeoff), and one STEADY window,
  nosource w02 1.000 → **2.101**: at the 2x hop the DP path wobble grows
  past the steady gate (span 12 > 8), so the gate fails to fire and the
  coarse path corrupts a good constant init.
- Expected steady/twin improvement did NOT appear: on steady windows the
  coarse stage is trust-gated OFF, so the finer frequency resolution never
  engages there.

Verdict: **not a replacement init**. As a hybrid it is only attractive
gated to FLY124-warmup-like windows; on DREGON the ramp machinery breaks
at the 2x hop, exactly the flagged risk.

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

## DREGON-degradation discriminators (2026-08-04, three-arm study)

Question: why does every pi_kalman variant slightly worsen dregon_cruise
(blind init 1.807 → 1.83–1.87) while clearly helping FLY124. Three arms,
predictions registered in advance; runner + raw results in the session
scratchpad (`dregon_arms/{armA,armB,armC,armC2}.json`); instrumentation
validated (raw-GT rescore reproduces the flagship report to <2e-3).

**Arm A — high-k anchor** (k floor 16, `k_caps=(16,28,40)`, `band_hz=12`,
`off_comb_hz=25`, joint twin path off): removes ~95% of the steady
degradation (pooled steady +0.034 → +0.002) and beats init on the two
worst windows (nosource w1 1.104→1.005, whitenoise w1 1.044→0.949 at ×1).
Never improves below init pooled — no recoverable error beyond the
displaced-set penalty.

**Arm B — rescore vs low-passed GT** (butter-4 filtfilt, 1.5/0.8/0.4 Hz):
the init→refined gap GROWS under smoothing (+0.024 raw → +0.047 @0.4 Hz,
every arm) — the added error is low-frequency bias, not under-tracked
fast jitter.

**Arm C — hover** (`hovering_nosource_room2`, 3 ad-hoc 16 s windows,
command-GT caveat): standard refinement NEUTRAL (w0 +0.004 vs cruise
+0.03–0.09; blind init failed capture on w1/w2 — octave-confused seed on
the 3–4× weaker hover comb — direction read from w0 + the probe).
Capture-safe telemetry+1.0 offset probe, 6 passes: on cruise the track
walks down 0.06–0.1 rev/s per pass and keeps walking PAST its MAE optimum
(bottom at ×4, rising after) toward a sub-telemetry attractor; on hover
the pull is ~5× weaker and MAE never turns up.

**Verdicts**: T3 (smoother bandwidth) REFUTED three ways (gap grows under
smoothed GT; identical RTS with high-k anchor does not degrade; neutral on
hover). T1 (self-consistent displaced low-k lock) SUPPORTED as the
mechanism — the loss lives entirely in the k≤13 displaced set, and
iteration compounds it. T2 premise CONFIRMED (displacement is real,
translation-linked physics; nothing telemetry-referenced left to recover)
but its "nothing to fix" conclusion REFUTED — the penalty is avoidable by
not measuring the displaced set.

**Implied final-form fix (not yet implemented)**: displacement-aware
harmonic admission in pi_kalman — floor/down-weight k<16 (or gate low-k
on disagreement between the k<16 and k≥16 comb-implied rates), widen the
band to ≥10–12 Hz so on-grid high-k lines stay capturable, keep the joint
twin path off when its harmonics fall in the displaced set. Converts the
stage from −0.024 harm to neutral/slightly-positive on DREGON; FLY124
machinery untouched. Realistic ceiling: DREGON cruise does not improve
below the blind init this way — that floor is init trajectory-shape error
plus GT jitter.

## Bandwidth-and-admission revision — frozen-protocol test (2026-08-05)

Question: does the revised pi_kalman (k-scaled bands `band_k = k*B0`,
posterior-annealed bandwidth over iterations, displacement-aware low-k
admission, collision-aware probes; tracker options commit 0e21d6e, wiring
03318de) beat the protocol configuration on the frozen validation set.
Jobs `bandadm-ladder-7fb2e4` + `bandadm-chans-afb8db` (uni-cpu, HEAD
03318de, dataset pin 54849c13ed3a). Registered success criteria: FLY124
cruise <= flagship; DREGON degradation removed (neutral vs init 1.807);
ramps unchanged. Synthetic tiers had shown large gains (k_scaled clean
0.091 -> 0.031, twins 0.081 -> 0.027).

**Protocol ladder (blind init, PIT-MAE; init: all 2.236, dregon_cruise
1.807, fly124_cruise 2.515)**:

| variant | best all | fly124_cruise best | dregon_cruise x3 naive |
|---|---|---|---|
| protocol | 2.202 (peeled x3) | 2.256 (peeled x4) | 1.854 (ratchet) |
| k_scaled | 2.235 (peeled x1) | 2.501 (peeled x4) | 1.808 (neutral) |
| k_anneal | 2.236 | 2.504 | 1.808 (neutral) |
| full (+admission+probes) | 2.232 | 2.502 | 1.806 (neutral) |

**Verdict: the revision FAILS criterion 1 and is REJECTED for the final
form.** All three revised variants make the iteration nearly inert on
real data: they remove the DREGON ratchet (criterion 2 PASS, 1.86 ->
1.81) but also remove the FLY124 twin gain (2.26 -> 2.50, criterion 1
FAIL). Ramps unchanged on naive; peeled drifts +0.06 by x4 (criterion 3
marginal PASS). Mechanism: the narrow annealed bands only accept
corrections when the init comb is already near-exact — that is exactly
the synthetic condition (hence the large synthetic gains) and exactly
NOT the FLY124 twin condition, where capture of the 6 Hz-distant true
line is the source of the gain. The peel-energy GUARD already gives the
good half of this trade (init fallback on DREGON-displaced and ramp
windows, FLY124 gains kept, guarded flagship 2.193): the revision is
dominated by the guard and adds nothing.

**Channel ablation (protocol variant, mic subsets seed 0)**: C=8 init
1.81/2.52 (dregon/fly124 cruise); C=4: 10.28/2.56; C=2: 9.77/2.51; C=1:
18.76/5.72. Two findings: (1) the SEED is the channel-hungry stage —
DREGON blind capture collapses without the full 8-mic array, while
FLY124 cruise is essentially intact at C=2; (2) the pi_kalman iteration
itself still helps at C=1-2 on FLY124 (C=2: 2.51 -> 2.26 peeled x3), so
the multi-channel pooling is not what makes the iteration work — this
answers the single-channel question raised in the explainer review. The
C=4 ramp blowup (28.7) is a seed octave failure on that specific subset,
not an iteration effect.

**Final form stays**: blind full-range Viterbi init + guarded peeled
alternation, protocol bands (6 Hz), peel-energy guard. The k-scaled
band option remains available as an opt-in for near-exact-init regimes
(synthetic, telemetry-seeded), where it is strictly better.

## Per-harmonic displacement measured (2026-08-05) — PARTLY WITHDRAWN, see the correction below

Direct measurement of the displaced comb (heterodyne each harmonic around
k*telemetry with the tracker's demod bank, envelope-spectrum ridge ->
shaft-rate offset; twin-collision frames gated out; 15 frozen-protocol
windows; scripts + `displacement.json` + figures in session scratchpad
`displacement/`).

**The displacement is k-graded, not flat**: DREGON k=1-4 sit 0.5-0.6
rev/s below telemetry, k=5-13 near -0.15, k>=16 on-grid (-0.02). Same
profile on steady and ramp controls, so it is not cruise-only. The
0.3-0.5 rev/s figure from the probe study holds only for k<=4. **FLY124
carries the same signature ~3x smaller** (low-k -0.043, high-k -0.002).

~~**Near-optimality on DREGON is REFUTED.** Three-way MAE vs telemetry,
pooled: low-k comb 0.364 / high-k comb 0.086 / flagship blind track
1.854; the high-k comb tracks telemetry to 0.06-0.09 rev/s, 20x below the
blind track, so the remaining blind error is capture/assignment error,
not physics.~~ **WITHDRAWN — this was a search-window artifact. See the
correction section below. The 20x-headroom claim does not stand, and
figure F3 must not be used.**

Consequence for the band debate: the displaced low-k set is a data-model
mismatch (low harmonics do not follow shaft rate) to be handled by
harmonic ADMISSION, not by band choice; algorithm selection must not be
driven by it.

## CORRECTION — the high-k claim was a window artifact (2026-08-05, null controls)

Same-day null controls (`displacement/nullcontrol.md`, `nullcontrol.py`,
figures F4/F5; same 15 windows, same pin). Four controls sharing the
measurement's band, search half-width, collision gate and weighting, with
only the carrier changed.

**The high-k number measures the search window, not the comb.** Running
the identical pipeline at a carrier where NO rotor line can exist
returns the same value:

| DREGON cruise, high k (16-40) | MAE vs telemetry (rev/s) |
|---|---|
| measured, carrier k*g_r(t) | **0.0856** |
| null, off-comb carrier (k+0.5)*g_r(t) | **0.0857** |
| null, telemetry from a different window | 0.0845 |
| analytic, peak uniform in the window | 0.1537 |

Ratio measured/null = 1.00 (FLY124 1.01). The search half-width is
min(1.5k, 8) Hz, i.e. <= 8/k rev/s, so combining ~25 harmonics averages
per-k peak-picks of NOISE down to ~0.086 whether or not a line exists.

> **SUSPENDED 2026-08-05 (same day, user review).** The "DREGON has no
> high-k line" paragraph below is itself very likely a measurement
> artifact and must NOT be cited. The peak-search half-width is
> `min(1.5k, 8)` Hz (`measure_displacement.py`: `SEARCH_REVS=1.5`,
> `SEARCH_HZ_CAP=8.0`), i.e. `min(1.5, 8/k)` rev/s — it SHRINKS as 1/k.
> The DREGON displacement is FLAT in k at ~-0.424 rev/s, so the true line
> leaves the search window at **k = 8/0.424 = 18.9** and is already at
> 74% of the half-width by k=14 — exactly where "nothing above k=14" was
> reported. On FLY124 the displacement is -0.051 rev/s, so its line stays
> inside until k~157, which is why FLY124 "keeps a set to k~20" and
> DREGON does not: **the dataset asymmetry is the bug's signature, not
> physics.** (The demod BAND, `min(3k, 0.45 fs_env)`, is wide and fine;
> only the SEARCH WINDOW is broken.) The high-k NULL EQUALITY still
> stands as measured — but it only shows the on-comb search found noise
> where it looked, which is expected if it looked in the wrong place.
> Re-measurement in flight: two-pass search re-centred on
> `k*(g_r + d_r)`, identically re-centred null, k extended to ~100 (the
> user reports visible comb structure at ~6 kHz, i.e. k~75, far above the
> current 3.2 kHz cap). Until it lands, treat DREGON high-k as UNKNOWN,
> not absent.

**Corroboration, stronger than the null: DREGON has no high-k line.**
Pooled over 9 cruise windows x 4 rotors x 25 harmonics (900 units), only
**3 of 900** harmonics with k>=16 clear 6 dB over their own in-band floor
— FEWER than the off-comb null's 8. Median prominence 1.79 dB on-comb vs
1.73 dB null. The 3-5 dB "SNR" in F2's lower panel is a noise floor's
peak-pick bias. The window-independent pulse-pair estimator does not
rescue the claim: it returns ~0 on the null too (in-band noise is
symmetric), so agreement between the two estimators was never evidence.
DREGON's only clearly usable harmonic is **k=2** (+6.1 dB over null, 89%
of units clear the bar), with a weak group at k=4-13 and nothing above
k=14. FLY124 carries a real set out to k≈20.

**Retracted**: the three-way error decomposition and the "20x headroom /
remaining error is estimator error, not physics" reading; also "the comb
returns to the grid by k>=16" — the high-k offsets are small because the
window is small. Honest statement: above k=16 the DREGON comb is not
measurable by this demodulation, and there is no evidence either way.
FLY124 keeps a trace (14/400 units over the bar, mean offset -0.055
rev/s). **The seed-first neural program keeps its historical
justification (every neural failure was a capture/assignment failure) but
LOSES this quantitative headroom number.**

**What survives and is LARGER than reported.** The low-k displacement is
real and the pooled -0.145 was diluted by units carrying no line (they
contribute noise centred on zero). Discriminated from the null two ways:
the null's signed mean is +0.008 rev/s while the measurement's is -0.124;
and 34 of 432 DREGON low-k units clear the 6 dB bar against the null's 2.
**Restricted to those 34 real lines the offset is -0.424 rev/s**
(pulse-pair -0.231, biased toward zero at low SNR; FLY124 -0.051 over 74
units). So the original 4-probe "0.3-0.5 rev/s" figure was right, and the
dilution explains the discrepancy.

**Consequence for Arm A (high-k anchor) — REINTERPRETATION NEEDED.** Arm
A ran with `k` floor 16 and `band_hz=12`. WP18 already measured 0 usable
DREGON harmonics at B=12, and the prominence map now shows nothing
measurable above k=14 at any band. So "the high-k anchor removes 95% of
the degradation" most likely means **the stage was made nearly inert**,
not that it anchored on an undisplaced high-k set. It is not fully inert
(it beat init on two windows), so this is a hypothesis, not a verdict —
but the T1 mechanism note above must not be read as "the k>=16 comb is
on-grid and usable".

**The wiggle is a real rate deviation, and points at the labels.** The
k=2 ridge visibly wiggles on DREGON but not on FLY124. Four tests agree
it is a genuine shaft-rate deviation from telemetry, not interference:
delta_k is FLAT in k (-0.51,-0.51,-0.53,-0.34 at k=2,4,6,8, where a
fixed-frequency artifact would give -0.51,-0.25,-0.17,-0.13); with the
(k=2,k=4) pair fixed a priori r = +0.65..+0.83 over four windows with
slope brackets excluding 0.5 and bracketing 1.0; the correlation tracks
prominence; and twin beating, rotor permutation and telemetry lag are all
ruled out. As a fraction of rate the constant part is **-0.54% on DREGON**
(per rotor -0.55/-0.34/-0.54/-0.45%) against **-0.063% on FLY124** — the
same defect class as Michael's +0.70% `rps_scale` (WP13/WP14), which was
found and corrected, and never checked on DREGON. At cruise -0.54% is
~0.42 rev/s, twice the assumed 0.2 rev/s label-jitter floor. Follow-up
running (documentation check, static-bench single-motor measurement,
regime ladder) to discriminate: telemetry scale error vs commanded-rather-
than-measured setpoint vs real aeroacoustic displacement.

## Wide-anneal arm + why a fixed-Hz band is the principled choice (2026-08-05)

Fair test of the annealing property the narrow ladder never exercised:
start wide and let the posterior anneal shrink the trust region
(`--band-b0 3.0`, `k_anneal`, job `bandadm-wide-85732e`, HEAD 10e7322).
Prediction registered before the result (scratchpad
`displacement/predictions.md`): FLY124 recovers toward the protocol 2.26;
DREGON degrades to ~1.85.

**Result — worst row of the whole ladder**, every pool degrading
monotonically with iteration (init -> x4): all 2.236 -> 2.494,
dregon_cruise 1.807 -> 2.201, fly124_cruise 2.515 -> 2.526 (best 2.409 at
peeled x1), dregon_steady 1.030 -> 1.497, warmup and ramp both up.
Prediction 1 partly right (wide start restores SOME capture: peeled x1
2.409 beats the narrow arm's 2.509) but it never reaches protocol 2.345
and then diverges. Prediction 2 right and worse than predicted (1.91 at
x1, not neutral, and no plateau). Mechanism: wide bands admit noise and
neighbouring-comb contamination, the pulse-pair estimate walks, and the
posterior anneal then shrinks the trust region around the ALREADY
CORRUPTED estimate — it protects the wrong lock instead of refining the
right one.

**The principled point (independent of the DREGON displacement) — this
was ALREADY MEASURED in WP18** (`rps-refine-precision.md` § WP18, table
of `alpha_raw`). Two facts, both measured, both about the band:

1. **A k-scaled band destroys the high-harmonic precision advantage.**
   At a FIXED band the per-harmonic rate variance v_k falls as k^-2 —
   fitted slope of log(1/v_k) vs log k is 1.97-2.00 (DREGON, identical at
   B=1.5/3/6 Hz) and 1.46-1.51 (Michael's); v_k drops 167x between k=2
   and k=30. Under k-scaled bands the same fit **collapses to ~0** (0.29
   DREGON, -0.26/-0.51 Michael's) with R^2 falling to 0.03-0.58, i.e. no
   power law describes the weight at all and every harmonic becomes
   equally uninformative. Measured exponent SHIFT fixed -> k-scaled:
   **-1.7 (DREGON) / -2.0 (Michael's)** — worse than the naive -1,
   because v_k scales nearer B^3 than B once the `1 - sinc(2 B dt)` band
   factor (pi_kalman's `c_noise`) is included.
2. **Wide bands empty the harmonic set.** Usable harmonics of 30 after
   the twin gate: DREGON **20.5 @ B=1.5, 15 @ B=6, 0 @ B=12** (median min
   rotor split 0.42 rev/s); Michael's 19 / 14.5 / 1. At a fixed band the
   collision radius in rev/s is B/k; a k-scaled band holds it CONSTANT at
   B0 for every k — which is the honest argument FOR k-scaling (uniform
   coverage), and is why the option was built. But at B0=3.0 that
   constant radius is 3 rev/s, seven times DREGON's rotor split, at every
   harmonic: coverage is zero everywhere. The wide-anneal divergence
   above was predictable from this July measurement.

So the trade is real, not one-sided: fixed bands keep the k^2 precision
law but leave LOW harmonics collision-prone (radius B/k is largest
exactly where the displaced set lives); k-scaled bands equalize coverage
but flatten precision to nothing. Fixed wins because the precision law is
worth more than low-k coverage — the low harmonics are the ones we least
want on DREGON anyway.

Separately (and this is a REACH argument, not the WP18 weight argument —
do not conflate them): the band also sets how far the demodulator can
capture a line the init missed. Rate half-width B/(2k) gives the protocol
+-1.5 rev/s at k=2 and +-0.5 at k=6, enough to reach a twin 0.5-1 rev/s
away; k_scaled B0=0.35 gives +-0.175 everywhere and cannot. That explains
the FLY124 result specifically.

| policy | rate half-width at k=2 / 6 / 30 (rev/s) | collision radius (rev/s) |
|---|---|---|
| protocol, fixed 6 Hz | 1.50 / 0.50 / 0.100 | 3.0 / 1.0 / 0.20 |
| k_scaled, B0=0.35 | 0.175 / 0.175 / 0.175 | 0.35 at every k |
| k_scaled, B0=3.0 | 1.50 / 1.50 / 0.469 | 3.0 at every k |

**Verdict, restated on principle rather than on DREGON.** The k-scaled
band family is rejected because it flattens the MEASURED k^2 weight law
(WP18, fact 1) and, at any B0 wide enough to capture, empties the
harmonic set (fact 2) — not because of the DREGON low-k data-model
mismatch, which is an ADMISSION problem (see the displacement section)
handled by the peel-energy guard and, if wanted, by low-k gating.
Caveat: Michael's exponent is 1.5, not 2.0, so its ideal band grows
mildly with k (B ~ k^0.25). Fixed-Hz (k^0) is exactly right for DREGON
and slightly conservative for Michael's; full k-scaling assumes k^1 and
is far off for both. A principled anneal, if revisited, must shrink B in
Hz and keep the near-flat band exponent, not flatten the rate shape.
