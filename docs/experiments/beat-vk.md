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

### Full scoreboard on the fixed raw protocol (2026-07-29)

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
