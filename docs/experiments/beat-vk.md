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

(pending)

## Conclusion

(pending)
