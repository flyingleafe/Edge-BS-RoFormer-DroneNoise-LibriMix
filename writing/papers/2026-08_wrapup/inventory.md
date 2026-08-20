# Wrap-up paper — full experiment inventory, 2026-05-20 → 2026-08-20

Decision (supervisor, 2026-08-20): stop the push on the decomposition direction.
Wrap the results we have — neural RPS prediction, noise generation, blind
annotation, telemetry refinement — into one solid paper.

Priority code:
- **P1** — core paper material. The paper's claims stand on these rows.
- **P2** — supporting material: ablations, baselines, appendix, honest caveats.
- **P3** — cut, or one sentence as a lesson. Negative or unfinished results.

Every row names its evidence. Verify numbers against the named doc before they
go into the paper. (The E7–E12 doc conclusions were backfilled from W&B and
the 2026-07-12 report on 2026-08-20; they are current now.)

## A — Neural RPS prediction from audio

| # | Experiment | When | Verdict + headline numbers | Evidence | Prio |
|---|---|---|---|---|---|
| A1 | Architecture sweep, 26 variants (V4-michaels) | 06-19 | Trio established: transformer / uni_gru128 / SimpleConv-v2. Best online: uni_gru128 PIT-MSE 7.33 | `docs/experiments/simpleconv-rps-architecture-search.md`, report `2026-06-19_rps-arch-sweep-v4-michaels` | P1 |
| A2 | Cross-drone generalization via FLY125 | 06-30 | DREGON+FLY125 training drops FLY124 PIT RMSE 7.96 → 1.63 (run 955yy1wv) | report `2026-06-08_channel-generalization-failure`, `docs/experiments/cross-drone-generalization-fly125.md` | P1 |
| A3 | Multichannel generalization failure + PIT loss | 06-08 | Channel identity does not transfer. PIT loss is the fix; `drone_seen` category defined | `docs/experiments/channel-generalization-pit-loss.md` | P2 |
| A4 | Salience-map baselines (multif0, Basic Pitch port) | 06-15 | Best salience model: narrow + super-res multif0, RMSE 6.30 → 4.03. Port verified identical | `docs/experiments/salience-map-rps-tracking.md`, report `2026-06-15` | P2 |
| A5 | E5 time-warp augmentation | 07-10 | Warp noise+RPS together (α ≤ 1.12): transformer val PIT-MSE 11.76 → 8.74 (−26%) | `docs/experiments/e5-timewarp.md` | P1 |
| A6 | Validation-set contamination fix | 07-12 | FLY124 ground warm-up leaked into validation. `min_motor_rps` 30 → 50; reversed the E7–E9 verdicts | report `2026-07-12_full-flight-sim2real-rps` | P1 |
| A7 | E10–E12: training-data coverage story | 07-12 | Punchline: the low-RPS failure was never sim2real. Real full-flight (`min_motor_rps=0`) transformer all-regime PIT-MSE 79.6 vs 338.4 real-cruise-only vs 131.9 sim curriculum. Mean-collapse refuted | `docs/experiments/e10-full-flight.md`, report `2026-07-12` | P1 |
| A7b | E9 hard-mix sim2real (post-fix) | 07-12 | Sim transfer is real: gen-only 17.8–25.4 PIT-MSE with positive R² on the clean valid (was "222, R² −10" on the contaminated one); real fine-tune 11.1–14.1 | `docs/experiments/e9-hard-combined.md` | P1 |
| A8 | E11 silence gate + sim curriculum | 07-12 | Smoothstep emitter gate: a stopped rotor is exactly silent. Synthetic is the only source of true silence; complement, not substitute | same batch doc | P2 |
| A9 | Sim2real transfer of generator-trained predictors | 07-13 | Predictors trained on generated noise fail on real audio; static comb helps the transformer only | report `2026-07-12`, memory `sim2real-rps-transfer-findings` | P2 |
| A10 | CKLA layer campaign | 07-28 (closed) | Phase-only readout best: 2.79 / 1.29 rev/s cruise. Triton scan kernel 230× | `docs/experiments/ckla.md`, report `2026-07-27_ckla-campaign` | P2 |
| A11 | G1–G3 VK-parity levers | 07-24 | Context windows and HCQT front-end refuted; IF channel marginal best 2.481 | `docs/experiments/g1-vk-parity.md`, report `2026-07-24_vk-parity-status` | P3 |
| A12 | Native 16 kHz baselines | 06 | Adapt at 16 kHz; resample only for pretrained weights | memory `native-16khz-baselines` | P3 |

## B — Blind annotation (classical tracking)

| # | Experiment | When | Verdict + headline numbers | Evidence | Prio |
|---|---|---|---|---|---|
| B1 | Blind annotation algorithm (Viterbi seed → ramp handling → per-rotor decoupling → envelope solve → peel iteration) | 07–08 | DREGON cruise 0.68 rev/s, FLY124 cruise 1.03 rev/s against ground truth | deck `2026-08-04_rps-tracking-results-and-paper-plan`, `docs/experiments/beat-vk.md` | P1 |
| B2 | Tachometer validation on the blind corpus | 08-14 | First external validation: 0.97–1.22 rev/s PIT-MAE (room1 cruise). Acceptance gates are arm-dependent | `docs/experiments/blind-corpus-annotation.md` | P1 |
| B3 | Beat-VK precision analysis | 08 | The remaining gap is precision, not coverage | `docs/experiments/beat-vk.md` | P2 |
| B4 | WP18 phase-noise covariance | 08-14 | Optimal harmonic weight 1/v_k ∝ k^2.0 (DREGON) / k^1.5 (FLY); coverage is the binding constraint | `docs/experiments/rps-refine-precision.md` § WP18 | P2 |
| B5 | Kalman harmonic tracker | 07-10 (killed) | Drift robustness refuted at gate K2: degrades faster than lstsq under RPS drift | `docs/experiments/bets/kalman-harmonic-tracker.md` | P3 |
| B6 | SRP rotor localization | 07 | SRP-PHAT floor ~28 cm for 4 coherent rotors; separation is the wall | memory `srp-rotor-localization-floor` | P3 |

## C — Telemetry refinement + label forensics

| # | Experiment | When | Verdict + headline numbers | Evidence | Prio |
|---|---|---|---|---|---|
| C1 | Refinement stages A–D | 07-10 → 08 | Stages validated; trust stage D (phase). Honest floor 0.2 rev/s; VK refine inert (capture range); pi_kalman converges in one pass | report `2026-07-10_rps-refinement`, `docs/experiments/rps-refine-precision.md` | P1 |
| C2 | DREGON telemetry forensics | 08-11 | Label bias decided as a range: quote 0.35–0.85% as a floor. Telemetry has no acoustic lock. Do NOT correct DREGON labels | `docs/experiments/dregon-telemetry-forensics.md`, `dregon-comb-*.md` | P1 |
| C3 | Michael's labels recalibrated | 07-31 | FLY124/125 telemetry has a time lag + 0.7% scale error; `michaels-frames` republished; pre-07-31 derivatives stale | memory `michaels-labels-recalibrated`, `docs/experiments/michaels-recalibration-generators.md` | P1 |
| C4 | Mic-array geometry calibration (stage-0 RTF) | 07-15 | DREGON 180° mic-frame fix + Michael's ring fix; free-field validation | report `2026-07-15_mic-array-geometry-calibration` | P2 |

## D — Neural noise generation

| # | Experiment | When | Verdict + headline numbers | Evidence | Prio |
|---|---|---|---|---|---|
| D1 | Generator label sensitivity | 08-11 | A constant ×0.99458 label bias alone costs −8.6 dB at k50–80; exact labels train flat to k=80. Scale-correct conditioning is required | `docs/experiments/generator-label-sensitivity.md` | P1 |
| D2 | Refined-labels generator verdict | 08-12 | Refined labels fix the k10–49 washout; only the refined arm places lines ON the comb. Sharp k≥25 stays unlearnable from MSSTFT | `docs/experiments/generator-refined-labels.md` | P1 |
| D3 | Per-rotor paradox resolved (checkpoint selection) | 08-12 | MSSTFT erodes combs after epoch 0; mrstft-based selection anti-correlates with comb quality. Comb-aware selection flips the ranking | `docs/experiments/generator-perrotor-dynamics.md` | P1 |
| D4 | E6 harmonic linewidth (jitter injection) | 07-11 | Comb error 9.00 → 7.04 dB (k<10, DREGON in-flight); per-drone σ needed | `docs/experiments/noise-gen-linewidth.md` | P2 |
| D5 | Likelihood objective (Rice/Whittle NLL) | 08-03 (paused) | Fixes the 1.6 dB stochastic underfit of magnitude losses, but the variance term buys out the comb (uncertainty is cheap, accuracy expensive). Aggregate metrics do not see the failure | `docs/experiments/wind-channel-likelihood.md`, GOALS.md open thread | P2 |
| D6 | Residual attribution | 08-14 | Per-rotor broadband attribution is UNIDENTIFIABLE in steady flight (VIF 21–118; non-additive < 1 kHz). Model: per-mic floors + one common driver | `docs/experiments/residual-attribution.md` | P2 |
| D7 | Synthetic RPS (OU modes) + full-flight envelope | 06-30, 07-12 | DREGON-calibrated OU-in-control-mode trajectories; rps^2.5 amplitude scaling | report `2026-06-30_synthetic-rps-trajectories` | P2 |
| D8 | Wind/wake channel | 07–08 (paused) | Physics gate predicts DREGON per-mic floor (Spearman 0.92), but no valid A/B test exists yet — three attempts, three invalidating conditions | GOALS.md open thread, `docs/experiments/wind-channel-likelihood.md` | P3 |
| D9 | Positional harmonic generator (per-rotor sources → observers) | 07 | Built + tested; training never wired | memory `positional-harmonic-noise-gen` | P3 |
| D10 | Corrected-geometry retrains, per-drone generators | 07-17 | Variant retrains on the fixed geometry stream | report `2026-07-17_generator-corrected-geometry-variants` | P3 |
| D11 | Noise-gen loss breakdown | 07 | MultiScaleSTFT already log-dominated; underfit is loss design, not log_weight | memory `noise-gen-loss-breakdown` | P3 |

## E — Decomposition + trajectory-fitness measure (support role)

| # | Experiment | When | Verdict + headline numbers | Evidence | Prio |
|---|---|---|---|---|---|
| E1 | F_VK profiled-residual measure | 08-14 | Oracle −0.596% inside the 6d CI; L-BFGS is the sole recoverer; the global-judge hypothesis is falsified (blind optimization wins wrongly on 3/5 windows) | `docs/experiments/telemetry-fitness.md`, `results/fvk_arms` | P2 |
| E2 | Unified Gaussian model + marginal-likelihood score J | 08-18 | J ranks trajectory quality correctly where F_VK fails: blind result last on 4/5 windows, refined first on 3/5. This is the paper's *evaluation instrument* for annotation quality | `docs/experiments/vk-decomposition.md`, `results/joint_rescore_v4`, deck `2026-08-18` | P1 |
| E3 | v4 decomposition runs of record (DREGON + FLY124/125) | 08-18 | Retained-excess gates at parity or better at k≥10; amplitude targets (H,S) extracted. Open: k1–9 regression on FLY (5–8% vs <1%), carrier moved up to 14 rev/s vs telemetry | `results/vk_decompose_v4*`, memory `vk-decomposition-campaign` | P2 |
| E4 | Performance profile | 08-18 | One J evaluation ≈70 s; full window decomposition 400–2100 s CPU; GPU estimate 30–100× | deck `2026-08-18`, `results/vk_decompose_v4f` | P3 |

## F — Speech enhancement baselines + benchmark integrity

| # | Experiment | When | Verdict + headline numbers | Evidence | Prio |
|---|---|---|---|---|---|
| F1 | SE blind baselines, corrected | 07-22 → 08-03 | Published numbers were wrong (silent zeroed samples, ~+5 dB at 0 dB). Fixed pipeline; ranking unchanged: MP-SENet > TF-GridNet > Edge-BS-RoF ≫ DCUNet | `docs/experiments/f1-se-blind-baselines.md`, report `2026-07-22_se-blind-baselines` | P2 |
| F2 | DCUNet replication + DN-LM leakage | 07-26 | Replication succeeds as a SEEN-NOISE result; the published DN-LM protocol leaks (99.2% noise overlap) and the model ranking inverts without leakage | `docs/experiments/f2-survey-replication.md`, report `2026-07-26_dcunet-generalization` | P2 |
| F3 | RPS-conditioned SE on DREGON | 06–07 | Oracle-RPS SE gain still unproven (goal review 07-20) | `docs/experiments/rps-conditioned-se-dregon.md` | P3 |
| F4 | Diffusion-buffer SE | 06 | Exploratory only | `docs/experiments/diffusion-buffer-se.md` | P3 |

## G — Datasets + reusable assets (paper contributions)

| # | Asset | State | Evidence | Prio |
|---|---|---|---|---|
| G1 | DREGON-LM V4-michaels cleaned valid (pin b6ece43d) + full-envelope split | published | memory `dregon-lm-v4-michaels-valid-cleaned` | P2 |
| G2 | Refined-label sidecars per recording | published, feeds D2/E2 | `src/data_processing/refined_labels/` | P1 |
| G3 | Recalibrated `michaels-frames` (lag + scale) | published @fdef8184 | memory `michaels-labels-recalibrated` | P2 |
| G4 | Online-mixing training framework + policies | in `src/` | `src/data_processing/online_mixing.py` | P2 |
| G5 | Blind-corpus annotation plan + gates (~52 CPU-h) | designed | `docs/experiments/blind-corpus-annotation.md` | P2 |
| G6 | AVQ dataset onboarded | on dload | memory `avq-dataset` | P3 |

## Prioritized reading of the whole window

The paper's spine, per the supervisor's framing:

1. **The loop** (P1 rows): telemetry is imperfect (C2, C3) → refinement fixes it
   (C1) → refined labels measurably improve the generator (D1, D2, D3) → blind
   annotation removes the telemetry requirement (B1, B2) → the
   marginal-likelihood score J validates trajectory quality without ground
   truth (E2) → neural predictors learn from the annotated/augmented data and
   the coverage story explains their failures (A1, A2, A5, A6, A7).
2. **Supporting layer** (P2): baselines (A4, A10), honest negative results
   (D5, D6), data assets (G1–G5), benchmark integrity (F1, F2).
3. **Cut or one-line lessons** (P3): killed bets, refuted levers, unwired
   prototypes.

Counts: 14 P1 rows, 19 P2 rows, 13 P3 rows.
