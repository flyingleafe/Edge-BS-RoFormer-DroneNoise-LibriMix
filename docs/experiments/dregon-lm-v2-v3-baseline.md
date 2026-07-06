# DREGON-LM V2/V3: Dataset Evolution and Baseline RPS Training

**Status:** done | **Dates:** 2026-06-01 – 2026-06-02

## Motivation

V1 had three weaknesses: train/validation clips drawn from the *same* recordings (memorisation shortcut), clips only 0.82 s (~26 STFT frames, too short for trajectory-following), and a single microphone channel with telemetry-aligned (post-hoc measured) RPS as target. **DREGON-LM-V2** fixed all three: a recording-level split (6 `in_flight_nosource` recordings for train, 2 `free-flight` recordings for valid), 3.0 s clips (~94 STFT frames), all 8 mic channels used as independent noise sources, and commanded (flight-controller setpoint) RPS via `clean_command_spikes`. V2 also added synthetic "motor combo" clips (individual motor recordings summed at constant speed across the same channel) to anchor rotor-output ordering under the permutation-invariant (PIT) loss.

V2 in turn exposed a new problem: clips built from motor combos have near-constant RPS targets, and models trained on V2 learned to exploit this, suppressing predicted variance up to 20× and generalising poorly to real in-flight recordings. Separately, a first V3 generator attempt crashed, surfacing a data-layout bug. **DREGON-LM-V3** followed as a bugfixed, restructured successor: it fixes an axis bug in chunk extraction (command RPS was `(N_samples, N_motors)` but sliced as `(N_motors, N_samples)`, producing 18 MB `rps.npy` files instead of 1 KB), pre-cleans motor commands once per recording rather than per sample, loads LibriSpeech on-demand from all 28.5K files for diversity, and switches to 1 s clips with an explicit SNR range and float32 RPS targets.

## Results

### V2: motor-combo-fraction sweep (BiGRUv2, best architecture)

| Combo% | PIT MSE | Std MSE | Fixed MAE | PIT-aware MAE | Std/PIT gap |
|--------|---------|---------|-----------|---------------|-------------|
| 20% | 71.13 | 73.88 | 4.34 | 3.94 | 3.9% |
| 5% | 65.92 | 87.20 | 5.94 | 4.06 | 24.4% |
| **2.5%** | **56.70** | 86.81 | 6.71 | 4.18 | 34.7% |
| 0% | 117.35 | 153.77 | 8.44 | 5.36 | 23.7% |

Motor combos act as an ordering regularizer for PIT: they teach the model which output slot maps to which physical rotor. Too many (20%) over-regularizes (worse per-rotor prediction); too few (0%) removes the anchor and destabilizes PIT optimization. The sweet spot is **2.5%**: best PIT MSE (56.7) with near-best PIT-aware MAE (4.18), though the model swaps rotor assignments aggressively (35% Std/PIT gap) and needs PIT re-matching at inference. "PIT-aware MAE" (reorder-then-measure) differs from the "fixed-order MAE" the standard `evaluate()` reports, which conflates rotor-identity errors with prediction quality — fixed-order MAE (4.34→8.44) looks like degradation with fewer combos, but PIT-aware MAE (3.94→5.36) shows the real, much smaller effect.

Best V2 checkpoint: BiGRUv2 (1.44M params) at 2.5% combos, 41 epochs, PIT MSE 56.7 (`results/rps_predictor_v4_2.5pct/simple_conv_bigru_v2/best_simple_conv_bigru_v2.pt`). The "V3"/"V4" prefixes in checkpoint names are training-run labels, not dataset versions — all sweep checkpoints were trained on **DREGON-LM-V2** variants.

**Old (V1) vs new (V2) cross-evaluation:** V1-trained checkpoints scored MSE 2.7–5.2 on their own (V1) valid set but collapsed 63–123× on V2 valid (OLD SimpleConv 5.24→331.87; OLD BiGRU-v2 2.67→327.26), proving the V1 validation scores were inflated by memorisation. V2-trained ("V3"-labeled) checkpoints degraded far less (SimpleConv 66.83→148.09, 2.2×; BiGRU-v2 15.26→71.13, 4.7×). Per-channel on V2 valid, V2-trained BiGRU-v2 shows only a 1.09× ch0-vs-ch1-7 gap (66.17 vs 71.92) vs 1.38× for the V1-trained model (247.16 vs 340.12), confirming multi-channel training reduces "home-field" bias toward channel 0.

**In-flight generalisation (from the V2 report):** after correcting an audio/RPS-alignment bug (motor telemetry starts 5–6 s after audio, ends 1–3 s before), full-sequence PIT-MSE on real in-flight recordings (speech-high / whitenoise-high) was: OLD SimpleConv 224.6 / 87.7; OLD BiGRU-v2 199.9 / 104.0; V2-trained SimpleConv 274.1 / 123.8; V2-trained BiGRU-v2 277.3 / 323.0. V1-trained models generalise *better* to in-flight audio than V2-trained ones: ~20% of V2's clips have near-constant synthetic RPS, and V2-trained models exploit this (predicted std ~0.05 rev/s vs GT std ~1.2 rev/s) — a shortcut that fails on real, time-varying flight.

### V3: dataset regeneration + baseline

V3 (6000 train + 600 valid, 1 s @ 16 kHz, SNR range [−30, 0] dB, 672 MB, per-sample `mixture.wav`/`vocals.wav`/`noise.wav`/`rps.npy` with 4 motors × 32 STFT frames) was trained once with SimpleConv, standard MSE (no PIT), AdamW LR 1e-3, converging in 40 epochs (patience 30): **val MSE 227.0 (RMSE 15.1 RPS), MAE/clip 8.14 RPS (~9% of the 1–90 RPS range), an 81.9% improvement over a naive baseline.**

Cross-eval of OLD (V1-trained) vs NEW (V3-trained) SimpleConv: OLD scores MSE 5.2/MAE 0.67 on its own (V1) set but MSE 477.8/MAE 12.1 on V3; NEW scores MSE 84.5/MAE 3.65 on V1 but MSE 229.0/MAE 8.14 on V3 (its own set) — again showing V1 scores were inflated by memorisation, and V3 is intrinsically harder even in-distribution (229 vs V1's 5). Per-channel MSE on V3 spans ~4× (ch4=97 easiest, ch6=93, ch3=394 hardest), consistent with microphone geometry rather than model artifact. R² is unusable on both V2 and V3 (near-constant per-clip RPS drives SS_total ≈ 0, R² → −∞); MSE/MAE are the trustworthy metrics.

## Conclusion

DREGON-LM-V3 is the bugfixed, restructured successor generator (`scripts/create_dregon_librimix_v3.py`) to V2, with a validated SimpleConv baseline (val MSE 227.0) but **no PIT-loss or BiGRU-v2 run recorded against V3** in these sources — the strongest documented multi-rotor result remains the V2-trained BiGRU-v2 at 2.5% motor combos (PIT MSE 56.7). V3's suggested next steps (per its checkpoint notes) were to retrain SimpleConvV2/BiGRU-family models on V3, compare a DCUNet/DCCRN-encoder RPS head, and integrate RPS prediction into the enhancement pipeline; it's not confirmed in these sources whether V3 actually fixes V2's constant-speed/motor-combo variance-suppression bias (it wasn't a stated V3 goal — V3's fixes were the axis bug, per-recording command cleaning, on-demand LibriSpeech, and float32 RPS).

**Superseded:** the repo has since moved to `DREGON-LM-V4` as its canonical multi-rotor RPS dataset (`conf/data/dregon_lm_v4.yaml`, `dregon_lm_v4_michaels.yaml`, `dregon_lm_v4_8ch_flat.yaml`; documented in `src/data_processing/AGENTS.md` § "DREGON-LM-V4 + Michael's"). V2/V3 as described here are historical stepping stones; the durable lessons — recording-level splits, PIT loss with a motor-combo ordering regularizer, PIT-aware vs fixed-order MAE, and the R²-is-broken-on-short-clips pitfall — carried forward into later dataset/training work.
