# Three-track plan — 2026-07-20

**Status:** initial proposal (user review pending) · supersedes the "bridge to SE
now" recommendation from the 2026-07-20 goal review after user objections were
upheld by evidence sweeps (see memory: `goal-review-2026-07-20-state-of-project`).

Three goals, run simultaneously by interleaving compute: GPU trains baselines
(Track 1) while CPU-bound VK work (Track 2) and GP fitting (Track 3) proceed;
switch tracks whenever a training run is in flight.

---

## Track 1 — Blind-SE floor, then the oracle question

### Why this order
No credible oracle-RPS conditioning gain exists (own b1/b2 replication: DCUNet
+1.73 dB SI-SDR only at −30…−20 dB with eSTOI/PESQ down, DCCRN −2.30 dB;
Gulli et al. 2025 concentrates its gains at −30/−20 dB and loses STOI/PESQ at
−10/0 dB — same shape). Any RPS-informed claim is meaningless without a strong
*modern blind floor* measured on our data first.

### M1.1 — Infra (CPU/dev work, ~days)
- SE-target mode in `OnlineMixIterableDataset`: yield `(mixture, clean_speech)`
  instead of `(audio, rps_target)`; speech source, SNR-controlled mixing,
  augmentations, and E5 time-warp already exist in the pipeline.
- Plain-audio noise-source kind (no RPS/telemetry track required) so MIMII /
  AeroSonicDB / HornBase / DroneAudioSet can serve as noise pools; per-category
  weight support (extends E9's weighted MixedNoisePool).
- Fixed deterministic SE validation sets: (a) **drone valid** — held-out drone
  noise × held-out LibriSpeech, SNR grid −30…0 dB in 5 dB steps; (b)
  **per-category harmonic valid** — same protocol per noise category.

### M1.2/M1.3 — Two baseline passes (GPU)
**Architectures (4 core + anchors):**

| Model | Role | Why |
|---|---|---|
| Edge-BS-RoFormer | band-split transformer, in-house SOTA | Paper 1 winner on DN-LM (−15 dB: +2.2 dB over DCUNet, +2.3 over HTDemucs); already implemented |
| TF-GridNet | dense full+sub-band TF dual-path | strongest external architecture family, never tested on drone/harmonic noise; sub-band structure suits combs; public code (ESPnet); use mid-size config |
| MP-SENet | parallel magnitude+phase decoding | modern perceptual-metric line; explicit phase matters for later harmonic subtraction; public code, moderate size |
| DCUNet | complex-UNet anchor | 2023 benchmark winner (12-model study); links new results to both prior papers; in repo |
| noisy input + Wiener | floors | trivial anchors in every table |
| (optional) diffusion-buffer | generative representative | already implemented (a2); run at extreme SNR only if budget allows |

4 architectures is the deliberate cap: each extra arch costs two more training
runs (both passes) for diminishing inference; the four span the four families
(band-split transformer / dense dual-path / mag+phase / complex UNet).

Full-library sweep notes (2026-07-20):
- **Ruled out by own prior data:** DPTNet/DPRNN/SepFormer-class dual-path
  without band structure (Paper 1: DPTNet −25 dB vs Edge-BS-RoFormer);
  FullSubNet (superseded by TF-GridNet within the same full+sub-band family);
  SEGAN/MetricGAN GAN line (2023 benchmark: generative arch class lost).
- **MP-SENet alternates if code/porting disappoints:** CMGAN, DB-AIAT/DBT-Net,
  dual-branch Mamba SSM (TASLP 2024) — all in library as stubs.
- **Generative arm, if expanded beyond diffusion-buffer:** the library holds the
  full SGMSE+ lineage (Interspeech 2022 → TASLP 2023, StoRM, universal
  diffusion SE, joint generative+predictive decoders, single/few-step) —
  SGMSE+ is the canonical citation baseline there.
- **Mechanism sources for M1.5b (not baselines):** PHASEN's harmonics-aware
  attention + phase stream; SN-Net's explicit two-branch speech+noise modeling
  (AAAI 2021); adaptive comb filtering for harmonic signals (TASSP 1986).
- **Recent drone-SE citers of the 2023 benchmark (related work, not
  baselines):** adapter-based transfer learning (IWAENC 2024), ASR-distortion
  mitigation under drone noise (2026), aerial bioacoustic monitoring (2026),
  LadderNet hybrid (2024).

**Update 2026-07-20:** the authoritative execution plan for M1.1–M1.4 is
`docs/se-baselines-plan.md`. It promotes SGMSE+ (score-diffusion, trained from
scratch) to a 5th core baseline — the prior Diffusion-Buffer batch (a2) was
eval-only with an unresolved chunking mismatch and contributes nothing
reusable.
Sample rate: native 16 kHz (per native-16khz-baselines memory). Single-channel
first; multichannel is a later differentiator (Gulli et al. explicitly left it
open).

**Pass A — drone-only noises:** uniform over drone sub-datasets (DREGON +
Michael's pools, DroneAudioSet, SPCUP19-egonoise, zenodo/new-drone-noises,
drone_audio) × LibriSpeech, online-mixed at −30…0 dB, augmentations
(random_gain, polarity, channel_drop) + time-warp (α ≤ 1.12).

**Pass B — all harmonic noises, uniformly weighted by category:** drones /
MIMII industrial (fan, pump, valve, slider) / MIMII-DG / AeroSonicDB propeller
aircraft / motors (HUSTmotor + KAIST) / horns (HornBase). Same recipe.

**Both passes evaluated on both fixed valids.** Pass B vs Pass A *on the drone
valid* answers: does diverse harmonic noise teach transferable structure
(gain) or dilute capacity (loss)? Per-category breakdown shows which
categories transfer. Optional third arm: hold one category out of Pass B
entirely (e.g. AeroSonicDB) → zero-shot cross-category generalization — the
cleanest C3 figure available.

### M1.5 — Oracle question, mechanistically (after floors exist)
- **(a) VK-reconstruct subtraction (CPU, no training):** `vk_reconstruct`
  envelopes give a coherent harmonic-noise estimate from oracle RPS; subtract
  from the mixture, score on the drone valid vs the blind floor. Cheapest
  possible Bet-3 MVP; uses Track 2 machinery.
- **(b) Oracle-RPS conditioning of the Pass-A winner** (one run, mechanistic
  fusion informed by the pending comb-filter literature evidence — PercepNet-
  style comb gating rather than feature concat).
- **Kill criterion (C5 gate):** if neither beats the blind floor by ≥0.5 dB
  SI-SDR *and* ≥0.02 eSTOI at −10/−15 dB → the "RPS-as-latent-for-SE" framing
  pivots (plausibly to tracking+generation as the thesis core).

---

## Track 2 — VK parity & drone-agnostic tracking (mostly CPU)

Evidence base: telemetry-init VK 0.24–0.8 rev/s vs neural 1.5–3.1 (cruise);
blind VK 0.68–1.5 but twins unresolved; VK-on-predictor is a no-op (<0.02
rev/s) → neural errors are non-local. Literature niche is empty (no neural
tracker beats VK-class anywhere; no published acoustic-only drone RPM
estimator).

- **M2.1 Twin resolution:** shared-comb-shape prior in blind seeding (matched-
  filter base scan + alias rejection; template from confidently-recovered
  rotors). Fixes the FLY124 spurious-peak failure.
- **M2.2 Drone-agnostic operation:** auto-derive the hand-tuned knobs
  (bw_hz 1.5↔7.0, k-range, couple_hz, update_gate 30↔8, max_step, per-recording
  base seeds) from confidence/SNR signals the pipeline already computes.
  Acceptance: one config, no per-rig retuning, pooled ≤1.5 rev/s incl. twins on
  DREGON + FLY124 cruise; SPCup as label-free stress test.
- **M2.3 VK→subtraction bridge:** package `vk_envelopes`/`vk_reconstruct` as
  the coherent noise estimator for M1.5a.
- **M2.4 (GPU, later):** learned tracker with VK structure — unrolled coupled
  VK / KalmanNet-style learned gain / VK-distilled targets. Publishable
  regardless of SE outcome.

---

## Track 3 — Generator vs GP honest head-to-head (CPU-heavy)

No head-to-head exists: neural gen scored by MR-STFT/comb-mask on n=48
free-flight clips; GP has only coeff-RMSE 0.10 (Michael's) + CONA-domain
metrics. Disjoint metrics and data.

- **M3.1:** train one GP per drone on the same swapped DREGON+Michael's split
  (`train_dregon_michaels.py`), H=24 first — `apocrita-cpu` backend.
- **M3.2:** H→60 via streaming per-frame Fourier design (bounds RAM); covers
  the 2–3 kHz band where drone combs live.
- **M3.3:** score GP with the identical MR-STFT + comb-masked |Δlog-mag|
  protocol on the same 48 clips; add GP rows (same RPS trajectory, matched
  mic) to `noise_gen_real_vs_generated.ipynb` incl. listening players.
- **M3.4 Decision gate:** if the artifact-free GP matches the neural generator
  on comb fidelity → GP becomes the preferred augmentation substrate (E4
  failed on exploitable neural artifacts) and the coherence-aware-loss work is
  re-scoped; if the neural generator wins clearly, coherence-aware multichannel
  loss proceeds (near-empty literature niche — novelty, but only after the
  baseline is honest).

---

## Interleaving schedule (indicative)

| When | GPU | CPU (fill while training) |
|---|---|---|
| W1 | Pass A runs for in-repo archs (Edge-BS-RoFormer, DCUNet) as M1.1 lands; port TF-GridNet/MP-SENet | M1.1 infra; M3.1 GP jobs on apocrita-cpu; M2.1 twin prior |
| W2 | Pass A remaining; then Pass B launches | M2.2 auto-tuning; M3.2/M3.3 GP eval harness; M1.5a VK-subtraction (needs only valid sets + M2.3) |
| W3 | Pass B finishes; M1.5b oracle run on Pass-A winner | M1.4 analysis + diversity answer; M3.4 gate; M2.4 design |
| End W3 | — | Reports per track; **overdue mid-point portfolio review → rewrite GOALS.md portfolio** with these results |

## Literature evidence (2026-07-20 bibliography sweep)

- **Gulli et al. 2025, exact conditioning deltas** (full text in library):
  rotor-vs-blind SI-SDR gain is largest at the lowest SNRs and on weaker
  architectures — on their strongest model (DCCRN) +39.9% at −30 dB shrinking
  to +0.9…+5% at −10…0 dB; STOI/PESQ deltas mixed-sign. First and only
  controlled side-info-vs-blind ablation in drone SE; the classical
  motor-guided line (Schmidt dictionary IROS'16, Hioka/Yen PSD-Wiener) never
  reported one.
- **Parametric-harmonic-subtraction + neural enhancement: the niche is
  EMPTY.** No published work subtracts a parametric harmonic noise model
  before/inside a neural enhancer at any SNR (external + library search).
  M1.5a (VK-reconstruct subtraction) would be first — novel if it works.
- **Multichannel + deep + low-SNR drone SE: unoccupied.** Best published
  multichannel results remain classical BSS (Wang & Cavallaro TASLP 2020,
  SNR < −15 dB); Gulli is single-mic and defers multichannel; the 2023
  benchmark is single-channel by design. Phase-2 multichannel baselines on
  our 8-mic data would occupy an empty cell.
- **Rotor geometry as a spatial prior for suppression: nobody does it.**
  Closest is DOA-sector weighting for localization. Consistent with the
  project's SRP-floor finding that the rigid-geometry prior is the open
  lever — a second empty novelty slot, enabled by our corrected geometry.

## Standing decisions
- Every comparison reports per-SNR SI-SDR / eSTOI / PESQ with the noisy+Wiener
  anchors; no aggregate-only claims (Gulli lesson: percentage aggregates at
  extreme SNR mislead).
- All training via `python train.py experiment=<name>` + omnirun; GPU =
  gpushort/colab/kaggle for small models, apocrita-long for TF-GridNet;
  CPU = apocrita-cpu.
