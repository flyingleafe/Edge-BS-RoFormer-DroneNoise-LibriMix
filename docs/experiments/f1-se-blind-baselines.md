**Status:** in progress · 2026-07-20 – present · plan:
[`docs/se-baselines-plan.md`](../se-baselines-plan.md) (Track 1 M1.1–M1.4 of
`docs/three-track-plan-2026-07-20.md`) · **report:**
[`writing/reports/2026-07-22_se-blind-baselines/`](../../writing/reports/2026-07-22_se-blind-baselines/report.pdf)

# F1 — SE blind (no-RPS) baselines on our data

## Motivation

Establish the **blind** (no-RPS) speech-enhancement floor on our harmonic-noise
data with modern architectures, so every later RPS-informed claim is measured
against an honest, strong no-side-information baseline. Two training passes over
the same architecture set, both scored on the same two fixed validation sets:

- **Pass A (`f1_<arch>_a`) — drone noises only**: the drone-focused floor.
- **Pass B (`f1_<arch>_b`) — all harmonic noises, category-uniform**: does
  *diverse* harmonic noise help models on drone noise (transferable harmonic
  structure) or hurt (capacity dilution)? Pass B − Pass A on `SE-valid-drone`
  answers the diversity question; the per-category breakdown on
  `SE-valid-harmonic` shows which categories transfer.

## Architecture set

| Arch | Family | Source | Params |
|---|---|---|---|
| `edge_bs_rof` (reuses `a1_edge_bs_rof_fa` model) | band-split transformer | in repo (Paper 1) | — |
| `dcunet` (reuses `a1_baseline_dcunet` model) | complex UNet | in repo | — |
| `tfgridnet` (`f1_tfgridnet`) | dense full+sub-band dual-path | port (ESPnet V1, mid-size) | 8.38 M |
| `mpsenet` (`f1_mpsenet`) | parallel magnitude+phase | port (yxlu-0102/MP-SENet, generator-only) | 1.71 M |
| `sgmse` (`f1_sgmse`) | score-based diffusion (generative) | port (sp-uhh/sgmse), trained from scratch | — |
| noisy input, Wiener | floors | trivial anchors | — |

## Setup

- **Data**: online-mixed SE stream (`OnlineMixIterableDataset` SE-target mode:
  yields `(mixture, clean_speech)`), 16 kHz mono, 1 s chunks, SNR ~ U(−30, 0) dB,
  `random_gain`/`random_polarity` augmentation. Noise pools:
  - Pass A (`conf/online_mix/se_drone_only.yaml`): DREGON + Michael's real
    drone frames (train recordings) + `audio_pool` over `drone_audio`,
    `DroneAudioSet`, `SPCUP19-egonoise`, `new-drone-noises` — roughly uniform
    over sub-datasets.
  - Pass B (`conf/online_mix/se_all_harmonic.yaml`): category-uniform over
    6 categories — drone (= Pass A pool), MIMII, MIMII-DG, AeroSonicDB (aircraft),
    motors (HUSTmotor + KAIST), horns (HornBase). Category-uniform weighting keeps
    258 GiB MIMII from dominating.
- **Validation (fixed, published, pinned)**: `SE-valid-drone` and
  `SE-valid-harmonic`, built by `scripts/build_se_valid.py`: held-out noise
  (per-shard/recording holdout) × held-out LibriSpeech speakers, SNR grid
  {−30,−25,−20,−15,−10,−5,0} dB, ≥50 mixtures/point/category, deterministic. Both
  passes monitor/early-stop on `SE-valid-drone` (`si_sdr`, max) — the primary
  drone floor — and are evaluated on **both** valids for the final tables.
  Published + pinned: `SE-valid-drone@68a9b184fcc5` (350 mixtures),
  `SE-valid-harmonic@855bdcd731fe` (2100 mixtures, 6 categories).
- **Loss/metrics**: `si_sdr_mrstft` composite train loss (negative SI-SDR +
  multi-resolution STFT); `separation_basic` (si_sdr, sdr) during training;
  `separation_full` (+ pesq, estoi) at eval. Noisy + Wiener anchors in every
  table (`scripts/eval_se_anchors.py`).
  - **Training length matches the 2023 IEEE Access 12-model benchmark**
    (Mukhutdinov et al., where DCUNet won): epoch duration ≈ their ~1,300
    steps/epoch (`samples_per_validation: 20000` → ~1,250 steps at batch 16),
    early-stop patience NE=30 (`patience: 30`), LR-on-plateau patience Nα=15
    (`optim.patience: 15`), cap 150 epochs. The first pass early-stopped at ~85k
    samples (12 min) — patience 8 on a noisy val metric fired on noise.
  - **Why not masked_mse**: the first pass used masked_mse (Paper-1-consistent);
    the resulting dcunet floor came out ~at the *noisy* floor on SI-SDR/eSTOI
    (SDR improved +6–10 dB — it denoises — but scale-invariant metrics did not,
    and `val/si_sdr` *degraded* with training). At ultra-low SNR MSE rewards
    attenuation-toward-silence, not intelligibility, so it fails to establish a
    meaningful floor. Switched to the metric-aligned SI-SDR + spectral composite
    (`losses.SISDRLoss`, standard for SE). The masked_mse dcunet result is kept
    as the motivating comparison.

### Deliberate deviations from the plan

- **MIMII is one category** (fan/pump/valve/slider combined), not four —
  category-uniform weighting already prevents domination and answers the
  diversity question; a per-machine split was a nice-to-have.
- **No time-warp augmentation** in the SE passes: `apply_time_warp` requires a
  rotor track, which telemetry-free `audio_pool` noise lacks. `random_gain` +
  `random_polarity` (the E5 workhorses) are used.
- `zenodo_drone_noises` excluded from `audio_pool` (published as a single ZIP
  blob, not per-file audio).
- **`audio_pool` training pool capped at `max_shards: 24`** per dataset: a
  random shard is drawn per sample, so an uncapped pool over 258 GiB MIMII /
  88 GiB DroneAudioSet would stream nearly the whole dataset from R2 over a run
  (infeasible I/O). 24 shards ≈ hundreds of diverse recordings — ample for a
  noise-augmentation pool — and bounds the pull to ~3 GB/dataset.
- **SGMSE+ trains via a bespoke score-matching loop** (`scripts/train_sgmse.py`),
  not `train.py`: the discriminative training loop cannot host score-based
  diffusion (needs the clean target fed in during training for the DSM loss,
  sampling only at eval, and a scalar loss the Series-based spec-validation
  doesn't model). Final scoring stays on `eval.py` (the SE codec calls
  `model(x)` with no target → the model runs the reverse-SDE sampler → enhanced).
  Caveat: a 65 M NCSN++ trained from scratch needs days of GPU (the SGMSE papers
  train ~160 epochs); within the queue budget it will be **undertrained** — its
  floor is reported with that caveat, as a compute-bounded generative baseline.

## Results

Paper-matched training (2026-07-22). Per-SNR on `SE-valid-drone`; all models on
the **same 25 mixtures/point** (fair cross-model subset — 50-vs-25 mixing is
unfair because SI-SDR is high-variance at 0 dB; the 25-subset ranking is stable,
absolute Δ at 0 dB carries ±few-dB variance). Noisy + Wiener anchors.

**SI-SDR improvement over noisy input (dB):**

| SNR | Wiener | **MP-SENet-A** | TF-GridNet-A | Edge-BS-RoF-A | DCUNet-A | DCUNet-B |
|---|---|---|---|---|---|---|
| −30 | −0.1 | **+23.2** | +17.0 | +21.1 | +10.2 | +11.9 |
| −20 | +0.1 | **+18.5** | +11.9 | +13.9 | +6.0 | +6.0 |
| −15 | +0.1 | **+21.4** | +16.8 | +14.4 | +5.0 | +7.6 |
| −10 | +0.1 | **+17.9** | +13.7 | +12.6 | +2.7 | +3.9 |
| −5 | +0.1 | **+14.0** | +11.6 | +9.6 | −0.2 | +0.7 |
| 0 | +0.1 | **+16.6** | +14.5 | +11.1 | +2.7 | +2.2 |

**eSTOI (noisy → model):**

| SNR | noisy | MP-SENet | TF-GridNet | Edge-BS-RoF | DCUNet-A |
|---|---|---|---|---|---|
| −15 | 0.234 | **0.540** | 0.436 | 0.370 | 0.228 |
| −10 | 0.276 | **0.643** | 0.533 | 0.454 | 0.249 |
| −5 | 0.423 | **0.693** | 0.603 | 0.534 | 0.350 |
| 0 | 0.468 | **0.789** | 0.687 | 0.609 | 0.418 |

### Findings

1. **Architecture dominates, and the newer SE ports win.** Ranking
   **MP-SENet > TF-GridNet ≳ Edge-BS-RoFormer ≫ DCUNet**. MP-SENet (parallel
   magnitude+phase, 2023) is far the strongest — SI-SDR +21 dB and eSTOI
   0.234→0.540 at −15 dB — and **both ports beat the Paper-1 Edge-BS-RoFormer**.
   Notably MP-SENet and TF-GridNet were *compute-limited* (see caveats), so their
   true ceiling is **higher** than shown. DCUNet (older complex-UNet) barely
   lifts intelligibility (≈ noisy eSTOI) — it denoises (SI-SDR/PESQ up) but does
   not restore speech. All beat the Wiener floor at ≥ −10 dB.
2. **Diversity is architecture-dependent (helps weak, hurts strong).** On the
   *drone* valid, DCUNet Pass B (all-harmonic) ≥ Pass A (drone-only) at most SNRs
   (Δ SI-SDR@−15: +5.0→+7.6) — extra harmonic data helps the under-fitting model.
   But **Edge-BS-RoFormer Pass B is markedly *worse*** (Δ@−15: +14.4→+10.0; eSTOI
   0.370→0.334) — under equal budget, diverse noise dilutes the strong model's
   drone-specific capacity. So data breadth should scale *with*, not substitute
   for, architecture/budget. (MP-SENet/TF-GridNet Pass B pending to confirm.)
3. **Loss + training length matter a lot.** masked-MSE gave ~noisy floors
   (attenuation, not intelligibility); the SI-SDR+MRSTFT composite + the
   paper-length schedule (NE=30 / Nα=15) turned DCUNet's −15 dB SI-SDR gain from
   +1.6 (12-min run) to +5.0 (converged).

### Status / caveats

- Fully converged + evaled (drone): Edge-BS-RoFormer-A, DCUNet-A, DCUNet-B.
- **Compute-limited** (fp32 is ~10× slower/step than DCUNet — MP-SENet 1.5 it/s,
  TF-GridNet slower at batch 4; hit the 16 h wall so marked "failed" but their
  best.ckpt was recovered from R2): MP-SENet-A (~epoch 41), TF-GridNet-A
  (undertrained). **Their reported numbers are lower bounds.** SGMSE+ deferred.
- Eval is CPU-bound here (heavy fp32 forwards); the 2100-clip `SE-valid-harmonic`
  per-category transfer table needs GPU eval (follow-up).
- Pending: Pass B for the stronger archs (running), harmonic transfer, Typst
  report.

## Conclusion

On our harmonic-noise data at −30…0 dB, the **blind speech-enhancement floor is
architecture-bound and set by the newer SE models**: **MP-SENet** (parallel
magnitude+phase) is the strongest baseline (eSTOI 0.23→0.54 at −15 dB, 0.47→0.79
at 0 dB), with **TF-GridNet and both beating the Paper-1 Edge-BS-RoFormer** even
while compute-limited; the classic complex-UNet (DCUNet) only denoises without
restoring intelligibility. **The value of diverse-harmonic training is
architecture-dependent** — it helps the weak DCUNet but *dilutes* the strong
Edge-BS-RoFormer on drone noise, so data breadth must scale with capacity, not
replace it. These floors — with noisy + Wiener anchors in every table — gate
later RPS-informed claims. Report:
`writing/reports/2026-07-22_se-blind-baselines/`. _(Per-category harmonic
transfer + MP-SENet/TF-GridNet Pass B to follow; fp16 would let the heavy models
fully converge.)_
