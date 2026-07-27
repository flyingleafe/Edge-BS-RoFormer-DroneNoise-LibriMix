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

> ⚠ **CORRECTED 2026-07-24 — the numbers below replace the original ones.**
> Two defects invalidated the first revision: (1) a **mixing-pipeline bug** —
> `_scale_source_to_snr` guards only the denominator, so a *digitally silent
> noise draw* (`drone_audio`, ~6% of 1 s draws) zeroed the clean target **and**
> the mixture; 5/350 `SE-valid-drone` clips are empty this way, 3 of them at
> 0 dB, and because SI-SDR vs an all-zero reference hits the −80 dB floor they
> dragged **every** method's 0 dB mean down by ~5 dB (the noisy anchor read
> −4.8 dB instead of −0.0) — this alone manufactured the apparent
> "non-monotonic 0 dB dip"; and (2) `--per-snr N` evaluation took the **first**
> N clips per group, which is not a random sample (at 0 dB the first 25 differ
> from the last 25 by ~10 dB). Pipeline fixed + regression-tested (commit on
> `se-baselines`); the published valid sets still contain the empty clips and
> need a rebuild+repin.

Paper-matched training (2026-07-22), corrected 2026-07-24. Per-SNR on
`SE-valid-drone`, **per-clip over the full 345-clip set** (5 corrupt clips
excluded). Noisy + Wiener anchors.

**SI-SDR improvement over noisy input (dB):**

| SNR | Wiener | **MP-SENet-A** | TF-GridNet-A | Edge-BS-RoF-A | DCUNet-A | DCUNet-B |
|---|---|---|---|---|---|---|
| −30 | −0.1 | **+23.3** | +15.4 | +20.1 | +11.1 | +11.5 |
| −25 | −0.0 | **+19.0** | +12.7 | +16.1 | +8.3 | +8.5 |
| −20 | +0.1 | **+18.3** | +11.6 | +14.0 | +6.0 | +6.0 |
| −15 | +0.1 | **+19.5** | +14.3 | +13.7 | +4.7 | +6.3 |
| −10 | +0.1 | **+17.0** | +13.4 | +11.8 | +2.6 | +3.0 |
| −5 | +0.1 | **+13.9** | +11.9 | +9.8 | +0.1 | +1.6 |
| 0 | +0.1 | **+11.7** | +9.7 | +6.3 | **−2.2** | **−2.6** |

**eSTOI (noisy → model):**

| SNR | noisy | MP-SENet | TF-GridNet | Edge-BS-RoF | DCUNet-A |
|---|---|---|---|---|---|
| −15 | 0.239 | **0.516** | 0.404 | 0.344 | 0.200 |
| −10 | 0.276 | **0.564** | 0.464 | 0.388 | 0.220 |
| −5 | 0.423 | **0.708** | 0.616 | 0.545 | 0.362 |
| 0 | 0.497 | **0.779** | 0.681 | 0.602 | 0.410 |

(DCUNet is **below the noisy input on eSTOI at every SNR** — it damages speech
while removing noise energy.)

### Findings

1. **Architecture dominates, and the newer SE ports win.** Ranking
   **MP-SENet > {TF-GridNet, Edge-BS-RoFormer} ≫ DCUNet** (Edge-BS-RoFormer is
   stronger below −20 dB, TF-GridNet from −15 dB up). MP-SENet (parallel
   magnitude+phase, 2023) is far the strongest — SI-SDR +19.5 dB and eSTOI
   0.239→0.516 at −15 dB — and **both ports are competitive with or ahead of the
   Paper-1 Edge-BS-RoFormer**.
   Notably MP-SENet and TF-GridNet were *compute-limited* (see caveats), so their
   true ceiling is **higher** than shown. DCUNet (older complex-UNet) barely
   lifts intelligibility (≈ noisy eSTOI) — it denoises (SI-SDR/PESQ up) but does
   not restore speech. All beat the Wiener floor at ≥ −10 dB.
2. **Diversity is capacity-dependent (helps only the weak model).** On the
   *drone* valid, DCUNet Pass B (all-harmonic) ≥ Pass A (drone-only) at most SNRs
   (Δ SI-SDR@−15: +1.5) — extra harmonic data helps the under-fitting model. But
   **all three stronger ports are hurt or unmoved** — the split is clean across
   every arch. Δ SI-SDR B−A (dB), per SNR {−30,−25,−20,−15,−10,−5,0}:

   | arch | −30 | −25 | −20 | −15 | −10 | −5 | 0 |
   |---|---|---|---|---|---|---|---|
   | DCUNet (weak) | **+0.5** | **+0.2** | **+0.1** | **+1.5** | **+0.4** | **+1.5** | −0.5 |
   | MP-SENet | −4.8 | −3.1 | −4.6 | −2.8 | −1.4 | +0.4 | +0.1 |
   | TF-GridNet | −0.8 | −0.4 | −1.3 | −1.8 | −0.8 | −0.6 | −0.1 |
   | Edge-BS-RoF | −3.8 | −4.6 | −5.0 | −3.8 | −4.0 | −2.7 | −1.6 |

   Edge-BS-RoFormer and TF-GridNet also lose eSTOI across the board (Edge-BS-RoF
   −0.036 @−15, TF-GridNet −0.031); MP-SENet is flat on eSTOI. Under equal budget,
   diverse noise dilutes drone-specific capacity — data breadth should scale
   *with*, not substitute for, architecture/budget. (MP-SENet/TF-GridNet Pass B
   read from best-drone-valid checkpoints while their runs continue; matched to
   same-regime Pass A ckpts, so the direction is a fair within-arch comparison.)
3. **Harmonic structure — not loudness — is what SE exploits.** Per-family floor
   on `SE-valid-harmonic` (Pass B, mean over SNR, MP-SENet SI-SDR-improve dB /
   eSTOI): the difficulty ordering is model-independent —
   motors (+19.8 / 0.58) > aircraft (+16.9 / 0.52) > horns (+15.8 / 0.51) >
   drone (+15.2 / 0.47) ≫ mimii (+9.8 / 0.34) > mimii_dg (+2.1 / 0.28). The
   *strongly-tonal* families are recovered far better than the *stochastic*
   industrial MIMII noise — direct evidence for the project premise (a
   rotating-source target is the favourable case). Transfer (B−A) per family:
   harmonic motors/horns gain across all archs; broadband MIMII families *lose*
   for the strong models (Edge −11.6 on mimii, MP-SENet −9.9 on mimii_dg) — they
   can't fit stochastic noise and lose focus. Diverse data helps only when added
   families share exploitable structure *and* the model has spare capacity.
4. **Loss + training length matter — but the loss is NOT DCUNet's ceiling.**
   masked-MSE gave ~noisy floors (attenuation, not intelligibility); the
   SI-SDR+MRSTFT composite + the paper-length schedule (NE=30 / Nα=15) turned
   DCUNet's −15 dB SI-SDR gain from +1.6 (12-min run) to +4.7 (converged).
   *Tested and refuted (2026-07-24):* the composite's MRSTFT term dominates the
   SI-SDR term's gradient at the model output by 4× (−30 dB) → **54×** (0 dB),
   which looks like a cause of over-suppression — but retraining DCUNet to
   convergence under **three** objectives moves 0 dB SI-SDR only between
   −2.17 (original composite), −2.07 (**pure SI-SDR**, `conf/loss/si_sdr.yaml`,
   job `f1-dcunet-lossa`, early-stop ep60) and −1.37 (**MRSTFT ×0.05**,
   `si_sdr_mrstft_w0p05`, ratio 53×→2.5×, job `f1-dcunet-lossc2`) — a <1 dB
   spread, **all still below the noisy input (−0.00)**, with eSTOI 0.410 / 0.386
   / 0.397 vs noisy 0.497.
   Also refuted: train/eval length mismatch (feeding 1 s halves, matching the
   training chunk, scores *worse* than whole 2 s clips: −3.7 vs −1.0 at 0 dB).
   **DCUNet's high-SNR degradation is an architecture/training-regime trait**
   (2.8M complex mask fitted over SNR~U(−30,0) learns a suppression strength
   tuned to the low-SNR bulk), not a loss artefact.
5. **SGMSE+ from scratch is non-viable in this budget (negative control).**
   The score-diffusion 5th arch, trained via the bespoke DSM loop, sits **below
   the noisy input at every SNR** (Pass B: SI-SDR −30…−57 vs noisy −5…−31; eSTOI
   ≈ 0 vs noisy 0.05→0.47). Its val SI-SDR is **flat from step 2k** (loss drops
   sharply then the sampler never converges to clean speech) — a 65M NCSN++ needs
   ~100× more compute than the discriminative models. So the blind floor here is
   a *discriminative* result; from-scratch generative SE is not competitive in
   the available budget. (Evaluated at ~40k of 200k steps; trajectory flat, so
   the checkpoint isn't the limiter.)

### Status / caveats

- Fully converged + evaled (drone): Edge-BS-RoFormer-A, DCUNet-A, DCUNet-B.
- **Compute-limited** (fp32 is ~10× slower/step than DCUNet — MP-SENet 1.5 it/s,
  TF-GridNet slower at batch 4; hit the 16 h wall so marked "failed" but their
  best.ckpt was recovered from R2): MP-SENet-A (~epoch 41), TF-GridNet-A
  (undertrained). **Their reported numbers are lower bounds.** SGMSE+ trained but
  non-viable (Finding #5).
- Eval is CPU-bound here (heavy fp32 forwards); `SE-valid-harmonic` per-category
  tables use a balanced 15-clips/(family,SNR) subsample for tractability.
- Pass A + Pass B complete on **both** valids (drone + harmonic per-category) for
  all four discriminative archs (mpsenet/tfgridnet from best-drone-valid ckpts).
- SGMSE+ trained (resume+R2-upload added to train_sgmse.py) + GPU-evaled (Pass B):
  non-viable (below noisy, eSTOI≈0). GPU eval REQUIRED (sampler ~30min/utt on CPU);
  eval_se_models.py self-fetches ckpt from R2 + `--r2-upload`. Pass A still queued
  (expected equivalent); SGMSE+ jobs run to 200k, re-eval at completion if changed.
- Pending: fp16 to fully converge the heavy ports (numbers are lower bounds);
  optional SGMSE+ full-200k re-eval + Pass A.

## Conclusion

On our harmonic-noise data at −30…0 dB, the **blind speech-enhancement floor is
architecture-bound and set by the newer SE models**: **MP-SENet** (parallel
magnitude+phase) is the strongest baseline (eSTOI 0.24→0.52 at −15 dB, 0.50→0.78
at 0 dB), with **TF-GridNet and Edge-BS-RoFormer competitive behind it** (Edge
stronger below −20 dB, TF-GridNet above) even while compute-limited; the classic
complex-UNet (DCUNet) denoises at low SNR but **degrades the input above −5 dB**
and sits below the noisy eSTOI everywhere. **Diverse-harmonic training is capacity-dependent** —
it helps only the weak DCUNet and *dilutes* all three stronger ports (MP-SENet,
TF-GridNet, Edge-BS-RoFormer) on drone noise; the split is clean across every
arch, so data breadth must scale with capacity, not replace it. Per family,
**harmonic structure — not loudness — is what SE exploits** (tonal
motors/aircraft/horns recover far better than stochastic MIMII). The 5th arch,
**from-scratch SGMSE+, is non-viable in this budget** (below noisy, eSTOI≈0) — the
blind floor here is discriminative. These floors — with noisy + Wiener anchors in
every table — gate later RPS-informed claims. Report:
`writing/reports/2026-07-22_se-blind-baselines/`. _(SGMSE+ Pass A + full-200k
re-eval optional; fp16 would let the heavy models fully converge.)_
