**Status:** scaffolded (not yet run) · 2026-07-25 – present · replicates
Mukhutdinov et al., *"Deep Learning Models for Single-Channel Speech
Enhancement on Drones"*, IEEE Access, 2023 · related batch:
[`f1-se-blind-baselines.md`](./f1-se-blind-baselines.md)

# F2 — replication of the 2023 IEEE Access drone-SE survey (DCUNet arm)

## Motivation

F1 found DCUNet to be the **weakest** of five modern architectures on our data
— it denoises at low SNR but sits below the noisy input on eSTOI everywhere and
degrades SI-SDR above −5 dB (see `f1-se-blind-baselines.md` § Findings 1 & 4).
That is awkward, because the 2023 IEEE Access survey — the project's own prior
work, whose 12-model benchmark DCUNet *won* — reports a clearly positive DCUNet
result on drone ego-noise. Before any further conclusion is drawn from the F1
ranking, we need to know whether we can **reproduce the survey's own DCUNet
number under the survey's own protocol**. If we can, the F1 result is a genuine
data/regime difference; if we cannot, something in our training or evaluation
pipeline is off and every F1 number inherits the defect.

So this batch is a *faithful* re-run of the paper's DCUNet setup, deliberately
changing as little as possible, and one thing at a time afterwards. The one
up-front exception is the sample rate: we run at the project-native **16 kHz**
rather than the paper's 8 kHz, keeping the paper's STFT resolution in
milliseconds (see § Deliberate deviations for the reasoning and its consequence
for the target numbers).

## The paper's setup (ground truth for this replication)

| Item | Paper | Here |
|---|---|---|
| Sample rate | 8 kHz (everything resampled) | **16 kHz** — the project's native rate (deliberate deviation, see below) |
| Speech | TIMIT, 90 % train / 10 % held out | **LibriSpeech train-clean-100** (TIMIT is not in our data lake), 25 of 246 speakers (~10 %) held out |
| Noise | 5 drone ego-noise sequences, **first channel only** | AVQ `S1_seq1`, `S1_seq2`, `S1_seq3`, `S2_seq1`, `S2_seq2`, channel 0 — published as the standalone `AVQ-egonoise` dataset (see § The noise dataset) |
| Train/valid noise | the **same** 5 recordings for both; only speech is split | same |
| Crop | T = 24000 samples = 3.0 s | same 3.0 s, i.e. T = 48000 samples @ 16 kHz |
| Train SNR | U(−25, −5) dB, mixed on the fly | same |
| Valid | fixed mixtures at SNR ∈ {−25,−20,−15,−10,−5} dB | same, 50 clips/point = 250 clips |
| Loss | SI-SDR only | `conf/loss/si_sdr.yaml` (the standard 16 kHz group) |
| Model | DCUNet-10 (10 complex conv layers) | `model_type: dcunet`, `dcunet_num_encoder_layers: 5` (5 enc + 5 dec = 10) |
| STFT | 64 ms window / 16 ms hop | **the same 64 ms / 16 ms**, i.e. n_fft 1024, hop 256, dim_f 512 @ 16 kHz |
| Optim | Adam, lr 1e-3, plateau patience 5 (×0.1), early-stop patience 10, batch 32 | same |

**Replication target.** The paper reports DCUNet at **−15 dB input SNR**:
SI-SDR **+3.7 dB**, eSTOI **0.4**, PESQ **1.9**. Caveat: the paper's headline
numbers are on a *different* drone (the AS dataset), whereas we substitute AVQ —
AVQ's ego-noise is a comparatively benign, stationary onboard recording, so if
anything the AVQ validation should be **easier** than the paper's. Second
caveat: those numbers were measured at 8 kHz on TIMIT and we run at 16 kHz on
LibriSpeech, so they are a **reference point rather than an exact bar** (see
§ Deliberate deviations). Landing near or above them is the pass condition;
landing far below still points at our pipeline, not at DCUNet.

## Wiring

| Component | Config |
|---|---|
| Experiment | `conf/experiment/f2_dcunet_avq_survey.yaml` |
| Model | `conf/model/f2_dcunet_survey.yaml` (16 kHz / n_fft 1024 / hop 256 / dim_f 512, chunk 48000) |
| Data | `conf/data/f2_avq_survey.yaml` |
| Train stream | `conf/online_mix/se_avq_survey.yaml` (`kind: audio_pool`, `dataset: AVQ-egonoise` — no key/channel filtering) |
| Noise dataset | `AVQ-egonoise` (dload, pinned) — publisher `scripts/publish_avq_egonoise.py` |
| Valid set | `scripts/build_se_valid.py --dataset avq --publish` → `SE-valid-avq-survey` (250 clips) |
| Loss / metrics | `si_sdr` / `separation_basic` (`separation_full` at eval for PESQ/eSTOI) |

Run:

```bash
python scripts/build_se_valid.py --dataset avq --publish && dload pin SE-valid-avq-survey  # once
python train.py experiment=f2_dcunet_avq_survey
python eval.py  experiment=f2_dcunet_avq_survey metrics=separation_full
```

## The noise dataset: `AVQ-egonoise`

All three arms draw their AVQ noise from **`AVQ-egonoise`**, a small derived
dataset published by `scripts/publish_avq_egonoise.py` that contains *exactly*
what this batch consumes: the 5 pure ego-noise sequences of `AVQ`
(`S1_seq1`/`S1_seq2`/`S1_seq3`/`S2_seq1`/`S2_seq2` — the only AVQ recordings
without an `angle_vad` entry, i.e. the only ones without the speech source),
**channel 0 only**, `soxr_hq`-resampled to **16 kHz mono**. 705 s in total: one
43 MiB shard, 5 samples, each carrying AVQ's per-recording `meta` plus
provenance (`source_dataset`/`source_version`/`source_channel`).

It exists for two concrete reasons, both costs of the previous
`dataset: AVQ` + `channel: 0` + `include_keys: [...]` formulation:

1. **Shard scan.** A dload manifest carries no per-shard key list, so keys are
   only known once a shard is opened. Filtering by key therefore forced the
   pool to download all **11 AVQ shards (~4 GiB** of 8-ch 44.1 kHz audio) to
   discover which hold the 5 wanted sequences — for ~12 minutes of audio.
2. **Multipart ETag.** AVQ's largest shard is a 352 MB, 42-part multipart
   upload. s3transfer's ETag validation on some boto3 builds (as shipped in the
   Kaggle image) rejects a multipart ETag outright:
   `S3DownloadFailedError ... did not match expected ETag`. The object is **not
   corrupt** — its sha256 matches its content-address digest — but the download
   fails on those backends regardless. `AVQ-egonoise`'s single 43 MiB shard is a
   single-part upload, so the check cannot fire.

`AVQ` and `AVQ-raw` are untouched and remain the canonical full dataset.

## Pre-flight diagnostic: the noise pool is NOT the cause

Before spending GPU time on the ladder, the **existing F1 DCUNet checkpoint**
(trained on the broad 6-dataset drone pool, 1 s crops, SI-SDR+MRSTFT loss) was
scored on the **AVQ-only** valid set — the paper's own noise. If the F1 floor had
been caused by noise-pool breadth, this should have looked much healthier.

It does not (16 kHz, 3 s, n=50/SNR; `results/f2_diag/`):

| SNR | noisy SI-SDR | F1 DCUNet | Δ | noisy eSTOI | F1 DCUNet |
|---|---|---|---|---|---|
| −25 | −25.76 | −24.02 | +1.7 | 0.022 | 0.016 |
| −20 | −20.41 | −17.00 | +3.4 | 0.058 | 0.051 |
| −15 | −15.07 | −12.99 | +2.1 | 0.121 | **0.101** |
| −10 | −9.99 | −7.50 | +2.5 | 0.218 | 0.201 |
| −5 | −5.00 | −3.61 | +1.4 | 0.310 | 0.276 |

On the paper's own noise the F1 model gains *less* SI-SDR than it does on the
broad pool (+2.1 vs +4.7 dB at −15 dB) and still pushes **eSTOI below the noisy
input at every SNR**. So the six-dataset pool was never the problem: the cause is
in the model/training configuration — STFT resolution, 1 s crops, SNR range or
loss — which is exactly what step 1 changes.

**A useful side-effect: the valid set is calibrated against the paper.** Its
noisy eSTOI at −15 dB is **0.121**, essentially the paper's quoted **0.1**
baseline. The test condition therefore matches the published one closely, so
"eSTOI 0.121 → ~0.4 at −15 dB" is a fair bar at 16 kHz and any shortfall is the
model's, not a mismatched benchmark.

Consequence for the ladder: steps 2–3 are no longer the prime suspects. Their
role shifts from *finding* the culprit to *confirming* that the corrected recipe
survives broader noise.

## Planned arms — the noise-pool ladder

The replication is step 1 of a three-step ladder that walks the *training noise
distribution* from the paper's setting to F1's, one step at a time. Everything
else is frozen: same model (`f2_dcunet_survey`, DCUNet-10), same loss
(`si_sdr`), same 16 kHz rate, same 3.0 s / 48000-sample crop, same train SNR
range U(−25, −5) dB, same optimizer and schedule (Adam 1e-3, plateau ×0.1
patience 5, early stop patience 10, batch 32, `samples_per_validation: 41580`),
same speech source and same 25 held-out speakers — **and the same fixed
validation set**.

| Step | Experiment | Training noise pool | Online-mix policy |
|---|---|---|---|
| 1 | `f2_dcunet_avq_survey` | AVQ ego-noise only (5 sequences, channel 0) | `conf/online_mix/se_avq_survey.yaml` |
| 2 | `f2_dcunet_alldrone` | **all drone** noise (F1 Pass A pool) + AVQ | `conf/online_mix/se_survey_alldrone.yaml` |
| 3 | `f2_dcunet_allharmonic` | **all harmonic** noise, category-uniform (F1 Pass B pool) + AVQ | `conf/online_mix/se_survey_allharmonic.yaml` |

Each pool is a strict **superset** of the previous one — AVQ is carried into
steps 2 and 3 (in step 3 inside the `drone` category, with that category's
weights renormalised to 1/7 each so it still sums to 1.0 and MIMII cannot
dominate). So the manipulation is purely *additive*: step 2 adds drone
diversity, step 3 adds non-drone harmonic sources, and neither removes the
in-domain source. A drop from step 1 to step 2/3 therefore measures the cost of
noise-distribution breadth at fixed capacity, not a domain shift away from the
test condition.

**All three arms validate on the same fixed `SE-valid-avq-survey` set**
(250 clips at SNR {−25,−20,−15,−10,−5} dB, 16 kHz) — `conf/data/f2_avq_survey.yaml`,
`f2_alldrone.yaml` and `f2_allharmonic.yaml` differ only in the `train:` block.
This is deliberate. Holding validation fixed is what makes the training-noise
pool the single manipulated variable: the monitored metric (`si_sdr`, max), the
LR-on-plateau trigger and the early-stopping criterion are all measured on
exactly the same distribution in each arm, so both the final numbers *and* the
optimisation trajectories (epochs to best, LR-drop points) are comparable. A
per-arm valid set matched to its own training pool would instead confound
"trained on more noise" with "scored on a different, generally harder test".

**Target numbers.** The pass condition for step 1 is the paper's DCUNet result
at −15 dB input SNR: SI-SDR **+3.7 dB**, eSTOI **0.4**, PESQ **1.9** (AVQ should
be no harder than the paper's AS drone). For steps 2 and 3 the number to watch
is the *delta* against step 1 on the identical valid set, and how far step 3
lands from the F1 DCUNet numbers in `f1-se-blind-baselines.md` — step 3 is the
F1 noise distribution under the survey protocol, so it is the bridge between the
two batches.

## Deliberate deviations (and why)

- **16 kHz, not the paper's 8 kHz — but the paper's STFT in milliseconds.**
  The survey's 8 kHz is a literal detail of that study, not a property worth
  replicating: the project's native rate is 16 kHz and the in-repo Paper-1
  replication already trains DCUNet successfully there (SI-SDR **−8.09 dB**
  overall on DN-LM, ahead of Edge-BS-RoFormer at **−9.94 dB**). So this batch
  runs at 16 kHz while preserving what actually defines the paper's front-end —
  its STFT *resolution in time*: window 64 ms / hop 16 ms, which at 16 kHz is
  n_fft **1024** / hop **256** (it was 512/128 at 8 kHz). Note that this is 2x
  finer in time than the F1 SE baseline's n_fft 2048 / hop 512 (128 ms / 32 ms),
  so the F2 arms are not merely F1 with a different noise pool.
  *Consequence, stated honestly:* the paper's absolute targets (SI-SDR +3.7 dB,
  eSTOI 0.4, PESQ 1.9 at −15 dB) were measured **at 8 kHz on TIMIT**, so at
  16 kHz on LibriSpeech they are a **reference point, not an exact bar** — both
  the bandwidth (the model must now also reconstruct 4–8 kHz, and eSTOI/PESQ are
  computed over a wider band) and the speech corpus differ. Read the targets as
  an order-of-magnitude pass condition: landing near them means the pipeline is
  sound, landing far below still points at our pipeline.
- **LibriSpeech instead of TIMIT.** TIMIT is not in the data lake. The speech
  split is by *speaker* (the F1 `HELDOUT_SPEAKERS` list, 25 of 246 ≈ 10 %),
  which is stricter than the paper's 90/10 utterance split — train and valid
  never share a speaker.
- **AVQ instead of the paper's AS drone.** The 5 AVQ ego-noise sequences are the
  closest onboard-array ego-noise recordings we have published; the other 7 AVQ
  recordings contain the speech source and are excluded — by construction, since
  the pools read the derived `AVQ-egonoise` dataset (§ The noise dataset).
- **Infinite online stream, not a fixed epoch.** The paper mixes on the fly too;
  our stream is infinite, so `samples_per_validation` defines the epoch. It is
  set to 41580 samples (≈1300 steps at batch 32) — the paper's own epoch (10
  passes over its 4158 training utterances), used because both patiences
  (Nα = 5, NE = 10) are counted in epochs. Same value in all three arms.
- **8 ms zero-padded tail.** 48000 is not a multiple of the 256-sample hop
  (48000/256 = 187.5), so the iSTFT returns 47872 samples and DCUNet zero-pads
  the last 128 (8 ms, 0.27 % of the clip) back to 48000, emitting
  `UserWarning: DCUNet output length mismatch: output=47872, input=48000,
  diff=-128. Consider adjusting chunk_size.` once per process. Kept as-is
  because T = 3.0 s is the paper's crop; a hop-aligned 47872 would silence it.
- **Valid set is published + pinned** (`SE-valid-avq-survey@95378d78ccf1`), so it
  streams from R2 on any backend. (It was rebuilt on `AVQ-egonoise`, replacing
  `@a88d9204506d`: same 5 recordings, same channel 0, same 16 kHz, same seed and
  SNR grid, but the noise draws land on different offsets because the pool's
  shard layout changed — so the set is distributionally identical, not
  clip-identical, and the pre-flight diagnostic table above was measured on the
  predecessor build. Checked: the rebuilt set's noisy anchors are
  −25.10/−19.96/−15.05/−10.02/−4.99 dB SI-SDR and eSTOI
  0.019/0.059/**0.125**/0.229/0.306, matching the old build's
  0.022/0.058/**0.121**/0.218/0.310 — the −15 dB eSTOI calibration against the
  paper's 0.1 baseline still holds.) Rebuild with
  `python scripts/build_se_valid.py --dataset avq --publish && dload pin SE-valid-avq-survey`;
  an unpublished local build can be read with a `local_root: <dir>` param.

## Results

_Not run yet._

## Conclusion

_Pending._
