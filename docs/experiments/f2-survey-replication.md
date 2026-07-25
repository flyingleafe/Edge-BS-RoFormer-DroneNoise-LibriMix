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
changing as little as possible, and one thing at a time afterwards.

## The paper's setup (ground truth for this replication)

| Item | Paper | Here |
|---|---|---|
| Sample rate | 8 kHz (everything resampled) | same |
| Speech | TIMIT, 90 % train / 10 % held out | **LibriSpeech train-clean-100** (TIMIT is not in our data lake), 25 of 246 speakers (~10 %) held out |
| Noise | 5 drone ego-noise sequences, **first channel only** | AVQ `S1_seq1`, `S1_seq2`, `S1_seq3`, `S2_seq1`, `S2_seq2`, channel 0 |
| Train/valid noise | the **same** 5 recordings for both; only speech is split | same |
| Crop | T = 24000 samples = 3.0 s | same |
| Train SNR | U(−25, −5) dB, mixed on the fly | same |
| Valid | fixed mixtures at SNR ∈ {−25,−20,−15,−10,−5} dB | same, 50 clips/point = 250 clips |
| Loss | SI-SDR only | `conf/loss/si_sdr.yaml` (rate overridden to 8 kHz) |
| Model | DCUNet-10 (10 complex conv layers) | `model_type: dcunet`, `dcunet_num_encoder_layers: 5` (5 enc + 5 dec = 10) |
| STFT | 64 ms window / 16 ms hop | n_fft 512, hop 128, dim_f 256 @ 8 kHz |
| Optim | Adam, lr 1e-3, plateau patience 5 (×0.1), early-stop patience 10, batch 32 | same |

**Replication target.** The paper reports DCUNet at **−15 dB input SNR**:
SI-SDR **+3.7 dB**, eSTOI **0.4**, PESQ **1.9**. Caveat: the paper's headline
numbers are on a *different* drone (the AS dataset), whereas we substitute AVQ —
AVQ's ego-noise is a comparatively benign, stationary onboard recording, so if
anything the AVQ validation should be **easier** than the paper's. Reaching or
beating the target is the pass condition; landing far below it points at our
pipeline, not at DCUNet.

## Wiring

| Component | Config |
|---|---|
| Experiment | `conf/experiment/f2_dcunet_avq_survey.yaml` |
| Model | `conf/model/f2_dcunet_survey.yaml` (8 kHz / 512 / 128 / 256, chunk 24000) |
| Data | `conf/data/f2_avq_survey.yaml` |
| Train stream | `conf/online_mix/se_avq_survey.yaml` (`kind: audio_pool`, `include_keys` = the 5 ego-noise keys, `channel: 0`) |
| Valid set | `scripts/build_se_valid.py --dataset avq --local-repo datasets/se-valid-local` → `SE-valid-avq-survey` (250 clips) |
| Loss / metrics | `si_sdr` / `separation_basic` (`separation_full` at eval for PESQ/eSTOI) |

Run:

```bash
python scripts/build_se_valid.py --dataset avq --local-repo datasets/se-valid-local   # once
python train.py experiment=f2_dcunet_avq_survey
python eval.py  experiment=f2_dcunet_avq_survey metrics=separation_full
```

## Deliberate deviations (and why)

- **LibriSpeech instead of TIMIT.** TIMIT is not in the data lake. The speech
  split is by *speaker* (the F1 `HELDOUT_SPEAKERS` list, 25 of 246 ≈ 10 %),
  which is stricter than the paper's 90/10 utterance split — train and valid
  never share a speaker.
- **AVQ instead of the paper's AS drone.** The 5 AVQ ego-noise sequences are the
  closest onboard-array ego-noise recordings we have published; the other 7 AVQ
  recordings contain the speech source and are excluded by key.
- **Infinite online stream, not a fixed epoch.** The paper mixes on the fly too;
  our stream is infinite, so `samples_per_validation` defines the epoch. It is
  set to 20000 samples (≈625 steps at batch 32) — the F1 convention.
- **8 ms zero-padded tail.** 24000 is not a multiple of the 128-sample hop
  (24000/128 = 187.5), so the iSTFT returns 23936 samples and DCUNet zero-pads
  the last 64 (8 ms, 0.27 % of the clip) back to 24000, emitting
  `UserWarning: DCUNet output length mismatch: output=23936, input=24000,
  diff=-64. Consider adjusting chunk_size.` once per process. Kept as-is because
  T = 24000 is the paper's number; a hop-aligned 23936 would silence it.
- **Valid set not published to dload** (yet) — it lives in a local dload
  repository under `datasets/se-valid-local` and is read via
  `SEValidFrameDataset(local_root=...)`. Publish (`--publish`) + `dload pin`
  before running on remote backends, since the local repo does not ship.

## Results

_Not run yet._

## Conclusion

_Pending._
