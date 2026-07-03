# Harmonic Noise Generator as an RPS-Training Augmentation Source

**Status:** in progress (infra wired, no completed result run yet) · **Dates:** 2026-06-26 – 2026-07-03

## Motivation

RPS-predictor training relies on a handful of real recordings (DREGON + Michael's
FLY-series) mixed online with LibriSpeech at ultra-low SNR. The bet — from the
2026-06-30 supervisor slides ("train RPS predictors with synthetic-trajectory
augmentation") — is that a **learned generative model of rotor harmonic noise**
(the inverse of RPS prediction: RPS + array geometry → multichannel noise) can
synthesize unlimited, perfectly-labelled training variety and thereby improve
RPS prediction, especially generalization to unseen trajectories/drones.

The work proceeded in three prerequisite stages: (1) build and train the
generator (`PositionalHarmonicNoiseGen`) jointly on two drones with per-drone
conditioning; (2) fix an artifact in the generated noise by correcting the
DREGON train/valid split and adding random-phase rendering + smoothness
regularization; (3) wire the (would-be) trained generator as a live
`kind: generated` source inside the RPS online-mixing pipeline, so a frozen
noise-gen checkpoint renders synthetic noise on the GPU alongside real
recordings during RPS-predictor training.

## Results

No GPU training run has completed for any of the three stages — all milestones
below are **implementation/plumbing verification (CPU smoke tests)**, not
trained-model performance results.

1. **Generator + online data pipeline (2026-06-26).** `PositionalHarmonicNoiseGen`
   (single-rotor harmonic + filtered-noise emitter, differentiable propagation
   to 8 mics via 1/r attenuation + fractional delay) trains end-to-end on a
   CPU smoke test (128 train / 32 valid samples, 1 epoch, 236,572 params) with
   the online per-frame-geometry slicer streaming DREGON `in_flight_noise` +
   Michael's `FLY125`/`FLY124`. Confirms plumbing/gradient flow only — never
   run on GPU.

2. **Swapped split + random phase + smoothness (2026-07-01).** The original
   DREGON split was backwards (trained on 1 excluded-room recording, validated
   on 5); corrected to train on room2 (5 recs) + FLY125, validate on room1
   (1 rec) + FLY124. Random per-harmonic initial phase during training (zero
   phase at eval) was added, plus opt-in squared-2nd-difference smoothness
   penalties on harmonic amplitude (time) and diffuse-noise filter shape
   (time+freq), targeting a known weak spot (DREGON mid-frequency harmonics). A
   CPU smoke test confirmed the split loads and both the smoothness loss and
   random-phase training run/log correctly — plumbing only. Checkpoint note:
   the suggested smoothness weight (`1e-2`) is likely too weak (raw penalty
   ~0.26 vs. spectral loss in the hundreds) and needs a sweep.

3. **RPS-training augmentation wiring (2026-07-02).** A `kind: generated`
   noise source (`GeneratedNoisePool`) was added to the online-mixing dataset:
   a spawned producer process owns the CUDA context for a frozen noise-gen
   checkpoint and renders batches into a shared-memory ring buffer (seqlock)
   read lock-free by forked DataLoader workers; the synthetic RPS trajectory
   driving each chunk doubles as its exact label. A clean A/B config pair was
   defined (baseline = real-noise-only; treatment = same + one `generated`
   source, drone `michaels`, `n_harmonics: 100` matching the checkpoint). CPU
   smoke test (`tests/data_processing/test_generated_noise.py`, 42 pass)
   verified the producer/buffer/seqlock/mixing plumbing end-to-end; no
   RPS-predictor accuracy numbers exist for either arm.

**2026-07-03 refactor update:** the move to the unified Hydra framework ported
all three stages into `conf/experiment/`: `e2_noise_gen_dregon_michaels.yaml`
(stage 1), `e3_noise_gen_swapped_smoothness.yaml` (stage 2), and
`e4_generated_noise_augment.yaml` / `e4_no_aug_baseline.yaml` (stage 3's A/B
pair). Per `REPLICATION.md`, all 48 experiment configs compose successfully
against the Hydra schema, but E2/E3/E4 remain **config-complete with no
numeric results** — E2/E3 wrap an offline time-holdout dataset rather than the
original per-frame-geometry streamer (documented deviation), and E4 also needs
the E3 checkpoint artifact synced to the training machine.

## Conclusion

Design and implementation work for this cluster is done; what remains is
purely a completed GPU run. Next step: on a GPU machine with DREGON +
Michael's + LibriSpeech data available, run
`python train.py experiment=e3_noise_gen_swapped_smoothness` (the corrected
successor to `e2_noise_gen_dregon_michaels`) to actually train the generator,
then use its checkpoint to run `experiment=e4_generated_noise_augment` vs.
`experiment=e4_no_aug_baseline` for the RPS-predictor A/B. These `train.py`
commands supersede the deleted historical entry points
(`train_noise_generation.py`, `train_rps_predictor.py`) named in the source
checkpoint docs. Blocked on GPU compute and data availability, not on
remaining design work.
