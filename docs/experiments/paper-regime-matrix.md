# The paper regime matrix

**Status:** in progress | 2026-09-05 → | plan: the approved session plan of
2026-09-05 (the wrap-up paper, ICASSP 2027 deadline 2026-09-16).

## Motivation

One table for the paper: every rotor-speed-prediction row at a stated training
regime, a stated trunk and a stated microphone count, so that a difference
between two rows has exactly one cause. The table must support five claims:

1. Regressors are precise on non-diverse data and do not transfer: a model
   trained on one microphone fails on the other microphones, a model trained on
   one drone fails on the other drone, and so on.
2. Generality is bought with precision: the more diverse the training data, the
   better the unseen-data numbers, and the worse the in-domain precision.
3. Harmonic tracking is not learned unless forced: augmentations force it
   partly, synthetic pre-training reliably. Open: how the salience models behave.
4. In the stochastic-comb limit the regressors collapse to a fixed fan of lines
   around the mean and the salience models alias (octaves).
5. Speech in the mixture makes the task about two times harder for every model
   and regime (to verify).

## The regimes

Every real rung: DREGON room2 recordings [+ FLY125], full powered envelope
(`min_motor_rps: 0.0`), 1 s chunks, online mixing, LibriSpeech at −30..0 dB,
the silence arm at one sixth of the chunks, `snr_ref_floor_rms: 0.02`, no
warm-up stage, no warm start. Validation: the frozen real split
`dload:DREGON-LM-V4-michaels-valid-full` (37 clips × 8 microphones).

| rung | policy | sources | mics | augmentations |
|---|---|---|---|---|
| R1 | `conf/online_mix/real_r1_dload.yaml` | DREGON | mic 0 | gain, polarity |
| R2 | `real_r2_dload.yaml` | DREGON | 8 | gain, polarity |
| R3 | `real_r3_dload.yaml` | DREGON + FLY125 | 8 | gain, polarity |
| R4 | `hb_m3s2_dload.yaml` | DREGON + FLY125 | 8 | gain, polarity, freq-scale [0.7, 1.3], time-warp |
| R4 nomix | `hb_m3s2_nomix_dload.yaml` | as R4 | 8 | as R4, `source_prob: 0.0` (no speech) |
| S1 | `salv2_comb.yaml` | static comb + silence | 8 | gain, polarity (rate scaling at generation) |
| S2 | `salv2_stoch.yaml` | stochastic comb + silence | 8 | same |
| C1 | `m3abl_comb_s1_dload.yaml` → R4 | comb, then real | 8 | comb stage with the full schedule |
| M | `hb_stochmixed_dload.yaml` | real + stochastic + comb, one pool | 8 | R4 schedule |

For magnitude front-ends `random_polarity` is an exact no-op and `random_gain`
a log-magnitude offset, so rungs R1–R3 are in effect "no augmentation".

Decisions of 2026-09-05 (user): the silence arm and the SNR floor are in every
real rung; the neural noise generator leaves the matrix (its rows R3 and R5 of
the earlier taxonomy are quoted at most as one ablation sentence); the salience
models run on all four real rungs; single seed; the June SimpleConv joins the
regressors on the lower rungs; the magnitude front-end is used for every trunk,
so the transformer column is rerun without the IF channel.

## Trunks

| key | family | model config |
|---|---|---|
| `sc` | regressor, June SimpleConv | `simple_conv` |
| `scv2` | regressor, BiGRU head | `simple_conv_v2` (`hb_scv2_mag_nogate` model file) |
| `tm` | regressor, transformer head, magnitude front-end | `simple_conv_v2_transformer` |
| `gru` | regressor, causal GRU-128 | `simple_conv_v2_uni_gru128` |
| `hppnet` | salience, HPPNet port, per-rotor layers + CRF readout | `hppnet_rps_l4` |
| `hf0` | salience, HarmoF0 port, per-rotor layers + CRF readout | `harmof0_rps_l4` |

## The matrix

Experiment names per cell. "old" marks a row that predates this campaign.

| regime | sc | scv2 | tm | gru | hppnet | hf0 |
|---|---|---|---|---|---|---|
| R1 | `real_r1_sc` | `real_r1_scv2` | `real_r1_tm` | `real_r1_gru` | `real_r1_hppnet` | `real_r1_hf0` |
| R2 | `real_r2_sc` | `real_r2_scv2` | `real_r2_tm` | `real_r2_gru` | `real_r2_hppnet` | `real_r2_hf0` |
| R3 | `real_r3_sc` | `real_r3_scv2` | `real_r3_tm` | `real_r3_gru` | `real_r3_hppnet` | `real_r3_hf0` |
| R4 | `real_r4_sc` | `real_r4_scv2` (old: `hb_scv2_mag_nogate`) | `real_r4_tm` (old: `tm_r2hb_nogate`; IF: `r2hb_tr_nogate`) | `real_r4_gru` (old: `r2hb_gru_nogate`) | `hppnet_r2hb_l4` | `hf0_r2hb_l4` |
| R4 nomix | — | `r2hb_scv2_nomix` | `r2hb_tm_nomix` | `r2hb_gru_nomix` | `hppnet_r2hb_nomix` | `hf0_r2hb_nomix` |
| S1 nomix / mix | — | old `salv2_scv2_comb_{nomix,mix}` | `salv2_tr_comb_{nomix,mix}` | `salv2_gru_comb_{nomix,mix}` | old `salv2_hppnet_comb_{nomix,mix}` | old `salv2_hf0_comb_{nomix,mix}` |
| S2 nomix / mix | — | old `salv2_scv2_stoch_{nomix,mix}` | `salv2_tr_stoch_{nomix,mix}` | `salv2_gru_stoch_{nomix,mix}` | old `salv2_hppnet_stoch_{nomix,mix}` | old `salv2_hf0_stoch_{nomix,mix}` |
| C1 comb → R4 | — | old `r4hb_scv2` (+ `r4hb_seed1/2`) | `tm_comb_s1` → `tm_r4hb` (IF: `r4hb_tr`) | old `r4hb_gru` | old `hppnet_r4_l4` | `hf0_r4_l4_v2` (old `hf0_r4_l4` was BCE-selected) |
| C2 stoch → R4 | — | old `r6hb_scv2` | — | — | — | — |
| M | — | old `r7hb_scv2` | `r7hb_tm` | `r7hb_gru` | — | — |

The old rows `hb_scv2_mag_nogate`, `r2hb_gru_nogate` and `r2hb_tr_nogate`
trained on `hb_silence_dload.yaml`, which adds a 50k-sample unaugmented warm-up
stage to the R4 pool. The new `real_r4_*` rows drop it, so that rungs R1–R4 and
the speech A/B share one recipe. The old rows stay as a cross-check.

## Block S: the adaptation ladder of the multi-pitch baselines

One regime (R4, the old `hb_silence` pool of `hb_sal_multif0`), four
architectures, three levels: L0 = the published architecture (its own
log-frequency front-end, its harmonic device, a semitone-resolution shared map,
threshold + Hungarian decode); L1 = a finer output grid only (June recipe);
L2 = per-rotor Gaussian layers on the 0–150 rev/s grid with the CRF readout,
original input; L3 = L2 plus the comb gather on the linear STFT in place of the
log-axis dilated convolutions (HarmoF0 and HPPNet only; LateDeep and Basic
Pitch have no such convolutions).

| model | L0 | L1 | L2 | L3 |
|---|---|---|---|---|
| LateDeep | old `hb_sal_multif0` | old `hb_sal_multif0_nsr` | `hb_sal_multif0_l4` | n/a |
| Basic Pitch | old `hb_sal_bp` | June only | `hb_sal_bp_l4` | n/a |
| HarmoF0 | `hb_sal_hf0_orig` | — | (optional) | `hf0_r2hb_l4` |
| HPPNet | `hb_sal_hppnet_orig` | — | (optional) | `hppnet_r2hb_l4` |

### L0 for HarmoF0 and HPPNet: what "the published architecture" is

`src/models/harmonic_ports/{harmof0,hppnet}_orig.py`, registry keys
`harmof0_orig` and `hppnet_orig`. Both keep the paper's harmonic device and the
paper's frequency axis, which is the pair of things the L3 ports replaced:

- HarmoF0 — `WaveformToLogSpecgram` (a 1024-point STFT at 16 kHz, linearly
  interpolated onto the log bins), `MRDConv` at 12 harmonics, the three
  octave-dilated context blocks, the two 1x1 convolutions.
- HPPNet — the nnAudio CQT, `HarmonicDilatedConv` (eight branches at
  `round(log2(k) * 48)`), `CNNTrunk`, `FreqGroupLSTM`, the FRAME head only.

Both emit a bit-identical 352-bin grid — 48 bins per octave from 27.5 Hz — so
`conf/loss/salience_bce_orig.yaml` and `conf/metrics/salience_bce_orig.yaml`
serve both arms and the two differ only in the trunk and the front end. The
harmonic blocks are checked bit-identical against the upstream source in
`tests/models/test_harmonic_orig.py`.

The grid is 4 bins per semitone, not one: HarmoF0 emits its 352-bin map
directly, and HPPNet's `[1, 4]` frequency pool (which is what makes its output a
88-bin piano roll) is OFF, because a semitone bin is 5.95% of the rate — 2.4
rev/s at 40 rev/s, an order of magnitude past the campaign's 0.2 rev/s floor —
and the arm would read as a measurement of the pool. `freq_pool: 4` restores it.
The other deviations are seam-level (the hop-512 frame grid, logits instead of a
sigmoid, HPPNet's half-rate frame-subnet time pool, the dropped
onset/offset/velocity heads) and each is listed in the module docstrings.

Under `f0 = rps` this axis spans 27.5-4371 rev/s. Rotors occupy bins 0-118 of
352; the other 233 hold no fundamental, a stopped rotor has no bin at all (the
same dark-column convention `hb_sal_multif0` trains under), and one bin is 1.45%
of the rate — 0.40 rev/s at 27.5, 0.58 at 40, 2.2 at 150 — against the ports'
uniform 0.5017. HPPNet's CQT has the resolution but not the window: a
48-bin-per-octave filter at 27.5 Hz needs 2.5 s and the chunks are 1 s.

The monitor is `bce`, and it is the only one available: `SalienceBCEMetric` is
the whole shared-map metric surface, and `LayerPeakRPSMetric` reads per-rotor
layers, which is level L2. Rotor-speed error comes from `scripts/rps_dump.py`
and the PIT suite, as for every other L0 row. Batch 16 frames, as
`hb_sal_multif0`: a forward+backward probe at 1 s clips retains 1.72 GiB for
`harmof0_orig` and 2.80 GiB for `hppnet_orig`, against 2.03 GiB for `hppnet_rps`
under the same probe.

## Protocol

- Every checkpoint is dumped with `scripts/rps_dump.py` on the parts `comb`,
  `stoch`, `real`, `comb_speech`, `stoch_speech` and `real_nospeech`, and the
  dump must reproduce the W&B monitored value to three decimals.
- The frozen real split carries NO mixed LibriSpeech: `mode: real_valid` cuts
  the raw recordings. Its speech is acoustic. 14 of the 37 clips come from the
  DREGON `free-flight_speech-low_room1` and `free-flight_whitenoise-low_room1`
  flights, where a loudspeaker played into the room; the other 23 clips
  (`free-flight_nosource_room1` and FLY124) are rotor noise only.
  `real_nospeech` = `dload:DREGON-LM-V4-michaels-valid-full-nospeech`, the 23
  noise-only clips cut byte for byte from the published set (pin
  `01dfb417af7b…`). The speech A/B on real data therefore reads each pair
  (trained with / without mixed speech) on the 23 clean clips and on the 14
  loudspeaker clips separately.
- Caveat: re-deriving the published split from today's pinned frames does not
  reproduce it (DREGON labels moved by up to 16 rev/s when the frame adapter
  started to prefer `motors_measured`; the FLY124 audio alignment changed with
  the 2026-07-31 calibration). Every row of this campaign is scored on the
  published bytes, as every earlier row was.
- The real split is reported by rig (DREGON room1 / FLY124) × regime
  (zero-frames / below-30 / ramp in-grid / cruise in-grid / ground) ×
  microphone group (ch 0 / ch 1–7 / all) with `scripts/rps_regime_table.py`.
- The cue probes of `scripts/rps_cue_probe.py`: the frequency-scaling probe
  (six DREGON cruise clips, α ∈ [0.7, 1.3], slopes over the full range and
  within ±4 %) and the harmonic-cutoff probe (orders 80/40/20/10 on the
  stochastic part; MAE, true-rate and half-rate fractions).
- Error classes and the predicted-spread statistic:
  `scripts/rps_error_profile.py`, `scripts/spread_eval.py`.

## Compute

One burst on vast.ai (A100-80 instances, `omnirun --backend vast`), with the
two `uni-gpushort` slots for the SimpleConv rows. Every vast job prefetches
`DREGON-frames`, `michaels-frames` and `librispeech` with `dload pull` before
training, so the DataLoader workers read local shards (dload 0.3.0 has no retry
in its shard-open path; `data_processing.streams` adds one as a second guard).
Real-regime jobs get 4 h, salience real-regime jobs 8 h, synthetic-family
regressor jobs 10 h; a job cut by its time limit keeps its `best.ckpt`, and the
ledger marks it truncated.

## Job ledger

Filled as jobs land: experiment, job id, backend, epochs, truncated, spend.

## Results

### Batch 1 (2026-09-05, six cells; single seed)

Per-frame PIT MAE (rev/s) on the frozen real split, all 8 microphones, by regime
and rig. `r4hb_scv2` is the paper's best row, for reference.

| regime (frames) | r4hb_scv2 | r2hb_scv2_nomix | real_r1_scv2 | real_r1_tm | real_r1_gru | real_r2_sc | real_r3_sc |
|---|---|---|---|---|---|---|---|
| zero-frames (9296) | 2.86 | 6.24 | 0.41 | 4.07 | 5.32 | 10.20 | 4.37 |
| below-30 (2728) | 7.77 | 12.18 | 14.22 | 20.13 | 19.56 | 12.82 | 8.06 |
| DREGON:ramp:in-grid (4176) | 4.43 | 4.51 | 4.32 | 8.52 | 3.70 | 7.18 | 5.39 |
| FLY124:ramp:in-grid (5888) | 3.12 | 4.05 | 53.99 | 22.78 | 27.09 | 15.75 | 3.62 |
| DREGON:cruise:all (32128) | 2.92 | 3.93 | 2.73 | 2.28 | 2.30 | 3.55 | 2.27 |
| FLY124:cruise:all (20080) | 1.24 | 1.68 | 65.88 | 8.60 | 10.19 | 11.46 | 1.30 |
| all (74296) | 2.74 | 3.95 | 24.08 | 6.84 | 7.49 | 8.03 | 2.77 |

Microphone split (ch 0 / ch 1-7): the mic-0 models are NOT worse on the other
DREGON microphones at cruise (`real_r1_scv2` 2.73 / 2.73, `real_r1_tm` 2.37 /
2.27, `real_r1_gru` 2.40 / 2.29). The microphone effect appears on the unseen
drone only (`real_r1_tm` FLY124 cruise 4.45 on ch 0 against 9.19 on ch 1-7).
The unseen-drone failure is large for the convolutional trunk (65.9) and
moderate for the transformer and GRU heads (8.6, 10.2). Adding FLY125 (rung 3)
closes it even for SimpleConv, whose rung-3 row (2.77 all frames) ties the
paper's best row (2.74). Full table with the mic axis:
`results/rps_profile/ladder_batch1_real.csv`.

Other parts (mean PIT MAE over all frames; comb / stoch / comb+speech /
stoch+speech / real / real_nospeech):

| model | comb | stoch | comb+sp | stoch+sp | real | real_nospeech |
|---|---|---|---|---|---|---|
| r4hb_scv2 | 15.4 | 28.1 | 14.1 | 28.5 | 2.76 | pending |
| r2hb_scv2_nomix | 11.8 | 31.3 | 14.7 | 31.6 | 3.96 | 3.59 |
| real_r1_scv2 | 45.7 | 32.9 | 45.0 | 32.8 | 24.1 | 37.1 |
| real_r1_tm | 32.6 | 26.4 | 29.4 | 26.6 | 6.85 | 8.43 |
| real_r1_gru | 39.0 | 27.6 | 38.3 | 27.7 | 7.49 | 9.97 |
| real_r2_sc | 14.6 | 23.2 | 14.8 | 23.3 | 8.03 | 8.78 |
| real_r3_sc | 31.5 | 25.1 | 28.6 | 24.8 | 2.77 | 2.36 |

No real-only cell transfers to either synthetic family (claim 1, "and so on").

Frequency-scaling probe (slope over the full range / within +-4 %): rung 1-3
cells are flat (`real_r1_scv2` 0.12 / 0.00, `real_r1_tm` 0.07 / 0.21,
`real_r1_gru` 0.04 / 0.18, `real_r2_sc` 0.10 / -0.18, `real_r3_sc` -0.09 /
-0.32); the no-speech rung-4 cell follows the shift (`r2hb_scv2_nomix` 1.03 /
1.32) where its with-speech sibling does not (`hb_scv2_mag_nogate` 0.89 /
0.14). The harmonic-cutoff probe is uninformative for real-only cells: they
already score 23-34 on the stochastic part at the full comb.

Frequency-scaling probe over every row probed so far (slope full / within
+-4 %). Read only where the base prediction is sane: a model that returns
near-zero or garbage on these clips gives a meaningless relative change
(`salv2_scv2_comb_nomix` 25 / 7, `xrig_michaels_only` 19 / 18).

| row | recipe | full | local |
|---|---|---|---|
| `real_r1_scv2`, `real_r1_tm`, `real_r1_gru` | rung 1, no label-transforming augs | 0.12, 0.07, 0.04 | 0.00, 0.21, 0.18 |
| `real_r2_sc`, `real_r3_sc` | rungs 2-3 | 0.10, -0.09 | -0.18, -0.32 |
| `c9_simple_conv_v2_8ch` (June, the paper's "no augmentation" curve) | fixed mixtures | 0.03 | 0.16 |
| `scv2_fs_v2`, `r2hb_gru_nogate` | R2 recipe (freq-scale + time-warp), with speech | 0.05, 0.02 | 0.00, 0.00 |
| `r2hb_tr_nogate`, `hb_scv2_mag_nogate` | R2 recipe, with speech | 0.54, 0.89 | 0.25, 0.14 |
| `r2hb_scv2_nomix` | R2 recipe, no speech | 1.03 | 1.32 |
| `r4hb_scv2`, `r4hb_tr`, `r4hb_gru` | comb -> R2 curriculum | 1.04, 1.03, 1.01 | 1.02, 1.26, 1.35 |
| `r6hb_scv2`, `r7hb_scv2`, `xrig_dregon_only` | stoch -> R2; real+stoch+comb pool; comb -> DREGON only | 0.99, 0.83, 0.91 | 1.22, 1.58, 1.30 |
| `m3mixv2_scv2`, `m3mixv2_transformer`, `m3mixv2_unigru128` | real+gen+comb pool | 0.98, 0.90, 0.96 | 1.06, 1.61, 1.52 |
| `hppnet_r4_l4`, `hf0_r4_l4` | salience ports, comb -> R2 | 0.87, 0.80 | 0.87, 1.46 |

Reading: no real-only rung reads frequency; the label-transforming
augmentations alone give an inconsistent response across trunks (0.02-0.89
full, 0.00-0.25 local, with speech); every row that saw a synthetic comb
(curriculum or pool) reads frequency (0.83-1.04 full), the salience ports
included. The no-speech R2 cell reads frequency where its with-speech twin
does not; `real_r4_scv2` (with speech, no warm-up) decides whether that is
the speech or the warm-up stage.

### Batch 2 (rung 2 regressors, gpushort chains; single seed)

| regime (frames) | real_r1_scv2 | real_r2_scv2 | real_r1_gru | real_r2_gru | real_r1_tm | real_r2_tm | real_r3_sc | r4hb_scv2 |
|---|---|---|---|---|---|---|---|---|
| zero-frames (9296) | 0.41 | 3.26 | 5.32 | 5.10 | 4.07 | 6.48 | 4.37 | 2.86 |
| below-30 (2728) | 14.22 | 13.62 | 19.56 | 15.35 | 20.13 | 16.03 | 8.06 | 7.77 |
| DREGON:ramp:in-grid (4176) | 4.32 | 4.34 | 3.70 | 6.77 | 8.52 | 8.75 | 5.39 | 4.43 |
| FLY124:ramp:in-grid (5888) | 53.99 | 17.31 | 27.09 | 26.85 | 22.78 | 15.13 | 3.62 | 3.12 |
| DREGON:cruise:all (32128) | 2.73 | 2.58 | 2.30 | 3.86 | 2.28 | 2.74 | 2.27 | 2.92 |
| FLY124:cruise:all (20080) | 65.88 | 10.39 | 10.19 | 10.75 | 8.60 | 8.53 | 1.30 | 1.24 |
| all (74296) | 24.08 | 6.45 | 7.49 | 8.28 | 6.84 | 6.58 | 2.77 | 2.74 |

Rung 1 -> rung 2 (one microphone -> eight, same drone): DREGON cruise moves
little (scv2 2.73 -> 2.58, tm 2.28 -> 2.74, gru 2.30 -> 3.86); the unseen-drone
cruise error falls from 66 to 10 for the convolutional trunk and stays at
8.5-10.8 for the transformer and GRU heads. Rung 3 (+ FLY125) is what closes
it (SimpleConv 1.30). Microphone split at cruise: identical on ch 0 and ch 1-7
for every rung-1 and rung-2 cell on DREGON; on FLY124 the transformer reads
mic 0 better than the others at both rungs (4.45 / 9.19 and 4.30 / 9.14), so
that asymmetry belongs to the FLY124 array, not to the training microphones.
Frequency-scaling probe, rung 2: 0.01 / 0.01 (scv2), 0.00 / 0.00 (gru),
0.06 / 0.03 (tm) — flat, as at rung 1. Table with the mic axis:
`results/rps_profile/ladder_batch2_real.csv`.

Block S level L0 on the same protocol (all mics): `hb_sal_multif0` all 12.65
(zero 52.6, below-30 35.6, DREGON cruise 2.96, FLY124 cruise 4.46, ramps
20.7); `hb_sal_multif0_nsr` 11.82; `hb_sal_bp` 27.30 (cruise 26); the HPPNet
port after the comb curriculum (`hppnet_r4_l4`, level L3) 6.04.

### Batch 3 (rung-3 scv2): the convolutional ladder end to end (all mics)

| rung | zero-frames | below-30 | DREGON ramp | FLY124 ramp | DREGON cruise | FLY124 cruise | all |
|---|---|---|---|---|---|---|---|
| `real_r1_scv2` (mic 0) | 0.41 | 14.22 | 4.32 | 53.99 | 2.73 | 65.88 | 24.08 |
| `real_r2_scv2` (8 mics) | 3.26 | 13.62 | 4.34 | 17.31 | 2.58 | 10.39 | 6.45 |
| `real_r3_scv2` (+ FLY125) | 4.46 | 9.07 | 3.96 | 3.81 | 2.15 | 2.27 | 2.96 |
| `hb_scv2_mag_nogate` (rung 4, old R2 row) | 3.30 | 10.83 | 3.98 | 3.38 | 2.62 | 1.25 | 2.77 |
| `r4hb_scv2` (comb -> rung 4) | 2.86 | 7.77 | 4.43 | 3.12 | 2.92 | 1.24 | 2.74 |

Claim 2 in one column: from rung 3 on, every step that buys generality
(FLY124 cruise 2.27 -> 1.25 -> 1.24, ramps, below-30) costs DREGON cruise
precision (2.15 -> 2.62 -> 2.92). Rung 3 stays flat to the frequency probe
near the operating point (local 0.00; the full-range 1.18 is the breakdown at
the extremes) and does not transfer to the synthetic parts (comb 39.7, stoch
28.2). On the 23 noise-only clips it scores 2.77. `real_r4_scv2` (rung 4
without the warm-up stage) is running on gpushort.

### Batch 4 (rung-3 GRU, rung-4 scv2 without warm-up; all mics)

| row | zero | below-30 | DREGON ramp | FLY124 ramp | DREGON cruise | FLY124 cruise | all | probe full / local |
|---|---|---|---|---|---|---|---|---|
| `real_r1_gru` | 5.32 | 19.56 | 3.70 | 27.09 | 2.30 | 10.19 | 7.49 | 0.04 / 0.18 |
| `real_r2_gru` | 5.10 | 15.35 | 6.77 | 26.85 | 3.86 | 10.75 | 8.28 | 0.00 / 0.00 |
| `real_r3_gru` | 6.82 | 16.33 | 4.71 | 6.75 | 2.15 | 2.28 | 3.80 | 0.01 / 0.00 |
| `r2hb_gru_nogate` (rung 4, old R2 row) | 6.05 | 13.75 | 11.04 | 5.44 | 2.13 | 3.64 | 4.22 | 0.02 / 0.00 |
| `r4hb_gru` (comb -> rung 4) | 6.14 | 12.11 | 5.10 | 4.44 | 3.16 | 1.45 | 3.61 | 1.01 / 1.35 |
| `real_r4_scv2` (rung 4, augmentations from step 0) | 3.07 | 10.19 | 5.18 | 6.47 | 4.40 | 4.25 | 4.61 | 0.84 / 0.92 |
| `hb_scv2_mag_nogate` (rung 4, old R2 row, 50k warm-up) | 3.30 | 10.83 | 3.98 | 3.38 | 2.62 | 1.25 | 2.77 | 0.89 / 0.14 |

Readings. (1) For the causal GRU the label-transforming augmentations buy
nothing on real data: rung 3 beats the R2 row on FLY124 cruise (2.28 vs 3.64)
and on all frames (3.80 vs 4.22); only the comb curriculum moves it (1.45).
(2) The augmentation SCHEDULE decides what the convolutional model reads. With
the augmentations on from the first step (`real_r4_scv2`) the model reads
frequency (local slope 0.92) and loses precision everywhere (DREGON cruise
4.40 against 2.15 at rung 3; validation stalled at 4.6 from epoch 5 and
early-stopped at 29). With the old R2 schedule (a 50k-sample unaugmented
warm-up, about one epoch) the model stays prior-driven (0.14) and precise
(2.62 / 1.25 / 2.77). This is the precision-for-generality trade of claim 2
as a mechanism, on single seeds. Consequence for the matrix: rung 4 of the
paper is the existing R2 rows (with warm-up); the warm-up-free `real_r4_*`
rows are a schedule ablation; the speech A/B arms are re-run with the warm-up
(`r2hb_{scv2,tm,gru}_nomix_wu`, policy `hb_silence_nomix_dload.yaml`).

### Batch 5 (rung-4 transformer and GRU without warm-up; all mics)

| row | zero | below-30 | DREGON ramp | FLY124 ramp | DREGON cruise | FLY124 cruise | all | probe full / local |
|---|---|---|---|---|---|---|---|---|
| `real_r1_tm` | 4.07 | 20.13 | 8.52 | 22.78 | 2.28 | 8.60 | 6.84 | 0.07 / 0.21 |
| `real_r2_tm` | 6.48 | 16.03 | 8.75 | 15.13 | 2.74 | 8.53 | 6.58 | 0.06 / 0.03 |
| `real_r4_tm` (augmentations from step 0, magnitude) | 4.27 | 15.36 | 8.36 | 4.31 | 2.99 | 1.96 | 3.73 | 0.93 / 1.01 |
| `r2hb_tr_nogate` (old R2 row, IF, warm-up) | 5.46 | 15.34 | 6.73 | 3.39 | 2.59 | 1.41 | 3.39 | 0.54 / 0.25 |
| `r4hb_tr` (comb -> R2, IF) | 5.94 | 12.73 | 8.01 | 4.06 | 3.22 | 1.51 | 3.78 | 1.03 / 1.26 |
| `real_r4_gru` (augmentations from step 0) | 5.23 | 14.28 | 4.93 | 4.34 | 3.55 | 2.42 | 3.99 | 0.73 / 0.54 |
| `r2hb_gru_nogate` (old R2 row, warm-up) | 6.05 | 13.75 | 11.04 | 5.44 | 2.13 | 3.64 | 4.22 | 0.02 / 0.00 |

The schedule ablation holds on all three trunks: with the label-transforming
augmentations on from the first step the model reads frequency (local slope
0.92 / 1.01 / 0.54 for scv2 / transformer / GRU) and pays on DREGON cruise
(4.40 / 2.99 / 3.55 against 2.62 / 2.59 / 2.13 for the warm-up rows); the
warm-up rows keep the prior (0.14 / 0.25 / 0.00). The transformer's rung 3
(`real_r3_tm`) was interrupted on vast at epoch ~100 and resumes there.

### Batch 6-7: the speech A/B on real data (R2 schedule; single seed)

Split of the frozen real set by clip origin: DREGON room 1 with a loudspeaker
playing (clips 0-13, 14 clips), DREGON room 1 without a source (clips 14-21,
8 clips), FLY124 (clips 22-36, 15 clips). PIT MAE over all 8 mics.

| model | DREGON clean (8) | DREGON loudspeaker (14) | ratio | FLY124 (15) | whole split | probe full / local |
|---|---|---|---|---|---|---|
| scv2 with mixed speech (`hb_scv2_mag_nogate`) | 2.52 | 3.37 | 1.34 | 2.39 | 2.77 | 0.89 / 0.14 |
| scv2 without (`r2hb_scv2_nomix_wu`) | 2.78 | 3.54 | 1.27 | 2.56 | 2.98 | 0.81 / 0.36 |
| GRU with (`r2hb_gru_nogate`) | 2.97 | 2.98 | 1.00 | 6.08 | 4.23 | 0.02 / 0.00 |
| GRU without (`r2hb_gru_nomix_wu`) | 2.55 | 4.33 | 1.70 | 3.51 | 3.61 | 0.71 / 0.26 |
| transformer-mag with (`tm_r2hb_nogate`, the R2 rerun) | 3.06 | 4.04 | 1.32 | 2.53 | 3.21 | 0.40 / 0.40 |
| transformer-mag without (`r2hb_tm_nomix_wu`) | 3.02 | 3.95 | 1.31 | 2.86 | 3.31 | 0.01 / 0.02 |
| transformer-IF with (`r2hb_tr_nogate`, old R2 row) | 2.94 | 4.10 | 1.39 | 3.00 | 3.40 | 0.54 / 0.25 |

Readings: an acoustic talker or noise source in the room costs 1.3-1.4x for
models trained with mixed speech and 1.7x for the one model that never saw
speech (the causal GRU), which the training speech makes immune to it (1.00)
at the price of its FLY124 cell (6.08 vs 3.51; single seed). Training with
mixed speech helps the convolutional trunk by 5-10 % on every subset; for the
transformer it helps on FLY124 only (2.53 vs 2.86) and is a wash on DREGON.
The magnitude transformer beats its IF predecessor on the same regime (3.21
vs 3.40 all frames), which settles decision 5.
The warm-up-free pair (`r2hb_scv2_nomix` 3.95 vs `real_r4_scv2` 4.61) is a
schedule artifact and is not used. Together with the synthetic pairs (2x on
the stochastic family, 1.0-1.65x on the static comb), claim 5 reads "1.3-2x,
family-dependent", not "2x for all models and regimes".

### Batches 8-9: the static-comb cells of all four trunks (salv2 streams; single seed)

PIT MAE on the salv2 comb validation set; "comb+sp" = the same 32 flights with
a LibriSpeech talker at -30..0 dB at evaluation.

| trained without speech | comb | comb+sp | ratio | trained with speech | comb | comb+sp | ratio |
|---|---|---|---|---|---|---|---|
| `salv2_scv2_comb_nomix` | 0.91 | 1.60 | 1.8 | `salv2_scv2_comb_mix` | 0.83 | 0.87 | 1.05 |
| `salv2_tr_comb_nomix` | 1.22 | 2.32 | 1.9 | `salv2_tr_comb_mix` | 1.34 | 1.42 | 1.06 |
| `salv2_hppnet_comb_nomix` | 0.46 | 1.42 | 3.1 | `salv2_hppnet_comb_mix` | 0.48 | 0.55 | 1.15 |
| `salv2_hf0_comb_nomix` | 1.19 | 2.02 | 1.7 | `salv2_hf0_comb_mix` | 0.87 | 0.98 | 1.13 |

Every comb-trained row scores 36-38 on the stochastic part and 37-66 on real
audio: no transfer across synthetic families or to real data. Claim 5, final
form across synthetic and real data: a talker makes the task 1.7-3x harder
for a model that never saw speech, and 1.05-1.4x for a model trained with
mixed speech, which stays as precise on clean input. Speech in training is a
robustness ingredient, not a handicap.

### Batch 11: first salience rungs (all mics; single seed)

| row | zero | below-30 | DREGON ramp | FLY124 ramp | DREGON cruise | FLY124 cruise | all | probe full / local |
|---|---|---|---|---|---|---|---|---|
| `real_r1_hppnet` (mic 0, DREGON) | 2.86 | 29.28 | 10.68 | 54.49 | 2.91 | 71.22 | 26.86 | 0.38 / 0.71 |
| `hppnet_r2hb_nomix` (rung 4 pool, no speech, no warm start) | 2.90 | 27.62 | 10.48 | 3.07 | 2.65 | 0.80 | 3.57 | 0.97 / 0.99 |
| `hppnet_r4_l4` (comb -> rung 4, old) | 8.93 | 27.38 | 17.14 | 11.96 | 3.77 | 1.40 | 6.04 | 0.87 / 0.87 |
| `hf0_r2hb_l4` (rung 4 pool with speech, no warm start) | 13.96 | 24.28 | 16.62 | 12.90 | 6.22 | 2.29 | 7.90 | 1.01 / 0.95 |
| `hf0_r4_l4` (comb -> rung 4, BCE-selected, old) | 1.49 | 17.46 | 58.27 | 27.77 | 51.79 | 8.16 | 30.91 | 0.80 / 1.46 |

Readings: the HPPNet port trained on the real pool alone (no comb stage, no
speech) is the best salience row and the best FLY124-cruise cell of the
campaign (0.80), reads frequency, and beats its comb-curriculum twin on every
regime: the synthetic stage hurt the port. The mic-0 HPPNet fails on the
unseen drone like the regressors (71) but already half-reads frequency
(0.71 local) without any augmentation, which the regressors never do. The
HarmoF0 port trails HPPNet by 2x on the same recipe.

### Batch 12: HarmoF0 rungs (all mics; single seed)

| row | zero | below-30 | DREGON ramp | FLY124 ramp | DREGON cruise | FLY124 cruise | all | probe full / local |
|---|---|---|---|---|---|---|---|---|
| `real_r1_hf0` (mic 0, DREGON) | 18.09 | 31.21 | 26.15 | 55.91 | 19.43 | 71.55 | 37.05 | 0.32 / 0.28 |
| `hf0_r2hb_l4` (rung 4 pool, with speech) | 13.96 | 24.28 | 16.62 | 12.90 | 6.22 | 2.29 | 7.90 | 1.01 / 0.95 |
| `hf0_r2hb_nomix` (rung 4 pool, no speech) | 19.64 | 31.97 | 16.38 | 11.44 | 4.65 | 2.31 | 8.09 | 1.03 / 1.12 |
| `real_r1_hppnet` (reference, same rung) | 2.86 | 29.28 | 10.68 | 54.49 | 2.91 | 71.22 | 26.86 | 0.38 / 0.71 |
| `hppnet_r2hb_nomix` (reference, same pool) | 2.90 | 27.62 | 10.48 | 3.07 | 2.65 | 0.80 | 3.57 | 0.97 / 0.99 |

Readings: HarmoF0 loses to HPPNet on every regime of the same recipe, most
of all on silence (14-20 vs 2.9, phantom rotors) and on cruise (4.7-6.2 vs
2.65); on the mic-0 rung it is not usable even on DREGON cruise (19.4).
Speech in training is a wash for it (7.90 vs 8.09).

### Batch 13: HPPNet speech pair, transformer comb curriculum, Basic Pitch l4 (all mics; single seed)

| row | zero | below-30 | DREGON ramp | FLY124 ramp | DREGON cruise | FLY124 cruise | all | DREGON clean / loudspeaker | probe full / local |
|---|---|---|---|---|---|---|---|---|---|
| `hppnet_r2hb_l4` (rung 4 pool, with speech) | 5.95 | 26.65 | 9.83 | 4.77 | 2.95 | 0.92 | 4.18 | 4.64 / 5.65 | 0.97 / 0.97 |
| `hppnet_r2hb_nomix` (no speech) | 2.90 | 27.62 | 10.48 | 3.07 | 2.65 | 0.80 | 3.57 | 3.90 / 4.86 | 0.97 / 0.99 |
| `tm_r4hb` (magnitude transformer, comb -> rung 4) | 5.14 | 12.91 | 8.30 | 3.20 | 2.65 | 1.43 | 3.37 | 3.05 / 4.28 | 0.89 / 1.40 |
| `tm_r2hb_nogate` (magnitude transformer, rung 4) | 4.65 | 13.05 | 6.22 | 3.33 | 2.79 | 1.21 | 3.21 | 3.06 / 4.04 | 0.40 / 0.40 |
| `hb_sal_bp_l4` (Basic Pitch, L2) | 0.50 | 18.02 | 48.57 | 32.42 | 43.88 | 9.47 | 27.56 | 36.6 / 37.4 | 2.31 / 3.08 |
| `hb_sal_bp` (Basic Pitch, L0) | 33.86 | 38.68 | 35.45 | 16.54 | 26.39 | 25.65 | 27.30 | — | — |

Readings: (1) mixed speech in training HURTS the HPPNet port on every regime,
the loudspeaker clips included (5.65 vs 4.86): the opposite of the
convolutional regressor. (2) The comb curriculum does not help the magnitude
transformer (3.37 vs 3.21), as it did not help the IF one (3.78 vs 3.40); it
makes the model read frequency (local slope 1.40 vs 0.40) at the same
precision. (3) Basic Pitch with the per-rotor layers is as unusable as the
published one (27.6 vs 27.3): it decides silence perfectly (0.50) and cannot
place cruise speeds (43.9). Its problem is the input representation, which
its architecture gives no handle on (no harmonic convolutions to replace), so
block S ends for it at L2.

## Conclusion

Pending.
