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
| R4 nomix | — | `r2hb_scv2_nomix_wu` (warm-up-free: `r2hb_scv2_nomix`) | `r2hb_tm_nomix_wu` | `r2hb_gru_nomix_wu` | `hppnet_r2hb_nomix` | `hf0_r2hb_nomix` |
| S1 nomix / mix | — | old `salv2_scv2_comb_{nomix,mix}` | `salv2_tr_comb_{nomix,mix}` | `salv2_gru_comb_{nomix,mix}` | old `salv2_hppnet_comb_{nomix,mix}` | old `salv2_hf0_comb_{nomix,mix}` |
| S2 nomix / mix | — | old `salv2_scv2_stoch_{nomix,mix}` | `salv2_tr_stoch_{nomix,mix}` | `salv2_gru_stoch_{nomix,mix}` | old `salv2_hppnet_stoch_{nomix,mix}` | old `salv2_hf0_stoch_{nomix,mix}` |
| C1 comb → R4 | — | old `r4hb_scv2` (+ `r4hb_seed1/2`) | `tm_comb_s1` → `tm_r4hb` (IF: `r4hb_tr`) | old `r4hb_gru` | old `hppnet_r4_l4` | `hf0_r4_l4_v2` (old `hf0_r4_l4` was BCE-selected) |
| C2 stoch → R4 | — | old `r6hb_scv2` | — | — | — | — |
| M | — | old `r7hb_scv2` | `r7hb_tm` | `r7hb_gru` | — | — |

The no-speech regressor rows are the `_wu` twins (with the 50k warm-up of
the old R4 rows, so that the speech A/B compares like with like); the
warm-up-free `r2hb_tm_nomix` and `r2hb_gru_nomix` were planned but never
trained, and `r2hb_scv2_nomix` (warm-up-free) stays as the schedule
ablation's no-speech arm.

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
- Error classes and the predicted-spread ("fan") statistic:
  `scripts/rps_error_profile.py` (`fan.csv`, `fan_slope`), `scripts/spread_eval.py`.
- Every claim table is emitted by `python scripts/rps_claim_tables.py` into
  `results/paper_regime_matrix/` (`ladder.csv`, `speech_ab.csv`, `blocks.csv`,
  `stochastic.csv`, `missing.txt`, `claims.md`); the tables in the batches
  below were typed from the same tool outputs and are reproduced by it.

## Compute

Plan: one 10-12 h burst on vast.ai A100-80 instances (`omnirun --backend
vast`, every job prefetching `DREGON-frames`, `michaels-frames` and
`librispeech` with `dload pull`), with the two `uni-gpushort` slots for
the SimpleConv rows. What happened (2026-09-05): the vast account ran dry
after about $180 (eight runs interrupted with `best.ckpt` and
`train_state.pt` on R2), ~60 % of vast placements failed on slow hosts
until `ssh_wait_timeout_s`/`provision_attempts` were raised, and a
follow-up $10 plan on 4090 hosts lost 9 of 17 rows to Cloudflare R2
egress failures. The free cluster carried the campaign instead: the
`uni-gpushort` partition as chains of 55-minute segments
(`scripts/chain_train.sh`, `resume=true` restoring optimizer state, two
chains at a time) and the `sae` partition by direct `sbatch` (the daemon
keeps only four `uni` jobs in flight). Two direct `sae` jobs started
without R2 credentials (a `set -a` omission), trained from scratch and
uploaded nothing; their checkpoints were copied to R2 by hand (the vast
fragments kept in `checkpoints_vast_ep32/` and `checkpoints_prev/`).
Total spend on vast.ai: about $200; every other row cost nothing.
Time-boxing: a run cut by its wall keeps `best.ckpt`; the one such case
(`salv2_tr_stoch_nomix` at the 10 h wall, epoch 180) was resumed from R2
by a gpushort chain and early-stopped at epoch 186.

## Job ledger

Final backend of every trained row, from the host name of its W&B run
(25 cluster gpushort chain, 15 cluster sae, 8 vast.ai; the vast rows are the survivors of the two bursts, the
rest of the vast attempts were resumed or restarted on the cluster).
Epochs are the W&B epoch count of the surviving run; "W&B runs" counts
runs under the name (interrupted or crashed attempts included).
Evaluation jobs: 23 dump batches on `uni-gpushort`, 23 probe batches on
`uni-cpu`, the OT recompute and two phase-probe runs on `uni-cpu`
(`burst/eval_ledger.tsv` in the session scratchpad).

| group | experiment | final backend | GPU | epochs | W&B runs |
|---|---|---|---|---|---|
| Real ladder, regressors | `real_r1_sc` | cluster gpushort chain | V100-16 | 38 | 1 |
| Real ladder, regressors | `real_r1_scv2` | cluster gpushort chain | A100-80 | 36 | 1 |
| Real ladder, regressors | `real_r1_tm` | vast.ai | A100-80 | 29 | 1 |
| Real ladder, regressors | `real_r1_gru` | cluster gpushort chain | A100-80 | 33 | 1 |
| Real ladder, regressors | `real_r2_sc` | cluster gpushort chain | A100-80 | 30 | 1 |
| Real ladder, regressors | `real_r2_scv2` | cluster gpushort chain | A100-40 | 39 | 1 |
| Real ladder, regressors | `real_r2_tm` | cluster gpushort chain | A100-40 | 36 | 1 |
| Real ladder, regressors | `real_r2_gru` | cluster gpushort chain | A100-80 | 21 | 1 |
| Real ladder, regressors | `real_r3_sc` | cluster gpushort chain | A100-40 | 106 | 1 |
| Real ladder, regressors | `real_r3_scv2` | cluster gpushort chain | A100-80 | 27 | 1 |
| Real ladder, regressors | `real_r3_tm` | cluster sae | A100-80 | 36 | 3 |
| Real ladder, regressors | `real_r3_gru` | cluster gpushort chain | A100-80 | 27 | 1 |
| Real ladder, regressors | `real_r4_sc` | cluster gpushort chain | A100-40 | 50 | 1 |
| Real ladder, regressors | `real_r4_scv2` | cluster gpushort chain | A100-80 | 28 | 1 |
| Real ladder, regressors | `real_r4_tm` | cluster gpushort chain | V100-32 | 29 | 1 |
| Real ladder, regressors | `real_r4_gru` | cluster gpushort chain | V100-16 | 25 | 1 |
| Real ladder, salience ports | `real_r1_hppnet` | vast.ai | RTX 4090 | 26 | 1 |
| Real ladder, salience ports | `real_r1_hf0` | vast.ai | RTX 4090 | 45 | 1 |
| Real ladder, salience ports | `real_r2_hppnet` | vast.ai | RTX 4090 | 35 | 1 |
| Real ladder, salience ports | `real_r2_hf0` | cluster gpushort chain | A100-40 | 42 | 2 |
| Real ladder, salience ports | `real_r3_hppnet` | cluster gpushort chain | A100-80 | 40 | 2 |
| Real ladder, salience ports | `real_r3_hf0` | vast.ai | RTX 4090 | 38 | 1 |
| Real ladder, salience ports | `hppnet_r2hb_l4` | cluster sae | A100-80 | 41 | 1 |
| Real ladder, salience ports | `hf0_r2hb_l4` | vast.ai | RTX 4090 | 61 | 1 |
| No-speech twins | `r2hb_scv2_nomix` | vast.ai | A100-80 | 48 | 1 |
| No-speech twins | `r2hb_scv2_nomix_wu` | cluster gpushort chain | V100-16 | 53 | 1 |
| No-speech twins | `r2hb_tm_nomix_wu` | cluster gpushort chain | A100-40 | 26 | 1 |
| No-speech twins | `r2hb_gru_nomix_wu` | cluster gpushort chain | A100-40 | 62 | 1 |
| No-speech twins | `hppnet_r2hb_nomix` | cluster sae | A100-80 | 72 | 1 |
| No-speech twins | `hf0_r2hb_nomix` | cluster sae | A100-80 | 43 | 2 |
| Synthetic cells | `salv2_tr_comb_nomix` | cluster sae | A100-80 | 42 | 2 |
| Synthetic cells | `salv2_tr_comb_mix` | cluster sae | A100-80 | 40 | 2 |
| Synthetic cells | `salv2_gru_comb_nomix` | cluster sae | A100-80 | 134 | 2 |
| Synthetic cells | `salv2_gru_comb_mix` | cluster sae | A100-80 | 100 | 1 |
| Synthetic cells | `salv2_tr_stoch_nomix` | cluster gpushort chain | A100-80 | 185 | 2 |
| Synthetic cells | `salv2_tr_stoch_mix` | cluster sae | A100-80 | 98 | 2 |
| Synthetic cells | `salv2_gru_stoch_nomix` | cluster sae | A100-80 | 123 | 2 |
| Synthetic cells | `salv2_gru_stoch_mix` | cluster sae | A100-80 | 133 (running) | 2 |
| Curricula and pools | `tm_comb_s1` | cluster sae | A100-80 | 42 | 2 |
| Curricula and pools | `tm_r4hb` | cluster sae | A100-80 | 43 | 1 |
| Curricula and pools | `tm_r2hb_nogate` | cluster gpushort chain | V100-16 | 50 | 1 |
| Curricula and pools | `hf0_r4_l4_v2` | cluster gpushort chain | A100-40 | 26 | 2 |
| Curricula and pools | `r7hb_tm` | cluster gpushort chain | A100-40 | 24 | 1 |
| Curricula and pools | `r7hb_gru` | cluster gpushort chain | A100-40 | 33 | 1 |
| Block S | `hb_sal_multif0_l4` | cluster gpushort chain | A100-80 | 37 (running) | 3 |
| Block S | `hb_sal_bp_l4` | cluster sae | A100-80 | 40 | 1 |
| Block S | `hb_sal_hf0_orig` | cluster sae | A100-80 | 32 | 1 |
| Block S | `hb_sal_hppnet_orig` | vast.ai | RTX 4090 | 33 | 1 |

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
| `salv2_gru_comb_nomix` (batch 15b) | 0.70 | 1.56 | 2.2 | `salv2_gru_comb_mix` (batch 15) | 0.91 | 0.93 | 1.02 |

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

### Batch 14: block S, the published architectures against their ports; HPPNet rung 2 (all mics; single seed)

| row | zero | below-30 | DREGON ramp | FLY124 ramp | DREGON cruise | FLY124 cruise | all |
|---|---|---|---|---|---|---|---|
| `hb_sal_hppnet_orig` (HPPNet as published, L0) | 17.48 | 17.52 | 40.14 | 7.50 | 3.81 | 1.63 | 7.77 |
| `hppnet_r2hb_l4` (HPPNet port, L3, same pool) | 5.95 | 26.65 | 9.83 | 4.77 | 2.95 | 0.92 | 4.18 |
| `hb_sal_hf0_orig` (HarmoF0 as published, L0) | 14.77 | 18.92 | 37.65 | 8.13 | 11.50 | 1.92 | 10.79 |
| `hf0_r2hb_l4` (HarmoF0 port, L3, same pool) | 13.96 | 24.28 | 16.62 | 12.90 | 6.22 | 2.29 | 7.90 |
| `hb_sal_multif0` (LateDeep, L0) | 52.57 | 35.63 | 14.22 | 18.66 | 2.96 | 4.46 | 12.65 |
| `hb_sal_bp` (Basic Pitch, L0) | 33.86 | 38.68 | 35.45 | 16.54 | 26.39 | 25.65 | 27.30 |
| `real_r2_hppnet` (HPPNet port, rung 2) | 1.77 | 25.85 | 13.16 | 54.28 | 2.41 | 71.15 | 26.48 |

Readings: the modifications (comb gather on the linear STFT + per-rotor layers
with the CRF readout) halve HPPNet's error (7.77 -> 4.18), mostly on silence
(17.5 -> 6.0) and DREGON ramps (40 -> 10), and cut HarmoF0's DREGON cruise
error in half (11.5 -> 6.2). Both published log-axis models are already far
ahead of LateDeep and Basic Pitch on this task, so the ranking of the
multi-pitch baselines is set by their harmonic device before any adaptation.
The rung-2 HPPNet has the best DREGON cruise of the ports (2.41) and, like
every rung-1/2 row, no transfer to the unseen drone.

Frequency-scaling probe on the block-S rows (slope full / within +-4 %):
`hb_sal_hppnet_orig` 1.01 / 0.98, `hppnet_r2hb_l4` 0.97 / 0.97,
`hppnet_r2hb_nomix` 0.97 / 0.99, `real_r2_hppnet` 0.83 / 0.55,
`real_r1_hppnet` 0.38 / 0.71; `hb_sal_hf0_orig` 0.34 / -0.14,
`hf0_r2hb_l4` 1.01 / 0.95; `hb_sal_multif0` 0.11 / 0.43,
`hb_sal_multif0_nsr` 0.38 / -0.57, `hb_sal_bp` 0.12 / -1.28.

Reading: HPPNet reads frequency as published and as a port, on real data
alone (the harmonic dilated convolutions on the log axis are a comb by
construction, so the probe answers 1 with no synthetic data). The
published HarmoF0 does not (0.34), and its port does (1.01): the comb
gather on the linear STFT is what makes it read frequency. LateDeep and
Basic Pitch do not read frequency in any adaptation. The harmonic-cutoff
probe is not informative for these real-only rows: `real_r2_hppnet`
returns zero on the synthetic probe clip (MAE 57 at every cutoff) and the
L0 rows score 12-27 at the full comb.

### Batch 15: SimpleConv rung 4, HarmoF0 rung 3, GRU static comb with speech (all mics; single seed)

| row | zero | below-30 | DREGON ramp | FLY124 ramp | DREGON cruise | FLY124 cruise | all | probe full / local |
|---|---|---|---|---|---|---|---|---|
| `real_r4_sc` (SimpleConv, rung 4 recipe, no warm-up) | 3.92 | 8.68 | 5.96 | 4.11 | 2.93 | 2.00 | 3.28 | 0.91 / 0.93 |
| `real_r3_sc` (rung 3, for comparison) | 4.37 | 8.06 | 5.39 | 3.62 | 2.27 | 1.30 | 2.77 | -0.09 / -0.32 |
| `real_r3_hf0` (HarmoF0 port, rung 3) | 6.15 | 21.58 | 21.84 | 17.11 | 5.38 | 2.38 | 7.11 | 0.61 / 0.24 |
| `hf0_r2hb_l4` (HarmoF0 port, rung 4, for comparison) | 13.96 | 24.28 | 16.62 | 12.90 | 6.22 | 2.29 | 7.90 | 1.01 / 0.95 |

Synthetic parts: `real_r4_sc` comb 10.9 / stoch 22.9 (rung 3: 31.5 / 25.1),
`real_r3_hf0` comb 47.8 / stoch 36.2. The GRU static-comb cell trained with
speech, `salv2_gru_comb_mix`: comb 0.91, comb+sp 0.93 (ratio 1.02), stoch
37.6, real 58.5; its probe is meaningless (the base prediction on the real
clips is garbage).

Readings. The SimpleConv obeys the ladder rule of the other regressors: the
label-transforming augmentations of rung 4 make it read frequency (local
slope -0.32 -> 0.93) and transfer to the static comb (31.5 -> 10.9), and
they cost precision on the real split (all 2.77 -> 3.28, DREGON cruise
2.27 -> 2.93, FLY124 cruise 1.30 -> 2.00; only the zero frames improve,
4.37 -> 3.92). HarmoF0 reaches its
rung-4 level at rung 3 already (7.11 against 7.90), so for this port the
augmentations buy no real-split precision; they only move the probe from
0.61 / 0.24 to 1.01 / 0.95. The HarmoF0 ladder (all-mic MAE): rung 1 37.05,
rung 3 7.11, rung 4 7.90; rung 2 is pending.

Checkpoint incident. The first cluster run of `salv2_gru_comb_nomix` (Slurm
25550715) started without R2 credentials: it could not restore the vast
fragment, trained from scratch (134 epochs, W&B best 0.699) and uploaded
nothing, so the R2 `best.ckpt` was still the vast fragment (epoch 32) and
the batch-15 dump scored that fragment (comb 1.08). The cluster run's
`best.ckpt`, `last.ckpt` and `train_state.pt` were copied to R2 by hand
(the vast fragment is kept in `checkpoints_vast_ep32/`), and the cell was
dumped again as batch 15b. `salv2_gru_stoch_nomix` (Slurm 25550718) has the
same condition and gets the same treatment when it finishes.

Batch 15b, `salv2_gru_comb_nomix` on the corrected checkpoint: comb 0.70
(W&B best 0.699, last 0.701), comb+sp 1.56 (ratio 2.2), stoch 37.6, real
58.8. The GRU pair joins the batches 8-9 speech table above: with speech in
training the ratio falls from 2.2 to 1.02 at a comb cost of 0.70 -> 0.91.

### Batch 15c: SimpleConv rung 1 (the last regressor rung; single seed)

`real_r1_sc` had no dump until now. All mics: zero 5.29 | below-30 11.77 |
DREGON ramp 7.60 | FLY124 ramp 29.14 | DREGON cruise 5.74 | FLY124 cruise
35.40 | all 15.88; comb 35.6, stoch 23.6; probe 0.16 / 0.05. Microphone
split at DREGON cruise: ch 0 3.56, ch 1-7 6.06.

Reading. The June SimpleConv is the one mic-0 model that IS worse on the
other DREGON microphones at cruise (3.56 -> 6.06, 1.7x), where the scv2,
transformer and GRU mic-0 rows are not (2.73 / 2.73, 2.37 / 2.27, 2.40 /
2.29). The unseen drone fails for all of them (FLY124 cruise 35.4 here).
The complete SimpleConv ladder, all-mic MAE / DREGON cruise / FLY124
cruise: rung 1 15.88 / 5.74 / 35.40, rung 2 8.03 / 3.55 / 11.46, rung 3
2.77 / 2.27 / 1.30, rung 4 3.28 / 2.93 / 2.00. Rung 3 is its best real
split, as for every other trunk, and rung 4 buys the probe (0.93) and the
static comb (10.9) with real-split precision.

### Batch 16a: the transformer in the one-pool mix (real + comb + stoch, no curriculum; single seed)

| row | zero | below-30 | DREGON ramp | FLY124 ramp | DREGON cruise | FLY124 cruise | all | comb / stoch | probe full / local |
|---|---|---|---|---|---|---|---|---|---|
| `r7hb_tm` (gpushort chain, 4 segments, 25 epochs) | 4.77 | 13.85 | 19.78 | 12.87 | 4.87 | 7.73 | 7.43 | 4.9 / 18.9 | 1.04 / 1.11 |
| `r7hb_scv2` (old row, for comparison) | 13.23 | 11.98 | 8.49 | 10.05 | 2.88 | 5.31 | 6.05 | 4.9 / 10.9 | 0.83 / 1.58 |

Reading: the one-pool mix reads frequency and is the only regime that
transfers to both synthetic families at once (comb 4.9, stoch 11-19), and
it pays for that on the real split: all 7.43 against 3.21 for the
transformer's rung-4 row, DREGON cruise 4.87 against 2.79, FLY124 cruise
7.73 against 1.21. The staging of the curricula (C1: 3.37 all, 2.65 / 1.43
cruise) is what keeps the real precision. `r7hb_gru` is pending.

### Batch 16b (in part): the leaderboard under one script; compute moves

The training-free rows and the blind tracker now live in the same dump
format as every neural row (`scripts/rps_dump_adopt.py`): the blind
tracker's stored trajectories are adopted as they are (8-channel units
replicated over the 8 microphone samples of a clip), the five classical
estimators are recomputed (their stored outputs held only per-regime
error sums), and the optimal-transport row (`otmp`) is recomputed on
`uni-cpu` (`eval-otmp-a7e0c7`, ~30 CPU-h). Under each source's own
protocol the adopted files reproduce the stored summaries of
`docs/experiments/unified-baseline-eval.md` exactly. Under this
campaign's readout (`scripts/rps_regime_table.py`: one PIT assignment per
8-second sample instead of per time frame, and the `below-30` cut
instead of the doc's `low` pool) the cells move: NMF flight 8.1 ->
DREGON cruise 11.6 / FLY124 cruise 17.8, blind tracker (no gates)
cruise 2.27 -> 0.92 / 9.19, gated zero 0.01 (unchanged). The paper's
leaderboard (`writing/papers/2026-08_wrapup/src/tables/leaderboard.tex`,
generated by `make_tables.py` from these dumps) reports cruise per rig
for this reason.

`r7hb_gru` finished on its gpushort chain (early stop at epoch 33, W&B
best 5.88). Its row (batch 16b, all mics): zero 11.92 | below-30 13.84 |
DREGON ramp 12.44 | FLY124 ramp 8.79 | DREGON cruise 3.39 | FLY124
cruise 3.75 | all 5.88; comb 3.3 / stoch 14.0; probe 0.94 / 1.50. The
one-pool mix is now complete for the three regressor trunks (all-frame
6.05 / 7.43 / 5.88 for scv2 / tm / gru): every trunk reads frequency and
transfers to both synthetic families, and every trunk loses 2-4 rev/s on
the real split against its rung-4 row.

Compute: after today's GPU use the account's Slurm priority on `sae`
fell from rank ~10 to ~420 of 453 (the five pending direct rows got a
start estimate of 2026-09-08), so `real_r3_tm`, `real_r3_hppnet`,
`real_r2_hf0`, `hf0_r4_l4_v2` and `hb_sal_multif0_l4` were re-queued as
`uni-gpushort` chains (10 x 55 min max, two at a time); the `sae`
copies stay queued as a fallback and exit at once if the chain finished
the run. The four stochastic regressor cells keep running on `sae`
(transformer pair under a 10 h wall, GRU pair under 12 h).

The paper: the ICASSP version (`writing/papers/2026-08_wrapup`, commit
`eadd447` of the submodule) is 4 pages + references, built around the
five claims; every table is generated from the claim CSVs by
`make_tables.py`, the probe figure by `make_probe_fig.py`, and pending
cells print as dashes until their dumps land.

### Batch 18: the transformer's rung 3 (gpushort chain, 5 segments; all mics; single seed)

| row | zero | below-30 | DREGON ramp | FLY124 ramp | DREGON cruise | FLY124 cruise | all | comb / stoch | probe full / local |
|---|---|---|---|---|---|---|---|---|---|
| `real_r3_tm` (+ FLY125, no label-transforming augs) | 4.91 | 14.93 | 5.88 | 3.65 | 2.42 | 1.48 | 3.23 | 28.1 / 27.3 | -0.17 / 0.21 |
| `tm_r2hb_nogate` (rung 4, warm-up) | 4.65 | 13.05 | 6.22 | 3.33 | 2.79 | 1.21 | 3.21 | 18.1 / 31.4 | 0.40 / 0.40 |
| `real_r4_tm` (rung 4, augs from step 0) | 4.27 | 15.36 | 8.36 | 4.31 | 2.99 | 1.96 | 3.73 | 9.1 / 29.3 | 0.93 / 1.01 |

The transformer ladder (all-mic MAE / DREGON cruise / FLY124 cruise /
probe local): rung 1 6.84 / 2.28 / 8.60 / 0.21, rung 2 6.58 / 2.74 / 8.53
/ 0.03, rung 3 3.23 / 2.42 / 1.48 / 0.21, rung 4 3.21 / 2.79 / 1.21 /
0.40. For this trunk rung 3 already reaches the rung-4 all-frame error,
so the label-transforming augmentations buy only the probe (0.21 -> 0.40
with warm-up, 1.01 from step 0) at a DREGON cruise cost of 2.42 -> 2.79
(-> 2.99 from step 0). Same shape as the convolutional ladder: the second
drone is the ingredient that fixes the unseen rig (FLY124 cruise 8.5 ->
1.5), the augmentations are the ingredient that fixes the probe.

### The optimal-transport row and the phase-coherence probe (2026-09-05, evening)

`otmp` (inverse harmonic clustering) was recomputed on `uni-cpu`
(`eval-otmp-a7e0c7`, 296 per-channel units, resampled from its 1 s frames
to the 2048/512 grid) and assembled locally: zero 69.2 | below-30 49.1 |
ramp 18.7 | DREGON cruise 14.4 | FLY124 cruise 18.6 | all 24.2 (the doc's
own-protocol all-frame value was 24.5). The leaderboard's training-free
side is complete under the one script.

For the journal version (OJSP), a phase-coherence probe of the harmonics
of one rotor was built on the WP18 covariance estimator
(`scripts/phase_coherence_probe.py`, statistics in
`src/tracking/phase_noise.py`: envelope autocorrelation -> coherence time
and Lorentzian half width per harmonic, Lorentzian-vs-Gaussian line-shape
fit, cross-harmonic correlation of the rate opinions and the rank-one
share, Cauchy-vs-Gaussian tail statistics of the per-harmonic residual).
Conditions: single motor on the bench (DREGON `motor_Motor{1-4}_{50..90}`,
20 units), all four motors on the bench (`motor_allMotors_70`, 4 units),
free flight from the TRAINING split only (DREGON room 2, 7 windows x 4
rotors, telemetry refined by `pi_kalman`; FLY125, 7 windows x 4 rotors,
refined sidecar). Job `phase-coherence-4fe127` on `uni-cpu`; outputs
`results/phase_coherence/{summary.csv,conditions.csv,report.md}`.

### Batch 19: the HPPNet port's rung 3 (gpushort chain, 4 segments; all mics; single seed)

| row | zero | below-30 | DREGON ramp | FLY124 ramp | DREGON cruise | FLY124 cruise | all | ch0 / ch1-7 | comb / stoch | probe full / local |
|---|---|---|---|---|---|---|---|---|---|---|
| `real_r3_hppnet` (+ FLY125) | 6.96 | 29.22 | 9.09 | 6.74 | 2.22 | 0.93 | 4.20 | 2.50 / 2.18 | 47.0 / 38.1 | 0.43 / 0.73 |
| `hppnet_r2hb_l4` (rung 4) | 5.95 | 26.65 | 9.83 | 4.77 | 2.95 | 0.92 | 4.18 | 3.04 / 2.94 | 39.1 / 34.1 | 0.97 / 0.97 |

The HPPNet ladder (all-mic MAE / DREGON cruise / FLY124 cruise / probe
local): rung 1 26.86 / 2.91 / 71.2 / 0.71, rung 2 26.48 / 2.41 / 71.2 /
0.55, rung 3 4.20 / 2.22 / 0.93 / 0.73, rung 4 4.18 / 2.95 / 0.92 / 0.97.
Rung 3 is the port's best real cell and the best unseen-drone cell of the
whole ladder (FLY124 cruise 0.93 against 1.30-2.28 for the regressors at
rung 3); rung 4 buys only the probe (0.73 -> 0.97) at a DREGON cruise
cost of 2.22 -> 2.95. Both papers' claim-2 paragraph now says so.

### Phase-coherence probe, run 2 (`phase-coherence-2-4617a1`, 78 units): what the harmonics of one rotor do

Estimator: `scripts/phase_coherence_probe.py` on `tracking.phase_noise`
(demodulate orders k = 1..30 along the refined reference, envelope
autocorrelation -> coherence time and Lorentzian half width, line-shape
verdict, per-order rate opinions -> cross-harmonic correlation under
low-pass smoothing, residual tails). Conditions: one motor on the bench
(20 recordings, 4 motors x 5 setpoints), four motors on the bench (one
merged comb, the rotors within 1.2 rev/s), free flight of the TRAINING
split (DREGON room 2, 7 windows x 4 rotors; FLY125, 7 x 4).

| condition | admitted orders | even / odd (clear of other rotors' lines) | width law (Theil-Sen, Hz per order) | Lorentzian verdict | rho at raw / 4 / 1 / 0.25 Hz | rank-1 share | kurtosis k6-15 | Cauchy-vs-Gauss LLR k1-5 / 6-15 / 16-30 |
|---|---|---|---|---|---|---|---|---|
| one rotor, bench | 329 (56 % censored at 2 s lag) | 140 / 6 (all clear) | 0.042 [0.027, 0.055]; k >= 8: 0.042 | 100 % at every band | 0.001 / 0.001 / 0.001 / 0.001 | 0.28-0.31 | 133 | 1.50 / 0.83 / 0.58 |
| four rotors, bench (merged) | 8 | 6 / 2 (0 / 0) | n/a (clusters k x 1.2 rev/s wide) | 67 % (k6-15), 20 % (k16-30) | 0.002 / 0.003 / 0.007 / 0.009 | 0.23-0.26 | 87 | 0.90 / 0.43 / 0.38 |
| free flight, DREGON room 2 | 0 (every order within the 10 Hz band of another rotor's line) | - | n/a | - | 0.000 / 0.001 / 0.002 / -0.003 | 0.29-0.31 | 86 | 0.55 / 0.48 / 0.36 |
| free flight, FLY125 | 47 | 37 / 1 (1 / 0) | n/a: flat 3-4 Hz plateau, 0.04 s coherence, Gaussian verdict = the band and the neighbours, not the line | 0 % | 0.002 / 0.004 / 0.005 / 0.011 | 0.22-0.25 | 102 | 0.53 / 0.48 / 0.35 |

Per-order medians, one rotor: gamma 0.22 Hz (k 8), 0.30 (10), 0.20 (12),
0.37 (14), 0.40 (16), 0.45 (18), 0.46 (20), 0.65 (22), 0.97 (24), 1.01
(26), 0.88 (28), 1.19 (30); coherence times 0.74 s -> 0.13 s.

Readings. (1) A single two-blade rotor at a fixed setpoint radiates the
blade-passage orders (even shaft orders, 140 admitted against 6 odd);
after the shaft rate is tracked, each order is a Lorentzian line whose
residual half width grows with order at 0.042 Hz per order, ten times
below the flight-time law of the tracking work (about 0.6 Hz per order,
measured against telemetry, which includes the shaft jitter the refined
reference here absorbs). (2) The envelope-autocorrelation width is
measurable only where one comb is alone: with four combs 2-7 rev/s
apart, another rotor's line sits inside the demodulation band of nearly
every order, and the estimator reads the band or the neighbour (flat
3-9 Hz plateaus with Gaussian verdicts). Flight-time per-order widths
therefore stay with the phase-increment variance law of the tracking
work (WP18), not with this estimator. (3) After the shared rate error
is fitted out by the refinement, the per-order rate opinions are
uncorrelated across harmonics at every smoothing scale in every
condition (rho <= 0.011, rank-one share 0.22-0.31), so the phase noise
has a per-harmonic component independent across orders; this probe
cannot size the shared term (the reference absorbs it; WP18 put it at
12-27 % of the off-diagonal energy against a telemetry reference). (4)
The per-harmonic residuals are heavy-tailed in every condition and band
(excess kurtosis 86-133; a Cauchy fit beats a Gaussian by 0.35-1.5 nats
per sample), as a Lorentzian line implies. Together (1), (3) and (4)
are the stochastic comb's assumptions: Lorentzian lines, independent
per harmonic, widths growing with order.

### Batch 20: the HarmoF0 comb curriculum (gpushort chain, 2 segments; all mics; single seed)

| row | zero | below-30 | DREGON ramp | FLY124 ramp | DREGON cruise | FLY124 cruise | all | comb / stoch |
|---|---|---|---|---|---|---|---|---|
| `hf0_r4_l4_v2` (comb -> R4, monitored on rps_mae) | 13.31 | 26.86 | 19.80 | 22.94 | 5.94 | 5.45 | 9.63 | 44.4 / 35.4 |
| `hf0_r2hb_l4` (R4, no warm start) | 13.96 | 24.28 | 16.62 | 12.90 | 6.22 | 2.29 | 7.90 | 39.3 / 33.1 |
| `hppnet_r4_l4` (comb -> R4, for comparison) | 8.93 | 27.38 | 17.14 | 11.96 | 3.77 | 1.40 | 6.04 | 47.5 / 38.1 |
| `hppnet_r2hb_l4` (R4) | 5.95 | 26.65 | 9.83 | 4.77 | 2.95 | 0.92 | 4.18 | 39.1 / 34.1 |

Reading: the static-comb curriculum, the best recipe for every
regressor, hurts both salience ports (HarmoF0 7.90 -> 9.63, HPPNet 4.18
-> 6.04 all-frame; FLY124 cruise 2.29 -> 5.45 and 0.92 -> 1.40). The
ports read frequency by construction, so the curriculum has nothing to
teach them and its stage-1 optimum is a worse start than random. The
probe of this cell follows (batch 20b).

### Batch 21: HarmoF0 rung 2 (gpushort chain, 1 segment, early stop at epoch 42); the curriculum probe

| row | zero | below-30 | DREGON ramp | FLY124 ramp | DREGON cruise | FLY124 cruise | all | ch0 / ch1-7 | comb / stoch | probe full / local |
|---|---|---|---|---|---|---|---|---|---|---|
| `real_r2_hf0` (DREGON, 8 mics) | 18.71 | 34.35 | 13.56 | 54.61 | 3.45 | 63.91 | 27.46 | 5.13 / 3.21 | 43.3 / 35.4 | 0.39 / 0.28 |
| `hf0_r4_l4_v2` (comb -> R4; probe only new) | 13.31 | 26.86 | 19.80 | 22.94 | 5.94 | 5.45 | 9.63 | 9.21 / 5.47 | 44.4 / 35.4 | 1.06 / 1.04 |

The HarmoF0 ladder (all-mic MAE / DREGON cruise / FLY124 cruise / probe
local): rung 1 37.05 / 19.43 / 71.6 / 0.28, rung 2 27.46 / 3.45 / 63.9 /
0.28, rung 3 7.11 / 5.38 / 2.38 / 0.24, rung 4 7.90 / 6.22 / 2.29 / 0.95.
Same shape as every other trunk: the second drone fixes the unseen rig,
the augmentations fix the probe (0.24 -> 0.95) and cost DREGON cruise
(5.38 -> 6.22). Its comb curriculum reads frequency (1.06 / 1.04) at the
worst real split of the column (9.63). With this batch every rung of
every trunk is scored except `salv2_tr_stoch_nomix`,
`salv2_gru_stoch_{nomix,mix}` (running on sae) and `hb_sal_multif0_l4`
(gpushort chain).

### Batch 22: GRU on the stochastic family, no speech (sae direct, 123 epochs, early stop; checkpoints uploaded by hand as for the comb twin)

`salv2_gru_stoch_nomix`: stoch 2.50 (the best stochastic regressor;
W&B best 2.501), +speech 3.71 (ratio 1.48), comb 8.1, real 31.6; probe
0.62 / 0.88; fan 6.3 / 7.4 / 7.8 / 8.6 / 6.3 across the true-spread
buckets, slope 0.14; error classes offset 37 %, wander 40 %, no alias,
no octave. The fixed fan now stands on all three regressor trunks
(slopes 0.13 / 0.10 / 0.14). Only `salv2_gru_stoch_mix` and
`salv2_tr_stoch_nomix` remain of the stochastic family.

### Batch 23: the transformer on the stochastic family, no speech (daemon sae job to the 10 h wall at epoch 180, then a gpushort resume chain; early stop at epoch 186)

`salv2_tr_stoch_nomix`: stoch 2.48 (the best stochastic regressor;
W&B best 2.488), +speech 2.81 (ratio 1.13, the most speech-robust
no-speech model of the family), comb 12.8, real 31.0; probe 0.81 /
1.34; fan 4.3 / 6.7 / 8.1 / 8.9 / 5.7, slope 0.11; classes offset 25 %,
wander 42 %, 5/4 alias 7 %, no octave. Claim 5's range for models that
never saw speech is now 1.1-4.5x on the stochastic family (transformer
1.13, GRU 1.48, conv 1.64, HarmoF0 1.77, HPPNet 4.48); both papers'
abstracts and claim-5 paragraphs say 1.1-4.5x. Note on the resume: the
chain's "nothing to resume" detection failed and it kept submitting
2-minute segments until killed by hand; the run was complete from
segment 1. Remaining: `salv2_gru_stoch_mix` (sae, 12 h wall) and
`hb_sal_multif0_l4` (gpushort chain).

### Batch 24: GRU on the stochastic family with speech (sae direct, early stop; the stochastic family is complete)

`salv2_gru_stoch_mix`: stoch 3.58, +speech 3.52 (ratio 0.98), comb 16.7,
real 31.8; probe 0.73 / 0.74; fan 6.1 / 7.7 / 8.1 / 9.1 / 6.5, slope
0.09; classes offset 32 %, wander 24 %, 5/4 alias 7 %, no octave. The
speech A/B on the stochastic family, final: trained without speech a
talker costs 1.13 (tm) / 1.48 (gru) / 1.64 (scv2) / 1.77 (HarmoF0) /
4.48 (HPPNet); trained with speech 0.97-0.99 for the three regressors
and 1.12-1.18 for the ports, at a clean-input price of 1.4-1.6x for the
regressors (2.49 -> 3.90, 2.50 -> 3.58, 3.00 -> 4.21) and 1.2x for the
ports. Six stochastic regressor cells, six fixed fans (slopes 0.09-0.14).
Only `hb_sal_multif0_l4` remains (sae, resumed at epoch 49).

### Batch 25: LateDeep L2, the last cell (gpushort chain to epoch 49, then the sae copy to epoch 72, early stop)

| row | zero | below-30 | DREGON ramp | FLY124 ramp | DREGON cruise | FLY124 cruise | all | comb / stoch | probe full / local |
|---|---|---|---|---|---|---|---|---|---|
| `hb_sal_multif0_l4` (LateDeep, L2: per-rotor layers + CRF readout, original HCQT input) | 4.98 | 15.76 | 9.48 | 7.99 | 2.32 | 1.69 | 3.83 | 25.2 / 33.9 | 1.17 / 0.28 |
| `hb_sal_multif0` (L0) | 52.57 | 35.63 | 14.22 | 18.66 | 2.96 | 4.46 | 12.65 | 31.3 / 39.3 | 0.11 / 0.43 |
| `hb_sal_multif0_nsr` (L1) | 48.50 | 25.82 | 23.80 | 11.95 | 3.53 | 3.66 | 11.82 | 24.1 / 30.0 | 0.38 / -0.57 |
| `hb_sal_bp_l4` (Basic Pitch, L2) | 0.50 | 18.02 | 48.57 | 32.42 | 43.88 | 9.47 | 27.56 | 42.6 / 37.3 | 2.31 / 3.08 |

Reading: the per-rotor layers with the joint readout rescue LateDeep
completely (12.65 -> 3.83, the best salience row of the frozen split,
better than the HPPNet port's rung-4 row 4.18 and within the regressor
band 2.7-4.2) and do nothing for Basic Pitch (27.30 -> 27.56, which
learns silence only). Block S's message changes accordingly: which
modification a music model needs depends on what it already has (the
harmonic architectures need the comb-gather input, the harmonic-stacking
CNN needs the readout, Basic Pitch responds to neither). LateDeep L2 does
not read frequency locally (0.28), so it is a precise prior-driven model
like the rung-3 regressors. Both papers' block-S paragraphs are updated.
With this batch every cell of the matrix is scored: `missing.txt` lists
no cell without a dump.

### Claim 4: the stochastic limit (stochastic part, cruise time-frames; regressor rows for tm / gru pending)

The fan statistic of `scripts/rps_error_profile.py` (`fan.csv`): on the
cruise time-frames of the stochastic validation set (true mean speed of
the four rotors at least 45 rev/s), the predicted rotor spread (max minus
min over the four predicted tracks) against the true spread, in buckets of
true spread. `fan_slope` is the least-squares slope of predicted against
true spread over all cruise time-frames: 1 = the model tracks four lines,
0 = a fixed fan.

| true spread bucket (rev/s) | 0-2 | 2-5 | 5-10 | 10-20 | 20+ | slope |
|---|---|---|---|---|---|---|
| true spread (mean) | 0.35 | 4.43 | 7.36 | 11.81 | 21.96 | 1 |
| frames | 200 | 4928 | 22384 | 3872 | 528 | |
| `salv2_scv2_stoch_nomix` | 5.16 | 7.12 | 8.04 | 9.15 | 6.22 | 0.13 |
| `salv2_scv2_stoch_mix` | 6.00 | 7.38 | 7.77 | 8.86 | 5.75 | 0.10 |
| `salv2_tr_stoch_mix` (batch 17) | 6.31 | 8.10 | 8.51 | 9.78 | 6.25 | 0.10 |
| `salv2_gru_stoch_nomix` (batch 22) | 6.31 | 7.36 | 7.75 | 8.62 | 6.28 | 0.14 |
| `salv2_tr_stoch_nomix` (batch 23) | 4.31 | 6.70 | 8.05 | 8.93 | 5.71 | 0.11 |
| `salv2_gru_stoch_mix` (batch 24) | 6.14 | 7.73 | 8.07 | 9.08 | 6.47 | 0.09 |
| `salv2_hppnet_stoch_nomix` | 0.06 | 10.98 | 6.56 | 7.68 | 14.86 | 0.16 |
| `salv2_hppnet_stoch_mix` | 0.08 | 9.27 | 6.53 | 8.05 | 12.74 | 0.21 |
| `salv2_hf0_stoch_nomix` | 0.38 | 15.71 | 9.18 | 10.14 | 7.26 | -0.36 |
| `salv2_hf0_stoch_mix` | 0.22 | 14.16 | 9.51 | 11.07 | 6.81 | -0.22 |
| `r4hb_scv2` (comb -> real) | 7.17 | 3.03 | 5.55 | 7.42 | 8.25 | 0.46 |
| `r4hb_gru` (comb -> real) | 10.21 | 2.81 | 5.27 | 5.39 | 8.83 | 0.32 |
| `salv2_scv2_comb_nomix`, `salv2_tr_comb_nomix`, `salv2_gru_comb_mix`, `hppnet_r4_l4` | 0.1-6.6 | 0.1-0.6 | 0.0-0.4 | 0.1-1.2 | 0.0-0.7 | 0.0 |

Error classes on the same part (share of the model's total error carried by
each class of failed rotor track; `classes.csv`):

| row | MAE | median | p90 | offset | alias 5/4 | alias 2 (octave) | wander | missed |
|---|---|---|---|---|---|---|---|---|
| `salv2_scv2_stoch_nomix` | 2.99 | 2.05 | 8.39 | 0.26 | 0.22 | 0.00 | 0.36 | 0.00 |
| `salv2_scv2_stoch_mix` | 4.20 | 2.99 | 9.72 | 0.43 | 0.07 | 0.02 | 0.32 | 0.00 |
| `salv2_tr_stoch_mix` (batch 17) | 3.90 | 2.05 | 12.72 | 0.38 | 0.08 | 0.00 | 0.26 | 0.00 |
| `salv2_gru_stoch_nomix` (batch 22) | 2.50 | 1.84 | 5.60 | 0.37 | 0.00 | 0.00 | 0.40 | 0.00 |
| `salv2_tr_stoch_nomix` (batch 23) | 2.48 | 1.63 | 5.64 | 0.25 | 0.07 | 0.00 | 0.42 | 0.00 |
| `salv2_gru_stoch_mix` (batch 24) | 3.58 | 2.16 | 10.61 | 0.32 | 0.07 | 0.00 | 0.24 | 0.00 |
| `salv2_hppnet_stoch_nomix` | 2.65 | 1.68 | 5.39 | 0.07 | 0.00 | 0.24 | 0.57 | 0.00 |
| `salv2_hppnet_stoch_mix` | 3.16 | 1.62 | 6.81 | 0.12 | 0.00 | 0.23 | 0.48 | 0.00 |
| `salv2_hf0_stoch_nomix` | 9.13 | 4.66 | 23.16 | 0.02 | 0.02 | 0.01 | 0.69 | 0.00 |
| `salv2_hf0_stoch_mix` | 10.65 | 6.42 | 26.71 | 0.08 | 0.00 | 0.08 | 0.64 | 0.00 |
| `r4hb_scv2` | 28.09 | 27.19 | 54.09 | 0.49 | 0.00 | 0.03 | 0.02 | 0.37 |
| comb-only rows | 36.8-37.8 | 44.7 | 69.5-69.9 | 0.02-0.08 | 0.00 | 0.00 | 0.00-0.01 | 0.91-0.97 |

Readings. The convolutional regressor trained on the stochastic family
predicts a 5-9 rev/s fan whatever the true spread (slope 0.13, 0.10 with
speech): at a true spread of 0.35 rev/s it still asserts 5.2, and at 22 it
asserts 6.2. This is the fixed fan: the model reads the mean speed and
places four lines around it at the spread the training distribution
favors. Its error is offsets (26 %) and the 5/4 alias (22 %), the
signature of a fan line landing on a neighbor's harmonic, and no octaves.
The salience ports fail differently: HPPNet collapses the four tracks onto
one line when the rotors are within 2 rev/s (0.06), overshoots at 2-5
(11.0), and carries 24 % of its error in whole-clip octave locks (alias 2)
and 57 % in wander; HarmoF0 spreads its lines too wide (9-16 rev/s at every
true spread above 2) and 69 % of its error is wander. The curriculum rows
(comb -> real) return near-zero on the stochastic part (missed 37-41 %),
and the static-comb-only rows return zero outright (missed 91-97 %, fan
0.0-0.6): no transfer from the static comb to the stochastic family. The
transformer trained on the stochastic family with speech (batch 17,
`salv2_tr_stoch_mix`: stoch 3.90, +speech 3.86, comb 5.9, real 30.2,
probe 0.89 / 1.37) shows the same fixed fan as the convolutional
regressor, 6.3-9.8 rev/s at every true spread with slope 0.10, and the
same error mix (offsets 38 %, wander 26 %, the 5/4 alias 8 %, no
octaves): the fan is a property of the regression readout, not of one
trunk. The GRU cells and the transformer without speech join the table
when they land.

## Slot-comb v2 test (2026-09-06, user request: implement and test `docs/slot-comb-v2-design.md`, synthetic first)

Implementation (merged into main, commits 1ae441f and 47e5e77, plumbing to
follow): all seven parameter groups of the design as opt-in flags of
`SlotCombNet`, each containing the C1 corner at initialization (init
equivalence tested to <= 2.4e-7 on the score and 1 float32 ULP on the
loss): `off_state` (3.1), `r_lo=10, n_grid=900, mask_below_grid` (3.2),
`learned_transition` with `trans_slew=30` (3.3), `emission="v2"` with the
parts `gap` (3.4), `cross_order` (3.5), `read_width_learned` and
`claim_width_learned` (3.7) in `src/models/comb_slots_emission_v2.py`,
and `rate_prior` (3.6, `src/models/comb_slots_prior.py`). Two findings
from the implementation: the gap and read-width groups have vanishing
gradients at the exact corner (interior "off" limits), so the trained
arms start from `warm_start(gap_mu=-2, read_sigma=...)`, one step off the
corner; and the wider chain (900 points, 81-wide band, OFF scalar) costs
1.4x per training step end to end (the emission dominates).

Arms, all at ONE microphone (the fair comparison of the C1 campaign),
2 s crops, batch 2-4, lr 1e-3, selection on 48 windows of the training
policy, the trainer's non-finite guard, gpushort chains:

| arm | data | init | what it tests |
|---|---|---|---|
| `v2_comb` | S1 static comb (mono, `salv2_comb`) | corner + warm start | the mechanism on its home family with every group on |
| `v2_stoch` | S2 stochastic comb (mono, `salv2_stoch`) | `v2_comb` best | the stochastic limit (claim 4) for the CRF family |
| `v2_real` | R4 real mono pool (`slot_real_dload`, silence arm) | `v2_stoch` best | the curriculum synthetic -> real, the paper's C1/C2 analogue |
| `v2_real_scratch` | R4 real mono pool | corner + warm start | the A8 recipe with the v2 groups, no synthetic stage |

Scoring: `scripts/slot_dump.py` on the six parts into `results/rps_dump`
(the paper's format), `scripts/rps_regime_table.py`, the cue probes,
`scripts/rps_error_profile.py` (fan and classes on the stochastic part).

Decision rule (user, 2026-09-06): the paper is restructured around v2
only if it beats the tested models ACROSS splits, i.e. the best regressor
and port rows of the ladder on the real split (all-frame 2.74, DREGON
cruise 2.1-2.9, FLY124 cruise 0.9-1.2, zero 1.6-2.9, ramps 3-5) and the
best synthetic rows on their own parts (static comb 0.46-0.9, stochastic
2.48-2.65), at one microphone; otherwise v2 is not mentioned.

## Conclusion

Written 2026-09-06; every cell of the matrix is trained and scored.

1. **Narrow data gives precise models that do not transfer.** Every
   trunk trained on DREGON alone reaches its best DREGON cruise cell of
   the ladder at rung 1 or 2 (regressors 2.3-2.7, HPPNet port 2.4) and
   fails on the unseen drone (FLY124 cruise 8.5-72); one flight of that
   drone (rung 3) fixes it (1.2-2.4). The microphone boundary is crossed
   by the recurrent and attention heads and by the ports (ch 0 = ch 1-7
   at DREGON cruise) and not by the plain SimpleConv (3.56 -> 6.06). No
   real rung crosses into either synthetic family and no synthetic cell
   crosses back.
2. **Generality is bought with precision.** Rung 3 is every trunk's
   best real split. The label-transforming augmentations (rung 4) buy
   the probe and static-comb transfer at a DREGON cruise cost of
   0.3-0.7 rev/s with a warm-up and 0.5-2.3 rev/s from step 0; the
   comb curriculum C1 is the cheapest frequency-reading recipe for the
   regressors (probe 1.0-1.4 at all-frame 2.74-3.61) and hurts both
   salience ports. The one-pool mix reads frequency and loses 2-4 rev/s
   on the real split for every trunk.
3. **No model reads harmonic positions unless forced.** Real-only
   regressors sit at |slope| <= 0.3; the augmentations raise it
   inconsistently (0.0-0.4 with a warm-up, 0.5-1.0 from step 0), the
   comb curriculum reliably (1.0-1.4); HPPNet reads frequency at every
   rung and as published (its log-axis dilated convolutions are a comb
   by construction), the published HarmoF0 does not and its comb-gather
   port does, LateDeep and Basic Pitch never do.
4. **The stochastic limit.** All three regressor trunks predict a
   fixed 5-9 rev/s fan of the four rotors whatever the true spread
   (slopes 0.10-0.14) with offsets and the 5/4 alias as their error and
   no octaves; the salience ports fail by octave locks (HPPNet, 24 %) and
   wander (HarmoF0, 69 %); nothing learned on the static comb transfers
   (missed 91-97 %). The phase-coherence probe grounds the stochastic
   comb: a single rotor's blade-passage orders are Lorentzian lines whose
   residual width grows with order (0.042 Hz per order after shaft
   tracking), the per-harmonic residuals are uncorrelated across orders
   at every smoothing scale and Cauchy-tailed in every condition; widths
   in flight are not separable by the envelope estimator and rest on the
   tracking work's variance law (0.6 Hz per order).
5. **Speech.** An evaluation-time talker costs a model that never saw
   speech 1.6-2.2x on the static comb and 1.1-4.5x on the stochastic
   comb (transformer 1.13 ... HPPNet 4.48), and 1.0-1.2x for the same
   trunk trained with mixed speech; on real audio the loudspeaker clips
   cost 1.3-2.0x without and 1.0-1.3x with speech in training, at
   0-40 % on clean input. "Twice as hard" holds for models that never saw
   speech.

Block S: the harmonic device decides the ranking before any adaptation
(published HPPNet 7.77 and HarmoF0 10.79 against LateDeep 12.65 and
Basic Pitch 27.30); the comb gather with per-rotor layers halves
HPPNet's error (4.18, silence 17.5 -> 6.0) and HarmoF0's DREGON cruise
error (11.5 -> 6.2); output resolution alone (L1) changes little; the
per-rotor layers with the joint readout (L2) rescue LateDeep completely
(3.83, the best salience row) and do not rescue Basic Pitch (27.56).

Leaderboard: the blind tracker remains the most precise DREGON cruise
method (0.92) at forty times a regressor's cost, loses on FLY124 (9.2)
and has no silence decision without its gates; the best learned cells
are C1 Conv+BiGRU (all 2.74) and the HPPNet port at rung 3 on the unseen
drone (FLY124 cruise 0.93). Training-free estimators stay above 11 rev/s
at cruise.

Deliverables: `writing/papers/2026-08_wrapup` (ICASSP `src/index.tex`,
4 + 1 pages; OJSP `src/journal.tex`, 8 + 1 pages), every table generated
from the dumps (`scripts/rps_claim_tables.py`, `make_tables.py`,
`make_phase_table.py`), the figures from `make_probe_fig.py` and
`make_phase_fig.py`. Lessons for the next campaign are in the session
memory: credentials in direct sbatch jobs, chain runners and dirty
trees, vast egress, subagents and local compute.
