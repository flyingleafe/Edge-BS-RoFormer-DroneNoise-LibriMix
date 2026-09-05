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

## Protocol

- Every checkpoint is dumped with `scripts/rps_dump.py` on the parts `comb`,
  `stoch`, `real`, `comb_speech`, `stoch_speech` and `real_nospeech` (the
  noise-only twin of the frozen real split), and the dump must reproduce the
  W&B monitored value to three decimals.
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

Pending.

## Conclusion

Pending.
