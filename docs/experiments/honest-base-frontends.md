# Honest base regime + front-end grid (HB campaign)

Status: DESIGNED 2026-08-24, runs pending.

## Motivation

The zero-RPS regime is the largest weakness of the real-only predictors.
The diagnosis (2026-08-24 probes, `docs/experiments/generator-refined-labels.md`
§ regime probe) has three parts:

1. **Coverage.** The real training pool holds 421.4 s of noise. The regime
   split of training time is 6.25% zero / 8.11% ramps / 85.63% flight.
   Only 26.4 s of unique zero material exists, almost all from one room.
   The validation split is 12.7% zero frames — double the training share.
2. **Level confound.** The mixer scales speech relative to the power of the
   noise chunk. A zero chunk has ~40 dB less power than a flight chunk, so
   its mixture is near-silent overall. "Quiet in, zero out" is a winning
   shortcut on the training set, and it breaks on validation zeros that
   carry content (the 41–50 Hz rumble clip, transitions).
3. **Output head.** The plain linear head has no off state. Under MSE an
   uncertain model outputs the conditional mean — the observed 10–45 rev/s
   drift on zero frames (57% of real-only scv2 zero-frame values; no model
   hallucinates flight speeds there).

The zero-labeled spans of the real pool are label-honest: their audio RMS is
1–6% of flight RMS and the 40–350 Hz rotor band holds 3–9% of the energy.
The problem is what they lack — level diversity and content — not what they
contain.

## Design

### Data: the `hb` regime (`conf/online_mix/hb_silence_dload.yaml`)

A clone of the fs_v2 recipe (`e12_fullflight_freqscale_v2_dload.yaml`:
DREGON room2 + FLY125 full envelope, warm-up stage, freq-scale + time-warp +
gain/polarity augmentations) with two additions:

1. **Silence arm** — a `kind: silence` engine source
   (`data_processing/silence_noise.py`, `SilenceNoisePool`) at weight 0.4
   against the merged real weight 2.0 = **16.7% of noise chunks**. Each chunk
   carries an all-zero RPS track and one of three floors:
   - `room_tone` (w 0.3): colored noise at the measured quiet-span level
     (RMS 5e-4 to 5e-3);
   - `colored` (w 0.4): spectral tilt U(0, 2), RMS 5e-3 to 8e-2 — up to
     flight level, the "loud audio without a comb" arm;
   - `lf_rumble` (w 0.3): a dominant 30–150 Hz band, RMS 1e-2 to 8e-2 —
     the rumble-clip failure mode.
   Expected training-time regime mix: ≈21.9% zero / 6.8% ramps / 71.3% flight.
2. **SNR reference floor** — `snr_ref_floor_rms: 0.02`. The speech scaling
   reference becomes `max(noise_power, 0.02^2)`, so quiet and silent chunks
   carry speech at a normal absolute level. Flight chunks (RMS ~0.07) are
   unchanged. This removes the level shortcut.

### Model: voicing gate

`GatedProjection` replaces the final linear layer of the three heads
(BiGRU, causal GRU-128, temporal transformer) when `voicing_gate: true`:
one linear layer to `2R`, output = `sigmoid(gate) * speed`. The loss stays
plain PIT-MSE; the gate turns rotor-off into a classification decision.
The mechanism is identical across the three architectures.

### Front-ends (all on the 2048/512 grid)

| Key | Channels | Content |
|-----|----------|---------|
| `stft_mag` | 1 | log1p magnitude |
| `stft_mag_if` | 2 | log1p magnitude + IF deviation (fractional bins) |
| `stft_ssq` | 1 | synchrosqueezed log magnitude — STFT power reassigned to the rounded IF bin (`scatter_add`), then `log1p(sqrt(.))` |

`stft_ssq` folds the IF evidence of `stft_mag_if` into a sharpened
one-channel magnitude: window leakage re-concentrates onto the true
frequency, so comb lines get thinner where rotor fundamentals sit.

## The grid (10 runs, all on the `hb` regime)

| Experiment | Trunk head | Front-end | Gate |
|---|---|---|---|
| `hb_scv2_mag` | BiGRU (`simple_conv_v2`) | `stft_mag` | yes |
| `hb_scv2_if` | BiGRU | `stft_mag_if` | yes |
| `hb_scv2_ssq` | BiGRU | `stft_ssq` | yes |
| `hb_tr_mag` | Transformer | `stft_mag` | yes |
| `hb_tr_if` | Transformer | `stft_mag_if` | yes |
| `hb_tr_ssq` | Transformer | `stft_ssq` | yes |
| `hb_gru_mag` | causal GRU-128 | `stft_mag` | yes |
| `hb_gru_if` | causal GRU-128 | `stft_mag_if` | yes |
| `hb_gru_ssq` | causal GRU-128 | `stft_ssq` | yes |
| `hb_scv2_mag_nogate` | BiGRU | `stft_mag` | no |

All runs: `pit_mse` loss, `rps` metrics, batch 128 frames,
`samples_per_validation` 40000, patience 20, lr 1e-3, wd 1e-4, validation on
`dload:DREGON-LM-V4-michaels-valid-full`. This grid is the first honest
front-end × architecture comparison: every earlier comparison mixed
front-ends between architectures (scv2 and uni_gru on `stft_mag`, the
transformer on `stft_mag_if`).

Controls outside the grid: `scv2_fs_v2` (old data, no gate) against
`hb_scv2_mag_nogate` isolates the data effect; `hb_scv2_mag_nogate` against
`hb_scv2_mag` isolates the gate effect.

## Readouts

1. Aggregate best `val/mse` per run (W&B history minimum, not the summary).
2. Per-regime PIT MAE/MSE (zero / low / flight) with the standard probe
   (`results/m3cur_regime_probe/regime_probe.py` pattern).
3. Zero-frame error modes: clean-off-call rate (all four |pred| < 5),
   mid-range drift mass (10–45 rev/s), and gate saturation statistics.
4. Front-end ranking per architecture and its consistency across
   architectures.

## Results

Pending.
