# E8 — Static-Comb Noise Model (force harmonic tracking)

**Status:** completed (2 of 3 arms; verdict revised after the valid-set fix) —
**Date:** 2026-07-11, conclusion backfilled 2026-08-20

## Motivation

E7 showed that training an RPS predictor on *neural-generated* drone noise —
even the good E6 per-drone generator, vicinally sampled — does **not** transfer
to real data: real val **PIT MSE ≈ 222, R² ≈ −10.5** (worse than predicting the
mean), identical across `uni_gru128` and `simple_conv_v2`, while train PIT MSE
converged to ~10 on the generated stream. Both metrics are PIT-aligned, and the
val set + arch are identical to the real-data baseline that scores **7.33**, so
the ~22× train→real blow-up is a pure domain gap.

**Hypothesis (why it fails).** On generated data the predictor *reverse-engineers
the amplitude dynamics* — the neural generator's harmonic/noise amplitudes
co-vary with RPS and time, giving a shortcut (amplitude → RPS) that the model
exploits instead of tracking the harmonic comb's *frequency*. That shortcut does
not exist in real recordings (whose amplitude↔RPS relationship differs), so the
model predicts a confidently-wrong RPS → strongly negative R².

## The fix: the simplest model that forces frequency-tracking

`data_processing/rotor_spectral_model.py` (`StaticCombNoisePool`, `kind:
static_comb`). Per clip:

- **Static harmonic comb** at `k·rps(t)` (matching the neural gen's `f0 = rps`
  convention, `harmonic_gen_new.py`: `f0s = ms`) with a per-harmonic amplitude
  profile `a_k` that is **fixed in time and independent of RPS** → the *only*
  RPS cue is comb spacing.
- **Static broadband floor** (pink-ish), with the comb/floor mix constrained so
  **≥30% of each rotor's in-band harmonics clear the floor** (real single-rotor
  combs measure 77–100% above; the high harmonics may still wash out).
- **Wide profile variety** per clip — rolloff `p`, blade-pass emphasis,
  per-harmonic irregularity, floor level/tilt — sampled across ranges
  **calibrated to real** DREGON individual-motor + Michael's combs (measured:
  rolloff `p ≈ 0.63–1.48`, floor only −1.6…−11.6 dB below the comb).
- Multichannel via per-(mic,rotor) gain spread; RPS from `rps_synthesis`
  (same driver as E7, so the comparison isolates the amplitude-shortcut fix).

Fully analytic — no neural net, no GPU producer, no geometry; renders in ~84 ms
per 1 s 8-mic chunk directly in the DataLoader workers. Validated: const-RPS comb
has per-bin temporal amplitude CoV ≈ 0.012 (genuinely static); 0/80 sampled
rotors fall below the 30%-above-floor floor.

## Arms

Three predictor heads, static-comb-only train, fixed real
`DREGON-LM-V4-michaels/valid`, PIT MSE, patience 8:

| Arch | Experiment | E7 neural-gen (same arch) | real-data ref |
|---|---|---|---|
| `simple_conv_v2_uni_gru128` | `e8_staticcomb_s1_unigru128` | 222.3 (R² −10.5) | 7.33 |
| `simple_conv_v2` | `e8_staticcomb_s1_scv2` | 222.8 (R² −10.6) | 9.71 |
| `simple_conv_v2_transformer` | `e8_staticcomb_s1_transformer` | *(pending)* | 8.85 |

The question: does removing the amplitude shortcut + guaranteeing harmonic
visibility pull the real val PIT MSE / R² toward the real-data baseline? A
Stage-2 real-only fine-tune (mirroring E7) follows if Stage 1 shows transfer.

## Running

```bash
python train.py experiment=e8_staticcomb_s1_unigru128            # local GPU
# cloud:
omnirun submit --backend colab --gpu-type L4 --gpus 1 --time 3h --yes -- \
  python train.py experiment=e8_staticcomb_s1_unigru128 \
    data.train.params.path=conf/online_mix/rps_static_comb_only_dload.yaml \
    "data.valid.params.data_dir='dload:DREGON-LM-V4-michaels-valid'"
```

## Conclusion

*(Backfilled 2026-08-20 from the W&B run summaries and the
[2026-07-12 report](../../writing/reports/2026-07-12_full-flight-sim2real-rps/).)*

Two arms ran on 2026-07-11; the `scv2` arm never ran (no run, no checkpoint).
Best validation PIT-MSE, both on the **contaminated** valid
(`min_motor_rps=30`, FLY124 ground warm-up included):

| Arch | E8 static-comb | E7 neural-gen (same arch) | R² (E8) |
|---|---|---|---|
| `transformer` | 188.7 | 225.3 | −7.3 |
| `uni_gru128` | 222.6 | 222.3 | −10.5 |

**Partial support for the amplitude-shortcut hypothesis:** removing the
amplitude cue helped the transformer (−36 PIT-MSE) and left the smaller head
unchanged. But both arms still looked like failures, which motivated E9's
"remove every single-source shortcut" design.

**The failure magnitude was an artifact of the yardstick.** The clean
free-flight-only valid (`min_motor_rps=50`) landed with E9; on it the E9
gen-only recipe (which uses this static comb as half its noise mix) scores
17.8–25.4 with positive R². The E8 checkpoints were **never rescored** on the
clean split. The durable E8 contributions are: the analytic
`StaticCombNoisePool` itself (later half of E9/E10/E11 training noise), and the
relative transformer-vs-small-head reading above, which survives a
yardstick change because both arms shared it.
