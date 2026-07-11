# E8 — Static-Comb Noise Model (force harmonic tracking)

**Status:** built, running — **Date:** 2026-07-11

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

_Pending run._
