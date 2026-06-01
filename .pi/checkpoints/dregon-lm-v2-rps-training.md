# DREGON-LM-V2 dataset + RPS predictor training + cross-evaluation

**Goal:** Better DREGON-LM train/validation sets → train RPS prediction models with PIT loss → cross-evaluate old vs new checkpoints.

**Status:** complete (results consolidated, ready for report)
**Last touched:** 2026-06-01
**Resume on:** local (vast-server GPU)

---

## What was built

### DREGON-LM-V2 dataset (`datasets/DREGON-LM-V2/`, 2.2 GB)
- 6000 train + 600 valid samples, 3 seconds at 16 kHz
- All 8 mic channels (each as independent noise source)
- Command RPS (cleaned via `clean_command_spikes`)
- Recording-level train/valid split (zero overlap)
- 20% synthetic motor combos (~1200 samples)
- Script: `create_dregon_librimix.py`
- Old `datasets/DREGON-LM/` still intact

### Training code updates (`train_rps_predictor.py`)
- **PIT (permutation-invariant) MSE loss**: 24 permutations over 4 rotors, pairwise MSE matrix, min-permutation
- **WandB**: project `rps-prediction`, entity `flyingleafe`. NB: tmux doesn't inherit `WANDB_API_KEY` — must pass explicitly.
- **PIT-aware validation**: `evaluate()` uses PIT MSE as primary metric, reports both PIT and Std MSE
- Data root default: `datasets/DREGON-LM-V2`

### Model checkpoints

| Name | Arch | Params | Checkpoint |
|------|------|--------|-----------|
| OLD simple_conv | SimpleConv | 538K | `results/rps_exp_simple_conv/best_simple_conv.pt` |
| OLD bigru_v2 | SimpleConvBiGRUV2 | 1.44M | `results/rps_exp_bigru_v2/best_simple_conv_bigru_v2.pt` |
| V3 simple_conv | SimpleConv | 538K | `results/rps_predictor_v3/simple_conv/best_simple_conv.pt` |
| V3 bigru_v2 | SimpleConvBiGRUV2 | 1.44M | `results/rps_predictor_v3/simple_conv_bigru_v2/best_simple_conv_bigru_v2.pt` |

---

## Results: Cross-evaluation (PIT MSE)

All metrics in `results/rps_cross_eval/validation_metrics.json`.

### Main table

| Model | OLD valid | V2 valid | Degradation |
|-------|-----------|----------|-------------|
| OLD simple_conv | **5.24** | 331.87 | 63× |
| OLD bigru_v2 | **2.67** | 327.26 | 123× |
| V3 simple_conv | 66.83 | 148.09 | 2.2× |
| V3 bigru_v2 | 15.26 | 71.13 | 4.7× |

### Per-channel (V2 valid only)

| Model | ch0 | ch1-7 | ch1-7/ch0 |
|-------|-----|-------|-----------|
| OLD simple_conv | 293.60 | 338.01 | 1.15× |
| OLD bigru_v2 | 247.16 | 340.12 | 1.38× |
| V3 simple_conv | 126.92 | 151.48 | 1.19× |
| V3 bigru_v2 | 66.17 | 71.92 | 1.09× |

### In-flight recordings (PIT MSE, median over 3s windows)

| Model | speech-high | whitenoise-high |
|-------|-------------|-----------------|
| OLD simple_conv | 499 | 542 |
| OLD bigru_v2 | 504 | 542 |
| V3 simple_conv | 639 | 509 |
| V3 bigru_v2 | 481 | 532 |

Note: in-flight recordings have source signals (speech/whitenoise) mixed in — all models struggle (MSE 400–500), no clear winner.

---

## Key conclusions

1. **Old dataset was trivially easy.** OLD models achieve PIT MSE 2.7–5.2 on OLD valid (1s samples, measured RPS, train/val overlap). They collapse completely on V2 (63–123× worse).

2. **V2 is genuinely harder** — not a recipe problem. Factors:
   - 3s samples require curve fitting (94 STFT frames) instead of scalar prediction (32 frames)
   - Command RPS is noisier than measured RPS
   - Recording-level split eliminates memorization

3. **V3 models generalize much better** (2.2–4.7× degradation OLD→V2 vs 63–123× for old), but are capacity-limited on the harder task — they plateau at PIT MSE 71–148.

4. **Channel matters but isn't dominant.** Old models show 15–38% ch0 advantage (home field). V3 models show 9–19%. Even on ch0, V3 beats old by 2.3–3.7×.

5. **PIT training causes rotor-order ambiguity in SimpleConv** (28% PIT/Std gap) but not in BiGRUv2 (4% gap). BiGRUv2 learns a stable ordering.

6. **All models fail on in-flight source recordings** — the interfering source signal (speech/whitenoise) destroys RPS prediction.

---

## How to reproduce

### Dataset
```bash
uv run python create_dregon_librimix.py \
    --speech_dir data/librispeech/LibriSpeech/train-clean-100 \
    --dregon_dir data/DREGON \
    --output_dir datasets/DREGON-LM-V2 \
    --num_train 6000 --num_valid 600 --motor_combo_fraction 0.2
```

### Training
```bash
WANDB_API_KEY="<key>" uv run python train_rps_predictor.py \
    --model simple_conv_bigru_v2 --device cuda:0 \
    --data_root datasets/DREGON-LM-V2 \
    --save_path results/rps_predictor_v3/simple_conv_bigru_v2 \
    --epochs 500 --patience 30 --batch_size 96 --lr 1e-3
```

### Cross-evaluation
```bash
uv run python eval_cross.py
# Output: results/rps_cross_eval/
#   validation_metrics.json  — full dataset metrics
#   samples/                 — 5 old + 5 V2 samples with preds
#   inflight/                — speech-high & whitenoise-high windows
```

### Training logs
- V2 runs: `results/rps_predictor_v2/{simple_conv,simple_conv_bigru_v2}.log`
- V3 runs: `results/rps_predictor_v3/{simple_conv,simple_conv_bigru_v2}.log`
- WandB: https://wandb.ai/flyingleafe/rps-prediction

---

## State
- Working tree: dirty (`eval_cross.py` untracked, `train_rps_predictor.py` modified for PIT eval)
- `datasets/DREGON-LM-V2/`: 2.2 GB, NOT in git
- `results/rps_cross_eval/`: cross-evaluation artifacts, NOT in git
- `results/rps_predictor_v2/`: V2 training results
- `results/rps_predictor_v3/`: V3 training results
- No running processes

## Open questions
- Can V3 models be pushed further? Current plateau at PIT MSE 71–148 may be a genuine capacity limit.
- In-flight source recordings: would a denoising-first-then-RPS pipeline work better?
- Should we train directly on in_flight_source data (with source as additional noise)?
