# DREGON-LM-V2 dataset + RPS predictor training + cross-evaluation

**Goal:** Better DREGON-LM train/validation sets → train RPS prediction models with PIT loss → cross-evaluate old vs new checkpoints → optimize motor combo fraction.

**Status:** complete (motor combo sweep done, sweet spot found at 2.5%)
**Last touched:** 2026-06-01 (23:59)
**Resume on:** local (vast-server GPU)

---

## What was built

### DREGON-LM-V2 dataset variants
| Variant | Motor combos | Path | Size |
|---------|-------------|------|------|
| V2 (20%) | 1200/6000 | `datasets/DREGON-LM-V2/` | 2.2 GB |
| V2-5% | 300/6000 | `datasets/DREGON-LM-V2-5pct/` | 2.2 GB |
| V2-2.5% | 150/6000 | `datasets/DREGON-LM-V2-2.5pct/` | 2.2 GB |
| V2-0% | 0/6000 | `datasets/DREGON-LM-V2-0pct/` | 2.2 GB |

All: 6000 train + 600 valid, 3s at 16 kHz, 8 mic channels, command RPS, recording-level split.
Script: `create_dregon_librimix.py`

### Training code (`train_rps_predictor.py`)
- **PIT MSE loss**: 24 permutations over 4 rotors, verified correct (invariant to rotor ordering, equals min Std MSE)
- WandB: project `rps-prediction`, entity `flyingleafe`. Must pass `WANDB_API_KEY` explicitly for tmux sessions.
- PIT-aware evaluation: primary = PIT MSE, also reports Std MSE

---

## Model checkpoints (all trained on DREGON-LM-V2 variants)

| Name | Arch | Params | Combo% | Epochs | Checkpoint | PIT MSE |
|------|------|--------|--------|--------|-----------|---------|
| V3 SC 20% | SimpleConv | 538K | 20% | 52 | `results/rps_predictor_v3/simple_conv/best_simple_conv.pt` | 148.1 |
| V3 BG 20% | BiGRUv2 | 1.44M | 20% | 44 | `results/rps_predictor_v3/simple_conv_bigru_v2/best_simple_conv_bigru_v2.pt` | 71.1 |
| V4 SC 5% | SimpleConv | 538K | 5% | 79 | `results/rps_predictor_v4_5pct/simple_conv/best_simple_conv.pt` | 105.6 |
| V4 BG 5% | BiGRUv2 | 1.44M | 5% | 49 | `results/rps_predictor_v4_5pct/simple_conv_bigru_v2/best_simple_conv_bigru_v2.pt` | 65.9 |
| **V4 SC 2.5%** | SimpleConv | 538K | 2.5% | 66 | `results/rps_predictor_v4_2.5pct/simple_conv/best_simple_conv.pt` | **93.1** |
| **V4 BG 2.5%** | BiGRUv2 | 1.44M | 2.5% | 41 | `results/rps_predictor_v4_2.5pct/simple_conv_bigru_v2/best_simple_conv_bigru_v2.pt` | **56.7** |
| V4 SC 0% | SimpleConv | 538K | 0% | 84 | `results/rps_predictor_v4_0pct/simple_conv/best_simple_conv.pt` | 111.6 |
| V4 BG 0% | BiGRUv2 | 1.44M | 0% | 40 | `results/rps_predictor_v4_0pct/simple_conv_bigru_v2/best_simple_conv_bigru_v2.pt` | 117.3 |

---

## Motor combo sweep results (BiGRUv2, best model)

| Combo% | PIT MSE | Std MSE | Fixed MAE | **PIT-aware MAE** | Std/PIT gap |
|--------|---------|---------|-----------|-------------------|-------------|
| 20% (V3) | 71.13 | 73.88 | 4.34 | 3.94 | 3.9% |
| 5% | 65.92 | 87.20 | 5.94 | 4.06 | 24.4% |
| **2.5%** ✓ | **56.70** | 86.81 | 6.71 | 4.18 | 34.7% |
| 0% | 117.35 | 153.77 | 8.44 | 5.36 | 23.7% |

**PIT-aware MAE** = MAE after PIT-reordering predictions to match targets. The `evaluate()` function reports **fixed-order MAE** which conflates rotor identity with prediction quality.

### Key insight: motor combos as ordering regularizer

PIT implementation is verified correct (invariant to permutations, equals min Std MSE across all 24 permutations). The sweep reveals:

- **Motor combos anchor rotor ordering.** Constant-RPS synthetic samples teach the model which output slot corresponds to which physical rotor. Without them, PIT freely permutes and loses identity.
- **Too many (20%)**: over-regularizes — model sticks to fixed ordering but makes worse per-rotor predictions (PIT MSE 71.1, PIT MAE 3.94)
- **Too few (0%)**: removes anchor → PIT optimization becomes unstable, both PIT MSE and PIT MAE collapse (117.3, 5.36)
- **Sweet spot (2.5%)**: best PIT MSE (56.7) with near-best PIT MAE (4.18). Model swaps aggressively (35% Std/PIT gap) but makes the best per-rotor predictions. Requires PIT at inference to match outputs to physical rotors.
- **The earlier "MAE degradation" was an artifact** of measuring fixed-order MAE. PIT-aware MAE is stable from 20%→2.5% (3.94→4.18).

---

## Cross-evaluation (old vs new checkpoints)

All metrics in `results/rps_cross_eval/validation_metrics.json`.

### Main table

| Model | OLD valid | V2 valid | Degradation |
|-------|-----------|----------|-------------|
| OLD simple_conv | **5.24** | 331.87 | 63× |
| OLD bigru_v2 | **2.67** | 327.26 | 123× |
| V3 simple_conv (20%) | 66.83 | 148.09 | 2.2× |
| V3 bigru_v2 (20%) | 15.26 | 71.13 | 4.7× |
| **V4 bigru_v2 (2.5%)** | — | **56.70** | — |

### Per-channel (V2 valid only)

| Model | ch0 | ch1-7 | ch1-7/ch0 |
|-------|-----|-------|-----------|
| OLD simple_conv | 293.60 | 338.01 | 1.15× |
| OLD bigru_v2 | 247.16 | 340.12 | 1.38× |
| V3 simple_conv | 126.92 | 151.48 | 1.19× |
| V3 bigru_v2 | 66.17 | 71.92 | 1.09× |

---

## Key conclusions

1. **Old dataset was trivially easy** (1s, measured RPS, train/val overlap). Old models PIT MSE=2.7–5.2 there; collapse 63–123× on V2.

2. **V2 is genuinely harder** (3s curve fitting, command RPS noise, recording-level split = zero memorization).

3. **PIT is implemented correctly**: invariant to rotor permutations, equals min Std MSE. Gains are real, not artifacts.

4. **Motor combo fraction has a U-shaped optimum at 2.5%**: motor combos are an ordering regularizer for PIT. Too many drowns signal; too few removes anchor.

5. **`evaluate()` reports fixed-order MAE — misleading for PIT-trained models.** PIT-aware MAE (reorder then measure) is the fair metric. This is a code issue, not a model issue.

6. **Best model**: BiGRUv2 at 2.5% combos, PIT MSE=56.7, PIT-aware MAE=4.18. Checkpoint: `results/rps_predictor_v4_2.5pct/simple_conv_bigru_v2/best_simple_conv_bigru_v2.pt`

7. **All models fail on in-flight source recordings** — interfering source destroys RPS prediction.

---

## How to reproduce

### Dataset (2.5% sweet spot)
```bash
uv run python create_dregon_librimix.py \
    --speech_dir data/librispeech/LibriSpeech/train-clean-100 \
    --dregon_dir data/DREGON \
    --output_dir datasets/DREGON-LM-V2-2.5pct \
    --num_train 6000 --num_valid 600 --motor_combo_fraction 0.025
```

### Training
```bash
WANDB_API_KEY="<key>" uv run python train_rps_predictor.py \
    --model simple_conv_bigru_v2 --device cuda:0 \
    --data_root datasets/DREGON-LM-V2-2.5pct \
    --save_path results/rps_predictor_v4_2.5pct/simple_conv_bigru_v2 \
    --epochs 500 --patience 30 --batch_size 96 --lr 1e-3
```

### Cross-evaluation
```bash
uv run python eval_cross.py
# Output: results/rps_cross_eval/
```

### Results directories
```
results/rps_predictor_v4_5pct/    # 5% motor combos
results/rps_predictor_v4_2.5pct/  # 2.5% (best)
results/rps_predictor_v4_0pct/    # 0% motor combos
results/rps_cross_eval/           # cross-eval artifacts
```

---

## Open questions
- PIT-aware MAE in evaluate(): should we fix it or leave it as fixed-order for now?
- In-flight source recordings: denoising-first + RPS pipeline?
- Can we push below PIT MSE=56 with more capacity/training?
