# DREGON-LM-V2 dataset + RPS predictor training

**Goal:** Better DREGON-LM train/validation sets → train RPS prediction models with PIT loss.
**Status:** in-progress (training done, overfitting observed)
**Last touched:** 2026-06-01
**Resume on:** local (vast-server GPU)

## Done
- Fixed motor recording discovery in `data_processing/dregon.py` (`glob` → `rglob`)
- Rewrote `create_dregon_librimix.py` (v2):
  - All 8 mic channels (each as independent noise source)
  - 3-second samples exactly (48k samples at 16 kHz)
  - Command speeds + `clean_command_spikes` (existing function in dregon.py)
  - Recording-level train/valid split (zero overlap): train = 6 in_flight_nosource recordings, valid = free-flight_whitenoise-low_room1 + free-flight_speech-low_room1
  - 20% synthetic motor combos (sum same channel across different motors, per-channel SPL interpolation)
  - Optional white noise augmentation
  - Output: `datasets/DREGON-LM-V2/` (6000 train + 600 valid, 2.2 GB)
- Updated `train_rps_predictor.py`:
  - WandB logging to project `rps-prediction` (same pattern as train.py)
  - PIT (permutation-invariant) MSE loss: 4! = 24 permutations, pairwise MSE, best-permutation minimum
  - Default data_root = `datasets/DREGON-LM-V2`
- Trained two models in parallel tmux windows on RTX 4070 Ti GPUs:
  - `simple_conv` (538K params) on cuda:0 → best val MSE=176, 32 epochs, 9.3 min
  - `simple_conv_bigru_v2` (1.44M params) on cuda:1 → best val MSE=74, 37 epochs, 13.7 min
- Results saved to `results/rps_predictor_v2/`
  - Checkpoints: `simple_conv/best_simple_conv.pt`, `simple_conv_bigru_v2/best_simple_conv_bigru_v2.pt`
  - Logs: `simple_conv.log`, `simple_conv_bigru_v2.log`
  - Comparison: `comparison.json`

## Pending
1. Address overfitting: train loss drops (2021→128 for SimpleConv) but val loss oscillates without improving past epoch 17/22
   - Options: stronger regularization (dropout, weight decay), data augmentation, reduce model capacity, longer training with better LR schedule
   - Compare with old dataset results: old SimpleConv trained 99 epochs (64 min) to val MSE=5.2 on much easier task
2. Evaluate PIT-aware metrics properly (per-sample R² is broken due to near-constant-RPS samples → SS_tot ≈ 0)
3. Consider training the denoising models (DCUNet etc.) on DREGON-LM-V2
4. Push dataset to DVC if needed for other machines

## State
- Working tree: clean (changes committed as `58da9d6` and `8817527`)
- `datasets/DREGON-LM-V2/`: 2.2 GB, 6600 samples, NOT in git (gitignored)
- `results/rps_predictor_v2/`: checkpoints and logs, NOT in git
- Old dataset `datasets/DREGON-LM/` still intact (not deleted)
- Two tmux sessions completed and exited (no running processes)

## Decisions (do not relitigate)
- Motor combos sum same channel across different motors (physically meaningful per-microphone simulation)
- PIT loss uses brute-force 24 permutations (4 rotors, cheap)
- PIT loss normalized by n_rotors=4 for comparability with standard MSE
- 3-second samples at 16 kHz (not 8.2s — RPS prediction task, not denoising)
- Old DREGON-LM preserved as-is; new dataset in separate directory

## Open questions
- Why does overfitting hit so much harder on V2? Is it truly harder task or is there a data issue?
  - Old dataset had train/val overlap (same recordings) → inflated val scores
  - V2 has 3× longer samples with more temporal variation
  - Command RPS harder to predict than measured? (measured is smoother)
  - Motor combos may create unrealistic easy/hard patterns
- Should we try without PIT loss? (PIT only gave ~8% MSE improvement)
- What training duration is actually needed? Models plateaued at epoch 17/22 — is that a genuine capacity limit or early stopping patience too low?

## Resume
```bash
cd /root/harmonic-noise-suppression
# Dataset is ready at datasets/DREGON-LM-V2/
# Checkpoints at results/rps_predictor_v2/

# To evaluate:
uv run python train_rps_predictor.py --model simple_conv --device cuda:0 --data_root datasets/DREGON-LM-V2 --epochs 1

# To resume training from checkpoint (not implemented in train_rps_predictor.py yet):
# Would need to add --resume flag

# To re-train with different params:
tmux new-session -d -s rps_retry \
    "uv run python train_rps_predictor.py \
        --model simple_conv_bigru_v2 \
        --device cuda:0 \
        --data_root datasets/DREGON-LM-V2 \
        --save_path results/rps_predictor_v2/simple_conv_bigru_v2_retry \
        --epochs 500 --patience 50 --batch_size 32 --lr 3e-4 \
        2>&1 | tee results/rps_predictor_v2/simple_conv_bigru_v2_retry.log"
```
