# Autoresearch Experiments — 20260617-012233-dregon-lm-v4-michaels-simple-conv-v2

## Fixed context

- Dataset: `/gpfs/scratch/acw592/datasets/DREGON-LM-V4-michaels`
- Results root: `/gpfs/scratch/acw592/results/autoresearch/20260617-012233-dregon-lm-v4-michaels-simple-conv-v2`
- Baseline: `simple_conv_v2`
- Target metrics: PIT MSE (lower is better), R^2 (higher is better, 1.0 is max)
- Training budget: 50 epochs, patience 10, gpushort <= 1:00:00
- Extra training args: `--batch_size 32 --lr 1e-3 --weight_decay 1e-4 --loss pit_mse --epoch-progress`

## Experiment log

Record exact commands, job IDs, log paths, failures, fixes, and conclusions.

### E0 — Baseline simple_conv_v2

Status: completed

Submitted: 2026-06-17 01:23 BST

Job:

- Slurm job id: `12513837`
- Job name: `ar_012233_simplecv2`
- Log: `/gpfs/scratch/acw592/logs/ar_012233_simplecv2.o12513837`
- Save path: `/gpfs/scratch/acw592/results/autoresearch/20260617-012233-dregon-lm-v4-michaels-simple-conv-v2/simple_conv_v2`

Command:

```bash
python train_rps_predictor.py --model simple_conv_v2 --device cuda:0 --data_root /gpfs/scratch/acw592/datasets/DREGON-LM-V4-michaels --save_path /gpfs/scratch/acw592/results/autoresearch/20260617-012233-dregon-lm-v4-michaels-simple-conv-v2/simple_conv_v2 --epochs 50 --patience 10 --batch_size 32 --lr 1e-3 --weight_decay 1e-4 --loss pit_mse --epoch-progress
```

Notes:

- Initial Slurm status was `PENDING`; per autoresearch rule, no candidate jobs were submitted in this cycle after the baseline entered the queue.
- Added a compatibility parser alias in `train_rps_predictor.py` so the fixed autoresearch flag `--loss pit_mse` maps to the existing PIT-MSE training path (`args.pit_loss=True`).
- After baseline submission, proposed H1–H4 in `ideas.md`; no candidate implementation/submission yet because the first Slurm job initially queued as `PENDING`, triggering the workflow stop condition for new submissions.

Results:

- Slurm status: `COMPLETED` (elapsed `00:23:04`)
- Early stopping: epoch 31
- Best/final checkpoint evaluation: PIT MSE `7.8920`, RMSE `2.81`, Std MSE `42.1642`, frame MAE `2.08`, clip MAE `1.62`, R² `0.8183`
- Best epoch by validation PIT: epoch 21 (`Val PIT=7.8920`, `R²=0.8183`)

### E1 — Candidate simple_conv_v2_transformer

Status: completed

Hypothesis: H1 — replace the BiGRU temporal head in `simple_conv_v2` with a small Transformer encoder while preserving the same STFT magnitude front-end, residual+SE encoder, and attention frequency pooling.

Implementation:

- Added `TemporalTransformerHead` and `SimpleConvV2Transformer` in `src/models/rps_predictor.py`.
- Registered model key `simple_conv_v2_transformer` in both `src/models/rps_predictor.py::RPS_MODEL_REGISTRY` and `train_rps_predictor.py::MODEL_REGISTRY`.

Smoke test:

```bash
python - <<'PY'
import torch
from train_rps_predictor import get_model
model_name = "simple_conv_v2_transformer"
model = get_model(model_name, n_fft=2048, hop_length=512, num_rotors=4).eval()
audio = torch.randn(2, 48000)
with torch.no_grad():
    out = model(audio)
print(model_name, tuple(out.shape))
assert out.ndim == 3, out.shape
assert out.shape[0] == 2 and out.shape[1] == 4, out.shape
PY
```

Output: `simple_conv_v2_transformer (2, 4, 94)`.

Job:

- Submitted: 2026-06-17 10:46 BST
- Slurm job id: `12521795`
- Job name: `ar_012233_v2trans`
- Initial submit status: `PENDING` (therefore no further candidate submissions this cycle)
- Final Slurm status: `COMPLETED` (elapsed `00:11:10`)
- Log: `/gpfs/scratch/acw592/logs/ar_012233_v2trans.o12521795`
- Save path: `/gpfs/scratch/acw592/results/autoresearch/20260617-012233-dregon-lm-v4-michaels-simple-conv-v2/simple_conv_v2_transformer`

Command:

```bash
python train_rps_predictor.py --model simple_conv_v2_transformer --device cuda:0 --data_root /gpfs/scratch/acw592/datasets/DREGON-LM-V4-michaels --save_path /gpfs/scratch/acw592/results/autoresearch/20260617-012233-dregon-lm-v4-michaels-simple-conv-v2/simple_conv_v2_transformer --epochs 50 --patience 10 --batch_size 32 --lr 1e-3 --weight_decay 1e-4 --loss pit_mse --epoch-progress
```

Results:

- Early stopping: epoch 21
- Best/final checkpoint evaluation: PIT MSE `43.5184`, RMSE `6.60`, Std MSE `70.5665`, frame MAE `5.03`, clip MAE `4.58`, R² `-0.6571`
- Best epoch by validation PIT: epoch 11 (`Val PIT=43.5184`, `R²=-0.6571`)
- Conclusion: substantially worse than `simple_conv_v2` baseline (`7.8920`, R² `0.8183`). This rejects the naive “replace BiGRU with small global Transformer” variant under the fixed training budget.

### E2 — Candidate simple_conv_v2_local_attn

Status: completed

Hypothesis: H2 — keep the `simple_conv_v2` STFT magnitude front-end, residual+SE encoder, and attention frequency pooling, but replace the BiGRU with a Transformer encoder constrained by a local temporal attention mask.

Implementation:

- Added `LocalTemporalTransformerHead` and `SimpleConvV2LocalAttention` in `src/models/rps_predictor.py`.
- Registered model key `simple_conv_v2_local_attn` in both `src/models/rps_predictor.py::RPS_MODEL_REGISTRY` and `train_rps_predictor.py::MODEL_REGISTRY`.
- Local window: 17 STFT frames (~0.54 s at 16 kHz / hop 512).

Smoke test output: `simple_conv_v2_local_attn (2, 4, 94)`.

Job:

- Submitted: 2026-06-17 11:05 BST
- Slurm job id: `12522911`
- Job name: `ar_012233_v2local`
- Initial submit status: `PENDING`; follow-up check within ~10 s showed `RUNNING`, so per corrected user policy the loop may continue submitting additional candidates.
- Final Slurm status: `COMPLETED` (elapsed `00:10:05`)
- Log: `/gpfs/scratch/acw592/logs/ar_012233_v2local.o12522911`
- Save path: `/gpfs/scratch/acw592/results/autoresearch/20260617-012233-dregon-lm-v4-michaels-simple-conv-v2/simple_conv_v2_local_attn`

Command:

```bash
python train_rps_predictor.py --model simple_conv_v2_local_attn --device cuda:0 --data_root /gpfs/scratch/acw592/datasets/DREGON-LM-V4-michaels --save_path /gpfs/scratch/acw592/results/autoresearch/20260617-012233-dregon-lm-v4-michaels-simple-conv-v2/simple_conv_v2_local_attn --epochs 50 --patience 10 --batch_size 32 --lr 1e-3 --weight_decay 1e-4 --loss pit_mse --epoch-progress
```

Results:

- Early stopping: epoch 19
- Best/final checkpoint evaluation: PIT MSE `18.5846`, RMSE `4.31`, Std MSE `89.7452`, frame MAE `3.25`, clip MAE `2.71`, R² `0.5213`
- Best epoch by validation PIT: epoch 9 (`Val PIT=18.5846`, `R²=0.5213`)
- Conclusion: local attention is better than global Transformer H1 but still much worse than the BiGRU baseline.

### E3 — Candidate simple_conv_v2_multires

Status: completed

Hypothesis: H3 — concatenate long-window and short-window STFT magnitudes before the `simple_conv_v2` encoder so the model can use both high frequency resolution and better temporal localization.

Implementation:

- Added `SimpleConvV2MultiRes` in `src/models/rps_predictor.py`.
- Uses default `n_fft=2048` STFT magnitude plus a short-window `n_fft=1024` STFT magnitude with the same hop (`512`).
- Interpolates the short-resolution feature map to the long-resolution `(F,T)` grid and concatenates along channel dimension, then uses the same residual+SE encoder, attention frequency pool, and BiGRU head shape as `simple_conv_v2`.
- Registered model key `simple_conv_v2_multires` in both `src/models/rps_predictor.py::RPS_MODEL_REGISTRY` and `train_rps_predictor.py::MODEL_REGISTRY`.

Smoke test output: `simple_conv_v2_multires (2, 4, 94)`.

Job:

- Submitted: 2026-06-17 11:08 BST
- Slurm job id: `12523268`
- Job name: `ar_012233_v2mres`
- Initial submit status: `PENDING`; follow-up after ~11 s showed `RUNNING`, so the loop may continue submitting additional candidates.
- Final Slurm status: `COMPLETED` (elapsed `00:13:10`)
- Log: `/gpfs/scratch/acw592/logs/ar_012233_v2mres.o12523268`
- Save path: `/gpfs/scratch/acw592/results/autoresearch/20260617-012233-dregon-lm-v4-michaels-simple-conv-v2/simple_conv_v2_multires`

Command:

```bash
python train_rps_predictor.py --model simple_conv_v2_multires --device cuda:0 --data_root /gpfs/scratch/acw592/datasets/DREGON-LM-V4-michaels --save_path /gpfs/scratch/acw592/results/autoresearch/20260617-012233-dregon-lm-v4-michaels-simple-conv-v2/simple_conv_v2_multires --epochs 50 --patience 10 --batch_size 32 --lr 1e-3 --weight_decay 1e-4 --loss pit_mse --epoch-progress
```

Results:

- Early stopping: epoch 18
- Best/final checkpoint evaluation: PIT MSE `8.9704`, RMSE `3.00`, Std MSE `104.2420`, frame MAE `2.17`, clip MAE `1.62`, R² `0.8088`
- Best epoch by validation PIT: epoch 8 (`Val PIT=8.9704`, `R²=0.8088`)
- Conclusion: near baseline but slightly worse; multi-resolution STFT did not improve the fixed-budget score.

### E4 — Candidate simple_conv_v2_dwt

Status: completed

Hypothesis: H4 — augment STFT magnitude with a lightweight wavelet-like temporal feature branch that exposes multi-scale time-domain change/periodicity cues while avoiding extra dependencies.

Implementation:

- Added `SimpleConvV2Wavelet` in `src/models/rps_predictor.py`.
- Computes fixed Haar-like Conv1d responses on raw audio at scales 128, 256, 512, and 1024 samples with stride equal to STFT hop (`512`).
- Applies `log1p(abs(.))`, a tiny Conv1d projection to one temporal channel, broadcasts across frequency, concatenates with STFT magnitude, then reuses the `simple_conv_v2` encoder/pool/BiGRU shape.
- Registered model key `simple_conv_v2_dwt` in both `src/models/rps_predictor.py::RPS_MODEL_REGISTRY` and `train_rps_predictor.py::MODEL_REGISTRY`.

Smoke test output: `simple_conv_v2_dwt (2, 4, 94)`.

Job:

- Submitted: 2026-06-17 11:11 BST
- Slurm job id: `12523611`
- Job name: `ar_012233_v2dwt`
- Initial submit status: `PENDING`; follow-up after ~11 s remained `PENDING` with reason `QOSMaxGRESPerUser`, so no further jobs submitted.
- Started after an earlier job completed; final Slurm status: `COMPLETED` (elapsed `00:13:59`)
- Log: `/gpfs/scratch/acw592/logs/ar_012233_v2dwt.o12523611`
- Save path: `/gpfs/scratch/acw592/results/autoresearch/20260617-012233-dregon-lm-v4-michaels-simple-conv-v2/simple_conv_v2_dwt`

Command:

```bash
python train_rps_predictor.py --model simple_conv_v2_dwt --device cuda:0 --data_root /gpfs/scratch/acw592/datasets/DREGON-LM-V4-michaels --save_path /gpfs/scratch/acw592/results/autoresearch/20260617-012233-dregon-lm-v4-michaels-simple-conv-v2/simple_conv_v2_dwt --epochs 50 --patience 10 --batch_size 32 --lr 1e-3 --weight_decay 1e-4 --loss pit_mse --epoch-progress
```

Results:

- Early stopping: epoch 27
- Best/final checkpoint evaluation: PIT MSE `8.8957`, RMSE `2.98`, Std MSE `74.7740`, frame MAE `2.12`, clip MAE `1.70`, R² `0.8133`
- Best epoch by validation PIT: epoch 17 (`Val PIT=8.8957`, `R²=0.8133`)
- Conclusion: closest candidate in this batch but still worse than baseline (`7.8920`, R² `0.8183`).

### E5 — Candidate simple_conv_v2_magphase

Status: completed

Hypothesis: H5 — keep the `simple_conv_v2` residual+SE encoder, attention frequency pooling, and BiGRU temporal head, but use `stft_magphase` input (log magnitude + cos/sin phase) instead of magnitude only.

Implementation:

- Added `SimpleConvV2MagPhase` in `src/models/rps_predictor.py`.
- Registered model key `simple_conv_v2_magphase` in both `src/models/rps_predictor.py::RPS_MODEL_REGISTRY` and `train_rps_predictor.py::MODEL_REGISTRY`.

Smoke test output: `simple_conv_v2_magphase (2, 4, 94)`.

Job:

- Submitted: 2026-06-17 11:37 BST
- Slurm job id: `12524918`
- Job name: `ar_012233_v2phase`
- Initial submit status: `PENDING`; follow-up after ~11 s showed `RUNNING`, so the loop may continue submitting additional candidates.
- Final Slurm status: `COMPLETED` (elapsed `00:13:26`)
- Log: `/gpfs/scratch/acw592/logs/ar_012233_v2phase.o12524918`
- Save path: `/gpfs/scratch/acw592/results/autoresearch/20260617-012233-dregon-lm-v4-michaels-simple-conv-v2/simple_conv_v2_magphase`

Command:

```bash
python train_rps_predictor.py --model simple_conv_v2_magphase --device cuda:0 --data_root /gpfs/scratch/acw592/datasets/DREGON-LM-V4-michaels --save_path /gpfs/scratch/acw592/results/autoresearch/20260617-012233-dregon-lm-v4-michaels-simple-conv-v2/simple_conv_v2_magphase --epochs 50 --patience 10 --batch_size 32 --lr 1e-3 --weight_decay 1e-4 --loss pit_mse --epoch-progress
```

Results:

- Early stopping: epoch 27
- Best/final checkpoint evaluation: PIT MSE `10.4266`, RMSE `3.23`, Std MSE `87.2740`, frame MAE `2.39`, clip MAE `1.85`, R² `0.7466`
- Best epoch by validation PIT: epoch 17 (`Val PIT=10.4266`, `R²=0.7466`)
- Conclusion: phase channels hurt relative to magnitude-only `simple_conv_v2`.

### E6 — Candidate simple_conv_v2_dual_pool

Status: completed

Hypothesis: H6 — preserve both learned attention frequency pooling and plain mean frequency pooling by concatenating them before the BiGRU head.

Implementation:

- Added `SimpleConvV2DualPool` in `src/models/rps_predictor.py`.
- Registered model key `simple_conv_v2_dual_pool` in both `src/models/rps_predictor.py::RPS_MODEL_REGISTRY` and `train_rps_predictor.py::MODEL_REGISTRY`.

Smoke test output: `simple_conv_v2_dual_pool (2, 4, 94)`.

Job:

- Submitted: 2026-06-17 11:40 BST
- Slurm job id: `12524982`
- Job name: `ar_012233_v2dpool`
- Initial submit status: `PENDING`; follow-up after ~11 s showed `RUNNING`, so the loop may continue submitting additional candidates.
- Final Slurm status: `COMPLETED` (elapsed `00:20:22`)
- Log: `/gpfs/scratch/acw592/logs/ar_012233_v2dpool.o12524982`
- Save path: `/gpfs/scratch/acw592/results/autoresearch/20260617-012233-dregon-lm-v4-michaels-simple-conv-v2/simple_conv_v2_dual_pool`

Command:

```bash
python train_rps_predictor.py --model simple_conv_v2_dual_pool --device cuda:0 --data_root /gpfs/scratch/acw592/datasets/DREGON-LM-V4-michaels --save_path /gpfs/scratch/acw592/results/autoresearch/20260617-012233-dregon-lm-v4-michaels-simple-conv-v2/simple_conv_v2_dual_pool --epochs 50 --patience 10 --batch_size 32 --lr 1e-3 --weight_decay 1e-4 --loss pit_mse --epoch-progress
```

Results:

- Early stopping: epoch 28
- Best/final checkpoint evaluation: PIT MSE `9.8217`, RMSE `3.13`, Std MSE `110.3518`, frame MAE `2.37`, clip MAE `1.93`, R² `0.7462`
- Best epoch by validation PIT: epoch 18 (`Val PIT=9.8217`, `R²=0.7462`)
- Conclusion: mean+attention pooling did not help; it underperformed the baseline and the closest feature-augmentation variants.

### E7 — Candidate simple_conv_v2_gru96

Status: completed

Hypothesis: H7 — keep `simple_conv_v2` unchanged except increase the BiGRU hidden size from 64 to 96.

Implementation:

- Added `SimpleConvV2GRU96` in `src/models/rps_predictor.py`.
- Registered model key `simple_conv_v2_gru96` in both `src/models/rps_predictor.py::RPS_MODEL_REGISTRY` and `train_rps_predictor.py::MODEL_REGISTRY`.

Smoke test output: `simple_conv_v2_gru96 (2, 4, 94)`.

Job:

- Submitted: 2026-06-17 11:43 BST
- Slurm job id: `12525024`
- Job name: `ar_012233_v2gru96`
- Initial submit status: `PENDING`; follow-up after ~11 s remained `PENDING` with reason `QOSMaxGRESPerUser`, so no further jobs submitted.
- Started after an earlier job completed; final Slurm status: `COMPLETED` (elapsed `00:21:50`)
- Log: `/gpfs/scratch/acw592/logs/ar_012233_v2gru96.o12525024`
- Save path: `/gpfs/scratch/acw592/results/autoresearch/20260617-012233-dregon-lm-v4-michaels-simple-conv-v2/simple_conv_v2_gru96`

Command:

```bash
python train_rps_predictor.py --model simple_conv_v2_gru96 --device cuda:0 --data_root /gpfs/scratch/acw592/datasets/DREGON-LM-V4-michaels --save_path /gpfs/scratch/acw592/results/autoresearch/20260617-012233-dregon-lm-v4-michaels-simple-conv-v2/simple_conv_v2_gru96 --epochs 50 --patience 10 --batch_size 32 --lr 1e-3 --weight_decay 1e-4 --loss pit_mse --epoch-progress
```

Results:

- Early stopping: epoch 45
- Best/final checkpoint evaluation: PIT MSE `8.6612`, RMSE `2.94`, Std MSE `12.9412`, frame MAE `2.07`, clip MAE `1.62`, R² `0.8216`
- Best epoch by validation PIT: epoch 35 (`Val PIT=8.6612`, `R²=0.8216`)
- Conclusion: best candidate so far and slightly better than baseline on R²/frame MAE, but worse on primary PIT MSE (`8.6612` vs `7.8920`).

### E8 — Candidate simple_conv_v2_uni_gru

Status: completed — failed/unstable

Hypothesis: H8 — replace only the BiGRU temporal head with a unidirectional, causal-prenet GRU while preserving the existing centered STFT and symmetric temporal encoder.

Implementation:

- Added `CausalGRUHead` with left-padded Conv1d prenet and unidirectional GRU in `src/models/rps_predictor.py`.
- Added `SimpleConvV2UniGRU` and registered model key `simple_conv_v2_uni_gru` in both registries.
- Note: this is causal in the recurrent head only; the existing STFT and 2D encoder still use lookahead.

Smoke test output: `simple_conv_v2_uni_gru (2, 4, 94)`.

Job:

- Submitted: 2026-06-17 12:33 BST
- Slurm job id: `12530583`
- Job name: `ar_012233_v2ugru`
- Initial submit status: `PENDING`; follow-up after ~11 s showed `RUNNING`, so the loop continued to H9.
- Final Slurm status: `COMPLETED` (elapsed `00:07:09`)
- Log: `/gpfs/scratch/acw592/logs/ar_012233_v2ugru.o12530583`
- Save path: `/gpfs/scratch/acw592/results/autoresearch/20260617-012233-dregon-lm-v4-michaels-simple-conv-v2/simple_conv_v2_uni_gru`

Command:

```bash
python train_rps_predictor.py --model simple_conv_v2_uni_gru --device cuda:0 --data_root /gpfs/scratch/acw592/datasets/DREGON-LM-V4-michaels --save_path /gpfs/scratch/acw592/results/autoresearch/20260617-012233-dregon-lm-v4-michaels-simple-conv-v2/simple_conv_v2_uni_gru --epochs 50 --patience 10 --batch_size 32 --lr 1e-3 --weight_decay 1e-4 --loss pit_mse --epoch-progress
```

Results:

- Early stopping: epoch 14
- Best/final checkpoint evaluation: PIT MSE `228.6723`, RMSE `15.12`, Std MSE `270.6812`, frame MAE `9.82`, clip MAE `9.13`, R² `-10.4445`
- Best epoch by validation PIT: epoch 4 (`Train=274.9730`, `Val PIT=228.6723`, `R²=-10.4445`)
- Failure detail: epoch rows became `nan` after epoch 10/11; final evaluation loaded the epoch-4 checkpoint. This is not overfitting; it is instability plus failure to fit/use the training data well.

### E9 — Candidate simple_conv_v2_causal_gru

Status: completed — failed to fit

Hypothesis: H9 — make the neural stack time-causal via causal STFT framing, left-padded temporal Conv2d blocks, and a unidirectional GRU head.

Implementation:

- Added `CausalSTFTMag` using left-padded `torch.stft(center=False)` with the same frame count as the centered STFT.
- Added `CausalResidualConvBlock2d` with left-only temporal padding.
- Added `SimpleConvV2CausalGRU` and registered model key `simple_conv_v2_causal_gru` in both registries.

Smoke test output: `simple_conv_v2_causal_gru (2, 4, 94)`.

Job:

- Submitted: 2026-06-17 12:34 BST
- Slurm job id: `12530599`
- Job name: `ar_012233_v2cgru`
- Initial submit status: `PENDING`; follow-up after ~11 s remained `PENDING` with reason `Priority`, so no further jobs were submitted at that moment.
- Later status check showed `RUNNING`, allowing H10 submission.
- Final Slurm status: `COMPLETED` (elapsed `00:24:53`)
- Log: `/gpfs/scratch/acw592/logs/ar_012233_v2cgru.o12530599`
- Save path: `/gpfs/scratch/acw592/results/autoresearch/20260617-012233-dregon-lm-v4-michaels-simple-conv-v2/simple_conv_v2_causal_gru`

Command:

```bash
python train_rps_predictor.py --model simple_conv_v2_causal_gru --device cuda:0 --data_root /gpfs/scratch/acw592/datasets/DREGON-LM-V4-michaels --save_path /gpfs/scratch/acw592/results/autoresearch/20260617-012233-dregon-lm-v4-michaels-simple-conv-v2/simple_conv_v2_causal_gru --epochs 50 --patience 10 --batch_size 32 --lr 1e-3 --weight_decay 1e-4 --loss pit_mse --epoch-progress
```

Results:

- Ran all 50 epochs (no early stopping).
- Best/final checkpoint evaluation: PIT MSE `83.5143`, RMSE `9.14`, Std MSE `87.9984`, frame MAE `5.50`, clip MAE `5.08`, R² `-2.8866`
- Best epoch by validation PIT: epoch 49 (`Train=15.4060`, `Val PIT=83.5143`, `R²=-2.8866`)
- Conclusion: severe underfit/failure to learn a good mapping. The causal STFT framing and left-padded encoder likely introduce an alignment/latency penalty beyond just removing bidirectional recurrence.

### E10 — Candidate simple_conv_v2_causal_gru96

Status: completed — failed to fit

Hypothesis: H10 — widen the fully causal unidirectional GRU to 96 hidden units to recover some capacity lost by removing bidirectional recurrence.

Implementation:

- Added `SimpleConvV2CausalGRU96`, registered as `simple_conv_v2_causal_gru96`, reusing the causal front-end and causal encoder from H9 with `hidden_ch=96`.

Smoke test output: `simple_conv_v2_causal_gru96 (2, 4, 94)`.

Job:

- Submitted: 2026-06-17 12:35 BST
- Slurm job id: `12530631`
- Job name: `ar_012233_v2cgr96`
- Initial submit status: `PENDING`; follow-up after ~11 s remained `PENDING` with reason `Priority`, so no further jobs submitted.
- Final Slurm status: `COMPLETED` (elapsed `00:08:59`)
- Log: `/gpfs/scratch/acw592/logs/ar_012233_v2cgr96.o12530631`
- Save path: `/gpfs/scratch/acw592/results/autoresearch/20260617-012233-dregon-lm-v4-michaels-simple-conv-v2/simple_conv_v2_causal_gru96`

Command:

```bash
python train_rps_predictor.py --model simple_conv_v2_causal_gru96 --device cuda:0 --data_root /gpfs/scratch/acw592/datasets/DREGON-LM-V4-michaels --save_path /gpfs/scratch/acw592/results/autoresearch/20260617-012233-dregon-lm-v4-michaels-simple-conv-v2/simple_conv_v2_causal_gru96 --epochs 50 --patience 10 --batch_size 32 --lr 1e-3 --weight_decay 1e-4 --loss pit_mse --epoch-progress
```

Results:

- Early stopping: epoch 18
- Best/final checkpoint evaluation: PIT MSE `253.5939`, RMSE `15.92`, Std MSE `291.9453`, frame MAE `8.14`, clip MAE `7.73`, R² `-12.6410`
- Best epoch by validation PIT: epoch 8 (`Train=39.3973`, `Val PIT=253.5939`, `R²=-12.6410`)
- Conclusion: widening the fully causal head did not recover capacity; it stalled at high training loss.

### E11 — Candidate simple_conv_v2_uni_gru128

Status: completed — failed/unstable

Hypothesis: H11 — use a unidirectional GRU with hidden size 128 to match the BiGRU baseline's output width (`2*64`) while preserving the baseline STFT and encoder.

Implementation:

- Added `SimpleConvV2UniGRU128`, registered as `simple_conv_v2_uni_gru128`.
- Reuses `CausalGRUHead` with `hidden_ch=128`.

Smoke test output: `simple_conv_v2_uni_gru128 (2, 4, 94)`.

Job:

- Submitted: 2026-06-17 13:09 BST
- Slurm job id: `12533051`
- Job name: `ar_012233_v2ug128`
- Initial submit status: `PENDING`; follow-up after ~11 s showed `RUNNING`, so the loop continued to H12.
- Final Slurm status: `COMPLETED` (elapsed `00:13:57`)
- Log: `/gpfs/scratch/acw592/logs/ar_012233_v2ug128.o12533051`
- Save path: `/gpfs/scratch/acw592/results/autoresearch/20260617-012233-dregon-lm-v4-michaels-simple-conv-v2/simple_conv_v2_uni_gru128`

Command:

```bash
python train_rps_predictor.py --model simple_conv_v2_uni_gru128 --device cuda:0 --data_root /gpfs/scratch/acw592/datasets/DREGON-LM-V4-michaels --save_path /gpfs/scratch/acw592/results/autoresearch/20260617-012233-dregon-lm-v4-michaels-simple-conv-v2/simple_conv_v2_uni_gru128 --epochs 50 --patience 10 --batch_size 32 --lr 1e-3 --weight_decay 1e-4 --loss pit_mse --epoch-progress
```

Results:

- Early stopping: epoch 17
- Best/final checkpoint evaluation: PIT MSE `39.8099`, RMSE `6.31`, Std MSE `91.8041`, frame MAE `4.12`, clip MAE `3.80`, R² `-0.5486`
- Best epoch by validation PIT: epoch 7 (`Train=9.0165`, `Val PIT=39.8099`, `R²=-0.5486`)
- Failure detail: one epoch logged `nan` train loss; capacity improved H8 but remained unstable and much worse than baseline.

### E12 — Candidate simple_conv_v2_uni_gru128_norm

Status: completed — overfit/generalization gap

Hypothesis: H12 — add GroupNorm after the causal Conv1d prenet of H11 to stabilize the fixed-LR/AMP training that produced NaNs in H8.

Implementation:

- Added `CausalGRUNormHead` and `SimpleConvV2UniGRU128Norm`, registered as `simple_conv_v2_uni_gru128_norm`.

Smoke test output: `simple_conv_v2_uni_gru128_norm (2, 4, 94)`.

Job:

- Submitted: 2026-06-17 13:10 BST
- Slurm job id: `12533200`
- Job name: `ar_012233_v2ugnrm`
- Initial submit status: `PENDING`; follow-up after ~11 s showed `RUNNING`.
- Final Slurm status: `COMPLETED` (elapsed `00:36:21`)
- Log: `/gpfs/scratch/acw592/logs/ar_012233_v2ugnrm.o12533200`
- Save path: `/gpfs/scratch/acw592/results/autoresearch/20260617-012233-dregon-lm-v4-michaels-simple-conv-v2/simple_conv_v2_uni_gru128_norm`

Command:

```bash
python train_rps_predictor.py --model simple_conv_v2_uni_gru128_norm --device cuda:0 --data_root /gpfs/scratch/acw592/datasets/DREGON-LM-V4-michaels --save_path /gpfs/scratch/acw592/results/autoresearch/20260617-012233-dregon-lm-v4-michaels-simple-conv-v2/simple_conv_v2_uni_gru128_norm --epochs 50 --patience 10 --batch_size 32 --lr 1e-3 --weight_decay 1e-4 --loss pit_mse --epoch-progress
```

Results:

- Ran all 50 epochs (no early stopping).
- Best/final checkpoint evaluation: PIT MSE `20.2943`, RMSE `4.50`, Std MSE `76.0374`, frame MAE `2.34`, clip MAE `1.81`, R² `0.7391`
- Best epoch by validation PIT: epoch 42 (`Train=1.5089`, `Val PIT=20.2943`, `R²=0.7391`)
- Minimum train loss: epoch 50 (`Train=1.2512`, `Val PIT=25.7291`)
- Conclusion: GroupNorm fixed NaNs and trained strongly, but the causal head overfits/generalizes poorly versus the BiGRU baseline.
