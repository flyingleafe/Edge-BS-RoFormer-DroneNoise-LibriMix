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

### E13 — Candidate simple_conv_v2_uni_gru128_norm_do03

Status: completed — failed to generalize

Hypothesis: H13 — keep H12's normalized capacity-matched unidirectional GRU head, but increase head dropout from `0.1` to `0.3` to combat the observed train/validation gap.

Implementation:

- Added `SimpleConvV2UniGRU128NormDO03`, registered as `simple_conv_v2_uni_gru128_norm_do03`.

Smoke test output: `simple_conv_v2_uni_gru128_norm_do03 (2, 4, 94)`.

Job:

- Submitted: 2026-06-17 14:22 BST
- Slurm job id: `12538666`
- Job name: `ar_012233_v2ugdo`
- Initial submit status: `PENDING`; follow-up after ~11 s showed `RUNNING`, so the loop continued to H14.
- Final Slurm status: `COMPLETED` (elapsed `00:09:09`)
- Log: `/gpfs/scratch/acw592/logs/ar_012233_v2ugdo.o12538666`
- Save path: `/gpfs/scratch/acw592/results/autoresearch/20260617-012233-dregon-lm-v4-michaels-simple-conv-v2/simple_conv_v2_uni_gru128_norm_do03`

Command:

```bash
python train_rps_predictor.py --model simple_conv_v2_uni_gru128_norm_do03 --device cuda:0 --data_root /gpfs/scratch/acw592/datasets/DREGON-LM-V4-michaels --save_path /gpfs/scratch/acw592/results/autoresearch/20260617-012233-dregon-lm-v4-michaels-simple-conv-v2/simple_conv_v2_uni_gru128_norm_do03 --epochs 50 --patience 10 --batch_size 32 --lr 1e-3 --weight_decay 1e-4 --loss pit_mse --epoch-progress
```

Results:

- Early stopping: epoch 16
- Best/final checkpoint evaluation: PIT MSE `218.0722`, RMSE `14.77`, Std MSE `257.2070`, frame MAE `7.49`, clip MAE `7.22`, R² `-11.7088`
- Best epoch by validation PIT: epoch 6 (`Train=9.2616`, `Val PIT=218.0722`, `R²=-11.7088`)
- Conclusion: dropout `0.3` with hidden 128 was too disruptive; failed to generalize despite no NaNs.

### E14 — Candidate simple_conv_v2_uni_gru96_norm_do03

Status: completed — best causal so far

Hypothesis: H14 — reduce normalized causal GRU hidden size to 96 and use dropout `0.3`, seeking a better bias/variance tradeoff than H12.

Implementation:

- Added `SimpleConvV2UniGRU96NormDO03`, registered as `simple_conv_v2_uni_gru96_norm_do03`.

Smoke test output: `simple_conv_v2_uni_gru96_norm_do03 (2, 4, 94)`.

Job:

- Submitted: 2026-06-17 14:23 BST
- Slurm job id: `12538698`
- Job name: `ar_012233_v2ug96d`
- Initial submit status: `PENDING`; follow-up after ~11 s showed `RUNNING`.
- Final Slurm status: `COMPLETED` (elapsed `00:12:47`)
- Log: `/gpfs/scratch/acw592/logs/ar_012233_v2ug96d.o12538698`
- Save path: `/gpfs/scratch/acw592/results/autoresearch/20260617-012233-dregon-lm-v4-michaels-simple-conv-v2/simple_conv_v2_uni_gru96_norm_do03`

Command:

```bash
python train_rps_predictor.py --model simple_conv_v2_uni_gru96_norm_do03 --device cuda:0 --data_root /gpfs/scratch/acw592/datasets/DREGON-LM-V4-michaels --save_path /gpfs/scratch/acw592/results/autoresearch/20260617-012233-dregon-lm-v4-michaels-simple-conv-v2/simple_conv_v2_uni_gru96_norm_do03 --epochs 50 --patience 10 --batch_size 32 --lr 1e-3 --weight_decay 1e-4 --loss pit_mse --epoch-progress
```

Results:

- Early stopping: epoch 25
- Best/final checkpoint evaluation: PIT MSE `13.1309`, RMSE `3.62`, Std MSE `60.7853`, frame MAE `2.64`, clip MAE `2.12`, R² `0.7340`
- Best epoch by validation PIT: epoch 15 (`Train=5.4321`, `Val PIT=13.1309`, `R²=0.7340`)
- Minimum train loss: epoch 25 (`Train=3.8700`, `Val PIT=28.4468`)
- Conclusion: best causal-head candidate so far, but still substantially worse than `simple_conv_v2`.

### E15 — Candidate simple_conv_v2_uni_gru96_norm_do02

Status: completed — worse than H14

Hypothesis: H15 — keep H14's hidden size 96 but reduce dropout to `0.2`, testing whether H14 was over-regularized.

Implementation:

- Added `SimpleConvV2UniGRU96NormDO02`, registered as `simple_conv_v2_uni_gru96_norm_do02`.

Smoke test output: `simple_conv_v2_uni_gru96_norm_do02 (2, 4, 94)`.

Job:

- Submitted: 2026-06-17 15:14 BST
- Slurm job id: `12542830`
- Job name: `ar_012233_v2ug96b`
- Initial submit status: `PENDING`; follow-up after ~11 s showed `RUNNING`, so the loop continued to H16.
- Final Slurm status: `COMPLETED` (elapsed `00:09:34`)
- Log: `/gpfs/scratch/acw592/logs/ar_012233_v2ug96b.o12542830`
- Save path: `/gpfs/scratch/acw592/results/autoresearch/20260617-012233-dregon-lm-v4-michaels-simple-conv-v2/simple_conv_v2_uni_gru96_norm_do02`

Command:

```bash
python train_rps_predictor.py --model simple_conv_v2_uni_gru96_norm_do02 --device cuda:0 --data_root /gpfs/scratch/acw592/datasets/DREGON-LM-V4-michaels --save_path /gpfs/scratch/acw592/results/autoresearch/20260617-012233-dregon-lm-v4-michaels-simple-conv-v2/simple_conv_v2_uni_gru96_norm_do02 --epochs 50 --patience 10 --batch_size 32 --lr 1e-3 --weight_decay 1e-4 --loss pit_mse --epoch-progress
```

Results:

- Early stopping: epoch 19
- Best/final checkpoint evaluation: PIT MSE `65.4811`, RMSE `8.09`, Std MSE `108.1616`, frame MAE `4.43`, clip MAE `3.88`, R² `-1.2644`
- Best epoch by validation PIT: epoch 9 (`Train=7.7301`, `Val PIT=65.4811`, `R²=-1.2644`)
- Conclusion: lower dropout than H14 generalized much worse.

### E16 — Candidate simple_conv_v2_uni_gru64_norm_do03

Status: completed — failed/unstable

Hypothesis: H16 — lower the normalized causal GRU hidden size to 64 while keeping dropout `0.3`, testing whether GroupNorm+dropout makes the original small unidirectional head viable.

Implementation:

- Added `SimpleConvV2UniGRU64NormDO03`, registered as `simple_conv_v2_uni_gru64_norm_do03`.

Smoke test output: `simple_conv_v2_uni_gru64_norm_do03 (2, 4, 94)`.

Job:

- Submitted: 2026-06-17 15:15 BST
- Slurm job id: `12542878`
- Job name: `ar_012233_v2ug64d`
- Initial submit status: `PENDING`; follow-up after ~11 s showed `RUNNING`.
- Final Slurm status: `COMPLETED` (elapsed `00:07:36`)
- Log: `/gpfs/scratch/acw592/logs/ar_012233_v2ug64d.o12542878`
- Save path: `/gpfs/scratch/acw592/results/autoresearch/20260617-012233-dregon-lm-v4-michaels-simple-conv-v2/simple_conv_v2_uni_gru64_norm_do03`

Command:

```bash
python train_rps_predictor.py --model simple_conv_v2_uni_gru64_norm_do03 --device cuda:0 --data_root /gpfs/scratch/acw592/datasets/DREGON-LM-V4-michaels --save_path /gpfs/scratch/acw592/results/autoresearch/20260617-012233-dregon-lm-v4-michaels-simple-conv-v2/simple_conv_v2_uni_gru64_norm_do03 --epochs 50 --patience 10 --batch_size 32 --lr 1e-3 --weight_decay 1e-4 --loss pit_mse --epoch-progress
```

Results:

- Early stopping: epoch 16
- Best/final checkpoint evaluation: PIT MSE `95.4777`, RMSE `9.77`, Std MSE `141.9276`, frame MAE `4.96`, clip MAE `4.60`, R² `-3.7860`
- Best checkpoint came from epoch 6, which had finite validation metrics but a `nan` training loss row.
- Conclusion: lower capacity did not fix causal-head instability/generalization.

### E17 — Candidate simple_conv_tcn

Status: completed — overfit/poor generalization

Hypothesis: H17 — benchmark the existing `simple_conv_tcn` under the same DREGON-LM-V4-michaels/50-epoch PIT-MSE protocol. The dilated TCN head may be much simpler than BiGRU while retaining useful temporal context.

Implementation notes:

- No code changes required; existing model key `simple_conv_tcn` is already registered.
- Inspection caveat: `TCNHead` currently uses symmetric `padding=` in `Conv1d`, and the front-end uses default centered STFT, so this benchmark is dilated-conv/simple but not strictly causal.

Smoke test output: `simple_conv_tcn (2, 4, 94)`.

Job:

- Submitted: 2026-06-17
- Slurm job id: `12562530`
- Job name: `ar_012233_tcn`
- Initial submit status: `PENDING`; follow-up after ~11 s showed `RUNNING`.
- Final Slurm status: `COMPLETED` (elapsed `00:19:36`)
- Log: `/gpfs/scratch/acw592/logs/ar_012233_tcn.o12562530`
- Save path: `/gpfs/scratch/acw592/results/autoresearch/20260617-012233-dregon-lm-v4-michaels-simple-conv-v2/simple_conv_tcn`

Command:

```bash
python train_rps_predictor.py --model simple_conv_tcn --device cuda:0 --data_root /gpfs/scratch/acw592/datasets/DREGON-LM-V4-michaels --save_path /gpfs/scratch/acw592/results/autoresearch/20260617-012233-dregon-lm-v4-michaels-simple-conv-v2/simple_conv_tcn --epochs 50 --patience 10 --batch_size 32 --lr 1e-3 --weight_decay 1e-4 --loss pit_mse --epoch-progress
```

Results:

- Early stopping: epoch 32
- Best/final checkpoint evaluation: PIT MSE `24.5623`, RMSE `4.96`, Std MSE `44.8049`, frame MAE `3.35`, clip MAE `2.68`, R² `0.3952`
- Best epoch by validation PIT: epoch 22 (`Train=2.5823`, `Val PIT=24.5623`, `R²=0.3952`)
- Minimum train loss: epoch 31 (`Train=1.7520`, `Val PIT=42.5755`)
- Conclusion: the existing TCN fits train very well but generalizes much worse than `simple_conv_v2`; likely needs the v2 residual/SE/attention encoder/pool and/or stronger normalization/causal padding.

### E18 — Candidate simple_conv_v2_tcn

Status: completed — useful but below baseline

Hypothesis: H18 — put the existing dilated TCN head on the stronger `simple_conv_v2` residual/SE encoder and attention frequency pool. This tests whether H17 failed because of the older SimpleConv encoder/pool rather than because of the TCN head.

Implementation:

- Added `SimpleConvV2TCN`, registered as `simple_conv_v2_tcn`.
- Uses `TCNHead` unchanged, so the temporal head still has symmetric padding.

Smoke test output: `simple_conv_v2_tcn (2, 4, 94)`.

Job:

- Submitted: 2026-06-17
- Slurm job id: `12566518`
- Job name: `ar_012233_v2tcn`
- Initial submit status: `PENDING`; follow-up after ~11 s showed `RUNNING`, so the loop continued to H19.
- Final Slurm status: `COMPLETED` (elapsed `00:18:00`)
- Log: `/gpfs/scratch/acw592/logs/ar_012233_v2tcn.o12566518`
- Save path: `/gpfs/scratch/acw592/results/autoresearch/20260617-012233-dregon-lm-v4-michaels-simple-conv-v2/simple_conv_v2_tcn`

Command:

```bash
python train_rps_predictor.py --model simple_conv_v2_tcn --device cuda:0 --data_root /gpfs/scratch/acw592/datasets/DREGON-LM-V4-michaels --save_path /gpfs/scratch/acw592/results/autoresearch/20260617-012233-dregon-lm-v4-michaels-simple-conv-v2/simple_conv_v2_tcn --epochs 50 --patience 10 --batch_size 32 --lr 1e-3 --weight_decay 1e-4 --loss pit_mse --epoch-progress
```

Results:

- Early stopping: epoch 37
- Best/final checkpoint evaluation: PIT MSE `10.7799`, RMSE `3.28`, Std MSE `58.4627`, frame MAE `2.22`, clip MAE `1.72`, R² `0.7606`
- Best epoch by validation PIT: epoch 27 (`Train=2.3410`, `Val PIT=10.7799`, `R²=0.7606`)
- Minimum train loss: epoch 37 (`Train=1.7195`, `Val PIT=11.2610`)
- Conclusion: v2 encoder/pool fixed much of H17's problem, but the symmetric TCN remains worse than the BiGRU baseline.

### E19 — Candidate simple_conv_v2_causal_tcn

Status: completed — simpler but below best causal GRU

Hypothesis: H19 — replace the symmetric TCN head with a left-padded dilated TCN head, avoiding temporal normalization, to test a simpler head-causal alternative.

Implementation:

- Added `CausalTCNHead` and `SimpleConvV2CausalTCN`, registered as `simple_conv_v2_causal_tcn`.
- Caveat: this is head-only causal; the v2 encoder still uses centered STFT and symmetric 2D conv padding.

Smoke test output: `simple_conv_v2_causal_tcn (2, 4, 94)`.

Job:

- Submitted: 2026-06-17
- Slurm job id: `12566528`
- Job name: `ar_012233_v2ctcn`
- Initial submit status: `PENDING`; follow-up after ~11 s showed `RUNNING`.
- Final Slurm status: `COMPLETED` (elapsed `00:19:55`)
- Log: `/gpfs/scratch/acw592/logs/ar_012233_v2ctcn.o12566528`
- Save path: `/gpfs/scratch/acw592/results/autoresearch/20260617-012233-dregon-lm-v4-michaels-simple-conv-v2/simple_conv_v2_causal_tcn`

Command:

```bash
python train_rps_predictor.py --model simple_conv_v2_causal_tcn --device cuda:0 --data_root /gpfs/scratch/acw592/datasets/DREGON-LM-V4-michaels --save_path /gpfs/scratch/acw592/results/autoresearch/20260617-012233-dregon-lm-v4-michaels-simple-conv-v2/simple_conv_v2_causal_tcn --epochs 50 --patience 10 --batch_size 32 --lr 1e-3 --weight_decay 1e-4 --loss pit_mse --epoch-progress
```

Results:

- Early stopping: epoch 42
- Best/final checkpoint evaluation: PIT MSE `14.1444`, RMSE `3.76`, Std MSE `65.4155`, frame MAE `2.70`, clip MAE `2.19`, R² `0.6025`
- Best epoch by validation PIT: epoch 32 (`Train=2.5846`, `Val PIT=14.1444`, `R²=0.6025`)
- Minimum train loss: epoch 42 (`Train=1.6237`, `Val PIT=15.2405`)
- Conclusion: left-padded TCN head is simple and fits training, but validation is worse than H14 causal GRU (`13.1309`).

### E20 — Candidate smolnet_rps_tcn

Status: completed

Hypothesis: H20 — adapt SMoLnet's compressed real/imag STFT, frequency-dilated Conv2d backbone, and late square Conv2d layers to RPS prediction with mean frequency pooling and TCN head.

Implementation notes:

- Read source: `../drone-audition/drone_audition/models/smolnet.py`.
- SMoLnet early layers are `(kernel, 1)` Conv2d with dilation on the first spatial axis, i.e. frequency-dilated for `(B,C,F,T)` tensors. Late square layers use symmetric time padding, so the reference is not strictly causal.
- Added local RPS adaptation `SMoLnetRPSTCN`, registered as `smolnet_rps_tcn`.
- The pure SMoLnet RPS variants use mean frequency pooling because attention over all 1025 STFT bins OOMed during smoke testing.

Smoke test output: `smolnet_rps_tcn (2, 4, 94)`.

Job:

- Submitted: 2026-06-17
- Slurm job id: `12567513`
- Job name: `ar_012233_smtcn`
- Initial submit status: `PENDING`; follow-up after ~11 s showed `RUNNING`, so the loop continued to H21.
- Final Slurm status: `COMPLETED` (elapsed `00:35:20`)
- Log: `/gpfs/scratch/acw592/logs/ar_012233_smtcn.o12567513`
- Save path: `/gpfs/scratch/acw592/results/autoresearch/20260617-012233-dregon-lm-v4-michaels-simple-conv-v2/smolnet_rps_tcn`

Command:

```bash
python train_rps_predictor.py --model smolnet_rps_tcn --device cuda:0 --data_root /gpfs/scratch/acw592/datasets/DREGON-LM-V4-michaels --save_path /gpfs/scratch/acw592/results/autoresearch/20260617-012233-dregon-lm-v4-michaels-simple-conv-v2/smolnet_rps_tcn --epochs 50 --patience 10 --batch_size 32 --lr 1e-3 --weight_decay 1e-4 --loss pit_mse --epoch-progress
```

Results:

- Early stopping: epoch 31
- Best/final checkpoint evaluation: PIT MSE `17.4362`, RMSE `4.18`, Std MSE `58.4671`, frame MAE `2.95`, clip MAE `2.35`, R² `0.4048`
- Conclusion: SMoLnet-style backbone alone is better than old `simple_conv_tcn` (H17, 24.56) but much worse than the v2 encoder/pool variants.

### E21 — Candidate smolnet_rps_causal_tcn

Status: completed

Hypothesis: H21 — make the SMoLnet-style RPS adaptation more causal-compatible with left-padded late square layers and a left-padded TCN head, omitting temporal normalization.

Implementation:

- Added `SMoLnetRPSCausalTCN`, registered as `smolnet_rps_causal_tcn`.

Smoke test output: `smolnet_rps_causal_tcn (2, 4, 94)`.

Job:

- Submitted: 2026-06-17
- Slurm job id: `12567530`
- Job name: `ar_012233_smctcn`
- Initial submit status: `PENDING`; follow-up after ~11 s showed `RUNNING`, so the loop continued to H22.
- Final Slurm status: `COMPLETED` (elapsed `00:08:01`)
- Log: `/gpfs/scratch/acw592/logs/ar_012233_smctcn.o12567530`
- Save path: `/gpfs/scratch/acw592/results/autoresearch/20260617-012233-dregon-lm-v4-michaels-simple-conv-v2/smolnet_rps_causal_tcn`

Command:

```bash
python train_rps_predictor.py --model smolnet_rps_causal_tcn --device cuda:0 --data_root /gpfs/scratch/acw592/datasets/DREGON-LM-V4-michaels --save_path /gpfs/scratch/acw592/results/autoresearch/20260617-012233-dregon-lm-v4-michaels-simple-conv-v2/smolnet_rps_causal_tcn --epochs 50 --patience 10 --batch_size 32 --lr 1e-3 --weight_decay 1e-4 --loss pit_mse --epoch-progress
```

Results:

- Early stopping: epoch 16
- Best/final checkpoint evaluation: PIT MSE `49.6064`, RMSE `7.04`, Std MSE `79.4002`, frame MAE `5.97`, clip MAE `4.86`, R² `-0.3305`
- Conclusion: making the SMoLnet backbone causal worsens it substantially (vs. E20 17.44). The causal penalty hit SMoLnet harder than the v2 encoder.

### E22 — Candidate simple_conv_v2_smol_tcn

Status: completed

Hypothesis: H22 — combine the strong v2 encoder/pool with a shallow SMoLnet-style frequency-dilated refinement before the symmetric TCN head.

Implementation:

- Added `SimpleConvV2SMoLTCN`, registered as `simple_conv_v2_smol_tcn`.

Smoke test output: `simple_conv_v2_smol_tcn (2, 4, 94)`.

Job:

- Submitted: 2026-06-17
- Slurm job id: `12567556`
- Job name: `ar_012233_v2smt`
- Initial submit status: `PENDING`; follow-up after ~11 s still `PENDING` with reason `QOSMaxGRESPerUser`, so no further jobs were submitted.
- Final Slurm status: `COMPLETED` (elapsed `00:15:31`)
- Log: `/gpfs/scratch/acw592/logs/ar_012233_v2smt.o12567556`
- Save path: `/gpfs/scratch/acw592/results/autoresearch/20260617-012233-dregon-lm-v4-michaels-simple-conv-v2/simple_conv_v2_smol_tcn`

Command:

```bash
python train_rps_predictor.py --model simple_conv_v2_smol_tcn --device cuda:0 --data_root /gpfs/scratch/acw592/datasets/DREGON-LM-V4-michaels --save_path /gpfs/scratch/acw592/results/autoresearch/20260617-012233-dregon-lm-v4-michaels-simple-conv-v2/simple_conv_v2_smol_tcn --epochs 50 --patience 10 --batch_size 32 --lr 1e-3 --weight_decay 1e-4 --loss pit_mse --epoch-progress
```

Results:

- Early stopping: epoch 31
- Best/final checkpoint evaluation: PIT MSE `9.0751`, RMSE `3.01`, Std MSE `14.6140`, frame MAE `1.89`, clip MAE `1.31`, R² `0.8318`
- **Best R² (0.8318) of any model so far**, beating the baseline (0.8183) and H7 gru96 (0.8216). PIT MSE (9.0751) is slightly worse than baseline (7.8920).
- Conclusion: the v2 encoder + SMoLnet frequency-dilated refinement is the best combination found so far for R²; the main gap to baseline is PIT MSE, not explained variance.

### E23 — Candidate simple_conv_v2_smol_causal_tcn

Status: completed

Hypothesis: H23 — combine v2 encoder with SMoLnet-style refinement and a left-padded TCN head.

Implementation:

- Added `SimpleConvV2SMoLCausalTCN`, registered as `simple_conv_v2_smol_causal_tcn`.
- Uses causal-time SMoLnet refinement and left-padded TCN head.
- Caveat: still uses centered STFT and symmetric 2D conv encoder; only the refinement and head are causal.

Smoke test output: `simple_conv_v2_smol_causal_tcn (2, 4, 94)`.

Job:

- Submitted: 2026-06-17
- Slurm job id: `12568662`
- Job name: `ar_012233_v2smct`
- Initial submit status: `PENDING`; follow-up after ~12 s showed `RUNNING`.
- Final Slurm status: `COMPLETED` (elapsed `00:24:04`)
- Log: `/gpfs/scratch/acw592/logs/ar_012233_v2smct.o12568662`
- Save path: `/gpfs/scratch/acw592/results/autoresearch/20260617-012233-dregon-lm-v4-michaels-simple-conv-v2/simple_conv_v2_smol_causal_tcn`

Command:

```bash
python train_rps_predictor.py --model simple_conv_v2_smol_causal_tcn --device cuda:0 --data_root /gpfs/scratch/acw592/datasets/DREGON-LM-V4-michaels --save_path /gpfs/scratch/acw592/results/autoresearch/20260617-012233-dregon-lm-v4-michaels-simple-conv-v2/simple_conv_v2_smol_causal_tcn --epochs 50 --patience 10 --batch_size 32 --lr 1e-3 --weight_decay 1e-4 --loss pit_mse --epoch-progress
```

Results:

- Ran all 50 epochs (no early stopping, patience not triggered)
- Best/final checkpoint evaluation: PIT MSE `8.3806`, RMSE `2.89`, Std MSE `105.0128`, frame MAE `1.93`, clip MAE `1.46`, R² `0.8331`
- **New best R² (0.8331)**, beating E22 (0.8318) and baseline (0.8183). PIT MSE (8.3806) close to baseline (7.8920).
- Conclusion: causal constraint in SMoL refinement + head acts as effective regularizer, improving both metrics vs. symmetric E22. Best overall model on the leaderboard by combined PIT MSE + R².

### E24 — Candidate smolnet_rps_simple_head

### E24 — Candidate smolnet_rps_simple_head

Status: completed

Hypothesis: H24 — use the SMoLnet-style frequency-dilated body with the simplest SimpleConv-style mean-frequency-pool + shallow Conv1d head.

Rationale:

- This is the missing clean ablation: it changes the body while preserving the simplest temporal head pattern, rather than introducing a TCN at the same time.
- Expected to show whether the SMoLnet body is useful before attributing effects to temporal-head changes.

Implementation:

- Added `SMoLnetRPSSimpleHead`, registered as `smolnet_rps_simple_head` in both registries.
- Uses the existing `SMoLnetRPSBackbone` and then `mean(dim=2)` frequency pooling followed by `Conv1d(16→64, k=5)`, ReLU, dropout `0.1`, and `Conv1d(64→4, k=1)`, matching the SimpleConv head pattern.

Smoke test output: `smolnet_rps_simple_head (2, 4, 94)`.

Job:

- Submitted: 2026-06-17
- Slurm job id: `12568116`
- Job name: `ar_012233_smsimp`
- Initial submit status: `PENDING`; follow-up after ~12 s remained `PENDING` with reason `Resources`, so no further jobs submitted.
- Final Slurm status: `COMPLETED` (elapsed `00:18:00`)
- Log: `/gpfs/scratch/acw592/logs/ar_012233_smsimp.o12568116`
- Save path: `/gpfs/scratch/acw592/results/autoresearch/20260617-012233-dregon-lm-v4-michaels-simple-conv-v2/smolnet_rps_simple_head`

Command:

```bash
python train_rps_predictor.py --model smolnet_rps_simple_head --device cuda:0 --data_root /gpfs/scratch/acw592/datasets/DREGON-LM-V4-michaels --save_path /gpfs/scratch/acw592/results/autoresearch/20260617-012233-dregon-lm-v4-michaels-simple-conv-v2/smolnet_rps_simple_head --epochs 50 --patience 10 --batch_size 32 --lr 1e-3 --weight_decay 1e-4 --loss pit_mse --epoch-progress
```

Results:

- Early stopping: epoch 15
- Best/final checkpoint evaluation: PIT MSE `141.0523`, RMSE `11.88`, Std MSE `171.6778`, frame MAE `10.03`, clip MAE `9.58`, R² `-3.8919`
- Conclusion: the SMoLnet body with a simple Conv1d head fails dramatically. The SMoLnet frequency-dilated backbone needs the TCN head's strong temporal receptive field.

### E25 — Candidate simple_conv_v2_smol_bigru

Status: completed

Hypothesis: H25 — combine the winning v2+SMoL encoder (E22, best R²) with the winning BiGRU temporal head (baseline, best PIT MSE).

Implementation:

- Added `SimpleConvV2SMoLBiGRU`, registered as `simple_conv_v2_smol_bigru`.
- Extends `SimpleConvV2` (not `SimpleConvV2TCN`) and inserts `SMoLnetRPSBackbone` (non-causal) between the v2 encoder and attention frequency pooling.
- Uses `BiGRUHead(128, hidden_ch=64, num_layers=2)` identical to the baseline.

Smoke test output: `simple_conv_v2_smol_bigru (2, 4, 94)`.

Job:

- Submitted: 2026-06-17
- Slurm job id: `12568655`
- Job name: `ar_012233_v2smbg`
- Initial submit status: `PENDING`; follow-up after ~12 s showed `RUNNING`.
- Final Slurm status: `COMPLETED` (elapsed `00:16:03`)
- Log: `/gpfs/scratch/acw592/logs/ar_012233_v2smbg.o12568655`
- Save path: `/gpfs/scratch/acw592/results/autoresearch/20260617-012233-dregon-lm-v4-michaels-simple-conv-v2/simple_conv_v2_smol_bigru`

Command:

```bash
python train_rps_predictor.py --model simple_conv_v2_smol_bigru --device cuda:0 --data_root /gpfs/scratch/acw592/datasets/DREGON-LM-V4-michaels --save_path /gpfs/scratch/acw592/results/autoresearch/20260617-012233-dregon-lm-v4-michaels-simple-conv-v2/simple_conv_v2_smol_bigru --epochs 50 --patience 10 --batch_size 32 --lr 1e-3 --weight_decay 1e-4 --loss pit_mse --epoch-progress
```

Results:

- Early stopping: epoch 32
- Best/final checkpoint evaluation: PIT MSE `11.3410`, RMSE `3.37`, Std MSE `27.4749`, frame MAE `2.34`, clip MAE `1.73`, R² `0.7461`
- Conclusion: combining v2+SMoL encoder with BiGRU head performs worse than either alone. The compound architecture (v2 encoder → SMoL refinement → BiGRU) likely introduces optimization difficulties or gradient conflicts within the fixed training budget.

## Online-mixing rerun — 200 epoch / patience 50 / 5000 samples per validation

Status: 25 completed, 1 timed out on gpushort.

Purpose: rerun the same 26 model keys under online mixture generation rather than fixed offline train samples, while keeping the original optimizer/model parameters (`--batch_size 32 --lr 1e-3 --weight_decay 1e-4 --loss pit_mse --epoch-progress`).

Setup:

- Train stream: `configs/online_mix_v4_michaels_train_no_room1_gpfs.yaml`
- Noise sources: DREGON `in_flight_noise` from `/gpfs/scratch/acw592/data/DREGON`, excluding `free-flight_nosource_room1`; plus Michael's `FLY125` from `/gpfs/scratch/acw592/data`.
- Speech source: LibriSpeech `train-clean-100` under `/gpfs/scratch/acw592/data/librispeech/LibriSpeech/train-clean-100-readable` (symlink tree excluding one unreadable FLAC: `669/129061/669-129061-0001.flac`, which raised `flac decoder lost sync`).
- Source cache: `/gpfs/scratch/acw592/cache/online_mix_sources`; created once by `simple_conv_tcn` before submitting the rest of the array.
- SNR: uniform `[-30, 0]` dB; `speech_per_channel: independent`; `snr_per_channel: false`.
- Augmentations: enabled after 50000 global samples with probability `0.5`, choosing one of random gain, polarity flip, or single-channel drop.
- Fixed validation set: `/gpfs/scratch/acw592/datasets/DREGON-LM-V4-michaels/valid`.
- Results root: `/gpfs/scratch/acw592/results/autoresearch/20260618-v4-michaels-online-mix-200ep-aug50k-gpushort`.
- Runner script: `/gpfs/scratch/acw592/run_online_mix_v4_michaels_gpushort_array.sh`.

Submission notes:

- A first concurrent cache attempt was cancelled because one LibriSpeech file was unreadable and because concurrent cache creation made diagnosis noisy.
- Final sequence: delete cache, run `simple_conv_tcn` alone until cache creation completed and training started, then submit the rest as array `12575857_[0-16,18-25]`.
- `simple_conv_tcn` single job: `12575848`, log `/gpfs/scratch/acw592/logs/om0618_one_tcn.o12575848`, completed in `00:17:08`.
- Rest array: `12575857`, logs `/gpfs/scratch/acw592/logs/om0618_gs_rest.o*`.

Results table (best/final checkpoint when completed; best observed validation when timed out):

| Task | Model | Slurm | Status | Epochs logged | Early stop | PIT MSE | RMSE | MAE/f | MAE/c | R² | Best epoch | Log |
|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | `simple_conv_v2` | 12575857_0 | completed | 92 | 92 | 8.5349 | 2.92 | 2.00 | 1.56 | 0.8332 | 42 | `/gpfs/scratch/acw592/logs/om0618_gs_rest.o12575858` |
| 1 | `simple_conv_v2_transformer` | 12575857_1 | completed | 65 | 65 | 8.4629 | 2.91 | 2.16 | 1.68 | 0.8085 | 15 | `/gpfs/scratch/acw592/logs/om0618_gs_rest.o12575874` |
| 2 | `simple_conv_v2_local_attn` | 12575857_2 | completed | 67 | 67 | 10.0549 | 3.17 | 2.37 | 1.86 | 0.7637 | 17 | `/gpfs/scratch/acw592/logs/om0618_gs_rest.o12575889` |
| 3 | `simple_conv_v2_multires` | 12575857_3 | completed | 107 | 107 | 8.7521 | 2.96 | 2.27 | 1.87 | 0.7710 | 57 | `/gpfs/scratch/acw592/logs/om0618_gs_rest.o12575901` |
| 4 | `simple_conv_v2_dwt` | 12575857_4 | completed | 63 | 63 | 8.8512 | 2.98 | 2.32 | 1.87 | 0.7828 | 13 | `/gpfs/scratch/acw592/logs/om0618_gs_rest.o12575918` |
| 5 | `simple_conv_v2_magphase` | 12575857_5 | completed | 87 | 87 | 8.1824 | 2.86 | 2.06 | 1.54 | 0.8348 | 37 | `/gpfs/scratch/acw592/logs/om0618_gs_rest.o12575940` |
| 6 | `simple_conv_v2_dual_pool` | 12575857_6 | completed | 66 | 66 | 8.4940 | 2.91 | 2.29 | 1.80 | 0.7888 | 16 | `/gpfs/scratch/acw592/logs/om0618_gs_rest.o12575941` |
| 7 | `simple_conv_v2_gru96` | 12575857_7 | completed | 56 | 56 | 10.7942 | 3.29 | 2.51 | 2.12 | 0.7886 | 6 | `/gpfs/scratch/acw592/logs/om0618_gs_rest.o12575979` |
| 8 | `simple_conv_v2_uni_gru` | 12575857_8 | completed | 102 | 102 | 8.7301 | 2.95 | 2.28 | 1.88 | 0.8030 | 52 | `/gpfs/scratch/acw592/logs/om0618_gs_rest.o12575994` |
| 9 | `simple_conv_v2_causal_gru` | 12575857_9 | completed | 75 | 75 | 14.6395 | 3.83 | 2.48 | 2.04 | 0.7703 | 25 | `/gpfs/scratch/acw592/logs/om0618_gs_rest.o12576022` |
| 10 | `simple_conv_v2_causal_gru96` | 12575857_10 | completed | 65 | 65 | 11.4611 | 3.39 | 2.50 | 2.03 | 0.7657 | 15 | `/gpfs/scratch/acw592/logs/om0618_gs_rest.o12576086` |
| 11 | `simple_conv_v2_uni_gru128` | 12575857_11 | completed | 67 | 67 | **7.3264** | 2.71 | 2.04 | 1.55 | 0.8224 | 17 | `/gpfs/scratch/acw592/logs/om0618_gs_rest.o12576104` |
| 12 | `simple_conv_v2_uni_gru128_norm` | 12575857_12 | completed | 75 | 75 | 7.9864 | 2.83 | 2.11 | 1.62 | 0.8024 | 25 | `/gpfs/scratch/acw592/logs/om0618_gs_rest.o12576193` |
| 13 | `simple_conv_v2_uni_gru128_norm_do03` | 12575857_13 | completed | 67 | 67 | 8.2826 | 2.88 | 2.15 | 1.69 | 0.8057 | 17 | `/gpfs/scratch/acw592/logs/om0618_gs_rest.o12576224` |
| 14 | `simple_conv_v2_uni_gru96_norm_do03` | 12575857_14 | completed | 75 | 75 | 8.3325 | 2.89 | 2.26 | 1.87 | 0.8059 | 25 | `/gpfs/scratch/acw592/logs/om0618_gs_rest.o12576255` |
| 15 | `simple_conv_v2_uni_gru96_norm_do02` | 12575857_15 | completed | 71 | 71 | 9.3224 | 3.05 | 2.31 | 1.88 | 0.7946 | 21 | `/gpfs/scratch/acw592/logs/om0618_gs_rest.o12576262` |
| 16 | `simple_conv_v2_uni_gru64_norm_do03` | 12575857_16 | completed | 60 | 60 | 88.8055 | 9.42 | 7.60 | 7.27 | -1.8765 | 10 | `/gpfs/scratch/acw592/logs/om0618_gs_rest.o12576299` |
| 17 | `simple_conv_tcn` | 12575848 | completed | 65 | 65 | 14.1221 | 3.76 | 2.77 | 2.17 | 0.6832 | 15 | `/gpfs/scratch/acw592/logs/om0618_one_tcn.o12575848` |
| 18 | `simple_conv_v2_tcn` | 12575857_18 | completed | 71 | 71 | 12.4689 | 3.53 | 2.80 | 2.34 | 0.6924 | 21 | `/gpfs/scratch/acw592/logs/om0618_gs_rest.o12576301` |
| 19 | `simple_conv_v2_causal_tcn` | 12575857_19 | completed | 76 | 76 | 11.3739 | 3.37 | 2.35 | 1.73 | 0.7458 | 26 | `/gpfs/scratch/acw592/logs/om0618_gs_rest.o12576334` |
| 20 | `smolnet_rps_tcn` | 12575857_20 | timed out | 96 | — | 11.8746* | — | 2.60* | 1.99* | 0.7270* | 48 | `/gpfs/scratch/acw592/logs/om0618_gs_rest.o12576345` |
| 21 | `smolnet_rps_causal_tcn` | 12575857_21 | completed | 100 | 100 | 12.9773 | 3.60 | 2.59 | 1.94 | 0.6755 | 50 | `/gpfs/scratch/acw592/logs/om0618_gs_rest.o12576372` |
| 22 | `simple_conv_v2_smol_tcn` | 12575857_22 | completed | 88 | 88 | 12.6057 | 3.55 | 2.49 | 1.83 | 0.6863 | 38 | `/gpfs/scratch/acw592/logs/om0618_gs_rest.o12576414` |
| 23 | `simple_conv_v2_smol_causal_tcn` | 12575857_23 | completed | 113 | 113 | 8.9874 | 3.00 | 2.02 | 1.56 | 0.8237 | 63 | `/gpfs/scratch/acw592/logs/om0618_gs_rest.o12576468` |
| 24 | `smolnet_rps_simple_head` | 12575857_24 | completed | 93 | 93 | 13.7097 | 3.70 | 2.51 | 1.92 | 0.6900 | 43 | `/gpfs/scratch/acw592/logs/om0618_gs_rest.o12576469` |
| 25 | `simple_conv_v2_smol_bigru` | 12575857_25 | completed | 60 | 60 | 9.5475 | 3.09 | 2.38 | 2.01 | 0.8052 | 10 | `/gpfs/scratch/acw592/logs/om0618_gs_rest.o12575857` |

\* `smolnet_rps_tcn` timed out before final checkpoint evaluation. Metrics marked with `*` are the best logged validation row, not a loaded-best final evaluation.

Timeout diagnosis:

- Only timed-out job: `smolnet_rps_tcn` (`12575857_20`, batch log `/gpfs/scratch/acw592/logs/om0618_gs_rest.o12576345`).
- Slurm status: `TIMEOUT`, elapsed `01:00:12`, time limit `01:00:00`.
- It reused the finalized cache and trained normally; no traceback/error before timeout.
- Best observed validation: epoch 48, `Val PIT=11.8746`, train `12.1482`, frame MAE `2.60`, clip MAE `1.99`, R² `0.7270`.
- At epoch 96, immediately before timeout, it had worsened to `Val PIT=13.4076`, train `9.9089`, frame MAE `2.79`, clip MAE `2.21`, R² `0.6784`; then Slurm cancelled during epoch 97.
- Interpretation: it was not on track to beat the best online-mix models. Because patience is 50, best epoch 48 would early-stop at epoch 98 if no improvement; the 1h cutoff likely occurred just before natural early stopping/final evaluation.

Online-mix conclusions:

- Best completed PIT MSE: `simple_conv_v2_uni_gru128` with PIT MSE `7.3264`, beating the offline fixed-loader baseline (`7.8920`) on the same validation set.
- Best completed R²: `simple_conv_v2_magphase` with R² `0.8348`; `simple_conv_v2` online also reached R² `0.8332`.
- Online mixing substantially changed the ranking: previously failed unidirectional GRU variants became competitive, while `simple_conv_v2_smol_causal_tcn` remained strong but no longer best (`8.9874`, R² `0.8237`).

## Validation prediction export

Training/evaluation logs do not save raw validation predictions; they only keep checkpoints, W&B IDs, and printed aggregate metrics. A separate export job was submitted to evaluate all 52 best checkpoints from both series (26 original fixed/offline checkpoints + 26 online-mix checkpoints) on `/gpfs/scratch/acw592/datasets/DREGON-LM-V4-michaels/valid`.

Job:

- Slurm job id: `12642388`
- Job name: `om0618_eval_preds`
- Partition: `gpushort`
- Script: `autoresearch/20260617-012233-dregon-lm-v4-michaels-simple-conv-v2/save_validation_predictions.py`
- Initial output root: `/gpfs/scratch/acw592/results/autoresearch/20260617-012233-dregon-lm-v4-michaels-simple-conv-v2/validation_rps_predictions`
- Final layout: each model checkpoint folder now contains its own `validation_rps_predictions/` subdirectory.
- Move manifest: `/gpfs/scratch/acw592/results/autoresearch/20260617-012233-dregon-lm-v4-michaels-simple-conv-v2/validation_rps_predictions/moved_manifest.json`

Per model checkpoint folder output files:

- `validation_rps_predictions/pred_raw.npy` — raw model output before PIT rotor-order matching, shape `(rows, 4, frames)`.
- `validation_rps_predictions/target.npy` — validation target on the prediction frame grid.
- `validation_rps_predictions/target_pit_matched_to_pred.npy` — target reordered by PIT-optimal rotor assignment for metric/plot overlays.
- `validation_rps_predictions/sample_ids.npy`, `validation_rps_predictions/channels.npy` — row metadata.
- `validation_rps_predictions/metadata.json` — checkpoint path, shapes, final prediction folder, and quick metrics computed from saved arrays.

Series roots containing per-model `validation_rps_predictions/` folders:

- Offline fixed-train checkpoints: `/gpfs/scratch/acw592/results/autoresearch/20260617-012233-dregon-lm-v4-michaels-simple-conv-v2/<model>/validation_rps_predictions/`
- Online-mix checkpoints: `/gpfs/scratch/acw592/results/autoresearch/20260618-v4-michaels-online-mix-200ep-aug50k-gpushort/<model>/validation_rps_predictions/`
