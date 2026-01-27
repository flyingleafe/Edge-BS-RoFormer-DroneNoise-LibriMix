# Debugging Diffusion Buffer Training (Remote Machine)

This note is specific to the `diffusion_buffer` model (`models/diffusion_buffer.py`) and the DN‑LM dataset in this repo. The goal is to validate the data → STFT → diffusion loss path with minimal compute, then scale up.

## Golden Rules

- Start with a tiny run and keep it deterministic.
- The diffusion loss is computed in the model `forward(mix, target)` (not in `train.py`).
- Most failures are shape mismatches in STFT frames or `buffer_size` > frames.

## 1) Minimal Sanity Run (Diffusion Buffer)

Use a minimal run to verify the full pipeline.

- Set `num_workers=0`.
- Use a tiny batch size.
- Disable AMP (`use_amp: false`) if you see `nan` or `inf`.

Example:

```bash
python train.py \
  --model_type diffusion_buffer \
  --config_path configs/9_Diffusion_Buffer_BBED.yaml \
  --results_path results/debug_diffusion \
  --data_path datasets/DN-LM/train \
  --valid_path datasets/DN-LM/valid \
  --dataset_type 1 \
  --device_ids 0 \
  --num_workers 0 \
  --metrics si_sdr sdr \
  --metric_for_scheduler si_sdr
```

## 2) Dataset and I/O Checks (DN‑LM)

Expected `__getitem__` output from `dataset.py`:

- `mix`: `[channels, time]`
- `target`: `[channels, time]`

For DN‑LM, `audio.chunk_size` should be 16000 (1 second at 16 kHz). If chunk size is larger than the file length, the loader pads with zeros.

Quick print in `dataset.py` (first batch only) if shapes look off.

## 3) STFT Shape Expectations (Diffusion Model)

In `models/diffusion_buffer.py`:

- STFT input is mono: `mix` and `target` are averaged to 1 channel.
- STFT shape is `[batch, freq, frames]` (complex tensor).
- The diffusion buffer size must be `<= frames`.

Fast check: log `frames` and `buffer_size` in `_forward_train()`:

- If `buffer_size > frames`, training will fail.
- With 1‑second clips and `n_fft=510, hop=256`, frames are roughly 61.

## 4) Loss and NaNs

Training loss is MSE on the score target:

- Target is `-z / sigma`, where `sigma` is from the BBED schedule.
- NaNs usually mean `sigma` hit zero or AMP overflowed.

If NaNs appear:

- Set `training.use_amp: false`.
- Log min/max of `sigma` and `score`.
- Verify `t_eps` and `t_max` are sane (`t_eps > 0`, `t_max <= 1`).

## 5) Buffer and Time Schedule

Key params in `configs/9_Diffusion_Buffer_BBED.yaml`:

- `diffusion_buffer.buffer_size` (B)
- `diffusion_buffer.t_eps`
- `diffusion_buffer.t_max`

If you change buffer size:

- Ensure it does not exceed STFT frames for 1‑second clips.
- Keep `t_eps > 0` to avoid division by zero.

## 6) Common Failure Modes (Diffusion‑Specific)

- **`ValueError: Buffer size must be at least 1.`**  
  The STFT returned zero frames. Check `audio.chunk_size`, `n_fft`, `hop_length`.

- **CUDA OOM**  
  Reduce `training.batch_size`, or lower `model.base_channels`.

- **NaNs or infs**  
  Disable AMP; check `sigma` range; log `score` stats.

- **Mismatch in shapes**  
  Confirm `mix` and `target` are `[B, C, T]` and mono conversion works.

## 7) Scale Back Up

Once the minimal run works:

1. Increase `training.batch_size`.
2. Re‑enable AMP if stable.
3. Increase `num_workers`.
4. Restore full epochs.

## 8) When You Need a New Log

Use `PYTHONFAULTHANDLER=1` for cleaner stack traces:

```bash
PYTHONFAULTHANDLER=1 python train.py ...
```
