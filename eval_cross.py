#!/usr/bin/env python3
"""Cross-evaluation: old vs V3 checkpoints on OLD and V2 validation sets,
plus sample-level inference and in-flight recording inference."""

import json, os, random, torch, numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, Subset

from train_rps_predictor import (
    evaluate, get_model, DREGONRPSDataset, pit_mse_loss, pairwise_mse,
)
from data_processing.dregon import (
    get_geometry,
    clean_command_spikes,
)

device = 'cuda:0'
OUT = Path('results/rps_cross_eval')
OUT.mkdir(parents=True, exist_ok=True)
(OUT / 'samples').mkdir(exist_ok=True)

# ── Models ──────────────────────────────────────────────────────────────────

MODELS = [
    ('old_simple_conv', 'simple_conv', 'results/rps_exp_simple_conv/best_simple_conv.pt'),
    ('old_bigru_v2', 'simple_conv_bigru_v2', 'results/rps_exp_bigru_v2/best_simple_conv_bigru_v2.pt'),
    ('v3_simple_conv', 'simple_conv', 'results/rps_predictor_v3/simple_conv/best_simple_conv.pt'),
    ('v3_bigru_v2', 'simple_conv_bigru_v2', 'results/rps_predictor_v3/simple_conv_bigru_v2/best_simple_conv_bigru_v2.pt'),
]

loaded_models = {}
for name, mtype, ckpt in MODELS:
    model = get_model(mtype).to(device)
    model.load_state_dict(torch.load(ckpt, weights_only=True))
    model.eval()
    loaded_models[name] = model
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded {name}: {n_params:,} params")

# ── Part 1: Full validation set metrics ─────────────────────────────────────

old_ds = DREGONRPSDataset('datasets/DREGON-LM/valid')
new_ds = DREGONRPSDataset('datasets/DREGON-LM-V2/valid')

all_metrics = {}
for ds_name, ds in [('OLD_valid', old_ds), ('V2_valid', new_ds)]:
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=4)
    for model_name, model in loaded_models.items():
        m = evaluate(model, loader, device, len(ds), pit_eval=True)
        key = f'{model_name}__{ds_name}'
        all_metrics[key] = {
            'pit_mse': float(m['mse']),
            'std_mse': float(m['std_mse']),
            'mae_frame': float(m['mae_frame']),
            'mae_clip': float(m['mae_clip']),
        }
        print(f"  {key}: PIT_MSE={m['mse']:.2f}")

with open(OUT / 'validation_metrics.json', 'w') as f:
    json.dump(all_metrics, f, indent=2)
print(f"Saved validation metrics to {OUT / 'validation_metrics.json'}")

# ── Part 2: Sample-level inference (5 random from each dataset) ─────────────

random.seed(42)
old_indices = random.sample(range(len(old_ds)), 5)
new_indices = random.sample(range(len(new_ds)), 5)

import soundfile as sf

for tag, ds, indices in [('old', old_ds, old_indices), ('v2', new_ds, new_indices)]:
    for idx in indices:
        sample_dir = ds.samples[idx]
        sample_id = os.path.basename(sample_dir)
        out_dir = OUT / 'samples' / f'{tag}_{sample_id}'
        out_dir.mkdir(exist_ok=True)

        audio, rps_target = ds[idx]
        audio = audio.unsqueeze(0).to(device)
        rps_target = rps_target.numpy()

        # Save ground truth
        np.save(out_dir / 'rps_target.npy', rps_target)
        sf.write(out_dir / 'mixture.wav', audio.cpu().squeeze().numpy(), 16000)
        sf.write(out_dir / 'vocals.wav',
                 sf.read(os.path.join(sample_dir, 'vocals.wav'))[0], 16000)
        sf.write(out_dir / 'noise.wav',
                 sf.read(os.path.join(sample_dir, 'noise.wav'))[0], 16000)

        # Inference from all models
        for model_name, model in loaded_models.items():
            with torch.no_grad():
                with torch.amp.autocast('cuda'):
                    pred = model(audio).cpu().squeeze(0)  # (4, T)
            np.save(out_dir / f'rps_pred_{model_name}.npy', pred.numpy())

            # Compute PIT MSE for this sample
            pw = pairwise_mse(pred.unsqueeze(0),
                            torch.from_numpy(rps_target).unsqueeze(0))
            best = float('inf')
            import itertools
            for perm in itertools.permutations(range(4)):
                loss = sum(pw[0, j, perm[j]].item() for j in range(4))
                if loss < best:
                    best = loss
            pit_mse_sample = best / 4.0  # normalize by n_rotors

            # Save per-sample metrics JSON
            metrics_sample = {
                'sample_id': sample_id,
                'dataset': tag,
                'model': model_name,
                'pit_mse': pit_mse_sample,
            }
            with open(out_dir / f'metrics_{model_name}.json', 'w') as f:
                json.dump(metrics_sample, f, indent=2)

print(f"Saved sample inferences to {OUT / 'samples'}/")

# ── Part 3: In-flight recording inference ───────────────────────────────────

# Load speech-high and whitenoise-high recordings
from data_processing.dregon import load_timeframe, discover_recordings
from utils.data import TimeFrame

dregon_dir = Path('data/DREGON')
geometry = get_geometry(dregon_dir)

# Discover recordings in the in_flight_source split
all_samples = discover_recordings(dregon_dir)
target_ids = {'free-flight_speech-high_room1', 'free-flight_whitenoise-high_room1'}

for sample in all_samples:
    if sample['recording_id'] not in target_ids:
        continue

    rid = sample['recording_id']
    print(f"\nProcessing in-flight recording: {rid}")

    tf = load_timeframe(sample, geometry=geometry, target_sr=16000)

    audio_us = tf["audio"]
    # UniformSeries stores (channels, N) — axis 0 = channels
    n_channels = audio_us.samples.shape[0] if audio_us.samples.ndim > 1 else 1
    total_duration = audio_us.duration
    print(f"  Duration: {total_duration:.1f}s, channels: {n_channels}")

    # Use channel 0 for simplicity
    ch = 0
    audio_full = audio_us.samples[ch, :] if audio_us.samples.ndim > 1 else audio_us.samples
    audio_full = torch.from_numpy(audio_full.astype(np.float32))

    # Extract command RPS (cleaned) for ground truth
    motor_key = "motors_command" if "motors_command" in tf else "motors_measured"
    if motor_key in tf:
        command = tf[motor_key].values.copy()
        command_cleaned = clean_command_spikes(command)
        rps_full = torch.from_numpy(command_cleaned.T.astype(np.float32))  # (4, T_motor)
    else:
        rps_full = None

    # Process in 3-second windows with 1.5s overlap
    window_samples = 3 * 16000  # 48000
    hop_samples = int(1.5 * 16000)  # 24000
    n_fft, hop = 2048, 512

    all_preds = {name: [] for name in loaded_models}
    all_targets_rps = []
    window_starts = []

    for start_sample in range(0, len(audio_full) - window_samples + 1, hop_samples):
        chunk = audio_full[start_sample:start_sample + window_samples].to(device)
        chunk = chunk.unsqueeze(0)  # (1, samples)

        # RPS target for this window
        if rps_full is not None:
            n_frames = chunk.shape[1] // hop + 1
            # Resample RPS to match
            rps_chunk = torch.nn.functional.interpolate(
                rps_full.unsqueeze(0),
                size=n_frames, mode='linear', align_corners=False
            ).squeeze(0)
            all_targets_rps.append(rps_chunk.numpy())

        for model_name, model in loaded_models.items():
            with torch.no_grad():
                with torch.amp.autocast('cuda'):
                    pred = model(chunk).cpu().squeeze(0)
            all_preds[model_name].append(pred.numpy())

        window_starts.append(start_sample / 16000.0)

    # Save concatenated predictions
    out_dir = OUT / 'inflight' / rid
    out_dir.mkdir(parents=True, exist_ok=True)

    for model_name in loaded_models:
        preds_stacked = np.stack(all_preds[model_name], axis=0)  # (n_windows, 4, T)
        np.save(out_dir / f'rps_pred_{model_name}.npy', preds_stacked)

    if rps_full is not None and len(all_targets_rps) > 0:
        targets_stacked = np.stack(all_targets_rps, axis=0)
        np.save(out_dir / 'rps_target.npy', targets_stacked)

    # Save metadata
    meta_inflight = {
        'recording_id': rid,
        'duration': float(total_duration),
        'channel': ch,
        'window_duration': 3.0,
        'hop_duration': 1.5,
        'n_windows': len(window_starts),
        'window_starts': window_starts,
    }
    with open(out_dir / 'metadata.json', 'w') as f:
        json.dump(meta_inflight, f, indent=2)

    # Per-window PIT MSE metrics
    if rps_full is not None and len(all_targets_rps) > 0:
        inflight_metrics = {}
        for model_name in loaded_models:
            preds = np.stack(all_preds[model_name], axis=0)
            targs = np.stack(all_targets_rps, axis=0)

            pit_mses = []
            for w in range(len(preds)):
                p = torch.from_numpy(preds[w]).unsqueeze(0)
                t = torch.from_numpy(targs[w]).unsqueeze(0)
                loss = pit_mse_loss(p, t).item()
                pit_mses.append(loss)

            inflight_metrics[model_name] = {
                'mean_pit_mse': float(np.mean(pit_mses)),
                'median_pit_mse': float(np.median(pit_mses)),
                'min_pit_mse': float(np.min(pit_mses)),
                'max_pit_mse': float(np.max(pit_mses)),
            }

        with open(out_dir / 'window_metrics.json', 'w') as f:
            json.dump(inflight_metrics, f, indent=2)
        print(f"  Inflight metrics: {inflight_metrics}")

print(f"\nSaved in-flight results to {OUT / 'inflight'}/")
print("\nDone! All results in results/rps_cross_eval/")
