"""Phase-A VK-parity eval: test-time smoothing of neural RPS predictors.

Campaign criterion 2.3 asks whether an audio-only neural RPS predictor can
reach parity with the best blind VK tracker on the SAME evaluation clips.
This script evaluates EXISTING checkpoints (no training) with test-time
temporal aggregation, on exactly the protocol of
``results/vk_eval/vk_valid_comparison.csv``:

* Clips: the 37 samples of ``DREGON-LM-V4-michaels-valid-full`` (3 DREGON
  free-flight room1 recordings + Michael's FLY124), 8 s each, contiguous
  slices of the source recordings starting at t=0 (verified: each clip's
  ``mixture.wav`` equals the soxr_hq-16k-resampled recording slice
  ``[8k, 8(k+1)] s`` to within 3.1e-5). Truth: each clip's ``rps.npy`` —
  the SAME truth the baseline ``pred_mae`` column of
  ``vk_valid_comparison.csv`` was computed against.
* Metric: per-clip PIT-aligned MAE (rev/s) — one alignment per clip via
  ``tasks.rps_prediction.align_rps_to_gt`` (MSE-Hungarian, identical to the
  baseline dump that produced ``results/vk_eval/predictor_baseline.npz``).
* Pooled numbers: mean of per-clip MAE over the DREGON cruise clips
  (regime 'cruise', n=18), DREGON regime_mean_rps>30 (n=19), and FLY124
  cruise (n=9) — the vk_valid_comparison regime labels are embedded below.
  Reference bars (same clips, from the CSV): E12 transformer per-clip
  baseline DREGON-cruise 3.186 / FLY124-cruise 1.766; telemetry-init VK
  (vk_rec) 0.729 / 0.282. Blind-VK campaign targets (different windows —
  20 s, vk_blind_sweep): DREGON pooled ~0.68-0.74, FLY124 3.24.

Aggregation arms (per model × input mode):

* ``none``      — independent per-clip inference; exact baseline-protocol
                  reproduction (verified against the embedded per-clip
                  reference MAEs for the two baseline checkpoints).
* ``stitch``    — sliding 8 s windows (hop 1.024 s) over each CONCATENATED
                  recording; each window's rotor rows are permutation-aligned
                  to the running stitched estimate on the overlap
                  (predictions are PIT-trained, so rotor order is arbitrary
                  per window); per-frame mean across overlapping windows.
* ``stitchmed`` — per-frame median across overlapping windows.
* ``ma{2,5,10,20}``  — boxcar moving average (span in seconds) on ``stitch``.
* ``med{2,5,10,20}`` — running median (span in seconds) on ``stitch``.

Input modes: ``ch0`` (mic channel 0, the baseline protocol) and ``chmean``
(run all 8 mics, align each mic's prediction to mic 0's within the window,
average — fair vs VK, which also uses all 8 channels).

Checkpoints (R2, resolved via ``training.artifacts.resolve_checkpoint_uri``):
E12 real-full-flight transformer best+last, E12 scv2 best, C11
DREGON+FLY125 scv2 best (wandb 955yy1wv).

Run:
  python scripts/rps_predictor_vk_eval.py                  # all models/modes
  ... --models e12_transformer_best --modes ch0 --quick    # smoke (1 rec)
  ... --data dload:DREGON-LM-V4-michaels-valid-full        # cloud boxes
Remote (kaggle P100):
  omnirun submit --backend kaggle --gpus 1 --time 1h -- \
    python scripts/rps_predictor_vk_eval.py --data dload:DREGON-LM-V4-michaels-valid-full

Outputs (``results/rps_predictor_vk_eval/`` unless ``--out``): per-clip
``per_clip.csv``, pooled ``report.json``, stitched trajectories
``traj_<model>_<mode>.npz``, and a pooled-MAE table on stdout.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

_HERE = Path(__file__).resolve().parent
# Pin the repo's src/ ahead of site-packages (same rationale as
# scripts/vk_blind_sweep.py: the editable install points at whatever checkout
# owns .venv, which on omnirun worktrees is NOT the job's checkout).
_ROOT = _HERE.parent if (_HERE.parent / "src").is_dir() else Path.cwd().resolve()
sys.path.insert(0, str(_ROOT / "src"))

SR = 16000
HOP = 512
FRAME_S = HOP / SR  # 0.032 s
CLIP_SAMPLES = 128000  # 8 s
CLIP_FRAMES = 251
CLIP_HOP_FRAMES = 250  # consecutive clips share one boundary frame
# Sliding-window defaults; overridable via --win-frames/--slide-frames.
# window samples = (win_frames - 1) * HOP; model output frames == win_frames.
# Defaults: 8 s windows (matching the baseline per-clip protocol), 1.024 s hop.
# E12 was TRAINED on 1.0 s chunks, so short windows (e.g. --win-frames 33 =
# 1.024 s, --slide-frames 8) probe inference at the training length.
DEFAULT_WIN_FRAMES = CLIP_FRAMES
DEFAULT_SLIDE_FRAMES = 32
N_ROTORS = 4
SPANS_S = (2.0, 5.0, 10.0, 20.0)

# (clip_id, recording_id, start_s, regime, regime_mean_rps,
#  ref_mae_e12_transformer_best, ref_mae_e12_scv2_best, ref_mae_vk_rec)
# — copied from results/vk_eval/vk_valid_comparison.csv (gitignored, hence
# embedded). regime: mean GT rps <5 ground, <45 warmup, else cruise.
CLIPS: list[tuple[str, str, float, str, float, float, float, float]] = [
    ("sample_00000", "free-flight_speech-low_room1", 0.0, "ground", 0.000, 13.6093, 7.9962, 0.0012),
    ("sample_00001", "free-flight_speech-low_room1", 8.0, "cruise", 59.147, 8.3382, 8.8653, 2.6009),
    (
        "sample_00002",
        "free-flight_speech-low_room1",
        16.0,
        "cruise",
        80.442,
        1.5895,
        5.0510,
        0.7315,
    ),
    (
        "sample_00003",
        "free-flight_speech-low_room1",
        24.0,
        "cruise",
        80.329,
        2.0178,
        4.9391,
        0.6351,
    ),
    (
        "sample_00004",
        "free-flight_speech-low_room1",
        32.0,
        "cruise",
        80.074,
        2.6838,
        4.6931,
        0.7418,
    ),
    (
        "sample_00005",
        "free-flight_speech-low_room1",
        40.0,
        "cruise",
        80.248,
        2.3678,
        4.8889,
        0.6251,
    ),
    (
        "sample_00006",
        "free-flight_speech-low_room1",
        48.0,
        "cruise",
        79.906,
        2.6135,
        4.5444,
        0.7988,
    ),
    (
        "sample_00007",
        "free-flight_whitenoise-low_room1",
        0.0,
        "ground",
        0.000,
        9.3396,
        2.3082,
        0.0000,
    ),
    (
        "sample_00008",
        "free-flight_whitenoise-low_room1",
        8.0,
        "warmup",
        36.183,
        17.9255,
        9.1599,
        0.2242,
    ),
    (
        "sample_00009",
        "free-flight_whitenoise-low_room1",
        16.0,
        "cruise",
        80.724,
        1.9118,
        5.3548,
        0.6091,
    ),
    (
        "sample_00010",
        "free-flight_whitenoise-low_room1",
        24.0,
        "cruise",
        80.299,
        2.2261,
        4.9071,
        0.4514,
    ),
    (
        "sample_00011",
        "free-flight_whitenoise-low_room1",
        32.0,
        "cruise",
        80.304,
        2.7455,
        4.9127,
        0.5415,
    ),
    (
        "sample_00012",
        "free-flight_whitenoise-low_room1",
        40.0,
        "cruise",
        80.097,
        3.0553,
        4.8191,
        0.6573,
    ),
    (
        "sample_00013",
        "free-flight_whitenoise-low_room1",
        48.0,
        "cruise",
        80.251,
        2.8681,
        5.0380,
        0.4916,
    ),
    ("sample_00014", "free-flight_nosource_room1", 0.0, "ground", 0.129, 1.2521, 0.3997, 0.0964),
    ("sample_00015", "free-flight_nosource_room1", 8.0, "cruise", 65.308, 12.9447, 8.0838, 1.1385),
    ("sample_00016", "free-flight_nosource_room1", 16.0, "cruise", 80.970, 1.4727, 5.5776, 0.5755),
    ("sample_00017", "free-flight_nosource_room1", 24.0, "cruise", 80.484, 1.8671, 5.0981, 0.6147),
    ("sample_00018", "free-flight_nosource_room1", 32.0, "cruise", 80.461, 2.4700, 5.0840, 0.4032),
    ("sample_00019", "free-flight_nosource_room1", 40.0, "cruise", 80.298, 2.1545, 5.0122, 0.4253),
    ("sample_00020", "free-flight_nosource_room1", 48.0, "cruise", 80.415, 1.7711, 5.0231, 0.5883),
    ("sample_00021", "free-flight_nosource_room1", 56.0, "cruise", 80.635, 2.2440, 5.2608, 0.4999),
    ("sample_00022", "michaels_FLY124", 0.0, "warmup", 31.718, 6.1208, 37.3438, 0.2276),
    ("sample_00023", "michaels_FLY124", 8.0, "warmup", 36.138, 1.1473, 12.6614, 0.1384),
    ("sample_00024", "michaels_FLY124", 16.0, "warmup", 36.142, 1.0953, 18.3958, 0.1300),
    ("sample_00025", "michaels_FLY124", 24.0, "warmup", 38.287, 1.6295, 13.4454, 0.1615),
    ("sample_00026", "michaels_FLY124", 32.0, "cruise", 78.278, 4.0703, 5.6848, 0.3402),
    ("sample_00027", "michaels_FLY124", 40.0, "cruise", 80.374, 2.4864, 5.2242, 0.2381),
    ("sample_00028", "michaels_FLY124", 48.0, "cruise", 80.261, 1.5397, 5.2159, 0.2712),
    ("sample_00029", "michaels_FLY124", 56.0, "cruise", 80.132, 1.1177, 4.7920, 0.2714),
    ("sample_00030", "michaels_FLY124", 64.0, "cruise", 80.333, 1.2952, 5.0142, 0.3152),
    ("sample_00031", "michaels_FLY124", 72.0, "cruise", 79.833, 1.3358, 4.5770, 0.2832),
    ("sample_00032", "michaels_FLY124", 80.0, "cruise", 80.542, 0.9309, 5.1544, 0.2449),
    ("sample_00033", "michaels_FLY124", 88.0, "cruise", 79.724, 1.4656, 4.5198, 0.3190),
    ("sample_00034", "michaels_FLY124", 96.0, "cruise", 79.566, 1.6482, 4.6080, 0.2588),
    ("sample_00035", "michaels_FLY124", 104.0, "warmup", 28.566, 17.6682, 10.2504, 0.3089),
    ("sample_00036", "michaels_FLY124", 112.0, "ground", -0.004, 34.3880, 25.6683, 0.0047),
]

DREGON_RECS = {
    "free-flight_speech-low_room1",
    "free-flight_whitenoise-low_room1",
    "free-flight_nosource_room1",
}

# name -> (experiment config, checkpoint URI, index into CLIPS ref columns
#          for the none-arm protocol verification, or None)
MODELS: dict[str, tuple[str, str, int | None]] = {
    "e12_transformer_best": (
        "e12_real_fullflight_transformer",
        "r2://ml-data/artifacts/e12_real_fullflight_transformer/checkpoints/best.ckpt",
        5,
    ),
    "e12_transformer_last": (
        "e12_real_fullflight_transformer",
        "r2://ml-data/artifacts/e12_real_fullflight_transformer/checkpoints/last.ckpt",
        None,
    ),
    "e12_scv2_best": (
        "e12_real_fullflight_scv2",
        "r2://ml-data/artifacts/e12_real_fullflight_scv2/checkpoints/best.ckpt",
        6,
    ),
    "c11_scv2_best": (
        "c11_dregon_fly125_retrain",
        "r2://ml-data/artifacts/c11_dregon_fly125_retrain/checkpoints/best.ckpt",
        None,
    ),
    # G1 phase-B (VK-parity): E12 recipe retrained with 4 s / 8 s native
    # context (wandb 9pf3rpoh / 4bwjujj7). best = early-stop optimum
    # (ep 16 / ep 8), last = final epoch (36 / 28).
    "g1_transformer_4s_best": (
        "g1_transformer_4s",
        "r2://ml-data/artifacts/g1_transformer_4s/checkpoints/best.ckpt",
        None,
    ),
    "g1_transformer_4s_last": (
        "g1_transformer_4s",
        "r2://ml-data/artifacts/g1_transformer_4s/checkpoints/last.ckpt",
        None,
    ),
    "g1_transformer_8s_best": (
        "g1_transformer_8s",
        "r2://ml-data/artifacts/g1_transformer_8s/checkpoints/best.ckpt",
        None,
    ),
    "g1_transformer_8s_last": (
        "g1_transformer_8s",
        "r2://ml-data/artifacts/g1_transformer_8s/checkpoints/last.ckpt",
        None,
    ),
    # G2 front-end arms (wandb 5c4p0rim / 4ki6b5ky) and the G3 GP-noise
    # augmentation arm (9wwaa7vb) — all E12 recipe, early-stop best ckpts.
    "g2_hcqt_best": (
        "g2_hcqt_transformer",
        "r2://ml-data/artifacts/g2_hcqt_transformer/checkpoints/best.ckpt",
        None,
    ),
    "g2_if_best": (
        "g2_if_transformer",
        "r2://ml-data/artifacts/g2_if_transformer/checkpoints/best.ckpt",
        None,
    ),
    "g3_gp_aug_best": (
        "g3_gp_aug_transformer",
        "r2://ml-data/artifacts/g3_gp_aug_transformer/checkpoints/best.ckpt",
        None,
    ),
    # G6 strong-augmentation arms (wandb 03zhc73x / t53eoon4).
    "g6_strongaug_if_best": (
        "g6_strongaug_if",
        "r2://ml-data/artifacts/g6_strongaug_if/checkpoints/best.ckpt",
        None,
    ),
    "g6_strongaug_transformer_best": (
        "g6_strongaug_transformer",
        "r2://ml-data/artifacts/g6_strongaug_transformer/checkpoints/best.ckpt",
        None,
    ),
    # CKLA P1 (wandb s4u1tb7w) — complex-KLA temporal head on the E12 recipe
    # (docs/experiments/ckla.md). best = ep 22 val optimum, last = ep 42.
    "ckla_p1_best": (
        "ckla_p1_if",
        "r2://ml-data/artifacts/ckla_p1_if/checkpoints/best.ckpt",
        None,
    ),
    "ckla_p1_last": (
        "ckla_p1_if",
        "r2://ml-data/artifacts/ckla_p1_if/checkpoints/last.ckpt",
        None,
    ),
    # Mechanistic-lever arms (wandb smwulrhf / hilihk2v): p_init 1.0 gain
    # restoration and freq-scale-only augmentation — docs/experiments/ckla.md.
    "ckla_pnoise_best": (
        "ckla_p1_pnoise",
        "r2://ml-data/artifacts/ckla_p1_pnoise/checkpoints/best.ckpt",
        None,
    ),
    "ckla_freqscale_best": (
        "ckla_p1_freqscale",
        "r2://ml-data/artifacts/ckla_p1_freqscale/checkpoints/best.ckpt",
        None,
    ),
    # Protocol-B (legacy-equivalent epochs/batch, augs live from epoch 10):
    # the pnoise arm re-run — interim reads use whatever best.ckpt the
    # still-training kaggle job has uploaded; re-eval after it finishes.
    "ckla_pnoise_pb_best": (
        "ckla_p1_pnoise",
        "r2://ml-data/artifacts/ckla_p1_pnoise_pb/checkpoints/best.ckpt",
        None,
    ),
    # Clean-pipeline arms (channel_drop-free, padding-free freq_scale):
    "ckla_pnfs_pb_best": (
        "ckla_p1_pnfs",
        "r2://ml-data/artifacts/ckla_p1_pnfs_pb/checkpoints/best.ckpt",
        None,
    ),
    "g2_if_freqscale_best": (
        "g2_if_freqscale",
        "r2://ml-data/artifacts/g2_if_freqscale/checkpoints/best.ckpt",
        None,
    ),
    # Freq-scale v2 (p=1.0, alpha [0.7,1.3], hard 5-epoch warmup) and v3
    # synthesis-first (freq_scale+warp always-on from sample 0, corruption
    # ramp 0->0.7 over epochs 5->25) arms:
    "g2_if_freqscale_v2_best": (
        "g2_if_freqscale_v2",
        "r2://ml-data/artifacts/g2_if_freqscale_v2/checkpoints/best.ckpt",
        None,
    ),
    "ckla_pnoise_fs_v2_best": (
        "ckla_pnoise_fs_v2",
        "r2://ml-data/artifacts/ckla_pnoise_fs_v2/checkpoints/best.ckpt",
        None,
    ),
    "g2_if_v3synth_best": (
        "g2_if_v3synth",
        "r2://ml-data/artifacts/g2_if_v3synth/checkpoints/best.ckpt",
        None,
    ),
    "ckla_pnoise_v3synth_best": (
        "ckla_pnoise_v3synth",
        "r2://ml-data/artifacts/ckla_pnoise_v3synth/checkpoints/best.ckpt",
        None,
    ),
}

POOLS: dict[str, Any] = {
    "dregon_cruise": lambda c: c[1] in DREGON_RECS and c[3] == "cruise",
    "dregon_gt30": lambda c: c[1] in DREGON_RECS and c[4] > 30.0,
    "fly124_cruise": lambda c: c[1] == "michaels_FLY124" and c[3] == "cruise",
    "all_37": lambda c: True,
}


def span_frames(span_s: float) -> int:
    n = int(round(span_s / FRAME_S))
    return n + 1 if n % 2 == 0 else n


def perm_align(pred: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Permute pred's rotor rows to best match ref (MSE Hungarian)."""
    from scipy.optimize import linear_sum_assignment

    cost = np.mean((pred[:, None, :] - ref[None, :, :]) ** 2, axis=-1)
    row, col = linear_sum_assignment(cost)
    out = np.empty_like(pred)
    out[col] = pred[row]
    return out


def moving_average(traj: np.ndarray, n: int) -> np.ndarray:
    """Boxcar along frames with proper edge normalization. traj (R, F)."""
    kernel = np.ones(n, dtype=np.float64)
    denom = np.convolve(np.ones(traj.shape[1], dtype=np.float64), kernel, mode="same")
    return np.stack(
        [np.convolve(traj[r], kernel, mode="same") / denom for r in range(traj.shape[0])]
    )


def running_median(traj: np.ndarray, n: int) -> np.ndarray:
    from scipy.ndimage import median_filter

    return median_filter(traj, size=(1, n), mode="nearest")


def load_clip_data(data_path: str) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Return (audio, gt): clip_id -> (C, 128000) float32 / (4, 251) float32."""
    from data_processing.frame_datasets import DregonLMFrameDataset

    ds = DregonLMFrameDataset(data_path, n_fft=2048, hop_length=HOP, sample_rate=SR, channel=None)
    names = [p.name for p in ds.samples]
    expected = [c[0] for c in CLIPS]
    if names != expected:
        raise RuntimeError(
            f"dataset clip list mismatch: got {names[:3]}... expected {expected[:3]}..."
        )
    audio: dict[str, np.ndarray] = {}
    gt: dict[str, np.ndarray] = {}
    for i, name in enumerate(names):
        fr = ds[i]
        a = np.atleast_2d(np.asarray(fr["mixture"].data, dtype=np.float32))
        g = np.asarray(fr["rps"].data, dtype=np.float32)
        if a.shape[-1] != CLIP_SAMPLES or g.shape != (N_ROTORS, CLIP_FRAMES):
            raise RuntimeError(f"{name}: unexpected shapes audio {a.shape} gt {g.shape}")
        audio[name] = a
        gt[name] = g
    return audio, gt


def load_model(experiment: str, ckpt_uri: str, device: str):
    import torch
    from hydra import compose, initialize_config_dir

    from training.artifacts import resolve_checkpoint_uri
    from training.config import instantiate_model, register_configs

    register_configs()
    with initialize_config_dir(config_dir=str(_ROOT / "conf"), version_base=None):
        cfg = compose(config_name="config", overrides=[f"experiment={experiment}"])
    model = instantiate_model(cfg.model)
    local = resolve_checkpoint_uri(ckpt_uri, str(_ROOT / ".cache" / "r2_checkpoints"))
    sd = torch.load(local, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    model.load_state_dict(sd)
    model.eval()
    return model.to(device)


def batched_forward(
    model, wins: np.ndarray, device: str, batch: int, out_frames: int
) -> np.ndarray:
    """wins (N, T) float32 -> (N, 4, out_frames) float32."""
    import torch

    outs = []
    with torch.no_grad():
        for i in range(0, wins.shape[0], batch):
            t = torch.from_numpy(wins[i : i + batch]).to(device)
            out = model(t)
            outs.append(out.float().cpu().numpy())
    pred = np.concatenate(outs, axis=0)
    if pred.shape[1] != N_ROTORS or pred.shape[2] != out_frames:
        raise RuntimeError(f"unexpected model output shape {pred.shape} (expected F={out_frames})")
    return pred


def window_starts(f_total: int, win_frames: int, slide_frames: int) -> list[int]:
    last = f_total - win_frames
    starts = list(range(0, last + 1, slide_frames))
    if starts[-1] != last:
        starts.append(last)
    return starts


def predict_windows(
    model,
    rec_audio: np.ndarray,
    starts: list[int],
    mode: str,
    device: str,
    batch: int,
    win_frames: int,
) -> np.ndarray:
    """Predict every sliding window. Returns (W, 4, win_frames).

    ``chmean``: forward all C mics per window, permutation-align each mic's
    rotor rows to mic 0's, average across mics.
    """
    win_samples = (win_frames - 1) * HOP
    if mode == "ch0":
        wins = np.stack([rec_audio[0, s * HOP : s * HOP + win_samples] for s in starts])
        return batched_forward(model, wins, device, batch, win_frames)
    if mode != "chmean":
        raise ValueError(f"unknown input mode {mode!r}")
    n_ch = rec_audio.shape[0]
    wins = np.stack(
        [rec_audio[c, s * HOP : s * HOP + win_samples] for s in starts for c in range(n_ch)]
    )
    flat = batched_forward(model, wins, device, batch, win_frames)  # (W*C, 4, F)
    per_win = flat.reshape(len(starts), n_ch, N_ROTORS, win_frames)
    out = np.empty((len(starts), N_ROTORS, win_frames), dtype=np.float32)
    for w in range(len(starts)):
        ref = per_win[w, 0]
        acc = ref.astype(np.float64).copy()
        for c in range(1, n_ch):
            acc += perm_align(per_win[w, c].astype(np.float64), ref)
        out[w] = (acc / n_ch).astype(np.float32)
    return out


def stitch_stack(preds: np.ndarray, starts: list[int], f_total: int, win_frames: int) -> np.ndarray:
    """Align each window to the running stitched mean; return (W, 4, F) NaN-padded."""
    stack = np.full((len(starts), N_ROTORS, f_total), np.nan, dtype=np.float32)
    acc = np.zeros((N_ROTORS, f_total), dtype=np.float64)
    cnt = np.zeros(f_total, dtype=np.float64)
    for w, s in enumerate(starts):
        p = preds[w].astype(np.float64)
        sl = slice(s, s + win_frames)
        seen = cnt[sl] > 0
        if seen.any():
            ref = acc[:, sl][:, seen] / cnt[sl][seen]
            p = perm_align_overlap(p, ref, seen)
        stack[w, :, sl] = p.astype(np.float32)
        acc[:, sl] += p
        cnt[sl] += 1.0
    return stack


def perm_align_overlap(pred: np.ndarray, ref_overlap: np.ndarray, seen: np.ndarray) -> np.ndarray:
    """Permute pred rows using only the overlap columns for the cost."""
    from scipy.optimize import linear_sum_assignment

    p_ov = pred[:, seen]
    cost = np.mean((p_ov[:, None, :] - ref_overlap[None, :, :]) ** 2, axis=-1)
    row, col = linear_sum_assignment(cost)
    out = np.empty_like(pred)
    out[col] = pred[row]
    return out


def arm_trajectories(stack: np.ndarray) -> dict[str, np.ndarray]:
    stitch = np.nanmean(stack, axis=0)
    arms = {"stitch": stitch, "stitchmed": np.nanmedian(stack, axis=0)}
    for span in SPANS_S:
        n = span_frames(span)
        arms[f"ma{span:g}"] = moving_average(stitch, n)
        arms[f"med{span:g}"] = running_median(stitch, n)
    return arms


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    ap.add_argument("--data", default=None, help="dataset dir or dload: URI")
    ap.add_argument("--models", nargs="+", choices=sorted(MODELS), default=sorted(MODELS))
    ap.add_argument("--modes", nargs="+", choices=["ch0", "chmean"], default=["ch0", "chmean"])
    ap.add_argument("--device", default=None, help="cuda|cpu (default: auto)")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--out", default="results/rps_predictor_vk_eval")
    ap.add_argument("--quick", action="store_true", help="speech-low recording only (smoke)")
    ap.add_argument(
        "--win-frames",
        type=int,
        default=DEFAULT_WIN_FRAMES,
        help="sliding-window length in STFT frames; samples = (F-1)*512 "
        "(default 251 = 8 s; 33 = 1.024 s, the E12 training chunk length)",
    )
    ap.add_argument(
        "--slide-frames",
        type=int,
        default=DEFAULT_SLIDE_FRAMES,
        help="sliding-window hop in frames (default 32 = 1.024 s)",
    )
    args = ap.parse_args()
    win_frames: int = args.win_frames
    slide_frames: int = args.slide_frames

    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    data_path = args.data
    if data_path is None:
        local = _ROOT / "datasets" / "DREGON-LM-V4-michaels-full" / "valid"
        data_path = str(local) if local.is_dir() else "dload:DREGON-LM-V4-michaels-valid-full"
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[vk_eval] src pin: {_ROOT / 'src'} | data: {data_path} | device: {device}", flush=True)

    audio, gt = load_clip_data(data_path)

    # Group clips per recording, ordered by start time (verified contiguous).
    recs: dict[str, list[tuple[str, float]]] = {}
    for c in CLIPS:
        recs.setdefault(c[1], []).append((c[0], c[2]))
    for rid in recs:
        recs[rid].sort(key=lambda x: x[1])
    if args.quick:
        recs = {"free-flight_speech-low_room1": recs["free-flight_speech-low_room1"]}
    clip_info = {c[0]: c for c in CLIPS}

    per_clip_rows: list[dict[str, Any]] = []
    pooled: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    verification: dict[str, float] = {}

    for model_name in args.models:
        experiment, ckpt_uri, ref_col = MODELS[model_name]
        t0 = time.time()
        print(f"\n=== {model_name} ({experiment}) ===", flush=True)
        model = load_model(experiment, ckpt_uri, device)
        pooled[model_name] = {}
        for mode in args.modes:
            arm_maes: dict[str, dict[str, float]] = {}
            traj_dump: dict[str, np.ndarray] = {}
            for rid, clip_list in recs.items():
                ids = [cid for cid, _ in clip_list]
                rec_audio = np.concatenate([audio[cid] for cid in ids], axis=-1)
                f_total = len(ids) * CLIP_HOP_FRAMES + 1
                starts = window_starts(f_total, win_frames, slide_frames)
                preds = predict_windows(
                    model, rec_audio, starts, mode, device, args.batch, win_frames
                )
                stack = stitch_stack(preds, starts, f_total, win_frames)
                arms = arm_trajectories(stack)
                traj_dump[rid] = arms["stitch"].astype(np.float32)

                # 'none' arm: independent per-clip inference (baseline protocol).
                clip_wins = np.stack(
                    [audio[cid][0] for cid in ids]
                    if mode == "ch0"
                    else [audio[cid][c] for cid in ids for c in range(audio[ids[0]].shape[0])]
                )
                if mode == "ch0":
                    none_preds = batched_forward(model, clip_wins, device, args.batch, CLIP_FRAMES)
                else:
                    n_ch = audio[ids[0]].shape[0]
                    flat = batched_forward(model, clip_wins, device, args.batch, CLIP_FRAMES)
                    per_clip = flat.reshape(len(ids), n_ch, N_ROTORS, CLIP_FRAMES)
                    none_preds = np.empty((len(ids), N_ROTORS, CLIP_FRAMES), dtype=np.float32)
                    for k in range(len(ids)):
                        ref = per_clip[k, 0]
                        acc = ref.astype(np.float64).copy()
                        for c in range(1, n_ch):
                            acc += perm_align(per_clip[k, c].astype(np.float64), ref)
                        none_preds[k] = (acc / n_ch).astype(np.float32)

                from tasks.rps_prediction import align_rps_to_gt

                for k, cid in enumerate(ids):
                    g = gt[cid]
                    g0 = k * CLIP_HOP_FRAMES
                    clip_arms = {"none": none_preds[k]}
                    for arm, traj in arms.items():
                        clip_arms[arm] = traj[:, g0 : g0 + CLIP_FRAMES]
                    for arm, seg in clip_arms.items():
                        pa = align_rps_to_gt(seg, g)
                        d = pa - g
                        mae = float(np.mean(np.abs(d)))
                        info = clip_info[cid]
                        per_clip_rows.append(
                            {
                                "model": model_name,
                                "mode": mode,
                                "arm": arm,
                                "clip": cid,
                                "recording": rid,
                                "regime": info[3],
                                "regime_mean_rps": info[4],
                                "mae": mae,
                                "mse": float(np.mean(d**2)),
                            }
                        )
                        arm_maes.setdefault(arm, {})[cid] = mae

            # Protocol verification: none-arm ch0 vs the embedded baseline MAEs.
            if mode == "ch0" and ref_col is not None:
                deltas = [
                    abs(arm_maes["none"][c[0]] - float(c[ref_col]))  # type: ignore[arg-type]
                    for c in CLIPS
                    if c[0] in arm_maes.get("none", {})
                ]
                if deltas:
                    verification[model_name] = float(np.max(deltas))
                    print(
                        f"  [verify] none-arm vs stored baseline per-clip MAE: "
                        f"max |delta| = {max(deltas):.4f} over {len(deltas)} clips",
                        flush=True,
                    )

            pooled[model_name][mode] = {}
            for arm, by_clip in arm_maes.items():
                pooled[model_name][mode][arm] = {}
                for pool_name, sel in POOLS.items():
                    vals = [by_clip[c[0]] for c in CLIPS if c[0] in by_clip and sel(c)]
                    if vals:
                        pooled[model_name][mode][arm][pool_name] = float(np.mean(vals))
            traj_arrays: dict[str, Any] = {f"stitch__{rid}": tr for rid, tr in traj_dump.items()}
            np.savez_compressed(out_dir / f"traj_{model_name}_{mode}.npz", **traj_arrays)
        del model
        print(f"  done in {time.time() - t0:.1f} s", flush=True)

    # ── report ──
    arm_order = (
        ["none", "stitch", "stitchmed"]
        + [f"ma{s:g}" for s in SPANS_S]
        + [f"med{s:g}" for s in SPANS_S]
    )
    lines = [
        "Pooled PIT-MAE (rev/s), mean of per-clip MAE",
        "reference: VK telemetry-init (same clips) dregon_cruise 0.729 fly124_cruise 0.282;",
        "           blind-VK campaign targets dregon ~0.68-0.74, fly124 3.24 (20 s windows)",
    ]
    header = f"{'model':<22}{'mode':<8}{'arm':<11}" + "".join(f"{p:>15}" for p in POOLS)
    for model_name in args.models:
        lines += ["", header, "-" * len(header)]
        for mode in args.modes:
            for arm in arm_order:
                cells = pooled.get(model_name, {}).get(mode, {}).get(arm)
                if not cells:
                    continue
                row = f"{model_name:<22}{mode:<8}{arm:<11}"
                for p in POOLS:
                    row += f"{cells.get(p, float('nan')):>15.3f}"
                lines.append(row)
    table = "\n".join(lines)
    print("\n" + table, flush=True)

    report = {
        "protocol": "vk_valid_comparison per-clip PIT-MAE (align_rps_to_gt once per clip)",
        "data": data_path,
        "win_frames": win_frames,
        "slide_frames": slide_frames,
        "spans_s": list(SPANS_S),
        "verification_max_abs_delta": verification,
        "pooled": pooled,
        "reference_pooled": {
            "e12_transformer_best_baseline": {"dregon_cruise": 3.1856, "fly124_cruise": 1.7655},
            "e12_scv2_best_baseline": {"dregon_cruise": 5.3974, "fly124_cruise": 4.9767},
            "vk_telemetry_init": {"dregon_cruise": 0.7294, "fly124_cruise": 0.2825},
            "vk_blind_targets_20s_windows": {"dregon": "0.68-0.74", "fly124": 3.24},
        },
    }
    with open(out_dir / "report.json", "w") as f:
        json.dump(report, f, indent=2)
    with open(out_dir / "per_clip.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_clip_rows[0].keys()))
        writer.writeheader()
        writer.writerows(per_clip_rows)
    with open(out_dir / "summary.txt", "w") as f:
        f.write(table + "\n")
    print(f"\n[vk_eval] wrote {out_dir}/report.json, per_clip.csv, summary.txt", flush=True)


if __name__ == "__main__":
    main()
