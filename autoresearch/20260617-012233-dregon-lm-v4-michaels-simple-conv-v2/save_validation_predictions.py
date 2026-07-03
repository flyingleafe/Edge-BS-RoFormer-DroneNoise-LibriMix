#!/usr/bin/env python3
"""Save raw validation-set RPS predictions for both checkpoint sweeps.

Evaluates every best_*.pt checkpoint from:
  1. the original fixed/offline DREGON-LM-V4-michaels training sweep, and
  2. the 2026-06-18 online-mix rerun,

on DREGON-LM-V4-michaels/valid and stores NumPy arrays for downstream analysis
and plotting. Predictions are saved in the model's native rotor order (before
PIT matching) so they are raw model outputs.
"""

from __future__ import annotations

import glob
import itertools
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset
from tqdm import tqdm

from losses.pit import pit_mse_loss
from models.registry import build_model as get_model

_ROTOR_PERMS = torch.tensor(list(itertools.permutations(range(4))), dtype=torch.long)


class DREGONRPSDataset(Dataset):
    """Load mixture.wav + rps.npy from DREGON-LM, resample RPS to STFT frames.

    Faithful, self-contained port of the former ``train_rps_predictor.py``'s
    class of the same name (see docs/refactor-unified-framework.md); this
    autoresearch session artifact is the only remaining consumer of the exact
    ``(audio, rps)`` tensor-tuple contract, so it is inlined here rather than
    reused from ``data_processing.frame_datasets.DregonLMFrameDataset`` (which
    returns a ``td.Frame``, a different shape).
    """

    def __init__(self, data_dir, n_fft=2048, hop_length=512):
        self.hop_length = hop_length
        self.samples = sorted(
            d
            for d in glob.glob(os.path.join(data_dir, "sample_*"))
            if os.path.isfile(os.path.join(d, "mixture.wav"))
            and os.path.isfile(os.path.join(d, "rps.npy"))
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        d = self.samples[idx]
        audio, _sr = torchaudio.load(os.path.join(d, "mixture.wav"))
        if audio.shape[0] == 1:
            audio = audio[0]  # (T,)
        rps = torch.from_numpy(np.load(os.path.join(d, "rps.npy"))).float()  # (4, rps_T)
        n_frames = audio.shape[-1] // self.hop_length + 1
        rps = F.interpolate(
            rps.unsqueeze(0), size=n_frames, mode="linear", align_corners=False
        ).squeeze(0)
        return audio, rps


MODELS = [
    "simple_conv_v2",
    "simple_conv_v2_transformer",
    "simple_conv_v2_local_attn",
    "simple_conv_v2_multires",
    "simple_conv_v2_dwt",
    "simple_conv_v2_magphase",
    "simple_conv_v2_dual_pool",
    "simple_conv_v2_gru96",
    "simple_conv_v2_uni_gru",
    "simple_conv_v2_causal_gru",
    "simple_conv_v2_causal_gru96",
    "simple_conv_v2_uni_gru128",
    "simple_conv_v2_uni_gru128_norm",
    "simple_conv_v2_uni_gru128_norm_do03",
    "simple_conv_v2_uni_gru96_norm_do03",
    "simple_conv_v2_uni_gru96_norm_do02",
    "simple_conv_v2_uni_gru64_norm_do03",
    "simple_conv_tcn",
    "simple_conv_v2_tcn",
    "simple_conv_v2_causal_tcn",
    "smolnet_rps_tcn",
    "smolnet_rps_causal_tcn",
    "simple_conv_v2_smol_tcn",
    "simple_conv_v2_smol_causal_tcn",
    "smolnet_rps_simple_head",
    "simple_conv_v2_smol_bigru",
]

DATA_ROOT = Path("/gpfs/scratch/acw592/datasets/DREGON-LM-V4-michaels")
SERIES = {
    "offline_fixed_train_50ep": Path(
        "/gpfs/scratch/acw592/results/autoresearch/"
        "20260617-012233-dregon-lm-v4-michaels-simple-conv-v2"
    ),
    "online_mix_200ep_aug50k": Path(
        "/gpfs/scratch/acw592/results/autoresearch/"
        "20260618-v4-michaels-online-mix-200ep-aug50k-gpushort"
    ),
}
MANIFEST_ROOT = Path(
    "/gpfs/scratch/acw592/results/autoresearch/"
    "20260617-012233-dregon-lm-v4-michaels-simple-conv-v2/validation_rps_predictions"
)
N_FFT = 2048
HOP = 512


def _flatten_one(audio: torch.Tensor, rps: torch.Tensor):
    if audio.dim() == 1:  # (T,)
        return audio.unsqueeze(0), rps.unsqueeze(0), [0]
    if audio.dim() == 2:  # (C,T)
        c = audio.shape[0]
        return audio, rps.unsqueeze(0).expand(c, -1, -1), list(range(c))
    raise ValueError(f"unexpected audio shape {tuple(audio.shape)}")


def _match_targets_to_preds(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Target reordered to the PIT-optimal assignment for fixed-order metrics."""
    _, best_idx = pit_mse_loss(pred, target, return_indices=True)
    perms = _ROTOR_PERMS.to(pred.device)[best_idx]
    return torch.gather(target, 1, perms.unsqueeze(-1).expand(-1, -1, target.shape[-1]))


def evaluate_checkpoint(
    *,
    series_name: str,
    checkpoint_root: Path,
    model_name: str,
    ds: DREGONRPSDataset,
    device: torch.device,
) -> dict:
    ckpt = checkpoint_root / model_name / f"best_{model_name}.pt"
    if not ckpt.is_file():
        raise FileNotFoundError(f"missing checkpoint for {series_name}/{model_name}: {ckpt}")

    model = get_model(model_name, n_fft=N_FFT, hop_length=HOP, num_rotors=4).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.eval()

    preds: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    targets_pit: list[np.ndarray] = []
    sample_ids: list[str] = []
    channels: list[int] = []

    label = f"{series_name}/{model_name}"
    with torch.no_grad():
        for idx in tqdm(range(len(ds)), desc=label, unit="sample"):
            audio, rps = ds[idx]
            audio_f, rps_f, chs = _flatten_one(audio.float(), rps.float())
            audio_f = audio_f.to(device)
            rps_f = rps_f.to(device)
            with torch.autocast("cuda", enabled=(device.type == "cuda")):
                pred = model(audio_f)
            if pred.shape[-1] != rps_f.shape[-1]:
                rps_f = F.interpolate(
                    rps_f, size=pred.shape[-1], mode="linear", align_corners=False
                )
            target_pit = _match_targets_to_preds(pred.float(), rps_f.float())

            preds.append(pred.float().cpu().numpy())
            targets.append(rps_f.float().cpu().numpy())
            targets_pit.append(target_pit.float().cpu().numpy())
            sample_name = Path(ds.samples[idx]).name
            sample_ids.extend([sample_name] * len(chs))
            channels.extend(chs)

    pred_np = np.concatenate(preds, axis=0).astype(np.float32)
    target_np = np.concatenate(targets, axis=0).astype(np.float32)
    target_pit_np = np.concatenate(targets_pit, axis=0).astype(np.float32)
    sample_ids_np = np.asarray(sample_ids)
    channels_np = np.asarray(channels, dtype=np.int16)

    out_dir = checkpoint_root / model_name / "validation_rps_predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "pred_raw.npy", pred_np)
    np.save(out_dir / "target.npy", target_np)
    np.save(out_dir / "target_pit_matched_to_pred.npy", target_pit_np)
    np.save(out_dir / "sample_ids.npy", sample_ids_np)
    np.save(out_dir / "channels.npy", channels_np)

    mse = float(np.mean((pred_np - target_pit_np) ** 2))
    mae = float(np.mean(np.abs(pred_np - target_pit_np)))
    clip_mae = float(np.mean(np.abs(pred_np.mean(axis=-1) - target_pit_np.mean(axis=-1))))
    meta = {
        "series": series_name,
        "model": model_name,
        "checkpoint": str(ckpt),
        "dataset": str(DATA_ROOT / "valid"),
        "n_rows": int(pred_np.shape[0]),
        "shape": list(pred_np.shape),
        "arrays": {
            "pred_raw": str(out_dir / "pred_raw.npy"),
            "target": str(out_dir / "target.npy"),
            "target_pit_matched_to_pred": str(out_dir / "target_pit_matched_to_pred.npy"),
            "sample_ids": str(out_dir / "sample_ids.npy"),
            "channels": str(out_dir / "channels.npy"),
        },
        "quick_metrics_from_saved_arrays": {
            "pit_matched_mse": mse,
            "pit_matched_rmse": mse**0.5,
            "pit_matched_mae_frame": mae,
            "pit_matched_mae_clip": clip_mae,
        },
        "note": "pred_raw.npy is raw model output before PIT rotor-order matching.",
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def main() -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")
    MANIFEST_ROOT.mkdir(parents=True, exist_ok=True)
    ds = DREGONRPSDataset(DATA_ROOT / "valid", n_fft=N_FFT, hop_length=HOP)
    print(f"valid samples={len(ds)}")

    all_meta = []
    for series_name, checkpoint_root in SERIES.items():
        print(f"=== series={series_name} root={checkpoint_root} ===")
        for model_name in MODELS:
            all_meta.append(
                evaluate_checkpoint(
                    series_name=series_name,
                    checkpoint_root=checkpoint_root,
                    model_name=model_name,
                    ds=ds,
                    device=device,
                )
            )
    summary = {
        "data_root": str(DATA_ROOT),
        "manifest_root": str(MANIFEST_ROOT),
        "output_layout": "<checkpoint_root>/<model>/validation_rps_predictions/",
        "series": {k: str(v) for k, v in SERIES.items()},
        "n_series": len(SERIES),
        "n_models_per_series": len(MODELS),
        "n_evaluations": len(all_meta),
        "evaluations": all_meta,
    }
    (MANIFEST_ROOT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"saved predictions under each checkpoint folder; summary at {MANIFEST_ROOT}")


if __name__ == "__main__":
    main()
