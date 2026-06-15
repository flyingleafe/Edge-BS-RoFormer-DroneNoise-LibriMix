#!/usr/bin/env python3
"""Evaluate the narrow-band + super-resolution salience RPS models on DREGON-LM-V4/valid.

Mirrors the eval that produced
``results/dregon_v4_eval/salience_baselines_final_valid.json`` (PIT, track_threshold=0.3),
for the two new checkpoints under ``results/rps_baselines_v4/*_narrow_sr/``.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from train_rps_predictor import DREGONRPSDataset, evaluate, get_model  # noqa: E402

DATASET = "datasets/DREGON-LM-V4/valid"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRACK_THRESHOLD = 0.3
OUT_JSON = "results/dregon_v4_eval/salience_narrow_sr_final_valid.json"

# narrow input + super-resolution output configs (from the experiment run config)
MULTIF0_CFG = dict(
    n_octaves=1,
    over_sample=10,
    harmonics=[1, 2, 3, 4],
    superres_out=True,
    out_fmin=55.0,
    out_fmax=110.0,
    out_bins=360,
)
BP_CFG = dict(
    bp_fmin=55.0,
    bins_per_semitone=4,
    n_contour_semitones=12,
    superres_out=True,
    out_fmin=55.0,
    out_fmax=110.0,
    out_bins=360,
)

CHECKPOINTS = {
    "multif0_salience_narrow_sr": (
        lambda: get_model("multif0_salience", hcqt_fmin=55.0, salience_cfg=MULTIF0_CFG),
        "results/rps_baselines_v4/multif0_salience_narrow_sr/best_multif0_salience.pt",
    ),
    "basic_pitch_narrow_sr": (
        lambda: get_model("basic_pitch_salience", salience_cfg=BP_CFG),
        "results/rps_baselines_v4/basic_pitch_narrow_sr/best_basic_pitch_salience.pt",
    ),
}


def main():
    smoke = "--smoke" in sys.argv
    models = {}
    for name, (ctor, ckpt) in CHECKPOINTS.items():
        m = ctor().to(DEVICE)
        sd = torch.load(ckpt, map_location=DEVICE, weights_only=True)
        m.load_state_dict(sd, strict=True)
        m.eval()
        models[name] = m
        n_params = sum(p.numel() for p in m.parameters())
        print(f"loaded {name}: {n_params:,} params, out_freqs={m.output_freqs().shape}")
    if smoke:
        wav = torch.randn(2, 16000, device=DEVICE)
        for name, m in models.items():
            with torch.no_grad():
                pred = m.predict_rps(wav, threshold=TRACK_THRESHOLD)
            print(f"  {name} predict_rps -> {tuple(pred.shape)}")
        print("smoke OK")
        return

    ds = DREGONRPSDataset(DATASET, n_fft=2048, hop_length=512)
    loader = DataLoader(ds, batch_size=2, shuffle=False, num_workers=2)
    results = {}
    for name, model in models.items():
        print(f"evaluating {name} ...", flush=True)
        t0 = time.time()
        m = evaluate(
            model,
            loader,
            DEVICE,
            len(ds),
            pit_eval=True,
            track_threshold=TRACK_THRESHOLD,
            progress=True,
        )
        m["rmse"] = m["mse"] ** 0.5
        m["eval_seconds"] = round(time.time() - t0, 1)
        results[name] = m
        print(
            f"  {name}: RMSE {m['rmse']:.3f}  R2 {m['r2']:.3f}  "
            f"MAE_frame {m['mae_frame']:.3f}  ({m['eval_seconds']}s)",
            flush=True,
        )

    os.makedirs("results/dregon_v4_eval", exist_ok=True)
    payload = {
        "dataset": DATASET,
        "n_samples": len(ds),
        "channels": 8,
        "track_threshold": TRACK_THRESHOLD,
        "pit_eval": True,
        "device": DEVICE,
        "checkpoints": {k: v[1] for k, v in CHECKPOINTS.items()},
        "model_configs": {
            "multif0_salience_narrow_sr": {
                "class": "LateDeepSalience",
                "fmin": 55.0,
                **MULTIF0_CFG,
            },
            "basic_pitch_narrow_sr": {"class": "BasicPitchSalience", **BP_CFG},
        },
        "results": results,
    }
    with open(OUT_JSON, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
