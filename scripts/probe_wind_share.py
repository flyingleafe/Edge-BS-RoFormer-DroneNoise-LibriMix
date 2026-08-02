"""How much of the predicted variance does the trained wind channel actually carry?

The wind arm lost to its no-wind control on both drones. That is only evidence
about the WAKE MODEL if the channel is contributing meaningfully to the
prediction. If its share of the predicted variance is negligible, the comparison
measures optimization noise from adding dead parameters, not physics — and the
wake model remains untested.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data_processing import sources  # noqa: E402
from data_processing.frame_datasets import _frames_spec_geometry  # noqa: E402
from models.registry import build_noise_gen_model  # noqa: E402
from tasks.noise_generation import geometry_to_rel_pos  # noqa: E402
from training.artifacts import resolve_checkpoint_uri  # noqa: E402

CKPT = "r2://ml-data/artifacts/gen_w4_lik_wind_mm/checkpoints/best.ckpt"


def main() -> None:
    model = build_noise_gen_model(
        "positional_harmonic_wind_gen",
        sample_rate=16000,
        n_harmonics=100,
        cond_dim=16,
        drone_names=["dregon", "michaels"],
        rps_jitter_sigma=0.6,
        rps_jitter_tau=0.016,
        learn_rps_jitter_sigma=True,
        z_noise_std=0.1,
        film_spectral_norm=True,
    )
    state = torch.load(resolve_checkpoint_uri(CKPT), map_location="cpu")
    for key in ("state_dict", "model"):
        if isinstance(state, dict) and key in state:
            state = state[key]
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"loaded (missing={len(missing)}, unexpected={len(unexpected)})")
    model.eval()

    geoms = {
        "dregon": _frames_spec_geometry("frames:DREGON-frames"),
        "michaels": sources.geometry("michaels"),
    }
    for drone, (mic, rotor) in geoms.items():
        mic_t = torch.tensor(np.asarray(mic), dtype=torch.float32).unsqueeze(0)
        rot_t = torch.tensor(np.asarray(rotor), dtype=torch.float32).unsqueeze(0)
        rel = geometry_to_rel_pos(mic_t, rot_t)
        rps = torch.full((1, rot_t.shape[1], 16000), 80.0)
        with torch.no_grad():
            total = model.spectral_stats(rps, rel, [drone])["noise_psd"]
            coherent_only = model.generator.coherent.spectral_stats(
                rps, rel, z=model._resolve_conditioning([drone], {})
            )["noise_psd"]
            wind = model.generator.wind.expected_power_rel(rps, rel)
        tp, wp = float(total.mean()), float(wind.mean())
        print(
            f"{drone:9s} total psd {tp:12.4e} | coherent {float(coherent_only.mean()):12.4e} "
            f"| wind {wp:12.4e} | wind share {wp / max(tp, 1e-30):.3e}"
        )


if __name__ == "__main__":
    main()
