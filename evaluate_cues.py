"""Run existing paper cue transforms/selectors on the unchanged pilot fixed sets."""

import hashlib
import json
import runpy
from pathlib import Path

from omegaconf import OmegaConf

from experiments.refiner_bench import campaign_config, fixed_dataset_size, load_campaign_model
from training.config import build_dataset
from utils.checkpoints import resolve_checkpoint_uri

cue = runpy.run_path("scripts/rps_cue_probe.py")
for stage in ("s1", "s2"):
    experiment = f"jhtr_cond_{stage}_nomix"
    cfg = campaign_config(experiment)
    dataset = build_dataset(cfg.data.valid)
    frames = [dataset[i] for i in range(fixed_dataset_size(dataset))]
    n_ch = 1 + max(int(frame["meta"]["channel"]) for frame in frames)
    clips = cue["cruise_clips"](frames, n_ch, 6)
    selected = cue["flight_frames"](frames, 16)
    assert clips and selected, "Existing fixed set has no eligible cue-probe coverage"
    checkpoint = resolve_checkpoint_uri(
        f"r2://ml-data/artifacts/{experiment}/checkpoints/best.ckpt",
        cache_dir="results/checkpoints",
    )
    model = load_campaign_model(cfg, checkpoint, device="cuda")
    result = {
        "experiment": experiment,
        "dataset": "unchanged pilot fixed validation; not the paper's real-data frequency part",
        "conditioning": "existing corruption/row alignment, held fixed while audio changes",
        "checkpoint_sha256": hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
        "frequency": cue["freq_probe"](model, frames, clips, n_ch, list(cue["ALPHAS"])),
        "cutoff": cue["cutoff_probe"](model, frames, selected, list(cue["K_CUTS"]), "fir"),
        "cutoff_protocol": "paper .fir-n16 selector/filter; original K_CUTS",
    }
    out = Path(f"results/jhtr-cues-{stage}")
    out.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, out / "resolved_config.yaml", resolve=True)
    (out / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)
    del model, frames, dataset
