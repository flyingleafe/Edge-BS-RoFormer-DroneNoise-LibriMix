"""Unified evaluation entry point (docs/refactor-unified-framework.md §
"train.py / eval.py"). Composition + dispatch only.

Loads a checkpoint (``cfg.checkpoint`` or
``results/<experiment_name>/best.ckpt``), runs the configured
:class:`~metrics.suite.MetricSuite` over ``cfg.data.valid``, and writes
``results/<experiment_name>/eval/{metrics.json,per_sample.csv,per_snr.csv}``
(the last only when any sample carries ``meta.input_snr``).

Examples::

    python eval.py experiment=rps_simple_conv_v2_v4
    python eval.py experiment=rps_simple_conv_v2_v4 \\
        checkpoint=results/rps_simple_conv_v2_v4/ep12_mse_3.1200.ckpt
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import hydra
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from data_processing.collate import batch_size as frame_batch_size
from data_processing.collate import frame_collate, slice_sample
from training.artifacts import ArtifactStore
from training.config import (
    build_dataset,
    build_metrics,
    build_task_and_codec,
    instantiate_model,
    register_configs,
)
from training.validate import validate_config
from utils.checkpoints import resolve_checkpoint_uri

register_configs()


def _checkpoint_path(cfg: DictConfig) -> Path:
    if cfg.checkpoint:
        # ``r2://`` URIs are downloaded (once, cached) to a local path;
        # plain paths pass through unchanged.
        return Path(resolve_checkpoint_uri(str(cfg.checkpoint)))
    return Path(cfg.results_root) / cfg.experiment_name / "best.ckpt"


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    problems = validate_config(cfg)
    if problems:
        print(f"Config validation found {len(problems)} problem(s):")
        for p in problems:
            print(f"  - {p}")
        raise SystemExit("Refusing to evaluate: pipeline failed validation (see problems above).")

    ckpt_path = _checkpoint_path(cfg)
    if not ckpt_path.is_file():
        raise FileNotFoundError(
            f"checkpoint not found: {ckpt_path} (pass checkpoint=<path> or train first)"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _task, codec = build_task_and_codec(cfg.model)
    model = instantiate_model(cfg.model).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()

    metric_suite = build_metrics(cfg.metrics)
    valid_ds = build_dataset(cfg.data.valid)
    batch_size = cfg.data.batch_size or cfg.batch_size
    num_workers = cfg.data.num_workers if cfg.data.num_workers is not None else cfg.num_workers
    loader = DataLoader(
        valid_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=frame_collate,
    )

    pairs = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.map_data(lambda t: t.to(device))
            inputs = codec.to_inputs(batch)
            outputs = codec.call_model(model, inputs)
            pred_frame = codec.to_frame(outputs, batch)
            pred_cpu = pred_frame.map_data(lambda t: t.detach().cpu())
            batch_cpu = batch.map_data(lambda t: t.detach().cpu())
            for i in range(frame_batch_size(pred_cpu)):
                pairs.append((slice_sample(pred_cpu, i), slice_sample(batch_cpu, i)))

    result = metric_suite.evaluate(pairs, group_by="input_snr")
    metric_names = list(metric_suite.metrics)
    means = result.aggregate("mean")

    out_dir = Path(cfg.results_root) / cfg.experiment_name / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "metrics.json").write_text(json.dumps(means, indent=2))

    with open(out_dir / "per_sample.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["sample", *metric_names, "input_snr"])
        writer.writeheader()
        for i, row in enumerate(result.rows):
            writer.writerow({"sample": i, **row})

    per_snr_written = False
    if any(row.get("input_snr") is not None for row in result.rows):
        grouped = result.grouped("mean")
        with open(out_dir / "per_snr.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["input_snr", *metric_names])
            writer.writeheader()
            for snr in sorted(grouped, key=lambda k: (k is None, k)):
                writer.writerow({"input_snr": snr, **grouped[snr]})
        per_snr_written = True

    # Publish eval outputs next to the experiment's checkpoints on R2
    # (``artifacts/<experiment>/eval/``) — the zoo cache harvests them, so
    # metrics travel with the artifact store. Same defensive store as
    # training: a no-op without creds / with artifacts.enabled=false.
    store: Any = ArtifactStore(
        experiment_name=cfg.experiment_name,
        bucket=cfg.artifacts.bucket,
        prefix=cfg.artifacts.prefix,
        enabled=cfg.artifacts.enabled,
    )
    uploaded = store.upload_file(out_dir / "metrics.json", "eval/metrics.json")
    if per_snr_written:
        store.upload_file(out_dir / "per_snr.csv", "eval/per_snr.csv")
    if uploaded:
        print(f"Eval metrics uploaded to {uploaded}")

    print(f"Eval results written to {out_dir}")
    print(json.dumps(means, indent=2))


if __name__ == "__main__":
    main()
