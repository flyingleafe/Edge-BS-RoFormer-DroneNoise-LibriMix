"""Thin JHTR recipe checks, hardware proof and selected-checkpoint diagnostics.

No trainer, scheduler, checkpoint selector or data generator lives here. Run
from the repository root; --help does not instantiate data or import torch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser(
        "check", help="Strict recipe parity and optional full fixed-data parity"
    )
    check.add_argument("--parent", required=True)
    check.add_argument("--experiment", required=True)
    check.add_argument("--conditional-bridge", action="store_true")
    check.add_argument(
        "--data",
        action="store_true",
        help="Independently instantiate all 256/paired512 examples three times",
    )
    check.add_argument(
        "--parent-config",
        type=Path,
        help="Saved as-run parent config instead of current composition",
    )
    check.add_argument("--config", type=Path, help="Saved as-run child config")
    check.add_argument("--out", type=Path, required=True)
    profile = commands.add_parser(
        "profile", help="CPU smoke or GPU inherited-batch 1/4s train and 8s eval profile"
    )
    profile.add_argument("--experiment", default="jhtr_cond_s1_nomix")
    profile.add_argument("--config", type=Path)
    profile.add_argument("--device", default="cpu")
    profile.add_argument(
        "--smoke", action="store_true", help="One 1s example; not matched experimental exposure"
    )
    profile.add_argument("--out", type=Path, required=True)
    evaluate = commands.add_parser(
        "evaluate", help="Measure one selected local checkpoint on unchanged fixed validation"
    )
    evaluate.add_argument("--experiment", required=True)
    evaluate.add_argument(
        "--checkpoint",
        required=True,
        help="Local best.ckpt or established checkpoint URI; never auto-selected by probes",
    )
    evaluate.add_argument("--config", type=Path, help="Saved resolved as-run config (recommended)")
    evaluate.add_argument("--device", default="cpu")
    evaluate.add_argument(
        "--batch", type=int, default=1, help="Inference batch only; training batch is unchanged"
    )
    evaluate.add_argument(
        "--n", type=int, help="Truncate ONLY for explicitly labeled smoke, not scientific results"
    )
    evaluate.add_argument(
        "--cases",
        nargs="+",
        help="Default all: standard/oracle/offset+0.5/.../locator and capture probes",
    )
    evaluate.add_argument(
        "--groups",
        type=Path,
        help="JSON list of original recording/flight IDs in fixed frame order",
    )
    evaluate.add_argument(
        "--observable", type=Path, help="NPZ mask N,R,T with certified existing-source evidence"
    )
    evaluate.add_argument(
        "--observable-provenance",
        help="Path/reference documenting source order powers, floors, separation and duration",
    )
    evaluate.add_argument("--out", type=Path, required=True)
    compare = commands.add_parser(
        "compare", help="Paired 10,000 original-group bootstrap of two selected runs"
    )
    compare.add_argument("--reference", type=Path, required=True)
    compare.add_argument("--candidate", type=Path, required=True)
    compare.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    from experiments import refiner_bench as bench

    if args.command == "compare":
        result = bench.compare_jhtr(args.reference, args.candidate, out_path=args.out)
    else:
        cfg = bench.campaign_config(args.experiment, config_path=args.config)
        if args.command == "check":
            parent = bench.campaign_config(args.parent, config_path=args.parent_config)
            result = {
                "recipe": bench.check_recipe_parity(
                    parent, cfg, conditional_bridge=args.conditional_bridge
                )
            }
            if args.data and result["recipe"]["pass"]:
                result["data"] = bench.check_fixed_frame_parity(cfg)
        elif args.command == "profile":
            result = bench.profile_jhtr(cfg, device=args.device, smoke=args.smoke)
        else:
            import numpy as np
            from omegaconf import OmegaConf

            from training.config import build_dataset

            dataset = build_dataset(cfg.data.valid)
            dataset_size = bench.fixed_dataset_size(dataset)
            if args.n is not None and not 0 < args.n <= dataset_size:
                parser.error("--n must be positive and no larger than the fixed validation set")
            frames = [dataset[i] for i in range(dataset_size if args.n is None else args.n)]
            groups = json.loads(args.groups.read_text()) if args.groups else None
            if groups is not None and args.n is not None:
                groups = groups[: args.n]
            observable = None
            if args.observable:
                with np.load(args.observable) as evidence:
                    observable = evidence["mask"].astype(bool)
                if args.n is not None:
                    observable = observable[: args.n]
            model = bench.load_campaign_model(cfg, args.checkpoint, device=args.device)
            args.out.mkdir(parents=True, exist_ok=True)
            OmegaConf.save(cfg, args.out / "resolved_config.yaml", resolve=True)
            params = cfg.data.valid.params
            independent = params.get("flight_reuse") == 1 and str(cfg.data.valid._target_).endswith(
                ("FixedSynthFrameDataset", "SpeechPairedSynthValidDataset")
            )
            strata = {}
            if str(cfg.data.valid._target_).endswith("SpeechPairedSynthValidDataset"):
                positions = np.arange(len(frames))
                strata = {
                    "speech:false": positions < dataset_size // 2,
                    "speech:true": positions >= dataset_size // 2,
                }
            result = bench.evaluate_jhtr(
                args.experiment,
                frames,
                out_dir=args.out,
                checkpoint=args.checkpoint,
                model=model,
                device=args.device,
                batch=args.batch,
                group_ids=groups,
                independent_sample_ids=independent,
                cases=tuple(args.cases) if args.cases else None,
                observable=observable,
                observable_provenance=args.observable_provenance,
                smoke=args.n is not None,
                sample_strata=strata,
            )
            checkpoint_path = Path(args.checkpoint)
            if checkpoint_path.is_file():
                with checkpoint_path.open("rb") as source:
                    result["checkpoint_sha256"] = hashlib.file_digest(source, "sha256").hexdigest()
            lock = Path("dload.lock")
            result["dload_lock_sha256"] = (
                hashlib.sha256(lock.read_bytes()).hexdigest() if lock.exists() else None
            )
            result["exposure"] = (
                "Record completed epochs and executed updates/examples from train_state.pt plus logs; configured ceilings are not observed exposure. Disclose truncation/replayed epochs."
            )
            (args.out / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    if args.command not in ("evaluate", "compare"):
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(result, indent=2, allow_nan=False))
    return 1 if args.command == "check" and not result["recipe"]["pass"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
