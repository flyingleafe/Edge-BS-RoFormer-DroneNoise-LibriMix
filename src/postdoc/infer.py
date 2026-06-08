"""`postdoc infer` — run a trained model on audio files.

Takes a path to a *run directory* that contains a `config.yaml` and a
`training/` subdir with checkpoints. Typical sources of that directory:

    * a locally-downloaded wandb artifact
    * an rsync'd output from the remote SkyPilot job
    * a manually organized dir with `config.yaml` + `training/*.ckpt`

Optionally, an `experiment.yaml` alongside `config.yaml` pins the model type;
otherwise we infer it from `config['model']['model']`.
"""

from __future__ import annotations

import glob as _glob
import re
from argparse import Namespace
from pathlib import Path

import typer


def _resolve_run(query: str) -> Path:
    """Return the run directory for ``query``. Must be a real path."""
    p = Path(query).expanduser().resolve()
    if not p.exists():
        raise typer.BadParameter(f"Path does not exist: {query}")
    if not p.is_dir():
        raise typer.BadParameter(f"Not a directory: {query}")
    if not (p / "config.yaml").exists():
        raise typer.BadParameter(f"Missing config.yaml in run dir: {p}")
    return p


def _resolve_checkpoint(run_path: Path, selector: str) -> Path:
    train_dir = run_path / "training"
    if not train_dir.exists():
        raise typer.BadParameter(f"Training dir not found: {train_dir}")
    ckpts = list(train_dir.glob("*.ckpt"))
    if not ckpts:
        raise typer.BadParameter(f"No checkpoints in {train_dir}")

    sel = selector.lower()
    if sel == "best":
        best = [c for c in ckpts if "best" in c.name.lower()]
        if best:
            return best[0]
        sel = "latest"
    if sel == "latest":
        ckpts.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return ckpts[0]

    epoch_m = re.match(r"^(\d+)$", selector)
    if epoch_m:
        num = int(epoch_m.group(1))
        for c in ckpts:
            if re.search(rf"epoch[_-]?{num}[^\d]", c.name, re.IGNORECASE):
                return c
            if re.search(rf"e[_-]?{num}[^\d]", c.name, re.IGNORECASE):
                return c
        for c in ckpts:
            if str(num) in c.stem:
                return c
    # Try substring filename match
    for c in ckpts:
        if selector in c.name:
            return c
    raise typer.BadParameter(
        f"Checkpoint not found: {selector}. Available: {[c.name for c in ckpts]}"
    )


def infer_cmd(
    run_query: str,
    checkpoint: str = "best",
    input: str = "",
    output: str = "",
    device: int = 0,
    rps_file: str | None = None,
) -> None:
    import numpy as np
    import soundfile as sf
    import torch
    import yaml as _yaml
    from ml_collections import ConfigDict

    from utils import (
        demix,
        get_model_from_config,
        load_start_checkpoint,
        read_audio_transposed,
    )

    run_path = _resolve_run(run_query)
    typer.echo(f"Using run: {run_path.name}")

    config_path = run_path / "config.yaml"
    if not config_path.exists():
        typer.echo(f"Resolved config not found: {config_path}")
        raise typer.Exit(1)
    with open(config_path) as f:
        config = _yaml.safe_load(f)

    # Model type from the submitted experiment YAML
    exp_path = run_path / "experiment.yaml"
    if exp_path.exists():
        with open(exp_path) as f:
            exp = _yaml.safe_load(f)
        model_type = exp.get("model", {}).get("type")
    else:
        model_type = config.get("model", {}).get("model")
    if not model_type:
        typer.echo("Could not determine model type.")
        raise typer.Exit(1)

    ckpt_path = _resolve_checkpoint(run_path, checkpoint)
    typer.echo(f"Using checkpoint: {ckpt_path.name}")

    # Resolve inputs
    ipath = Path(input)
    if ipath.is_dir():
        audio_files = sorted(_glob.glob(str(ipath / "*.wav"))) + sorted(
            _glob.glob(str(ipath / "*.flac"))
        )
    elif _glob.has_magic(input):
        audio_files = sorted(_glob.glob(input))
    elif ipath.is_file():
        audio_files = [str(ipath)]
    else:
        typer.echo(f"Input not found: {input}")
        raise typer.Exit(1)
    if not audio_files:
        typer.echo(f"No audio files for input: {input}")
        raise typer.Exit(1)
    typer.echo(f"Processing {len(audio_files)} file(s)...")

    opath = Path(output)
    if len(audio_files) > 1 and (opath.is_file() or opath.suffix):
        opath = opath.parent
    opath.mkdir(parents=True, exist_ok=True)

    device_obj = torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")
    typer.echo(f"Device: {device_obj}")

    model, cfg = get_model_from_config(model_type, str(config_path))
    model = model.to(device_obj)
    fake_args = Namespace(
        start_check_point=str(ckpt_path),
        model_type=model_type,
        lora_checkpoint=None,
    )
    load_start_checkpoint(fake_args, model, type_="infer")
    model.eval()

    use_rps = config.get("use_rps", False)
    predict_rps = config.get("predict_rps", False)
    typer.echo(f"Model: {model_type}  |  RPS cond: {use_rps}  |  RPS pred: {predict_rps}")

    cfg_obj = cfg if isinstance(cfg, ConfigDict) else ConfigDict(cfg)

    for audio_path_str in audio_files:
        audio_path = Path(audio_path_str)
        stem = audio_path.stem

        rps_path: Path | None = None
        if rps_file:
            rps_path = Path(rps_file)
        elif use_rps or predict_rps:
            cand = audio_path.with_suffix(".npy")
            if cand.exists():
                rps_path = cand
            else:
                typer.echo(f"  [WARN] RPS file not found for {stem}")

        rps_data = None
        if rps_path and rps_path.exists():
            rps_data = np.load(str(rps_path))
            typer.echo(f"  RPS: {rps_path.name} {rps_data.shape}")

        mix, sr = read_audio_transposed(str(audio_path))
        mix_tensor = torch.from_numpy(mix).float()
        typer.echo(f"  {audio_path.name}  ({mix.shape[-1] / sr:.2f}s)")  # pyright: ignore[reportOptionalMemberAccess]

        with (
            torch.no_grad(),
            torch.cuda.amp.autocast(enabled=getattr(cfg_obj.training, "use_amp", True)),
        ):
            result = demix(
                cfg_obj,
                model,
                mix_tensor,
                device_obj,
                model_type=model_type,
                rps=rps_data,
            )
        if isinstance(result, dict):
            for instr, audio in result.items():
                out_wav = opath / f"{stem}_out_{instr}.wav"
                sf.write(str(out_wav), audio.T, sr)
                typer.echo(f"    -> {out_wav.name}")
        else:
            out_wav = opath / f"{stem}_out.wav"
            sf.write(str(out_wav), result.T, sr)
            typer.echo(f"    -> {out_wav.name}")

        if rps_data is not None:
            out_rps = opath / f"{stem}_out_rps.npy"
            np.save(str(out_rps), rps_data)
            typer.echo(f"    -> {out_rps.name}")

    typer.echo(f"\nDone. Outputs in: {opath}")
