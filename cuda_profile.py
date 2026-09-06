import hashlib
import json
import time
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

from models.jhtr import JHTR


# Disposable copy of the canonical profile loop, with only native construction
# made dependency-light and completed rows printed before any later OOM.
def profile_jhtr(cfg: Any, *, device: str = "cpu", smoke: bool = False) -> dict[str, Any]:
    """Run the actual model forward/backward; no optimizer/schedule/data changes.

    Full profiling uses the inherited batch at 1/4 s training and 8 s validation,
    as the trainer does. CPU smoke uses one 1 s example and is not a matched profile.
    """

    torch.manual_seed(int(cfg.seed))
    model = JHTR(**cfg.model.params).to(device)
    cuda = torch.device(device).type == "cuda"
    if not smoke and not cuda:
        raise ValueError("matched full profile requires a GPU; use --smoke for bounded CPU proof")
    rows = []
    for seconds in (1,) if smoke else (1, 4, 8):
        training = seconds != 8
        batch = 1 if smoke else int(cfg.data.batch_size or cfg.batch_size)
        samples = seconds * 16000
        # Known finite nonzero input, not a new dataset or training regimen.
        audio = torch.randn(batch, samples, device=device) * 0.1
        cond = torch.full((batch, 4, samples // 512 + 1), 80.0, device=device)
        model.train(training)
        model.zero_grad(set_to_none=True)
        if cuda:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        with (
            torch.set_grad_enabled(training),
            torch.autocast(
                device_type=torch.device(device).type,
                enabled=cuda and bool(cfg.amp),
                dtype=torch.bfloat16
                if getattr(cfg, "amp_dtype", "float16") == "bfloat16"
                else torch.float16,
            ),
        ):
            prediction = model(
                audio, cond if bool(cfg.model.task_params.get("use_cond", False)) else None
            )
            objective = prediction.square().mean()
        if cuda:
            torch.cuda.synchronize()
        forward_seconds = time.perf_counter() - start
        if prediction.shape != cond.shape or not torch.isfinite(prediction).all():
            raise AssertionError("model smoke/profile produced invalid output")
        start = time.perf_counter()
        if training:
            objective.backward()
            grads = [p.grad for p in model.parameters() if p.grad is not None]
            if not grads or not all(torch.isfinite(g).all() for g in grads):
                raise AssertionError("model smoke/profile produced absent or nonfinite gradients")
        if cuda:
            torch.cuda.synchronize()
        rows.append(
            {
                "seconds": seconds,
                "batch": batch,
                "training": training,
                "frames": prediction.shape[-1],
                "forward_seconds": forward_seconds,
                "backward_seconds": time.perf_counter() - start if training else None,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated() if cuda else None,
                "peak_reserved_bytes": torch.cuda.max_memory_reserved() if cuda else None,
            }
        )
        print(json.dumps(rows[-1]), flush=True)
        del audio, cond, prediction, objective
    if not all(torch.isfinite(p).all() for p in model.parameters()):
        raise AssertionError("nonfinite model parameters")
    return {
        "device": device,
        "hardware": torch.cuda.get_device_name() if cuda else "CPU",
        "parameters": sum(p.numel() for p in model.parameters()),
        "smoke": smoke,
        "amp": cuda and bool(cfg.amp),
        "timing": "single cold forward/backward; not steady-state latency",
        "profiles": rows,
    }


manifest = json.loads(Path("source_revision.json").read_text())
for name, digest in manifest["sha256"].items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == digest, name
assert torch.cuda.is_available(), "CUDA is required"
raw_cfg = json.loads(Path("profile-config.json").read_text())
assert raw_cfg["model"]["_target_"] == "models.jhtr.JHTR"
assert raw_cfg["model"]["model_type"] is None
raw_cfg["model"] = SimpleNamespace(**raw_cfg["model"])
raw_cfg["data"] = SimpleNamespace(**raw_cfg["data"])
cfg = SimpleNamespace(**raw_cfg)
out = Path("results/kaggle-matched-profile.json")
out.parent.mkdir(parents=True, exist_ok=True)
try:
    result = profile_jhtr(cfg, device="cuda", smoke=False)
except Exception:
    result = {
        "pass": False,
        "error": traceback.format_exc(),
        "hardware": torch.cuda.get_device_name(),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
    }
    out.write_text(json.dumps(result, indent=2))
    raise
result.update(
    {"source_commit": manifest["source_commit"], "pass": True, "torch": torch.__version__}
)
out.write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2), flush=True)
