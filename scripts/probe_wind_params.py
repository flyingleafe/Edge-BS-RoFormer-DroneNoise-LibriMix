"""Report what the wind-wake channel actually learned in a trained checkpoint.

The wind arm (`gen_w4_lik_wind_mm`) scored WORSE than its no-wind control on
both drones, and — diagnostically — worse on Michael's, whose array sits out of
the wake where the gate is ~7500x weaker and the channel should be inert. Two
very different failures produce that:

- the channel learned a LARGE level and is injecting variance where the geometry
  says it should not (the wake model is wrong), or
- it stayed inert and merely perturbed the optimization of the shared coherent
  generator (the wake model is untested, the experiment is confounded).

This prints the learned scalars so the two can be told apart.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from models.generative.wind_wake_gen import _pos  # noqa: E402
from training.artifacts import resolve_checkpoint_uri  # noqa: E402

DEFAULT_CKPT = "r2://ml-data/artifacts/gen_w4_lik_wind_mm/checkpoints/best.ckpt"


def main() -> None:
    uri = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CKPT
    state = torch.load(resolve_checkpoint_uri(uri), map_location="cpu")
    for key in ("state_dict", "model"):
        if isinstance(state, dict) and key in state:
            state = state[key]
    print(f"checkpoint: {uri}")
    print("=== learned wind scalars ===")
    for name, value in sorted(state.items()):
        if ".wind." not in name or value.numel() > 4:
            continue
        raw = float(value.reshape(-1)[0])
        print(f"  {name:56s} raw={raw:+9.4f}  softplus={float(_pos(torch.tensor(raw))):10.5f}")
    shapes = {n: tuple(v.shape) for n, v in state.items() if ".wind." in n and v.numel() > 4}
    if shapes:
        print("=== larger wind tensors (shape, |mean|) ===")
        for name, shape in sorted(shapes.items()):
            print(f"  {name:56s} {str(shape):16s} {float(state[name].abs().mean()):.6f}")


if __name__ == "__main__":
    main()
