"""Per-drone conditioning codebook + geometry helper for the noise generators.

Canonical home of :class:`DroneCodebook` and :func:`geometry_to_rel_pos`
(moved from ``tasks.noise_generation`` in the 2026-08 refactor, so that
``data_processing`` and ``models`` consumers do not import the task layer;
``tasks.noise_generation`` re-exports both names).
"""

from __future__ import annotations

import numpy as np
import torch

# ── Geometry → model input ──────────────────────────────────────────────────


def geometry_to_rel_pos(
    mic_positions: np.ndarray | torch.Tensor,
    rotor_positions: np.ndarray | torch.Tensor,
) -> np.ndarray | torch.Tensor:
    """Build per-(mic, rotor) relative position vectors.

    Two call shapes, dispatched on argument type:

    - **numpy, unbatched** (single recording's geometry): ``mic_positions
      (M, 3)``, ``rotor_positions (R, 3)`` -> ``(M, R, 3)`` float32 numpy
      array. The original single-recording contract (report/notebook figure
      scripts, ``data_processing.generated_noise``).
    - **torch, batched** (a training batch, one geometry per sample):
      ``mic_positions (B, M, 3)``, ``rotor_positions (B, R, 3)`` -> ``(B, M,
      R, 3)`` tensor, same dtype/device as the inputs. Used by
      :class:`tasks.codecs.NoiseGenerationCodec` — geometry arrives as a
      batched Frame entry (``tasks.task.noise_generation``'s input spec),
      and building the relative-position tensor via plain broadcasting
      (rather than round-tripping through numpy) keeps it differentiable
      and on-device.

    In both cases: ``rel_pos[..., m, r, :] = mic[..., m, :] - rotor[..., r, :]``
    — the vector from rotor ``r`` to mic ``m`` that the generator propagates
    along.
    """
    if isinstance(mic_positions, torch.Tensor) or isinstance(rotor_positions, torch.Tensor):
        mic_t = torch.as_tensor(mic_positions)
        rotor_t = torch.as_tensor(rotor_positions)
        if mic_t.shape[-1] != 3:
            raise ValueError(f"mic_positions must end in dim 3 (xyz), got {tuple(mic_t.shape)}")
        if rotor_t.shape[-1] != 3:
            raise ValueError(f"rotor_positions must end in dim 3 (xyz), got {tuple(rotor_t.shape)}")
        if mic_t.dim() == 3 and rotor_t.dim() == 3:
            return mic_t.unsqueeze(2) - rotor_t.unsqueeze(1)  # (B, M, R, 3)
        if mic_t.dim() == 2 and rotor_t.dim() == 2:
            return mic_t.unsqueeze(1) - rotor_t.unsqueeze(0)  # (M, R, 3)
        raise ValueError(
            "mic_positions/rotor_positions must both be (M,3)/(R,3) (unbatched) or "
            f"(B,M,3)/(B,R,3) (batched); got {tuple(mic_t.shape)} and {tuple(rotor_t.shape)}"
        )

    mic = np.asarray(mic_positions, dtype=np.float64)
    rotor = np.asarray(rotor_positions, dtype=np.float64)
    if mic.ndim != 2 or mic.shape[-1] != 3:
        raise ValueError(f"mic_positions must be (M, 3), got {mic.shape}")
    if rotor.ndim != 2 or rotor.shape[-1] != 3:
        raise ValueError(f"rotor_positions must be (R, 3), got {rotor.shape}")
    return (mic[:, None, :] - rotor[None, :, :]).astype(np.float32)  # (M, R, 3)


# ── Per-drone conditioning codebook (external, name-keyed) ──────────────────


class DroneCodebook(torch.nn.Module):
    """Name-keyed table of learnable per-drone conditioning codes.

    Deliberately **decoupled** from the generator. The model takes a code ``z``
    ``(B, d)`` as an input (like geometry); this owns the ``drone_name -> z``
    map. Keeping it external means:

    * the generator's parameter shape is **fixed** regardless of how many drones
      exist — adding a drone never resizes model weights;
    * keys are **names**, not positional indices, so adding/removing a drone
      never disturbs existing codes and there is no index drift between datasets
      (``load_state_dict(strict=False)`` loads the intersection by name);
    * **few-shot adaptation** to an unseen drone = freeze the generator, add a
      fresh code here, and optimise just that ``d``-vector.

    ``d`` is fixed by the generator (it sizes the FiLM generator), so build the
    codebook with the same ``dim``.
    """

    def __init__(
        self,
        dim: int,
        names: list[str] | tuple[str, ...] = (),
        *,
        init_std: float = 0.01,
    ):
        super().__init__()
        self.dim = int(dim)
        self.init_std = float(init_std)
        self.codes = torch.nn.ParameterDict()
        for name in names:
            self.add(name)

    @staticmethod
    def _key(name: str) -> str:
        # nn.ParameterDict keys are state_dict path components, so '.' is unsafe.
        key = str(name)
        if "." in key:
            raise ValueError(f"drone name must not contain '.': {name!r}")
        if not key:
            raise ValueError("drone name must be a non-empty string")
        return key

    def add(self, name: str, init: torch.Tensor | None = None) -> torch.nn.Parameter:
        """Register a drone by name (idempotent). Returns its code parameter.

        ``init`` warm-starts the code (e.g. from a nearby known drone); otherwise
        it is small-random so a fresh drone starts near the FiLM near-identity.
        """
        key = self._key(name)
        if key in self.codes:
            return self.codes[key]
        if init is None:
            vec = torch.randn(self.dim) * self.init_std
        else:
            vec = init.detach().clone().reshape(-1)
            if vec.shape[-1] != self.dim:
                raise ValueError(f"init must have dim {self.dim}, got {tuple(init.shape)}")
        param = torch.nn.Parameter(vec)
        self.codes[key] = param
        return param

    def names(self) -> list[str]:
        return list(self.codes.keys())

    def __contains__(self, name: str) -> bool:
        return self._key(name) in self.codes

    def forward(self, names: list[str] | tuple[str, ...]) -> torch.Tensor:
        """Look up a batch of codes by name -> ``(B, d)`` (gradient-tracked)."""
        missing = [n for n in names if self._key(n) not in self.codes]
        if missing:
            raise KeyError(f"unknown drone(s) {missing}; known: {self.names()}")
        return torch.stack([self.codes[self._key(n)] for n in names], dim=0)


__all__ = ["DroneCodebook", "geometry_to_rel_pos"]
