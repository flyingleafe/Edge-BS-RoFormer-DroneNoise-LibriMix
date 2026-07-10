"""Generated-noise source pool: a trained noise-gen model as a data source.

This is *option C* of the design: a single background **producer process**
(the only extra CUDA context) continuously renders drone-noise chunks with a
trained ``PositionalHarmonicNoiseGen`` and writes them into a **shared-memory
ring buffer**; the ordinary CPU ``DataLoader`` workers read finished chunks out
of that buffer with no GPU access of their own.

Why a separate process (not GPU-inside-``sample_timeframe``): the online-mixing
dataset runs inside forked ``DataLoader`` workers, and CUDA cannot be
re-initialised in a forked subprocess. A dedicated **spawn** producer owns one
CUDA context; the **fork** workers only ever touch CPU shared memory. See the
handoff/design discussion for the full rationale.

Key properties:

* **Rate-decoupled.** Workers sample-with-replacement from whatever slots are
  currently filled; if the GPU can't keep up, chunks are simply reused (epoch
  reuse of an augmentation source — harmless). No backpressure needed.
* **Lock-free reads (seqlock).** Each slot carries an int64 ``version``: the
  producer bumps it odd before writing and even after. A reader retries if the
  version is odd or changed across its copy, so it never observes a torn chunk —
  no per-slot mutex in the hot path (which would be awkward across the
  spawn-producer / fork-worker process split anyway).
* **Excitation == label.** The synthetic RPS trajectory that drives the
  generator is stored next to the audio, so for RPS-prediction training the
  generated noise comes with an exact, noise-free RPS label.
* **Deterministic-bank mode.** With ``refresh: false`` the producer fills the
  buffer once (fixed seeds) and stops — a reproducible fixed generated set,
  usable where a live stream would break reproducibility.

The pool implements the same ``sample_timeframe(rng, duration_s) -> td.Frame``
interface as :class:`data_processing.online_mixing.TimeFrameNoisePool`, so
``kind: generated`` slots into an online-mix ``sources.noise`` list exactly like
a real recording.
"""

from __future__ import annotations

import atexit
import contextlib
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import tdseries as td
import torch
import torch.multiprocessing as tmp

from data_processing import rps_synthesis
from data_processing.frames import make_recording_frame

# Convenience blend factor for `rps_synthesis.generate_intermittent(drone_profile=...)`
# keyed by codebook drone name. Unknown names default to the DREGON end (0.0).
_DRONE_PROFILE_BLEND = {"dregon": 0.0, "michaels": 1.0}

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _ensure_src_on_path() -> None:
    """Make ``models``/``tasks`` importable in a spawned child interpreter."""
    src = str(_REPO_ROOT / "src")
    for p in (str(_REPO_ROOT), src):
        if p not in sys.path:
            sys.path.insert(0, p)


def load_geometry(
    drone: str, dregon_dir: str | Path = "data/DREGON"
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(mic_positions (M,3), rotor_positions (R,3))`` for a drone name."""
    _ensure_src_on_path()
    if drone == "michaels":
        from data_processing.michaels import get_geometry as _mich

        return _mich()
    from data_processing.dregon import get_geometry as _dreg
    from data_processing.streams import resolve_source

    # `dregon_dir` may be a plain path or a `dload:NAME[/sub]` URI.
    return _dreg(resolve_source(dregon_dir))


def _build_generator(params: dict[str, Any], device: str):
    """Load the model + codebook bundle and validate the requested drone."""
    _ensure_src_on_path()
    from models.generative import PositionalHarmonicNoiseGen
    from tasks.noise_generation import DroneCodebook

    bundle = torch.load(params["checkpoint"], map_location=device, weights_only=False)
    cond_dim = int(bundle["cond_dim"])
    model = PositionalHarmonicNoiseGen(
        sample_rate=params["sample_rate"],
        n_harmonics=params["n_harmonics"],
        use_diff_noise=not params["no_diff_noise"],
        cond_dim=cond_dim,
    )
    model.load_state_dict(bundle["model"])
    model.to(device).eval()  # eval => deterministic modules; phase variety is explicit below
    names = list(bundle["drone_names"])
    if params["drone"] not in names:
        raise ValueError(
            f"drone {params['drone']!r} not in checkpoint codebook {names}; "
            "the generated source must name a drone the checkpoint was trained on"
        )
    codebook = DroneCodebook(cond_dim, names=names).to(device)
    codebook.load_state_dict(bundle["codebook"])
    return model, codebook


def _producer_loop(shared: dict[str, torch.Tensor], params: dict[str, Any]) -> None:
    """Background process: render batches on ``device`` into the shared ring.

    Runs until ``shared['run'][0]`` is cleared (live mode) or the buffer has been
    filled once (deterministic ``refresh: false`` mode).
    """
    _ensure_src_on_path()
    torch.set_num_threads(1)
    from tasks.noise_generation import geometry_to_rel_pos

    device = params["device"]
    try:
        model, codebook = _build_generator(params, device)
    except Exception as exc:  # surface the cause; readers time out on an empty buffer
        print(f"[generated-noise producer] failed to start: {exc}", file=sys.stderr)
        raise

    mic_pos, rotor_pos = load_geometry(params["drone"], params["dregon_dir"])
    rel = torch.from_numpy(geometry_to_rel_pos(mic_pos, rotor_pos)).float().to(device)  # (M,R,3)
    n_rotors = rotor_pos.shape[0]
    n_harm = params["n_harmonics"]

    audio_buf = shared["audio"]
    rps_buf = shared["rps"]
    version = shared["version"]
    ready = shared["ready"]
    write_pos = shared["write_pos"]
    filled = shared["filled"]
    run = shared["run"]

    n_slots = audio_buf.shape[0]
    gen_bs = int(params["gen_batch"])
    duration_s = params["chunk_s"]
    sr = params["sample_rate"]
    rng = np.random.default_rng(params["seed"])
    blend = _DRONE_PROFILE_BLEND.get(params["drone"], 0.0)

    while bool(run[0].item()):
        rps_np = rps_synthesis.generate_intermittent_batch(
            gen_bs,
            duration_s,
            sr,
            drone_profile=blend,
            aggressiveness=params["aggressiveness"],
            rng=rng,
        )  # (bs, R, T)
        rps_t = torch.from_numpy(rps_np).float().to(device)
        rel_b = rel.unsqueeze(0).expand(gen_bs, -1, -1, -1)
        z = codebook([params["drone"]] * gen_bs).to(device)
        init_phase = None
        if params["random_phase"]:
            # Per-chunk random harmonic phases decorrelate the waveform texture
            # even at identical RPS — extra augmentation variety (model stays in
            # eval, so this is the ONLY stochastic knob).
            init_phase = (
                torch.from_numpy(rng.uniform(0.0, 2.0 * np.pi, size=(gen_bs, n_rotors, n_harm)))
                .float()
                .to(device)
            )
        with torch.no_grad():
            audio = model(rps_t, rel_b, z, initial_phases=init_phase).cpu()  # (bs, M, T)

        for b in range(gen_bs):
            i = int(write_pos[0].item())
            write_pos[0] = (i + 1) % n_slots
            v = int(version[i].item())
            version[i] = v + 1  # odd => writing
            audio_buf[i].copy_(audio[b])
            rps_buf[i].copy_(torch.from_numpy(rps_np[b]))
            version[i] = v + 2  # even => complete
            ready[i] = 1
            filled[0] = min(int(filled[0].item()) + 1, n_slots)

        if not params["refresh"] and int(filled[0].item()) >= n_slots:
            break  # deterministic bank: fill once, then idle


class GeneratedNoisePool:
    """Trained noise generator exposed as a ``sample_timeframe`` noise source."""

    def __init__(
        self,
        checkpoint: str | Path,
        drone: str,
        *,
        sample_rate: int = 16000,
        duration_s: float = 1.0,
        n_harmonics: int = 100,
        no_diff_noise: bool = False,
        aggressiveness: float = 1.0,
        random_phase: bool = True,
        n_slots: int = 512,
        gen_batch: int = 32,
        warmup: int = 16,
        refresh: bool = True,
        device: str = "cuda:0",
        dregon_dir: str | Path = "data/DREGON",
        seed: int = 0,
        warmup_timeout_s: float = 60.0,
    ):
        self.drone = str(drone)
        self.sample_rate = int(sample_rate)
        self.chunk_s = float(duration_s)
        self.chunk_len = int(round(self.chunk_s * self.sample_rate))
        self.warmup = int(warmup)
        self.warmup_timeout_s = float(warmup_timeout_s)

        self.mic_pos, self.rotor_pos = load_geometry(self.drone, dregon_dir)
        n_mics, n_rotors = self.mic_pos.shape[0], self.rotor_pos.shape[0]

        # Shared ring buffer (fork workers inherit these; the spawn producer gets
        # them via torch's shared-memory reduction). All time-last (…, T).
        self.shared: dict[str, torch.Tensor] = {
            "audio": torch.zeros(n_slots, n_mics, self.chunk_len).share_memory_(),
            "rps": torch.zeros(n_slots, n_rotors, self.chunk_len).share_memory_(),
            "version": torch.zeros(n_slots, dtype=torch.int64).share_memory_(),
            "ready": torch.zeros(n_slots, dtype=torch.uint8).share_memory_(),
            "write_pos": torch.zeros(1, dtype=torch.int64).share_memory_(),
            "filled": torch.zeros(1, dtype=torch.int64).share_memory_(),
            "run": torch.ones(1, dtype=torch.uint8).share_memory_(),
        }
        self._params: dict[str, Any] = {
            "checkpoint": str(checkpoint),
            "drone": self.drone,
            "sample_rate": self.sample_rate,
            "chunk_s": self.chunk_s,
            "n_harmonics": int(n_harmonics),
            "no_diff_noise": bool(no_diff_noise),
            "aggressiveness": float(aggressiveness),
            "random_phase": bool(random_phase),
            "gen_batch": int(gen_batch),
            "refresh": bool(refresh),
            "device": str(device),
            "dregon_dir": str(dregon_dir),
            "seed": int(seed),
        }
        self._proc: Any = None
        self._owner_pid = None
        self._start_producer()
        atexit.register(self.close)

    # -- lifecycle ---------------------------------------------------------------

    def _start_producer(self) -> None:
        # Only the process that built the pool starts the producer; forked
        # DataLoader workers inherit an already-running one and must not spawn
        # their own.
        import os

        ctx = tmp.get_context("spawn")
        self._proc = ctx.Process(
            target=_producer_loop, args=(self.shared, self._params), daemon=True
        )
        self._proc.start()
        self._owner_pid = os.getpid()

    @classmethod
    def from_config(cls, cfg: Any, *, duration_s: float, sample_rate: int) -> GeneratedNoisePool:
        def g(key, default=None):
            if isinstance(cfg, dict):
                return cfg.get(key, default)
            return getattr(cfg, key, default)

        rps = g("rps", {}) or {}
        rps_kind = rps.get("kind", "synthetic_intermittent")
        if rps_kind != "synthetic_intermittent":
            raise ValueError(
                f"generated noise supports rps.kind 'synthetic_intermittent' only, got {rps_kind!r}"
            )
        buf = g("buffer", {}) or {}
        checkpoint = g("checkpoint")
        if not checkpoint:
            raise ValueError("generated noise source requires 'checkpoint'")
        return cls(
            checkpoint=checkpoint,
            drone=g("drone", "dregon"),
            sample_rate=sample_rate,
            duration_s=duration_s,
            n_harmonics=int(g("n_harmonics", 100)),
            no_diff_noise=bool(g("no_diff_noise", False)),
            aggressiveness=float(rps.get("aggressiveness", 1.0)),
            random_phase=bool(g("random_phase", True)),
            n_slots=int(buf.get("slots", 512)),
            gen_batch=int(g("gen_batch", 32)),
            warmup=int(buf.get("warmup", 16)),
            refresh=bool(g("refresh", True)),
            device=str(g("device", "cuda:0")),
            dregon_dir=str(g("dregon_dir", "data/DREGON")),
            seed=int(g("seed", 0)),
        )

    def close(self) -> None:
        """Stop the producer and let the OS reclaim the shared buffers."""
        import os

        if self._proc is None or self._owner_pid != os.getpid():
            return  # never tear down from a forked worker
        self.shared["run"][0] = 0
        self._proc.join(timeout=5.0)
        if self._proc.is_alive():
            self._proc.terminate()
        self._proc = None

    def __del__(self):
        with contextlib.suppress(Exception):
            self.close()

    # -- sampling ----------------------------------------------------------------

    def _wait_warmup(self) -> np.ndarray:
        deadline = time.time() + self.warmup_timeout_s
        while True:
            ready_idx = np.flatnonzero(self.shared["ready"].numpy())
            if ready_idx.size >= min(self.warmup, self.shared["ready"].shape[0]):
                return ready_idx
            if time.time() > deadline:
                if ready_idx.size > 0:
                    return ready_idx  # partially warm is fine
                alive = self._proc is not None and self._proc.is_alive()
                raise RuntimeError(
                    "generated-noise buffer never warmed up "
                    f"(producer alive={alive}); check the checkpoint/device/geometry"
                )
            time.sleep(0.02)

    def sample_timeframe(self, rng: np.random.Generator, duration_s: float) -> td.Frame:
        ready_idx = self._wait_warmup()
        version = self.shared["version"]
        while True:
            i = int(rng.choice(ready_idx))
            v0 = int(version[i].item())
            if v0 & 1:
                continue  # producer mid-write
            audio = self.shared["audio"][i].clone().numpy()  # (M, T)
            rps = self.shared["rps"][i].clone().numpy()  # (R, T)
            if int(version[i].item()) == v0:
                break  # seqlock: slot unchanged across the copy

        target_len = int(round(duration_s * self.sample_rate))
        if audio.shape[-1] > target_len:
            off = int(rng.integers(0, audio.shape[-1] - target_len + 1))
            audio = audio[..., off : off + target_len]
            rps = rps[..., off : off + target_len]
        elif audio.shape[-1] < target_len:
            pad = target_len - audio.shape[-1]
            audio = np.pad(audio, ((0, 0), (0, pad)))
            rps = np.pad(rps, ((0, 0), (0, pad)))

        audio_us = td.uniform(
            np.ascontiguousarray(audio, dtype=np.float32),
            self.sample_rate,
            dims=("mic", "time"),
            t_start=0.0,
        )
        t = np.arange(audio.shape[-1], dtype=np.float64) / self.sample_rate
        rps_es = td.events(
            t, np.ascontiguousarray(rps, dtype=np.float32), dims=("rotor", "time"), t_start=0.0
        )
        return make_recording_frame(
            {"audio": audio_us, "rps": rps_es},
            meta={"recording_id": f"generated_{self.drone}"},
            mic_pos=self.mic_pos,
            rotor_pos=self.rotor_pos,
        )
