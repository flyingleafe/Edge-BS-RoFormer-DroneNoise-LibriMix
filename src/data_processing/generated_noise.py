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


def _rps_excitation_batch(
    rps_kind: str,
    gen_bs: int,
    duration_s: float,
    sr: int,
    *,
    drone_profile: float,
    aggressiveness: float,
    rng: np.random.Generator,
    flight_fs: float = 200.0,
) -> np.ndarray:
    """RPS excitation for one producer batch → ``(gen_bs, R, T)`` at audio rate.

    ``synthetic_intermittent`` = cruise-only (each item a fresh hover trajectory);
    ``full_flight`` generates ONE full flight (ground→warm-up→takeoff→cruise→
    landing→ground) at a modest ``flight_fs`` and windows ``gen_bs`` slices from
    it, so the batch spans the low-/zero-RPS regimes (the generator can then be
    driven into silence at zero RPS — provided it was trained on those regions).
    """
    if rps_kind == "synthetic_intermittent":
        return rps_synthesis.generate_intermittent_batch(
            gen_bs,
            duration_s,
            sr,
            drone_profile=drone_profile,
            aggressiveness=aggressiveness,
            rng=rng,
        )
    if rps_kind != "full_flight":
        raise ValueError(f"unsupported rps.kind {rps_kind!r}")
    flight = rps_synthesis.generate_full_flight(
        None, flight_fs, drone_profile=drone_profile, aggressiveness=aggressiveness, rng=rng
    )  # (R, Nlow)
    r_n, n_low = flight.shape
    t_low = np.arange(n_low) / flight_fs
    total_s = float(t_low[-1])
    n_t = int(round(duration_s * sr))
    out = np.empty((gen_bs, r_n, n_t), dtype=np.float32)
    max_start = max(0.0, total_s - duration_s)
    for b in range(gen_bs):
        start = float(rng.uniform(0.0, max_start)) if max_start > 0 else 0.0
        t_win = start + np.arange(n_t) / sr
        out[b] = np.stack([np.interp(t_win, t_low, flight[r]) for r in range(r_n)])
    return out


_REPO_ROOT = Path(__file__).resolve().parent.parent


def _ensure_src_on_path() -> None:
    """Make ``models``/``tasks`` importable in a spawned child interpreter."""
    src = str(_REPO_ROOT / "src")
    for p in (str(_REPO_ROOT), src):
        if p not in sys.path:
            sys.path.insert(0, p)


def _to_plain(cfg: Any) -> Any:
    """Convert an OmegaConf node (or nested dict/list) to plain Python
    containers so it pickles cleanly into the spawn producer process."""
    try:
        from omegaconf import DictConfig, ListConfig, OmegaConf

        if isinstance(cfg, (DictConfig, ListConfig)):
            return OmegaConf.to_container(cfg, resolve=True)
    except Exception:
        pass
    if isinstance(cfg, dict):
        return {k: _to_plain(v) for k, v in cfg.items()}
    if isinstance(cfg, (list, tuple)):
        return [_to_plain(v) for v in cfg]
    return cfg


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


class _GenBundle:
    """Everything the producer needs from a generator checkpoint: the emitter,
    each drone's conditioning code ``z`` and learned jitter ``sigma``, plus the
    codebook dimensionality/names."""

    __slots__ = ("model", "z_map", "sigma_map", "cond_dim", "names")

    def __init__(self, model, z_map, sigma_map, cond_dim, names):
        self.model = model
        self.z_map: dict[str, torch.Tensor] = z_map
        self.sigma_map: dict[str, float | None] = sigma_map
        self.cond_dim = int(cond_dim)
        self.names = list(names)


def _load_generator(params: dict[str, Any], device: str) -> _GenBundle:
    """Load a trained generator, its per-drone codes and learned jitter σ.

    Handles both checkpoint formats transparently:

    * the modern flat ``_CodebookConditionedNoiseGen.state_dict()`` (keys
      ``generator.*`` + ``codebook.codes.<name>`` + optional
      ``log_jitter_sigma.<name>``) — the one the unified ``training.loop``
      writes, and the only one carrying the learned per-drone σ; and
    * the legacy ``save_bundle`` dict ``{model, codebook, cond_dim,
      drone_names}`` (no σ → resolves to ``None``).

    ``params['checkpoint']`` may be a local path or an ``r2://`` URI.
    """
    _ensure_src_on_path()
    from models.generative import PositionalHarmonicNoiseGen
    from training.artifacts import resolve_checkpoint_uri

    ckpt_path = resolve_checkpoint_uri(params["checkpoint"])
    obj = torch.load(ckpt_path, map_location=device, weights_only=False)

    def _build_emitter(cond_dim: int) -> Any:
        return PositionalHarmonicNoiseGen(
            sample_rate=params["sample_rate"],
            n_harmonics=params["n_harmonics"],
            use_diff_noise=not params["no_diff_noise"],
            cond_dim=cond_dim,
        )

    is_bundle = (
        isinstance(obj, dict) and "model" in obj and "codebook" in obj and "drone_names" in obj
    )
    if is_bundle:
        from tasks.noise_generation import DroneCodebook

        cond_dim = int(obj["cond_dim"])
        names = list(obj["drone_names"])
        model = _build_emitter(cond_dim)
        model.load_state_dict(obj["model"])
        codebook = DroneCodebook(cond_dim, names=names).to(device)
        codebook.load_state_dict(obj["codebook"])
        with torch.no_grad():
            z_map = {n: codebook([n])[0].detach().clone() for n in names}
        sigma_map: dict[str, float | None] = {n: None for n in names}
    else:
        # Flat _CodebookConditionedNoiseGen state_dict. The FiLM generator may
        # be spectral-norm parametrized (E6 latreg) and the wrapper may carry
        # per-drone log_jitter_sigma — plain PositionalHarmonicNoiseGen keys
        # won't match. So rebuild the exact composite via the registry (which
        # applies the same spectral-norm/codebook/σ structure), load strictly,
        # then extract the emitter + codes + learned σ.
        from models.registry import build_noise_gen_model

        sd = obj  # flat state_dict
        code_keys = [k for k in sd if k.startswith("codebook.codes.")]
        if not code_keys:
            raise ValueError(
                "generated-noise checkpoint is neither a save_bundle nor a "
                "conditioned state_dict (no 'codebook.codes.*' keys)"
            )
        names = [k[len("codebook.codes.") :] for k in code_keys]
        cond_dim = int(sd[code_keys[0]].shape[-1])
        film_sn = any(".parametrizations." in k for k in sd)
        learn_sig = any(k.startswith("log_jitter_sigma.") for k in sd)
        composite: Any = build_noise_gen_model(
            str(params.get("model_name", "positional_harmonic_gen")),
            sample_rate=params["sample_rate"],
            n_harmonics=params["n_harmonics"],
            use_diff_noise=not params["no_diff_noise"],
            cond_dim=cond_dim,
            drone_names=names,
            rps_jitter_sigma=float(params.get("rps_jitter_sigma_init", 0.6)),
            rps_jitter_tau=float(params.get("rps_jitter_tau", 0.016)),
            learn_rps_jitter_sigma=learn_sig,
            z_noise_std=0.0,  # inference: we inject our own vicinal z-noise
            film_spectral_norm=film_sn,
        )
        composite.load_state_dict(sd)
        composite.to(device).eval()
        model = composite.generator
        with torch.no_grad():
            z_map = {n: composite.codebook([n])[0].detach().clone() for n in names}
            sig_t = composite._resolve_jitter_sigma(names) if learn_sig else None
        sigma_map = {
            n: (float(sig_t[i]) if sig_t is not None else None) for i, n in enumerate(names)
        }

    model.to(device).eval()  # eval => deterministic modules; phase/jitter variety is explicit below
    z_map = {n: v.to(device).float() for n, v in z_map.items()}
    return _GenBundle(model, z_map, sigma_map, cond_dim, names)


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
        gb = _load_generator(params, device)
    except Exception as exc:  # surface the cause; readers time out on an empty buffer
        print(f"[generated-noise producer] failed to start: {exc}", file=sys.stderr)
        raise
    model = gb.model
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

    def _gen_batch() -> tuple[np.ndarray, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """One producer batch → ``(rps_np (bs,R,T), rel_b, z, sigma_t|None)``."""
        raise NotImplementedError  # replaced below by the mode-specific sampler

    interp = params.get("interp")
    if interp:
        # -- Vicinal embedding + geometry sampling along the DREGON↔Michael's
        #    segment. Per batch: draw one α; build z as a Gaussian ball around
        #    the segment point z(α); interpolate rotor positions and jitter σ at
        #    the same α; per chunk pick a mic-array rig and jitter each mic.
        e0, e1 = interp["endpoints"]
        z0, z1 = gb.z_map[e0], gb.z_map[e1]
        seg_len = float((z1 - z0).norm().item()) or 1.0
        s0, s1 = gb.sigma_map.get(e0), gb.sigma_map.get(e1)
        rot0 = np.asarray(load_geometry(e0, params["dregon_dir"])[1], dtype=np.float64)  # (R,3)
        rot1 = np.asarray(load_geometry(e1, params["dregon_dir"])[1], dtype=np.float64)
        n_rotors = int(rot0.shape[0])
        mic_cfg = interp["mic_sampling"]
        rigs = list(mic_cfg["rigs"])
        rig_probs = np.asarray(mic_cfg.get("prob", [1.0 / len(rigs)] * len(rigs)), dtype=np.float64)
        rig_probs = rig_probs / rig_probs.sum()
        mic_arrays = {rig: load_geometry(rig, params["dregon_dir"])[0] for rig in rigs}  # (M,3)
        n_mics = int(mic_arrays[rigs[0]].shape[0])
        mic_jit = float(mic_cfg.get("jitter_std", 0.0))
        a_lo, a_hi = float(interp["alpha"]["low"]), float(interp["alpha"]["high"])
        emb_noise = float(interp.get("embedding_noise", 0.0))
        rotor_interp = bool(interp.get("rotor_interp", True))
        js = interp.get("jitter_sigma", "interp")

        def _gen_batch() -> tuple[np.ndarray, torch.Tensor, torch.Tensor, torch.Tensor | None]:
            alpha = float(rng.uniform(a_lo, a_hi))
            z_base = (1.0 - alpha) * z0 + alpha * z1
            if emb_noise > 0.0:
                z_base = z_base + torch.from_numpy(
                    rng.normal(0.0, emb_noise * seg_len, size=int(z_base.shape[-1]))
                ).float().to(device)
            z = z_base.unsqueeze(0).expand(gen_bs, -1)
            rotor_pos = (
                (1.0 - alpha) * rot0 + alpha * rot1
                if rotor_interp
                else (rot1 if alpha >= 0.5 else rot0)
            )
            sig_val: float | None
            if js == "interp" and s0 is not None and s1 is not None:
                sig_val = (1.0 - alpha) * s0 + alpha * s1
            elif isinstance(js, (int, float)):
                sig_val = float(js)
            else:
                sig_val = None
            # α doubles as the rps_synthesis drone_profile blend (0=DREGON, 1=Michael's).
            rps_np = _rps_excitation_batch(
                params["rps_kind"],
                gen_bs,
                duration_s,
                sr,
                drone_profile=alpha,
                aggressiveness=params["aggressiveness"],
                rng=rng,
                flight_fs=params["flight_fs"],
            )
            rels = np.empty((gen_bs, n_mics, n_rotors, 3), dtype=np.float32)
            for b in range(gen_bs):
                rig = rigs[int(rng.choice(len(rigs), p=rig_probs))]
                mic_b = mic_arrays[rig] + rng.normal(0.0, mic_jit, mic_arrays[rig].shape)
                rels[b] = geometry_to_rel_pos(mic_b, rotor_pos)
            rel_b = torch.from_numpy(rels).to(device)
            sigma_t = (
                torch.full((gen_bs,), float(sig_val), device=device)
                if sig_val is not None
                else None
            )
            return rps_np, rel_b, z, sigma_t
    else:
        drone = params["drone"]
        z_single = gb.z_map[drone]
        sigma_single = gb.sigma_map.get(drone)
        mic_pos, rotor_pos_s = load_geometry(drone, params["dregon_dir"])
        rel = torch.from_numpy(geometry_to_rel_pos(mic_pos, rotor_pos_s)).float().to(device)
        n_rotors = int(rotor_pos_s.shape[0])
        blend = _DRONE_PROFILE_BLEND.get(drone, 0.0)

        def _gen_batch() -> tuple[np.ndarray, torch.Tensor, torch.Tensor, torch.Tensor | None]:
            rps_np = _rps_excitation_batch(
                params["rps_kind"],
                gen_bs,
                duration_s,
                sr,
                drone_profile=blend,
                aggressiveness=params["aggressiveness"],
                rng=rng,
                flight_fs=params["flight_fs"],
            )  # (bs, R, T)
            rel_b = rel.unsqueeze(0).expand(gen_bs, -1, -1, -1)
            z = z_single.unsqueeze(0).expand(gen_bs, -1)
            sigma_t = (
                torch.full((gen_bs,), float(sigma_single), device=device)
                if sigma_single is not None
                else None
            )
            return rps_np, rel_b, z, sigma_t

    while bool(run[0].item()):
        rps_np, rel_b, z, sigma_t = _gen_batch()
        rps_t = torch.from_numpy(rps_np).float().to(device)
        init_phase = None
        if params["random_phase"]:
            # Per-chunk random harmonic phases decorrelate the waveform texture
            # even at identical RPS — extra augmentation variety (model stays in
            # eval, so this is the ONLY stochastic knob besides α + geometry).
            init_phase = (
                torch.from_numpy(rng.uniform(0.0, 2.0 * np.pi, size=(gen_bs, n_rotors, n_harm)))
                .float()
                .to(device)
            )
        kwargs: dict[str, Any] = {"initial_phases": init_phase}
        if sigma_t is not None:
            # Force the OU linewidth ON at eval (the emitter's gate is
            # training-only unless rps_jitter is explicitly overridden).
            kwargs["rps_jitter"] = True
            kwargs["rps_jitter_sigma"] = sigma_t
        with torch.no_grad():
            audio = model(rps_t, rel_b, z, **kwargs).cpu()  # (bs, M, T)

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
        rps_kind: str = "synthetic_intermittent",
        flight_fs: float = 200.0,
        random_phase: bool = True,
        n_slots: int = 512,
        gen_batch: int = 32,
        warmup: int = 16,
        refresh: bool = True,
        device: str = "cuda:0",
        dregon_dir: str | Path = "data/DREGON",
        seed: int = 0,
        warmup_timeout_s: float = 60.0,
        interp: dict[str, Any] | None = None,
    ):
        # In interp mode the producer varies z/geometry per chunk; ``drone`` is
        # only the *nominal* rig used to size the ring buffer + reconstruct frame
        # meta (both rigs share M=8/R=4, so any endpoint works). Default it to
        # the first segment endpoint.
        self.interp = dict(interp) if interp else None
        if self.interp and (not drone or drone == "dregon"):
            drone = str(self.interp["endpoints"][0])
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
            "rps_kind": str(rps_kind),
            "flight_fs": float(flight_fs),
            "random_phase": bool(random_phase),
            "gen_batch": int(gen_batch),
            "refresh": bool(refresh),
            "device": str(device),
            "dregon_dir": str(dregon_dir),
            "seed": int(seed),
            "interp": self.interp,
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
        if rps_kind not in ("synthetic_intermittent", "full_flight"):
            raise ValueError(
                "generated noise supports rps.kind 'synthetic_intermittent' or "
                f"'full_flight', got {rps_kind!r}"
            )
        buf = g("buffer", {}) or {}
        checkpoint = g("checkpoint")
        if not checkpoint:
            raise ValueError("generated noise source requires 'checkpoint'")
        interp_cfg = g("interp")
        interp = _to_plain(interp_cfg) if interp_cfg else None
        return cls(
            checkpoint=checkpoint,
            drone=g("drone", "dregon"),
            interp=interp,
            sample_rate=sample_rate,
            duration_s=duration_s,
            n_harmonics=int(g("n_harmonics", 100)),
            no_diff_noise=bool(g("no_diff_noise", False)),
            aggressiveness=float(rps.get("aggressiveness", 1.0)),
            rps_kind=str(rps_kind),
            flight_fs=float(rps.get("flight_fs", 200.0)),
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
