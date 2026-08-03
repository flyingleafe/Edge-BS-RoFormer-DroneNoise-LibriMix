"""Online mixing, as dload pipeline composition.

An online-mix policy YAML (``conf/online_mix/*.yaml``) compiles to ONE infinite
``dload.Pipeline`` of per-sample ``td.Frame``s — no pool classes, no bespoke
sampling loops. The mapping (see ``docs/refactor-data-pipelines.md``):

- **real recordings** (``kind: frames``) — loaded once via
  :func:`mixing.load_noise_source_frames` (published frames; fixes baked in),
  then ``dload.choice`` over per-record window streams weighted by valid
  duration; each window's start is a uniform draw from a ``random_stream``;
- **synthetic engines** (``kind: generated`` / ``gp`` / ``static_comb``) — the
  engine object (a genuinely stateful resource: CUDA producer, GP coefficient
  table) is built once, then ``random_stream(seed).map(render)`` renders a
  chunk per draw;
- **audio pools** (``kind: audio_pool``) — ``Dataset.samples()`` over a
  (possibly shard-subset) manifest, ``.shuffle().repeat()``, key/holdout
  filters as ``.filter(...)``, decode+channel+window draws zipped in from
  ``random_stream``s (per-encounter randomness, dload-deterministic);
- **speech** — the pinned librispeech dataset streamed the same way; the
  historical bespoke ``packed_int16`` cache is superseded by the
  ``librispeech-pcm16`` derived dataset (memoized preprocessing is what
  ``Repository.derive`` IS) — decode dispatches on the manifest layout, one
  code path;
- **mixing** — ``dload.zip_with(render, ids, noise, speech)``: the id stream
  (``from_iterable(count(start), shard=True)`` — its worker striping
  reproduces the old ``worker_id + k * num_workers`` global-id assignment
  exactly) carries the global sample id that drives curriculum staging and
  the per-sample augmentation RNG (``make_rng``), exactly as before;
- **augmentation firing** stays on the per-sample-id RNG so the check_stream
  control-stream methodology (draw-count stability) keeps working.

Determinism model: stream content is deterministic per
``(base_seed, epoch, worker, position)`` — dload's model. The pre-refactor
mixer's stronger per-``(base_seed, gid)`` worker-count-independence of *chunk
content* is intentionally dropped (chunk↔id pairing follows the per-worker
streams); curriculum/augmentation decisions remain pure functions of ``gid``.
Same-config, same-``num_workers`` runs reproduce exactly.
"""

from __future__ import annotations

import hashlib
import io
import itertools
from collections.abc import Iterator, Mapping
from dataclasses import replace as _dc_replace
from fractions import Fraction
from functools import partial
from pathlib import Path
from typing import Any, Protocol, cast, runtime_checkable

import dload
import librosa
import numpy as np
import soundfile as sf
import tdseries as td
from omegaconf import DictConfig, ListConfig, OmegaConf

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - python-dotenv is a project dependency.
    load_dotenv = None

if load_dotenv is not None:
    # Let configs use ${oc.env:...} values from the project .env while still
    # respecting variables already provided by the shell/job launcher.
    load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)

from data_processing import mixing
from data_processing.frames import (
    audio_series,
    get_meta,
    rps_series,
)
from data_processing.mixing import (
    MAX_DRAW_RETRIES,
    find_inflight_window,
    is_silent,
    mix_at_source_to_noise_snr,
    resolve_motor_tracks,
    scale_source_to_snr,
)
from data_processing.noise_augmentations import (
    freq_scale_source_factor,
    maybe_apply_noise_augmentation,
)
from data_processing.streams import open_repository
from data_processing.time_warp import (
    WarpParams,
    apply_time_warp,
    sample_warp_params,
)

# Back-compat aliases for long-lived private names (tests, streams.mix_frames,
# time_warp docstring). New code imports these from data_processing.mixing.
_resolve_motor_tracks = resolve_motor_tracks
_inflight_window = find_inflight_window
_mix_at_source_to_noise_snr = mix_at_source_to_noise_snr
_scale_source_to_snr = scale_source_to_snr
_is_silent = is_silent
_MAX_DRAW_RETRIES = MAX_DRAW_RETRIES

DEFAULT_SAMPLE_RATE = 16_000
DEFAULT_DURATION_S = 1.0
DEFAULT_N_FFT = 2048
DEFAULT_HOP_LENGTH = 512


@runtime_checkable
class NoiseEngine(Protocol):
    """A stateful synthetic-noise resource renderable per chunk."""

    def sample_timeframe(self, rng: np.random.Generator, duration_s: float) -> td.Frame: ...


# =============================================================================
# Config access + per-sample-id determinism (unchanged semantics)
# =============================================================================


def _to_plain(cfg: Any) -> Any:
    if isinstance(cfg, (DictConfig, ListConfig)):
        return OmegaConf.to_container(cfg, resolve=True)
    return cfg


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, Mapping):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def make_rng(base_seed: int, global_sample_id: int) -> np.random.Generator:
    """Deterministic per-sample RNG independent of worker process."""
    payload = f"{int(base_seed)}:{int(global_sample_id)}".encode()
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    seed = int.from_bytes(digest, "little", signed=False)
    return np.random.default_rng(seed)


# =============================================================================
# Policy resolution (curriculum staging + probability ramps; unchanged)
# =============================================================================


def _sample_snr_db(policy: Mapping[str, Any], rng: np.random.Generator) -> float:
    spec = policy.get("snr_db", {"uniform": {"low": -30.0, "high": 0.0}})
    if isinstance(spec, (int, float)):
        return float(spec)
    if isinstance(spec, Mapping) and "uniform" in spec:
        u = cast(Mapping[str, Any], spec["uniform"])
        return float(rng.uniform(float(u.get("low", -30.0)), float(u.get("high", 0.0))))
    raise ValueError(f"unsupported snr_db spec: {spec!r}")


#: Floor for a ramped probability: a ramp NEVER resolves to exactly 0.0, so the
#: block consumes its single fire-decision RNG draw on every sample. This keeps
#: the per-sample RNG structure identical on both sides of the ramp window and
#: keeps check_stream's 1e-9 control stream draw-aligned with the real stream.
_RAMP_PROBABILITY_FLOOR = 1e-9

#: Policy keys whose value is a probabilistic augmentation block —
#: ``noise_augmentations`` may also be a LIST of blocks (applied sequentially).
_AUG_BLOCK_KEYS = ("augmentations", "noise_augmentations", "noise_time_warp")


def _resolve_probability(value: Any, global_sample_id: int) -> float:
    """Resolve a ``probability`` field — a plain float, or a linear-ramp mapping
    (floored at ``_RAMP_PROBABILITY_FLOOR``; see its comment)."""
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, Mapping) and "ramp" in value:
        ramp = cast(Mapping[str, Any], value["ramp"])
        x0, x1 = float(ramp["from"]), float(ramp["until"])
        y0, y1 = float(ramp.get("start", 0.0)), float(ramp.get("end", 1.0))
        gid = float(global_sample_id)
        if gid <= x0:
            p = y0
        elif gid >= x1:
            p = y1
        else:
            p = y0 + (y1 - y0) * (gid - x0) / (x1 - x0)
        return max(p, _RAMP_PROBABILITY_FLOOR)
    raise ValueError(f"unsupported probability spec: {value!r}")


def _materialize_ramps(stage: Mapping[str, Any], global_sample_id: int) -> Mapping[str, Any]:
    """Resolve any ramp-valued ``probability`` in ``stage`` to a float (on a
    copy — the source policy is never mutated)."""
    out: dict[str, Any] | None = None
    for key in _AUG_BLOCK_KEYS:
        spec = stage.get(key)
        blocks = spec if isinstance(spec, list) else [spec] if isinstance(spec, Mapping) else []
        if not any(isinstance(b.get("probability"), Mapping) for b in blocks):
            continue
        new_blocks = [
            {**b, "probability": _resolve_probability(b["probability"], global_sample_id)}
            if isinstance(b.get("probability"), Mapping)
            else b
            for b in blocks
        ]
        if out is None:
            out = dict(stage)
        out[key] = new_blocks if isinstance(spec, list) else new_blocks[0]
    return out if out is not None else stage


def _resolve_policy(policy: Mapping[str, Any], global_sample_id: int) -> Mapping[str, Any]:
    """Resolve constant or staged policy for a global sample id.

    The first stage whose ``until`` is ``None`` or greater than the current id
    is active; ramp-valued probabilities are materialized for this id.
    """
    stages = policy.get("stages") if isinstance(policy, Mapping) else None
    if not stages:
        return _materialize_ramps(policy, global_sample_id)
    for stage in stages:
        until = stage.get("until")
        if until is None or int(global_sample_id) < int(until):
            return _materialize_ramps(stage, global_sample_id)
    return _materialize_ramps(stages[-1], global_sample_id)


# =============================================================================
# Noise chunk extraction helpers (frame slicing + STFT-grid labels)
# =============================================================================


def _as_audio_ct(audio: np.ndarray, *, target_len: int | None = None) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 1:
        audio = audio[None, :]
    elif audio.ndim != 2:
        raise ValueError(f"audio must be 1-D or 2-D, got shape {audio.shape}")
    if target_len is not None:
        if audio.shape[-1] < target_len:
            audio = np.pad(audio, ((0, 0), (0, target_len - audio.shape[-1])))
        elif audio.shape[-1] > target_len:
            audio = audio[..., :target_len]
    return np.ascontiguousarray(audio, dtype=np.float32)


def _extract_audio_array(tf: td.Frame, *, target_len: int) -> np.ndarray:
    audio = tf["audio"]
    return _as_audio_ct(audio.data, target_len=target_len)


def _stft_frame_times(audio: td.Series, n_frames: int, hop_length: int) -> np.ndarray:
    # Frame i is placed at t_start + i * hop / sr, built from the exact sr/hop
    # frame-rate fraction (never a float division) via a throwaway GridIndex.
    ti = cast(td.GridIndex, audio.tindex)
    frame_rate = Fraction(ti.sr_num, ti.sr_den) / hop_length
    grid = td.GridIndex.create(
        (frame_rate.numerator, frame_rate.denominator), n_frames, t_start=ti.t_start_ticks
    )
    return grid.sample_times()


def interpolate_rps_to_stft_grid(
    tf: td.Frame,
    *,
    n_frames: int,
    hop_length: int,
) -> np.ndarray:
    """Interpolate a sliced Frame's RPS track to the model STFT grid."""
    _, rps_key, needs_clean = resolve_motor_tracks(tf)
    audio = tf["audio"]
    rps = tf[rps_key]
    if rps.data is None or rps.dim_size("time") == 0:
        return np.zeros((4, n_frames), dtype=np.float32)

    frame_times = _stft_frame_times(audio, n_frames, hop_length)
    if needs_clean:
        from data_processing.sources.dregon import clean_command_spikes

        rps = rps.map_data(clean_command_spikes)
    return rps.interpolate(frame_times).astype(np.float32)


# =============================================================================
# Noise source streams
# =============================================================================


def _load_real_records(
    spec: Mapping[str, Any], *, sample_rate: int, window_s: float
) -> list[dict[str, Any]]:
    """Load one real source spec (``kind: frames``) into chunkable records."""
    frames = mixing.load_noise_source_frames([dict(spec)], sample_rate=sample_rate)
    min_motor_rps = float(_cfg_get(spec, "min_motor_rps", 30.0))
    records: list[dict[str, Any]] = []
    for tf in frames:
        if "audio" not in tf:
            continue
        try:
            detect_key, _rps_key, needs_clean = resolve_motor_tracks(tf)
            audio = tf["audio"]
            detect = tf[detect_key]
            valid_start = max(audio.t_start, detect.t_start)
            valid_end = min(audio.t_end, detect.t_end)
            if min_motor_rps > 0:
                flight_start, flight_end = find_inflight_window(
                    tf, detect_key, min_motor_rps, clean=needs_clean
                )
                valid_start = max(valid_start, flight_start)
                valid_end = min(valid_end, flight_end)
            if valid_end - valid_start >= window_s:
                records.append({"tf": tf, "valid_start": valid_start, "valid_end": valid_end})
        except Exception as exc:
            print(f"Warning: skipping noise frame {get_meta(tf, 'recording_id', '?')}: {exc}")
    if not records:
        raise ValueError(f"no usable noise recordings found for spec {dict(spec)!r}")
    return records


def _cut_record_window(record: dict[str, Any], window_s: float, u: float) -> td.Frame:
    """Cut a uniform-random ``window_s`` chunk from a record's valid span."""
    vs, ve = float(record["valid_start"]), float(record["valid_end"])
    start = vs + float(u) * max(0.0, (ve - vs) - window_s)
    return record["tf"].time[start : start + window_s]


def _engine_chunk(engine: NoiseEngine, window_s: float, u: float) -> td.Frame:
    """Render one chunk from a synthetic engine, seeded by the stream draw."""
    rng = np.random.default_rng(int(float(u) * 2**53))
    return engine.sample_timeframe(rng, window_s)


def _build_engine(spec: Mapping[str, Any], *, window_s: float, sample_rate: int) -> NoiseEngine:
    kind = str(_cfg_get(spec, "kind"))
    if kind == "generated":
        from data_processing.generated_noise import GeneratedNoisePool

        return GeneratedNoisePool.from_config(spec, duration_s=window_s, sample_rate=sample_rate)
    if kind == "gp":
        from data_processing.gp_noise import GPRotorNoisePool

        return GPRotorNoisePool.from_config(spec, duration_s=window_s, sample_rate=sample_rate)
    if kind == "static_comb":
        from data_processing.rotor_spectral_model import StaticCombNoisePool

        return StaticCombNoisePool.from_config(spec, duration_s=window_s, sample_rate=sample_rate)
    raise ValueError(f"unsupported engine kind: {kind!r}")


# ─── audio_pool (telemetry-free dload audio datasets) ─────────────────────────

_AUDIO_EXTS = ("wav", "flac", "ogg", "mp3")


def _holdout_shards(
    manifest: Any, holdout: Mapping[str, Any] | None, max_shards: Any
) -> tuple[Any, tuple[int, int] | None]:
    """Shard-level train/valid split + shard cap.

    Returns ``(manifest_or_subset, index_range_or_none)``: multi-shard datasets
    reserve the last ``valid_shards`` whole shards (= whole recording groups)
    as the valid partition; ``max_shards`` caps the train side (a bounded
    shard subset of a huge dataset is plenty for a noise-augmentation pool and
    keeps the total R2 pull feasible). Datasets with too few shards to reserve
    one fall back to a per-shard sample-index window (returned as ``(lo_frac,
    hi_frac)`` and applied by key membership at filter time).
    """
    side = str(_cfg_get(holdout, "split", "")) if holdout else ""
    valid_shards = max(1, int(_cfg_get(holdout, "valid_shards", 1))) if holdout else 1
    shards = list(manifest.shards)
    index_range: tuple[int, int] | None = None
    if side in {"train", "valid"} and len(shards) > valid_shards:
        shards = shards[-valid_shards:] if side == "valid" else shards[:-valid_shards]
    elif side in {"train", "valid"}:
        frac = float(_cfg_get(holdout, "fraction", 0.1))
        if not 0.0 < frac < 1.0:
            raise ValueError(f"audio_pool holdout.fraction must be in (0,1), got {frac}")
        index_range = (0, 1)  # marker; the exact window is applied per shard below
    if side != "valid" and max_shards is not None and len(shards) > int(max_shards):
        shards = shards[: int(max_shards)]
    sub = _dc_replace(manifest, shards=tuple(shards))
    return sub, index_range


def _key_in_index_window(
    allowed_keys: frozenset[str] | None,
    include: tuple[str, ...],
    exclude: tuple[str, ...],
    sample: tuple[str, dict[str, bytes]],
) -> bool:
    """Sample filter: exact-or-substring key match against include/exclude,
    optionally narrowed to the holdout window's allowed key set."""
    key = str(sample[0])
    if allowed_keys is not None:
        return key in allowed_keys
    if include and not any(p == key or p in key for p in include):
        return False
    return not any(p == key or p in key for p in exclude)


def _index_window_keys(repo: Any, manifest: Any, side: str, frac: float) -> frozenset[str]:
    """The allowed sample keys of a few-shard dataset's index-window holdout.

    Reads the (≤ ``valid_shards``) shards' key lists once — only invoked when
    the shard-level split degenerated (tiny datasets), so this is a handful of
    small shards at most."""
    from dload.pack import PackReader

    allowed: set[str] = set()
    for shard in manifest.shards:
        pin = repo.open_shard(shard)
        try:
            with PackReader(pin.path) as reader:
                n = len(reader)
                cut = int(np.floor(n * (1.0 - frac)))
                cut = min(max(1, cut), n - 1) if n > 1 else 0
                lo, hi = (cut, n) if side == "valid" else (0, cut if n > 1 else n)
                allowed.update(str(reader.keys[i]) for i in range(lo, hi))
        finally:
            pin.release()
    return frozenset(allowed)


def _decode_audio_sample(
    layout: str | None, sample: tuple[str, dict[str, bytes]]
) -> tuple[np.ndarray, int] | None:
    """Decode one sample to ``((C, T), sr)`` — or ``None`` when it holds no
    audio (some datasets interleave non-audio samples, e.g. csv flight logs)."""
    key, fields = sample
    if layout == "tdframe-v1":
        from data_processing.streams import sample_to_frame

        frame = sample_to_frame(fields)
        if "audio" not in frame:
            return None
        series = frame["audio"]
        arr = np.asarray(series.data, dtype=np.float32)
        sr = int(cast(td.GridIndex, series.tindex).sr)
        return arr, sr
    ext = next((e for e in _AUDIO_EXTS if e in fields), None)
    if ext is None:
        return None
    raw, sr = sf.read(io.BytesIO(fields[ext]), dtype="float32", always_2d=True)  # (T, C)
    return np.ascontiguousarray(raw.T), int(sr)  # (C, T)


def _cycle_dataset_samples(
    manifest: Any, shuffle_buffer: int, seed: int
) -> Iterator[tuple[str, dict[str, bytes]]]:
    """Endlessly cycle a (sub-)manifest's samples, reshuffled per cycle.

    Built for ``dload.from_iterable(..., shard=True)``: each DataLoader worker
    takes an interleaved stripe of one logical stream, so workers never
    deadlock on empty shard stripes (the ``.repeat()``-over-SourceNode failure
    mode when ``num_workers > shards`` — an empty stripe cycles forever
    yielding nothing) and an empty source simply ends the worker's stream.
    The repository is opened lazily inside the iterating process (fork-safe).
    """
    repo = open_repository()
    ds = dload.Dataset(repo, manifest)
    pipe = ds.samples()
    if shuffle_buffer > 1:
        pipe = pipe.shuffle(shuffle_buffer, seed=seed)
    while True:
        empty = True
        for sample in pipe:  # each pass re-plans with the bumped epoch (reshuffled)
            empty = False
            yield sample
        if empty:
            return


def _audio_pool_chunk(
    sample_rate: int,
    window_len: int,
    channel: str | int,
    layout: str | None,
    sample: tuple[str, dict[str, bytes]],
    u_channel: float,
    u_cut: float,
) -> td.Frame | None:
    """Decode + mono-pick + fit-length one audio_pool sample (None = skip)."""
    decoded = _decode_audio_sample(layout, sample)
    if decoded is None:
        return None
    arr, sr = decoded
    if arr.ndim == 2:
        if isinstance(channel, str) and channel == "random":
            c = int(float(u_channel) * arr.shape[0]) % arr.shape[0]
        else:
            c = int(channel) % arr.shape[0]
        mono = arr[c]
    else:
        mono = arr
    mono = np.ascontiguousarray(mono, dtype=np.float32)
    if sr != sample_rate:
        mono = librosa.resample(
            mono, orig_sr=float(sr), target_sr=float(sample_rate), res_type="soxr_hq"
        )
    length = int(mono.shape[0])
    if length == 0:
        return None
    if length > window_len:
        start = int(float(u_cut) * (length - window_len + 1)) % (length - window_len + 1)
        mono = mono[start : start + window_len]
    elif length < window_len:
        reps = int(np.ceil(window_len / length))
        tiled = np.tile(mono, reps)
        start = int(float(u_cut) * (tiled.shape[0] - window_len + 1)) % (
            tiled.shape[0] - window_len + 1
        )
        mono = tiled[start : start + window_len]
    return td.Frame({"audio": audio_series(mono[None, :], sample_rate)})


def _audio_pool_stream(
    spec: Mapping[str, Any], *, sample_rate: int, window_s: float, seed: int
) -> dload.Pipeline:
    """An infinite chunk stream over a telemetry-free dload audio dataset."""
    dataset = str(_cfg_get(spec, "dataset"))
    if not dataset:
        raise ValueError("noise source kind 'audio_pool' requires a 'dataset' name")
    repo = open_repository()
    manifest = repo.manifest(dataset, _cfg_get(spec, "version", None))
    sub_manifest, index_window = _holdout_shards(
        manifest, _cfg_get(spec, "holdout", None), _cfg_get(spec, "max_shards", None)
    )
    if not sub_manifest.shards:
        raise ValueError(f"audio_pool dataset {dataset!r} has no shards after the holdout split")
    layout = (manifest.meta or {}).get("layout")
    allowed_keys: frozenset[str] | None = None
    if index_window is not None:
        frac = float(_cfg_get(_cfg_get(spec, "holdout"), "fraction", 0.1))
        allowed_keys = _index_window_keys(
            repo, sub_manifest, str(_cfg_get(_cfg_get(spec, "holdout"), "split")), frac
        )
    include = tuple(str(k) for k in (_cfg_get(spec, "include_keys", None) or ()))
    exclude = tuple(str(k) for k in (_cfg_get(spec, "exclude_keys", None) or ()))
    channel = _cfg_get(spec, "channel", "random")
    window_len = int(round(window_s * sample_rate))

    pipe = dload.from_iterable(
        partial(_cycle_dataset_samples, sub_manifest, 4096, seed), shard=True
    )
    pipe = pipe.filter(partial(_key_in_index_window, allowed_keys, include, exclude))
    pipe = dload.zip_with(
        partial(_audio_pool_chunk, sample_rate, window_len, channel, layout),
        pipe,
        dload.random_stream(dload.seeded(seed, "channel")),
        dload.random_stream(dload.seeded(seed, "cut")),
    )
    return pipe.filter(_not_none)


def _not_none(x: Any) -> bool:
    return x is not None


def _records_window_stream(
    records: list[dict[str, Any]], *, window_s: float, seed: int
) -> dload.Pipeline:
    """One duration-weighted random-window stream over chunkable records."""
    per_record = []
    durations = []
    for rec in records:
        rid = str(get_meta(rec["tf"], "recording_id", len(per_record)))
        rseed = dload.seeded(seed, "noise-window", rid)
        per_record.append(
            dload.random_stream(rseed).map(partial(_cut_record_window, rec, window_s))
        )
        durations.append(float(rec["valid_end"]) - float(rec["valid_start"]))
    if len(per_record) == 1:
        return per_record[0]
    return dload.choice(per_record, durations, seed=dload.seeded(seed, "records-choice"))


def build_noise_stream(
    specs: Any, *, sample_rate: int, window_s: float, seed: int
) -> tuple[dload.Pipeline, int]:
    """Compile ``sources.noise`` (one spec or a list) into one chunk Pipeline.

    Returns ``(pipeline, channel_ceiling)`` — the ceiling sizes the speech
    lane count for ``speech_per_channel: independent`` mixes.

    Unweighted real (``kind: frames``) sources merge into one
    duration-weighted ``dload.choice`` over per-record window streams (the
    merged pool's weight is the number of merged items, as before). A real
    source with an explicit ``weight``, and every engine/audio_pool source,
    becomes its own sub-stream at that pool-level weight.
    """
    specs = _to_plain(specs)
    items = list(specs) if isinstance(specs, list) else [specs]
    engine_kinds = {"generated", "static_comb", "gp", "audio_pool"}
    standalone = [c for c in items if _cfg_get(c, "kind") in engine_kinds]
    real_items = [c for c in items if _cfg_get(c, "kind") not in engine_kinds]
    weighted_real = [c for c in real_items if _cfg_get(c, "weight", None) is not None]
    merged_real = [c for c in real_items if _cfg_get(c, "weight", None) is None]

    streams: list[dload.Pipeline] = []
    weights: list[float] = []
    ceiling = 1

    if merged_real:
        records: list[dict[str, Any]] = []
        for spec in merged_real:
            records.extend(_load_real_records(spec, sample_rate=sample_rate, window_s=window_s))
        streams.append(_records_window_stream(records, window_s=window_s, seed=seed))
        weights.append(float(len(merged_real)))
        ceiling = max(ceiling, _records_channels(records))
    for c in weighted_real:
        records = _load_real_records(c, sample_rate=sample_rate, window_s=window_s)
        streams.append(_records_window_stream(records, window_s=window_s, seed=seed))
        weights.append(float(_cfg_get(c, "weight", 1.0)))
        ceiling = max(ceiling, _records_channels(records))
    for c in standalone:
        kind = str(_cfg_get(c, "kind"))
        cseed = dload.seeded(seed, "engine", kind, json_fingerprint(c))
        if kind == "audio_pool":
            streams.append(
                _audio_pool_stream(c, sample_rate=sample_rate, window_s=window_s, seed=cseed)
            )
        else:
            engine = _build_engine(c, window_s=window_s, sample_rate=sample_rate)
            streams.append(dload.random_stream(cseed).map(partial(_engine_chunk, engine, window_s)))
            ceiling = max(ceiling, 8)  # the project rigs are 8-mic
        weights.append(float(_cfg_get(c, "weight", 1.0)))

    if not streams:
        raise ValueError("online mix config requires sources.noise")
    if len(streams) == 1:
        return streams[0], ceiling
    return (
        dload.choice(streams, weights, seed=dload.seeded(seed, "pool-choice")),
        ceiling,
    )


def _records_channels(records: list[dict[str, Any]]) -> int:
    n = 1
    for rec in records:
        audio = rec["tf"]["audio"]
        n = max(n, int(audio.dim_size("mic")) if "mic" in audio.dims else 1)
    return n


def json_fingerprint(cfg: Any) -> str:
    """A short stable digest of a config spec (for per-source seed derivation)."""
    import json

    return hashlib.blake2b(
        json.dumps(_to_plain(cfg), sort_keys=True, default=str).encode(), digest_size=8
    ).hexdigest()


# =============================================================================
# Speech source stream
# =============================================================================


def _decode_speech_chunk(
    sample_rate: int, window_len: int, layout: str | None, sample: Any, u_cut: float
) -> np.ndarray | None:
    """Decode one speech sample to a mono ``(window_len,)`` chunk (None = skip).

    Decode dispatches on the dataset layout: raw audio files (flac/wav bytes)
    or the ``pcm16-mono-v1`` derived dataset (int16 PCM bytes — the memoized
    form of exactly this decode+resample).
    """
    key, fields = sample
    if layout == "pcm16-mono-v1":
        raw = np.frombuffer(fields["pcm"], dtype=np.int16)
        audio = raw.astype(np.float32) / 32767.0
        sr = sample_rate  # the derivation renders at the pinned rate
    else:
        ext = next((e for e in _AUDIO_EXTS if e in fields), None)
        if ext is None:
            return None
        audio, sr = sf.read(io.BytesIO(fields[ext]), dtype="float32", always_2d=False)
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        if int(sr) != int(sample_rate):
            audio = librosa.resample(audio, orig_sr=float(sr), target_sr=float(sample_rate))
        audio = np.asarray(audio, dtype=np.float32)
    n = int(audio.shape[0])
    if n >= window_len:
        start = int(float(u_cut) * (n - window_len + 1)) % (n - window_len + 1)
        audio = audio[start : start + window_len]
    else:
        audio = np.pad(audio, (0, window_len - n))
    return np.ascontiguousarray(audio, dtype=np.float32)


def _speech_key_allowed(subpath: str | None, exclude: tuple[str, ...], sample: Any) -> bool:
    key = str(sample[0])
    if subpath and f"/{subpath.strip('/')}/" not in f"/{key}/":
        return False
    return not any(f"/{t}/" in f"/{key}/" for t in exclude)


def build_speech_stream(
    spec: Mapping[str, Any], *, sample_rate: int, window_s: float, lanes: int, seed: int
) -> dload.Pipeline:
    """The speech utterance stream: a dload audio dataset (librispeech pins),
    shuffled + repeated, decoded and cut per encounter.

    ``lanes`` > 1 batches consecutive utterances into C-lane lists (the
    ``speech_per_channel: independent`` draw); the render fn tiles lane 0 for
    ``shared``.
    """
    spec = _to_plain(spec)
    dataset = str(_cfg_get(spec, "dataset", "librispeech"))
    subpath = _cfg_get(spec, "subpath", "LibriSpeech/train-clean-100")
    exclude = tuple(str(t) for t in (_cfg_get(spec, "exclude", None) or ()))
    window_len = int(round(window_s * sample_rate))

    ds = open_repository().dataset(dataset, _cfg_get(spec, "version", None))
    layout = (ds.manifest.meta or {}).get("layout")
    pipe = dload.from_iterable(
        partial(
            _cycle_dataset_samples,
            ds.manifest,
            int(_cfg_get(spec, "shuffle_buffer", 512)),
            seed,
        ),
        shard=True,
    )
    pipe = pipe.filter(partial(_speech_key_allowed, subpath, exclude))
    pipe = dload.zip_with(
        partial(_decode_speech_chunk, sample_rate, window_len, layout),
        pipe,
        dload.random_stream(dload.seeded(seed, "speech-cut")),
    )
    pipe = pipe.filter(_not_none)
    if lanes > 1:
        return pipe.window(lanes)
    return pipe.map(_as_lane_list)


def _as_lane_list(x: Any) -> list[Any]:
    return [x]


# =============================================================================
# Per-sample renderers (the zip_with bodies)
# =============================================================================


def _max_window_s(policy: Mapping[str, Any], duration_s: float) -> float:
    """Static worst-case noise-window length (seconds) over all policy stages.

    Covers freq_scale compression (``freq_scale_source_factor``, compounding
    over list blocks) and the time-warp's worst-case source read
    (``1 + dev_const + dev_sine`` + a fixed margin), so the noise stream can
    cut fixed-length windows that always cover whatever a per-sample stage
    needs. A warp block whose probability is a ramp Mapping counts as active.
    """
    from data_processing.time_warp import (
        DEFAULT_DEV_CONST,
        DEFAULT_DEV_SINE,
        WARP_SOURCE_MARGIN_S,
    )

    stages = policy.get("stages") if isinstance(policy, Mapping) else None
    stage_list = list(stages) if stages else [policy]
    window = duration_s
    for stage in stage_list:
        if not isinstance(stage, Mapping):
            continue
        base = duration_s * float(freq_scale_source_factor(stage.get("noise_augmentations")))
        warp = stage.get("noise_time_warp")
        if isinstance(warp, Mapping):
            p = warp.get("probability", 0.0)
            active = isinstance(p, Mapping) or float(p or 0.0) > 0.0
            if active:
                dev_c = float(warp.get("dev_const", DEFAULT_DEV_CONST))
                dev_s = float(warp.get("dev_sine", DEFAULT_DEV_SINE))
                base = base * (1.0 + dev_c + dev_s) + WARP_SOURCE_MARGIN_S
        window = max(window, base)
    return window


def _maybe_sample_time_warp(
    spec: Mapping[str, Any] | None,
    rng: np.random.Generator,
) -> WarpParams | None:
    """Fire-and-sample the noise time-warp (absent key / p<=0 consumes no RNG)."""
    if not spec:
        return None
    probability = float(spec.get("probability", 0.0))
    if probability <= 0.0 or rng.random() >= probability:
        return None
    return sample_warp_params(spec, rng)


def _apply_one_augmentation(
    audio: np.ndarray,
    spec: Mapping[str, Any] | None,
    rng: np.random.Generator,
) -> np.ndarray:
    if not spec:
        return audio
    probability = float(spec.get("probability", 0.0))
    if probability <= 0.0 or rng.random() >= probability:
        return audio
    choices = list(spec.get("choices", []))
    if not choices:
        return audio
    choice = choices[int(rng.integers(0, len(choices)))]
    if isinstance(choice, str):
        name, params = choice, {}
    elif isinstance(choice, Mapping):
        if len(choice) != 1:
            raise ValueError(f"augmentation choice must have one key, got {choice!r}")
        name, params = next(iter(choice.items()))
        params = params or {}
    else:
        raise ValueError(f"unsupported augmentation choice: {choice!r}")

    out = audio.copy()
    if name == "random_gain":
        min_db = float(params.get("min_db", -6.0))
        max_db = float(params.get("max_db", 6.0))
        gain = 10.0 ** (float(rng.uniform(min_db, max_db)) / 20.0)
        out *= np.float32(gain)
    elif name == "random_polarity":
        out *= -1.0
    elif name == "channel_drop":
        if out.shape[0] <= 1:
            return out
        max_channels = int(params.get("max_channels", 1))
        n_drop = int(rng.integers(1, min(max_channels, out.shape[0] - 1) + 1))
        drop = rng.choice(out.shape[0], size=n_drop, replace=False)
        out[drop, :] = 0.0
    else:
        raise ValueError(f"unsupported augmentation: {name!r}")
    return out.astype(np.float32, copy=False)


def _apply_one_augmentation_pair(
    mixture: np.ndarray,
    target: np.ndarray,
    spec: Mapping[str, Any] | None,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply one augmentation to the ``(mixture, clean target)`` pair *identically*."""
    if not spec:
        return mixture, target
    probability = float(spec.get("probability", 0.0))
    if probability <= 0.0 or rng.random() >= probability:
        return mixture, target
    choices = list(spec.get("choices", []))
    if not choices:
        return mixture, target
    choice = choices[int(rng.integers(0, len(choices)))]
    if isinstance(choice, str):
        name, params = choice, {}
    elif isinstance(choice, Mapping):
        if len(choice) != 1:
            raise ValueError(f"augmentation choice must have one key, got {choice!r}")
        name, params = next(iter(choice.items()))
        params = params or {}
    else:
        raise ValueError(f"unsupported augmentation choice: {choice!r}")

    mix = mixture.copy()
    tgt = target.copy()
    if name == "random_gain":
        min_db = float(params.get("min_db", -6.0))
        max_db = float(params.get("max_db", 6.0))
        gain = np.float32(10.0 ** (float(rng.uniform(min_db, max_db)) / 20.0))
        mix *= gain
        tgt *= gain
    elif name == "random_polarity":
        mix *= -1.0
        tgt *= -1.0
    elif name == "channel_drop":
        if mix.shape[0] <= 1:
            return mix, tgt
        max_channels = int(params.get("max_channels", 1))
        n_drop = int(rng.integers(1, min(max_channels, mix.shape[0] - 1) + 1))
        drop = rng.choice(mix.shape[0], size=n_drop, replace=False)
        mix[drop, :] = 0.0
        tgt[drop, :] = 0.0
    else:
        raise ValueError(f"unsupported SE augmentation: {name!r}")
    return mix.astype(np.float32, copy=False), tgt.astype(np.float32, copy=False)


def _render_rps_sample(
    static: Mapping[str, Any], gid: int, noise_tf: td.Frame, speech_lanes: list[Any]
) -> td.Frame:
    """One RPS-prediction sample: (warped/augmented noise + speech mix) + RPS.

    The per-sample-id RNG drives — in the historical fixed draw order — the
    time-warp fire/params, the noise-augmentation blocks, the source_prob
    fire, the SNR draw, and the post-mix augmentation. Chunk *content* comes
    from the streams (see the module docstring's determinism note).
    """
    rng = make_rng(int(static["base_seed"]), int(gid))
    policy = _resolve_policy(cast(Mapping[str, Any], static["policy"]), int(gid))
    target_len = int(static["target_len"])
    sample_rate = int(static["sample_rate"])
    hop_length = int(static["hop_length"])
    duration_s = float(static["duration_s"])

    aug_spec = policy.get("noise_augmentations")
    aug_blocks: list[Mapping[str, Any]] = (
        list(aug_spec) if isinstance(aug_spec, list) else [aug_spec] if aug_spec else []
    )
    # Per-stage oversampling factor (the resolved stage's freq_scale worst
    # case); the stream window is statically sized to cover every stage.
    aug_factor = float(freq_scale_source_factor(aug_spec))
    base_len = int(round(duration_s * aug_factor * sample_rate))

    warp = _maybe_sample_time_warp(
        cast("Mapping[str, Any] | None", policy.get("noise_time_warp")), rng
    )
    if warp is not None:
        noise_tf = apply_time_warp(noise_tf, warp, target_len=base_len, sample_rate=sample_rate)

    for aug_block in aug_blocks:
        noise_tf = maybe_apply_noise_augmentation(
            noise_tf, aug_block, rng, target_len=base_len, sample_rate=sample_rate
        )

    audio_track = noise_tf["audio"]
    noise_audio = _extract_audio_array(noise_tf, target_len=target_len)
    n_frames = noise_audio.shape[-1] // hop_length + 1

    mixture = noise_audio
    if speech_lanes and static.get("speech_present"):
        source_prob = float(policy.get("source_prob", 1.0))
        if rng.random() < source_prob:
            n_ch = noise_audio.shape[0]
            mode = str(policy.get("speech_per_channel", "independent"))
            if mode == "shared":
                source = np.tile(speech_lanes[0][None, :], (n_ch, 1)).astype(np.float32)
            else:
                lanes = [speech_lanes[i % len(speech_lanes)] for i in range(n_ch)]
                source = np.stack(lanes, axis=0).astype(np.float32)
            snr_db = _sample_snr_db(policy, rng)
            mixture = mix_at_source_to_noise_snr(
                source,
                noise_audio,
                snr_db,
                per_channel=bool(policy.get("snr_per_channel", False)),
            )

    mixture = _apply_one_augmentation(
        mixture,
        cast(Mapping[str, Any], policy.get("augmentations")),
        rng,
    )
    # Keep amplitude policy deliberately simple: no peak normalization (it
    # would alter the configured SNR/gain regime).
    rps = interpolate_rps_to_stft_grid(
        noise_tf.select(["audio", resolve_motor_tracks(noise_tf)[1]]),
        n_frames=n_frames,
        hop_length=hop_length,
    )
    audio_sr = cast(td.GridIndex, audio_track.tindex).sr
    if int(round(audio_sr)) != sample_rate:
        raise ValueError(f"noise audio sr {audio_sr} != configured {sample_rate}")

    return td.Frame(
        {
            "mixture": audio_series(mixture, sample_rate),
            "rps": rps_series(rps, sample_rate=sample_rate, hop_length=hop_length),
            "meta": td.Frame({"sample_id": int(gid), "task": "rps_prediction"}),
        }
    )


def _render_se_sample(
    static: Mapping[str, Any], gid: int, noise_tf: td.Frame, speech_lanes: list[Any]
) -> td.Frame:
    """One speech-enhancement sample: ``(mixture, clean_target)`` — the clean
    target being the gain-scaled speech exactly as mixed (post-mix augmentation
    applied identically to both). Silence rejection happens upstream (stream
    filters), so telemetry-free noise sources work and no RPS is touched."""
    rng = make_rng(int(static["base_seed"]), int(gid))
    policy = _resolve_policy(cast(Mapping[str, Any], static["policy"]), int(gid))
    target_len = int(static["target_len"])
    sample_rate = int(static["sample_rate"])

    audio_track = noise_tf["audio"]
    audio_sr = cast(td.GridIndex, audio_track.tindex).sr
    if int(round(audio_sr)) != sample_rate:
        raise ValueError(f"noise audio sr {audio_sr} != configured {sample_rate}")
    noise_audio = _extract_audio_array(noise_tf, target_len=target_len)
    if noise_audio.shape[0] > 1:
        ch = int(rng.integers(0, noise_audio.shape[0]))
        noise_audio = np.ascontiguousarray(noise_audio[ch : ch + 1])

    source = speech_lanes[0]
    snr_db = _sample_snr_db(policy, rng)
    per_channel = bool(policy.get("snr_per_channel", False))
    scaled_source = scale_source_to_snr(
        source[None, :], noise_audio, snr_db, per_channel=per_channel
    )
    mixture = (noise_audio + scaled_source).astype(np.float32)

    mixture, target = _apply_one_augmentation_pair(
        mixture,
        scaled_source,
        cast(Mapping[str, Any], policy.get("augmentations")),
        rng,
    )
    return td.Frame(
        {
            "mixture": audio_series(np.ascontiguousarray(mixture), sample_rate),
            "target": audio_series(np.ascontiguousarray(target), sample_rate),
            "meta": td.Frame({"sample_id": int(gid), "task": "speech_enhancement"}),
        }
    )


# =============================================================================
# The compiler: policy config -> infinite sample Frame pipeline
# =============================================================================


def build_online_mix_pipeline(cfg: Any) -> dload.Pipeline:
    """Compile an online-mix policy config into the infinite sample stream.

    The returned Pipeline yields one ``td.Frame`` per sample —
    ``{mixture, rps, meta}`` for ``task: rps_prediction`` or
    ``{mixture, target, meta}`` for ``task: speech_enhancement`` — with
    ``meta.sample_id`` carrying the global sample id (curriculum staging and
    per-sample augmentation RNG key off it downstream, e.g. the flatten /
    rps_corruption stages in ``frame_datasets.OnlineMixFrameDataset``).
    """
    cfg = _to_plain(cfg)
    sample_rate = int(_cfg_get(cfg, "sample_rate", DEFAULT_SAMPLE_RATE))
    duration_s = float(_cfg_get(cfg, "duration_s", DEFAULT_DURATION_S))
    n_fft = int(_cfg_get(cfg, "n_fft", DEFAULT_N_FFT))
    hop_length = int(_cfg_get(cfg, "hop_length", DEFAULT_HOP_LENGTH))
    base_seed = int(_cfg_get(cfg, "base_seed", 1234))
    start_sample_id = int(_cfg_get(cfg, "start_sample_id", 0))
    task = str(_cfg_get(cfg, "task", "rps_prediction"))
    policy = cast(Mapping[str, Any], _cfg_get(cfg, "policy", {}))
    if task not in {"rps_prediction", "speech_enhancement"}:
        raise ValueError(f"unsupported online-mix task: {task!r}")

    # Provenance print: which policy this PROCESS actually trains on.
    stages = policy.get("stages") if isinstance(policy, Mapping) else None
    stage_desc = (
        " | ".join(
            f"until={s.get('until')}:" + ",".join(sorted(k for k in s if k != "until"))
            for s in stages
        )
        if stages
        else "flat:" + ",".join(sorted(policy))
        if isinstance(policy, Mapping)
        else "?"
    )
    print(f"[online-mix] task={task} base_seed={base_seed} stages: {stage_desc}", flush=True)

    window_s = _max_window_s(policy, duration_s)
    sources = _cfg_get(cfg, "sources", {})
    noise_cfg = _cfg_get(sources, "noise", None)
    if noise_cfg is None:
        raise ValueError("online mix config requires sources.noise")
    noise_stream, channel_ceiling = build_noise_stream(
        noise_cfg, sample_rate=sample_rate, window_s=window_s, seed=base_seed
    )

    speech_cfgs = _cfg_get(sources, "speech", None)
    speech_stream = None
    speech_present = speech_cfgs is not None
    if speech_present:
        speech_cfg = speech_cfgs[0] if isinstance(speech_cfgs, list) else speech_cfgs
        lanes = 1 if task == "speech_enhancement" else channel_ceiling
        speech_stream = build_speech_stream(
            speech_cfg,
            sample_rate=sample_rate,
            # Speech is always cut at the final duration; the oversampled
            # window applies only to the noise (aug/warp source material).
            window_s=duration_s,
            lanes=lanes,
            seed=dload.seeded(base_seed, "speech"),
        )
    if task == "speech_enhancement" and speech_stream is None:
        raise ValueError("speech_enhancement online mixing requires a sources.speech pool")

    # Silence rejection (SE only): filtered upstream so a silent chunk never
    # enters a sample (a silent NOISE draw would collapse the clean target to
    # all-zeros via the source-to-noise scaling — see mixing.is_silent).
    if task == "speech_enhancement":
        target_len = int(round(duration_s * sample_rate))
        noise_stream = noise_stream.filter(partial(_nonsilent_frame, target_len))
        speech_stream = speech_stream.filter(_nonsilent_lanes)

    ids = dload.from_iterable(partial(itertools.count, start_sample_id), shard=True)
    static = {
        "policy": policy,
        "base_seed": base_seed,
        "target_len": int(round(duration_s * sample_rate)),
        "sample_rate": sample_rate,
        "n_fft": n_fft,
        "hop_length": hop_length,
        "duration_s": duration_s,
        "speech_present": speech_present,
        "task": task,
    }
    render = _render_se_sample if task == "speech_enhancement" else _render_rps_sample
    if speech_stream is not None:
        pipe = dload.zip_with(partial(render, static), ids, noise_stream, speech_stream)
    else:
        pipe = dload.zip_with(partial(_render_no_speech, static), ids, noise_stream)
    return pipe


def _render_no_speech(static: Mapping[str, Any], gid: int, noise_tf: td.Frame) -> td.Frame:
    render = _render_se_sample if static["task"] == "speech_enhancement" else _render_rps_sample
    return render(static, gid, noise_tf, [])


def _nonsilent_frame(target_len: int, tf: td.Frame) -> bool:
    return not is_silent(_extract_audio_array(tf, target_len=target_len))


def _nonsilent_lanes(lanes: list[Any]) -> bool:
    return all(not is_silent(np.asarray(lane)) for lane in lanes)
