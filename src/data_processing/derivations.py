"""dload *derived-dataset* declarations — every derived dataset, exactly once.

A derived dataset (dload 0.3.0 ``Repository.derive``) is a finite, deterministic
pipeline that dload runs *once*, commits as an ordinary content-addressed
version, and memoizes: every later caller of the identical pipeline hits the
same snapshot by *fingerprint* instead of recomputing. This module declares
**every** derived dataset in ``dload.lock`` as a frozen JSON spec (``SPECS``)
plus the module-level generator functions that materialize them. It is the
only way derived datasets are made — the historical bespoke scripts
(``create_dregon_librimix.py``, ``create_dataset.py``, ``build_se_valid.py``,
``publish_*.py``) are deleted; their per-sample cores live in
:mod:`data_processing.mixing` and :mod:`data_processing.sources`.

Generator families (all module-level, hence fingerprintable):

- ``generate_dregon_lm_split`` / ``generate_dn_lm_split`` — the LibriMix-style
  mixed datasets (``sample-dir-v1`` samples), reading parents through dload
  URIs — never ``data/`` paths. Noise pools are composed from *published
  frames datasets* (``noise_sources`` spec dicts — the same schema as the
  online-mix ``kind: frames`` source).
- ``generate_source_frames`` — a :mod:`data_processing.sources` builder as a
  derivation (``*``-frames datasets, MIMII/AVQ/...). DREGON gets no special
  treatment: its frames dataset is this same generator over the same registry.
- ``generate_frame_subset`` — a filtered/transformed subset of a published
  frames dataset (AVQ-egonoise: key filter + channel pick + resample).
- ``generate_raw_subset`` — a byte-exact raw-file subset (AVQ-raw).
- ``generate_pcm16_mono`` — a memoized mono int16 decode of a raw-audio
  dataset (``librispeech-pcm16``; the derived-dataset form of the deleted
  bespoke ``packed_int16`` speech cache).
- ``generate_beatvk_valid`` — the frozen beat-VK raw validation set (a pure
  frames→frames transform over pinned parents).
- ``generate_se_valid`` — the fixed SE validation sets (held-out noise ×
  held-out speakers over an SNR grid).
- ``generate_avq_vkrps`` — adopt-only placeholder (a GPU annotator sits in the
  loop; the spec pins the annotator commit in its note).

Fingerprint mechanics (why the shapes below matter):
    build_pipeline(name) == dload.from_iterable(partial(<gen_fn>, gen_spec))
dload fingerprints this by the generator's *module + qualname* plus the
``gen_spec`` dict (recursively, keys sorted). So:

- The generator must stay a top-level function (dload rejects lambdas/locals).
- Editing a generator's *behavior* without renaming it would silently serve the
  stale snapshot — **bump the spec's ``recipe_version`` on any behavioral
  change** (review-enforced convention; ``recipe_version`` is inside the
  fingerprint, so a bump forces a fresh identity).
- The *parent pins live inside ``gen_spec``* and thus inside the fingerprint
  (``from_iterable`` pipelines have no ``SourceNode``s). That is what ties a
  derived version to exact parent versions.
- ``meta``/``fields`` are **registry metadata, not generation inputs** — they
  are kept out of ``gen_spec`` and forwarded to ``derive(..., meta=...)``.

Determinism caveat: cross-machine *byte* determinism of the mixing generators
is not guaranteed (numpy RNG draw order, librosa/soundfile versions). That is
benign for memoization — the derivation ref settles once and every later
reader resolves that snapshot — but it means (a) materialize from one
designated box, (b) all glob listings are sorted here, and (c) the historical
pins are **adopted in place** (``scripts/derive.py adopt``) rather than
re-derived, which would upload a near-duplicate copy. See
``docs/derived-datasets-plan.md`` + ``docs/data-and-artifacts.md``.
"""

from __future__ import annotations

import io
import json
from collections.abc import Iterator
from functools import partial
from pathlib import Path
from typing import Any

import dload
import numpy as np
import soundfile as sf
import tdseries as td

Sample = tuple[str, dict[str, bytes]]

#: Resolved parent dataset pins (from ``dload.lock``). Frozen into the specs so
#: they participate in the derivation fingerprint. Update deliberately (a change
#: mints a new derived identity); ``tests/data_processing/test_derivations.py``
#: asserts these match the lock file (pin-drift guard).
PARENTS = {
    "DREGON": "dload:DREGON@db39bcf762d0b2beb3433fc2760da6a55e078f8134f8bb074ee8bf985a5ffc03",
    "librispeech": "dload:librispeech@b674a6d0c4e9d598e7f12400d75e7f21b9bea72845aa3bcb37f5b96d56f73783",
    "drone_audio": "dload:drone_audio@b6c77a68c55dedec11750a3784c10833e7db981fab6ef00380300a9e4d382b95",
    "DREGON-frames": "dload:DREGON-frames@298e77d4cb96fd1fcce052360b6c669ea403596c89aaed8c00e1e83d1d159279",
    "michaels-frames": "dload:michaels-frames@fdef818432e99f0909762b9a9d45b76ae95ef3d1f1b7b9aa8012bcc91cd9200a",
    "AVQ": "dload:AVQ@50dd53d1a6c0ab81fe02e4a40a57557a0a2b1c1b85152470edd12aa6d0725f39",
    "AVQ-egonoise": "dload:AVQ-egonoise@b43b374b007a0d5c9575dd2feacd31a05097d0c436629819b936273f17cf7703",
    "DREGON-LM-V4-michaels-valid-full": "dload:DREGON-LM-V4-michaels-valid-full@9604f3ffc2c935e2ba2be52bd96c602d02a6999f1d683ee89fa1b0e28fafc4a9",
}


# ─── Sample encoding ──────────────────────────────────────────────────────────


def wav_bytes(arr: np.ndarray, sample_rate: int) -> bytes:
    """Encode ``(T,)`` or ``(T, C)`` float audio to WAV bytes, byte-for-byte as
    the historical disk writers' ``sf.write(path, arr, sr)`` would."""
    buf = io.BytesIO()
    sf.write(buf, arr, sample_rate, format="WAV")
    return buf.getvalue()


def _sample_dir_meta(fields: dict[str, str]) -> dict[str, Any]:
    """The manifest ``meta`` for a ``sample-dir-v1`` derived dataset.

    ``fields`` maps sample field name → on-disk filename (stem+ext), consumed by
    ``streams.ensure_local``/``_field_relpath`` to reconstruct a ``sample_NNNNN/``
    tree; ``layout`` selects the decode dispatch; ``meta_sample`` restores the
    split-level ``metadata.json`` from the ``_meta`` bookkeeping sample.
    """
    return {
        "layout": "sample-dir-v1",
        "fields": dict(fields),
        "meta_sample": {"key": "_meta", "fields": {"metadata": "metadata.json"}},
    }


def _split_dload_uri(uri: str) -> tuple[str, str | None]:
    """``"dload:NAME@sha"`` -> ``("NAME", "sha")``; a bare name -> version ``None``."""
    name, _, version = str(uri).removeprefix("dload:").partition("@")
    return name, version or None


def _resolve_librispeech(parent_uri: str, subpath: str | None) -> list[str]:
    """Sorted speech-file list from a pinned librispeech dload URI."""
    from data_processing.streams import resolve_source

    root = resolve_source(parent_uri)
    speech_dir = Path(root) / subpath if subpath else Path(root)
    files = sorted(str(p) for ext in ("*.wav", "*.flac") for p in speech_dir.rglob(ext))
    if not files:
        raise ValueError(f"No speech files under {speech_dir}")
    return files


# ─── LibriMix generators ──────────────────────────────────────────────────────


def generate_dregon_lm_split(gen: dict[str, Any]) -> Iterator[Sample]:
    """Yield the samples of one multichannel DREGON-LM split.

    ``gen`` carries ``seed``/``num_samples``/``split``, ``params`` (mixing
    knobs), ``parents`` (dload URIs), ``noise_sources`` (frames spec dicts —
    ``None`` selects the split defaults), and ``mode``: ``"synthesized"``
    (LibriSpeech mixed onto noise chunks) or ``"real_valid"`` (raw recording
    clips, no mixing — the real-valid recipe). All per-sample math is
    :mod:`data_processing.mixing`.
    """
    import random

    from data_processing import mixing

    params = gen["params"]
    split = gen["split"]
    mode = gen.get("mode", "synthesized")
    sample_rate = int(params["sample_rate"])
    sample_duration = float(params["sample_duration"])
    target_length = int(sample_duration * sample_rate)
    min_motor_rps = float(params.get("min_motor_rps", 30.0))

    # ── noise pool (published frames; fixes baked in) ──────────────────────
    noise_specs = gen.get("noise_sources")
    if noise_specs is None:
        if split == "train":
            noise_specs = [
                {"dataset": PARENTS["DREGON-frames"], "splits": mixing.TRAIN_NOISE_SPLITS}
            ]
        else:
            noise_specs = [
                {
                    "dataset": PARENTS["DREGON-frames"],
                    "recording_ids": mixing.VALID_NOISE_RECORDING_IDS,
                }
            ]
    noise_records = mixing.load_noise_source_frames(noise_specs, sample_rate=sample_rate)
    if not noise_records:
        raise ValueError("No valid multichannel noise records found")

    random.seed(gen["seed"])
    np.random.seed(gen["seed"])

    metadata_list = []
    if mode == "real_valid":
        # Raw clips — no speech, no mixing. Fields: mixture + rps only.
        clips = mixing.iter_real_valid_clips(
            noise_records,
            num_samples=int(gen["num_samples"]),
            sample_duration=sample_duration,
            sample_rate=sample_rate,
            seed=int(gen["seed"]),
            min_motor_rps=min_motor_rps,
            max_non_overlapping=bool(params.get("max_non_overlapping", False)),
        )
        for idx, (audio, rps, clip_meta) in enumerate(clips):
            sample_id = f"sample_{idx:05d}"
            sample_meta = {"id": sample_id, **clip_meta, "rps_shape": list(rps.shape)}
            metadata_list.append(sample_meta)
            yield (
                sample_id,
                {
                    "mixture": wav_bytes(audio.T.astype(np.float32), sample_rate),
                    "rps": dload.codecs.npy_bytes(rps),
                },
            )
        yield "_meta", {"metadata": dload.codecs.json_bytes({split: metadata_list})}
        return

    # ── synthesized split ────────────────────────────────────────────────────
    speech_files = _resolve_librispeech(gen["parents"]["librispeech"], params.get("speech_subpath"))
    snr_range = (float(params["snr_range"][0]), float(params["snr_range"][1]))
    for idx in range(int(gen["num_samples"])):
        sample_id = f"sample_{idx:05d}"
        arrays, sample_meta = mixing.render_multichannel_sample(
            noise_records,
            speech_files,
            target_length=target_length,
            sample_rate=sample_rate,
            sample_duration=sample_duration,
            snr_range=snr_range,
            speech_per_channel=params["speech_per_channel"],
            source_white_noise_prob=float(params["source_white_noise_prob"]),
            white_noise_prob=float(params["white_noise_prob"]),
            white_noise_snr=float(params["white_noise_snr"]),
            min_motor_rps=min_motor_rps,
        )
        sample_meta = {"id": sample_id, **sample_meta}
        metadata_list.append(sample_meta)
        yield (
            sample_id,
            {
                "mixture": wav_bytes(arrays["mixture"], sample_rate),
                "vocals": wav_bytes(arrays["vocals"], sample_rate),
                "noise": wav_bytes(arrays["noise"], sample_rate),
                "rps": dload.codecs.npy_bytes(arrays["rps"]),
                "meta": dload.codecs.json_bytes(sample_meta),
            },
        )

    yield "_meta", {"metadata": dload.codecs.json_bytes({split: metadata_list})}


def generate_dregon_lm_subset(gen: dict[str, Any]) -> Iterator[Sample]:
    """Yield a byte-verbatim subset of a published ``sample-dir-v1`` split.

    ``gen``: ``parent`` (the pinned ``dload:NAME@sha`` of the split),
    ``split`` (the key of the parent's ``metadata.json``), ``fields``
    (field -> filename, copied as they are) and ``keep_source_types`` (the
    per-clip ``source_type`` values to keep). Kept clips are renumbered from
    zero in parent order and each row keeps its parent id under ``source_id``.

    Every field is copied as bytes, never decoded and re-encoded, so a subset
    clip is bit-identical to its parent clip. That is the point: a twin whose
    audio or labels moved would not be comparable with the set it is cut from,
    and the parent's own generator no longer reproduces its labels (the
    published DREGON-LM-V4 valid sets predate both the michaels telemetry
    calibration of 2026-07-31 and the ``motors_measured`` preference of
    2026-08-05).
    """
    from data_processing.streams import resolve_source

    root = Path(resolve_source(gen["parent"]))
    split = gen["split"]
    fields = dict(gen["fields"])
    keep = {str(t) for t in gen["keep_source_types"]}

    rows = json.loads((root / "metadata.json").read_text())[split]
    metadata_list = []
    for idx, row in enumerate(r for r in rows if str(r.get("source_type")) in keep):
        sample_id = f"sample_{idx:05d}"
        metadata_list.append({**row, "id": sample_id, "source_id": row["id"]})
        yield (
            sample_id,
            {name: (root / row["id"] / fname).read_bytes() for name, fname in fields.items()},
        )
    if not metadata_list:
        raise ValueError(f"no {split} clip of {gen['parent']} has source_type in {sorted(keep)}")
    yield "_meta", {"metadata": dload.codecs.json_bytes({split: metadata_list})}


def generate_dn_lm_split(gen: dict[str, Any]) -> Iterator[Sample]:
    """Yield the samples of one DN-LM (DroneNoise-LibriMix, mono) split.

    Reads LibriSpeech + a drone-noise dataset via dload, then reuses
    :func:`data_processing.mixing.mix_dn_lm` (length/normalize/inverse-distance/
    SNR/anti-clip). No ``rps`` field — DN-LM is a plain speech-enhancement
    mixture; consume it via ``ensure_local`` + the folder loader.
    """
    import random

    from data_processing import mixing
    from data_processing.streams import resolve_source

    params = gen["params"]
    split = gen["split"]
    sample_rate = int(params["sample_rate"])
    target_length = int(float(params["sample_duration"]) * sample_rate)

    speech_files = _resolve_librispeech(gen["parents"]["librispeech"], params.get("speech_subpath"))

    noise_root = resolve_source(gen["parents"]["noise"])
    noise_subpath = params.get("noise_subpath")
    noise_dir = Path(noise_root) / noise_subpath if noise_subpath else Path(noise_root)
    noise_exts = ("*.wav", "*.WAV", "*.flac", "*.FLAC", "*.mp3", "*.MP3", "*.ogg", "*.OGG")
    noise_files = sorted(str(p) for ext in noise_exts for p in noise_dir.rglob(ext))
    if not noise_files:
        raise ValueError(f"No noise files under {noise_dir}")

    random.seed(gen["seed"])
    np.random.seed(gen["seed"])
    snr_range = tuple(params["target_snr_range"]) if params.get("target_snr_range") else None
    dist_range = tuple(params["speech_distance_range"])

    metadata_list = []
    for idx in range(int(gen["num_samples"])):
        sample_id = f"sample_{idx:05d}"
        speech = mixing.load_audio(random.choice(speech_files), target_sr=sample_rate)
        noise_path = random.choice(noise_files)
        noise = mixing.load_audio(noise_path, target_sr=sample_rate)
        arrays, actual_snr, speech_distance = mixing.mix_dn_lm(
            speech,
            noise,
            target_length=target_length,
            speech_distance_range=dist_range,
            noise_distance=float(params["noise_distance"]),
            target_snr_range=snr_range,
        )
        sample_meta = {
            "id": sample_id,
            "input_snr": actual_snr,
            "noise_source": Path(noise_path).name,
            "speech_distance": speech_distance,
        }
        metadata_list.append(sample_meta)
        yield (
            sample_id,
            {
                "mixture": wav_bytes(arrays["mixture"], sample_rate),
                "vocals": wav_bytes(arrays["vocals"], sample_rate),
                "noise": wav_bytes(arrays["noise"], sample_rate),
                "meta": dload.codecs.json_bytes(sample_meta),
            },
        )

    yield "_meta", {"metadata": dload.codecs.json_bytes({split: metadata_list})}


# ─── Source-frames generator (uniform over the sources registry) ──────────────


def generate_source_frames(gen: dict[str, Any]) -> Iterator[Sample]:
    """Yield ``tdframe-v1`` recording samples built by a sources-registry entry.

    ``gen``: ``{"source": <registry name>, "recipe_version": N, "raw": ...}``
    where ``raw`` is either ``{"kind": "dload", "uri": "dload:NAME@sha"}``
    (project-local raw tree) or ``{"kind": "download"}`` (fetch via the
    registry entry's pinned ``DownloadSpec``/fetcher into the raw cache).
    """
    from data_processing import sources
    from data_processing.streams import frame_to_sample, resolve_source

    name = gen["source"]
    raw = gen.get("raw") or {}
    root = resolve_source(raw["uri"]) if raw.get("kind") == "dload" else sources.raw_root(name)
    for key, frame in sources.get(name).builder(root):  # type: ignore[misc]
        yield key, frame_to_sample(frame)


# ─── Frame-subset generator ───────────────────────────────────────────────────


def generate_frame_subset(gen: dict[str, Any]) -> Iterator[Sample]:
    """Yield a filtered/transformed subset of a published frames dataset.

    ``gen``: ``parent`` (``dload:NAME@sha``), ``include_keys`` (recording ids to
    keep), optional ``channel`` (pick one audio channel), ``sample_rate``
    (soxr-resample), ``forbid_entries`` (raise if present — guards the
    "pure ego-noise" invariant), ``meta_updates`` (per-sample meta additions).
    Geometry entries are dropped when a channel is picked (a single channel
    has no array).
    """
    from data_processing.frames import (
        audio_series,
        get_meta,
        resample_audio_series,
        with_meta,
    )
    from data_processing.streams import frame_to_sample, iter_published_frames

    name, version = _split_dload_uri(str(gen["parent"]))
    include = [str(k) for k in gen["include_keys"]]
    wanted = set(include)
    seen: set[str] = set()
    channel = gen.get("channel")
    sample_rate = int(gen.get("sample_rate", 0)) or None
    forbid = set(gen.get("forbid_entries", []))
    meta_updates = dict(gen.get("meta_updates", {}))
    for frame in iter_published_frames(name, version):
        rid = str(get_meta(frame, "recording_id", ""))
        if rid not in wanted:
            continue
        missing_labels = forbid & set(frame.keys())
        if missing_labels:
            raise ValueError(f"{rid}: forbidden entries present: {sorted(missing_labels)}")
        audio = frame["audio"]
        idx = audio.tindex
        if not isinstance(idx, td.GridIndex):
            raise ValueError(f"{rid}: audio is not uniformly sampled")
        data = np.asarray(audio.data, dtype=np.float32)
        native_sr = int(idx.sr)
        if channel is not None and data.ndim == 2:
            data = data[int(channel) : int(channel) + 1]
        series = audio_series(np.ascontiguousarray(data), native_sr)
        if sample_rate is not None and sample_rate != native_sr:
            series = resample_audio_series(series, sample_rate)
        n = int(np.asarray(series.data).shape[-1])
        out_sr = sample_rate or native_sr
        out = with_meta(
            td.Frame({"audio": series, "meta": frame["meta"]}),
            sample_rate=out_sr,
            n_channels=1 if channel is not None else int(data.shape[0] if data.ndim == 2 else 1),
            duration_s=round(n / out_sr, 3),
            source_dataset=name,
            source_version=version or "lock-pin",
            **meta_updates,
        )
        if channel is not None:
            out = with_meta(out, source_channel=int(channel))
        seen.add(rid)
        yield rid, frame_to_sample(out)
    missing = wanted - seen
    if missing:
        raise ValueError(f"{name} did not yield the required keys {sorted(missing)}")


# ─── Raw-subset generator (byte-exact file passthrough) ───────────────────────


def generate_raw_subset(gen: dict[str, Any]) -> Iterator[Sample]:
    """Yield raw files of a source's download, byte-exact, minus a skip set.

    ``gen``: ``source`` (registry name — the raw files are fetched via its
    pinned download spec), ``skip`` (``{"extensions": [...], "stem_prefix":
    ...}``). Sample key = archive-relative path (filesystem-neutral); field =
    file extension.
    """
    from data_processing import sources
    from data_processing.sources._common import safe_key

    root = sources.raw_root(gen["source"])
    skip = gen.get("skip") or {}
    exts = {str(e).lower() for e in skip.get("extensions", [])}
    stem_prefix = str(skip.get("stem_prefix", ""))
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        if p.suffix.lower() in exts and p.stem.lower().startswith(stem_prefix):
            continue
        rel = p.relative_to(root)
        key = safe_key(str(rel.with_suffix("")))
        ext = p.suffix.lower().lstrip(".") or "bin"
        yield key, {ext: p.read_bytes()}


# ─── Decoded-speech cache (memoized decode+resample) ──────────────────────────

#: Manifest layout of :func:`generate_pcm16_mono` output — the decode dispatch
#: key in ``online_mixing._decode_speech_chunk``.
PCM16_LAYOUT = "pcm16-mono-v1"


def generate_pcm16_mono(gen: dict[str, Any]) -> Iterator[Sample]:
    """Decode a raw-audio dload dataset once to mono int16 PCM at a fixed rate.

    This is the derived-dataset form of the deleted bespoke ``packed_int16``
    speech cache: the online mixer's per-encounter FLAC decode + resample is
    pure, deterministic preprocessing, which is exactly what
    ``Repository.derive`` memoizes. Consumed by pointing a policy's
    ``sources.speech[].dataset`` at this dataset instead of ``librispeech``;
    ``online_mixing._decode_speech_chunk`` dispatches on the manifest layout,
    so nothing else changes. It matters for the 8-lane RPS streams, which
    decode one utterance per mic per sample.

    ``gen``: ``parent`` (dload URI), ``sample_rate``, optional ``subpath``.
    """
    import librosa

    from data_processing.streams import resolve_source

    root = Path(resolve_source(gen["parent"]))
    subpath = gen.get("subpath")
    audio_root = root / subpath if subpath else root
    sample_rate = int(gen["sample_rate"])
    for path in sorted(p for ext in ("*.wav", "*.flac") for p in audio_root.rglob(ext)):
        audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        if int(sr) != sample_rate:
            audio = librosa.resample(audio, orig_sr=float(sr), target_sr=float(sample_rate))
        pcm = np.clip(audio * 32767.0, -32768.0, 32767.0).astype(np.int16)
        key = str(path.relative_to(root).with_suffix("")).replace("\\", "/")
        yield key, {"pcm": pcm.tobytes()}


# ─── beat-VK frozen validation set ────────────────────────────────────────────

_BEATVK_WINDOW_S = 16.0
_BEATVK_FRAME_S = 0.032  # the scorer's fixed evaluation grid (= 512 / 16000)
_BEATVK_WINDOW_FRAMES = int(round(_BEATVK_WINDOW_S / _BEATVK_FRAME_S))  # 500
_BEATVK_GROUND_MAX = 5.0  # mean rps below -> "ground"
_BEATVK_WARMUP_MAX = 45.0  # mean rps below -> "warmup"; else "cruise"


def _trim_constant_runs(ts: np.ndarray, vals: np.ndarray) -> tuple[float, float, float, float]:
    """Trim leading/trailing exact-constant telemetry runs (logger not live)."""
    same = np.all(vals[:, 1:] == vals[:, :-1], axis=0)  # (M-1,) consecutive equality
    lead = 0
    while lead < len(same) and same[lead]:
        lead += 1
    trail = 0
    while trail < len(same) and same[len(same) - 1 - trail]:
        trail += 1
    t0, t1 = float(ts[lead]), float(ts[len(ts) - 1 - trail])
    if t1 <= t0:
        raise ValueError("telemetry is entirely constant — no live span")
    return t0, t1, t0 - float(ts[0]), float(ts[-1]) - t1


def _window_mean_rps(ts: np.ndarray, vals: np.ndarray, start: float) -> float:
    """Mean raw RPS over ``[start, start + window)`` on the 0.032 s grid."""
    grid = start + np.arange(_BEATVK_WINDOW_FRAMES) * _BEATVK_FRAME_S
    per_rotor = [np.interp(grid, ts, vals[r]) for r in range(vals.shape[0])]
    return float(np.mean(per_rotor))


def _beatvk_regime(mean_rps: float) -> str:
    if mean_rps < _BEATVK_GROUND_MAX:
        return "ground"
    if mean_rps < _BEATVK_WARMUP_MAX:
        return "warmup"
    return "cruise"


def _beatvk_frame(
    source: str, source_version: str, key: str, rps_entry: str, src: td.Frame
) -> td.Frame:
    """One source recording Frame -> the published beatvk-valid-raw Frame."""
    audio_src = src["audio"]
    rps_src = src[rps_entry]
    sr = audio_src.tindex.sr
    if float(sr) != int(sr):
        raise ValueError(f"{key}: non-integer audio sample rate {sr}")

    # Re-anchor time so the audio starts at t=0; telemetry stamps shift by the
    # same offset, so audio/telemetry alignment is exactly preserved. The
    # audio data array itself is byte-identical to the source.
    shift = float(audio_src.t_start)
    audio = td.uniform(np.asarray(audio_src.data), int(sr), dims=("mic", "time"), t_start=0.0)
    ts = np.asarray(rps_src.tindex.abs_stamps, dtype=np.float64) - shift
    vals = np.asarray(rps_src.data)
    rps_raw = td.events(ts, vals, dims=("rotor", "time"))

    # Window manifest over the live span (times in re-anchored seconds).
    live_t0, live_t1, lead_s, trail_s = _trim_constant_runs(ts, vals)
    span_start = max(float(audio.t_start), live_t0)
    span_end = min(float(audio.t_end), live_t1)
    windows: list[dict[str, Any]] = []
    start = span_start
    while start + _BEATVK_WINDOW_S <= span_end + 1e-9:
        mean_rps = _window_mean_rps(ts, vals, start)
        windows.append(
            {
                "index": len(windows),
                "start_s": round(start, 6),
                "end_s": round(start + _BEATVK_WINDOW_S, 6),
                "regime": _beatvk_regime(mean_rps),
                "mean_rps": round(mean_rps, 4),
            }
        )
        start += _BEATVK_WINDOW_S
    spans = {
        "audio": [float(audio.t_start), float(audio.t_end)],
        "telemetry": [float(ts[0]), float(ts[-1])],
        "eval": [span_start, span_end],
        "telemetry_const_trim_s": [round(lead_s, 4), round(trail_s, 4)],
    }
    rate_hz = float(1.0 / np.median(np.diff(ts)))
    meta = {
        "recording_id": key,
        "source_dataset": source,
        "source_version": source_version,
        "source_rps_entry": rps_entry,
        "sample_rate": int(sr),
        "n_channels": int(audio.dim_size("mic")),
        "rps_rate_hz": round(rate_hz, 2),
        "time_anchor_offset_s": shift,
        "spans": spans,
        "window_s": _BEATVK_WINDOW_S,
        "regime_thresholds": {
            "ground_max": _BEATVK_GROUND_MAX,
            "warmup_max": _BEATVK_WARMUP_MAX,
        },
        "windows": windows,
    }
    return td.Frame({"audio": audio, "rps_raw": rps_raw, "meta": td.Frame(meta)})


def generate_beatvk_valid(gen: dict[str, Any]) -> Iterator[Sample]:
    """The frozen beat-VK raw validation set: native-rate audio + RAW measured
    telemetry (re-anchored to t=0) + a frozen 16 s window manifest, built from
    pinned frames datasets (``gen["recordings"]`` = ``[[parent, key, rps_entry],
    ...]``; ``gen["parents"]`` pins them)."""
    from data_processing.frames import get_meta
    from data_processing.streams import frame_to_sample, iter_published_frames

    parents = gen["parents"]
    for parent, key, rps_entry in gen["recordings"]:
        name, version = _split_dload_uri(parents[parent])
        seen = False
        for frame in iter_published_frames(name, version):
            rid = str(get_meta(frame, "recording_id", ""))
            if rid != key:
                continue
            seen = True
            yield (
                key,
                frame_to_sample(_beatvk_frame(name, version or "lock-pin", key, rps_entry, frame)),
            )
            break
        if not seen:
            raise KeyError(f"recording {key!r} not found in {name}")


# ─── decomp-frames (adopt-only: a cluster VK solve sits in the loop) ─────────

#: The decomposition's envelope grid stride, in audio samples
#: (``tracking.fitness_vk.FVKConfig.fs_env`` = 100 Hz at 16 kHz). Every
#: window of ``scripts/vk_decompose.py`` is snapped to it, so the whole
#: recording's envelope grid is one uniform ``(sample_rate, stride)`` grid.
_DECOMP_ENV_STRIDE = 160
#: The protocol frame grid the refined labels sit on (``vk_decompose.HOP_S``).
_DECOMP_HOP_S = 0.032


def _decomp_frame_grid(n_t: int, sr: int) -> np.ndarray:
    """Verbatim ``scripts/vk_decompose.py::frame_grid`` — the label frame grid.

    Duplicated rather than imported: ``scripts/`` is never importable from
    ``src/`` (see the scripts table in AGENTS.md), and the two implementations
    are pinned together by ``tests/data_processing/test_decomp_frames.py``,
    which asserts they agree on the same input.
    """
    return np.arange(0.0, n_t / float(sr) - _DECOMP_HOP_S / 2, _DECOMP_HOP_S)


def _decomp_interp_rps(vals: Any, stamps: Any, ft: Any) -> np.ndarray:
    """Verbatim ``scripts/vk_decompose.py::interp_rps`` (see above on duplication)."""
    ts = np.asarray(stamps, dtype=np.float64)
    _, uniq = np.unique(ts, return_index=True)
    uniq = np.sort(uniq)
    ts = ts[uniq]
    v = np.asarray(vals, dtype=np.float64)[:, uniq]
    q = np.clip(np.asarray(ft, dtype=np.float64), ts[0], ts[-1])
    return np.stack([np.interp(q, ts, row) for row in v])


def _decomp_to_audio_grid(r: Any, ft: Any, n_t: int, sr: int) -> np.ndarray:
    """Verbatim ``scripts/vk_decompose.py::to_audio_grid`` (see above)."""
    r2 = np.atleast_2d(np.asarray(r, dtype=np.float64))
    t_audio = np.arange(n_t, dtype=np.float64) / float(sr)
    return np.stack([np.interp(t_audio, np.asarray(ft, dtype=np.float64), row) for row in r2])


def decomp_carrier(frame: td.Frame, rps_key: str, sr: int) -> np.ndarray:
    """The audio-rate rotor-speed CARRIER the decomposition was solved against.

    ``(R, T)``, reproducing ``scripts/vk_decompose.py::load_recordings`` exactly:
    telemetry (already label-overridden and trimmed by
    ``noise_rps_dataset.load_published_noise_sources``) is interpolated onto the
    0.032 s protocol frame grid first and only then onto the audio grid. Doing it
    in one step instead would give a *different* piecewise-linear function, and
    the amplitude targets are only meaningful against the carrier that produced
    them.
    """
    audio_s = frame["audio"]
    rps_s = frame[rps_key]
    t0 = int(audio_s.tindex.t_start_ticks)
    ticks = np.asarray(rps_s.tindex.abs_stamps_ticks, dtype=np.int64)
    stamps = (ticks - t0) / float(td.TICKS_PER_SECOND)
    n_t = int(np.asarray(audio_s.data).shape[-1])
    ft = _decomp_frame_grid(n_t, sr)
    r_ref = _decomp_interp_rps(np.asarray(rps_s.data), stamps, ft)
    return _decomp_to_audio_grid(r_ref, ft, n_t, sr)


def dense_envelopes(
    amp: np.ndarray, valid: np.ndarray, rotor: np.ndarray, k: np.ndarray, k_max: int
) -> tuple[np.ndarray, np.ndarray]:
    """Sparse ``(mic, track, time)`` envelopes -> dense ``(mic, rotor, k, time)``.

    The track set of a recording is ``rotor x k`` up to that recording's own
    ``k_hi`` (``vk_decompose.recording_k_hi``), so two recordings have different
    track counts and could not be collated into one training batch. The dense
    layout fixes the shape at ``k_max`` for every recording, indexes directly as
    ``[mic, rotor, k - 1, t]`` — the same order the emitter's ``harm_amps``
    carries — and carries its own validity mask, which is ``False`` both above a
    recording's ``k_hi`` and on frames no solve window covered.
    """
    rot = np.asarray(rotor, dtype=np.int64)
    ks = np.asarray(k, dtype=np.int64)
    keep = ks <= int(k_max)
    n_mic, _, n_env = amp.shape
    n_rotor = int(rot.max()) + 1
    dense = np.zeros((n_mic, n_rotor, int(k_max), n_env), dtype=np.float32)
    mask = np.zeros((n_rotor, int(k_max), n_env), dtype=bool)
    dense[:, rot[keep], ks[keep] - 1, :] = np.asarray(amp, dtype=np.float32)[:, keep]
    mask[rot[keep], ks[keep] - 1, :] = np.asarray(valid, dtype=bool)[keep]
    return dense, mask


def _decomp_artifact(gen: dict[str, Any], rid: str, name: str) -> str:
    """Local path of one ``artifacts/vk-decompose/<rid>/<name>`` R2 object."""
    from utils.checkpoints import resolve_checkpoint_uri

    uri = f"r2://{gen['artifact_bucket']}/{gen['artifact_prefix']}/{rid}/{name}"
    return resolve_checkpoint_uri(uri, cache_dir=".cache/vk_decompose_artifacts")


def _decomp_frame(gen: dict[str, Any], source: dict[str, Any], src: Any) -> tuple[str, td.Frame]:
    """One decomposed recording -> the published ``decomp-frames-v1`` Frame.

    Everything is re-anchored to the decomposition span: ``t = 0`` is
    ``span_samples[0]`` of the solve, so the audio-rate carrier, the 100 Hz
    envelope bank and the residual all start together and one
    ``frame.time[a:b]`` cut stays aligned across the three rates.
    """
    import json

    from data_processing.frames import make_recording_frame, meta_dict

    sr = int(gen["sample_rate"])
    rid = str(meta_dict(src.frame).get("recording_id"))
    env = np.load(_decomp_artifact(gen, rid, "envelopes.npz"))
    res = np.load(_decomp_artifact(gen, rid, "residual.npz"))
    report = json.loads(Path(_decomp_artifact(gen, rid, "report.json")).read_text())

    a0, a1 = (int(v) for v in np.asarray(env["span_samples"]))
    env_sr, env_stride = int(env["sample_rate"]), int(env["stride"])
    # A decomposition solved at an integer multiple of the spec rate (the v2
    # solve runs at 32 kHz so the line cap reaches 8 kHz = the 16 kHz Nyquist)
    # joins by decimation: the 100 Hz envelope grid is rate-agnostic, the
    # residual is resampled, and the span converts exactly because the span
    # start sits on the envelope grid.
    decim, rem = divmod(env_sr, sr)
    if rem or decim < 1 or env_stride != _DECOMP_ENV_STRIDE * decim:
        raise ValueError(
            f"{rid}: decomposition grid ({env_sr} Hz / stride {env_stride}) "
            f"does not match this spec ({sr} Hz / stride {_DECOMP_ENV_STRIDE})"
        )
    if a0 % env_stride:
        raise ValueError(f"{rid}: decomposition span start {a0} is not on the envelope grid")
    amp, mask = dense_envelopes(
        np.asarray(env["amp"]), np.asarray(env["valid"]), env["rotor"], env["k"], gen["k_max"]
    )
    residual = np.asarray(res["residual"], dtype=np.float64)
    if residual.shape[-1] != a1 - a0:
        raise ValueError(f"{rid}: residual length disagrees with the span {(a0, a1)}")
    if decim > 1:
        from scipy.signal import resample_poly

        residual = resample_poly(residual, 1, decim, axis=-1)
        a0, a1 = a0 // decim, a1 // decim
    residual = np.asarray(residual, dtype=np.float32)
    carrier = decomp_carrier(src.frame, src.rps_key, sr)[:, a0:a1]
    n_env = amp.shape[-1]
    if residual.shape[-1] != a1 - a0 or carrier.shape[-1] != a1 - a0:
        raise ValueError(f"{rid}: residual/carrier length disagrees with the span {(a0, a1)}")

    grid = td.GridIndex.create((sr, _DECOMP_ENV_STRIDE), n_env, t_start=0.0)
    tracks = {
        "rps": td.uniform(carrier.astype(np.float32), sr, dims=("rotor", "time"), t_start=0.0),
        "residual": td.uniform(residual, sr, dims=("mic", "time"), t_start=0.0),
        "amp": td.Series(amp, ("mic", "rotor", "k", "time"), {"time": grid}),
        "amp_valid": td.Series(mask, ("rotor", "k", "time"), {"time": grid}),
    }
    mic_pos, rotor_pos = _decomp_geometry(source["parent_uri"])
    meta = {
        "recording_id": rid,
        "drone": source["drone"],
        "split": source["drone"],
        "sample_rate": sr,
        "n_channels": int(residual.shape[0]),
        "env_rate_hz": sr / float(_DECOMP_ENV_STRIDE),
        "env_stride": _DECOMP_ENV_STRIDE,
        "k_max": int(gen["k_max"]),
        "k_hi": int(report["k_hi"]),
        "n_tracks": int(report["n_tracks"]),
        "source_spec": source["spec"],
        "source_rps_key": source["rps_key"],
        "label_variant": "refined" if source.get("rps_override_dir") else "published",
        # The span this frame covers, in the SOURCE frame's own time base, so a
        # consumer can line it up with the parent recording's audio (which this
        # dataset deliberately does not duplicate).
        "source_span_s": [round(a0 / float(sr), 6), round(a1 / float(sr), 6)],
        "source_t0_offset_s": float(env["t0_offset_s"]),
        "energy": {
            "track_fraction": report["energy"]["track_fraction"],
            "residual_fraction": report["energy"]["residual_fraction"],
            "band_share_of_tracks": report["energy"]["band_share_of_tracks"],
        },
        "bw_rps": report["params"]["bw_rps"],
        "decomposer": gen["decomposer"],
    }
    env.close()
    res.close()
    return rid, make_recording_frame(tracks, meta=meta, mic_pos=mic_pos, rotor_pos=rotor_pos)


def _decomp_geometry(parent_uri: str) -> tuple[np.ndarray, np.ndarray]:
    """The array geometry of a pinned parent frames dataset.

    Read off the published recordings (``mic_pos``/``rotor_pos`` baked in by
    ``frames.make_recording_frame``) — the same source
    ``frame_datasets._frames_spec_geometry`` reads for a ``frames:`` spec, so an
    amplitude-target model and an audio-target model see identical positions,
    and no raw tree is needed.
    """
    from data_processing.streams import iter_published_frames

    name, version = _split_dload_uri(parent_uri)
    for tf in iter_published_frames(name, version):
        if "mic_pos" in tf and "rotor_pos" in tf:
            return (
                np.asarray(tf["mic_pos"].data, dtype=np.float64),
                np.asarray(tf["rotor_pos"].data, dtype=np.float64),
            )
    raise ValueError(f"no recording in {parent_uri!r} carries mic_pos/rotor_pos geometry")


def generate_decomp_frames(gen: dict[str, Any]) -> Iterator[Sample]:
    """Vold-Kalman decompositions of the real drone recordings, as frames.

    Per recording: the per-``(mic, rotor, harmonic)`` amplitude envelope bank at
    100 Hz, its validity mask, the broadband residual waveform, and the exact
    audio-rate rotor-speed carrier the solve used. NOT re-derivable here — the
    envelopes come from a multi-hour cluster solve
    (``scripts/vk_decompose.py``, pinned in the spec's ``decomposer`` field),
    whose outputs live as R2 artifacts; this generator joins those artifacts to
    the pinned parent frames datasets and is run ONCE to publish the result.
    """
    from data_processing.frames import meta_dict
    from data_processing.noise_rps_dataset import load_published_noise_sources
    from data_processing.streams import frame_to_sample

    for source in gen["sources"]:
        parent_uri = gen["parents"][source["parent"]]
        name, version = _split_dload_uri(parent_uri)
        spec = f"frames:{name}@{version}" if version else f"frames:{name}"
        wanted = list(source["recording_ids"])
        found: list[str] = []
        for src in load_published_noise_sources(
            spec,
            int(gen["sample_rate"]),
            origin=source["drone"],
            rps_key=source["rps_key"],
            splits=source.get("splits"),
            rps_override_dir=source.get("rps_override_dir"),
        ):
            if str(meta_dict(src.frame).get("recording_id")) not in wanted:
                continue
            rid, frame = _decomp_frame(gen, {**source, "spec": spec, "parent_uri": parent_uri}, src)
            found.append(rid)
            yield rid, frame_to_sample(frame)
        missing = sorted(set(wanted) - set(found))
        if missing:
            raise KeyError(f"{spec}: no decomposed recording for {missing}")


def _decomp_spec(artifact_prefix: str, *, recipe_version: int, note: str) -> dict[str, Any]:
    """One ``decomp-frames-v*`` spec — everything but the decomposition run.

    The join (which recordings, which labels, which parents, the dense ``k``
    grid) is identical across decomposition versions; what changes is the solve
    that produced the artifacts, which is exactly ``artifact_prefix`` plus the
    ``recipe_version`` that mints a fresh derivation identity.
    """
    return {
        "generator": "decomp_frames",
        "adopt_only": True,
        "note": note,
        "gen": {
            "recipe_version": int(recipe_version),
            "parents": {
                "dregon": PARENTS["DREGON-frames"],
                "michaels": PARENTS["michaels-frames"],
            },
            "decomposer": "scripts/vk_decompose.py@508ffcb",
            "artifact_bucket": "ml-data",
            "artifact_prefix": artifact_prefix,
            "sample_rate": 16000,
            "k_max": 80,
            "sources": [
                {
                    "parent": "dregon",
                    "drone": "dregon",
                    "rps_key": "motors_measured",
                    "splits": ["in_flight_noise"],
                    "rps_override_dir": "src/data_processing/refined_labels",
                    "recording_ids": ["free-flight_nosource_room1"],
                },
                {
                    "parent": "michaels",
                    "drone": "michaels",
                    "rps_key": "rps",
                    "splits": None,
                    "rps_override_dir": None,
                    "recording_ids": ["FLY124", "FLY125"],
                },
            ],
        },
    }


# ─── AVQ-egonoise-vkrps (adopt-only: GPU annotator in the loop) ───────────────


def generate_avq_vkrps(gen: dict[str, Any]) -> Iterator[Sample]:
    """AVQ-egonoise joined with blind-VK RPS pseudo-labels. NOT re-derivable
    here: the labels come from running the blind-VK annotator (a GPU model) on
    the audio — the spec pins the annotator commit in its ``note`` and the
    dataset is adopted in place. This generator exists so the spec has a
    stable fingerprint identity; running it raises."""
    raise RuntimeError(
        "AVQ-egonoise-vkrps is adopt-only: its RPS pseudo-labels are produced by "
        "the blind-VK annotator (see the spec note for the pinned annotator "
        "commit); it cannot be re-materialized by a pure dload pipeline."
    )
    yield  # pragma: no cover - generator form required for fingerprinting


# ─── SE validation sets (specs; generator body wired to the stream builders) ──

#: LibriSpeech train-clean-100 speaker ids reserved for SE validation (every
#: 10th id, sorted numerically — 25 of 246). Training speech excludes these
#: (policy ``sources.speech[].exclude``), so no speaker is shared train/valid.
SE_HELDOUT_SPEAKERS = [
    "19", "89", "211", "307", "445", "831", "1098", "1502", "1898", "2182",
    "2764", "3168", "3526", "3982", "4267", "4813", "5322", "5703", "6081",
    "6476", "7067", "7312", "7800", "8238", "8630",
]  # fmt: skip

#: Valid-side noise pools per SE category (the same spec schema the online-mix
#: stream builders consume). ``holdout`` reserves whole shards (= whole
#: recording groups) as the valid partition; training pools use the
#: complementary ``split: train`` side. Real DREGON/michaels drone noise is
#: held out by *recording id* (valid: room1 + FLY124; train: the rest/FLY125).
_SE_HOLDOUT_VALID = {"split": "valid", "valid_shards": 2}
#: The complementary (training) side — what the models actually saw.
_SE_HOLDOUT_TRAIN = {"split": "train", "valid_shards": 2}
#: LibriSpeech subtree the speaker-split speech is drawn from.
_LIBRISPEECH_SUBPATH = "LibriSpeech/train-clean-100"
SE_CATEGORY_NOISE: dict[str, list[dict[str, Any]]] = {
    "drone": [
        {
            "kind": "frames",
            "dataset": "DREGON-frames",
            "recording_ids": ["free-flight_nosource_room1"],
            "min_motor_rps": 30.0,
        },
        {
            "kind": "frames",
            "dataset": "michaels-frames",
            "recording_ids": ["FLY124"],
            "min_motor_rps": 0.0,
        },
        {"kind": "audio_pool", "dataset": "drone_audio", "holdout": _SE_HOLDOUT_VALID},
        {"kind": "audio_pool", "dataset": "DroneAudioSet", "holdout": _SE_HOLDOUT_VALID},
    ],
    "mimii": [{"kind": "audio_pool", "dataset": "MIMII", "holdout": _SE_HOLDOUT_VALID}],
    "mimii_dg": [{"kind": "audio_pool", "dataset": "MIMII-DG", "holdout": _SE_HOLDOUT_VALID}],
    "aircraft": [{"kind": "audio_pool", "dataset": "AeroSonicDB", "holdout": _SE_HOLDOUT_VALID}],
    "motors": [
        {"kind": "audio_pool", "dataset": "HUSTmotor", "holdout": _SE_HOLDOUT_VALID},
        {"kind": "audio_pool", "dataset": "KAIST-rotating-acoustic", "holdout": _SE_HOLDOUT_VALID},
    ],
    "horns": [{"kind": "audio_pool", "dataset": "HornBase", "holdout": _SE_HOLDOUT_VALID}],
    # The generalization probe's IN-DISTRIBUTION arm: a byte-for-byte mirror of
    # the F1 Pass-A *training* noise pool (conf/online_mix/se_drone_only.yaml) —
    # the complementary side of every holdout the `drone` category uses. Paired
    # with the SAME held-out speakers, so the only thing differing between
    # `drone_seen` and `drone` is whether the model trained on that noise.
    # Consumed on demand by notebooks/generalization_lib.py; NOT published.
    "drone_seen": [
        {
            "kind": "frames",
            "dataset": "DREGON-frames",
            "splits": ["in_flight_noise"],
            "exclude_recording_ids": ["free-flight_nosource_room1"],
            "min_motor_rps": 30.0,
        },
        {
            "kind": "frames",
            "dataset": "michaels-frames",
            "recording_ids": ["FLY125"],
            "min_motor_rps": 0.0,
        },
        {"kind": "audio_pool", "dataset": "drone_audio", "holdout": _SE_HOLDOUT_TRAIN},
        {"kind": "audio_pool", "dataset": "DroneAudioSet", "holdout": _SE_HOLDOUT_TRAIN},
        {"kind": "audio_pool", "dataset": "SPCUP19-egonoise", "holdout": _SE_HOLDOUT_TRAIN},
        {"kind": "audio_pool", "dataset": "new-drone-noises", "holdout": _SE_HOLDOUT_TRAIN},
    ],
    # F2 replication: no noise holdout on purpose — the paper reuses the same 5
    # ego-noise recordings for train and valid and splits only the speech.
    "avq_ego": [{"kind": "audio_pool", "dataset": "AVQ-egonoise"}],
    "avq_ego_s1": [
        {
            "kind": "audio_pool",
            "dataset": "AVQ-egonoise",
            "include_keys": ["S1_seq1", "S1_seq2", "S1_seq3"],
        }
    ],
    "avq_ego_s2": [
        {"kind": "audio_pool", "dataset": "AVQ-egonoise", "include_keys": ["S2_seq1", "S2_seq2"]}
    ],
}


def _se_valid_frame(
    category: str,
    sample_rate: int,
    snr_grid: list[float],
    per_snr: int,
    index: int,
    noise_tf: td.Frame,
    speech_lanes: list[np.ndarray],
    u_channel: float,
) -> td.Frame:
    """One fixed SE-valid clip: mono noise + speech scaled to the grid SNR."""
    from data_processing.frames import audio_series
    from data_processing.mixing import scale_source_to_snr

    snr = float(snr_grid[index // per_snr])
    noise = np.asarray(noise_tf["audio"].data, dtype=np.float32)
    if noise.ndim == 2:
        noise = noise[int(float(u_channel) * noise.shape[0]) % noise.shape[0]]
    noise = np.ascontiguousarray(noise)[None, :]
    speech = np.asarray(speech_lanes[0], np.float32)[None, :]
    # The noise window may run one sample long (inclusive time-slice bounds);
    # speech lanes are cut to the exact target length. Crop both to the
    # common length so the mix broadcasts.
    n = min(noise.shape[-1], speech.shape[-1])
    noise = noise[:, :n]
    speech = speech[:, :n]
    target = scale_source_to_snr(speech, noise, snr)
    sample_id = f"{category}_snr{int(snr):+03d}_{index:04d}"
    return td.Frame(
        {
            "mixture": audio_series((noise + target).astype(np.float32), sample_rate),
            "target": audio_series(target, sample_rate),
            "meta": td.Frame(
                {
                    "recording_id": sample_id,
                    "id": sample_id,
                    "input_snr": snr,
                    "category": category,
                }
            ),
        }
    )


def iter_se_valid_category(
    category: str,
    noise_specs: list[dict[str, Any]],
    *,
    per_snr: int,
    snr_grid: list[float],
    duration_s: float,
    sample_rate: int,
    seed: int,
    heldout_speakers: list[str],
    librispeech: str,
) -> Iterator[tuple[str, td.Frame]]:
    """Yield one category's fixed SE-valid clips as ``(sample_id, Frame)``.

    Built from the *same* stream builders the training policies compile to
    (:mod:`data_processing.online_mixing`), with the complementary filters:
    the noise pools take the ``split: valid`` side of every holdout and the
    speech stream ``include``s exactly the held-out speakers. Silent draws are
    filtered upstream — a silent noise draw would zero both the mixture and
    the clean target through the source-to-noise scaling (the bug that
    invalidated the v1 sets; see ``mixing.MIN_DRAW_POWER``).

    Public so the generalization probe (``notebooks/generalization_lib.py``)
    can materialize unpublished categories (e.g. ``drone_seen``) on demand.
    """
    import itertools

    from data_processing.online_mixing import (
        _nonsilent_frame,
        _nonsilent_lanes,
        build_noise_stream,
        build_speech_stream,
    )

    cseed = dload.seeded(seed, "se-valid", category)
    target_len = int(round(duration_s * sample_rate))
    noise, _ = build_noise_stream(
        noise_specs, sample_rate=sample_rate, window_s=duration_s, seed=cseed
    )
    speech = build_speech_stream(
        {
            "dataset": _split_dload_uri(librispeech)[0],
            "version": _split_dload_uri(librispeech)[1],
            "subpath": _LIBRISPEECH_SUBPATH,
            "include": list(heldout_speakers),
        },
        sample_rate=sample_rate,
        window_s=duration_s,
        lanes=1,
        seed=dload.seeded(cseed, "speech"),
    )
    pipe = dload.zip_with(
        partial(_se_valid_frame, category, sample_rate, list(snr_grid), int(per_snr)),
        dload.from_iterable(itertools.count),
        noise.filter(partial(_nonsilent_frame, target_len)),
        speech.filter(_nonsilent_lanes),
        dload.random_stream(dload.seeded(cseed, "channel")),
    )
    for frame in itertools.islice(pipe, int(per_snr) * len(snr_grid)):
        yield str(frame["meta"]["id"]), frame


def generate_se_valid(gen: dict[str, Any]) -> Iterator[Sample]:
    """A fixed SE validation set: held-out noise x held-out LibriSpeech
    speakers over an SNR grid — ``{mixture, target, meta}`` frames.

    ``gen``: ``categories`` + ``category_noise`` (the per-category valid-side
    noise specs), ``heldout_speakers``, ``librispeech`` (pinned dload URI),
    ``per_snr`` / ``snr_grid`` / ``duration_s`` / ``sample_rate``, ``seed``.
    """
    from data_processing.streams import frame_to_sample

    for category in gen["categories"]:
        for sample_id, frame in iter_se_valid_category(
            category,
            gen["category_noise"][category],
            per_snr=int(gen["per_snr"]),
            snr_grid=[float(s) for s in gen["snr_grid"]],
            duration_s=float(gen["duration_s"]),
            sample_rate=int(gen["sample_rate"]),
            seed=int(gen["seed"]),
            heldout_speakers=list(gen["heldout_speakers"]),
            librispeech=str(gen["librispeech"]),
        ):
            yield sample_id, frame_to_sample(frame)


# ─── Spec registry ────────────────────────────────────────────────────────────
#
# One entry per derived dataset. ``gen`` is the fingerprinted sub-spec (feeds
# ``partial(generator, gen)``); everything else (``generator``, ``fields``,
# ``adopt_only``, ``note``) is registry metadata. ``adopt_only`` datasets are
# published by pointing their derivation ref at an already-uploaded historical
# version (``scripts/derive.py adopt``) rather than re-materialized.

_DREGON_LM_FIELDS = {
    "mixture": "mixture.wav",
    "vocals": "vocals.wav",
    "noise": "noise.wav",
    "rps": "rps.npy",
    "meta": "meta.json",
}
_REAL_VALID_FIELDS = {
    "mixture": "mixture.wav",
    "rps": "rps.npy",
}
_DN_LM_FIELDS = {
    "mixture": "mixture.wav",
    "vocals": "vocals.wav",
    "noise": "noise.wav",
    "meta": "meta.json",
}

# Shared V4 mixing knobs (the DREGON-LM-V4 recipe).
_V4_PARAMS = {
    "sample_duration": 1.0,
    "sample_rate": 16000,
    "snr_range": [-30.0, 0.0],
    "speech_per_channel": "independent",
    "source_white_noise_prob": 0.3,
    "white_noise_prob": 0.0,
    "white_noise_snr": 30.0,
    "min_motor_rps": 30.0,
    "speech_subpath": "LibriSpeech/train-clean-100",
}
_V4_VALID_PARAMS = {
    **_V4_PARAMS,
    "sample_duration": 8.0,
}

#: The valid-side real-recording pool shared by the V4 valid specs.
_V4_VALID_SOURCES = [
    {
        "dataset": PARENTS["DREGON-frames"],
        "recording_ids": ["free-flight_speech-low_room1", "free-flight_whitenoise-low_room1"],
    }
]
_V4_MICHAELS_VALID_SOURCES = [
    {
        "dataset": PARENTS["DREGON-frames"],
        "recording_ids": [
            "free-flight_nosource_room1",
            "free-flight_speech-low_room1",
            "free-flight_whitenoise-low_room1",
        ],
    },
    {"dataset": PARENTS["michaels-frames"], "recording_ids": ["FLY124"]},
]

SPECS: dict[str, dict[str, Any]] = {
    # ── DREGON-LM V4 (active pins; adopted in place) ────────────────────────
    "DREGON-LM-V4-train": {
        "generator": "dregon_lm",
        "fields": _DREGON_LM_FIELDS,
        "adopt_only": True,
        "note": "Adopt-in-place: re-derivation would upload a near-duplicate "
        "(RNG not byte-stable). Generator reproduces the synthesized train "
        "recipe but the published bytes are the historical upload.",
        "gen": {
            "recipe_version": 1,
            "seed": 42,
            "num_samples": 6000,
            "split": "train",
            "noise_sources": None,
            "parents": {"librispeech": PARENTS["librispeech"]},
            "params": _V4_PARAMS,
        },
    },
    "DREGON-LM-V4-valid": {
        "generator": "dregon_lm",
        "fields": _REAL_VALID_FIELDS,
        "adopt_only": True,
        "note": "Adopt-in-place. The V4 valid split is real-valid raw clips "
        "(speech-low + whitenoise-low, 30 random 8 s clips); the real_valid "
        "mode reproduces the recipe.",
        "gen": {
            "recipe_version": 1,
            "seed": 43,
            "num_samples": 30,
            "split": "valid",
            "mode": "real_valid",
            "noise_sources": _V4_VALID_SOURCES,
            "parents": {},
            "params": _V4_VALID_PARAMS,
        },
    },
    "DREGON-LM-V4-michaels-train": {
        "generator": "dregon_lm",
        "fields": _DREGON_LM_FIELDS,
        "adopt_only": True,
        "note": "Adopt-in-place. Composed noise pool (DREGON in_flight_noise + "
        "michaels FLY125) over published frames; 9000 samples as published.",
        "gen": {
            "recipe_version": 1,
            "seed": 42,
            "num_samples": 9000,
            "split": "train",
            "noise_sources": [
                {"dataset": PARENTS["DREGON-frames"], "splits": ["in_flight_noise"]},
                {"dataset": PARENTS["michaels-frames"], "recording_ids": ["FLY125"]},
            ],
            "parents": {"librispeech": PARENTS["librispeech"]},
            "params": _V4_PARAMS,
        },
    },
    "DREGON-LM-V4-michaels-valid": {
        "generator": "dregon_lm",
        "fields": _REAL_VALID_FIELDS,
        "adopt_only": True,
        "note": "Adopt-in-place. Real-valid clips over DREGON room1 recordings "
        "+ michaels FLY124 (28 clips as published).",
        "gen": {
            "recipe_version": 1,
            "seed": 43,
            "num_samples": 28,
            "split": "valid",
            "mode": "real_valid",
            "noise_sources": _V4_MICHAELS_VALID_SOURCES,
            "parents": {},
            "params": _V4_VALID_PARAMS,
        },
    },
    "DREGON-LM-V4-michaels-valid-full": {
        "generator": "dregon_lm",
        "fields": _REAL_VALID_FIELDS,
        "adopt_only": True,
        "note": "Adopt-in-place. FULL-envelope real valid (min_motor_rps 0.0, "
        "max_non_overlapping; spans warm-up/takeoff/cruise/landing/ground) — "
        "37 clips as published.",
        "gen": {
            "recipe_version": 1,
            "seed": 43,
            "num_samples": 37,
            "split": "valid",
            "mode": "real_valid",
            "noise_sources": _V4_MICHAELS_VALID_SOURCES,
            "parents": {},
            "params": {**_V4_VALID_PARAMS, "min_motor_rps": 0.0, "max_non_overlapping": True},
        },
    },
    "DREGON-LM-V4-michaels-valid-full-nospeech": {
        "generator": "dregon_lm_subset",
        "fields": _REAL_VALID_FIELDS,
        "adopt_only": False,
        "note": "Noise-only twin of DREGON-LM-V4-michaels-valid-full: its 23 "
        "source_type=nosource clips (free-flight_nosource_room1 + michaels "
        "FLY124), copied byte for byte, so audio and rps match the full set "
        "exactly. real_valid does no mixing, so a clip's mixture IS the raw "
        "8-mic recording; the speech and the white noise of the other 14 clips "
        "are a loudspeaker in the room during the DREGON in_flight_source "
        "flights, and no noise-only version of those windows exists. Cut from "
        "the published bytes and not re-derived, because the parent's own "
        "generator no longer reproduces its labels (it predates the michaels "
        "calibration of 2026-07-31 and the motors_measured preference of "
        "2026-08-05).",
        "gen": {
            "recipe_version": 1,
            "parent": PARENTS["DREGON-LM-V4-michaels-valid-full"],
            "split": "valid",
            "fields": _REAL_VALID_FIELDS,
            "keep_source_types": ["nosource"],
        },
    },
    # ── DN-LM (derived; recipe_version 1 materialized 2026-07) ──────────────
    # Noise source = drone-only: `drone_audio/Binary_Drone_Audio/yes_drone`
    # (the 1332 label-1 recordings) — the raw tree mixes ESC-50/WN/silence
    # negatives under */unknown/, which the paper's DN-LM excluded. Sizes
    # 6480/720 follow the paper's "2 hours". NOTE: the materialized pins were
    # derived against the PRE-REPAIR librispeech parent (one truncated flac,
    # since republished); the spec keeps that pin so the derivation identity
    # keeps hitting the materialized version. Bump recipe_version and re-pin
    # the parent together for any future re-derivation.
    "DN-LM-train": {
        "generator": "dn_lm",
        "fields": _DN_LM_FIELDS,
        "adopt_only": False,
        "note": "Drone-only noise from drone_audio/Binary_Drone_Audio/yes_drone "
        "(label-1 recordings; excludes ESC-50/WN 'unknown'). Bump "
        "recipe_version on any recipe change.",
        "gen": {
            "recipe_version": 1,
            "seed": 42,
            "num_samples": 6480,
            "split": "train",
            "parents": {
                "librispeech": "dload:librispeech@fda9aad8dfb545c82752f33e8ff563feaee82f5f9ba7b50efafff4cd8ff73ae5",
                "noise": PARENTS["drone_audio"],
            },
            "params": {
                "sample_duration": 1.0,
                "sample_rate": 16000,
                "target_snr_range": [-30.0, 0.0],
                "speech_distance_range": [5.0, 20.0],
                "noise_distance": 0.5,
                "speech_subpath": "LibriSpeech/train-clean-100",
                "noise_subpath": "Binary_Drone_Audio/yes_drone",
            },
        },
    },
    "DN-LM-valid": {
        "generator": "dn_lm",
        "fields": _DN_LM_FIELDS,
        "adopt_only": False,
        "note": "Drone-only noise from drone_audio/Binary_Drone_Audio/yes_drone "
        "(label-1 recordings; excludes ESC-50/WN 'unknown'). Bump "
        "recipe_version on any recipe change.",
        "gen": {
            "recipe_version": 1,
            "seed": 43,
            "num_samples": 720,
            "split": "valid",
            "parents": {
                "librispeech": "dload:librispeech@fda9aad8dfb545c82752f33e8ff563feaee82f5f9ba7b50efafff4cd8ff73ae5",
                "noise": PARENTS["drone_audio"],
            },
            "params": {
                "sample_duration": 1.0,
                "sample_rate": 16000,
                "target_snr_range": [-30.0, 0.0],
                "speech_distance_range": [5.0, 20.0],
                "noise_distance": 0.5,
                "speech_subpath": "LibriSpeech/train-clean-100",
                "noise_subpath": "Binary_Drone_Audio/yes_drone",
            },
        },
    },
    # ── Source frames (uniform: every source's builder as a derivation) ─────
    "DREGON-frames": {
        "generator": "source_frames",
        "adopt_only": True,
        "note": "Adopt-in-place (published by the deleted "
        "scripts/publish_frame_datasets.py; the sources.dregon builder "
        "reproduces it).",
        "gen": {
            "recipe_version": 1,
            "source": "DREGON",
            "raw": {"kind": "dload", "uri": PARENTS["DREGON"]},
        },
    },
    "michaels-test-frames": {
        "generator": "source_frames",
        "adopt_only": False,
        "note": "The HELD-OUT TEST recordings of Michael's rig: FLY103/FLY108 "
        "(raw tree `new-drone-noises`), MONO at native 48 kHz, same rich layout "
        "as `michaels-frames` minus mic_pos. Derivable — unlike michaels-frames "
        "these bytes have never been published, so the builder IS the recipe. "
        "recipe_version 1 = the 2026-08 calibration (coarse comb alignment then "
        "the WP13/WP14 VK fit; MICHAELS_TEST_FILES + MICHAELS_RPS_SCALE). "
        "Reserved as a TEST set: no training derivation may root on it.",
        "gen": {
            "recipe_version": 1,
            "source": "michaels-test",
            "raw": {
                "kind": "dload",
                "uri": "dload:new-drone-noises@158b780aaaf27af03ed995707d8a93e62e3fce7246132b815d52cfc225dcc774",
            },
        },
    },
    "michaels-frames": {
        "generator": "source_frames",
        "adopt_only": True,
        "note": "Adopt-in-place (published by the deleted "
        "scripts/publish_frame_datasets.py). The sources.michaels builder "
        "reproduces the tracks and the meta, except that the provenance strings "
        "name the current builder rather than the deleted script. "
        "recipe_version 2 = the measured telemetry calibration of 2026-07-31 "
        "(MICHAELS_FILES offsets/dilations + the new MICHAELS_RPS_SCALE); "
        "version 1 frames carry uncalibrated labels, so every number derived "
        "from them is stale.",
        "gen": {
            "recipe_version": 2,
            "source": "michaels",
            "raw": {
                "kind": "dload",
                "uri": "dload:recording_with_motor_speed@5b7eab554710c3d83667085c8f5ca256322ec10e2cffae6002443f63b05257b4",
            },
        },
    },
    # ── External harmonic-noise frames (the 10 externals; adopted) ──────────
    "MIMII": {
        "generator": "source_frames",
        "adopt_only": True,
        "note": "Adopt-in-place; raw via the registry's pinned zenodo spec.",
        "gen": {"recipe_version": 1, "source": "MIMII", "raw": {"kind": "download"}},
    },
    "MIMII-DG": {
        "generator": "source_frames",
        "adopt_only": True,
        "note": "Adopt-in-place; raw via the registry's pinned zenodo spec.",
        "gen": {"recipe_version": 1, "source": "MIMII-DG", "raw": {"kind": "download"}},
    },
    "AeroSonicDB": {
        "generator": "source_frames",
        "adopt_only": True,
        "note": "Adopt-in-place; raw via the registry's pinned zenodo spec.",
        "gen": {"recipe_version": 1, "source": "AeroSonicDB", "raw": {"kind": "download"}},
    },
    "DroneAudioSet": {
        "generator": "source_frames",
        "adopt_only": True,
        "note": "Adopt-in-place; raw via the registry's pinned HF spec.",
        "gen": {"recipe_version": 1, "source": "DroneAudioSet", "raw": {"kind": "download"}},
    },
    "drone-detection-samples": {
        "generator": "source_frames",
        "adopt_only": True,
        "note": "Adopt-in-place; raw via the registry's pinned HF spec.",
        "gen": {
            "recipe_version": 1,
            "source": "drone-detection-samples",
            "raw": {"kind": "download"},
        },
    },
    "HornBase": {
        "generator": "source_frames",
        "adopt_only": True,
        "note": "Adopt-in-place; raw via the registry's pinned mendeley spec.",
        "gen": {"recipe_version": 1, "source": "HornBase", "raw": {"kind": "download"}},
    },
    "KAIST-rotating-acoustic": {
        "generator": "source_frames",
        "adopt_only": True,
        "note": "Adopt-in-place; raw via the registry's pinned mendeley spec.",
        "gen": {
            "recipe_version": 1,
            "source": "KAIST-rotating-acoustic",
            "raw": {"kind": "download"},
        },
    },
    "HUSTmotor": {
        "generator": "source_frames",
        "adopt_only": True,
        "note": "Adopt-in-place; raw via the registry's pinned gdrive spec.",
        "gen": {"recipe_version": 1, "source": "HUSTmotor", "raw": {"kind": "download"}},
    },
    "SPCUP19-egonoise": {
        "generator": "source_frames",
        "adopt_only": True,
        "note": "Adopt-in-place; raw via the registry's pinned http spec.",
        "gen": {"recipe_version": 1, "source": "SPCUP19-egonoise", "raw": {"kind": "download"}},
    },
    "AVQ": {
        "generator": "source_frames",
        "adopt_only": True,
        "note": "Adopt-in-place; raw via the registry's pinned http spec.",
        "gen": {"recipe_version": 1, "source": "AVQ", "raw": {"kind": "download"}},
    },
    # ── Purpose-built subsets / companions ───────────────────────────────────
    "AVQ-egonoise": {
        "generator": "frame_subset",
        "adopt_only": True,
        "note": "Adopt-in-place (published by the deleted "
        "scripts/publish_avq_egonoise.py). The 5 pure rotor ego-noise "
        "sequences of AVQ (no angle_vad entry), channel 0, 16 kHz mono.",
        "gen": {
            "recipe_version": 1,
            "parent": PARENTS["AVQ"],
            "include_keys": ["S1_seq1", "S1_seq2", "S1_seq3", "S2_seq1", "S2_seq2"],
            "forbid_entries": ["angle_vad"],
            "channel": 0,
            "sample_rate": 16000,
            "meta_updates": {
                "derived_note": "Derived subset of AVQ: the 5 pure rotor ego-noise "
                "sequences, channel 0 only, soxr_hq-resampled 44100 -> 16000 Hz mono."
            },
        },
    },
    "AVQ-raw": {
        "generator": "raw_subset",
        "adopt_only": True,
        "note": "Adopt-in-place (published by the deleted "
        "scripts/publish_avq_raw.py). Byte-exact AVQ companion: everything "
        "except the per-channel MONO-*.wav (the audio in AVQ).",
        "gen": {
            "recipe_version": 1,
            "source": "AVQ",
            "skip": {"extensions": [".wav"], "stem_prefix": "mono"},
        },
    },
    "AVQ-egonoise-vkrps": {
        "generator": "avq_vkrps",
        "adopt_only": True,
        "note": "Adopt-in-place. AVQ-egonoise joined with blind-VK RPS "
        "pseudo-labels from the annotator scripts/vk_pseudolabel.py @ "
        "fa5053fc (a GPU model sits in the loop — not re-derivable as a pure "
        "pipeline; see generate_avq_vkrps).",
        "gen": {
            "recipe_version": 1,
            "parent": PARENTS["AVQ-egonoise"],
            "annotator": "scripts/vk_pseudolabel.py@fa5053fc",
            "min_seg_s": 10.0,
        },
    },
    # ── VK decompositions of the real recordings ─────────────────────────────
    # Version-parameterized: one spec per DECOMPOSITION run. The v1 solve used a
    # flat 1 Hz envelope bandwidth for every track, which under-resolves the real
    # linewidth (it grows with k) — a measured share of the mid-k line energy
    # therefore leaks into what v1 calls the residual. v2 re-solves with a
    # linewidth-matched bandwidth schedule and lands under its own artifact
    # prefix; switching a training arm over is one `dataset:` line in
    # conf/data/decomp_frames*.yaml.
    "decomp-frames-v1": _decomp_spec(
        "artifacts/vk-decompose",
        recipe_version=1,
        note="Adopt-in-place. The coupled Vold-Kalman decomposition of the three "
        "decomposable real recordings (DREGON free-flight_nosource_room1 on the "
        "REFINED labels, Michael's FLY124/FLY125 on the recalibrated telemetry), as "
        "amplitude-envelope frames: per-(mic, rotor, harmonic) amplitude at 100 Hz + "
        "validity mask + the broadband residual waveform + the exact audio-rate carrier "
        "the solve used. Not re-derivable here — the envelopes come from a multi-hour "
        "cluster solve (scripts/vk_decompose.py @ 508ffcb, jobs vk-decompose-6bdede + "
        "vk-decompose-michaels-be071d) whose outputs are R2 artifacts; this generator "
        "joins them to the pinned parents and is run ONCE. FLAT 1 Hz per-track "
        "bandwidth, so mid/high-k line amplitudes are underestimated and the residual "
        "carries the leaked comb energy — see decomp-frames-v2. Consumers: the "
        "amplitude-target generator arms "
        "(docs/experiments/amplitude-target-training.md).",
    ),
    "decomp-frames-v2": _decomp_spec(
        "artifacts/vk-decompose-v2",
        recipe_version=2,
        note="Adopt-in-place, SAME join as decomp-frames-v1 over a re-run solve with a "
        "LINEWIDTH-MATCHED per-track bandwidth schedule (v1's flat 1 Hz clamp left a "
        "measured majority of the k10-24 stripe contrast in the residual). Materialize "
        "once the v2 artifacts are on R2 under artifacts/vk-decompose-v2/<recording>/.",
    ),
    # ── Fixed SE validation sets ─────────────────────────────────────────────
    "SE-valid-drone": {
        "generator": "se_valid",
        "adopt_only": False,
        "note": "Held-out drone noise x held-out LibriSpeech speakers over the SNR grid. "
        "recipe_version 2: rebuilt with the silent-draw filter (the v1 upload, "
        "adopted from the deleted scripts/build_se_valid.py, carried 5 all-zero "
        "clips from silent drone_audio noise draws — see mixing.MIN_DRAW_POWER).",
        "gen": {
            "recipe_version": 2,
            "seed": 20260720,
            "categories": ["drone"],
            "category_noise": {"drone": SE_CATEGORY_NOISE["drone"]},
            "heldout_speakers": SE_HELDOUT_SPEAKERS,
            "librispeech": PARENTS["librispeech"],
            "per_snr": 50,
            "snr_grid": [-30, -25, -20, -15, -10, -5, 0],
            "duration_s": 2.0,
            "sample_rate": 16000,
        },
    },
    "SE-valid-harmonic": {
        "generator": "se_valid",
        "adopt_only": False,
        "note": "Per-category harmonic-noise transfer table set. recipe_version 2: "
        "rebuilt with the silent-draw filter (the v1 upload carried 5 all-zero "
        "drone-category clips — see mixing.MIN_DRAW_POWER).",
        "gen": {
            "recipe_version": 2,
            "seed": 20260720,
            "categories": ["drone", "mimii", "mimii_dg", "aircraft", "motors", "horns"],
            "category_noise": {
                k: SE_CATEGORY_NOISE[k]
                for k in ("drone", "mimii", "mimii_dg", "aircraft", "motors", "horns")
            },
            "heldout_speakers": SE_HELDOUT_SPEAKERS,
            "librispeech": PARENTS["librispeech"],
            "per_snr": 50,
            "snr_grid": [-30, -25, -20, -15, -10, -5, 0],
            "duration_s": 2.0,
            "sample_rate": 16000,
        },
    },
    "SE-valid-avq-survey": {
        "generator": "se_valid",
        "adopt_only": True,
        "note": "Adopt-in-place. F2 survey-replication valid set (no noise "
        "holdout by design — the paper reuses the same 5 ego-noise recordings "
        "for train and valid).",
        "gen": {
            "recipe_version": 1,
            "seed": 20260720,
            "categories": ["avq_ego"],
            "category_noise": {"avq_ego": SE_CATEGORY_NOISE["avq_ego"]},
            "heldout_speakers": SE_HELDOUT_SPEAKERS,
            "librispeech": PARENTS["librispeech"],
            "per_snr": 50,
            "snr_grid": [-25, -20, -15, -10, -5],
            "duration_s": 3.0,
            "sample_rate": 16000,
        },
    },
    "SE-valid-avq-split": {
        "generator": "se_valid",
        "adopt_only": True,
        "note": "Adopt-in-place. Session-split memorisation probe (S1 seen / "
        "S2 unseen noise halves).",
        "gen": {
            "recipe_version": 1,
            "seed": 20260720,
            "categories": ["avq_ego_s1", "avq_ego_s2"],
            "category_noise": {k: SE_CATEGORY_NOISE[k] for k in ("avq_ego_s1", "avq_ego_s2")},
            "heldout_speakers": SE_HELDOUT_SPEAKERS,
            "librispeech": PARENTS["librispeech"],
            "per_snr": 50,
            "snr_grid": [-25, -20, -15, -10, -5],
            "duration_s": 3.0,
            "sample_rate": 16000,
        },
    },
    # ── Decoded-speech cache (derivable; not yet materialized) ──────────────
    "librispeech-pcm16": {
        "generator": "pcm16_mono",
        "adopt_only": False,
        "note": "Memoized mono int16 decode of the librispeech pin at 16 kHz — "
        "the derived-dataset replacement for the deleted bespoke packed_int16 "
        "speech cache. Point a policy's sources.speech[].dataset here to skip "
        "per-encounter FLAC decode (matters for 8-lane RPS streams).",
        "gen": {
            "recipe_version": 1,
            "parent": PARENTS["librispeech"],
            "subpath": _LIBRISPEECH_SUBPATH,
            "sample_rate": 16000,
        },
    },
    # ── beat-VK frozen validation set ────────────────────────────────────────
    "beatvk-valid-raw": {
        "generator": "beatvk_valid",
        "adopt_only": True,
        "note": "Adopt-in-place (published by the deleted "
        "scripts/publish_beatvk_valid.py). The frozen beat-VK raw validation "
        "set: 4 recordings, native-rate audio + raw measured telemetry + the "
        "16 s window manifest.",
        "gen": {
            "recipe_version": 1,
            "parents": {
                "DREGON-frames": PARENTS["DREGON-frames"],
                "michaels-frames": PARENTS["michaels-frames"],
            },
            "recordings": [
                ["DREGON-frames", "free-flight_nosource_room1", "motors_measured"],
                ["DREGON-frames", "free-flight_speech-low_room1", "motors_measured"],
                ["DREGON-frames", "free-flight_whitenoise-low_room1", "motors_measured"],
                ["michaels-frames", "FLY124", "rps"],
            ],
        },
    },
}


#: Historical datasets kept as plain dload pins (no derivation spec): the
#: superseded DREGON-LM V1/V2/V3/test recipes and the rps_* probe sets. Their
#: recipes live in git history (the deleted creation CLIs); they are consumed
#: as pinned uploads and are NOT re-derived. Tests assert every dload.lock
#: dataset is either a SPECS entry, a sources.REGISTRY entry, or listed here.
HISTORICAL_PINS = {
    "DREGON-LM-train": "mono V1 recipe (Paper 2 baseline), superseded by V4",
    "DREGON-LM-valid": "mono V1 recipe, superseded by V4",
    "DREGON-LM-V2-train": "V2 recipe (motor combos), superseded by V4",
    "DREGON-LM-V2-valid": "V2 recipe, superseded by V4",
    "DREGON-LM-V3-train": "V3 recipe (per-channel mono), superseded by V4",
    "DREGON-LM-V3-valid": "V3 recipe, superseded by V4",
    "DREGON-LM-test-train": "smoke-test recipe, superseded",
    "DREGON-LM-test-valid": "smoke-test recipe, superseded",
    "DREGON-LM-rps_eval_long_samples": "one-off RPS eval probe set",
    "DREGON-LM-rps_eval_specific_samples": "one-off RPS eval probe set",
    "DREGON-LM-rps_train_specific_samples": "one-off RPS eval probe set",
}


_GENERATORS = {
    "decomp_frames": generate_decomp_frames,
    "dregon_lm": generate_dregon_lm_split,
    "dregon_lm_subset": generate_dregon_lm_subset,
    "dn_lm": generate_dn_lm_split,
    "source_frames": generate_source_frames,
    "frame_subset": generate_frame_subset,
    "raw_subset": generate_raw_subset,
    "pcm16_mono": generate_pcm16_mono,
    "beatvk_valid": generate_beatvk_valid,
    "avq_vkrps": generate_avq_vkrps,
    "se_valid": generate_se_valid,
}

#: Manifest layout per generator family (adopt sanity check + derive meta).
_LAYOUTS = {
    "decomp_frames": "tdframe-v1",
    "dregon_lm": "sample-dir-v1",
    "dregon_lm_subset": "sample-dir-v1",
    "dn_lm": "sample-dir-v1",
    "source_frames": "tdframe-v1",
    "frame_subset": "tdframe-v1",
    "raw_subset": "raw-files",
    "pcm16_mono": PCM16_LAYOUT,
    "beatvk_valid": "tdframe-v1",
    "avq_vkrps": "tdframe-v1",
    "se_valid": "tdframe-v1",
}


# ─── Public helpers (used by scripts/derive.py + tests) ───────────────────────


def build_pipeline(name: str) -> dload.Pipeline:
    """The fingerprintable pipeline for derived dataset ``name``."""
    entry = SPECS[name]
    fn = _GENERATORS[entry["generator"]]
    return dload.from_iterable(partial(fn, entry["gen"]))


def dataset_meta(name: str) -> dict[str, Any]:
    """The manifest meta to pass to ``derive(..., meta=...)``."""
    entry = SPECS[name]
    kind = entry["generator"]
    layout = _LAYOUTS[kind]
    if layout == "sample-dir-v1":
        return _sample_dir_meta(entry["fields"])
    if kind == "source_frames":
        from data_processing import sources

        return sources.dataset_meta(entry["gen"]["source"])
    meta: dict[str, Any] = {"layout": layout}
    if kind == "frame_subset":
        meta["description"] = entry["gen"].get("meta_updates", {}).get("derived_note", "")
    if kind == "beatvk_valid":
        meta["protocol"] = {
            "window_s": _BEATVK_WINDOW_S,
            "frame_s": _BEATVK_FRAME_S,
            "regime_thresholds": {
                "ground_max": _BEATVK_GROUND_MAX,
                "warmup_max": _BEATVK_WARMUP_MAX,
            },
        }
    return meta


def spec_layout(name: str) -> str:
    """The manifest layout a derived/adopted version of ``name`` must carry."""
    return _LAYOUTS[SPECS[name]["generator"]]


def fingerprint(name: str) -> str:
    """The derivation fingerprint (offline; used by adopt-in-place)."""
    return build_pipeline(name).fingerprint()
