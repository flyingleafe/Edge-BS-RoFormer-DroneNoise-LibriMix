"""dload *derived-dataset* declarations for DREGON-LM and DN-LM.

A derived dataset (dload 0.3.0 ``Repository.derive``) is a finite, deterministic
pipeline that dload runs *once*, commits as an ordinary content-addressed
version, and memoizes: every later caller of the identical pipeline hits the
same snapshot by *fingerprint* instead of recomputing. This module holds the
two pieces that make our mixed datasets expressible that way:

1. **Module-level generator functions** (``generate_dregon_lm_split``,
   ``generate_dn_lm_split``) that yield storable ``(key, {field: bytes})``
   samples in the ``sample-dir-v1`` convention (per-sample ``mixture``/
   ``vocals``/``noise``/``rps`` + a dataset-level ``_meta`` sample), reading
   their parent datasets through dload URIs — never ``data/`` paths.
2. A **registry of frozen JSON specs** (``SPECS``) — one per dataset — carrying
   every generation parameter, the seed, a ``recipe_version``, and the
   *resolved parent pins*. The spec is the durable, reviewed declaration; a
   dataset is materialized (or adopted) from it by ``scripts/derive.py``.

Fingerprint mechanics (why the shapes below matter):
    build_pipeline(name) == dload.from_iterable(partial(<gen_fn>, gen_spec))
dload fingerprints this by the generator's *module + qualname* plus the
``gen_spec`` dict (recursively, keys sorted). So:

- The generator must stay a top-level function (dload rejects lambdas/locals).
- Editing a generator's *behavior* without renaming it would silently serve the
  stale snapshot — **bump the spec's ``recipe_version`` on any behavioral
  change** (review-enforced convention; ``recipe_version`` is inside the
  fingerprint, so a bump forces a fresh identity).
- ``from_iterable`` pipelines have no ``SourceNode``s, so dload's own
  ``derived_from`` is empty; the *parent pins live inside ``gen_spec``* and thus
  inside the fingerprint. That is what ties a derived version to exact parent
  versions.
- ``meta``/``fields``/``layout`` are **registry metadata, not generation
  inputs** — they are kept out of ``gen_spec`` (mirroring dload, whose
  fingerprint deliberately excludes ``meta``). They are forwarded to
  ``derive(..., meta=...)`` so the derived manifest carries the same
  ``sample-dir-v1`` decode/reconstruct keys a hand-committed version would.

Determinism caveat: cross-machine *byte* determinism of these generators is not
guaranteed (numpy RNG draw order, librosa/soundfile versions). That is benign
for memoization — the derivation ref settles once and every later reader
resolves that snapshot — but it means (a) materialize from one designated box,
(b) all glob listings are sorted here, and (c) the historical ``DREGON-LM-V4-*``
pins are **adopted in place** (their ref is pointed at the already-uploaded
version) rather than re-derived, which would upload a near-duplicate copy. See
``docs/derived-datasets-plan.md``.

Layering note: the generation cores live in ``scripts/create_dregon_librimix.py``
and ``scripts/create_dataset.py`` (which also drive the disk-writing CLIs).
``scripts/`` is not an importable package, and those modules pull torch
transitively, so the generator bodies below import them **lazily** via a small
``sys.path`` shim — keeping *this* module importable without torch, so
fingerprinting (``scripts/derive.py adopt``) runs on any box. A future cleanup
could hoist the pure cores into this package and drop the shim.
"""

from __future__ import annotations

import io
import sys
from collections.abc import Iterator
from functools import partial
from pathlib import Path
from typing import Any

import dload
import numpy as np
import soundfile as sf

Sample = tuple[str, dict[str, bytes]]

#: Repo root, for the deferred ``scripts/`` import shim (see module docstring).
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS_DIR = _REPO_ROOT / "scripts"

#: Resolved parent dataset pins (from ``dload.lock``). Frozen into the specs so
#: they participate in the derivation fingerprint. Update deliberately (a change
#: mints a new derived identity).
PARENTS = {
    "DREGON": "dload:DREGON@db39bcf762d0b2beb3433fc2760da6a55e078f8134f8bb074ee8bf985a5ffc03",
    "librispeech": "dload:librispeech@fda9aad8dfb545c82752f33e8ff563feaee82f5f9ba7b50efafff4cd8ff73ae5",
    "drone_audio": "dload:drone_audio@b6c77a68c55dedec11750a3784c10833e7db981fab6ef00380300a9e4d382b95",
}


# ─── Sample encoding ──────────────────────────────────────────────────────────


def wav_bytes(arr: np.ndarray, sample_rate: int) -> bytes:
    """Encode ``(T,)`` or ``(T, C)`` float audio to WAV bytes, byte-for-byte as
    the disk-writing CLIs' ``sf.write(path, arr, sr)`` would (same default
    subtype inferred from the ``WAV`` format)."""
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


# ─── Generators ───────────────────────────────────────────────────────────────


def generate_dregon_lm_split(gen: dict[str, Any]) -> Iterator[Sample]:
    """Yield the samples of one multichannel DREGON-LM split.

    ``gen`` is the frozen generation sub-spec (see ``SPECS``): ``seed``,
    ``num_samples``, ``split``, ``parents`` (dload URIs), ``noise_sources``
    (``None`` → DREGON split defaults, else a ``load_noise_sources`` spec
    string), and ``params`` (mixing knobs). Reads LibriSpeech + DREGON via
    dload, then reuses the exact per-sample mixing core the CLI uses.
    """
    import random

    if str(_SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(_SCRIPTS_DIR))
    # Deferred (pulls torch transitively via data_processing.*): keeps this
    # module importable for fingerprinting without torch.
    import create_dregon_librimix as clm  # type: ignore[import-not-found]

    from data_processing.streams import resolve_source

    params = gen["params"]
    split = gen["split"]
    sample_rate = int(params["sample_rate"])
    sample_duration = float(params["sample_duration"])
    target_length = int(sample_duration * sample_rate)

    speech_root = resolve_source(gen["parents"]["librispeech"])
    speech_subpath = params.get("speech_subpath")
    speech_dir = Path(speech_root) / speech_subpath if speech_subpath else Path(speech_root)
    speech_files = sorted(str(p) for ext in ("*.wav", "*.flac") for p in speech_dir.rglob(ext))
    if not speech_files:
        raise ValueError(f"No speech files under {speech_dir}")

    dregon_dir = Path(resolve_source(gen["parents"]["DREGON"]))
    noise_sources = gen.get("noise_sources")
    if noise_sources:
        raise NotImplementedError(
            "generate_dregon_lm_split does not yet source composed noise pools "
            f"(noise_sources={noise_sources!r}) via dload — this spec is "
            "adopt-only; see scripts/derive.py adopt."
        )
    if split == "train":
        noise_records = clm.load_dregon_multichannel_records(
            dregon_dir, splits=clm.TRAIN_NOISE_SPLITS
        )
    else:
        noise_records = clm.load_dregon_multichannel_records(
            dregon_dir, recording_ids=clm.VALID_NOISE_RECORDING_IDS
        )
    if not noise_records:
        raise ValueError("No valid multichannel noise records found")

    random.seed(gen["seed"])
    np.random.seed(gen["seed"])
    snr_range = (float(params["snr_range"][0]), float(params["snr_range"][1]))

    metadata_list = []
    for idx in range(int(gen["num_samples"])):
        sample_id = f"sample_{idx:05d}"
        arrays, sample_meta = clm.render_multichannel_sample(
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
            min_motor_rps=float(params["min_motor_rps"]),
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


def generate_dn_lm_split(gen: dict[str, Any]) -> Iterator[Sample]:
    """Yield the samples of one DN-LM (DroneNoise-LibriMix, mono) split.

    Reads LibriSpeech + a local drone-noise dataset via dload, then reuses the
    CLI's ``mix_dn_lm`` core (length/normalize/inverse-distance/SNR/anti-clip).
    No ``rps`` field — DN-LM is a plain speech-enhancement mixture; consume it
    via ``ensure_local`` + the folder loader, not the RPS streaming decoder.
    """
    import random

    if str(_SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(_SCRIPTS_DIR))
    import create_dataset as cds  # type: ignore[import-not-found]

    from data_processing.streams import resolve_source

    params = gen["params"]
    split = gen["split"]
    sample_rate = int(params["sample_rate"])
    target_length = int(float(params["sample_duration"]) * sample_rate)

    speech_root = resolve_source(gen["parents"]["librispeech"])
    speech_subpath = params.get("speech_subpath")
    speech_dir = Path(speech_root) / speech_subpath if speech_subpath else Path(speech_root)
    speech_files = sorted(str(p) for ext in ("*.wav", "*.flac") for p in speech_dir.rglob(ext))
    if not speech_files:
        raise ValueError(f"No speech files under {speech_dir}")

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
        speech = cds.load_audio(random.choice(speech_files), target_sr=sample_rate)
        noise_path = random.choice(noise_files)
        noise = cds.load_audio(noise_path, target_sr=sample_rate)
        arrays, actual_snr, speech_distance = cds.mix_dn_lm(
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


_GENERATORS = {
    "dregon_lm": generate_dregon_lm_split,
    "dn_lm": generate_dn_lm_split,
}


# ─── Spec registry ────────────────────────────────────────────────────────────
#
# One entry per derived dataset. ``gen`` is the fingerprinted sub-spec (feeds
# ``partial(generator, gen)``); everything else (``generator``, ``fields``,
# ``adopt_only``, ``note``) is registry metadata. ``adopt_only`` datasets are
# published by pointing their derivation ref at an already-uploaded historical
# version (``scripts/derive.py adopt``) rather than re-materialized — either
# because re-running would upload a near-duplicate (V4-train) or because the
# generator cannot yet reproduce them (real-valid / michaels-sourced splits).

_DREGON_LM_FIELDS = {
    "mixture": "mixture.wav",
    "vocals": "vocals.wav",
    "noise": "noise.wav",
    "rps": "rps.npy",
    "meta": "meta.json",
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

SPECS: dict[str, dict[str, Any]] = {
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
            "parents": {"DREGON": PARENTS["DREGON"], "librispeech": PARENTS["librispeech"]},
            "params": _V4_PARAMS,
        },
    },
    "DREGON-LM-V4-valid": {
        "generator": "dregon_lm",
        "fields": _DREGON_LM_FIELDS,
        "adopt_only": True,
        "note": "Adopt-in-place. The V4 valid split is the raw real-valid "
        "recording clips, which the synthesized generator does not reproduce.",
        "gen": {
            "recipe_version": 1,
            "seed": 43,
            "num_samples": 600,
            "split": "valid",
            "noise_sources": None,
            "parents": {"DREGON": PARENTS["DREGON"], "librispeech": PARENTS["librispeech"]},
            "params": _V4_PARAMS,
        },
    },
    "DREGON-LM-V4-michaels-train": {
        "generator": "dregon_lm",
        "fields": _DREGON_LM_FIELDS,
        "adopt_only": True,
        "note": "Adopt-in-place. Composed noise pool "
        "(dregon in_flight_noise + michaels FLY125) is not yet sourced via "
        "dload; generator raises if run.",
        "gen": {
            "recipe_version": 1,
            "seed": 42,
            "num_samples": 6000,
            "split": "train",
            "noise_sources": "dregon-split:in_flight_noise,michaels:FLY125",
            "parents": {"DREGON": PARENTS["DREGON"], "librispeech": PARENTS["librispeech"]},
            "params": _V4_PARAMS,
        },
    },
    "DREGON-LM-V4-michaels-valid": {
        "generator": "dregon_lm",
        "fields": _DREGON_LM_FIELDS,
        "adopt_only": True,
        "note": "Adopt-in-place. Real-valid michaels FLY124 clips; not "
        "reproduced by the synthesized generator.",
        "gen": {
            "recipe_version": 1,
            "seed": 43,
            "num_samples": 600,
            "split": "valid",
            "noise_sources": "michaels:FLY124",
            "parents": {"DREGON": PARENTS["DREGON"], "librispeech": PARENTS["librispeech"]},
            "params": _V4_PARAMS,
        },
    },
    # DN-LM: absent from the bucket → fresh materialization (no adoption).
    # Noise source = drone-only. The raw `drone_audio` dataset (Sara Al-Emadi
    # DroneAudioDataset) mixes real drone recordings with ESC-50 / white-noise /
    # silence negatives under `*/unknown/`; the paper's DN-LM used only the
    # drone class (the CLI's HF path filters label==1). So we point noise_subpath
    # at `Binary_Drone_Audio/yes_drone` (the 1332 label-1 recordings; equals the
    # bebop_1+membo_1 union) rather than globbing the whole tree — which would
    # contaminate the "noise" with non-drone audio. Sizes 6480/720 follow the
    # paper's "2 hours" (README + AGENTS); a 10× scale-up (64800) existed in the
    # deleted replicate_paper.py but is not the canonical figure.
    "DN-LM-train": {
        "generator": "dn_lm",
        "fields": _DN_LM_FIELDS,
        "adopt_only": False,
        "note": "Drone-only noise from drone_audio/Binary_Drone_Audio/yes_drone "
        "(label-1 recordings; excludes ESC-50/WN 'unknown'). Fresh "
        "materialization; bump recipe_version on any recipe change.",
        "gen": {
            "recipe_version": 1,
            "seed": 42,
            "num_samples": 6480,
            "split": "train",
            "parents": {
                "librispeech": PARENTS["librispeech"],
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
        "(label-1 recordings; excludes ESC-50/WN 'unknown'). Fresh "
        "materialization; bump recipe_version on any recipe change.",
        "gen": {
            "recipe_version": 1,
            "seed": 43,
            "num_samples": 720,
            "split": "valid",
            "parents": {
                "librispeech": PARENTS["librispeech"],
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
}


# ─── Public helpers (used by scripts/derive.py + tests) ───────────────────────


def build_pipeline(name: str) -> dload.Pipeline:
    """The fingerprintable pipeline for derived dataset ``name``."""
    entry = SPECS[name]
    fn = _GENERATORS[entry["generator"]]
    return dload.from_iterable(partial(fn, entry["gen"]))


def dataset_meta(name: str) -> dict[str, Any]:
    """The ``sample-dir-v1`` manifest meta to pass to ``derive(..., meta=...)``."""
    return _sample_dir_meta(SPECS[name]["fields"])


def fingerprint(name: str) -> str:
    """The derivation fingerprint (offline; used by adopt-in-place)."""
    return build_pipeline(name).fingerprint()
