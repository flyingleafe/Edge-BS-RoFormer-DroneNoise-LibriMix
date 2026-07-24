#!/usr/bin/env python3
"""Build + publish the fixed speech-enhancement validation sets (F1 baselines).

See ``docs/se-baselines-plan.md`` § "Fixed validation sets". Two ``tdframe-v1``
datasets, each sample a mixture Frame ``{mixture, target, meta}`` consumed by
:class:`data_processing.frame_datasets.SEValidFrameDataset` and scored by
``eval.py`` (per-SNR grouping on ``meta.input_snr``):

- **SE-valid-drone** — held-out drone noise × held-out LibriSpeech speakers over
  the SNR grid. The drone-focused floor + the Pass-B−Pass-A diversity delta.
- **SE-valid-harmonic** — the same protocol per Pass-B category (drone + the
  harmonic categories), for the per-category transfer table.

Leak-free splits:
- **Noise**: every ``audio_pool`` source draws its *valid* side of the per-shard
  index holdout (``holdout.split=valid``); training draws ``split=train``. Real
  DREGON/michaels drone noise is held out by *recording id* (valid uses DREGON
  ``free-flight_nosource_room1`` + michaels ``FLY124``; training excludes room1
  and uses ``FLY125``).
- **Speech**: valid draws only from ``HELDOUT_SPEAKERS`` (LibriSpeech
  train-clean-100 speaker ids reserved here); training excludes the same ids
  (the online-mix policy ``sources.speech[].exclude``).

Deviation from the plan (documented): MIMII is treated as ONE category
(fan/pump/valve/slider combined) rather than four — category-uniform weighting
still prevents its 258 GiB from dominating, and the diversity question is
unaffected. Time-warp augmentation is not applied to the valid (it is fixed).

Run (needs R2 creds in ``.env``; pulls a few shards per source):

    python scripts/build_se_valid.py --dataset both --publish
    dload pin SE-valid-drone && dload pin SE-valid-harmonic
    git add dload.lock && git commit
"""

from __future__ import annotations

import argparse
import hashlib
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import tdseries as td

import data_processing.streams as streams
from data_processing.frames import audio_series
from data_processing.online_mixing import (
    _MAX_DRAW_RETRIES,
    AudioFileSourcePool,
    _extract_audio_array,
    _is_silent,
    _scale_source_to_snr,
    build_noise_pool,
)
from utils.paths import get_data_root

SR = 16000
SNR_GRID = [-30, -25, -20, -15, -10, -5, 0]

# LibriSpeech train-clean-100 speaker ids reserved for validation (every 10th
# id, sorted numerically — 25 of 246). Training excludes these (policy
# `sources.speech[].exclude`), so no speaker is shared train/valid.
HELDOUT_SPEAKERS = [
    "19", "89", "211", "307", "445", "831", "1098", "1502", "1898", "2182",
    "2764", "3168", "3526", "3982", "4267", "4813", "5322", "5703", "6081",
    "6476", "7067", "7312", "7800", "8238", "8630",
]  # fmt: skip

# valid draws the last 2 whole shards of each dataset (whole-recording holdout;
# bounds the valid build to ~2 shards/dataset even on 2003-shard MIMII). Training
# uses the complementary shards (holdout.split=train, same valid_shards).
HOLDOUT_VALID = {"split": "valid", "valid_shards": 2}

# Per-category noise-source specs consumed by `build_noise_pool`. Every
# audio_pool draws its valid holdout side; drone real-frames are held out by id.
CATEGORY_NOISE: dict[str, list[dict]] = {
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
        {"kind": "audio_pool", "dataset": "drone_audio", "holdout": HOLDOUT_VALID},
        {"kind": "audio_pool", "dataset": "DroneAudioSet", "holdout": HOLDOUT_VALID},
    ],
    "mimii": [{"kind": "audio_pool", "dataset": "MIMII", "holdout": HOLDOUT_VALID}],
    "mimii_dg": [{"kind": "audio_pool", "dataset": "MIMII-DG", "holdout": HOLDOUT_VALID}],
    "aircraft": [{"kind": "audio_pool", "dataset": "AeroSonicDB", "holdout": HOLDOUT_VALID}],
    "motors": [
        {"kind": "audio_pool", "dataset": "HUSTmotor", "holdout": HOLDOUT_VALID},
        {"kind": "audio_pool", "dataset": "KAIST-rotating-acoustic", "holdout": HOLDOUT_VALID},
    ],
    "horns": [{"kind": "audio_pool", "dataset": "HornBase", "holdout": HOLDOUT_VALID}],
}

HARMONIC_CATEGORIES = ["drone", "mimii", "mimii_dg", "aircraft", "motors", "horns"]


def _heldout_speech_pool(duration_s: float) -> AudioFileSourcePool:
    root = get_data_root() / "data" / "librispeech" / "LibriSpeech" / "train-clean-100"
    if not root.is_dir():
        raise FileNotFoundError(
            f"LibriSpeech train-clean-100 not found at {root}; `dload pull librispeech` first"
        )
    files: list[Path] = []
    for spk in HELDOUT_SPEAKERS:
        files.extend((root / spk).glob("**/*.flac"))
    if not files:
        raise ValueError("no held-out speaker files found")
    return AudioFileSourcePool(
        sorted(files), sample_rate=SR, duration_s=duration_s, cache_mode="none"
    )


def _mono(audio: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """(C, T) -> a single (T,) channel (random for multichannel noise)."""
    if audio.ndim == 2:
        return audio[int(rng.integers(0, audio.shape[0]))]
    return audio


def _iter_category_samples(
    category: str, *, per_snr: int, duration_s: float, seed: int
) -> Iterator[tuple[str, dict]]:
    target_len = int(round(duration_s * SR))
    noise_pool = build_noise_pool(CATEGORY_NOISE[category], duration_s=duration_s, sample_rate=SR)
    speech_pool = _heldout_speech_pool(duration_s)
    # Deterministic per (category) RNG so the set is fixed/reproducible. Uses a
    # stable digest (Python's hash() is per-process randomized via PYTHONHASHSEED).
    digest = hashlib.blake2b(f"{seed}:{category}".encode(), digest_size=8).digest()
    rng = np.random.default_rng(int.from_bytes(digest, "little"))
    idx = 0
    for snr in SNR_GRID:
        for _ in range(per_snr):
            # Reject digitally-silent draws: a silent NOISE draw makes
            # _scale_source_to_snr return scale == 0, zeroing BOTH the target and
            # the mixture (the v1 sets shipped 5 such empty clips, which alone
            # shifted the 0 dB noisy anchor by 4.8 dB). See _is_silent.
            for _attempt in range(_MAX_DRAW_RETRIES):
                noise_tf = noise_pool.sample_timeframe(rng, duration_s)
                noise = _mono(_extract_audio_array(noise_tf, target_len=target_len), rng)
                speech = speech_pool.sample_mono(rng)
                if not _is_silent(noise) and not _is_silent(speech):
                    break
            scaled = _scale_source_to_snr(speech[None, :], noise[None, :], float(snr))[0]
            mixture = (noise + scaled).astype(np.float32)
            sample_id = f"{category}_snr{snr:+03d}_{idx:04d}"
            frame = td.Frame(
                {
                    "mixture": audio_series(mixture[None, :], SR),
                    "target": audio_series(scaled[None, :].astype(np.float32), SR),
                    "meta": td.Frame(
                        {
                            "recording_id": sample_id,
                            "id": sample_id,
                            "input_snr": float(snr),
                            "category": category,
                        }
                    ),
                }
            )
            yield sample_id, streams.frame_to_sample(frame)
            idx += 1


def _publish(name: str, categories: list[str], *, per_snr, duration_s, seed, publish: bool):
    def gen() -> Iterator[tuple[str, dict]]:
        for cat in categories:
            print(f"  [{name}] category={cat} ...")
            yield from _iter_category_samples(
                cat, per_snr=per_snr, duration_s=duration_s, seed=seed
            )

    if not publish:
        # Dry run: exercise a few samples per category without committing.
        n = 0
        for cat in categories:
            it = _iter_category_samples(cat, per_snr=2, duration_s=duration_s, seed=seed)
            for _ in range(2):
                next(it)
                n += 1
        print(f"[dry-run] {name}: built {n} sample(s) across {len(categories)} categories OK")
        return

    repo = streams.open_repository()
    recipe = Path(__file__).read_text(encoding="utf-8")
    manifest = repo.commit(
        name,
        gen(),
        meta={
            streams.LAYOUT_META_KEY: streams.TDFRAME_LAYOUT,
            "description": (
                f"SE blind-baseline validation set ({name}): held-out noise x held-out "
                f"LibriSpeech speakers, SNR grid {SNR_GRID} dB, {per_snr}/point/category, "
                f"{duration_s}s @ {SR} Hz mono. Frame {{mixture,target,meta}}."
            ),
            "source": "scripts/build_se_valid.py",
            "snr_grid": SNR_GRID,
            "categories": categories,
            "per_snr": per_snr,
            "duration_s": duration_s,
        },
        recipe=recipe,
        progress=print,
    )
    print(f"{name}@{manifest.version[:12]}: {manifest.num_samples} samples")


def main() -> None:
    p = argparse.ArgumentParser(description="Build/publish the SE validation sets.")
    p.add_argument("--dataset", choices=["drone", "harmonic", "both"], default="both")
    p.add_argument("--per-snr", type=int, default=50)
    p.add_argument("--duration", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=20260720)
    p.add_argument("--publish", action="store_true", help="commit to dload (else dry-run)")
    args = p.parse_args()

    if args.dataset in ("drone", "both"):
        _publish(
            "SE-valid-drone",
            ["drone"],
            per_snr=args.per_snr,
            duration_s=args.duration,
            seed=args.seed,
            publish=args.publish,
        )
    if args.dataset in ("harmonic", "both"):
        _publish(
            "SE-valid-harmonic",
            HARMONIC_CATEGORIES,
            per_snr=args.per_snr,
            duration_s=args.duration,
            seed=args.seed,
            publish=args.publish,
        )


if __name__ == "__main__":
    main()
