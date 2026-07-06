"""Cloudflare R2 artifact store: checkpoints + validation samples.

See docs/refactor-unified-framework.md § "Future expansions (design
headroom)": "checkpoints AND selected validation samples are uploaded as
artifacts to Cloudflare R2 (bucket ``ml-data``, creds via ``.env``:
``R2_ACCOUNT_ID`` + AWS keys, s3fs client)".

Client library is ``s3fs`` (a direct project dependency; see
``pyproject.toml``), pointed at the R2 S3-compatible endpoint
``https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com``.

Bucket layout (``bucket`` defaults to ``"ml-data"``, ``prefix`` to
``"artifacts"``)::

    <bucket>/<prefix>/<experiment_name>/checkpoints/<filename>.ckpt
    <bucket>/<prefix>/<experiment_name>/val_samples/epoch_<N>/<sample_id>__<role>.{wav,png}
    <bucket>/<prefix>/<experiment_name>/val_samples/epoch_<N>/manifest.json

:class:`ArtifactStore` is deliberately defensive: every public method is a
no-op (one log line, no exception) when ``enabled=False`` or the R2
credentials (``R2_ACCOUNT_ID``, ``AWS_ACCESS_KEY_ID``,
``AWS_SECRET_ACCESS_KEY`` — loaded from ``.env`` via ``python-dotenv``) are
missing from the environment — this is what keeps headless/CI training runs
artifact-free but crash-free. Any exception raised *during* an upload
(network error, bad credentials, etc.) is caught, logged, and swallowed the
same way: a broken artifact store must never take training down.

The underlying filesystem object is dependency-injected via the ``fs``
constructor argument (anything shaped like ``s3fs.S3FileSystem``: ``.open(path,
mode)`` context manager + ``.makedirs(path, exist_ok=True)``) so tests can pass
an in-memory fake instead of monkeypatching ``s3fs`` internals.
"""

from __future__ import annotations

import io
import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

__all__ = ["ArtifactStore", "ValSample", "R2_ENV_VARS"]

logger = logging.getLogger(__name__)

R2_ENV_VARS: tuple[str, str, str] = ("R2_ACCOUNT_ID", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY")


@dataclass
class ValSample:
    """One validation sample's audio/figure payload, ready for R2 upload.

    ``audio`` maps a role name (``"mixture"``/``"target"``/``"output"`` for
    speech enhancement, ``"mixture"`` for RPS prediction) to ``(waveform,
    sample_rate)``. ``figures`` maps a role name (e.g. ``"rps_overlay"``) to
    already-encoded PNG bytes. ``metrics``/``input_snr`` are folded into the
    per-epoch manifest alongside the sample's R2 keys.
    """

    sample_id: str
    input_snr: float | None = None
    audio: dict[str, tuple[np.ndarray, int]] = field(default_factory=dict)
    figures: dict[str, bytes] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)


def _load_r2_env() -> dict[str, str] | None:
    """Load R2 credentials from the environment (``.env`` via python-dotenv).

    Returns ``None`` if any of :data:`R2_ENV_VARS` is missing.
    """
    import os

    from dotenv import load_dotenv

    load_dotenv()
    account_id = os.environ.get("R2_ACCOUNT_ID")
    access_key = os.environ.get("AWS_ACCESS_KEY_ID")
    secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
    if not account_id or not access_key or not secret_key:
        return None
    return {
        "R2_ACCOUNT_ID": account_id,
        "AWS_ACCESS_KEY_ID": access_key,
        "AWS_SECRET_ACCESS_KEY": secret_key,
    }


class ArtifactStore:
    """Upload checkpoints and validation samples to Cloudflare R2.

    Parameters
    ----------
    experiment_name:
        Names the ``<bucket>/<prefix>/<experiment_name>/...`` root.
    bucket, prefix:
        Bucket + key prefix (``conf/artifacts/*.yaml`` :class:`~training.config.ArtifactsConfig`).
    enabled:
        When ``False``, every method is a no-op (one log line).
    fs:
        Pre-built filesystem object (``s3fs.S3FileSystem``-shaped, or a test
        fake). When omitted, a real ``s3fs.S3FileSystem`` is lazily built
        from ``.env`` credentials on first use; if those are missing, the
        store silently degrades to no-op mode (headless CI safety).
    """

    def __init__(
        self,
        *,
        experiment_name: str,
        bucket: str = "ml-data",
        prefix: str = "artifacts",
        enabled: bool = True,
        fs: Any | None = None,
    ) -> None:
        self.experiment_name = experiment_name
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self.enabled = enabled
        self._fs: Any | None = fs
        self._env_checked = fs is not None

    @property
    def active(self) -> bool:
        """Whether uploads will actually reach R2 (enabled + fs resolvable)."""
        return self.enabled and self._get_fs() is not None

    def _get_fs(self) -> Any | None:
        if self._fs is not None:
            return self._fs
        if not self.enabled or self._env_checked:
            return None
        self._env_checked = True
        env = _load_r2_env()
        if env is None:
            logger.warning(
                "ArtifactStore: R2 credentials missing (%s) — artifact uploads disabled.",
                ", ".join(R2_ENV_VARS),
            )
            return None
        import s3fs

        self._fs = s3fs.S3FileSystem(
            endpoint_url=f"https://{env['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com",
            key=env["AWS_ACCESS_KEY_ID"],
            secret=env["AWS_SECRET_ACCESS_KEY"],
        )
        return self._fs

    def _root(self) -> str:
        return f"{self.bucket}/{self.prefix}/{self.experiment_name}"

    def upload_checkpoint(self, path: str | Path) -> str | None:
        """Upload a checkpoint file; returns its ``r2://...`` URI, or ``None``
        (disabled / missing creds / upload failed)."""
        if not self.enabled:
            logger.info("ArtifactStore: disabled; skipping checkpoint upload for %s", path)
            return None
        fs = self._get_fs()
        if fs is None:
            return None
        ckpt_path = Path(path)
        checkpoints_dir = f"{self._root()}/checkpoints"
        key = f"{checkpoints_dir}/{ckpt_path.name}"
        try:
            fs.makedirs(checkpoints_dir, exist_ok=True)
            with open(ckpt_path, "rb") as src, fs.open(key, "wb") as dst:
                dst.write(src.read())
        except Exception:
            logger.warning(
                "ArtifactStore: failed to upload checkpoint %s", ckpt_path, exc_info=True
            )
            return None
        return f"r2://{key}"

    def upload_val_samples(self, epoch: int, samples: Sequence[ValSample]) -> str | None:
        """Upload one epoch's validation-sample audio/figures + a manifest.

        Returns the manifest's ``r2://...`` URI, or ``None`` (disabled /
        missing creds / no samples / upload failed).
        """
        if not self.enabled:
            logger.info("ArtifactStore: disabled; skipping val-sample upload for epoch %d", epoch)
            return None
        if not samples:
            return None
        fs = self._get_fs()
        if fs is None:
            return None

        epoch_root = f"{self._root()}/val_samples/epoch_{epoch}"
        manifest: dict[str, Any] = {"epoch": epoch, "samples": []}
        try:
            fs.makedirs(epoch_root, exist_ok=True)
            for sample in samples:
                keys: dict[str, str] = {}
                for role, (wav, sr) in sample.audio.items():
                    key = f"{epoch_root}/{sample.sample_id}__{role}.wav"
                    buf = io.BytesIO()
                    sf.write(buf, np.asarray(wav, dtype=np.float32), sr, format="WAV")
                    with fs.open(key, "wb") as dst:
                        dst.write(buf.getvalue())
                    keys[role] = f"r2://{key}"
                for role, png_bytes in sample.figures.items():
                    key = f"{epoch_root}/{sample.sample_id}__{role}.png"
                    with fs.open(key, "wb") as dst:
                        dst.write(png_bytes)
                    keys[role] = f"r2://{key}"
                manifest["samples"].append(
                    {
                        "sample_id": sample.sample_id,
                        "input_snr": sample.input_snr,
                        "metrics": dict(sample.metrics),
                        "keys": keys,
                    }
                )

            manifest_key = f"{epoch_root}/manifest.json"
            with fs.open(manifest_key, "wb") as dst:
                dst.write(json.dumps(manifest, indent=2).encode("utf-8"))
        except Exception:
            logger.warning(
                "ArtifactStore: failed to upload val samples for epoch %d", epoch, exc_info=True
            )
            return None
        return f"r2://{manifest_key}"
