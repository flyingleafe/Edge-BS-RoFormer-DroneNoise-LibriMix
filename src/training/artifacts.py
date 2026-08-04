"""Cloudflare R2 artifact store: checkpoints + validation samples.

See docs/refactor-unified-framework.md § "Future expansions (design
headroom)": "checkpoints AND selected validation samples are uploaded as
artifacts to Cloudflare R2 (bucket ``ml-data``, creds via ``.env``:
``R2_ACCOUNT_ID`` + AWS keys, boto3 client)".

Client library is ``boto3`` (already a dependency via ``dload-ml``; declared
directly in ``pyproject.toml``), pointed at the R2 S3-compatible endpoint
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

The underlying S3 client is dependency-injected via the ``client``
constructor argument (anything shaped like ``boto3.client("s3")``:
``.put_object(Bucket=..., Key=..., Body=...)`` +
``.upload_file(Filename, Bucket, Key)``) so tests can pass an in-memory fake
instead of monkeypatching ``boto3`` internals.
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

# Credential loading + r2:// checkpoint resolution live in utils.checkpoints
# (bottom layer) so model/data code can use them without importing training;
# re-exported here for backward compatibility.
from utils.checkpoints import R2_ENV_VARS, resolve_checkpoint_uri
from utils.checkpoints import load_r2_env as _load_r2_env

__all__ = ["ArtifactStore", "ValSample", "R2_ENV_VARS", "resolve_checkpoint_uri"]

logger = logging.getLogger(__name__)


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
    client:
        Pre-built S3 client (``boto3.client("s3")``-shaped, or a test fake).
        When omitted, a real boto3 S3 client is lazily built from ``.env``
        credentials on first use; if those are missing, the store silently
        degrades to no-op mode (headless CI safety).
    """

    def __init__(
        self,
        *,
        experiment_name: str,
        bucket: str = "ml-data",
        prefix: str = "artifacts",
        enabled: bool = True,
        client: Any | None = None,
    ) -> None:
        self.experiment_name = experiment_name
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self.enabled = enabled
        self._client: Any | None = client
        self._env_checked = client is not None

    @property
    def active(self) -> bool:
        """Whether uploads will actually reach R2 (enabled + client resolvable)."""
        return self.enabled and self._get_client() is not None

    def _get_client(self) -> Any | None:
        if self._client is not None:
            return self._client
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
        import boto3

        self._client = boto3.client(
            "s3",
            endpoint_url=f"https://{env['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com",
            aws_access_key_id=env["AWS_ACCESS_KEY_ID"],
            aws_secret_access_key=env["AWS_SECRET_ACCESS_KEY"],
            region_name="auto",
        )
        return self._client

    def _key_root(self) -> str:
        """Key root inside the bucket (no bucket component)."""
        return f"{self.prefix}/{self.experiment_name}"

    def _uri(self, key: str) -> str:
        return f"r2://{self.bucket}/{key}"

    def upload_checkpoint(self, path: str | Path) -> str | None:
        """Upload a checkpoint file; returns its ``r2://...`` URI, or ``None``
        (disabled / missing creds / upload failed)."""
        if not self.enabled:
            logger.info("ArtifactStore: disabled; skipping checkpoint upload for %s", path)
            return None
        client = self._get_client()
        if client is None:
            return None
        ckpt_path = Path(path)
        key = f"{self._key_root()}/checkpoints/{ckpt_path.name}"
        try:
            client.upload_file(str(ckpt_path), self.bucket, key)
        except Exception:
            logger.warning(
                "ArtifactStore: failed to upload checkpoint %s", ckpt_path, exc_info=True
            )
            return None
        return self._uri(key)

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
        client = self._get_client()
        if client is None:
            return None

        epoch_root = f"{self._key_root()}/val_samples/epoch_{epoch}"
        manifest: dict[str, Any] = {"epoch": epoch, "samples": []}
        try:
            for sample in samples:
                keys: dict[str, str] = {}
                for role, (wav, sr) in sample.audio.items():
                    key = f"{epoch_root}/{sample.sample_id}__{role}.wav"
                    buf = io.BytesIO()
                    sf.write(buf, np.asarray(wav, dtype=np.float32), sr, format="WAV")
                    client.put_object(Bucket=self.bucket, Key=key, Body=buf.getvalue())
                    keys[role] = self._uri(key)
                for role, png_bytes in sample.figures.items():
                    key = f"{epoch_root}/{sample.sample_id}__{role}.png"
                    client.put_object(Bucket=self.bucket, Key=key, Body=png_bytes)
                    keys[role] = self._uri(key)
                manifest["samples"].append(
                    {
                        "sample_id": sample.sample_id,
                        "input_snr": sample.input_snr,
                        "metrics": dict(sample.metrics),
                        "keys": keys,
                    }
                )

            manifest_key = f"{epoch_root}/manifest.json"
            client.put_object(
                Bucket=self.bucket,
                Key=manifest_key,
                Body=json.dumps(manifest, indent=2).encode("utf-8"),
            )
        except Exception:
            logger.warning(
                "ArtifactStore: failed to upload val samples for epoch %d", epoch, exc_info=True
            )
            return None
        return self._uri(manifest_key)
