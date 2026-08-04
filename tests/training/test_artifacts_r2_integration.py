"""One opt-in integration test hitting real Cloudflare R2.

Marked ``network`` (deselected by default, see ``addopts`` in
``pyproject.toml``) — run with ``pytest -m network`` to opt in. When opted
in, it is still skipped unless ``R2_ACCOUNT_ID`` is present in the
environment (loaded from ``.env`` here, same as the real ``ArtifactStore``
does internally). Uploads under the ``_connectivity_check/test-artifacts/``
prefix in the ``ml-data`` bucket and deletes everything it wrote before
returning, pass or fail.
"""

from __future__ import annotations

import os
import uuid

import numpy as np
import pytest
from dotenv import load_dotenv

load_dotenv()

pytestmark = [
    pytest.mark.network,
    pytest.mark.skipif(
        not os.environ.get("R2_ACCOUNT_ID"),
        reason="R2_ACCOUNT_ID not set; skipping live Cloudflare R2 integration test",
    ),
]


def test_artifact_store_roundtrip_against_real_r2():
    from training.artifacts import ArtifactStore, ValSample

    run_id = uuid.uuid4().hex[:8]
    store = ArtifactStore(
        experiment_name=f"run-{run_id}",
        bucket="ml-data",
        prefix="_connectivity_check/test-artifacts",
        enabled=True,
    )
    assert store.active, "ArtifactStore did not activate against real R2 — check .env creds"
    client = store._get_client()
    assert client is not None
    bucket = store.bucket
    key_root = f"{store.prefix}/{store.experiment_name}"

    def _exists(key: str) -> bool:
        try:
            client.head_object(Bucket=bucket, Key=key)
            return True
        except client.exceptions.ClientError:
            return False

    def _list_keys() -> list[str]:
        resp = client.list_objects_v2(Bucket=bucket, Prefix=f"{key_root}/")
        return [obj["Key"] for obj in resp.get("Contents", [])]

    try:
        # ── Checkpoint upload ──
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".ckpt", delete=False) as f:
            f.write(b"integration-test-checkpoint-bytes")
            ckpt_path = f.name
        try:
            ckpt_uri = store.upload_checkpoint(ckpt_path)
        finally:
            os.unlink(ckpt_path)

        assert ckpt_uri == f"r2://{bucket}/{key_root}/checkpoints/{os.path.basename(ckpt_path)}"
        assert ckpt_uri is not None
        ckpt_key = ckpt_uri.removeprefix(f"r2://{bucket}/")
        assert _exists(ckpt_key)
        body = client.get_object(Bucket=bucket, Key=ckpt_key)["Body"].read()
        assert body == b"integration-test-checkpoint-bytes"

        # ── Validation-sample upload (audio + manifest) ──
        wav = (np.linspace(-0.5, 0.5, 1600, dtype=np.float32), 16000)
        samples = [
            ValSample(
                sample_id="sample_000",
                input_snr=-9.0,
                audio={"mixture": wav},
                metrics={"mse": 0.42},
            )
        ]
        manifest_uri = store.upload_val_samples(0, samples)
        assert manifest_uri == f"r2://{bucket}/{key_root}/val_samples/epoch_0/manifest.json"
        assert manifest_uri is not None
        manifest_key = manifest_uri.removeprefix(f"r2://{bucket}/")
        assert _exists(manifest_key)
        wav_key = f"{key_root}/val_samples/epoch_0/sample_000__mixture.wav"
        assert _exists(wav_key)
    finally:
        # Always clean up, even on assertion failure.
        leftover = _list_keys()
        if leftover:
            client.delete_objects(Bucket=bucket, Delete={"Objects": [{"Key": k} for k in leftover]})
        assert _list_keys() == []
