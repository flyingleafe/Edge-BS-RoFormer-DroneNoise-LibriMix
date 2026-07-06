"""One opt-in integration test hitting real Cloudflare R2.

Skipped unless ``R2_ACCOUNT_ID`` is present in the environment (loaded from
``.env`` here, same as the real ``ArtifactStore`` does internally). Uploads
under the ``_connectivity_check/test-artifacts/`` prefix in the ``ml-data``
bucket and deletes everything it wrote before returning, pass or fail.
"""

from __future__ import annotations

import os
import uuid

import numpy as np
import pytest
from dotenv import load_dotenv

load_dotenv()

pytestmark = pytest.mark.skipif(
    not os.environ.get("R2_ACCOUNT_ID"),
    reason="R2_ACCOUNT_ID not set; skipping live Cloudflare R2 integration test",
)


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
    fs = store._get_fs()
    assert fs is not None
    root = store._root()

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

        assert ckpt_uri == f"r2://{root}/checkpoints/{os.path.basename(ckpt_path)}"
        assert ckpt_uri is not None
        ckpt_key = ckpt_uri.removeprefix("r2://")
        assert fs.exists(ckpt_key)
        with fs.open(ckpt_key, "rb") as remote:
            assert remote.read() == b"integration-test-checkpoint-bytes"

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
        assert manifest_uri == f"r2://{root}/val_samples/epoch_0/manifest.json"
        assert manifest_uri is not None
        manifest_key = manifest_uri.removeprefix("r2://")
        assert fs.exists(manifest_key)
        wav_key = f"{root}/val_samples/epoch_0/sample_000__mixture.wav"
        assert fs.exists(wav_key)
    finally:
        # Always clean up, even on assertion failure.
        if fs.exists(root):
            fs.rm(root, recursive=True)
        assert not fs.exists(root)
