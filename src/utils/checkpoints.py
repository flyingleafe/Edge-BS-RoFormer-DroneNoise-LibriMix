"""Checkpoint URI resolution + R2 credential loading (bottom-layer helpers).

Relocated from ``training.artifacts`` so that model code (e.g.
``models.htdemucs_ft``) can resolve ``r2://`` checkpoints without importing
the training package (import-linter contract "models must not import
training"). ``training.artifacts`` re-exports both names, so existing
``from training.artifacts import resolve_checkpoint_uri`` callers are
unaffected.
"""

from __future__ import annotations

from pathlib import Path

R2_ENV_VARS: tuple[str, str, str] = ("R2_ACCOUNT_ID", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY")


def load_r2_env() -> dict[str, str] | None:
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


def resolve_checkpoint_uri(uri: str | Path, cache_dir: str | Path | None = None) -> str:
    """Resolve a checkpoint reference to a local file path.

    A plain path is returned unchanged. An ``r2://<bucket>/<key>`` URI is
    downloaded (once, cached) via a boto3 client built from ``.env`` R2 creds
    and the local cache path is returned. Used for warm-start
    (``cfg.checkpoint``) and for the generated-noise producer's checkpoint,
    so both work identically on a laptop or a fresh cloud GPU box.
    """
    uri = str(uri)
    if not uri.startswith("r2://"):
        return uri
    bucket, _, key = uri[len("r2://") :].partition("/")
    if not bucket or not key:
        raise ValueError(f"malformed r2:// checkpoint uri: {uri!r}")
    cache = Path(cache_dir) if cache_dir else Path(".cache") / "r2_checkpoints"
    cache.mkdir(parents=True, exist_ok=True)
    dst = cache / f"{bucket}__{key.replace('/', '__')}"
    if dst.exists():
        return str(dst)
    env = load_r2_env()
    if env is None:
        raise RuntimeError(f"r2:// checkpoint {uri!r} requested but R2 creds missing in .env")
    import boto3

    client = boto3.client(
        "s3",
        endpoint_url=f"https://{env['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com",
        aws_access_key_id=env["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=env["AWS_SECRET_ACCESS_KEY"],
        region_name="auto",
    )
    client.download_file(bucket, key, str(dst))
    return str(dst)
