from __future__ import annotations

import json
import shutil
from pathlib import Path, PurePosixPath
from typing import BinaryIO

from postdoc.interfaces.storage import StorageBackend


class LocalStorage(StorageBackend):

    def __init__(self, results_dir: Path | str):
        self._root = Path(results_dir)

    def _resolve(self, job_id: str, rel_path: PurePosixPath) -> Path:
        return self._root / job_id / str(rel_path)

    def put(self, job_id: str, rel_path: PurePosixPath, data: bytes | BinaryIO) -> None:
        path = self._resolve(job_id, rel_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(data, bytes):
            path.write_bytes(data)
        else:
            with open(path, "wb") as f:
                shutil.copyfileobj(data, f)

    def get(self, job_id: str, rel_path: PurePosixPath) -> bytes:
        path = self._resolve(job_id, rel_path)
        if not path.exists():
            raise FileNotFoundError(f"Artifact not found: {path}")
        return path.read_bytes()

    def get_to_file(self, job_id: str, rel_path: PurePosixPath, dest: Path) -> None:
        data = self.get(job_id, rel_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)

    def list(self, job_id: str, prefix: PurePosixPath | None = None) -> list[PurePosixPath]:
        base = self._root / job_id
        if prefix:
            base = base / str(prefix)
        if not base.exists():
            return []
        return [
            PurePosixPath(p.relative_to(self._root / job_id))
            for p in base.rglob("*")
            if p.is_file()
        ]

    def exists(self, job_id: str, rel_path: PurePosixPath) -> bool:
        return self._resolve(job_id, rel_path).exists()

    def job_root_path(self, job_id: str) -> str:
        return str(self._root / job_id)
