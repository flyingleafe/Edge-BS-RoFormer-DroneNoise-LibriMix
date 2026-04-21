from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path, PurePosixPath
from typing import BinaryIO


class StorageBackend(ABC):

    @abstractmethod
    def put(self, job_id: str, rel_path: PurePosixPath, data: bytes | BinaryIO) -> None:
        ...

    @abstractmethod
    def get(self, job_id: str, rel_path: PurePosixPath) -> bytes:
        ...

    @abstractmethod
    def get_to_file(self, job_id: str, rel_path: PurePosixPath, dest: Path) -> None:
        ...

    @abstractmethod
    def list(self, job_id: str, prefix: PurePosixPath | None = None) -> list[PurePosixPath]:
        ...

    @abstractmethod
    def exists(self, job_id: str, rel_path: PurePosixPath) -> bool:
        ...

    def put_json(self, job_id: str, rel_path: PurePosixPath, data: dict) -> None:
        self.put(job_id, rel_path, json.dumps(data, indent=2).encode())

    def get_json(self, job_id: str, rel_path: PurePosixPath) -> dict:
        return json.loads(self.get(job_id, rel_path))

    @abstractmethod
    def job_root_path(self, job_id: str) -> str:
        ...
