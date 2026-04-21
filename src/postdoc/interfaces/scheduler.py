from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from postdoc.interfaces.tracker import JobTracker


class NoCapacityError(Exception):
    pass


@dataclass
class SchedulerSubmitResult:
    process_handle: str
    gpu_ids: list[int]


class Scheduler(ABC):

    @abstractmethod
    def submit(self, job_id: str, resolved_config: Path, experiment: dict) -> SchedulerSubmitResult:
        ...

    @abstractmethod
    def cancel(self, job_id: str, process_handle: str) -> None:
        ...

    @abstractmethod
    def is_alive(self, process_handle: str) -> bool:
        ...

    @abstractmethod
    def available_capacity(self) -> int:
        ...

    @abstractmethod
    def drain_queue(self, tracker: JobTracker) -> None:
        ...
