from __future__ import annotations

from typing import Protocol

from camel.domain.entities.trace import Trace
from camel.domain.value_objects.dataset_record import DatasetRecord


class AgentPort(Protocol):
    async def invoke(self, record: DatasetRecord, system_prompt: str) -> Trace: ...

    async def flush(self) -> None: ...
