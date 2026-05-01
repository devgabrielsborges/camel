from __future__ import annotations

from pydantic import BaseModel

from camel.domain.entities.trace import Trace
from camel.domain.value_objects.dataset_record import DatasetRecord


class Session(BaseModel):
    session_id: str
    evaluation_id: str
    traces: list[Trace] = []
    dataset_record: DatasetRecord

    def add_trace(self, trace: Trace) -> None:
        self.traces.append(trace)

    @property
    def is_single_turn(self) -> bool:
        return len(self.traces) == 1
