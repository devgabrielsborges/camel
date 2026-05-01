from __future__ import annotations

from typing import Protocol

from camel.domain.entities.trace import Trace


class TraceRepository(Protocol):
    def save(self, trace: Trace) -> None: ...

    def find_by_session(self, session_id: str) -> list[Trace]: ...

    def find_by_evaluation(self, evaluation_id: str) -> list[Trace]: ...

    def exists(self, session_id: str) -> bool: ...
