from __future__ import annotations

from typing import Protocol

from camel.domain.entities.evaluation import Evaluation, EvaluationStatus


class EvaluationRepository(Protocol):
    def save(self, evaluation: Evaluation) -> None: ...

    def find_by_id(self, evaluation_id: str) -> Evaluation | None: ...

    def update_status(self, evaluation_id: str, status: EvaluationStatus) -> None: ...
