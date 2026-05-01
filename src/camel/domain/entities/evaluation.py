from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field

from camel.domain.entities.session import Session
from camel.domain.value_objects.model_config import ModelConfig


class EvaluationStatus(str, Enum):
    PENDING = "pending"
    INFERRING = "inferring"
    EVALUATING = "evaluating"
    COMPLETE = "complete"
    FAILED = "failed"


_VALID_TRANSITIONS: dict[EvaluationStatus, set[EvaluationStatus]] = {
    EvaluationStatus.PENDING: {EvaluationStatus.INFERRING, EvaluationStatus.FAILED},
    EvaluationStatus.INFERRING: {EvaluationStatus.EVALUATING, EvaluationStatus.FAILED},
    EvaluationStatus.EVALUATING: {EvaluationStatus.COMPLETE, EvaluationStatus.FAILED},
    EvaluationStatus.COMPLETE: set(),
    EvaluationStatus.FAILED: set(),
}


class Evaluation(BaseModel):
    evaluation_id: str
    experiment_name: str
    eval_model: ModelConfig
    prompt_version: str
    dataset_name: str
    status: EvaluationStatus = EvaluationStatus.PENDING
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime | None = None
    sessions: list[Session] = []

    def transition_to(self, new_status: EvaluationStatus) -> None:
        allowed = _VALID_TRANSITIONS.get(self.status, set())
        if new_status not in allowed:
            msg = f"Invalid transition: {self.status.value} → {new_status.value}"
            raise ValueError(msg)
        self.status = new_status
        if new_status in {EvaluationStatus.COMPLETE, EvaluationStatus.FAILED}:
            self.completed_at = datetime.now(timezone.utc)

    def add_session(self, session: Session) -> None:
        self.sessions.append(session)
