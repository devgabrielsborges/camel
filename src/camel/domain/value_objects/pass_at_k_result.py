from __future__ import annotations

from pydantic import BaseModel

from camel.domain.value_objects.score import Score


class PassAtKResult(BaseModel, frozen=True):
    """Result of a Pass@k evaluation for a single question."""

    question_id: str
    k: int
    responses: list[str]
    scores: list[Score]
    passed: bool
    best_score: float
