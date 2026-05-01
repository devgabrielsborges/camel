from __future__ import annotations

from pydantic import BaseModel

from camel.domain.value_objects.score import Score


class EvaluationResult(BaseModel, frozen=True):
    trace_id: str
    scores: list[Score]
