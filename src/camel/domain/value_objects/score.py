from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class Score(BaseModel, frozen=True):
    scorer_name: str
    value: float | bool
    rationale: str | None = None
    metadata: dict[str, Any] = {}

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Score):
            return NotImplemented
        return self.scorer_name == other.scorer_name and self.value == other.value

    def __hash__(self) -> int:
        return hash((self.scorer_name, self.value))
