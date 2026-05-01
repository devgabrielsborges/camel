from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field

from camel.domain.value_objects import TokenUsage, ToolCall
from camel.domain.value_objects.score import Score


class Trace(BaseModel):
    trace_id: str
    session_id: str
    input_text: str
    output_text: str
    tool_calls: list[ToolCall] = []
    token_usage: TokenUsage
    model: str
    latency_ms: int
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    scores: list[Score] = []

    def add_score(self, score: Score) -> None:
        self.scores.append(score)

    def validate_output(self) -> None:
        if not self.output_text.strip():
            msg = f"Trace {self.trace_id}: output_text must not be empty"
            raise ValueError(msg)
