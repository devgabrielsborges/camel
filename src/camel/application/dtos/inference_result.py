from __future__ import annotations

from pydantic import BaseModel

from camel.domain.value_objects import TokenUsage, ToolCall


class InferenceResult(BaseModel, frozen=True):
    trace_id: str
    session_id: str
    input_text: str
    output_text: str
    token_usage: TokenUsage
    model: str
    latency_ms: int
    tool_calls: list[ToolCall] = []
