from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from camel.domain.value_objects.pass_at_k_result import PassAtKResult

__all__ = [
    "Chunk",
    "ClassDef",
    "PassAtKResult",
    "TokenUsage",
    "ToolCall",
]


class TokenUsage(BaseModel, frozen=True):
    input_tokens: int
    output_tokens: int
    total_tokens: int


class Chunk(BaseModel, frozen=True):
    content: str
    score: float


class ClassDef(BaseModel, frozen=True):
    class_name: str
    context: str
    class_id: str


class ToolCall(BaseModel, frozen=True):
    tool_name: str
    arguments: dict[str, Any]
    result: str | None = None
