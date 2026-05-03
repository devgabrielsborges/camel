from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from camel.domain.value_objects.category_score_collection import CategoryScoreCollection
from camel.domain.value_objects.metric_threshold import MetricThreshold
from camel.domain.value_objects.metric_type import MetricType
from camel.domain.value_objects.pass_at_k_result import PassAtKResult
from camel.domain.value_objects.test_result import TestResult
from camel.domain.value_objects.threshold_profile import ThresholdProfile

__all__ = [
    "CategoryScoreCollection",
    "Chunk",
    "ClassDef",
    "MetricThreshold",
    "MetricType",
    "PassAtKResult",
    "TestResult",
    "ThresholdProfile",
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
