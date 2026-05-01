from __future__ import annotations

from pydantic import BaseModel


class ModelConfig(BaseModel, frozen=True):
    model_name: str
    temperature: float = 0.0
    max_tokens: int | None = None
    top_p: float = 1.0
