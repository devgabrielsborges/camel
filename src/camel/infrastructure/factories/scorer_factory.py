from __future__ import annotations

from typing import Any

from camel.infrastructure.adapters.mlflow_scorer import (
    get_deterministic_scorers,
    get_llm_judge_scorers,
)
from camel.infrastructure.config.settings import Settings


def create_scorers(settings: Settings, *, no_llm_judge: bool = False) -> list[Any]:
    scorers = get_deterministic_scorers()
    if not no_llm_judge:
        scorers.extend(get_llm_judge_scorers(settings.judge_model))
    return scorers
