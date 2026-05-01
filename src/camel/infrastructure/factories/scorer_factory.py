from __future__ import annotations

from camel.infrastructure.adapters.mlflow_scorer import (
    DeterministicScorer,
    LLMJudgeScorer,
)
from camel.infrastructure.config.settings import Settings


def create_deterministic_scorer() -> DeterministicScorer:
    return DeterministicScorer()


def create_llm_judge_scorer(settings: Settings) -> LLMJudgeScorer:
    return LLMJudgeScorer(judge_model=settings.judge_model)
