from __future__ import annotations

from typing import Any

from camel.application.ports.groundedness_port import GroundednessPort
from camel.infrastructure.adapters.mlflow_scorer import (
    get_deterministic_scorers,
    get_llm_judge_scorers,
)
from camel.infrastructure.adapters.trulens_groundedness import TruLensGroundednessAdapter
from camel.infrastructure.config.provider_validation import validate_provider_credentials
from camel.infrastructure.config.settings import Settings


def create_scorers(settings: Settings, *, no_llm_judge: bool = False) -> list[Any]:
    scorers = get_deterministic_scorers()
    if not no_llm_judge:
        validate_provider_credentials(settings.judge_model)
        scorers.extend(get_llm_judge_scorers(settings.judge_model))
    return scorers


def create_groundedness_scorer(settings: Settings) -> GroundednessPort:
    """Create a groundedness scorer using TruLens LiteLLM provider."""
    return TruLensGroundednessAdapter(model_engine=settings.judge_model)
