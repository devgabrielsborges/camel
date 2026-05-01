from __future__ import annotations

from camel.infrastructure.adapters.openai_agent import OpenAIAgentAdapter
from camel.infrastructure.config.settings import Settings


def create_agent_adapter(settings: Settings) -> OpenAIAgentAdapter:
    return OpenAIAgentAdapter(model=settings.openai_model)
