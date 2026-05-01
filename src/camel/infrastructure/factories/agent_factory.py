from __future__ import annotations

from camel.application.ports.agent_port import AgentPort
from camel.infrastructure.config.settings import Settings


def create_agent_adapter(settings: Settings, *, model_override: str | None = None) -> AgentPort:
    model = model_override or settings.openai_model
    provider = settings.llm_provider.lower()

    if provider == "litellm":
        from camel.infrastructure.adapters.litellm_agent import LiteLLMAgentAdapter

        return LiteLLMAgentAdapter(model=model)

    from camel.infrastructure.adapters.openai_agent import OpenAIAgentAdapter

    return OpenAIAgentAdapter(model=model)
