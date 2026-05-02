from __future__ import annotations

from camel.application.ports.agent_port import AgentPort
from camel.infrastructure.config.settings import Settings


def _resolve_api_base(model: str, settings: Settings) -> str | None:
    """Only pass a custom base URL for ollama models; let LiteLLM handle other providers."""
    if model.startswith("ollama/") or model.startswith("ollama_chat/"):
        return settings.ollama_api_base
    return None


def create_agent_adapter(settings: Settings, *, model_override: str | None = None) -> AgentPort:
    model = model_override or settings.openai_model
    provider = settings.llm_provider.lower()
    api_base = _resolve_api_base(model, settings)

    if provider == "litellm":
        from camel.infrastructure.adapters.litellm_agent import LiteLLMAgentAdapter

        return LiteLLMAgentAdapter(model=model, api_base=api_base)

    from camel.infrastructure.adapters.openai_agent import OpenAIAgentAdapter

    return OpenAIAgentAdapter(model=model, api_base=api_base)
