from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

PROVIDER_ENV_KEYS: dict[str, list[str]] = {
    "openai": ["OPENAI_API_KEY"],
    "anthropic": ["ANTHROPIC_API_KEY"],
    "azure": ["AZURE_API_KEY", "AZURE_API_BASE"],
    "bedrock": ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"],
    "vertex_ai": ["GOOGLE_APPLICATION_CREDENTIALS"],
    "google": ["GOOGLE_API_KEY"],
    "groq": ["GROQ_API_KEY"],
    "mistral": ["MISTRAL_API_KEY"],
    "deepseek": ["DEEPSEEK_API_KEY"],
    "together_ai": ["TOGETHER_API_KEY"],
    "ollama": [],
    "ollama_chat": [],
}


def _extract_provider(model: str) -> str:
    """Extract the provider prefix from a LiteLLM-style model string.

    'gpt-4o-mini' -> 'openai'
    'anthropic/claude-3-5-sonnet' -> 'anthropic'
    'ollama/llama3' -> 'ollama'
    """
    if "/" in model:
        return model.split("/", 1)[0]
    return "openai"


def validate_provider_credentials(model: str) -> None:
    """Raise ValueError if required env vars for the model's provider are missing."""
    provider = _extract_provider(model)
    required_keys = PROVIDER_ENV_KEYS.get(provider)

    if required_keys is None:
        logger.debug(
            "Provider '%s' not in known providers list; skipping credential check",
            provider,
        )
        return

    missing = [key for key in required_keys if not os.environ.get(key)]
    if missing:
        raise ValueError(
            f"Model '{model}' requires provider '{provider}' credentials. "
            f"Missing environment variables: {', '.join(missing)}"
        )
