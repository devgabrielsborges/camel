from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from camel.infrastructure.config.provider_validation import (
    _extract_provider,
    validate_provider_credentials,
)


class TestExtractProvider:
    @pytest.mark.parametrize(
        ("model", "expected_provider"),
        [
            ("gpt-4o-mini", "openai"),
            ("o3-mini", "openai"),
            ("anthropic/claude-3-5-sonnet", "anthropic"),
            ("azure/my-deployment", "azure"),
            ("bedrock/anthropic.claude-v2", "bedrock"),
            ("vertex_ai/gemini-2.0-flash", "vertex_ai"),
            ("google/gemini-2.0-flash", "google"),
            ("groq/llama-3.1-8b-instant", "groq"),
            ("mistral/mistral-large-latest", "mistral"),
            ("deepseek/deepseek-chat", "deepseek"),
            ("together_ai/meta-llama/Llama-3-70b", "together_ai"),
            ("ollama/llama3", "ollama"),
            ("ollama_chat/phi3", "ollama_chat"),
        ],
    )
    def test_provider_extraction(self, model: str, expected_provider: str) -> None:
        assert _extract_provider(model) == expected_provider


class TestValidateProviderCredentials:
    def test_openai_passes_when_key_set(self) -> None:
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}):
            validate_provider_credentials("gpt-4o-mini")

    def test_openai_raises_when_key_missing(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="OPENAI_API_KEY"):
                validate_provider_credentials("gpt-4o-mini")

    def test_anthropic_passes_when_key_set(self) -> None:
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-test"}):
            validate_provider_credentials("anthropic/claude-3-5-sonnet")

    def test_anthropic_raises_when_key_missing(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
                validate_provider_credentials("anthropic/claude-3-5-sonnet")

    def test_azure_raises_when_partial_keys(self) -> None:
        with patch.dict(os.environ, {"AZURE_API_KEY": "key"}, clear=True):
            with pytest.raises(ValueError, match="AZURE_API_BASE"):
                validate_provider_credentials("azure/my-deployment")

    def test_azure_passes_when_all_keys_set(self) -> None:
        env = {"AZURE_API_KEY": "key", "AZURE_API_BASE": "https://x.openai.azure.com/"}
        with patch.dict(os.environ, env, clear=True):
            validate_provider_credentials("azure/my-deployment")

    def test_bedrock_requires_both_aws_keys(self) -> None:
        with patch.dict(os.environ, {"AWS_ACCESS_KEY_ID": "id"}, clear=True):
            with pytest.raises(ValueError, match="AWS_SECRET_ACCESS_KEY"):
                validate_provider_credentials("bedrock/anthropic.claude-v2")

    def test_ollama_requires_no_keys(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            validate_provider_credentials("ollama/llama3")

    def test_ollama_chat_requires_no_keys(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            validate_provider_credentials("ollama_chat/phi3")

    def test_unknown_provider_skips_validation(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            validate_provider_credentials("some_new_provider/some-model")

    def test_groq_raises_when_key_missing(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="GROQ_API_KEY"):
                validate_provider_credentials("groq/llama-3.1-8b-instant")

    def test_groq_passes_when_key_set(self) -> None:
        with patch.dict(os.environ, {"GROQ_API_KEY": "gsk_test"}):
            validate_provider_credentials("groq/llama-3.1-8b-instant")

    def test_error_message_includes_model_and_provider(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="mistral/mistral-large-latest") as exc_info:
                validate_provider_credentials("mistral/mistral-large-latest")
            assert "mistral" in str(exc_info.value)
            assert "MISTRAL_API_KEY" in str(exc_info.value)
