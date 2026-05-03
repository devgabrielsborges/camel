from __future__ import annotations

import pytest

from camel.infrastructure.adapters.mlflow_scorer import _build_judge_model_uri


class TestBuildJudgeModelUri:
    @pytest.mark.parametrize(
        ("model", "expected_uri"),
        [
            ("gpt-4o-mini", "openai:/gpt-4o-mini"),
            ("gpt-4o", "openai:/gpt-4o"),
            ("o3-mini", "openai:/o3-mini"),
            ("anthropic/claude-3-5-sonnet-20241022", "anthropic:/claude-3-5-sonnet-20241022"),
            ("anthropic/claude-3-haiku-20240307", "anthropic:/claude-3-haiku-20240307"),
            ("bedrock/anthropic.claude-v2", "bedrock:/anthropic.claude-v2"),
            ("azure/my-deployment", "azure:/my-deployment"),
            ("vertex_ai/gemini-2.0-flash", "vertex_ai:/gemini-2.0-flash"),
            ("google/gemini-2.0-flash", "google:/gemini-2.0-flash"),
            ("groq/llama-3.1-8b-instant", "groq:/llama-3.1-8b-instant"),
            ("mistral/mistral-large-latest", "mistral:/mistral-large-latest"),
            ("deepseek/deepseek-chat", "deepseek:/deepseek-chat"),
            ("together_ai/meta-llama/Llama-3-70b", "together_ai:/meta-llama/Llama-3-70b"),
            ("ollama/llama3", "ollama:/llama3"),
            ("ollama_chat/phi3", "ollama_chat:/phi3"),
        ],
    )
    def test_model_uri_construction(self, model: str, expected_uri: str) -> None:
        assert _build_judge_model_uri(model) == expected_uri

    def test_bare_model_defaults_to_openai(self) -> None:
        assert _build_judge_model_uri("chatgpt-4o-latest").startswith("openai:/")

    def test_provider_slash_model_splits_on_first_slash(self) -> None:
        uri = _build_judge_model_uri("together_ai/meta-llama/Llama-3-70b")
        assert uri == "together_ai:/meta-llama/Llama-3-70b"
