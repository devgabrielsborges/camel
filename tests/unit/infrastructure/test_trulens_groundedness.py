from __future__ import annotations

from unittest.mock import MagicMock, patch

from camel.application.ports.groundedness_port import GroundednessPort
from camel.infrastructure.adapters.trulens_groundedness import TruLensGroundednessAdapter


class TestTruLensGroundednessAdapter:
    def test_implements_port(self) -> None:
        with patch("camel.infrastructure.adapters.trulens_groundedness.TruLensLiteLLM"):
            adapter = TruLensGroundednessAdapter(model_engine="gpt-4o-mini")
            assert isinstance(adapter, GroundednessPort)

    def test_score_returns_tuple(self) -> None:
        with patch("camel.infrastructure.adapters.trulens_groundedness.TruLensLiteLLM") as mock_cls:
            mock_provider = MagicMock()
            mock_provider.groundedness_measure_with_cot_reasons.return_value = (
                0.85,
                "Statement is supported by source.",
            )
            mock_cls.return_value = mock_provider

            adapter = TruLensGroundednessAdapter(model_engine="gpt-4o-mini")
            score, reasons = adapter.score(
                source="The return policy is 30 days.",
                statement="You can return items within 30 days.",
            )
            assert score == 0.85
            assert "supported" in reasons.lower()

    def test_score_clamps_to_valid_range(self) -> None:
        with patch("camel.infrastructure.adapters.trulens_groundedness.TruLensLiteLLM") as mock_cls:
            mock_provider = MagicMock()
            mock_provider.groundedness_measure_with_cot_reasons.return_value = (
                1.5,
                "Over-scored",
            )
            mock_cls.return_value = mock_provider

            adapter = TruLensGroundednessAdapter(model_engine="gpt-4o-mini")
            score, _ = adapter.score(source="src", statement="stmt")
            assert score == 1.0

    def test_score_handles_negative(self) -> None:
        with patch("camel.infrastructure.adapters.trulens_groundedness.TruLensLiteLLM") as mock_cls:
            mock_provider = MagicMock()
            mock_provider.groundedness_measure_with_cot_reasons.return_value = (
                -0.5,
                "Negative",
            )
            mock_cls.return_value = mock_provider

            adapter = TruLensGroundednessAdapter(model_engine="gpt-4o-mini")
            score, _ = adapter.score(source="src", statement="stmt")
            assert score == 0.0

    def test_score_handles_exception(self) -> None:
        with patch("camel.infrastructure.adapters.trulens_groundedness.TruLensLiteLLM") as mock_cls:
            mock_provider = MagicMock()
            mock_provider.groundedness_measure_with_cot_reasons.side_effect = RuntimeError(
                "API error"
            )
            mock_cls.return_value = mock_provider

            adapter = TruLensGroundednessAdapter(model_engine="gpt-4o-mini")
            score, reasons = adapter.score(source="src", statement="stmt")
            assert score == 0.0
            assert reasons == "scoring_error"

    def test_passes_model_engine_to_provider(self) -> None:
        with patch("camel.infrastructure.adapters.trulens_groundedness.TruLensLiteLLM") as mock_cls:
            TruLensGroundednessAdapter(model_engine="ollama/llama3.2:3b")
            mock_cls.assert_called_once_with(model_engine="ollama/llama3.2:3b")
