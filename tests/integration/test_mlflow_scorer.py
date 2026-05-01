from __future__ import annotations

import os

import pytest
from mlflow.entities import Feedback

from camel.infrastructure.adapters.mlflow_scorer import (
    class_exact_match,
    get_deterministic_scorers,
    get_llm_judge_scorers,
    refusal_detection,
    token_overlap_f1,
)

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")


class TestTokenOverlapF1Scorer:
    def test_returns_feedback(self) -> None:
        result = token_overlap_f1(
            outputs={"response": "returns within 30 days"},
            expectations={"expected_response": "returns accepted within 30 days"},
        )
        assert isinstance(result, Feedback)
        assert result.name == "token_overlap_f1"
        assert result.feedback.value > 0.0

    def test_no_overlap(self) -> None:
        result = token_overlap_f1(
            outputs={"response": "xyz"},
            expectations={"expected_response": "abc def ghi"},
        )
        assert result.feedback.value == 0.0


class TestClassExactMatchScorer:
    def test_match(self) -> None:
        result = class_exact_match(
            outputs={"response": "This is classified as P1."},
            expectations={"chosen_class_id": "P1"},
        )
        assert isinstance(result, Feedback)
        assert result.name == "class_exact_match"
        assert result.feedback.value is True

    def test_no_expected_class(self) -> None:
        result = class_exact_match(
            outputs={"response": "some answer"},
            expectations={"chosen_class_id": ""},
        )
        assert result.feedback.value is False


class TestRefusalDetectionScorer:
    def test_refusal_detected(self) -> None:
        result = refusal_detection(
            outputs={"response": "I don't have that information."},
        )
        assert isinstance(result, Feedback)
        assert result.name == "refusal_detection"
        assert result.feedback.value is True

    def test_no_refusal(self) -> None:
        result = refusal_detection(
            outputs={"response": "Here is the answer to your question."},
        )
        assert result.feedback.value is False


class TestGetDeterministicScorers:
    def test_returns_three(self) -> None:
        scorers = get_deterministic_scorers()
        assert len(scorers) == 3
        names = {getattr(s, "name", "") for s in scorers}
        assert "token_overlap_f1" in names
        assert "class_exact_match" in names
        assert "refusal_detection" in names


@pytest.mark.skipif(not OPENAI_API_KEY, reason="OPENAI_API_KEY not set")
class TestLLMJudgeScorers:
    def test_returns_list(self) -> None:
        judge_model = os.environ.get("JUDGE_MODEL", "gpt-4o-mini")
        scorers = get_llm_judge_scorers(judge_model)
        assert isinstance(scorers, list)
        assert len(scorers) == 2
