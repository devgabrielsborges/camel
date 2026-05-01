from __future__ import annotations

import os

import pytest

from camel.infrastructure.adapters.mlflow_scorer import (
    DeterministicScorer,
    LLMJudgeScorer,
)

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")


class TestDeterministicScorer:
    def test_score_returns_scores(self) -> None:
        scorer = DeterministicScorer()
        scores = scorer.score(
            inputs={"question": "What is the return policy?"},
            outputs={"response": "Returns accepted within 30 days."},
            expectations={
                "expected_response": "Our return policy allows returns within 30 days.",
                "chosen_class_id": "P1",
            },
        )

        names = {s.scorer_name for s in scores}
        assert "token_overlap_f1" in names
        assert "class_exact_match" in names
        assert "refusal_detection" in names

    def test_f1_score_is_positive_for_overlap(self) -> None:
        scorer = DeterministicScorer()
        scores = scorer.score(
            inputs={"question": "test"},
            outputs={"response": "returns within 30 days"},
            expectations={"expected_response": "returns accepted within 30 days"},
        )
        f1 = next(s for s in scores if s.scorer_name == "token_overlap_f1")
        assert f1.value > 0.0


@pytest.mark.skipif(not OPENAI_API_KEY, reason="OPENAI_API_KEY not set")
class TestLLMJudgeScorer:
    def test_score_returns_list(self) -> None:
        judge_model = os.environ.get("JUDGE_MODEL", "gpt-4o-mini")
        scorer = LLMJudgeScorer(judge_model=judge_model)
        scores = scorer.score(
            inputs={"question": "What is your return policy?"},
            outputs={"response": "We accept returns within 30 days."},
            expectations={
                "expected_response": "Returns are accepted within 30 days of purchase.",
                "guidelines": "Be polite and accurate.",
            },
        )
        assert isinstance(scores, list)
