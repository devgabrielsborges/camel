from __future__ import annotations

from camel.domain.services.scoring import pass_at_k, token_overlap_f1
from camel.domain.value_objects.pass_at_k_result import PassAtKResult


class TestPassAtK:
    def test_all_responses_pass(self) -> None:
        reference = "Returns are accepted within 30 days."
        responses = [
            "Returns are accepted within 30 days.",
            "You can return items within 30 days.",
            "Our return policy is 30 days.",
        ]
        result = pass_at_k("q1", responses, reference, token_overlap_f1)
        assert result.passed is True
        assert result.best_score > 0.0
        assert result.k == 3
        assert result.question_id == "q1"

    def test_one_response_passes(self) -> None:
        reference = "Returns are accepted within 30 days."
        responses = [
            "xyz completely unrelated",
            "abc another unrelated",
            "Returns are accepted within 30 days.",
        ]
        result = pass_at_k("q2", responses, reference, token_overlap_f1)
        assert result.passed is True
        assert result.best_score >= 0.3

    def test_no_responses_pass(self) -> None:
        reference = "Returns are accepted within 30 days."
        responses = [
            "xyz completely unrelated text here",
            "abc more unrelated content",
            "def still not matching at all",
        ]
        result = pass_at_k("q3", responses, reference, token_overlap_f1, threshold=0.9)
        assert result.passed is False

    def test_custom_threshold(self) -> None:
        reference = "The sky is blue and the grass is green."
        responses = ["The sky is blue.", "Unrelated.", "Also unrelated."]
        result_low = pass_at_k("q4", responses, reference, token_overlap_f1, threshold=0.1)
        result_high = pass_at_k("q5", responses, reference, token_overlap_f1, threshold=0.99)
        assert result_low.passed is True
        assert result_high.passed is False

    def test_returns_correct_number_of_scores(self) -> None:
        reference = "test"
        responses = ["a", "b", "c"]
        result = pass_at_k("q6", responses, reference, token_overlap_f1)
        assert len(result.scores) == 3
        assert len(result.responses) == 3

    def test_single_response(self) -> None:
        reference = "Hello world"
        responses = ["Hello world"]
        result = pass_at_k("q7", responses, reference, token_overlap_f1)
        assert result.k == 1
        assert result.passed is True
        assert result.best_score == 1.0

    def test_result_is_frozen(self) -> None:
        reference = "test"
        responses = ["test"]
        result = pass_at_k("q8", responses, reference, token_overlap_f1)
        assert isinstance(result, PassAtKResult)

    def test_best_score_is_maximum(self) -> None:
        reference = "The return policy allows returns within 30 days of purchase."
        responses = [
            "Returns within 30 days.",
            "The return policy allows returns within 30 days of purchase.",
            "Unrelated text.",
        ]
        result = pass_at_k("q9", responses, reference, token_overlap_f1)
        individual_scores = [
            token_overlap_f1(r, reference).value for r in responses
        ]
        assert result.best_score == round(max(float(s) for s in individual_scores if s is not None), 4)  # type: ignore[arg-type]
