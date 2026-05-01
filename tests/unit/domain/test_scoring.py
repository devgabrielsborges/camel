from __future__ import annotations

from camel.domain.services.scoring import (
    class_exact_match,
    refusal_detection,
    token_overlap_f1,
)


class TestTokenOverlapF1:
    def test_identical_texts(self) -> None:
        score = token_overlap_f1("hello world", "hello world")
        assert score.value == 1.0
        assert score.scorer_name == "token_overlap_f1"

    def test_no_overlap(self) -> None:
        score = token_overlap_f1("cats are great", "xyz123")
        assert score.value == 0.0

    def test_partial_overlap(self) -> None:
        score = token_overlap_f1("the quick brown fox", "the quick red dog")
        assert 0.0 < score.value < 1.0

    def test_empty_response(self) -> None:
        score = token_overlap_f1("", "some reference")
        assert score.value == 0.0

    def test_empty_reference(self) -> None:
        score = token_overlap_f1("some response", "")
        assert score.value == 0.0

    def test_value_between_0_and_1(self) -> None:
        score = token_overlap_f1(
            "Our return policy allows returns within 30 days of purchase.",
            "Returns are accepted within 30 days. Contact support for details.",
        )
        assert 0.0 <= score.value <= 1.0


class TestClassExactMatch:
    def test_match_found(self) -> None:
        score = class_exact_match("I classified this as [P1] Returns.", "P1")
        assert score.value is True

    def test_no_match(self) -> None:
        score = class_exact_match("This is about shipping.", "P1")
        assert score.value is False

    def test_case_insensitive(self) -> None:
        score = class_exact_match("Category: p1 returns", "P1")
        assert score.value is True

    def test_scorer_name(self) -> None:
        score = class_exact_match("response", "A1")
        assert score.scorer_name == "class_exact_match"


class TestRefusalDetection:
    def test_no_refusal(self) -> None:
        score = refusal_detection("Here is the answer to your question.")
        assert score.value is False

    def test_english_refusal(self) -> None:
        score = refusal_detection("I don't have information about that.")
        assert score.value is True

    def test_portuguese_refusal(self) -> None:
        score = refusal_detection("Não possuo informações sobre isso.")
        assert score.value is True

    def test_spanish_refusal(self) -> None:
        score = refusal_detection("No tengo esa información.")
        assert score.value is True

    def test_cannot_refusal(self) -> None:
        score = refusal_detection("I cannot answer that question.")
        assert score.value is True

    def test_scorer_name(self) -> None:
        score = refusal_detection("some text")
        assert score.scorer_name == "refusal_detection"
