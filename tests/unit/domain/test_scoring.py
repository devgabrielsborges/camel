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

    def test_match_by_class_name(self) -> None:
        score = class_exact_match(
            "Sobre comparação de aeroportos, o melhor é...",
            "A2",
            class_name="comparação de aeroportos",
        )
        assert score.value is True

    def test_match_by_class_name_underscore(self) -> None:
        score = class_exact_match(
            "Suas preferencias pessoais são importantes.",
            "A3",
            class_name="preferencias_pessoais",
        )
        assert score.value is True

    def test_no_match_with_class_name(self) -> None:
        score = class_exact_match(
            "O tempo está bom hoje.",
            "A3",
            class_name="preferencias_pessoais",
        )
        assert score.value is False

    def test_id_takes_priority_over_class_name(self) -> None:
        score = class_exact_match("[A3] resposta.", "A3", class_name="pagamento")
        assert score.value is True


class TestRefusalDetection:
    def test_no_refusal(self) -> None:
        score = refusal_detection("Here is the answer to your question.")
        assert score.value is False

    def test_english_dont_have(self) -> None:
        score = refusal_detection("I don't have information about that.")
        assert score.value is True

    def test_english_cannot(self) -> None:
        score = refusal_detection("I cannot answer that question.")
        assert score.value is True

    def test_english_unable(self) -> None:
        score = refusal_detection("I'm unable to find it.")
        assert score.value is True

    def test_english_never_find(self) -> None:
        score = refusal_detection("I could never find that record.")
        assert score.value is True

    def test_portuguese_nao_possuo(self) -> None:
        score = refusal_detection("Não possuo informações sobre isso.")
        assert score.value is True

    def test_portuguese_nao_consigo(self) -> None:
        score = refusal_detection("Não consigo encontrar essa informação.")
        assert score.value is True

    def test_portuguese_nao_tenho(self) -> None:
        score = refusal_detection("Não tenho essa informação.")
        assert score.value is True

    def test_portuguese_nao_encontrei(self) -> None:
        score = refusal_detection("Não encontrei informações sobre isso.")
        assert score.value is True

    def test_spanish_no_tengo(self) -> None:
        score = refusal_detection("No tengo esa información.")
        assert score.value is True

    def test_spanish_no_puedo(self) -> None:
        score = refusal_detection("No puedo ayudarte con eso.")
        assert score.value is True

    def test_non_refusal_with_negation(self) -> None:
        score = refusal_detection("No problem, here is your answer.")
        assert score.value is False

    def test_non_refusal_portuguese(self) -> None:
        score = refusal_detection("Aqui está a resposta para sua pergunta.")
        assert score.value is False

    def test_metadata_contains_language(self) -> None:
        score = refusal_detection("I don't have that.")
        assert score.metadata is not None
        assert "detected_language" in score.metadata

    def test_scorer_name(self) -> None:
        score = refusal_detection("some text")
        assert score.scorer_name == "refusal_detection"
