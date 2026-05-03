from __future__ import annotations

from camel.domain.services.scoring import (
    chunk_attribution,
    class_exact_match,
    hedging_detection,
    question_response_overlap,
    refusal_detection,
    response_length_ratio,
    rouge_l,
    self_consistency,
    token_overlap_f1,
)
from camel.domain.value_objects import Chunk


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


class TestSelfConsistency:
    def test_identical_responses(self) -> None:
        score = self_consistency(["hello world", "hello world", "hello world"])
        assert score.value == 1.0
        assert score.scorer_name == "self_consistency"

    def test_different_responses(self) -> None:
        score = self_consistency(["cats are great", "xyz 123 abc", "foo bar baz"])
        assert score.value is not None
        assert score.value < 1.0

    def test_single_response_returns_none(self) -> None:
        score = self_consistency(["only one response"])
        assert score.value is None

    def test_two_responses(self) -> None:
        score = self_consistency(["the quick brown fox", "the quick red dog"])
        assert score.value is not None
        assert 0.0 < score.value < 1.0
        assert score.metadata["n_pairs"] == 1

    def test_metadata_shape(self) -> None:
        score = self_consistency(["a", "b", "c"])
        assert "variance" in score.metadata
        assert "n_pairs" in score.metadata
        assert score.metadata["n_pairs"] == 3

    def test_high_agreement_low_variance(self) -> None:
        score = self_consistency(["hello world test", "hello world test", "hello world test"])
        assert score.metadata["variance"] == 0.0


class TestHedgingDetection:
    def test_no_hedging(self) -> None:
        score = hedging_detection("Here is the definitive answer to your question.")
        assert score.value is False
        assert score.scorer_name == "hedging_detection"

    def test_english_maybe(self) -> None:
        score = hedging_detection("Maybe the return policy is 30 days.")
        assert score.value is True
        assert score.metadata["hedging_count"] >= 1

    def test_english_perhaps(self) -> None:
        score = hedging_detection("Perhaps this is the right answer.")
        assert score.value is True

    def test_english_possibly(self) -> None:
        score = hedging_detection("The answer is possibly related to returns.")
        assert score.value is True

    def test_english_i_think(self) -> None:
        score = hedging_detection("I think the return policy is 30 days.")
        assert score.value is True

    def test_english_it_seems(self) -> None:
        score = hedging_detection("It seems that shipping takes 5 days.")
        assert score.value is True

    def test_english_i_believe(self) -> None:
        score = hedging_detection("I believe the policy allows 30 day returns.")
        assert score.value is True

    def test_english_apparently(self) -> None:
        score = hedging_detection("Apparently the deadline is next week.")
        assert score.value is True

    def test_portuguese_talvez(self) -> None:
        score = hedging_detection("Talvez o prazo de devolução seja de 30 dias.")
        assert score.value is True

    def test_portuguese_eu_acho(self) -> None:
        score = hedging_detection("Eu acho que o prazo é de 30 dias.")
        assert score.value is True

    def test_portuguese_possivelmente(self) -> None:
        score = hedging_detection("Possivelmente a resposta está aqui.")
        assert score.value is True

    def test_spanish_quizas(self) -> None:
        score = hedging_detection("Quizás el plazo de devolución es de 30 días.")
        assert score.value is True

    def test_spanish_yo_creo(self) -> None:
        score = hedging_detection("Yo creo que la política es de 30 días.")
        assert score.value is True

    def test_spanish_posiblemente(self) -> None:
        score = hedging_detection("Posiblemente la respuesta sea correcta.")
        assert score.value is True

    def test_metadata_contains_stem_info(self) -> None:
        score = hedging_detection("Maybe this is correct.")
        assert "matches" in score.metadata
        assert len(score.metadata["matches"]) > 0
        match = score.metadata["matches"][0]
        assert "language" in match
        assert "stem" in match
        assert "type" in match

    def test_multiple_hedging_stems(self) -> None:
        score = hedging_detection("I think it's probably the right answer, perhaps.")
        assert score.metadata["hedging_count"] >= 2

    def test_confident_answer_no_hedging(self) -> None:
        score = hedging_detection("The return policy allows returns within 30 days of purchase.")
        assert score.value is False


class TestQuestionResponseOverlap:
    def test_high_overlap(self) -> None:
        score = question_response_overlap(
            "The return policy allows returns within 30 days.",
            "What is the return policy?",
        )
        assert score.value is not None
        assert score.value > 0.0
        assert score.scorer_name == "question_response_overlap"

    def test_no_overlap(self) -> None:
        score = question_response_overlap("cats dogs animals", "xyz 123 abc")
        assert score.value == 0.0

    def test_empty_response(self) -> None:
        score = question_response_overlap("", "What is the policy?")
        assert score.value == 0.0

    def test_empty_question(self) -> None:
        score = question_response_overlap("Some response text", "")
        assert score.value == 0.0

    def test_identical(self) -> None:
        score = question_response_overlap("hello world", "hello world")
        assert score.value == 1.0

    def test_value_between_0_and_1(self) -> None:
        score = question_response_overlap(
            "Returns are accepted within 30 days of purchase for full refund.",
            "What is the return and refund policy?",
        )
        assert 0.0 < score.value < 1.0


class TestResponseLengthRatio:
    def test_equal_length(self) -> None:
        score = response_length_ratio("hello world", "hello world")
        assert score.value is not None
        assert score.value == 1.0
        assert score.scorer_name == "response_length_ratio"

    def test_longer_response(self) -> None:
        score = response_length_ratio(
            "This is a much longer response with many tokens.",
            "Short ref.",
        )
        assert score.value is not None
        assert score.value > 1.0

    def test_shorter_response(self) -> None:
        score = response_length_ratio("Short.", "This is a much longer reference text.")
        assert score.value is not None
        assert score.value < 1.0

    def test_empty_reference_returns_none(self) -> None:
        score = response_length_ratio("Some response", "")
        assert score.value is None

    def test_metadata_contains_counts(self) -> None:
        score = response_length_ratio("hello world", "hello world")
        assert "response_tokens" in score.metadata
        assert "reference_tokens" in score.metadata
        assert score.metadata["response_tokens"] > 0
        assert score.metadata["reference_tokens"] > 0


class TestRougeL:
    def test_identical_texts(self) -> None:
        score = rouge_l("hello world", "hello world")
        assert score.value == 1.0
        assert score.scorer_name == "rouge_l"

    def test_no_overlap(self) -> None:
        score = rouge_l("cats are great", "xyz 123 abc")
        assert score.value == 0.0

    def test_partial_subsequence(self) -> None:
        score = rouge_l("the quick brown fox", "the quick red dog")
        assert 0.0 < score.value < 1.0

    def test_empty_response(self) -> None:
        score = rouge_l("", "some reference")
        assert score.value == 0.0

    def test_empty_reference(self) -> None:
        score = rouge_l("some response", "")
        assert score.value == 0.0

    def test_preserves_order(self) -> None:
        ordered = rouge_l("a b c d e", "a b c d e")
        shuffled = rouge_l("e d c b a", "a b c d e")
        assert ordered.value >= shuffled.value

    def test_value_between_0_and_1(self) -> None:
        score = rouge_l(
            "Our return policy allows returns within 30 days of purchase.",
            "Returns are accepted within 30 days. Contact support for details.",
        )
        assert 0.0 <= score.value <= 1.0


class TestChunkAttribution:
    def test_high_attribution_to_relevant_chunk(self) -> None:
        chunks = [
            Chunk(content="Returns are accepted within 30 days of purchase.", score=1.5),
            Chunk(content="Shipping takes 5-7 business days.", score=0.3),
        ]
        score = chunk_attribution("Returns are accepted within 30 days of purchase.", chunks)
        assert score.value is not None
        assert score.value > 0.5
        assert score.scorer_name == "chunk_attribution"

    def test_no_match(self) -> None:
        chunks = [
            Chunk(content="This is about cats and dogs.", score=1.0),
            Chunk(content="Weather forecast for today.", score=0.5),
        ]
        score = chunk_attribution("xyz 123 completely unrelated", chunks)
        assert score.value is not None
        assert score.value < 0.1

    def test_empty_chunks_returns_none(self) -> None:
        score = chunk_attribution("Some response", [])
        assert score.value is None

    def test_single_chunk(self) -> None:
        chunks = [Chunk(content="Returns accepted within 30 days.", score=1.0)]
        score = chunk_attribution("Returns accepted within 30 days.", chunks)
        assert score.value is not None
        assert score.value > 0.5

    def test_metadata_contains_entropy(self) -> None:
        chunks = [
            Chunk(content="Returns accepted within 30 days.", score=1.5),
            Chunk(content="Shipping takes 5 business days.", score=0.3),
            Chunk(content="Contact support for refund details.", score=0.8),
        ]
        score = chunk_attribution("Returns accepted within 30 days.", chunks)
        assert "entropy" in score.metadata
        assert "score_correlation" in score.metadata
        assert "per_chunk" in score.metadata
        assert len(score.metadata["per_chunk"]) == 3

    def test_correlation_positive_for_aligned_attribution(self) -> None:
        chunks = [
            Chunk(content="Returns accepted within 30 days of purchase.", score=2.0),
            Chunk(content="Unrelated content about weather.", score=0.1),
            Chunk(content="Somewhat related refund information.", score=0.5),
        ]
        score = chunk_attribution("Returns accepted within 30 days of purchase.", chunks)
        assert score.metadata["score_correlation"] > 0.0

    def test_two_chunks_skips_correlation(self) -> None:
        chunks = [
            Chunk(content="Returns accepted.", score=1.0),
            Chunk(content="Shipping info.", score=0.5),
        ]
        score = chunk_attribution("Returns accepted.", chunks)
        assert score.metadata["score_correlation"] == 0.0
