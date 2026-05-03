from __future__ import annotations

from enum import StrEnum

from camel.domain.value_objects.score import Score


class FailureMode(StrEnum):
    CORRECT_EXTRACTION = "correct_extraction"
    CORRECT_REFUSAL = "correct_refusal"
    FALSE_REFUSAL = "false_refusal"
    HALLUCINATION = "hallucination"
    HEDGING_ANSWER = "hedging_answer"
    OFF_TOPIC = "off_topic"
    PARTIAL_ANSWER = "partial_answer"


def classify_failure_mode(
    scores: dict[str, Score],
    category: str,
) -> FailureMode:
    """Classify a prediction into a failure mode based on scorer results and category.

    Decision logic:
    - refusal + positivo → false_refusal
    - refusal + negativo → correct_refusal
    - high overlap + positivo → correct_extraction
    - moderate overlap + negativo → hallucination
    - very low overlap + no refusal → off_topic
    - hedging + positivo + moderate overlap → hedging_answer
    - otherwise → partial_answer
    """
    refusal_score = scores.get("refusal_detection")
    overlap_score = scores.get("token_overlap_f1")
    hedging_score = scores.get("hedging_detection")

    is_refusal = bool(refusal_score and refusal_score.value is True)
    is_hedging = bool(hedging_score and hedging_score.value is True)
    overlap_value = (
        float(overlap_score.value) if overlap_score and overlap_score.value is not None else 0.0
    )

    if is_refusal and category == "positivo":
        return FailureMode.FALSE_REFUSAL
    if is_refusal and category == "negativo":
        return FailureMode.CORRECT_REFUSAL
    if overlap_value > 0.5 and category == "positivo":
        return FailureMode.CORRECT_EXTRACTION
    if overlap_value > 0.3 and category == "negativo":
        return FailureMode.HALLUCINATION
    if overlap_value < 0.1 and not is_refusal:
        return FailureMode.OFF_TOPIC
    if is_hedging and category == "positivo":
        return FailureMode.HEDGING_ANSWER

    return FailureMode.PARTIAL_ANSWER
