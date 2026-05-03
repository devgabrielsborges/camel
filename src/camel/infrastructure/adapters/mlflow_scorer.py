from __future__ import annotations

import logging
from typing import Any

from mlflow.entities import Feedback
from mlflow.genai.scorers import scorer

from camel.domain.services.scoring import chunk_attribution as _chunk_attribution
from camel.domain.services.scoring import class_exact_match as _class_exact_match
from camel.domain.services.scoring import hedging_detection as _hedging_detection
from camel.domain.services.scoring import question_response_overlap as _question_response_overlap
from camel.domain.services.scoring import refusal_detection as _refusal_detection
from camel.domain.services.scoring import response_length_ratio as _response_length_ratio
from camel.domain.services.scoring import rouge_l as _rouge_l
from camel.domain.services.scoring import token_overlap_f1 as _token_overlap_f1
from camel.domain.value_objects import Chunk

logger = logging.getLogger(__name__)


@scorer
def token_overlap_f1(
    *,
    inputs: dict[str, Any] | None = None,
    outputs: dict[str, Any] | None = None,
    expectations: dict[str, Any] | None = None,
) -> Feedback:
    result = _token_overlap_f1(
        (outputs or {}).get("response", ""),
        (expectations or {}).get("expected_response", ""),
    )
    return Feedback(name="token_overlap_f1", value=result.value)


@scorer
def class_exact_match(
    *,
    inputs: dict[str, Any] | None = None,
    outputs: dict[str, Any] | None = None,
    expectations: dict[str, Any] | None = None,
) -> Feedback:
    exp = expectations or {}
    expected_class = exp.get("chosen_class_id", "")
    if not expected_class:
        return Feedback(
            name="class_exact_match",
            error="N/A: no expected class provided",
        )

    classes: list[dict[str, str]] = exp.get("classes", [])
    class_name = next(
        (c.get("class", "") for c in classes if c.get("id") == expected_class),
        "",
    )

    if not class_name:
        return Feedback(
            name="class_exact_match",
            error=f"N/A: class {expected_class} not in classes list",
        )

    result = _class_exact_match(
        (outputs or {}).get("response", ""),
        expected_class,
        class_name=class_name,
    )
    return Feedback(name="class_exact_match", value=bool(result.value))


@scorer
def refusal_detection(
    *,
    inputs: dict[str, Any] | None = None,
    outputs: dict[str, Any] | None = None,
    expectations: dict[str, Any] | None = None,
) -> Feedback:
    result = _refusal_detection((outputs or {}).get("response", ""))
    return Feedback(name="refusal_detection", value=bool(result.value))


@scorer
def groundedness(
    *,
    inputs: dict[str, Any] | None = None,
    outputs: dict[str, Any] | None = None,
    expectations: dict[str, Any] | None = None,
) -> Feedback:
    score_val = (outputs or {}).get("groundedness_score")
    if score_val is None:
        return Feedback(name="groundedness", error="N/A: not computed")
    return Feedback(name="groundedness", value=float(score_val))


@scorer
def hedging_detection(
    *,
    inputs: dict[str, Any] | None = None,
    outputs: dict[str, Any] | None = None,
    expectations: dict[str, Any] | None = None,
) -> Feedback:
    result = _hedging_detection((outputs or {}).get("response", ""))
    return Feedback(name="hedging_detection", value=bool(result.value))


@scorer
def question_response_overlap(
    *,
    inputs: dict[str, Any] | None = None,
    outputs: dict[str, Any] | None = None,
    expectations: dict[str, Any] | None = None,
) -> Feedback:
    result = _question_response_overlap(
        (outputs or {}).get("response", ""),
        (inputs or {}).get("question", ""),
    )
    return Feedback(name="question_response_overlap", value=result.value)


@scorer
def response_length_ratio(
    *,
    inputs: dict[str, Any] | None = None,
    outputs: dict[str, Any] | None = None,
    expectations: dict[str, Any] | None = None,
) -> Feedback:
    result = _response_length_ratio(
        (outputs or {}).get("response", ""),
        (expectations or {}).get("expected_response", ""),
    )
    if result.value is None:
        return Feedback(name="response_length_ratio", error="N/A: empty reference")
    return Feedback(name="response_length_ratio", value=result.value)


@scorer
def rouge_l(
    *,
    inputs: dict[str, Any] | None = None,
    outputs: dict[str, Any] | None = None,
    expectations: dict[str, Any] | None = None,
) -> Feedback:
    result = _rouge_l(
        (outputs or {}).get("response", ""),
        (expectations or {}).get("expected_response", ""),
    )
    return Feedback(name="rouge_l", value=result.value)


@scorer
def chunk_attribution(
    *,
    inputs: dict[str, Any] | None = None,
    outputs: dict[str, Any] | None = None,
    expectations: dict[str, Any] | None = None,
) -> Feedback:
    exp = expectations or {}
    raw_chunks: list[dict[str, Any]] = exp.get("chunks", [])
    if not raw_chunks:
        return Feedback(name="chunk_attribution", error="N/A: no chunks provided")
    chunks = [Chunk(content=c["content"], score=c["score"]) for c in raw_chunks]
    result = _chunk_attribution((outputs or {}).get("response", ""), chunks)
    if result.value is None:
        return Feedback(name="chunk_attribution", error="N/A: empty chunks")
    return Feedback(name="chunk_attribution", value=result.value)


def get_deterministic_scorers() -> list[Any]:
    return [
        token_overlap_f1,
        class_exact_match,
        refusal_detection,
        hedging_detection,
        question_response_overlap,
        response_length_ratio,
        rouge_l,
        chunk_attribution,
    ]


def _build_judge_model_uri(judge_model: str) -> str:
    """Build MLflow model URI from a LiteLLM-style model string.

    Maps 'provider/model' to 'provider:/model' (MLflow URI scheme).
    Bare model names without a provider prefix default to 'openai:/'.

    Examples:
        'gpt-4o-mini' -> 'openai:/gpt-4o-mini'
        'anthropic/claude-3-5-sonnet' -> 'anthropic:/claude-3-5-sonnet'
        'bedrock/anthropic.claude-v2' -> 'bedrock:/anthropic.claude-v2'
        'azure/my-deployment' -> 'azure:/my-deployment'
    """
    if "/" in judge_model:
        provider, model_path = judge_model.split("/", 1)
        return f"{provider}:/{model_path}"
    return f"openai:/{judge_model}"


def get_llm_judge_scorers(judge_model: str) -> list[Any]:
    try:
        from mlflow.genai.scorers import Correctness, Guidelines

        model_uri = _build_judge_model_uri(judge_model)
        return [
            Correctness(model=model_uri),
            Guidelines(guidelines="Be helpful and accurate.", model=model_uri),
        ]
    except ImportError:
        logger.warning("MLflow genai scorers not available")
        return []
