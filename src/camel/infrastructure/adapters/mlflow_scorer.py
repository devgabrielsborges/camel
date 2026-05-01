from __future__ import annotations

import logging
from typing import Any

from mlflow.entities import Feedback
from mlflow.genai.scorers import scorer

from camel.domain.services.scoring import class_exact_match as _class_exact_match
from camel.domain.services.scoring import refusal_detection as _refusal_detection
from camel.domain.services.scoring import token_overlap_f1 as _token_overlap_f1

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


def get_deterministic_scorers() -> list[Any]:
    return [token_overlap_f1, class_exact_match, refusal_detection]


def get_llm_judge_scorers(judge_model: str) -> list[Any]:
    try:
        from mlflow.genai.scorers import Correctness, Guidelines

        model_uri = f"openai:/{judge_model}"
        return [
            Correctness(model=model_uri),
            Guidelines(guidelines="Be helpful and accurate.", model=model_uri),
        ]
    except ImportError:
        logger.warning("MLflow genai scorers not available")
        return []
