from __future__ import annotations

import logging
from typing import Any

from camel.domain.services.scoring import (
    class_exact_match,
    refusal_detection,
    token_overlap_f1,
)
from camel.domain.value_objects.score import Score

logger = logging.getLogger(__name__)


class DeterministicScorer:
    """Runs all deterministic scorers against a single trace."""

    def score(
        self,
        inputs: dict[str, str],
        outputs: dict[str, str],
        expectations: dict[str, str],
    ) -> list[Score]:
        scores: list[Score] = []
        response = outputs.get("response", "")
        reference = expectations.get("expected_response", "")

        scores.append(token_overlap_f1(response, reference))

        expected_class = expectations.get("chosen_class_id", "")
        if expected_class:
            scores.append(class_exact_match(response, expected_class))

        scores.append(refusal_detection(response))

        return scores


class LLMJudgeScorer:
    """Wraps MLflow's built-in Correctness and Guidelines judges."""

    def __init__(self, judge_model: str) -> None:
        self._judge_model = judge_model

    def score(
        self,
        inputs: dict[str, str],
        outputs: dict[str, str],
        expectations: dict[str, str],
    ) -> list[Score]:
        scores: list[Score] = []

        try:
            from mlflow.genai.scorers import Correctness, Guidelines

            correctness = Correctness()
            guideline_text = expectations.get("guidelines", "")
            if not guideline_text:
                guideline_text = "Be helpful and accurate."
            guidelines = Guidelines(guidelines=guideline_text)

            eval_input: dict[str, Any] = {
                "inputs": inputs,
                "outputs": outputs,
                "expectations": expectations,
            }

            try:
                c_result = correctness(**eval_input)
                if c_result is not None:
                    scores.append(
                        Score(
                            scorer_name="correctness",
                            value=float(getattr(c_result, "value", 0.0)),
                            rationale=getattr(c_result, "rationale", None),
                        )
                    )
            except Exception:
                logger.warning("Correctness scorer failed", exc_info=True)

            try:
                g_result = guidelines(**eval_input)
                if g_result is not None:
                    scores.append(
                        Score(
                            scorer_name="guidelines",
                            value=float(getattr(g_result, "value", 0.0)),
                            rationale=getattr(g_result, "rationale", None),
                        )
                    )
            except Exception:
                logger.warning("Guidelines scorer failed", exc_info=True)

        except ImportError:
            logger.warning("MLflow genai scorers not available")

        return scores
