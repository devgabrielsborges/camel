from __future__ import annotations

import inspect
import logging
from collections import defaultdict
from typing import Any

from mlflow.entities import Feedback

from camel.domain.entities.evaluation import Evaluation, EvaluationStatus
from camel.domain.services.aggregation import (
    AggregatedMetric,
    CategoryBreakdown,
    aggregate_by_category,
    aggregate_scores,
)
from camel.domain.value_objects.score import Score
from camel.infrastructure.adapters.mlflow_tracker import MLflowTrackerAdapter

logger = logging.getLogger(__name__)


_YES_NO_MAP: dict[str, float] = {"yes": 1.0, "no": 0.0}


def _feedback_to_score(feedback: Feedback, fallback_name: str) -> Score:
    name = feedback.name if feedback.name != "feedback" else fallback_name
    raw = feedback.feedback.value if feedback.feedback else None
    if raw is None:
        value: float | bool | None = None
    elif isinstance(raw, bool):
        value = raw
    elif isinstance(raw, (int, float)):
        value = float(raw)
    elif isinstance(raw, str):
        value = _YES_NO_MAP.get(raw.strip().lower(), 0.0)
    else:
        value = 0.0
    return Score(scorer_name=name, value=value, rationale=feedback.rationale)


class RunEvaluation:
    def __init__(
        self,
        scorers: list[Any],
        tracker_adapter: MLflowTrackerAdapter,
    ) -> None:
        self._scorers = scorers
        self._tracker = tracker_adapter

    def execute(
        self,
        evaluation: Evaluation,
        run_id: str,
    ) -> tuple[list[AggregatedMetric], list[CategoryBreakdown]]:
        if evaluation.status != EvaluationStatus.EVALUATING:
            evaluation.transition_to(EvaluationStatus.EVALUATING)

        scorer_names = [getattr(s, "name", "unknown") for s in self._scorers]
        self._tracker.set_run_tags(
            run_id,
            {
                "evaluation.scorers": ",".join(scorer_names),
                "evaluation.scorer_count": str(len(self._scorers)),
                "evaluation.session_count": str(len(evaluation.sessions)),
            },
        )

        all_scores: list[Score] = []
        scores_by_category: dict[str, list[Score]] = defaultdict(list)
        scored_count = 0

        try:
            for session in evaluation.sessions:
                record = session.dataset_record
                for trace_obj in session.traces:
                    inputs = {"question": record.question}
                    outputs = {"response": trace_obj.output_text}
                    expectations = {
                        "expected_response": record.content,
                        "guidelines": "; ".join(record.instructions),
                        "chosen_class_id": record.chosen_class_id,
                        "classes": [
                            {"id": c.class_id, "class": c.class_name} for c in record.classes
                        ],
                    }

                    for scorer_fn in self._scorers:
                        try:
                            kwargs: dict[str, Any] = {
                                "inputs": inputs,
                                "outputs": outputs,
                            }
                            sig = inspect.signature(scorer_fn)
                            if "expectations" in sig.parameters:
                                kwargs["expectations"] = expectations
                            result = scorer_fn(**kwargs)
                            if isinstance(result, Feedback):
                                score = _feedback_to_score(
                                    result, getattr(scorer_fn, "name", "unknown")
                                )
                            else:
                                score = Score(
                                    scorer_name=getattr(scorer_fn, "name", "unknown"),
                                    value=result if result is not None else 0.0,
                                )
                            trace_obj.add_score(score)
                            all_scores.append(score)
                            scores_by_category[record.data_category_qa].append(score)
                        except Exception:
                            logger.warning(
                                "Scorer %s failed",
                                getattr(scorer_fn, "name", "unknown"),
                                exc_info=True,
                            )

                    scored_count += 1

            overall = aggregate_scores(all_scores)
            by_category = aggregate_by_category(scores_by_category)

            flat_metrics: dict[str, float] = {}
            for m in overall:
                flat_metrics[f"{m.scorer_name}_mean"] = m.mean
                flat_metrics[f"{m.scorer_name}_std"] = m.std
            flat_metrics["total_scored"] = float(scored_count)

            self._tracker.log_metrics(run_id, flat_metrics)

            evaluation.transition_to(EvaluationStatus.COMPLETE)

        except Exception:
            evaluation.transition_to(EvaluationStatus.FAILED)
            raise

        logger.info("Evaluated %d traces", scored_count)
        return overall, by_category
