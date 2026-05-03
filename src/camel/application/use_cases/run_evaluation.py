from __future__ import annotations

import inspect
import logging
from collections import defaultdict
from collections.abc import Callable
from typing import Any

from mlflow.entities import Feedback

from camel.application.ports.groundedness_port import GroundednessPort
from camel.domain.entities.evaluation import Evaluation, EvaluationStatus
from camel.domain.services.aggregation import (
    AggregatedMetric,
    CategoryBreakdown,
    aggregate_by_category,
    aggregate_scores,
)
from camel.domain.services.failure_modes import classify_failure_mode
from camel.domain.services.scoring import pass_at_k, self_consistency, token_overlap_f1
from camel.domain.value_objects.score import Score
from camel.infrastructure.adapters.mlflow_tracker import MLflowTrackerAdapter

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[int, int], None]


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
        groundedness_scorer: GroundednessPort | None = None,
        pass_at_k_threshold: float = 0.3,
    ) -> None:
        self._scorers = scorers
        self._tracker = tracker_adapter
        self._groundedness = groundedness_scorer
        self._pass_at_k_threshold = pass_at_k_threshold

    def execute(
        self,
        evaluation: Evaluation,
        run_id: str,
        on_progress: ProgressCallback | None = None,
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
        total_sessions = len(evaluation.sessions)

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
                        "chunks": [
                            {"content": ch.content, "score": ch.score} for ch in record.chunks_big
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

                    if self._groundedness and record.content:
                        try:
                            g_score, g_reasons = self._groundedness.score(
                                source=record.content,
                                statement=trace_obj.output_text,
                            )
                            groundedness_score = Score(
                                scorer_name="groundedness",
                                value=round(g_score, 4),
                                rationale=g_reasons,
                            )
                            trace_obj.add_score(groundedness_score)
                            all_scores.append(groundedness_score)
                            scores_by_category[record.data_category_qa].append(groundedness_score)
                        except Exception:
                            logger.warning("Groundedness scoring failed", exc_info=True)

                    failure_mode = classify_failure_mode(
                        {s.scorer_name: s for s in trace_obj.scores},
                        record.data_category_qa,
                    )
                    fm_score = Score(
                        scorer_name="failure_mode",
                        value=None,
                        metadata={"failure_mode": str(failure_mode)},
                    )
                    trace_obj.add_score(fm_score)

                    scored_count += 1

                if len(session.traces) > 1:
                    responses = [t.output_text for t in session.traces]
                    pk_result = pass_at_k(
                        question_id=record.id,
                        responses=responses,
                        reference=record.content,
                        scorer_fn=token_overlap_f1,
                        threshold=self._pass_at_k_threshold,
                    )
                    pk_score = Score(
                        scorer_name="pass_at_k",
                        value=pk_result.passed,
                        metadata={
                            "k": pk_result.k,
                            "best_score": pk_result.best_score,
                        },
                    )
                    session.traces[0].add_score(pk_score)
                    all_scores.append(pk_score)
                    scores_by_category[record.data_category_qa].append(pk_score)

                    sc_score = self_consistency(responses)
                    if sc_score.value is not None:
                        session.traces[0].add_score(sc_score)
                        all_scores.append(sc_score)
                        scores_by_category[record.data_category_qa].append(sc_score)

                if on_progress is not None:
                    on_progress(scored_count, total_sessions)

            overall = aggregate_scores(all_scores)
            by_category = aggregate_by_category(scores_by_category)

            flat_metrics: dict[str, float] = {}
            for m in overall:
                flat_metrics[f"{m.scorer_name}_mean"] = m.mean
                flat_metrics[f"{m.scorer_name}_std"] = m.std
            flat_metrics["total_scored"] = float(scored_count)

            fm_counts: dict[str, int] = defaultdict(int)
            for s in all_scores:
                if s.scorer_name == "failure_mode":
                    fm = s.metadata.get("failure_mode", "unknown")
                    fm_counts[fm] += 1
            for fm_name, count in fm_counts.items():
                flat_metrics[f"failure_mode_{fm_name}_count"] = float(count)
            if scored_count > 0 and fm_counts:
                for fm_name, count in fm_counts.items():
                    flat_metrics[f"failure_mode_{fm_name}_rate"] = round(count / scored_count, 4)

            self._tracker.log_metrics(run_id, flat_metrics)

            evaluation.transition_to(EvaluationStatus.COMPLETE)

        except Exception:
            evaluation.transition_to(EvaluationStatus.FAILED)
            raise

        logger.info("Evaluated %d traces", scored_count)
        return overall, by_category
