from __future__ import annotations

import logging
from collections import defaultdict

from camel.domain.entities.evaluation import Evaluation, EvaluationStatus
from camel.domain.services.aggregation import (
    AggregatedMetric,
    CategoryBreakdown,
    aggregate_by_category,
    aggregate_scores,
)
from camel.domain.value_objects.score import Score
from camel.infrastructure.adapters.mlflow_scorer import (
    DeterministicScorer,
    LLMJudgeScorer,
)
from camel.infrastructure.adapters.mlflow_tracker import MLflowTrackerAdapter

logger = logging.getLogger(__name__)


class RunEvaluation:
    def __init__(
        self,
        deterministic_scorer: DeterministicScorer,
        llm_scorer: LLMJudgeScorer | None,
        tracker_adapter: MLflowTrackerAdapter,
    ) -> None:
        self._det_scorer = deterministic_scorer
        self._llm_scorer = llm_scorer
        self._tracker = tracker_adapter

    def execute(
        self,
        evaluation: Evaluation,
        run_id: str,
    ) -> tuple[list[AggregatedMetric], list[CategoryBreakdown]]:
        if evaluation.status != EvaluationStatus.EVALUATING:
            evaluation.transition_to(EvaluationStatus.EVALUATING)

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
                    }

                    det_scores = self._det_scorer.score(inputs, outputs, expectations)
                    for s in det_scores:
                        trace_obj.add_score(s)
                        all_scores.append(s)
                        scores_by_category[record.data_category_qa].append(s)

                    if self._llm_scorer is not None:
                        llm_scores = self._llm_scorer.score(
                            inputs, outputs, expectations
                        )
                        for s in llm_scores:
                            trace_obj.add_score(s)
                            all_scores.append(s)
                            scores_by_category[record.data_category_qa].append(s)

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
