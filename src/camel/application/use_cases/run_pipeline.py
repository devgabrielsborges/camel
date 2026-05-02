from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

from camel.application.use_cases.export_results import ExportResults
from camel.application.use_cases.register_dataset import RegisterDataset
from camel.application.use_cases.run_evaluation import RunEvaluation
from camel.application.use_cases.run_inference import RunInference
from camel.domain.entities.evaluation import Evaluation, EvaluationStatus
from camel.domain.services.aggregation import AggregatedMetric, CategoryBreakdown
from camel.domain.services.verdict import VerdictResult, compute_verdict
from camel.domain.value_objects.prompt_template import PromptTemplate
from camel.infrastructure.adapters.mlflow_tracker import MLflowTrackerAdapter

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[int, int], None]
StepCallback = Callable[[int, str], None]


@dataclass(frozen=True)
class PipelineResult:
    evaluation: Evaluation
    run_id: str
    overall_metrics: list[AggregatedMetric]
    category_breakdowns: list[CategoryBreakdown]
    exported_rows: int
    verdict: VerdictResult | None = None


class RunPipeline:
    def __init__(
        self,
        tracker_adapter: MLflowTrackerAdapter,
        register_dataset: RegisterDataset,
        run_inference: RunInference,
        run_evaluation: RunEvaluation,
        export_results: ExportResults,
    ) -> None:
        self._tracker = tracker_adapter
        self._register_dataset = register_dataset
        self._run_inference = run_inference
        self._run_evaluation = run_evaluation
        self._export_results = export_results

    async def execute(
        self,
        evaluation: Evaluation,
        categories: list[str],
        output_path: str,
        prompt_template: PromptTemplate | None = None,
        limit: int | None = None,
        prompt_version_uri: str = "",
        on_step: StepCallback | None = None,
        on_inference_progress: ProgressCallback | None = None,
        on_evaluation_progress: ProgressCallback | None = None,
        on_export_progress: ProgressCallback | None = None,
        inference_total: int | None = None,
    ) -> PipelineResult:
        self._tracker.set_experiment(evaluation.experiment_name)

        logger.info("Pipeline step 1/6: Register prompt")
        if on_step is not None:
            on_step(1, "Registering prompt")
        if prompt_template is not None:
            prompt_version_uri = self._tracker.register_prompt(prompt_template)
            evaluation.prompt_version = prompt_version_uri
            logger.info("Registered prompt: %s", prompt_version_uri)

        logger.info("Pipeline step 2/6: Register evaluation dataset")
        if on_step is not None:
            on_step(2, "Registering dataset")
        dataset_count = self._register_dataset.execute(
            dataset_name=evaluation.dataset_name,
            categories=categories,
            limit=limit,
        )
        logger.info("Registered %d records in evaluation dataset", dataset_count)

        logger.info("Pipeline step 3/6: Inference (with MLflow autolog tracing)")
        if on_step is not None:
            on_step(3, "Running inference")
        run_id = self._tracker.start_run(evaluation)
        self._tracker.enable_autolog()
        try:
            evaluation = await self._run_inference.execute(
                evaluation=evaluation,
                categories=categories,
                limit=limit,
                prompt_version_uri=prompt_version_uri,
                on_progress=on_inference_progress,
                total=inference_total,
            )

            self._tracker.log_metrics(
                run_id,
                {"total_sessions": float(len(evaluation.sessions))},
            )

            logger.info("Pipeline step 4/6: Evaluation (reusing traces)")
            if on_step is not None:
                on_step(4, "Scoring traces")
            evaluation.transition_to(EvaluationStatus.EVALUATING)
            overall, by_category = self._run_evaluation.execute(
                evaluation=evaluation,
                run_id=run_id,
                on_progress=on_evaluation_progress,
            )

            logger.info("Pipeline step 5/6: Export (gold)")
            if on_step is not None:
                on_step(5, "Exporting results")
            exported_rows = self._export_results.execute(
                evaluation=evaluation,
                output_path=output_path,
                on_progress=on_export_progress,
            )

            logger.info("Pipeline step 6/6: Computing verdict")
            if on_step is not None:
                on_step(6, "Computing verdict")
            verdict_result = compute_verdict(by_category)

            verdict_metrics: dict[str, float] = {
                "verdict_positivo_overlap_mean": verdict_result.positivo_overlap_mean,
                "verdict_negativo_overlap_mean": verdict_result.negativo_overlap_mean,
                "verdict_positivo_refusal_rate": verdict_result.positivo_refusal_rate,
                "verdict_negativo_refusal_rate": verdict_result.negativo_refusal_rate,
                "verdict_discrimination_delta": verdict_result.discrimination_delta,
            }
            self._tracker.log_metrics(run_id, verdict_metrics)
            self._tracker.set_run_tags(
                run_id,
                {
                    "verdict": str(verdict_result.verdict),
                    "verdict.reasons": "; ".join(verdict_result.reasons),
                },
            )
            logger.info("Verdict: %s", verdict_result.verdict)

        except Exception:
            self._tracker.end_run(run_id)
            self._tracker.disable_autolog()
            raise
        else:
            self._tracker.end_run(run_id)
            self._tracker.disable_autolog()

        logger.info(
            "Pipeline complete: %d rows exported to %s",
            exported_rows,
            output_path,
        )

        return PipelineResult(
            evaluation=evaluation,
            run_id=run_id,
            overall_metrics=overall,
            category_breakdowns=by_category,
            exported_rows=exported_rows,
            verdict=verdict_result,
        )
