from __future__ import annotations

import logging
from dataclasses import dataclass

from camel.application.use_cases.export_results import ExportResults
from camel.application.use_cases.register_dataset import RegisterDataset
from camel.application.use_cases.run_evaluation import RunEvaluation
from camel.application.use_cases.run_inference import RunInference
from camel.domain.entities.evaluation import Evaluation, EvaluationStatus
from camel.domain.services.aggregation import AggregatedMetric, CategoryBreakdown
from camel.domain.value_objects.prompt_template import PromptTemplate
from camel.infrastructure.adapters.mlflow_tracker import MLflowTrackerAdapter

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PipelineResult:
    evaluation: Evaluation
    run_id: str
    overall_metrics: list[AggregatedMetric]
    category_breakdowns: list[CategoryBreakdown]
    exported_rows: int


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
    ) -> PipelineResult:
        logger.info("Pipeline step 1/5: Register prompt")
        if prompt_template is not None:
            prompt_version_uri = self._tracker.register_prompt(prompt_template)
            evaluation.prompt_version = prompt_version_uri
            logger.info("Registered prompt: %s", prompt_version_uri)

        logger.info("Pipeline step 2/5: Register evaluation dataset")
        dataset_count = self._register_dataset.execute(
            dataset_name=evaluation.dataset_name,
            categories=categories,
            limit=limit,
        )
        logger.info("Registered %d records in evaluation dataset", dataset_count)

        logger.info("Pipeline step 3/5: Inference (with MLflow autolog tracing)")
        run_id = self._tracker.start_run(evaluation)
        self._tracker.enable_autolog()
        try:
            evaluation = await self._run_inference.execute(
                evaluation=evaluation,
                categories=categories,
                limit=limit,
                prompt_version_uri=prompt_version_uri,
            )

            self._tracker.log_metrics(
                run_id,
                {"total_sessions": float(len(evaluation.sessions))},
            )

            logger.info("Pipeline step 4/5: Evaluation (reusing traces)")
            evaluation.transition_to(EvaluationStatus.EVALUATING)
            overall, by_category = self._run_evaluation.execute(
                evaluation=evaluation,
                run_id=run_id,
            )

            logger.info("Pipeline step 5/5: Export (gold)")
            exported_rows = self._export_results.execute(
                evaluation=evaluation,
                output_path=output_path,
            )

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
        )
