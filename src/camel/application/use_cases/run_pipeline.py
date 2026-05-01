from __future__ import annotations

import logging
from dataclasses import dataclass

from camel.application.use_cases.export_results import ExportResults
from camel.application.use_cases.run_evaluation import RunEvaluation
from camel.application.use_cases.run_inference import RunInference
from camel.domain.entities.evaluation import Evaluation
from camel.domain.services.aggregation import AggregatedMetric, CategoryBreakdown

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PipelineResult:
    evaluation: Evaluation
    overall_metrics: list[AggregatedMetric]
    category_breakdowns: list[CategoryBreakdown]
    exported_rows: int


class RunPipeline:
    def __init__(
        self,
        run_inference: RunInference,
        run_evaluation: RunEvaluation,
        export_results: ExportResults,
    ) -> None:
        self._run_inference = run_inference
        self._run_evaluation = run_evaluation
        self._export_results = export_results

    async def execute(
        self,
        evaluation: Evaluation,
        categories: list[str],
        run_id: str,
        output_path: str,
        limit: int | None = None,
        prompt_version_uri: str = "",
    ) -> PipelineResult:
        logger.info("Pipeline phase 1/3: Inference")
        evaluation = await self._run_inference.execute(
            evaluation=evaluation,
            categories=categories,
            limit=limit,
            prompt_version_uri=prompt_version_uri,
        )

        logger.info("Pipeline phase 2/3: Evaluation")
        overall, by_category = self._run_evaluation.execute(
            evaluation=evaluation,
            run_id=run_id,
        )

        logger.info("Pipeline phase 3/3: Export")
        exported_rows = self._export_results.execute(
            evaluation=evaluation,
            output_path=output_path,
        )

        logger.info(
            "Pipeline complete: %d rows exported to %s",
            exported_rows,
            output_path,
        )

        return PipelineResult(
            evaluation=evaluation,
            overall_metrics=overall,
            category_breakdowns=by_category,
            exported_rows=exported_rows,
        )
