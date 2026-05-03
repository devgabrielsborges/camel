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
from camel.domain.value_objects.statistical_verdict_result import StatisticalVerdictResult
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
    statistical_verdict: StatisticalVerdictResult | None = None


class RunPipeline:
    def __init__(
        self,
        tracker_adapter: MLflowTrackerAdapter,
        register_dataset: RegisterDataset,
        run_inference: RunInference,
        run_evaluation: RunEvaluation,
        export_results: ExportResults,
        *,
        threshold_profile_path: str = "",
        reference_db_path: str = "",
        legacy_verdict_only: bool = False,
    ) -> None:
        self._tracker = tracker_adapter
        self._register_dataset = register_dataset
        self._run_inference = run_inference
        self._run_evaluation = run_evaluation
        self._export_results = export_results
        self._threshold_profile_path = threshold_profile_path
        self._reference_db_path = reference_db_path
        self._legacy_verdict_only = legacy_verdict_only

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

            self._tracker.disable_autolog()

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
                run_id=run_id,
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

            statistical_verdict_result = self._run_statistical_verdict(
                evaluation,
                run_id,
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
            verdict=verdict_result,
            statistical_verdict=statistical_verdict_result,
        )

    def _run_statistical_verdict(
        self,
        evaluation: Evaluation,
        run_id: str,
    ) -> StatisticalVerdictResult | None:
        if self._legacy_verdict_only:
            return None

        if not self._threshold_profile_path:
            logger.info("No threshold profile path configured — skipping statistical verdict")
            return None

        from camel.infrastructure.adapters.threshold_repository import ThresholdProfileRepository

        repo = ThresholdProfileRepository()
        profile = repo.load(self._threshold_profile_path)
        if profile is None:
            logger.warning(
                "ThresholdProfile not found at %s — statistical verdict skipped. "
                "Run 'camel derive-thresholds' to generate one.",
                self._threshold_profile_path,
            )
            return None

        from camel.domain.services.score_collector import collect_raw_scores

        candidate_collections = collect_raw_scores(evaluation)
        if not candidate_collections:
            logger.warning("No candidate score collections — statistical verdict skipped")
            return None

        reference_collections = []
        if self._reference_db_path:
            from camel.infrastructure.adapters.duckdb_reference_scores import load_reference_scores

            reference_collections = load_reference_scores(
                self._reference_db_path,
                list(profile.reference_models),
            )

        from camel.domain.services.hypothesis_tests import run_all_tests
        from camel.domain.services.statistical_verdict import compute_statistical_verdict

        test_results = run_all_tests(candidate_collections, reference_collections, profile)

        critical_metrics = {
            ("token_overlap_f1", "positivo"),
            ("token_overlap_f1", "negativo"),
            ("refusal_detection", "negativo"),
            ("discrimination_delta", "global"),
        }
        stat_verdict = compute_statistical_verdict(
            test_results,
            critical_metrics,
            alpha=profile.alpha,
            correction_method=profile.correction_method,
            profile_version=profile.version,
        )

        v2_metrics: dict[str, float] = {}
        for tr in test_results:
            prefix = f"verdict_v2_{tr.category}_{tr.metric_name}"
            v2_metrics[f"{prefix}_p_value"] = tr.p_value
            v2_metrics[f"{prefix}_p_adjusted"] = tr.p_value_adjusted
            v2_metrics[f"{prefix}_effect_size"] = tr.effect_size
            v2_metrics[f"{prefix}_candidate_value"] = tr.candidate_value
        self._tracker.log_metrics(run_id, v2_metrics)

        v2_tags: dict[str, str] = {
            "verdict_v2": str(stat_verdict.verdict),
            "verdict_v2.profile_version": profile.version,
            "verdict_v2.correction": profile.correction_method,
            "verdict_v2.reasons": "; ".join(stat_verdict.reasons[:5]),
        }
        self._tracker.set_run_tags(run_id, v2_tags)

        logger.info("Statistical verdict: %s", stat_verdict.verdict)
        return stat_verdict
