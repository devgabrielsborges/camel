from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Optional

import typer
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from camel.infrastructure.cli.app import app

logger = logging.getLogger(__name__)

DEFAULT_CATEGORIES = ["positivo", "negativo"]

_STEP_LABELS: dict[int, str] = {
    1: "Registering prompt",
    2: "Registering dataset",
    3: "Running inference",
    4: "Scoring traces",
    5: "Exporting results",
    6: "Computing verdict",
}


def _create_progress() -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    )


@app.command(name="run")
def run_pipeline(
    limit: Optional[int] = typer.Option(
        None,
        "--limit",
        "-l",
        help="Max number of records to process (all if omitted)",
    ),
    batch_size: Optional[int] = typer.Option(
        None,
        "--batch-size",
        "-b",
        help="Number of records per batch (overrides BATCH_SIZE env var)",
    ),
    concurrency: Optional[int] = typer.Option(
        None,
        "--concurrency",
        "-c",
        help="Max concurrent agent calls (overrides CONCURRENCY env var)",
    ),
    categories: Optional[str] = typer.Option(
        None,
        "--categories",
        help="Comma-separated data_category_QA filter values",
    ),
    experiment: Optional[str] = typer.Option(
        None,
        "--experiment",
        "-e",
        help="MLflow experiment name (default: WeniEval)",
    ),
    output: Optional[str] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output CSV path (overrides RESULTS_DIR env var)",
    ),
    model: Optional[str] = typer.Option(
        None,
        "--model",
        "-m",
        help="OpenAI model to use for inference (overrides OPENAI_MODEL env var)",
    ),
    no_llm_judge: bool = typer.Option(
        False,
        "--no-llm-judge",
        help="Skip LLM-as-judge scorers (deterministic only)",
    ),
    no_cache: bool = typer.Option(
        False,
        "--no-cache",
        help="Disable inference cache (always call the LLM)",
    ),
) -> None:
    """Execute the full pipeline: register prompt -> register dataset -> infer -> evaluate -> export."""
    from camel.application.use_cases.export_results import ExportResults
    from camel.application.use_cases.register_dataset import RegisterDataset
    from camel.application.use_cases.run_evaluation import RunEvaluation
    from camel.application.use_cases.run_inference import RunInference
    from camel.application.use_cases.run_pipeline import RunPipeline
    from camel.domain.entities.evaluation import Evaluation
    from camel.domain.services.prompt_renderer import PromptRenderer
    from camel.domain.value_objects.model_config import ModelConfig
    from camel.domain.value_objects.prompt_template import PromptTemplate
    from camel.infrastructure.adapters.duckdb_dataset import DuckDBDatasetAdapter
    from camel.infrastructure.adapters.mlflow_tracker import MLflowTrackerAdapter
    from camel.infrastructure.config.settings import Settings
    from camel.infrastructure.factories.agent_factory import create_agent_adapter
    from camel.infrastructure.factories.scorer_factory import (
        create_groundedness_scorer,
        create_scorers,
    )

    settings = Settings()

    cat_list = categories.split(",") if categories else DEFAULT_CATEGORIES
    exp_name = experiment or settings.experiment_name
    bs = batch_size or settings.batch_size
    conc = concurrency or settings.concurrency
    output_path = output or f"{settings.results_dir}/predictions.csv"
    model_name = model or settings.openai_model

    from camel.infrastructure.adapters.cached_agent import CachedAgentAdapter

    dataset_adapter = DuckDBDatasetAdapter(db_path=settings.duckdb_path)
    raw_agent = create_agent_adapter(settings, model_override=model_name)
    if no_cache:
        agent_adapter = raw_agent
    else:
        agent_adapter = CachedAgentAdapter(
            agent=raw_agent,
            cache_dir=settings.inference_cache_dir,
            model=model_name,
        )
    tracker_adapter = MLflowTrackerAdapter(tracking_uri=settings.mlflow_tracking_uri)
    renderer = PromptRenderer(template_path=settings.prompt_template_path)
    scorers = create_scorers(settings, no_llm_judge=no_llm_judge)

    total_records = dataset_adapter.count(cat_list)
    effective_total = min(total_records, limit) if limit else total_records

    evaluation = Evaluation(
        evaluation_id=str(uuid.uuid4()),
        experiment_name=exp_name,
        eval_model=ModelConfig(model_name=model_name, temperature=0.0),
        prompt_version="",
        dataset_name=settings.dataset_name,
    )

    prompt_template = PromptTemplate(
        template_path=settings.prompt_template_path,
        version_uri="",
    )

    register_dataset = RegisterDataset(
        dataset_adapter=dataset_adapter,
        tracker_adapter=tracker_adapter,
    )

    run_inference = RunInference(
        dataset_adapter=dataset_adapter,
        agent_adapter=agent_adapter,
        tracker_adapter=tracker_adapter,
        prompt_renderer=renderer,
        batch_size=bs,
        concurrency=conc,
        pass_at_k=settings.pass_at_k,
    )

    groundedness_scorer = create_groundedness_scorer(settings) if not no_llm_judge else None

    run_evaluation = RunEvaluation(
        scorers=scorers,
        tracker_adapter=tracker_adapter,
        groundedness_scorer=groundedness_scorer,
    )

    export_results = ExportResults()

    pipeline = RunPipeline(
        tracker_adapter=tracker_adapter,
        register_dataset=register_dataset,
        run_inference=run_inference,
        run_evaluation=run_evaluation,
        export_results=export_results,
    )

    with _create_progress() as progress:
        steps_task = progress.add_task("Pipeline", total=6)
        inference_task = progress.add_task("Inference", total=effective_total, visible=False)
        eval_task = progress.add_task("Scoring", total=effective_total, visible=False)
        export_task = progress.add_task("Exporting", total=effective_total, visible=False)

        def _on_step(step: int, _description: str) -> None:
            label = _STEP_LABELS.get(step, _description)
            progress.update(steps_task, description=f"[{step}/6] {label}")
            if step == 3:
                progress.update(inference_task, visible=True)
            elif step == 4:
                progress.update(inference_task, visible=False)
                progress.update(eval_task, visible=True)
            elif step == 5:
                progress.update(eval_task, visible=False)
                progress.update(export_task, visible=True)
            elif step == 6:
                progress.update(export_task, visible=False)

        def _on_inference_progress(current: int, _total: int) -> None:
            progress.update(inference_task, completed=current)

        def _on_eval_progress(current: int, _total: int) -> None:
            progress.update(eval_task, completed=current)

        def _on_export_progress(current: int, _total: int) -> None:
            progress.update(export_task, completed=current)

        try:
            result = asyncio.run(
                pipeline.execute(
                    evaluation=evaluation,
                    categories=cat_list,
                    output_path=output_path,
                    prompt_template=prompt_template,
                    limit=limit,
                    on_step=_on_step,
                    on_inference_progress=_on_inference_progress,
                    on_evaluation_progress=_on_eval_progress,
                    on_export_progress=_on_export_progress,
                    inference_total=effective_total,
                )
            )
        except RuntimeError as exc:
            progress.stop()
            typer.echo(f"Pipeline failed: {exc}", err=True)
            raise typer.Exit(code=1) from exc

        progress.update(steps_task, completed=6, description="Pipeline complete")

    typer.echo(f"\nPipeline complete: {result.exported_rows} rows exported to {output_path}")
    typer.echo(f"MLflow run ID: {result.run_id}")
    typer.echo(f"Sessions: {len(result.evaluation.sessions)}")
    typer.echo(f"Status: {result.evaluation.status.value}")

    if result.overall_metrics:
        typer.echo("\nScorer Summary:")
        for m in result.overall_metrics:
            typer.echo(f"  {m.scorer_name}: mean={m.mean:.4f}, std={m.std:.4f}, n={m.count}")

    if result.verdict:
        typer.echo(f"\nVerdict: {result.verdict.verdict.value}")
        for reason in result.verdict.reasons:
            typer.echo(f"  - {reason}")

    if isinstance(agent_adapter, CachedAgentAdapter):
        typer.echo(
            f"\nCache: {agent_adapter.hit_count} hits, " f"{agent_adapter.miss_count} misses"
        )
