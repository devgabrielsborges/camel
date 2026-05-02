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


@app.command()
def infer(
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
    model: Optional[str] = typer.Option(
        None,
        "--model",
        "-m",
        help="OpenAI model to use for inference (overrides OPENAI_MODEL env var)",
    ),
) -> None:
    """Run batch inference: register prompt -> register dataset -> run inferences with MLflow tracing."""
    from camel.application.use_cases.register_dataset import RegisterDataset
    from camel.application.use_cases.run_inference import RunInference
    from camel.domain.entities.evaluation import Evaluation
    from camel.domain.services.prompt_renderer import PromptRenderer
    from camel.domain.value_objects.model_config import ModelConfig
    from camel.domain.value_objects.prompt_template import PromptTemplate
    from camel.infrastructure.adapters.duckdb_dataset import DuckDBDatasetAdapter
    from camel.infrastructure.adapters.mlflow_tracker import MLflowTrackerAdapter
    from camel.infrastructure.config.settings import Settings
    from camel.infrastructure.factories.agent_factory import create_agent_adapter

    settings = Settings()

    cat_list = categories.split(",") if categories else DEFAULT_CATEGORIES
    exp_name = experiment or settings.experiment_name
    bs = batch_size or settings.batch_size
    conc = concurrency or settings.concurrency
    model_name = model or settings.openai_model

    dataset_adapter = DuckDBDatasetAdapter(db_path=settings.duckdb_path)
    agent_adapter = create_agent_adapter(settings, model_override=model_name)
    tracker_adapter = MLflowTrackerAdapter(tracking_uri=settings.mlflow_tracking_uri)
    renderer = PromptRenderer(template_path=settings.prompt_template_path)

    evaluation = Evaluation(
        evaluation_id=str(uuid.uuid4()),
        experiment_name=exp_name,
        eval_model=ModelConfig(model_name=model_name, temperature=0.0),
        prompt_version="",
        dataset_name=settings.dataset_name,
    )

    tracker_adapter.set_experiment(exp_name)

    total_records = dataset_adapter.count(cat_list)
    effective_total = min(total_records, limit) if limit else total_records

    with _create_progress() as progress:
        steps_task = progress.add_task("Pipeline steps", total=3)

        progress.update(steps_task, description="[1/3] Registering prompt")
        prompt_version_uri = ""
        try:
            tpl = PromptTemplate(
                template_path=settings.prompt_template_path,
                version_uri="",
            )
            prompt_version_uri = tracker_adapter.register_prompt(tpl)
            evaluation.prompt_version = prompt_version_uri
        except Exception:
            logger.warning("Could not register prompt with MLflow, continuing without")
        progress.advance(steps_task)

        progress.update(steps_task, description="[2/3] Registering dataset")
        reg_uc = RegisterDataset(
            dataset_adapter=dataset_adapter,
            tracker_adapter=tracker_adapter,
        )
        reg_uc.execute(
            dataset_name=evaluation.dataset_name,
            categories=cat_list,
            limit=limit,
        )
        progress.advance(steps_task)

        progress.update(steps_task, description="[3/3] Running inference")
        inference_task = progress.add_task("Inference", total=effective_total)

        def _on_inference_progress(current: int, _total: int) -> None:
            progress.update(inference_task, completed=current)

        run_id = tracker_adapter.start_run(evaluation)
        tracker_adapter.enable_autolog()

        use_case = RunInference(
            dataset_adapter=dataset_adapter,
            agent_adapter=agent_adapter,
            tracker_adapter=tracker_adapter,
            prompt_renderer=renderer,
            batch_size=bs,
            concurrency=conc,
        )

        try:
            result = asyncio.run(
                use_case.execute(
                    evaluation=evaluation,
                    categories=cat_list,
                    limit=limit,
                    prompt_version_uri=prompt_version_uri,
                    on_progress=_on_inference_progress,
                    total=effective_total,
                )
            )
            tracker_adapter.log_metrics(
                run_id,
                {"total_sessions": float(len(result.sessions))},
            )
        finally:
            tracker_adapter.end_run(run_id)
            tracker_adapter.disable_autolog()

        progress.update(inference_task, completed=effective_total)
        progress.advance(steps_task)

    typer.echo(
        f"\nInference complete: {len(result.sessions)} sessions, " f"status={result.status.value}"
    )
    typer.echo(f"MLflow run ID: {run_id}")
    typer.echo(f"Run 'camel evaluate --run-id {run_id}' to score these traces")
